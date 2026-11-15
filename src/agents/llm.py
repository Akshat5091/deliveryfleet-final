"""Gemini access for both frameworks: model factory, rate limiter, retries.

Free-tier Gemini keys allow a limited number of requests per minute for each
model, so every LLM call goes through that model's sliding-window limiter
(LLM_RPM). Transient 429 / 503 errors are retried with exponential backoff plus
jitter (the Lab-04 run_with_retry pattern).
"""
from __future__ import annotations

import asyncio
import random
import threading
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import TypeVar

from crewai import LLM
from google import genai
from google.genai import types

from src.config import LLM_RPM, ROUTE_MODEL, WORKER_MODEL
from src.observability import tracing

T = TypeVar("T")
RETRYABLE = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "overloaded", "rate limit", "500 INTERNAL")


class RateLimiter:
    """Thread-safe sliding-window limit: at most `rpm` calls in any 60 seconds.

    Gemini's quota is per minute, so calls inside the budget go straight through (one demo run makes
    about 13 calls) and a call waits only when the last 60 s already hold `rpm` calls, until the
    oldest of them leaves the window.
    """

    WINDOW_S = 60.0

    def __init__(self, rpm: int, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.rpm = max(rpm, 1)
        self._clock, self._sleep = clock, sleep
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def wait(self) -> None:
        while True:
            with self._lock:
                now = self._clock()
                while self._calls and now - self._calls[0] >= self.WINDOW_S:
                    self._calls.popleft()
                if len(self._calls) < self.rpm:
                    self._calls.append(now)
                    return
                delay = self.WINDOW_S - (now - self._calls[0])
            self._sleep(delay)


# Set by the ADK route service at start-up: the shared model's budget is split between processes.
_route_process = False
ROUTE_SHARE = 0.2  # measured: the Route Agent makes 3 of a booking's 14 calls (6 when it also reroutes)


def mark_route_service() -> None:
    global _route_process
    _route_process = True


def shared_budget(rpm: int, route_process: bool) -> int:
    """ROUTE_MODEL is called from two processes (the web app and the ADK route service), so the
    key's per-minute budget is split by how many calls each side really makes."""
    route = max(1, round(rpm * ROUTE_SHARE))
    return route if route_process else max(1, rpm - route)


# Gemini counts requests per model, so each model gets its own limiter.
_limiters: dict[str, RateLimiter] = {}
_limiters_lock = threading.Lock()


def limiter_for(model: str) -> RateLimiter:
    with _limiters_lock:
        if model not in _limiters:
            rpm = shared_budget(LLM_RPM, _route_process) if model == ROUTE_MODEL else LLM_RPM
            _limiters[model] = RateLimiter(rpm)
        return _limiters[model]


def daily_quota_spent(exc: BaseException) -> bool:
    """A per-day free-tier quota only resets at midnight Pacific time; retrying cannot help."""
    return "PerDay" in f"{exc}"


def is_retryable(exc: BaseException) -> bool:
    if daily_quota_spent(exc):
        return False
    text = f"{type(exc).__name__} {exc}"
    return any(marker.lower() in text.lower() for marker in RETRYABLE)


def crew_llm(model: str = WORKER_MODEL, temperature: float = 0.2) -> LLM:
    """CrewAI routes 'gemini/...' to its native Gemini provider; rate limiting is
    applied by the before_llm_call hook registered in install_llm_hooks()."""
    return LLM(model=f"gemini/{model}", temperature=temperature, max_tokens=4096,
               thinking_config=types.ThinkingConfig(thinking_level="low"))


_hooks_installed = False


def install_llm_hooks() -> None:
    global _hooks_installed
    if _hooks_installed:
        return
    from crewai.hooks import register_before_llm_call_hook

    def _throttle(context) -> None:
        limiter_for(WORKER_MODEL).wait()
        return None

    register_before_llm_call_hook(_throttle)
    _hooks_installed = True


def rate_limited(exc: BaseException) -> bool:
    text = f"{exc}"
    return "429" in text or "RESOURCE_EXHAUSTED" in text


async def with_retry(fn: Callable[[], Awaitable[T]], attempts: int = 4) -> T:
    """Retry transient Gemini errors. A 429 means the per-minute quota is spent, so wait out
    the window (30 s, 60 s, 90 s); a 503 overload clears quickly (2 s, 4 s, 8 s)."""
    for attempt in range(attempts):
        try:
            return await fn()
        except Exception as exc:
            if attempt == attempts - 1 or not is_retryable(exc):
                raise
            delay = 30 * (attempt + 1) if rate_limited(exc) else 2 ** (attempt + 1)
            await asyncio.sleep(delay + random.uniform(0, 1.5))
    raise RuntimeError("unreachable")


_client: genai.Client | None = None


def genai_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


async def gemini_generate(prompt: str, agent: str, model: str = WORKER_MODEL, max_tokens: int = 1024,
                          response_schema: type | None = None, temperature: float = 0.0,
                          thinking: str = "low") -> str:
    """Direct google-genai call (guard judge, LLM-as-judge), recorded as an LLM event.

    Gemini 3.x models think before answering and thinking tokens count against
    max_tokens, so the thinking level is set explicitly (Lab-07 uses "low").
    """
    config = types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_tokens,
                                         thinking_config=types.ThinkingConfig(thinking_level=thinking))
    if response_schema is not None:
        config.response_mime_type = "application/json"
        config.response_schema = response_schema

    async def call() -> types.GenerateContentResponse:
        await asyncio.to_thread(limiter_for(model).wait)
        return await genai_client().aio.models.generate_content(model=model, contents=prompt, config=config)

    run = tracing.current()
    ctx = tracing.RunContext(run.run_id, agent, "google-genai") if run else None
    start = time.perf_counter()
    try:
        resp = await with_retry(call)
    except Exception as exc:
        tracing.record_llm_call(model, 0, 0, None, status="ERROR", preview=str(exc), ctx=ctx)
        raise
    usage = resp.usage_metadata
    output_tokens = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
    tracing.record_llm_call(model, usage.prompt_token_count or 0, output_tokens,
                            round((time.perf_counter() - start) * 1000, 1), preview=resp.text or "", ctx=ctx)
    return resp.text or ""
