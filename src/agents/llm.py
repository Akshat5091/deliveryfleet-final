"""Gemini access for both frameworks: model factory, rate limiter, retries.

Free-tier Gemini keys allow a limited number of requests per minute for each
model, so every LLM call goes through that model's sliding-window limiter
(LLM_RPM). Transient 429 / 503 errors are retried with exponential backoff plus
jitter (the Lab-04 run_with_retry pattern).

When the worker model is overloaded ("503 high demand"), hangs past CALL_TIMEOUT_S or
runs out of quota, it is marked busy for BUSY_FOR_S and calls go to FALLBACK_MODEL,
which has its own free quota, instead of waiting on the busy model.
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

from src.config import CALL_TIMEOUT_S, FALLBACK_MODEL, LLM_RPM, ROUTE_MODEL, WORKER_MODEL
from src.observability import tracing

T = TypeVar("T")
RETRYABLE = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "504", "DEADLINE_EXCEEDED", "overloaded", "rate limit",
             "500 INTERNAL", "high demand", "timeout", "timed out")


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


# set by the route service at start-up (it gets a smaller share of the per-minute budget)
_route_process = False
ROUTE_SHARE = 0.2  # Route Agent: about 3 of a booking's 14 calls


def mark_route_service() -> None:
    global _route_process
    _route_process = True


def shared_budget(rpm: int, route_process: bool) -> int:
    """ROUTE_MODEL is called from two processes (the web app and the ADK route service), so the
    key's per-minute budget is split by how many calls each side really makes."""
    route = max(1, round(rpm * ROUTE_SHARE))
    return route if route_process else max(1, rpm - route)


# Gemini rate limits are per model
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
    return "PerDay" in str(exc)


def is_retryable(exc: BaseException | str) -> bool:
    if daily_quota_spent(exc):
        return False
    text = f"{type(exc).__name__} {exc}"
    return any(marker.lower() in text.lower() for marker in RETRYABLE)


BUSY_FOR_S = 120.0
_busy_until: dict[str, float] = {}
_busy_lock = threading.Lock()


def _now() -> float:
    return time.monotonic()


def note_failure(model: str, exc: BaseException | str) -> None:
    """A model that is overloaded, slow or out of quota is skipped for BUSY_FOR_S. Other errors
    (a bad request, a refused key) are not the model's load, so they change nothing."""
    if not (is_retryable(exc) or daily_quota_spent(exc)):
        return
    with _busy_lock:
        _busy_until[model.split("/")[-1]] = _now() + BUSY_FOR_S


def is_busy(model: str) -> bool:
    with _busy_lock:
        return _busy_until.get(model, 0.0) > _now()


def active_model(model: str) -> str:
    """The model to call now: `model` unless it is busy, else the fallback. When both are busy,
    the one that was marked busy first (and so frees up first)."""
    candidates = [model] if model == FALLBACK_MODEL else [model, FALLBACK_MODEL]
    with _busy_lock:
        until = {m: _busy_until.get(m, 0.0) for m in candidates}
    now = _now()
    free = [m for m in candidates if until[m] <= now]
    return free[0] if free else min(candidates, key=until.__getitem__)


def crew_llm(model: str = WORKER_MODEL, temperature: float = 0.2) -> LLM:
    """CrewAI routes 'gemini/...' to its native Gemini provider; rate limiting and the busy-model
    switch are applied by the before_llm_call hook registered in install_llm_hooks().
    The Gemini provider sends only max_output_tokens to the API; max_tokens alone is not a cap."""
    return LLM(model=f"gemini/{model}", temperature=temperature, max_tokens=4096, max_output_tokens=4096,
               thinking_config=types.ThinkingConfig(thinking_level="low"),
               client_params={"http_options": types.HttpOptions(timeout=int(CALL_TIMEOUT_S * 1000))})  # ms


def choose_crew_model(context) -> None:
    """before_llm_call hook: send the call to the fallback while the worker model is busy."""
    model = active_model(WORKER_MODEL)
    llm = getattr(context, "llm", None)
    if llm is not None:
        llm.model = model
    limiter_for(model).wait()


_hooks_installed = False


def install_llm_hooks() -> None:
    global _hooks_installed
    if _hooks_installed:
        return
    from crewai.hooks import register_before_llm_call_hook

    register_before_llm_call_hook(choose_crew_model)
    _hooks_installed = True


def rate_limited(exc: BaseException) -> bool:
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def _backoff_s(exc: BaseException, attempt: int) -> float:
    """A 429 means the per-minute quota is spent, so wait out the window (30 s, 60 s, 90 s);
    a 503 overload clears quickly (2 s, 4 s, 8 s)."""
    return (30 * (attempt + 1) if rate_limited(exc) else 2 ** (attempt + 1)) + random.uniform(0, 1.5)


async def with_retry(fn: Callable[[], Awaitable[T]], attempts: int = 4, model: str | None = None) -> T:
    """Retry transient Gemini errors. With `model`, a failure marks that model busy, so the retry
    already runs on the fallback (the CrewAI hook picks it) and starts after one second."""
    for attempt in range(attempts):
        try:
            return await fn()
        except Exception as exc:
            if model:
                note_failure(model, exc)
            if attempt == attempts - 1 or not (is_retryable(exc) or (model and daily_quota_spent(exc))):
                raise
            switched = model is not None and not is_busy(active_model(model))
            await asyncio.sleep(1.0 if switched else _backoff_s(exc, attempt))
    raise RuntimeError("unreachable")


async def with_fallback(call: Callable[[str], Awaitable[T]], model: str, attempts: int = 4) -> tuple[T, str]:
    """Run call(model). If the model is busy, slow or out of quota, retry at once on the fallback
    model; back off only when there is nowhere else to go. Returns (result, model used)."""
    for attempt in range(attempts):
        current = active_model(model)
        try:
            return await call(current), current
        except Exception as exc:
            note_failure(current, exc)
            nxt = active_model(model)
            moved = nxt != current and not is_busy(nxt)
            if attempt == attempts - 1 or not (is_retryable(exc) or moved):
                raise
            await asyncio.sleep(0 if moved else _backoff_s(exc, attempt))  # both busy: wait, then try again
    raise RuntimeError("unreachable")


_client: genai.Client | None = None


def genai_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(http_options=types.HttpOptions(timeout=90_000))  # ms
    return _client


async def gemini_generate(prompt: str, agent: str, model: str = WORKER_MODEL, max_tokens: int = 1024,
                          response_schema: type | None = None, temperature: float = 0.0,
                          thinking: str = "low", timeout_s: float | None = None) -> str:
    """Direct google-genai call (guard judge, LLM-as-judge), recorded as an LLM event.

    Gemini 3.x models think before answering and thinking tokens count against
    max_tokens, so the thinking level is set explicitly (Lab-07 uses "low").
    Each attempt is cut off after timeout_s (default CALL_TIMEOUT_S) and moves to the fallback.
    """
    config = types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_tokens,
                                         thinking_config=types.ThinkingConfig(thinking_level=thinking))
    if response_schema is not None:
        config.response_mime_type = "application/json"
        config.response_schema = response_schema

    async def call(current: str) -> types.GenerateContentResponse:
        await asyncio.to_thread(limiter_for(current).wait)
        return await asyncio.wait_for(
            genai_client().aio.models.generate_content(model=current, contents=prompt, config=config),
            timeout_s or CALL_TIMEOUT_S)

    run = tracing.current()
    ctx = tracing.RunContext(run.run_id, agent, "google-genai") if run else None
    start = time.perf_counter()
    try:
        resp, used = await with_fallback(call, model)
    except Exception as exc:
        tracing.record_llm_call(model, 0, 0, None, status="ERROR", preview=str(exc), ctx=ctx)
        raise
    usage = resp.usage_metadata
    output_tokens = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
    tracing.record_llm_call(used, usage.prompt_token_count or 0, output_tokens,
                            round((time.perf_counter() - start) * 1000, 1), preview=resp.text or "", ctx=ctx)
    return resp.text or ""
