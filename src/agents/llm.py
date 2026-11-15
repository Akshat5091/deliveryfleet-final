"""Gemini access for both frameworks: model factory, rate limiter, retries.

Free-tier Gemini keys allow a limited number of requests per minute, so every
LLM call in this process waits on one token bucket (LLM_RPM). Transient 429 /
503 errors are retried with exponential backoff plus jitter (the Lab-04
run_with_retry pattern).
"""
from __future__ import annotations

import asyncio
import random
import threading
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from crewai import LLM
from google import genai
from google.genai import types

from src.config import LLM_RPM, WORKER_MODEL
from src.observability import tracing

T = TypeVar("T")
RETRYABLE = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "overloaded", "rate limit", "500 INTERNAL")


class RateLimiter:
    """Thread-safe minimum spacing between LLM calls in this process."""

    def __init__(self, rpm: int) -> None:
        self.interval = 60.0 / max(rpm, 1)
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self.interval
        if slot > now:
            time.sleep(slot - now)


# Two processes call Gemini (the web app and the ADK route service), each with its own limiter,
# so each gets half of the key's per-minute budget and together they stay under LLM_RPM.
limiter = RateLimiter(max(LLM_RPM // 2, 1))


def is_retryable(exc: BaseException) -> bool:
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
        limiter.wait()
        return None

    register_before_llm_call_hook(_throttle)
    _hooks_installed = True


async def with_retry(fn: Callable[[], Awaitable[T]], attempts: int = 4) -> T:
    for attempt in range(attempts):
        try:
            return await fn()
        except Exception as exc:
            if attempt == attempts - 1 or not is_retryable(exc):
                raise
            await asyncio.sleep(2 ** (attempt + 1) + random.uniform(0, 1.5))
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
        await asyncio.to_thread(limiter.wait)
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
