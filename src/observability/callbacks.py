"""CrewAI callbacks: monitoring (event listener) and guarding (tool hooks).

Monitoring: every LLM call (model, tokens, latency) and tool call (args,
result, latency) made inside a CrewAI agent is written to the event store
under the current run and agent.

Guarding: before a tool runs, the agent must be on that tool's allow-list;
after it runs, the result is type/range checked (Tier 2 guardrail). A failed
check replaces the result with an error the agent can reason about.
The ADK equivalents live in src/agents/adk_route_agent.py.
"""
from __future__ import annotations

import json
import threading
import time

from crewai.events import (
    BaseEventListener,
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
)
from crewai.hooks import register_after_tool_call_hook, register_before_tool_call_hook
from crewai.hooks.tool_hooks import ToolCallHookContext

from src.guardrails.tool_validator import ToolOutputError, validate
from src.observability import run_logger, tracing

TOOL_ALLOW_LIST: dict[str, set[str]] = {
    "retrieval_agent": {"search_policies"},
    "dispatch_agent": {"rank_drivers", "get_driver", "forecast_delay"},
    "exception_agent": {"rank_drivers", "get_driver", "request_reroute_via_a2a"},
    "supervisor": set(),
}


def base_tool_name(name: str) -> str:
    """CrewAI names MCP tools 'mcp_<host>_<port>_mcp_<tool>'; policies use the plain tool name."""
    return name.rsplit("_mcp_", 1)[-1]


def _usage(usage: dict | None) -> tuple[int, int]:
    if not usage:
        return 0, 0
    tin = usage.get("prompt_tokens") or usage.get("input_tokens") or usage.get("prompt_token_count") or 0
    tout = usage.get("completion_tokens") or usage.get("output_tokens") or usage.get("candidates_token_count") or 0
    return int(tin), int(tout)


class FleetEventListener(BaseEventListener):
    def __init__(self) -> None:
        self._started: dict[str, float] = {}
        self._lock = threading.Lock()
        super().__init__()

    def setup_listeners(self, bus) -> None:
        @bus.on(LLMCallStartedEvent)
        def _llm_start(source, event: LLMCallStartedEvent) -> None:
            with self._lock:
                self._started[event.call_id] = time.perf_counter()

        @bus.on(LLMCallCompletedEvent)
        def _llm_done(source, event: LLMCallCompletedEvent) -> None:
            with self._lock:
                t0 = self._started.pop(event.call_id, None)
            tin, tout = _usage(event.usage)
            ms = round((time.perf_counter() - t0) * 1000, 1) if t0 else None
            preview = event.response if isinstance(event.response, str) else json.dumps(event.response, default=str)
            tracing.record_llm_call(event.model or "", tin, tout, ms, preview=preview or "")

        @bus.on(LLMCallFailedEvent)
        def _llm_fail(source, event: LLMCallFailedEvent) -> None:
            with self._lock:
                self._started.pop(event.call_id, None)
            tracing.record_llm_call(event.model or "", 0, 0, None, status="ERROR", preview=event.error)

        @bus.on(ToolUsageFinishedEvent)
        def _tool_done(source, event: ToolUsageFinishedEvent) -> None:
            ms = (event.finished_at - event.started_at).total_seconds() * 1000
            tracing.record_tool_call(base_tool_name(event.tool_name), event.tool_args, round(ms, 1),
                                     result_preview=str(event.output))

        @bus.on(ToolUsageErrorEvent)
        def _tool_err(source, event: ToolUsageErrorEvent) -> None:
            tracing.record_tool_call(base_tool_name(event.tool_name), event.tool_args, None, status="ERROR",
                                     result_preview=str(getattr(event, "error", "")))


def _before_tool(ctx: ToolCallHookContext) -> bool | None:
    run = tracing.current()
    if run is None:
        return None
    allowed = TOOL_ALLOW_LIST.get(run.agent)
    if allowed is not None and base_tool_name(ctx.tool_name) not in allowed:
        run_logger.log_event(run.run_id, "guardrail", run.agent, run.framework, name="tool_allow_list",
                             status="BLOCKED", payload={"tool": ctx.tool_name})
        return False
    return None


def _after_tool(ctx: ToolCallHookContext) -> str | None:
    run = tracing.current()
    try:
        validate(base_tool_name(ctx.tool_name), ctx.tool_result)
    except (ToolOutputError, json.JSONDecodeError, KeyError, TypeError) as exc:
        if run:
            run_logger.log_event(run.run_id, "guardrail", run.agent, run.framework, name="tool_output_validation",
                                 status="BLOCKED", payload={"tool": ctx.tool_name, "error": str(exc)})
        return json.dumps({"error": f"tool output rejected by validator: {exc}"})
    return None


_installed = False


def install() -> None:
    """Register the listener and hooks once per process."""
    global _installed, _listener
    if _installed:
        return
    _listener = FleetEventListener()
    register_before_tool_call_hook(_before_tool)
    register_after_tool_call_hook(_after_tool)
    _installed = True
