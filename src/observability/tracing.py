"""OpenTelemetry tracing plus the run context every callback reads.

Trace tree per run:  mas.run  ->  agent.<name>  ->  llm / tool sub-spans.
Spans are exported to the shared SQLite store so the website can draw a
waterfall. The current run and agent live in context variables; CrewAI copies
them into its handler threads, and they travel to the ADK service inside the
A2A message metadata.
"""
from __future__ import annotations

import contextvars
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import Status, StatusCode

from src.config import PRICE_PER_M
from src.observability import run_logger


@dataclass(frozen=True)
class RunContext:
    run_id: str
    agent: str = ""
    framework: str = ""


_current: contextvars.ContextVar[RunContext | None] = contextvars.ContextVar("fleet_run", default=None)


def current() -> RunContext | None:
    return _current.get()


def set_context(ctx: RunContext | None) -> contextvars.Token:
    return _current.set(ctx)


def reset_context(token: contextvars.Token) -> None:
    _current.reset(token)


class SQLiteSpanExporter(SpanExporter):
    def export(self, spans: list[ReadableSpan]) -> SpanExportResult:
        for s in spans:
            attrs = dict(s.attributes or {})
            run_logger.save_span(
                span_id=format(s.context.span_id, "016x"), trace_id=format(s.context.trace_id, "032x"),
                parent_id=format(s.parent.span_id, "016x") if s.parent else None,
                run_id=str(attrs.get("run_id", "")), name=s.name, agent=str(attrs.get("agent", "")),
                start=s.start_time / 1e9, end=(s.end_time or s.start_time) / 1e9,
                status=s.status.status_code.name, attributes=attrs,
            )
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


def _provider(service: str) -> TracerProvider:
    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    provider.add_span_processor(SimpleSpanProcessor(SQLiteSpanExporter()))
    return provider


_tracer: trace.Tracer | None = None


def init(service: str) -> None:
    global _tracer
    _tracer = _provider(service).get_tracer("deliveryfleet")


def tracer() -> trace.Tracer:
    if _tracer is None:
        init("deliveryfleet-web")
    return _tracer


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    key = next((k for k in PRICE_PER_M if model and model.endswith(k)), None)
    if key is None:
        return 0.0
    price_in, price_out = PRICE_PER_M[key]
    return round((input_tokens * price_in + output_tokens * price_out) / 1_000_000, 7)


@contextmanager
def agent_span(run_id: str, agent: str, framework: str, **attrs) -> Iterator[RunContext]:
    """Span + start/end events around one agent's whole turn."""
    ctx = RunContext(run_id, agent, framework)
    token = set_context(ctx)
    start = time.perf_counter()
    run_logger.log_event(run_id, "agent_start", agent, framework, payload=attrs)
    with tracer().start_as_current_span(f"agent.{agent}", attributes={"run_id": run_id, "agent": agent,
                                                                      "framework": framework, **attrs}) as span:
        status = "OK"
        try:
            yield ctx
        except Exception as exc:
            status = "ERROR"
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            run_logger.log_event(run_id, "error", agent, framework, name=type(exc).__name__,
                                 status="ERROR", payload={"error": str(exc)[:500]})
            raise
        finally:
            ms = (time.perf_counter() - start) * 1000
            run_logger.log_event(run_id, "agent_end", agent, framework, duration_ms=round(ms, 1), status=status)
            reset_context(token)


def record_llm_call(model: str, input_tokens: int, output_tokens: int, duration_ms: float | None,
                    status: str = "OK", preview: str = "", ctx: RunContext | None = None) -> None:
    ctx = ctx or current()
    if ctx is None:
        return
    run_logger.log_event(ctx.run_id, "llm_call", ctx.agent, ctx.framework, name=model,
                         duration_ms=duration_ms, input_tokens=input_tokens, output_tokens=output_tokens,
                         cost_usd=cost_usd(model, input_tokens, output_tokens), status=status,
                         payload={"preview": preview[:400]})


def record_tool_call(tool: str, args: dict | str, duration_ms: float | None, status: str = "OK",
                     result_preview: str = "", ctx: RunContext | None = None) -> None:
    ctx = ctx or current()
    if ctx is None:
        return
    run_logger.log_event(ctx.run_id, "tool_call", ctx.agent, ctx.framework, name=tool, duration_ms=duration_ms,
                         status=status, payload={"args": args, "result": result_preview[:600], "via": "MCP"})
