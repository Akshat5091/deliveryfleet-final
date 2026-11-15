"""Turns the final shared state of a run into the FinalDeliveryReport (Pydantic)."""
from __future__ import annotations

import time

from src.mcp.tools.driver_db import get_driver
from src.observability import run_logger
from src.orchestration.base import RunSpec, final_assignment, totals
from src.schemas.decisions import FinalDeliveryReport


def outcome_of(state: dict) -> str:
    """Outcome decided in code from the blackboard; the LLM review alone can never mark a late or
    illegal delivery as dispatched."""
    exc, review, check = state.get("exception"), state.get("review") or {}, state.get("business_check") or {}
    if exc and exc.get("action") == "ESCALATE":
        return "ESCALATED"
    if review.get("verdict") not in (None, "APPROVED") or check.get("action") == "BLOCK":
        return "ESCALATED"
    _, eta = final_assignment(state)
    deadline = (state.get("request") or {}).get("deadline_mins")
    if eta is not None and deadline is not None and float(eta) > float(deadline):
        return "ESCALATED"
    return "RECOVERED_AFTER_DISRUPTION" if exc else "DISPATCHED"


def build_report(spec: RunSpec, architecture: str, state: dict, started: float,
                 error: str | None = None) -> FinalDeliveryReport:
    calls, tokens, cost = totals(spec.run_id)
    request, dispatch = state.get("request") or {}, state.get("dispatch") or {}
    route, exc, review = state.get("route") or {}, state.get("exception") or {}, state.get("review") or {}
    outcome = "FAILED" if error else outcome_of(state)
    driver_id = exc.get("new_driver_id") or dispatch.get("driver_id")
    driver_name = dispatch.get("driver_name")
    if exc.get("new_driver_id") and exc["new_driver_id"] != dispatch.get("driver_id"):
        driver_name = (get_driver(exc["new_driver_id"]) or {}).get("name")
    closed = (state.get("disruption") or {}).get("closed_corridor")
    # never report a route over the closed leg
    route_open = {} if closed and (route.get("corridor") or "").startswith(closed) else route
    message = review.get("customer_message") or exc.get("customer_update") or (
        "Sorry, we hit a technical problem while planning your delivery. A dispatcher will call you shortly."
        if error else "Your delivery has been scheduled.")
    return FinalDeliveryReport(
        run_id=spec.run_id, architecture=architecture, persona=spec.persona,
        guard_status=(state.get("guard") or {}).get("status", "ERROR"),
        order_id=request.get("order_id") or (spec.order or {}).get("order_id"), outcome=outcome,
        driver_id=driver_id, driver_name=driver_name,
        corridor=exc.get("new_corridor") or route_open.get("corridor"),
        eta_mins=exc.get("new_eta_mins") or route_open.get("eta_mins"),
        p_on_time=dispatch.get("p_on_time"), deadline_mins=request.get("deadline_mins"),
        exception_action=exc.get("action"), supervisor_verdict=review.get("verdict"),
        customer_message=message, total_latency_s=round(time.perf_counter() - started, 3),
        llm_calls=calls, total_tokens=tokens, cost_usd=cost,
    )


def finish(spec: RunSpec, report: FinalDeliveryReport, state: dict, provenance: list[dict]) -> FinalDeliveryReport:
    run_logger.finish_run(spec.run_id, report.outcome, report.model_dump(), state, provenance)
    return report
