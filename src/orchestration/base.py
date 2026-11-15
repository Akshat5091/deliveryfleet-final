"""Pieces every architecture shares: run setup, guard, disruption events, reporting."""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field

from src.guardrails import input_guard
from src.guardrails.business_guardrails import check_assignment
from src.mcp.tools.driver_db import deadhead_minutes, get_driver
from src.observability import run_logger, tracing
from src.schemas.decisions import (
    DeliveryRequest,
    DisruptionEvent,
    FinalDeliveryReport,
    GuardVerdict,
)

SCENARIOS = ("normal", "corridor_closed", "breakdown", "rain_spike")


@dataclass
class RunSpec:
    message: str
    persona: str = "polite"
    prompt_style: str = "ReAct"
    scenario: str = "normal"
    order: dict | None = None
    hour_of_day: int = 14
    source: str = "web"
    run_id: str = field(default_factory=lambda: f"RUN-{time.strftime('%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}")


async def guard_input(spec: RunSpec) -> GuardVerdict:
    ctx = tracing.RunContext(spec.run_id, "input_guard", "Guardrail")
    token = tracing.set_context(ctx)
    try:
        verdict = await input_guard.screen(spec.message)
    finally:
        tracing.reset_context(token)
    # Never keep the raw message: PII and abuse are stored only in masked form.
    run_logger.set_input_text(spec.run_id, verdict.sanitized_text)
    run_logger.log_event(spec.run_id, "guardrail", "input_guard", "Guardrail", name="input_screen",
                         duration_ms=verdict.latency_ms, status=verdict.status, payload=verdict.model_dump())
    return verdict


def make_disruption(scenario: str, route: dict) -> DisruptionEvent | None:
    if scenario == "normal":
        return None
    corridor = route.get("corridor", "")
    zones = [z.strip() for z in corridor.split("->")]
    first_leg = f"{zones[0]} -> {zones[1]}" if len(zones) > 1 else None
    if scenario == "corridor_closed" and first_leg:
        return DisruptionEvent(kind="CORRIDOR_CLOSED", closed_corridor=first_leg,
                               detail=f"Police barricade after an accident closes {first_leg} for 2 hours.")
    if scenario == "breakdown":
        return DisruptionEvent(kind="VEHICLE_BREAKDOWN",
                               detail="Assigned vehicle reports a flat tyre and cannot continue; roads are open.")
    if scenario == "rain_spike":
        return DisruptionEvent(kind="WEATHER_SPIKE", rainfall_mm=18.0,
                               detail="Cloudburst: rainfall jumps to 18 mm/h across Zone D.")
    return DisruptionEvent(kind="CORRIDOR_CLOSED", closed_corridor=None,
                           detail="Local road works on the last kilometre (intra-zone trip).")


def business_check(request: DeliveryRequest, driver_id: str, trip_minutes: float, hour: int) -> dict:
    """Hard-rule check in code. Shift hours count the empty drive to the pickup as well as the trip."""
    driver = get_driver(driver_id)
    if driver is None:
        return {"driver_id": driver_id, "action": "BLOCK", "violations": [f"{driver_id} not found"], "checks": []}
    empty = deadhead_minutes(driver["current_zone"], request.pickup)
    r = check_assignment(driver, request.weight_kg, request.pickup, request.dropoff, trip_minutes + empty, hour)
    return {"driver_id": driver_id, "action": r.action, "violations": r.violations, "checks": r.checks,
            "trip_minutes": round(trip_minutes, 1), "empty_drive_minutes": empty}


def final_assignment(state: dict) -> tuple[str | None, float | None]:
    """The driver and ETA that actually stand after any exception replan."""
    dispatch, route, exc = state.get("dispatch") or {}, state.get("route") or {}, state.get("exception") or {}
    driver = exc.get("new_driver_id") or dispatch.get("driver_id")
    eta = exc.get("new_eta_mins") or route.get("eta_mins")
    return driver, eta


def final_check(state: dict, request: DeliveryRequest, hour: int) -> dict | None:
    """Re-run the hard rules on the final driver and final ETA (after a reroute or reassignment)."""
    driver, eta = final_assignment(state)
    if not driver or eta is None:
        return None
    return business_check(request, driver, float(eta), hour)


def mark_failed(spec: RunSpec, architecture: str, state: dict, provenance: list[dict], reason: str) -> None:
    """Last-resort close of a run so it never stays RUNNING (cancellation, store errors)."""
    try:
        run_logger.finish_run(spec.run_id, "FAILED", {
            "run_id": spec.run_id, "architecture": architecture, "persona": spec.persona, "outcome": "FAILED",
            "customer_message": "Sorry, we hit a technical problem while planning your delivery.",
            "error": reason[:300]}, state, provenance)
    except Exception:  # the store itself may be the thing that failed; nothing more can be recorded
        pass


def no_legal_driver(candidates: dict) -> dict | None:
    """If every driver is BLOCKed, return the business_check record explaining why; else None."""
    rows = candidates.get("candidates", [])
    if any(c["rule_action"] != "BLOCK" for c in rows):
        return None
    reasons = sorted({v.split(":")[0] for c in rows for v in c["violations"]})
    return {"driver_id": None, "action": "BLOCK", "checks": [],
            "violations": [f"No legal driver for this order ({', '.join(reasons) or 'no candidates'})"]}


def totals(run_id: str) -> tuple[int, int, float]:
    per_agent = run_logger.agent_totals(run_id)
    calls = sum(a["llm_calls"] for a in per_agent.values())
    tokens = sum(a["input_tokens"] + a["output_tokens"] for a in per_agent.values())
    cost = round(sum(a["cost_usd"] for a in per_agent.values()), 6)
    return calls, tokens, cost


def blocked_report(spec: RunSpec, architecture: str, verdict: GuardVerdict, started: float) -> FinalDeliveryReport:
    calls, tokens, cost = totals(spec.run_id)
    return FinalDeliveryReport(
        run_id=spec.run_id, architecture=architecture, persona=spec.persona, guard_status=verdict.status,
        order_id=(spec.order or {}).get("order_id"), outcome="BLOCKED",
        customer_message="We could not process this request because it contained instructions our system "
                         "is not allowed to follow. Please resend the delivery details only.",
        total_latency_s=round(time.perf_counter() - started, 3), llm_calls=calls, total_tokens=tokens, cost_usd=cost,
    )


async def gather_all(*aws):
    """Like asyncio.gather, but waits for every branch before raising the first error, so no
    agent keeps spending tokens after the run has already been marked failed."""
    results = await asyncio.gather(*aws, return_exceptions=True)
    for r in results:
        if isinstance(r, BaseException):
            raise r
    return results
