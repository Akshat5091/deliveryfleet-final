"""Pieces every architecture shares: run setup, guard, disruption events, reporting."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from src.guardrails import input_guard
from src.guardrails.business_guardrails import check_assignment
from src.mcp.tools.driver_db import get_driver
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
        return DisruptionEvent(kind="VEHICLE_BREAKDOWN", closed_corridor=first_leg if first_leg else None,
                               detail="Assigned vehicle reports a flat tyre and cannot continue.")
    if scenario == "rain_spike":
        return DisruptionEvent(kind="WEATHER_SPIKE", rainfall_mm=18.0,
                               detail="Cloudburst: rainfall jumps to 18 mm/h across Zone D.")
    return DisruptionEvent(kind="CORRIDOR_CLOSED", closed_corridor=None,
                           detail="Local road works on the last kilometre (intra-zone trip).")


def business_check(request: DeliveryRequest, driver_id: str, trip_minutes: float, hour: int) -> dict:
    driver = get_driver(driver_id)
    if driver is None:
        return {"action": "BLOCK", "violations": [f"{driver_id} not found"], "checks": []}
    r = check_assignment(driver, request.weight_kg, request.pickup, request.dropoff, trip_minutes, hour)
    return {"driver_id": driver_id, "action": r.action, "violations": r.violations, "checks": r.checks}


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
