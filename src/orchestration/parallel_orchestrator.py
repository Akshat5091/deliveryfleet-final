"""Parallel (broadcast) architecture.

After intake the Supervisor broadcasts the same request to the Retrieval,
Dispatch and Route agents at once. They work blind to each other: Dispatch
has no compliance brief, Route does not know the vehicle and plans for a
standard Van. The Supervisor then reconciles. Fastest wall-clock, but
conflicts are only caught at the end. Results still land on a blackboard so
the run is auditable.
"""
from __future__ import annotations

import asyncio
import time

from src.agents import crew_dispatch_agent, crew_exception_agent, crew_retrieval_agent, crew_supervisor
from src.agents.a2a_client import request_route
from src.mcp.client import call_tool
from src.observability import run_logger, tracing
from src.orchestration.base import (
    RunSpec,
    blocked_report,
    business_check,
    guard_input,
    make_disruption,
    no_legal_driver,
)
from src.orchestration.blackboard_manager import Blackboard
from src.orchestration.hierarchical_orchestrator import _bb_logger
from src.orchestration.reporting import build_report, finish
from src.schemas.decisions import DeliveryRequest, DisruptionEvent, FinalDeliveryReport

ARCHITECTURE = "parallel"


async def run(spec: RunSpec) -> FinalDeliveryReport:
    started = time.perf_counter()
    run_logger.start_run(spec.run_id, architecture=ARCHITECTURE, persona=spec.persona, prompt_style=spec.prompt_style,
                         scenario=spec.scenario, order_id=(spec.order or {}).get("order_id"), source=spec.source,
                         input_text=spec.message)
    bb = Blackboard(spec.run_id, on_change=_bb_logger(spec.run_id))
    with tracing.tracer().start_as_current_span("mas.run", attributes={
            "run_id": spec.run_id, "architecture": ARCHITECTURE, "persona": spec.persona,
            "prompt_style": spec.prompt_style, "scenario": spec.scenario}):
        try:
            verdict = await guard_input(spec)
            bb.write("guard", verdict, "input_guard", verdict.status)
            if verdict.status == "BLOCKED":
                return finish(spec, blocked_report(spec, ARCHITECTURE, verdict, started), bb.snapshot(), bb.provenance())

            request = await crew_supervisor.intake(spec.run_id, verdict.sanitized_text, spec.order, spec.prompt_style)
            bb.write("request", request, "supervisor", "broadcast to all workers")
            run_logger.log_event(spec.run_id, "broadcast", "supervisor", "CrewAI", name="request",
                                 payload={"to": ["retrieval_agent", "dispatch_agent", "route_agent"]})
            ctx = tracing.RunContext(spec.run_id, "dispatch_agent", "CrewAI")

            async def blind_dispatch():
                candidates = await call_tool("rank_drivers", {
                    "pickup": request.pickup, "dropoff": request.dropoff, "weight_kg": request.weight_kg,
                    "deadline_mins": request.deadline_mins, "hour_of_day": spec.hour_of_day}, ctx)
                if no_legal_driver(candidates):
                    return candidates, None
                return candidates, await crew_dispatch_agent.decide(spec.run_id, request, None, candidates,
                                                                    spec.hour_of_day, spec.prompt_style)

            compliance, (candidates, decision), plan = await asyncio.gather(
                crew_retrieval_agent.retrieve_rules(spec.run_id, request, spec.hour_of_day, spec.prompt_style),
                blind_dispatch(),
                request_route({"pickup": request.pickup, "dropoff": request.dropoff, "weight_kg": request.weight_kg,
                               "deadline_mins": request.deadline_mins, "hour_of_day": spec.hour_of_day,
                               "vehicle_type": "Van", "closed_corridors": []},
                              spec.run_id, spec.prompt_style, sender="supervisor"),
            )
            bb.write("compliance", compliance, "retrieval_agent")
            bb.write("candidates", candidates, "supervisor")
            bb.write("route", plan, "route_agent", "planned for a generic Van")
            if decision is None:
                bb.write("business_check", no_legal_driver(candidates), "business_guardrail", "no legal driver")
            else:
                bb.write("dispatch", decision, "dispatch_agent")
                bb.write("business_check", business_check(request, decision.driver_id, plan.eta_mins,
                                                          spec.hour_of_day), "business_guardrail")

            event = make_disruption(spec.scenario, plan.model_dump()) if decision else None
            if event:
                bb.write("disruption", event, "event_bus", f"{event.kind}: {event.detail}")
                replan = await crew_exception_agent.recover(spec.run_id, request, decision.model_dump(), plan.model_dump(),
                                                            DisruptionEvent.model_validate(event.model_dump()),
                                                            spec.hour_of_day, spec.prompt_style)
                bb.write("exception", replan, "exception_agent", replan.action)

            review = await crew_supervisor.review(spec.run_id, bb.view_for("supervisor"), request.deadline_mins,
                                                  spec.prompt_style)
            bb.write("review", review, "supervisor", review.verdict)
            report = build_report(spec, ARCHITECTURE, bb.snapshot(), started)
        except Exception as exc:
            run_logger.log_event(spec.run_id, "error", "supervisor", "CrewAI", name=type(exc).__name__,
                                 status="ERROR", payload={"error": str(exc)[:800]})
            report = build_report(spec, ARCHITECTURE, bb.snapshot(), started, error=str(exc))
        return finish(spec, report, bb.snapshot(), bb.provenance())


