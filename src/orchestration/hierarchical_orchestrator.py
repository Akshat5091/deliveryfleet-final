"""Hierarchical architecture (the one we deploy): 1 Supervisor + 4 workers on a Blackboard.

Flow (unchanged from Eval 2):
  Input Guard -> Supervisor intake
  -> in parallel: Retrieval Agent (RAG)  ||  driver pre-fetch from the fleet DB (MCP)
  -> Dispatch Agent (CrewAI) -> A2A -> Route Agent (Google ADK service)
  -> business-rule check in code
  -> Exception Agent, only if a disruption event is posted
  -> Supervisor review (Lab-03 manager loop: approve, escalate, or rerun one worker, max 2 reruns)

Workers never call each other's code. Each reads its scoped view of the
blackboard and writes its own section; the supervisor reads everything.
"""
from __future__ import annotations

import asyncio
import time

from src.agents import crew_dispatch_agent, crew_exception_agent, crew_retrieval_agent, crew_supervisor
from src.agents.a2a_client import request_route
from src.mcp.client import call_tool
from src.observability import run_logger, tracing
from src.orchestration.base import RunSpec, blocked_report, business_check, guard_input, make_disruption
from src.orchestration.blackboard_manager import Blackboard
from src.orchestration.reporting import build_report, finish
from src.schemas.decisions import ComplianceBrief, DeliveryRequest, DisruptionEvent, FinalDeliveryReport

ARCHITECTURE = "hierarchical"
MAX_RERUNS = 2


def _bb_logger(run_id: str):
    def log(p) -> None:
        run_logger.log_event(run_id, "blackboard", p.agent, "Blackboard", name=p.section, status=p.op,
                             payload=p.as_dict())
    return log


async def run(spec: RunSpec) -> FinalDeliveryReport:
    started = time.perf_counter()
    run_logger.start_run(spec.run_id, architecture=ARCHITECTURE, persona=spec.persona, prompt_style=spec.prompt_style,
                         scenario=spec.scenario, order_id=(spec.order or {}).get("order_id"), source=spec.source,
                         input_text=spec.message)
    bb = Blackboard(spec.run_id, on_change=_bb_logger(spec.run_id))
    sup = tracing.RunContext(spec.run_id, "supervisor", "CrewAI")
    with tracing.tracer().start_as_current_span("mas.run", attributes={
            "run_id": spec.run_id, "architecture": ARCHITECTURE, "persona": spec.persona,
            "prompt_style": spec.prompt_style, "scenario": spec.scenario}):
        try:
            verdict = await guard_input(spec)
            bb.write("guard", verdict, "input_guard", verdict.status)
            if verdict.status == "BLOCKED":
                return finish(spec, blocked_report(spec, ARCHITECTURE, verdict, started), bb.snapshot(), bb.provenance())

            request = await crew_supervisor.intake(spec.run_id, verdict.sanitized_text, spec.order, spec.prompt_style)
            bb.write("request", request, "supervisor", f"{request.order_id} {request.pickup} -> {request.dropoff}")

            async def gather_context() -> None:
                req = DeliveryRequest.model_validate(bb.read("request", "retrieval_agent"))
                compliance, candidates = await asyncio.gather(
                    crew_retrieval_agent.retrieve_rules(spec.run_id, req, spec.hour_of_day, spec.prompt_style),
                    call_tool("rank_drivers", {"pickup": req.pickup, "dropoff": req.dropoff, "weight_kg": req.weight_kg,
                                               "deadline_mins": req.deadline_mins, "hour_of_day": spec.hour_of_day}, sup),
                )
                bb.write("compliance", compliance, "retrieval_agent", compliance.summary[:120])
                bb.write("candidates", candidates, "supervisor", f"{len(candidates['candidates'])} drivers scored")

            async def dispatch() -> None:
                view = bb.view_for("dispatch_agent")
                decision = await crew_dispatch_agent.decide(
                    spec.run_id, DeliveryRequest.model_validate(view["request"]),
                    ComplianceBrief.model_validate(view["compliance"]), view["candidates"],
                    spec.hour_of_day, spec.prompt_style)
                bb.write("dispatch", decision, "dispatch_agent", f"{decision.driver_id} p_on_time={decision.p_on_time}")

            async def route() -> None:
                view = bb.view_for("route_agent")
                req = view["request"]
                task = {"pickup": req["pickup"], "dropoff": req["dropoff"], "weight_kg": req["weight_kg"],
                        "deadline_mins": req["deadline_mins"], "hour_of_day": spec.hour_of_day,
                        "vehicle_type": view["dispatch"]["vehicle_type"], "closed_corridors": []}
                plan = await request_route(task, spec.run_id, spec.prompt_style, sender="dispatch_agent")
                bb.write("route", plan, "route_agent", f"{plan.corridor} eta={plan.eta_mins}")
                bb.write("business_check", business_check(DeliveryRequest.model_validate(req), view["dispatch"]["driver_id"],
                                                          plan.eta_mins, spec.hour_of_day),
                         "business_guardrail", "hard-rule check")

            await gather_context()
            await dispatch()
            await route()

            event = make_disruption(spec.scenario, bb.snapshot()["route"])
            if event:
                bb.write("disruption", event, "event_bus", f"{event.kind}: {event.detail}")
                view = bb.view_for("exception_agent")
                replan = await crew_exception_agent.recover(
                    spec.run_id, DeliveryRequest.model_validate(view["request"]), view["dispatch"], view["route"],
                    DisruptionEvent.model_validate(view["disruption"]), spec.hour_of_day, spec.prompt_style)
                bb.write("exception", replan, "exception_agent", replan.action)
                if replan.new_driver_id:
                    bb.write("business_check", business_check(request, replan.new_driver_id, replan.new_eta_mins or 60.0,
                                                              spec.hour_of_day), "business_guardrail", "re-check after replan")

            for attempt in range(MAX_RERUNS + 1):
                review = await crew_supervisor.review(spec.run_id, bb.view_for("supervisor"), request.deadline_mins,
                                                      spec.prompt_style)
                bb.write("review", review, "supervisor", f"{review.verdict} rerun={review.rerun}")
                if review.rerun == "none" or attempt == MAX_RERUNS:
                    break
                run_logger.log_event(spec.run_id, "supervisor_rerun", "supervisor", "CrewAI", name=review.rerun,
                                     payload={"issues": review.issues, "attempt": attempt + 1})
                if review.rerun == "retrieval_agent":
                    await gather_context()
                    await dispatch()
                elif review.rerun == "dispatch_agent":
                    await dispatch()
                await route()

            report = build_report(spec, ARCHITECTURE, bb.snapshot(), started)
        except Exception as exc:
            run_logger.log_event(spec.run_id, "error", "supervisor", "CrewAI", name=type(exc).__name__,
                                 status="ERROR", payload={"error": str(exc)[:800]})
            report = build_report(spec, ARCHITECTURE, bb.snapshot(), started, error=str(exc))
        return finish(spec, report, bb.snapshot(), bb.provenance())
