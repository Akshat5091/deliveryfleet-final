"""Sequential architecture with peer-to-peer messaging (Lab-03 P2P pattern).

Same agents, but no shared blackboard and no supervisor loop: each agent gets
only the message the previous agent put in its inbox, and passes its own
result (plus what it received) on to the next one. Nothing runs in parallel,
and the Dispatch Agent has to fetch candidates itself with its MCP tools.
Used as a comparison point for latency, context size and decision quality.
"""
from __future__ import annotations

import json
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
from src.orchestration.blackboard_manager import PeerInbox
from src.orchestration.reporting import build_report, finish
from src.schemas.decisions import ComplianceBrief, DeliveryRequest, DisruptionEvent, FinalDeliveryReport

ARCHITECTURE = "sequential"


async def run(spec: RunSpec) -> FinalDeliveryReport:
    started = time.perf_counter()
    run_logger.start_run(spec.run_id, architecture=ARCHITECTURE, persona=spec.persona, prompt_style=spec.prompt_style,
                         scenario=spec.scenario, order_id=(spec.order or {}).get("order_id"), source=spec.source,
                         input_text=spec.message)
    inbox = PeerInbox(spec.run_id, on_change=lambda e: run_logger.log_event(
        spec.run_id, "p2p_message", e["agent"], "P2P", name=e["section"], payload=e))
    state: dict = {}
    with tracing.tracer().start_as_current_span("mas.run", attributes={
            "run_id": spec.run_id, "architecture": ARCHITECTURE, "persona": spec.persona,
            "prompt_style": spec.prompt_style, "scenario": spec.scenario}):
        try:
            verdict = await guard_input(spec)
            state["guard"] = verdict.model_dump()
            if verdict.status == "BLOCKED":
                return finish(spec, blocked_report(spec, ARCHITECTURE, verdict, started), state, inbox.provenance())

            request = await crew_supervisor.intake(spec.run_id, verdict.sanitized_text, spec.order, spec.prompt_style)
            state["request"] = request.model_dump()
            inbox.send("supervisor", "retrieval_agent", {"request": state["request"]})

            msg = inbox.receive("retrieval_agent")
            req = DeliveryRequest.model_validate(msg["request"])
            compliance = await crew_retrieval_agent.retrieve_rules(spec.run_id, req, spec.hour_of_day, spec.prompt_style)
            state["compliance"] = compliance.model_dump()
            inbox.send("retrieval_agent", "dispatch_agent", {**msg, "compliance": state["compliance"]})

            msg = inbox.receive("dispatch_agent")
            ctx = tracing.RunContext(spec.run_id, "dispatch_agent", "CrewAI")
            candidates = await call_tool("rank_drivers", {
                "pickup": req.pickup, "dropoff": req.dropoff, "weight_kg": req.weight_kg,
                "deadline_mins": req.deadline_mins, "hour_of_day": spec.hour_of_day}, ctx)
            state["candidates"] = candidates
            nobody = no_legal_driver(candidates)
            if nobody:
                state["business_check"] = nobody
                inbox.send("dispatch_agent", "supervisor", {**msg, "business_check": nobody})
                review = await crew_supervisor.review(spec.run_id, inbox.receive("supervisor"), req.deadline_mins,
                                                      spec.prompt_style)
                state["review"] = review.model_dump()
                return finish(spec, build_report(spec, ARCHITECTURE, state, started), state, inbox.provenance())
            decision = await crew_dispatch_agent.decide(spec.run_id, req, ComplianceBrief.model_validate(msg["compliance"]),
                                                        candidates, spec.hour_of_day, spec.prompt_style)
            state["candidates"], state["dispatch"] = candidates, decision.model_dump()
            inbox.send("dispatch_agent", "route_agent", {**msg, "dispatch": state["dispatch"]})

            msg = inbox.receive("route_agent")
            plan = await request_route({"pickup": req.pickup, "dropoff": req.dropoff, "weight_kg": req.weight_kg,
                                        "deadline_mins": req.deadline_mins, "hour_of_day": spec.hour_of_day,
                                        "vehicle_type": msg["dispatch"]["vehicle_type"], "closed_corridors": []},
                                       spec.run_id, spec.prompt_style, sender="dispatch_agent")
            state["route"] = plan.model_dump()
            state["business_check"] = business_check(req, decision.driver_id, plan.eta_mins, spec.hour_of_day)
            inbox.send("route_agent", "exception_agent", {**msg, "route": state["route"]})

            msg = inbox.receive("exception_agent")
            event = make_disruption(spec.scenario, state["route"])
            if event:
                state["disruption"] = event.model_dump()
                replan = await crew_exception_agent.recover(spec.run_id, req, msg["dispatch"], msg["route"],
                                                            DisruptionEvent.model_validate(state["disruption"]),
                                                            spec.hour_of_day, spec.prompt_style)
                state["exception"] = replan.model_dump()
                msg = {**msg, "exception": state["exception"]}
            inbox.send("exception_agent", "supervisor", msg)

            final_msg = inbox.receive("supervisor")
            run_logger.log_event(spec.run_id, "context_size", "supervisor", "P2P", name="inbox_chars",
                                 payload={"chars": len(json.dumps(final_msg, default=str))})
            review = await crew_supervisor.review(spec.run_id, final_msg, req.deadline_mins, spec.prompt_style)
            state["review"] = review.model_dump()
            report = build_report(spec, ARCHITECTURE, state, started)
        except Exception as exc:
            run_logger.log_event(spec.run_id, "error", "supervisor", "CrewAI", name=type(exc).__name__,
                                 status="ERROR", payload={"error": str(exc)[:800]})
            report = build_report(spec, ARCHITECTURE, state, started, error=str(exc))
        return finish(spec, report, state, inbox.provenance())
