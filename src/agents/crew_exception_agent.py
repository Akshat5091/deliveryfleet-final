"""Worker 4: Exception Agent (CrewAI), event-driven.

Sleeps until a disruption event (road closure, vehicle breakdown, rain spike)
is posted to the blackboard. Then it decides between rerouting, reassigning,
both, or escalating to a human. Rerouting is done by asking the ADK Route
Agent over A2A from inside a CrewAI tool, so this is CrewAI -> A2A -> ADK.
"""

import asyncio
import json
from typing import Any

from crewai import Agent, Task
from crewai.tasks.task_output import TaskOutput
from crewai.tools import tool

from src.agents.a2a_client import request_route
from src.agents.crew_common import PROMPT_STYLES, mcp_server, run_task
from src.agents.llm import crew_llm
from src.observability import tracing
from src.orchestration.base import business_check
from src.schemas.decisions import DeliveryRequest, DisruptionEvent, ExceptionReplan


def make_reroute_tool(request: DeliveryRequest, hour_of_day: int, prompt_style: str, plans: list[dict]):
    """`plans` collects every RoutePlan the Route Agent returned, so the guardrail and the orchestrator
    can hold the agent's ETA to the tool's number."""
    @tool("request_reroute_via_a2a")
    def request_reroute_via_a2a(vehicle_type: str, closed_corridors: list[str], zone_d_rainfall_mm: float = 0.0) -> str:
        """Ask the Route Agent (Google ADK service, over the A2A protocol) for a new corridor that avoids
        the closed corridors. closed_corridors use the form 'Zone A -> Zone B'. Returns the RoutePlan JSON."""
        ctx = tracing.current()
        run_id = ctx.run_id if ctx else "exception-standalone"
        task = {**request.model_dump(include={"pickup", "dropoff", "weight_kg", "deadline_mins"}),
                "hour_of_day": hour_of_day, "vehicle_type": vehicle_type,
                "closed_corridors": closed_corridors, "zone_d_rainfall_mm": zone_d_rainfall_mm or None}
        plan = asyncio.run(request_route(task, run_id, prompt_style, sender="exception_agent"))
        plans.append(plan.model_dump())
        return plan.model_dump_json(exclude={"alternative_corridor", "alternative_eta_mins"})
    return request_reroute_via_a2a


def routed_eta(replan: ExceptionReplan, plans: list[dict]) -> float | None:
    """The Route Agent's ETA for the corridor the replan chose (None if it never asked for that corridor)."""
    match = [p for p in plans if p["corridor"] == replan.new_corridor]
    return match[-1]["eta_mins"] if match else None


def make_guardrail(request: DeliveryRequest, hour_of_day: int, original_driver: str, plans: list[dict]):
    """Any plan that keeps a driver on the road (same or new) is re-checked against the hard rules, and a
    reroute must use the corridor and ETA the Route Agent actually returned."""
    def _legal(output: TaskOutput) -> tuple[bool, Any]:
        replan: ExceptionReplan | None = output.pydantic
        if replan is None:
            return False, "Return an ExceptionReplan."
        if replan.action == "ESCALATE":
            return True, output
        if replan.action in ("REROUTE", "REROUTE_AND_REASSIGN"):
            eta = routed_eta(replan, plans)
            if eta is None:
                return False, ("Call request_reroute_via_a2a and use the corridor it returns; "
                               "a reroute ETA must come from the Route Agent.")
            if abs((replan.new_eta_mins or 0) - eta) > 1.0:
                return False, (f"new_eta_mins must be the Route Agent's eta_mins ({eta}) for {replan.new_corridor}. "
                               f"If {eta} is above the {request.deadline_mins}-minute deadline, choose ESCALATE.")
            if eta > request.deadline_mins:
                return False, (f"The Route Agent's ETA {eta} min misses the {request.deadline_mins}-minute deadline; "
                               "choose ESCALATE or a faster legal option.")
        if replan.action in ("REASSIGN", "REROUTE_AND_REASSIGN") and (
                not replan.new_driver_id or replan.new_driver_id == original_driver):
            return False, "A reassignment needs a new_driver_id different from the original driver."
        driver = replan.new_driver_id or original_driver
        verdict = business_check(request, driver, replan.new_eta_mins or 60.0, hour_of_day)
        if verdict["action"] == "BLOCK":
            return False, f"{driver} is illegal for this plan: {'; '.join(verdict['violations'])}"
        return True, output
    return _legal


async def recover(run_id: str, request: DeliveryRequest, dispatch: dict, route: dict, event: DisruptionEvent,
                  hour_of_day: int, prompt_style: str) -> ExceptionReplan:
    plans: list[dict] = []
    agent = Agent(
        role="Dynamic Exception & Recovery Coordinator",
        goal="Keep a disrupted delivery on time without breaking labour or zone rules, or escalate honestly.",
        backstory=("You only wake up when something goes wrong mid-delivery. Statutory driver-hour limits can "
                   "never be overridden, even for critical orders. " + PROMPT_STYLES[prompt_style]),
        llm=crew_llm(),
        tools=[make_reroute_tool(request, hour_of_day, prompt_style, plans)],
        mcps=[mcp_server("rank_drivers", "get_driver")],
        max_iter=6,
        verbose=False,
    )
    remaining = max(request.deadline_mins - event.at_minute, 1)
    # The old plan's alternative is a base time without a delay forecast; showing it invites a wrong ETA.
    route = {k: v for k, v in route.items() if not k.startswith("alternative_")}
    task = Task(
        description=(
            f"DISRUPTION at minute {event.at_minute:.0f} of the trip: {event.kind} - {event.detail}\n"
            f"closed_corridor: {event.closed_corridor}, rainfall_mm: {event.rainfall_mm}\n\n"
            f"Order: {json.dumps(request.model_dump())}\nMinutes left to the deadline: {remaining:.0f}\n"
            f"Current assignment: {json.dumps(dispatch)}\nCurrent route plan: {json.dumps(route)}\n\n"
            "Decide:\n- CORRIDOR_CLOSED or WEATHER_SPIKE: call request_reroute_via_a2a for the current vehicle.\n"
            "- VEHICLE_BREAKDOWN: call rank_drivers excluding the current driver (exclude_ids) with "
            f"deadline_mins={remaining:.0f}, pick a legal driver, then reroute if a corridor is also closed.\n"
            "- If no legal option reaches the drop before the deadline, choose ESCALATE.\n"
            "minutes_vs_doing_nothing = (old ETA + expected disruption delay) - new ETA. Write a short, polite "
            "customer_update without blaming anyone."
        ),
        expected_output="An ExceptionReplan.",
        agent=agent,
        output_pydantic=ExceptionReplan,
        guardrail=make_guardrail(request, hour_of_day, dispatch.get("driver_id", ""), plans),
        guardrail_max_retries=2,
    )
    replan = await run_task(run_id, "exception_agent", agent, task, ExceptionReplan)
    if replan.action in ("REROUTE", "REROUTE_AND_REASSIGN"):
        # Code is the source of truth for the ETA: always the Route Agent's number, never the LLM's copy.
        eta = routed_eta(replan, plans)
        if eta is None or eta > request.deadline_mins:
            replan = replan.model_copy(update={
                "action": "ESCALATE", "new_driver_id": None, "new_eta_mins": eta,
                "customer_update": ("A road on your delivery route has closed and the detour cannot reach you within "
                                    "the promised time. A dispatcher will call you shortly with the options.")})
        else:
            replan = replan.model_copy(update={"new_eta_mins": eta})
    return replan
