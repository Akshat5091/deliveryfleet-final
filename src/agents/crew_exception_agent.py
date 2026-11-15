"""Worker 4: Exception Agent (CrewAI), event-driven.

Sleeps until a disruption event (road closure, vehicle breakdown, rain spike)
is posted to the blackboard. Then it decides between rerouting, reassigning,
both, or escalating to a human. Rerouting is done by asking the ADK Route
Agent over A2A from inside a CrewAI tool, so this is CrewAI -> A2A -> ADK.
"""
from __future__ import annotations

import asyncio
import json

from crewai import Agent, Task
from crewai.tasks.task_output import TaskOutput
from crewai.tools import tool

from src.agents.a2a_client import request_route
from src.agents.crew_common import PROMPT_STYLES, mcp_server, run_task
from src.agents.llm import crew_llm
from src.guardrails.business_guardrails import check_assignment
from src.mcp.tools.driver_db import get_driver
from src.observability import tracing
from src.schemas.decisions import DeliveryRequest, DisruptionEvent, ExceptionReplan


def make_reroute_tool(request: DeliveryRequest, hour_of_day: int, prompt_style: str):
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
        return plan.model_dump_json()
    return request_reroute_via_a2a


def make_guardrail(request: DeliveryRequest, hour_of_day: int, original_driver: str):
    def _legal(output: TaskOutput) -> tuple[bool, object]:
        replan: ExceptionReplan | None = output.pydantic
        if replan is None:
            return False, "Return an ExceptionReplan."
        if replan.action in ("REASSIGN", "REROUTE_AND_REASSIGN"):
            if not replan.new_driver_id or replan.new_driver_id == original_driver:
                return False, "A reassignment needs a new_driver_id different from the original driver."
            driver = get_driver(replan.new_driver_id)
            if driver is None:
                return False, f"{replan.new_driver_id} is not in the fleet database."
            verdict = check_assignment(driver, request.weight_kg, request.pickup, request.dropoff,
                                       replan.new_eta_mins or 60.0, hour_of_day)
            if verdict.action == "BLOCK":
                return False, f"{replan.new_driver_id} is illegal: {'; '.join(verdict.violations)}"
        return True, output
    return _legal


async def recover(run_id: str, request: DeliveryRequest, dispatch: dict, route: dict, event: DisruptionEvent,
                  hour_of_day: int, prompt_style: str) -> ExceptionReplan:
    agent = Agent(
        role="Dynamic Exception & Recovery Coordinator",
        goal="Keep a disrupted delivery on time without breaking labour or zone rules, or escalate honestly.",
        backstory=("You only wake up when something goes wrong mid-delivery. Statutory driver-hour limits can "
                   "never be overridden, even for critical orders. " + PROMPT_STYLES[prompt_style]),
        llm=crew_llm(),
        tools=[make_reroute_tool(request, hour_of_day, prompt_style)],
        mcps=[mcp_server("rank_drivers", "get_driver")],
        max_iter=6,
        verbose=False,
    )
    remaining = max(request.deadline_mins - event.at_minute, 1)
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
        guardrail=make_guardrail(request, hour_of_day, dispatch.get("driver_id", "")),
        guardrail_max_retries=2,
    )
    return await run_task(run_id, "exception_agent", agent, task, ExceptionReplan)
