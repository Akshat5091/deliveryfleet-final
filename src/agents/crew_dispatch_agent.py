"""Worker 1: Dispatch Agent (CrewAI, ReAct).

Chooses the driver. It reads the order, the compliance brief and the
pre-fetched candidate list from the blackboard, may call rank_drivers /
get_driver through MCP, and explains the trade-off it made (on-time odds vs
distance vs fatigue). The Task guardrail re-checks the chosen driver against
the hard business rules in code; an illegal choice is sent back to the agent.
"""
from __future__ import annotations

import json

from crewai import Agent, Task
from crewai.tasks.task_output import TaskOutput

from src.agents.crew_common import PROMPT_STYLES, mcp_server, run_task
from src.agents.llm import crew_llm
from src.guardrails.business_guardrails import check_assignment
from src.mcp.tools.driver_db import get_driver
from src.schemas.decisions import ComplianceBrief, DeliveryRequest, DispatchDecision


def make_guardrail(request: DeliveryRequest, hour_of_day: int, eta_by_driver: dict[str, float]):
    def _legal(output: TaskOutput) -> tuple[bool, object]:
        decision: DispatchDecision | None = output.pydantic
        if decision is None:
            return False, "Return a DispatchDecision."
        driver = get_driver(decision.driver_id)
        if driver is None:
            return False, f"{decision.driver_id} is not in the fleet database."
        trip = eta_by_driver.get(driver["driver_id"], decision.total_eta_mins)
        verdict = check_assignment(driver, request.weight_kg, request.pickup, request.dropoff, trip, hour_of_day)
        if verdict.action == "BLOCK":
            return False, (f"{decision.driver_id} is illegal for this order: {'; '.join(verdict.violations)}. "
                           "Choose a driver whose rule_action is ALLOW or FLAG_FOR_REVIEW.")
        return True, output
    return _legal


async def decide(run_id: str, request: DeliveryRequest, compliance: ComplianceBrief | None, candidates: dict,
                 hour_of_day: int, prompt_style: str) -> DispatchDecision:
    agent = Agent(
        role="Fleet Dispatch & Allocation Strategist",
        goal="Assign the driver most likely to deliver on time without breaking any labour or zone rule.",
        backstory=("Senior dispatcher for an urban last-mile fleet. You weigh on-time probability, the empty "
                   "drive to the pickup, vehicle suitability and driver fatigue, and you always say why the "
                   "runner-up lost. " + PROMPT_STYLES[prompt_style]),
        llm=crew_llm(),
        mcps=[mcp_server("rank_drivers", "get_driver")],
        max_iter=6,
        verbose=False,
    )
    shortlist = [
        {k: c[k] for k in ("driver_id", "name", "vehicle_type", "hours_worked", "deadhead_km", "total_eta_mins",
                           "p_on_time", "rule_action", "violations")}
        for c in candidates.get("candidates", [])
    ]
    task = Task(
        description=(
            f"Order:\n{json.dumps(request.model_dump(), indent=1)}\nTime now: {hour_of_day:02d}:00.\n\n"
            f"Compliance brief from the Retrieval Agent:\n"
            f"{json.dumps(compliance.model_dump(), indent=1) if compliance else 'not available - call rank_drivers and apply its rule verdicts'}\n\n"
            f"Candidate drivers from the fleet database (rank_drivers, ML on-time probability included):\n"
            f"{json.dumps(shortlist, indent=1)}\n\n"
            "Pick one driver. Never pick a driver whose rule_action is BLOCK. Prefer ALLOW over FLAG_FOR_REVIEW "
            "unless the on-time probability gap is large. Fill rule_checks with each hard rule you verified and "
            "rejected with one line per serious alternative. Confidence is LOW if p_on_time < 0.6."
        ),
        expected_output="A DispatchDecision for exactly one driver.",
        agent=agent,
        output_pydantic=DispatchDecision,
        guardrail=make_guardrail(request, hour_of_day, {c["driver_id"]: c["total_eta_mins"] for c in shortlist}),
        guardrail_max_retries=2,
    )
    return await run_task(run_id, "dispatch_agent", agent, task, DispatchDecision)
