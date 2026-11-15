"""Lead Supervisor (CrewAI): intake at the start, review at the end.

Intake turns the customer's free-text message (already screened by the Input
Guard) into a validated DeliveryRequest. Review is the Lab-03 manager step:
read the whole blackboard, approve, escalate to a human, or name the one
worker that must run again.
"""
from __future__ import annotations

import json

from crewai import Agent, Task

from src.agents.crew_common import PROMPT_STYLES, run_task
from src.agents.llm import crew_llm
from src.schemas.decisions import DeliveryRequest, SupervisorReview


def _supervisor(prompt_style: str) -> Agent:
    return Agent(
        role="Lead Fleet Supervisor",
        goal="Turn customer requests into correct dispatch jobs and sign off only decisions that are legal and on time.",
        backstory=("You run the dispatch desk. You delegate work to specialist agents, read their results on the "
                   "shared blackboard, and you are the last check before a driver is sent. "
                   + PROMPT_STYLES[prompt_style]),
        llm=crew_llm(),
        allow_delegation=False,
        max_iter=3,
        verbose=False,
    )


async def intake(run_id: str, message: str, order_hint: dict | None, prompt_style: str) -> DeliveryRequest:
    agent = _supervisor(prompt_style)
    hint = f"\nOrder record from the order system: {json.dumps(order_hint)}" if order_hint else ""
    task = Task(
        description=(
            "Customer message (already screened; [REDACTED_...] marks removed abuse or personal data):\n"
            f"<<<\n{message}\n>>>{hint}\n\n"
            "Extract the delivery job. pickup and dropoff must be copied exactly in the form 'Zone X - Place'. "
            "If the order record is given, its numbers win over the message. priority is one of LOW, NORMAL, "
            "HIGH, CRITICAL. handling_notes keeps only operational instructions, rewritten politely."
        ),
        expected_output="A DeliveryRequest.",
        agent=agent,
        output_pydantic=DeliveryRequest,
    )
    return await run_task(run_id, "supervisor", agent, task, DeliveryRequest)


async def review(run_id: str, blackboard: dict, deadline_mins: int, prompt_style: str) -> SupervisorReview:
    agent = _supervisor(prompt_style)
    task = Task(
        description=(
            f"Blackboard snapshot:\n{json.dumps(blackboard, indent=1, default=str)}\n\n"
            "Check, in this order: is there a compliance brief, a dispatch decision and a route plan? "
            "Did the business_check section say BLOCK? Is the final ETA within the deadline "
            f"({deadline_mins} min)? If an exception section exists, does its plan fix the problem?\n"
            "- Something missing or inconsistent that a worker can fix: verdict REJECTED and rerun = that worker.\n"
            "- Legal but late (ETA beyond deadline) or confidence LOW, or exception action ESCALATE: "
            "verdict ESCALATE_TO_HUMAN, rerun = none.\n- Otherwise APPROVED, rerun = none.\n"
            "customer_message: professional and calm (max 3 sentences), driver first name + ETA when approved."
        ),
        expected_output="A SupervisorReview.",
        agent=agent,
        output_pydantic=SupervisorReview,
    )
    return await run_task(run_id, "supervisor", agent, task, SupervisorReview)
