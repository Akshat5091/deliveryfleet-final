"""Worker 2: Retrieval Agent (CrewAI + ChromaDB RAG through MCP).

Answers one question: which company and municipal rules bind THIS order?
It searches the policy knowledge base (search_policies MCP tool) and returns
a ComplianceBrief with quoted citations. The Task guardrail rejects briefs
that cite nothing or cite a file that is not in the knowledge base.
"""
from __future__ import annotations

import json

from crewai import Agent, Task
from crewai.tasks.task_output import TaskOutput

from src.agents.crew_common import PROMPT_STYLES, mcp_server, run_task
from src.agents.llm import crew_llm
from src.config import KB_DIR
from src.schemas.decisions import ComplianceBrief, DeliveryRequest

KB_FILES = {p.name for p in KB_DIR.glob("*.txt")}


def _grounded(output: TaskOutput) -> tuple[bool, object]:
    brief = output.pydantic
    if brief is None or not brief.citations:
        return False, "Cite at least one rule returned by search_policies."
    unknown = [c.source for c in brief.citations if c.source not in KB_FILES]
    if unknown:
        return False, f"Unknown sources {unknown}. Use the exact file names returned by search_policies: {sorted(KB_FILES)}."
    return True, output


async def retrieve_rules(run_id: str, request: DeliveryRequest, hour_of_day: int, prompt_style: str) -> ComplianceBrief:
    agent = Agent(
        role="Fleet Compliance Retrieval Specialist",
        goal="Find every company and municipal rule that constrains this delivery, quoting the source.",
        backstory=("You keep dispatch legal. You never rely on memory for rules: you search the policy "
                   "knowledge base and quote what it says. " + PROMPT_STYLES[prompt_style]),
        llm=crew_llm(),
        mcps=[mcp_server("search_policies")],
        max_iter=5,
        verbose=False,
    )
    task = Task(
        description=(
            f"Order (from the blackboard):\n{json.dumps(request.model_dump(), indent=1)}\n"
            f"Current time: {hour_of_day:02d}:00.\n\n"
            "Search the policy knowledge base at least twice: once for the pickup and dropoff zones' transit "
            "rules, once for driver working-hour limits and priority rules. Then report which rules apply now, "
            "whether an electric vehicle is mandatory, the tightest payload cap on the route (null if none), "
            "and the maximum trip length allowed for a driver near the fatigue threshold."
        ),
        expected_output="A ComplianceBrief with citations (source file + rule text) and a two-sentence summary.",
        agent=agent,
        output_pydantic=ComplianceBrief,
        guardrail=_grounded,
        guardrail_max_retries=2,
    )
    return await run_task(run_id, "retrieval_agent", agent, task, ComplianceBrief)
