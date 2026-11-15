"""Shared plumbing for the CrewAI agents.

run_task() runs one CrewAI Task with one Agent inside an observability span:
CrewAI's event listener records the LLM and tool calls, the Task's
output_pydantic forces the structured result, and the Task guardrail (a plain
Python check) can reject the result and make CrewAI retry with feedback.
"""
from __future__ import annotations

from typing import TypeVar

from crewai import Agent, Crew, Process, Task
from crewai.mcp import MCPServerHTTP
from crewai.mcp.filters import create_static_tool_filter
from pydantic import BaseModel

from src.agents.llm import install_llm_hooks, with_retry
from src.config import MCP_URL
from src.observability import callbacks, tracing

M = TypeVar("M", bound=BaseModel)

PROMPT_STYLES = {
    "ReAct": ("Work in explicit Thought -> Action -> Observation steps: say what you need, call a tool, "
              "read what it returned, and only then decide. Ground every number in a tool observation."),
    "Direct": "Answer directly and briefly. Use tools only where a number is needed. No step-by-step narration.",
}


class AgentOutputError(RuntimeError):
    pass


def mcp_server(*tool_names: str) -> MCPServerHTTP:
    return MCPServerHTTP(url=MCP_URL, tool_filter=create_static_tool_filter(allowed_tool_names=list(tool_names)),
                         cache_tools_list=True)


def setup() -> None:
    callbacks.install()
    install_llm_hooks()


async def run_task(run_id: str, agent_name: str, agent: Agent, task: Task, output_model: type[M]) -> M:
    setup()
    with tracing.agent_span(run_id, agent_name, "CrewAI", role=agent.role):
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
        result = await with_retry(lambda: crew.kickoff_async())
        if result.pydantic is None:
            raise AgentOutputError(f"{agent_name} returned no structured output: {str(result.raw)[:300]}")
        return output_model.model_validate(result.pydantic.model_dump())
