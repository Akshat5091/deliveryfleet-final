"""MCP client used by the orchestrator for its own tool calls (driver pre-fetch).

Goes over the same streamable-HTTP transport as the agents, and records each
call as a tool_call event attributed to the calling agent.
"""
from __future__ import annotations

import json
import time

from fastmcp import Client

from src.config import MCP_URL
from src.guardrails.tool_validator import validate
from src.observability import tracing


async def call_tool(name: str, args: dict, ctx: tracing.RunContext) -> dict:
    start = time.perf_counter()
    try:
        async with Client(MCP_URL, timeout=60) as client:
            result = await client.call_tool(name, args)
        data = validate(name, result.structured_content if result.structured_content is not None else result.data)
    except Exception as exc:
        tracing.record_tool_call(name, args, round((time.perf_counter() - start) * 1000, 1), status="ERROR",
                                 result_preview=str(exc), ctx=ctx)
        raise
    tracing.record_tool_call(name, args, round((time.perf_counter() - start) * 1000, 1),
                             result_preview=json.dumps(data, default=str), ctx=ctx)
    return data
