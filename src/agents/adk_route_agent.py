"""Worker 3: Route Agent (Google ADK), served over A2A as its own process.

The Dispatch side sends a JSON task over A2A. This agent calls two MCP tools
(plan_route, then forecast_delay for the chosen corridor and vehicle) and
returns a RoutePlan. All six ADK callback hook points are used:

  before_agent  read run id from A2A metadata, screen the payload, log start
  before_model  shared rate limiter, start the LLM timer
  after_model   record tokens, latency and cost of every Gemini call
  before_tool   validate tool arguments; a returned dict short-circuits the tool
  after_tool    validate the tool result's types and ranges (Tier 2 guardrail)
  after_agent   log end of turn
"""
from __future__ import annotations

import json
import time
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.planners import PlanReActPlanner
from google.adk.tools import BaseTool, ToolContext
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.genai import types

from src.agents.llm import limiter
from src.config import MCP_URL, WORKER_MODEL
from src.guardrails.input_guard import regex_screen
from src.guardrails.tool_validator import ToolOutputError, validate
from src.observability import run_logger, tracing
from src.schemas.decisions import RoutePlan

AGENT = "route_agent"
FRAMEWORK = "Google-ADK"
_llm_started: dict[str, float] = {}

INSTRUCTION = """You are the Route Agent of a last-mile delivery fleet in four city zones (A-D).
You receive one JSON routing task. {style_hint?}

Steps:
1. Call plan_route with pickup, dropoff, closed_corridors and zone_d_rainfall_mm from the task.
2. Take the PRIMARY corridor. Call forecast_delay with its distance_km and speed_factor, the
   rainfall of the dropoff zone, hour_of_day, weight_kg, the driver's vehicle_type, both zones,
   and slack_mins = deadline_mins - primary base_eta_mins.
3. Return the RoutePlan: corridor, distance_km and base_eta_mins from the primary route,
   delay_mean/p10/p90 from the forecast, eta_mins = base_eta_mins + delay_mean_mins,
   the alternative corridor and its base ETA if one exists, and a two-sentence rationale
   naming the congestion level and why this corridor beats the alternative.
Use only numbers returned by the tools. Never invent a corridor."""

STYLE_HINTS = {
    "ReAct": "Reason step by step: state what you need, call a tool, read the observation, then continue.",
    "Direct": "Be concise: call the tools in order and return the result without commentary.",
}


def _ctx(callback_context: CallbackContext | ToolContext) -> tracing.RunContext | None:
    run_id = callback_context.state.get("run_id")
    return tracing.RunContext(run_id, AGENT, FRAMEWORK) if run_id else None


def _task_text(callback_context: CallbackContext) -> str:
    content = callback_context.user_content
    return "".join(p.text or "" for p in (content.parts or [])) if content else ""


def before_agent(callback_context: CallbackContext) -> types.Content | None:
    metadata: dict[str, Any] = {}
    run_config = getattr(callback_context._invocation_context, "run_config", None)
    if run_config and run_config.custom_metadata:
        metadata = run_config.custom_metadata.get("a2a_metadata", {}) or {}
    text = _task_text(callback_context)
    try:
        task = json.loads(text)
    except json.JSONDecodeError:
        task = {}
    run_id = metadata.get("run_id") or task.get("run_id") or "adk-standalone"
    style = metadata.get("prompt_style") or task.get("prompt_style") or "ReAct"
    callback_context.state["run_id"] = run_id
    callback_context.state["style_hint"] = STYLE_HINTS.get(style, STYLE_HINTS["ReAct"])
    callback_context.state["turn_started"] = time.time()
    run_logger.log_event(run_id, "agent_start", AGENT, FRAMEWORK, payload={"via": "A2A", "prompt_style": style})

    injection, _, _, _ = regex_screen(text)
    if injection:
        run_logger.log_event(run_id, "guardrail", AGENT, FRAMEWORK, name="adk_before_agent",
                             status="BLOCKED", payload={"hits": injection})
        return types.Content(role="model", parts=[types.Part(text=json.dumps({"error": "task rejected by guardrail"}))])
    return None


def before_model(callback_context: CallbackContext, llm_request: LlmRequest) -> LlmResponse | None:
    limiter.wait()
    _llm_started[callback_context.invocation_id] = time.perf_counter()
    return None


def after_model(callback_context: CallbackContext, llm_response: LlmResponse) -> LlmResponse | None:
    ctx = _ctx(callback_context)
    started = _llm_started.pop(callback_context.invocation_id, None)
    usage = llm_response.usage_metadata
    text = "".join(p.text or "" for p in (llm_response.content.parts or [])) if llm_response.content else ""
    calls = [p.function_call.name for p in (llm_response.content.parts or []) if p.function_call] if llm_response.content else []
    tracing.record_llm_call(
        WORKER_MODEL, (usage.prompt_token_count or 0) if usage else 0,
        ((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else 0,
        round((time.perf_counter() - started) * 1000, 1) if started else None,
        status="ERROR" if llm_response.error_code else "OK",
        preview=text or (f"function_call: {', '.join(calls)}" if calls else ""), ctx=ctx,
    )
    _llm_started[callback_context.invocation_id] = time.perf_counter()
    return None


def before_tool(tool: BaseTool, args: dict[str, Any], tool_context: ToolContext) -> dict | None:
    for key in ("pickup", "dropoff"):
        if key in args and not str(args[key]).startswith("Zone "):
            return {"error": f"{key} must look like 'Zone X - Place', got {args[key]!r}"}
    tool_context.state["tool_started"] = time.perf_counter()
    return None


def _structured(tool_response: Any) -> Any:
    if isinstance(tool_response, dict):
        if tool_response.get("structuredContent") is not None:
            return tool_response["structuredContent"]
        content = tool_response.get("content")
        if isinstance(content, list) and content and isinstance(content[0], dict) and "text" in content[0]:
            return json.loads(content[0]["text"])
    return tool_response


def after_tool(tool: BaseTool, args: dict[str, Any], tool_context: ToolContext, tool_response: Any) -> dict | None:
    ctx = _ctx(tool_context)
    started = tool_context.state.get("tool_started")
    ms = round((time.perf_counter() - started) * 1000, 1) if started else None
    try:
        data = validate(tool.name, _structured(tool_response))
    except (ToolOutputError, KeyError, TypeError, json.JSONDecodeError) as exc:
        tracing.record_tool_call(tool.name, args, ms, status="ERROR", result_preview=str(exc), ctx=ctx)
        if ctx:
            run_logger.log_event(ctx.run_id, "guardrail", AGENT, FRAMEWORK, name="tool_output_validation",
                                 status="BLOCKED", payload={"tool": tool.name, "error": str(exc)})
        return {"error": f"tool output rejected by validator: {exc}"}
    tracing.record_tool_call(tool.name, args, ms, result_preview=json.dumps(data, default=str), ctx=ctx)
    return None


def after_agent(callback_context: CallbackContext) -> types.Content | None:
    ctx = _ctx(callback_context)
    started = callback_context.state.get("turn_started")
    if ctx:
        run_logger.log_event(ctx.run_id, "agent_end", AGENT, FRAMEWORK,
                             duration_ms=round((time.time() - started) * 1000, 1) if started else None)
    return None


def build_route_agent() -> LlmAgent:
    tools = McpToolset(
        connection_params=StreamableHTTPConnectionParams(url=MCP_URL, timeout=30),
        tool_filter=["plan_route", "forecast_delay"],
    )
    return LlmAgent(
        name=AGENT,
        model=WORKER_MODEL,
        description="Computes the fastest legal road corridor and a delay forecast with an 80% band "
                    "for one delivery, using MCP routing and ML tools.",
        instruction=INSTRUCTION,
        tools=[tools],
        planner=PlanReActPlanner(),
        output_schema=RoutePlan,
        output_key="route_plan",
        generate_content_config=types.GenerateContentConfig(
            temperature=0.1,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
            safety_settings=[types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                                                 threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE)],
        ),
        before_agent_callback=before_agent,
        before_model_callback=before_model,
        after_model_callback=after_model,
        before_tool_callback=before_tool,
        after_tool_callback=after_tool,
        after_agent_callback=after_agent,
    )
