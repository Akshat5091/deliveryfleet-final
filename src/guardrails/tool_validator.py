"""Tier 2 guardrail: type and range checks on whatever a tool returns.

Wired as CrewAI after_tool_call hooks and ADK after_tool_callback so a broken
or tampered tool result is caught before the model reasons over it.
"""
from __future__ import annotations

import json
import math
from typing import Any


class ToolOutputError(ValueError):
    pass


def _finite(value: Any, name: str, lo: float = 0.0, hi: float = 1e6) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
        raise ToolOutputError(f"{name}={value!r} outside [{lo}, {hi}]")


def validate(tool_name: str, result: Any) -> dict | list:
    """Parse and validate a tool result. Returns the parsed object or raises ToolOutputError."""
    data = json.loads(result) if isinstance(result, str) else result
    if isinstance(data, dict) and "error" in data:
        return data
    if tool_name == "rank_drivers":
        if not isinstance(data, dict) or not isinstance(data.get("candidates"), list):
            raise ToolOutputError("rank_drivers must return {'candidates': [...]}")
        for c in data["candidates"]:
            _finite(c["hours_worked"], "hours_worked", 0, 24)
            _finite(c["capacity_kg"], "capacity_kg", 1, 1000)
            _finite(c["total_eta_mins"], "total_eta_mins", 0, 600)
            _finite(c["p_on_time"], "p_on_time", 0, 1)
    elif tool_name == "plan_route":
        if data.get("status") == "OK":
            _finite(data["primary"]["distance_km"], "distance_km", 0, 500)
            _finite(data["primary"]["base_eta_mins"], "base_eta_mins", 0, 600)
    elif tool_name == "forecast_delay":
        for key in ("mean_mins", "p10_mins", "p90_mins"):
            _finite(data[key], key, 0, 300)
        if not data["p10_mins"] <= data["mean_mins"] <= data["p90_mins"]:
            raise ToolOutputError("forecast band not ordered p10 <= mean <= p90")
    elif tool_name == "search_policies":
        if not isinstance(data.get("matches"), list):
            raise ToolOutputError("search_policies must return {'matches': [...]}")
    return data
