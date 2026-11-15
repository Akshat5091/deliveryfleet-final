"""Pydantic v2 contracts for everything agents write to the blackboard."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src.mcp.tools.routing_engine import zone_of

Priority = Literal["LOW", "NORMAL", "HIGH", "CRITICAL"]


class DeliveryRequest(BaseModel):
    """Structured order extracted by the Supervisor from the customer's message."""
    order_id: str
    pickup: str = Field(description="Exact location string, e.g. 'Zone A - Warehouse 1'")
    dropoff: str = Field(description="Exact location string, e.g. 'Zone B - Tech Park'")
    weight_kg: float = Field(gt=0, le=300)
    priority: Priority
    deadline_mins: int = Field(gt=0, le=480)
    handling_notes: str = Field(default="", description="Operational notes only, rewritten politely; no insults")

    @field_validator("pickup", "dropoff")
    @classmethod
    def _inside_network(cls, value: str) -> str:
        zone_of(value)
        return value.strip()


class GuardVerdict(BaseModel):
    status: Literal["ALLOW", "SANITIZED", "BLOCKED"]
    injection_hits: list[str] = []
    toxicity_hits: list[str] = []
    pii_masked: list[str] = []
    judge_label: str | None = None
    sanitized_text: str
    latency_ms: float


class PolicyCitation(BaseModel):
    source: str = Field(description="Knowledge-base file the rule came from")
    rule: str = Field(description="The rule text, quoted or tightly paraphrased")


class ComplianceBrief(BaseModel):
    """Retrieval Agent output: zone and labour rules that bind this order."""
    citations: list[PolicyCitation]
    ev_required: bool = Field(description="True if an EV is mandatory for this order at this time")
    max_payload_kg: float | None = Field(default=None, description="Tightest payload cap on the route, if any")
    max_trip_minutes_for_fatigued: float = Field(default=30.0)
    summary: str = Field(description="Two sentences a dispatcher can act on")


class RoutePlan(BaseModel):
    """Route Agent (ADK) output, returned over A2A."""
    corridor: str
    distance_km: float
    base_eta_mins: float
    delay_mean_mins: float
    delay_p10_mins: float
    delay_p90_mins: float
    eta_mins: float = Field(description="base_eta_mins + delay_mean_mins")
    alternative_corridor: str | None = None
    alternative_eta_mins: float | None = Field(
        default=None, description="the alternative corridor's base travel time from plan_route, without a delay "
                                  "forecast; compare it with base_eta_mins, not with eta_mins")
    rationale: str


class DispatchDecision(BaseModel):
    """Dispatch Agent output."""
    driver_id: str
    driver_name: str
    vehicle_type: str
    total_eta_mins: float = Field(description="Deadhead + route ETA, minutes")
    p_on_time: float = Field(ge=0, le=1)
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    rule_checks: list[str] = Field(description="Each hard rule checked, e.g. 'hours 4.5 + trip <= 9.0: pass'")
    rejected: list[str] = Field(description="'D03: 8.8 h worked, trip > 30 min' style lines")
    reasoning: str = Field(description="Short trade-off explanation")


class ExceptionReplan(BaseModel):
    """Exception Agent output after a mid-transit disruption."""
    trigger: str
    action: Literal["REROUTE", "REASSIGN", "REROUTE_AND_REASSIGN", "ESCALATE"]
    new_driver_id: str | None = None
    new_corridor: str | None = None
    new_eta_mins: float | None = None
    minutes_vs_doing_nothing: float | None = Field(default=None, description="Positive = minutes saved")
    customer_update: str


class SupervisorReview(BaseModel):
    verdict: Literal["APPROVED", "ESCALATE_TO_HUMAN", "REJECTED"]
    rerun: Literal["none", "retrieval_agent", "dispatch_agent", "route_agent"] = Field(
        default="none", description="Lab-03 manager loop: which worker must run again, or 'none'")
    issues: list[str]
    customer_message: str = Field(description="Professional message to the customer, max 3 sentences")


class DisruptionEvent(BaseModel):
    kind: Literal["CORRIDOR_CLOSED", "VEHICLE_BREAKDOWN", "WEATHER_SPIKE"]
    detail: str
    closed_corridor: str | None = None
    rainfall_mm: float | None = None
    at_minute: float = 10.0


class FinalDeliveryReport(BaseModel):
    """The single structured output of one MAS run."""
    run_id: str
    architecture: str
    persona: str
    guard_status: str
    order_id: str | None
    outcome: Literal["DISPATCHED", "RECOVERED_AFTER_DISRUPTION", "ESCALATED", "BLOCKED", "FAILED"]
    driver_id: str | None = None
    driver_name: str | None = None
    corridor: str | None = None
    eta_mins: float | None = None
    p_on_time: float | None = None
    deadline_mins: int | None = None
    exception_action: str | None = None
    supervisor_verdict: str | None = None
    customer_message: str
    total_latency_s: float
    llm_calls: int
    total_tokens: int
    cost_usd: float

    def to_markdown(self) -> str:
        rows = [(k, v) for k, v in self.model_dump().items() if v not in (None, "")]
        lines = ["| Field | Value |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in rows]
        return "\n".join(lines)
