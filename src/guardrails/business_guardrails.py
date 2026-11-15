"""Tier 3 guardrail: hard statutory and municipal rules checked in code.

An LLM may *propose* a driver; only this module decides whether the proposal
is legal. Prompts can be jailbroken, arithmetic cannot.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.config import FATIGUE_MAX_TRIP_MINS, FATIGUE_WARN_HOURS, MAX_SHIFT_HOURS
from src.mcp.tools.routing_engine import zone_of

ZONE_B_PEAK = range(17, 20)  # 17:00-20:00
ZONE_B_PEAK_LIMIT_KG = 50.0
ZONE_A_TRUCK_LIMIT_KG = 150.0
ZONE_A_EV_WINDOW = range(9, 19)
EV_TYPES = {"Electric Van", "Electric Scooter"}


@dataclass
class RuleResult:
    action: str = "ALLOW"  # ALLOW | FLAG_FOR_REVIEW | BLOCK
    violations: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)

    def _flag(self, level: str, message: str) -> None:
        self.violations.append(message)
        order = {"ALLOW": 0, "FLAG_FOR_REVIEW": 1, "BLOCK": 2}
        if order[level] > order[self.action]:
            self.action = level


def check_assignment(driver: dict, weight_kg: float, pickup: str, dropoff: str,
                     trip_minutes: float, hour_of_day: int) -> RuleResult:
    result = RuleResult()
    hours = float(driver["hours_worked"])
    trip_h = trip_minutes / 60.0

    if driver.get("status") != "AVAILABLE":
        result._flag("BLOCK", f"DRIVER_UNAVAILABLE: {driver['driver_id']} is {driver.get('status')}")
    if hours + trip_h > MAX_SHIFT_HOURS:
        result._flag("BLOCK", f"SAFETY_HOURS_VIOLATION: {hours:.1f} h + {trip_h:.2f} h trip > {MAX_SHIFT_HOURS} h")
    elif hours >= FATIGUE_WARN_HOURS and trip_minutes > FATIGUE_MAX_TRIP_MINS:
        result._flag("BLOCK", f"FATIGUE_RISK: {hours:.1f} h worked and trip {trip_minutes:.0f} min > {FATIGUE_MAX_TRIP_MINS:.0f} min")
    result.checks.append(f"hours {hours:.1f} + trip {trip_h:.2f} <= {MAX_SHIFT_HOURS}")

    if weight_kg > float(driver["capacity_kg"]):
        result._flag("BLOCK", f"CAPACITY_OVERLOAD: {weight_kg} kg > {driver['capacity_kg']} kg")
    result.checks.append(f"weight {weight_kg} <= capacity {driver['capacity_kg']}")

    zones = {zone_of(pickup), zone_of(dropoff)}
    if "Zone B" in zones and weight_kg > ZONE_B_PEAK_LIMIT_KG and hour_of_day in ZONE_B_PEAK:
        result._flag("BLOCK", f"ZONE_B_PEAK_FREIGHT: {weight_kg} kg > {ZONE_B_PEAK_LIMIT_KG} kg between 17:00-20:00")
    if "Zone A" in zones and driver["vehicle_type"] == "Truck" and weight_kg > ZONE_A_TRUCK_LIMIT_KG:
        result._flag("BLOCK", "ZONE_A_HEAVY_TRUCK: trucks > 150 kg barred from residential avenues")
    if "Zone A" in zones and hour_of_day in ZONE_A_EV_WINDOW and driver["vehicle_type"] not in EV_TYPES:
        result._flag("FLAG_FOR_REVIEW", "ZONE_A_EV_PREFERENCE: non-EV in Zone A between 09:00-19:00")
    result.checks.append("zone rules A/B/D evaluated")
    return result
