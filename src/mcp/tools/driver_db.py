"""Fleet database (SQLite) and candidate-driver ranking.

rank_drivers combines three things for every driver: the database record,
the routing engine's ETA and the delay model's P(on time), then runs the
business guardrail so the Dispatch Agent sees which drivers are legal and why
the others are not. The agent still makes the final choice and explains it.
"""
from __future__ import annotations

import csv
import sqlite3
import threading
from contextlib import closing

from src.config import DRIVERS_CSV, RUNTIME_DIR
from src.guardrails.business_guardrails import check_assignment
from src.mcp.tools.delay_predictor import forecast
from src.mcp.tools.routing_engine import FREE_FLOW_KMH, deadhead_km, plan_route, zone_of

DB_PATH = RUNTIME_DIR / "fleet_db.sqlite3"
_init_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.row_factory = sqlite3.Row
    return con


def init_db(reset: bool = False) -> None:
    """Create the drivers table from drivers.csv (idempotent)."""
    with _init_lock, closing(_connect()) as con:
        if reset:
            con.execute("DROP TABLE IF EXISTS drivers")
        con.execute("""CREATE TABLE IF NOT EXISTS drivers (
            driver_id TEXT PRIMARY KEY, name TEXT, current_zone TEXT, vehicle_type TEXT,
            capacity_kg REAL, status TEXT, current_orders INTEGER, hours_worked REAL, rating REAL)""")
        if con.execute("SELECT COUNT(*) FROM drivers").fetchone()[0] == 0:
            with open(DRIVERS_CSV, encoding="utf-8") as fh:
                rows = [(r["driver_id"], r["name"], r["current_zone"], r["vehicle_type"], float(r["capacity_kg"]),
                         r["status"], int(r["current_orders"]), float(r["hours_worked"]), float(r["rating"]))
                        for r in csv.DictReader(fh)]
            con.executemany("INSERT INTO drivers VALUES (?,?,?,?,?,?,?,?,?)", rows)
        con.commit()


def all_drivers() -> list[dict]:
    init_db()
    with closing(_connect()) as con:
        return [dict(r) for r in con.execute("SELECT * FROM drivers ORDER BY driver_id")]


def get_driver(driver_id: str) -> dict | None:
    init_db()
    with closing(_connect()) as con:
        row = con.execute("SELECT * FROM drivers WHERE driver_id = ?", (driver_id.strip().upper(),)).fetchone()
        return dict(row) if row else None


def deadhead_minutes(driver_zone: str, pickup: str) -> float:
    """Minutes of empty driving from a driver's zone to the pickup."""
    return round(deadhead_km(driver_zone, pickup) / (FREE_FLOW_KMH * 0.8) * 60.0, 1)


def rank_drivers(pickup: str, dropoff: str, weight_kg: float, deadline_mins: int, hour_of_day: int = 14,
                 closed_corridors: list[str] | None = None, exclude_ids: list[str] | None = None,
                 rainfall_override: dict[str, float] | None = None) -> dict:
    route = plan_route(pickup, dropoff, closed_corridors, rainfall_override)
    if route["status"] != "OK":
        return {"error": f"no route between {pickup} and {dropoff}", "candidates": []}
    primary = route["primary"]
    rain = max(route["rainfall_mm"].get(zone_of(pickup), 0.0), route["rainfall_mm"].get(zone_of(dropoff), 0.0))
    excluded = {d.upper() for d in (exclude_ids or [])}
    candidates = []
    for d in all_drivers():
        if d["driver_id"] in excluded:
            continue
        empty_km = deadhead_km(d["current_zone"], pickup)
        empty_min = deadhead_minutes(d["current_zone"], pickup)
        # The deadline runs from pickup; the empty drive to the pickup is reported separately,
        # but it still counts towards the driver's shift hours.
        slack = deadline_mins - primary["base_eta_mins"]
        f = forecast(primary["distance_km"], primary["speed_factor"], rain, hour_of_day, weight_kg,
                     d["vehicle_type"], zone_of(pickup), zone_of(dropoff), slack_mins=slack)
        total = round(primary["base_eta_mins"] + f.mean_mins, 1)
        rules = check_assignment(d, weight_kg, pickup, dropoff, total + empty_min, hour_of_day)
        candidates.append({
            "driver_id": d["driver_id"], "name": d["name"], "vehicle_type": d["vehicle_type"],
            "current_zone": d["current_zone"], "capacity_kg": d["capacity_kg"], "hours_worked": d["hours_worked"],
            "status": d["status"], "rating": d["rating"], "deadhead_km": empty_km,
            "deadhead_mins": empty_min, "total_eta_mins": total, "delay_mean_mins": f.mean_mins, "delay_band": [f.p10_mins, f.p90_mins],
            "p_on_time": f.p_on_time, "rule_action": rules.action, "violations": rules.violations,
        })
    order = {"ALLOW": 0, "FLAG_FOR_REVIEW": 1, "BLOCK": 2}
    # Legal first, then on-time odds (to 0.05 resolution), then the nearest driver.
    candidates.sort(key=lambda c: (order[c["rule_action"]], -round(c["p_on_time"] * 20) / 20, c["deadhead_km"]))
    return {"corridor": primary["corridor"], "route_base_eta_mins": primary["base_eta_mins"], "candidates": candidates}
