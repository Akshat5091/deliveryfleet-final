"""Zone road network and time-weighted Dijkstra routing.

The network keeps the Eval-2 inter-zone distances and the directional traffic
speed factors from traffic.csv. A delivery location such as
"Zone B - Tech Park" hangs off its zone hub with a short local leg.
"""
from __future__ import annotations

import csv
import heapq
import zlib
from dataclasses import dataclass
from functools import lru_cache

from src.config import TRAFFIC_CSV, WEATHER_CSV, ZONES

ZONE_DISTANCES_KM: dict[frozenset[str], float] = {
    frozenset({"Zone A", "Zone B"}): 14.5,
    frozenset({"Zone A", "Zone C"}): 18.0,
    frozenset({"Zone A", "Zone D"}): 26.5,
    frozenset({"Zone B", "Zone C"}): 12.0,
    frozenset({"Zone B", "Zone D"}): 21.0,
    frozenset({"Zone C", "Zone D"}): 16.5,
}
FREE_FLOW_KMH = 60.0
LOCAL_SPEED_KMH = 25.0
HEAVY_RAIN_MM = 10.0
HEAVY_RAIN_SPEED_CUT_KMH = 15.0


class UnknownLocationError(ValueError):
    pass


@dataclass(frozen=True)
class Leg:
    src: str
    dst: str
    km: float
    minutes: float
    congestion: str


def zone_of(location: str) -> str:
    """'Zone B - Tech Park' -> 'Zone B'. Raises for anything outside the four zones."""
    head = location.split("-")[0].strip().lower()
    for zone in ZONES:
        if head == zone.lower():
            return zone
    raise UnknownLocationError(f"'{location}' is not inside Zone A-D")


def local_leg_km(location: str) -> float:
    """Deterministic 0.8-2.3 km leg from a location to its zone hub."""
    return round(0.8 + (zlib.crc32(location.encode()) % 16) / 10.0, 1)


@lru_cache(maxsize=1)
def traffic_table() -> dict[tuple[str, str], tuple[float, str]]:
    table: dict[tuple[str, str], tuple[float, str]] = {}
    with open(TRAFFIC_CSV, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            src, dst = (part.strip() for part in row["route"].split("->"))
            table[(src, dst)] = (float(row["speed_factor"]), row["congestion_level"])
    return table


@lru_cache(maxsize=1)
def weather_table() -> dict[str, dict[str, float | str]]:
    with open(WEATHER_CSV, encoding="utf-8") as fh:
        return {
            row["zone"]: {"condition": row["condition"], "rainfall_mm": float(row["rainfall_mm"])}
            for row in csv.DictReader(fh)
        }


def _congestion_rank(level: str) -> int:
    return {"LOW": 0, "MODERATE": 1, "HIGH": 2, "SEVERE": 3}.get(level, 1)


def _edge_speed(src: str, dst: str, rainfall: dict[str, float]) -> tuple[float, str]:
    factor, level = traffic_table().get((src, dst), (0.8, "MODERATE"))
    speed = FREE_FLOW_KMH * factor
    # Municipal directive Zone D: 15 km/h cut when precipitation exceeds 10 mm.
    if "Zone D" in (src, dst) and rainfall.get("Zone D", 0.0) > HEAVY_RAIN_MM:
        speed -= HEAVY_RAIN_SPEED_CUT_KMH
    return max(speed, 8.0), level


def _dijkstra(origin: str, target: str, closed: set[frozenset[str]], rainfall: dict[str, float]) -> list[Leg] | None:
    best: dict[str, float] = {origin: 0.0}
    prev: dict[str, Leg] = {}
    heap: list[tuple[float, str]] = [(0.0, origin)]
    while heap:
        cost, node = heapq.heappop(heap)
        if node == target:
            break
        if cost > best.get(node, float("inf")):
            continue
        for nxt in ZONES:
            pair = frozenset({node, nxt})
            if nxt == node or pair in closed:
                continue
            speed, level = _edge_speed(node, nxt, rainfall)
            km = ZONE_DISTANCES_KM[pair]
            minutes = km / speed * 60.0
            if cost + minutes < best.get(nxt, float("inf")):
                best[nxt] = cost + minutes
                prev[nxt] = Leg(node, nxt, km, round(minutes, 1), level)
                heapq.heappush(heap, (cost + minutes, nxt))
    if target != origin and target not in prev:
        return None
    legs: list[Leg] = []
    node = target
    while node != origin:
        legs.append(prev[node])
        node = prev[node].src
    return list(reversed(legs))


def plan_route(pickup: str, dropoff: str, closed_corridors: list[str] | None = None,
               rainfall_override: dict[str, float] | None = None) -> dict:
    """Fastest route plus the best alternative that avoids the primary's first corridor.

    closed_corridors uses the 'Zone A -> Zone D' notation (closure applies both ways).
    """
    src_zone, dst_zone = zone_of(pickup), zone_of(dropoff)
    closed = {frozenset(p.strip() for p in c.split("->")) for c in (closed_corridors or [])}
    rainfall = {z: float(w["rainfall_mm"]) for z, w in weather_table().items()}
    rainfall.update(rainfall_override or {})

    def summarise(legs: list[Leg]) -> dict:
        first = local_leg_km(pickup)
        last = local_leg_km(dropoff) if dropoff != pickup else 0.0
        km = first + last + sum(l.km for l in legs)
        minutes = (first + last) / LOCAL_SPEED_KMH * 60.0 + sum(l.minutes for l in legs)
        corridor = " -> ".join([src_zone] + [l.dst for l in legs]) if legs else f"{src_zone} (intra-zone)"
        return {
            "corridor": corridor,
            "distance_km": round(km, 1),
            "base_eta_mins": round(minutes, 1),
            "legs": [l.__dict__ for l in legs],
            "worst_congestion": max((l.congestion for l in legs), key=_congestion_rank, default="LOW"),
            "speed_factor": round(min((traffic_table().get((l.src, l.dst), (0.9, ""))[0] for l in legs), default=0.92), 2),
        }

    primary = _dijkstra(src_zone, dst_zone, closed, rainfall)
    if primary is None:
        return {"status": "NO_ROUTE", "pickup": pickup, "dropoff": dropoff, "closed_corridors": closed_corridors or []}
    alternative = None
    if primary:
        alt = _dijkstra(src_zone, dst_zone, closed | {frozenset({primary[0].src, primary[0].dst})}, rainfall)
        alternative = summarise(alt) if alt else None
    return {"status": "OK", "pickup": pickup, "dropoff": dropoff, "primary": summarise(primary),
            "alternative": alternative, "closed_corridors": closed_corridors or [], "rainfall_mm": rainfall}


def deadhead_km(driver_zone: str, pickup: str) -> float:
    """Distance a driver must travel empty to reach the pickup."""
    pickup_zone = zone_of(pickup)
    hub = 0.0 if driver_zone == pickup_zone else ZONE_DISTANCES_KM[frozenset({driver_zone, pickup_zone})]
    return round(hub + local_leg_km(pickup), 1)
