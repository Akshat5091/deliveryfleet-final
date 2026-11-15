"""FastMCP server: the only way agents reach tools.

Runs as its own process (streamable HTTP on MCP_PORT). CrewAI agents connect
with MCPServerHTTP, the ADK Route Agent with McpToolset; neither framework
imports the tool code directly. Every argument is validated by the type
hints (FastMCP turns them into a JSON schema) before the tool runs.
"""
from __future__ import annotations

from typing import Annotated, Literal

from fastmcp import FastMCP
from pydantic import Field

from src.config import HOST, MCP_PORT
from src.mcp.tools import chroma_rag, delay_predictor, driver_db, routing_engine

mcp = FastMCP("DeliveryFleetTools")

Location = Annotated[str, Field(pattern=r"^Zone [A-D] - .{2,60}$", description="e.g. 'Zone A - Warehouse 1'")]
Corridor = Annotated[str, Field(pattern=r"^Zone [A-D] -> Zone [A-D]$")]
ZoneName = Literal["Zone A", "Zone B", "Zone C", "Zone D"]
Vehicle = Literal["Bike", "Electric Scooter", "Van", "Electric Van", "Truck"]


@mcp.tool
def search_policies(query: Annotated[str, Field(min_length=3, max_length=300)],
                    k: Annotated[int, Field(ge=1, le=6)] = 4) -> dict:
    """Semantic search over company and municipal delivery rules (ChromaDB RAG).
    Returns the most relevant rule blocks with their source file and similarity."""
    return chroma_rag.search_policies(query, k)


@mcp.tool
def plan_route(pickup: Location, dropoff: Location, closed_corridors: list[Corridor] | None = None,
               zone_d_rainfall_mm: Annotated[float | None, Field(ge=0, le=200)] = None) -> dict:
    """Fastest road corridor between two locations (time-weighted Dijkstra over the
    zone network with live traffic factors) plus the best alternative corridor."""
    override = {"Zone D": zone_d_rainfall_mm} if zone_d_rainfall_mm is not None else None
    return routing_engine.plan_route(pickup, dropoff, closed_corridors, override)


@mcp.tool
def forecast_delay(distance_km: Annotated[float, Field(gt=0, le=300)],
                   traffic_speed_factor: Annotated[float, Field(gt=0, le=1)],
                   rainfall_mm: Annotated[float, Field(ge=0, le=200)],
                   hour_of_day: Annotated[int, Field(ge=0, le=23)],
                   weight_kg: Annotated[float, Field(gt=0, le=300)],
                   vehicle_type: Vehicle, pickup_zone: ZoneName, dropoff_zone: ZoneName,
                   slack_mins: float | None = None) -> dict:
    """Random-Forest delay forecast in minutes with an 80% band (p10-p90) and,
    if slack_mins is given, the probability the delay stays within the slack."""
    return delay_predictor.forecast(distance_km, traffic_speed_factor, rainfall_mm, hour_of_day,
                                    weight_kg, vehicle_type, pickup_zone, dropoff_zone, slack_mins).as_dict()


@mcp.tool
def rank_drivers(pickup: Location, dropoff: Location, weight_kg: Annotated[float, Field(gt=0, le=300)],
                 deadline_mins: Annotated[int, Field(gt=0, le=480)],
                 hour_of_day: Annotated[int, Field(ge=0, le=23)] = 14,
                 closed_corridors: list[Corridor] | None = None,
                 exclude_ids: list[Annotated[str, Field(pattern=r"^D\d{2}$")]] | None = None) -> dict:
    """Every driver in the fleet database scored for this order: ETA, delay band,
    P(on time) and the business-rule verdict (ALLOW / FLAG_FOR_REVIEW / BLOCK with reasons)."""
    return driver_db.rank_drivers(pickup, dropoff, weight_kg, deadline_mins, hour_of_day, closed_corridors, exclude_ids)


@mcp.tool
def get_driver(driver_id: Annotated[str, Field(pattern=r"^D\d{2}$")]) -> dict:
    """One driver's record from the fleet database."""
    return driver_db.get_driver(driver_id) or {"error": f"{driver_id} not found"}


def main() -> None:
    driver_db.init_db()
    chroma_rag.corpus_stats()
    delay_predictor.forecast(10, 0.8, 0, 14, 5, "Van", "Zone A", "Zone B")
    mcp.run(transport="http", host=HOST, port=MCP_PORT, path="/mcp", show_banner=False)


if __name__ == "__main__":
    main()
