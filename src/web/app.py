"""FastAPI web app: starts runs, streams their events live, serves the UI.

A run executes as a background asyncio task; the browser follows it through
Server-Sent Events that tail the shared event store, so events written by the
ADK service (another process) appear in the same live timeline.
"""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src.config import A2A_URL, ARTIFACTS_DIR, MCP_URL, MODEL_CARD_PATH, WORKER_MODEL, JUDGE_MODEL, llm_available
from src.evaluation.personas import DEMO_CASES, PERSONAS, message_for, order_hint, orders
from src.mcp.tools.chroma_rag import corpus_stats
from src.mcp.tools.driver_db import all_drivers
from src.observability import run_logger, tracing
from src.orchestration import get_orchestrator
from src.orchestration.base import SCENARIOS, RunSpec

STATIC = Path(__file__).parent / "static"
MAX_ACTIVE_RUNS = 2
_active: dict[str, asyncio.Task] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    tracing.init("deliveryfleet-web")
    yield
    for task in _active.values():
        task.cancel()


app = FastAPI(title="DeliveryFleet MAS", lifespan=lifespan, docs_url="/api/docs")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    return response


class RunRequest(BaseModel):
    case_id: str | None = None
    order_id: str | None = Field(default=None, pattern=r"^ORD-\d{3}$")
    persona: Literal["polite", "urgent", "abusive", "injection", "custom"] = "polite"
    message: str | None = Field(default=None, max_length=1200)
    architecture: Literal["hierarchical", "sequential", "parallel"] = "hierarchical"
    prompt_style: Literal["ReAct", "Direct"] = "ReAct"
    scenario: Literal["normal", "corridor_closed", "breakdown", "rain_spike"] = "normal"
    hour_of_day: int = Field(default=14, ge=6, le=23)


def build_spec(req: RunRequest) -> RunSpec:
    order_id, persona, message, scenario = req.order_id, req.persona, req.message, req.scenario
    if req.case_id:
        case = next((c for c in DEMO_CASES if c["id"] == req.case_id), None)
        if case is None:
            raise HTTPException(404, f"unknown case {req.case_id}")
        order_id, persona = case["order_id"], case["persona"]
        # An explicitly chosen event overrides the scenario's built-in one.
        scenario = req.scenario if "scenario" in req.model_fields_set else case["scenario"]
        message = case.get("message")
    if not message:
        if not order_id or order_id not in orders() or persona == "custom":
            raise HTTPException(422, "give a message, or an order_id with a persona")
        message = message_for(order_id, persona)
    hint = order_hint(order_id) if order_id and order_id in orders() and persona != "custom" else None
    return RunSpec(message=message, persona=persona, prompt_style=req.prompt_style, scenario=scenario,
                   order=hint, hour_of_day=req.hour_of_day, source="web")


def launch(spec: RunSpec, architecture: str) -> dict:
    if not llm_available():
        raise HTTPException(503, "GEMINI_API_KEY is not configured on the server")
    for rid, task in list(_active.items()):
        if task.done():
            _active.pop(rid)
    if len(_active) >= MAX_ACTIVE_RUNS:
        raise HTTPException(429, "Two runs are already in progress. Please wait a few seconds.")
    _active[spec.run_id] = asyncio.create_task(get_orchestrator(architecture)(spec))
    return {"run_id": spec.run_id, "message": spec.message}


@app.post("/api/runs")
async def start_run(req: RunRequest):
    return launch(build_spec(req), req.architecture)


URGENCY = {  # customer-facing choice -> (priority, deadline minutes from pickup)
    "standard": ("NORMAL", 90),
    "express": ("HIGH", 60),
    "critical": ("CRITICAL", 45),
}


class BookingRequest(BaseModel):
    pickup: str = Field(pattern=r"^Zone [A-D] - .{2,60}$")
    dropoff: str = Field(pattern=r"^Zone [A-D] - .{2,60}$")
    weight_kg: float = Field(gt=0, le=300)
    urgency: Literal["standard", "express", "critical"] = "standard"
    note: str = Field(default="", max_length=600)
    scenario: Literal["normal", "corridor_closed", "breakdown", "rain_spike"] = "normal"
    hour_of_day: int = Field(default=14, ge=6, le=23)


@app.post("/api/bookings")
async def book(req: BookingRequest):
    """Customer app: the form becomes an ordinary customer message, so the agents do the real work."""
    priority, deadline = URGENCY[req.urgency]
    booking_id = f"BK-{int(time.time() * 1000) % 100000:05d}"
    message = (f"Booking {booking_id}: please pick up a {req.weight_kg:g} kg parcel from {req.pickup} and deliver it "
               f"to {req.dropoff} within {deadline} minutes ({priority} priority). {req.note}".strip())
    spec = RunSpec(message=message, persona="customer", scenario=req.scenario, hour_of_day=req.hour_of_day,
                   source="booking")
    return {**launch(spec, "hierarchical"), "booking_id": booking_id, "deadline_mins": deadline, "priority": priority}


@app.get("/api/runs/{run_id}/stream")
async def stream(run_id: str, request: Request):
    async def events():
        last, idle_since = 0, time.time()
        while True:
            if await request.is_disconnected():
                return
            batch = run_logger.events_after(run_id, last)
            for ev in batch:
                last = ev["id"]
                yield {"event": "agent_event", "data": json.dumps(ev, default=str)}
            run = run_logger.get_run(run_id)
            if run and run["status"] != "RUNNING" and not batch:
                yield {"event": "done", "data": json.dumps(_run_payload(run_id), default=str)}
                return
            if batch:
                idle_since = time.time()
            elif time.time() - idle_since > 300:
                yield {"event": "timeout", "data": "{}"}
                return
            await asyncio.sleep(0.4)
    return EventSourceResponse(events())


def _run_payload(run_id: str) -> dict:
    run = run_logger.get_run(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    return {"run": run, "agents": run_logger.agent_totals(run_id), "spans": run_logger.spans_for(run_id)}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    payload = _run_payload(run_id)
    payload["events"] = run_logger.events_after(run_id, 0)
    return payload


@app.get("/api/runs")
async def list_runs(source: str | None = None):
    rows = run_logger.list_runs(source)
    keep = ("run_id", "started", "finished", "architecture", "persona", "prompt_style", "scenario", "order_id",
            "status", "source", "report", "judge")
    return [{k: r.get(k) for k in keep} for r in rows]


async def _probe(url: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            await client.get(url)
        return True
    except httpx.HTTPError:
        return False


@app.get("/api/health")
async def health():
    mcp, a2a = await asyncio.gather(_probe(MCP_URL), _probe(f"{A2A_URL}/.well-known/agent-card.json"))
    return {"web": True, "mcp_server": mcp, "a2a_route_service": a2a, "llm_key": llm_available(),
            "worker_model": WORKER_MODEL, "judge_model": JUDGE_MODEL, "active_runs": len(_active)}


@app.get("/api/meta")
async def meta():
    agent_card = None
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            agent_card = (await client.get(f"{A2A_URL}/.well-known/agent-card.json")).json()
    except (httpx.HTTPError, ValueError):
        pass
    model_card = json.loads(MODEL_CARD_PATH.read_text()) if MODEL_CARD_PATH.exists() else None
    locations = sorted({o[k] for o in orders().values() for k in ("pickup", "dropoff")})
    return {
        "locations": locations,
        "orders": list(orders().values()), "personas": PERSONAS, "demo_cases": DEMO_CASES,
        "scenarios": SCENARIOS, "drivers": all_drivers(), "rag": corpus_stats(),
        "agent_card": agent_card, "model_card": model_card,
    }


@app.get("/api/evaluation")
async def evaluation():
    summary = ARTIFACTS_DIR / "evaluation_summary.json"
    if not summary.exists():
        return JSONResponse({"available": False})
    return {"available": True, **json.loads(summary.read_text())}


app.mount("/artifacts", StaticFiles(directory=ARTIFACTS_DIR), name="artifacts")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")
