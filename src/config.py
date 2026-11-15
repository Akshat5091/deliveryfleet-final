"""Central settings: paths, service ports, model names and statutory limits."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# CrewAI prints emoji status lines; the Windows console codec cannot encode them.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# .env may live in the project folder or one level up (DeliveryFleet-Final/.env).
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT.parent / ".env")

os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "false")

DATA_DIR = PROJECT_ROOT / "src" / "data"
KB_DIR = DATA_DIR / "knowledge_base"
RUNTIME_DIR = Path(os.getenv("FLEET_RUNTIME_DIR", PROJECT_ROOT / "runtime"))
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

ORDERS_CSV = DATA_DIR / "orders.csv"
DRIVERS_CSV = DATA_DIR / "drivers.csv"
TRAFFIC_CSV = DATA_DIR / "traffic.csv"
WEATHER_CSV = DATA_DIR / "weather.csv"
HISTORY_CSV = DATA_DIR / "delivery_history.csv"

MODEL_PATH = RUNTIME_DIR / "delay_model.joblib"
MODEL_CARD_PATH = ARTIFACTS_DIR / "delay_model_card.json"
CHROMA_DIR = RUNTIME_DIR / "chroma"
STORE_DB = RUNTIME_DIR / "fleet.sqlite3"

HOST = "127.0.0.1"
WEB_PORT = int(os.getenv("PORT", "7860"))
MCP_PORT = int(os.getenv("MCP_PORT", "8001"))
A2A_PORT = int(os.getenv("A2A_PORT", "8002"))
MCP_URL = os.getenv("MCP_URL", f"http://{HOST}:{MCP_PORT}/mcp")
A2A_URL = os.getenv("A2A_URL", f"http://{HOST}:{A2A_PORT}")

# Every agent and the input guard run on Gemini 3.1 Flash-Lite, the cheapest Gemini model open to
# new keys. The judge uses a different model so it does not grade its own model's answers.
# Gemini rate limits are per model, so each distinct model has its own budget.
WORKER_MODEL = os.getenv("WORKER_MODEL", "gemini-3.1-flash-lite")
ROUTE_MODEL = os.getenv("ROUTE_MODEL", "gemini-3.1-flash-lite")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3.5-flash-lite")
LLM_RPM = int(os.getenv("LLM_RPM", "14"))  # requests per minute allowed for each model

# USD per 1M tokens (input, output incl. thinking) for cost estimates: Gemini API
# paid-tier standard prices from aistudio.google.com/docs/pricing (read Nov 2026;
# 3.6 Flash at its promotional rate valid until 31 Dec 2026). Override via env.
PRICE_PER_M = {
    "gemini-3.5-flash-lite": (float(os.getenv("PRICE_LITE_IN", "0.30")), float(os.getenv("PRICE_LITE_OUT", "2.50"))),
    "gemini-3.1-flash-lite": (float(os.getenv("PRICE_ROUTE_IN", "0.25")), float(os.getenv("PRICE_ROUTE_OUT", "1.50"))),
    "gemini-3.6-flash": (float(os.getenv("PRICE_FLASH_IN", "0.75")), float(os.getenv("PRICE_FLASH_OUT", "3.75"))),
}

# Statutory and company limits (SOP-2026-v4, Labor Safety Directive 4.1).
MAX_SHIFT_HOURS = 9.0
FATIGUE_WARN_HOURS = 8.5
FATIGUE_MAX_TRIP_MINS = 30.0
CRITICAL_DELAY_TRIGGER_MINS = 15.0
ZONES = ("Zone A", "Zone B", "Zone C", "Zone D")


def llm_available() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
