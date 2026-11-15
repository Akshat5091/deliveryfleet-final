"""SQLite event store shared by every process (web app, MCP server, ADK service).

Tables
  runs    one row per MAS run (inputs, final report, totals)
  events  append-only log: agent start/end, LLM calls, tool calls, guardrail
          verdicts, blackboard reads/writes, A2A messages, errors
  spans   OpenTelemetry spans (trace tree for the waterfall view)
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import closing
from typing import Any

from src.config import STORE_DB

_lock = threading.Lock()
_initialised = False

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, started REAL, finished REAL, architecture TEXT, persona TEXT,
    prompt_style TEXT, scenario TEXT, order_id TEXT, status TEXT, source TEXT,
    input_text TEXT, report TEXT, blackboard TEXT, provenance TEXT, judge TEXT);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, ts REAL, kind TEXT, agent TEXT,
    framework TEXT, name TEXT, duration_ms REAL, input_tokens INTEGER, output_tokens INTEGER,
    cost_usd REAL, status TEXT, payload TEXT);
CREATE INDEX IF NOT EXISTS ix_events_run ON events(run_id, id);
CREATE TABLE IF NOT EXISTS spans (
    span_id TEXT PRIMARY KEY, trace_id TEXT, parent_id TEXT, run_id TEXT, name TEXT,
    agent TEXT, start REAL, end REAL, status TEXT, attributes TEXT);
CREATE INDEX IF NOT EXISTS ix_spans_run ON spans(run_id);
"""


def _connect() -> sqlite3.Connection:
    global _initialised
    con = sqlite3.connect(STORE_DB, timeout=15, check_same_thread=False)
    con.row_factory = sqlite3.Row
    # Safe with WAL and much faster per commit than the default FULL (no fsync on every event).
    con.execute("PRAGMA synchronous=NORMAL")
    if not _initialised:
        with _lock:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript(SCHEMA)
            _initialised = True
    return con


def log_event(run_id: str, kind: str, agent: str = "", framework: str = "", name: str = "",
              duration_ms: float | None = None, input_tokens: int = 0, output_tokens: int = 0,
              cost_usd: float = 0.0, status: str = "OK", payload: dict | None = None) -> None:
    with closing(_connect()) as con:
        con.execute(
            "INSERT INTO events (run_id, ts, kind, agent, framework, name, duration_ms, input_tokens,"
            " output_tokens, cost_usd, status, payload) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, time.time(), kind, agent, framework, name, duration_ms, input_tokens, output_tokens,
             cost_usd, status, json.dumps(payload or {}, default=str)),
        )
        con.commit()


def events_after(run_id: str, after_id: int = 0) -> list[dict]:
    with closing(_connect()) as con:
        rows = con.execute("SELECT * FROM events WHERE run_id = ? AND id > ? ORDER BY id", (run_id, after_id))
        return [_row(r) for r in rows]


def save_span(span_id: str, trace_id: str, parent_id: str | None, run_id: str, name: str, agent: str,
              start: float, end: float, status: str, attributes: dict) -> None:
    with closing(_connect()) as con:
        con.execute("INSERT OR REPLACE INTO spans VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (span_id, trace_id, parent_id, run_id, name, agent, start, end, status,
                     json.dumps(attributes, default=str)))
        con.commit()


def spans_for(run_id: str) -> list[dict]:
    with closing(_connect()) as con:
        return [_row(r) for r in con.execute("SELECT * FROM spans WHERE run_id = ? ORDER BY start", (run_id,))]


def start_run(run_id: str, **fields: Any) -> None:
    """Create (or reset) a run. A re-run of the same id starts from an empty trace."""
    with closing(_connect()) as con:
        con.execute("DELETE FROM events WHERE run_id = ?", (run_id,))
        con.execute("DELETE FROM spans WHERE run_id = ?", (run_id,))
        con.execute(
            "INSERT OR REPLACE INTO runs (run_id, started, architecture, persona, prompt_style, scenario,"
            " order_id, status, source, input_text) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (run_id, time.time(), fields.get("architecture"), fields.get("persona"), fields.get("prompt_style"),
             fields.get("scenario"), fields.get("order_id"), "RUNNING", fields.get("source", "web"),
             fields.get("input_text")),
        )
        con.commit()


def set_input_text(run_id: str, text: str) -> None:
    """Replace the stored customer message with the guard's masked version."""
    with closing(_connect()) as con:
        con.execute("UPDATE runs SET input_text = ? WHERE run_id = ?", (text, run_id))
        con.commit()


def finish_run(run_id: str, status: str, report: dict, blackboard: dict, provenance: list[dict]) -> None:
    with closing(_connect()) as con:
        con.execute(
            "UPDATE runs SET finished = ?, status = ?, report = ?, blackboard = ?, provenance = ? WHERE run_id = ?",
            (time.time(), status, json.dumps(report, default=str), json.dumps(blackboard, default=str),
             json.dumps(provenance, default=str), run_id),
        )
        con.commit()


def save_judge(run_id: str, judge: dict) -> None:
    with closing(_connect()) as con:
        con.execute("UPDATE runs SET judge = ? WHERE run_id = ?", (json.dumps(judge), run_id))
        con.commit()


def get_run(run_id: str) -> dict | None:
    with closing(_connect()) as con:
        row = con.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return _row(row) if row else None


LIST_COLUMNS = ("run_id, started, finished, architecture, persona, prompt_style, scenario, order_id, "
                "status, source, report, judge")


def list_runs(source: str | None = None, limit: int = 500) -> list[dict]:
    """Summary rows only (no blackboard/provenance blobs)."""
    with closing(_connect()) as con:
        if source:
            rows = con.execute(f"SELECT {LIST_COLUMNS} FROM runs WHERE source = ? ORDER BY started DESC LIMIT ?",
                               (source, limit))
        else:
            rows = con.execute(f"SELECT {LIST_COLUMNS} FROM runs ORDER BY started DESC LIMIT ?", (limit,))
        return [_row(r) for r in rows]


def run_status(run_id: str) -> str | None:
    with closing(_connect()) as con:
        row = con.execute("SELECT status FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return row["status"] if row else None


def fail_stale_runs() -> int:
    """Runs left RUNNING by a crash or restart can never finish; mark them FAILED."""
    with closing(_connect()) as con:
        n = con.execute("UPDATE runs SET status = 'FAILED', finished = ? WHERE status = 'RUNNING'",
                        (time.time(),)).rowcount
        con.commit()
        return n


def agent_totals(run_id: str) -> dict[str, dict]:
    """Per-agent latency, tokens, cost and call counts for one run."""
    with closing(_connect()) as con:
        rows = con.execute(
            "SELECT agent, kind, COUNT(*) n, SUM(COALESCE(duration_ms,0)) ms, SUM(input_tokens) it,"
            " SUM(output_tokens) ot, SUM(cost_usd) c, SUM(status != 'OK') errs"
            " FROM events WHERE run_id = ? AND agent != '' GROUP BY agent, kind", (run_id,))
        out: dict[str, dict] = {}
        for r in rows:
            a = out.setdefault(r["agent"], {"latency_ms": 0.0, "llm_calls": 0, "tool_calls": 0,
                                            "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "errors": 0})
            a["input_tokens"] += r["it"] or 0
            a["output_tokens"] += r["ot"] or 0
            a["cost_usd"] += r["c"] or 0.0
            a["errors"] += r["errs"] or 0
            if r["kind"] == "agent_end":
                a["latency_ms"] += r["ms"] or 0.0
            elif r["kind"] == "llm_call":
                a["llm_calls"] += r["n"]
            elif r["kind"] == "tool_call":
                a["tool_calls"] += r["n"]
        return out


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    for key in ("payload", "attributes", "report", "blackboard", "provenance", "judge"):
        if key in d and isinstance(d[key], str):
            try:
                d[key] = json.loads(d[key])
            except json.JSONDecodeError:
                pass
    return d
