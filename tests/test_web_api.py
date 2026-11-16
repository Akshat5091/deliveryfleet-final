"""Web API contract tests (no LLM calls: runs are never actually started here)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.web import app as web


@pytest.fixture()
def client(monkeypatch):
    started: list = []
    monkeypatch.setattr(web, "launch", lambda spec, arch, ip: started.append((spec, arch)) or {"run_id": spec.run_id, "message": spec.message})
    with TestClient(web.app) as c:
        c.started = started
        yield c


def test_startup_closes_stale_runs_in_the_test_store_and_never_the_live_one():
    import sqlite3
    from contextlib import closing

    from src.config import STORE_DB
    from src.observability import run_logger

    run_logger.start_run("RUN-STALE-TEST", architecture="hierarchical")
    with TestClient(web.app):
        pass
    assert run_logger.run_status("RUN-STALE-TEST") == "FAILED"
    if STORE_DB.exists():
        with closing(sqlite3.connect(STORE_DB)) as live:
            assert live.execute("SELECT 1 FROM runs WHERE run_id = 'RUN-STALE-TEST'").fetchone() is None


def test_meta_lists_locations_cases_and_drivers(client):
    meta = client.get("/api/meta").json()
    assert "Zone A - Warehouse 1" in meta["locations"]
    assert {c["id"] for c in meta["demo_cases"]} >= {"standard", "exception", "injection", "overweight"}
    assert len(meta["drivers"]) == 10


def test_booking_builds_a_customer_message(client):
    res = client.post("/api/bookings", json={"pickup": "Zone A - Warehouse 1", "dropoff": "Zone B - Tech Park",
                                             "weight_kg": 12, "urgency": "express", "note": "Fragile"})
    assert res.status_code == 200
    body = res.json()
    assert body["priority"] == "HIGH" and body["deadline_mins"] == 60
    spec, arch = client.started[0]
    assert arch == "hierarchical" and spec.source == "booking"
    assert "Zone A - Warehouse 1" in spec.message and "Fragile" in spec.message


@pytest.mark.parametrize("bad", [
    {"pickup": "Mars - Base", "dropoff": "Zone B - Tech Park", "weight_kg": 5},
    {"pickup": "Zone A - Warehouse 1", "dropoff": "Zone B - Tech Park", "weight_kg": 900},
    {"pickup": "Zone A - Warehouse 1", "dropoff": "Zone B - Tech Park", "weight_kg": 5, "note": "x" * 700},
    {"pickup": "Zone A - Warehouse 1", "dropoff": "Zone B - Tech Park", "weight_kg": 5, "urgency": "yesterday"},
])
def test_booking_rejects_out_of_range_input(client, bad):
    assert client.post("/api/bookings", json=bad).status_code == 422
    assert client.started == []


def test_case_event_override_only_when_explicit(client):
    client.post("/api/runs", json={"case_id": "exception"})
    client.post("/api/runs", json={"case_id": "exception", "scenario": "normal"})
    assert client.started[0][0].scenario == "corridor_closed"
    assert client.started[1][0].scenario == "normal"


def test_unknown_case_is_404(client):
    assert client.post("/api/runs", json={"case_id": "nope"}).status_code == 404


def test_a_fresh_server_starts_from_the_saved_runs_and_never_overwrites_a_store(tmp_path):
    import sqlite3
    from contextlib import closing

    from src.observability import run_logger

    seed = tmp_path / "seed.sqlite3"
    with closing(sqlite3.connect(seed)) as con:
        con.executescript(run_logger.SCHEMA)
        con.execute("INSERT INTO runs (run_id, status) VALUES ('EVAL-SEED', 'DISPATCHED')")
        con.commit()
    assert run_logger.seed_if_empty(seed) is True
    assert run_logger.run_status("EVAL-SEED") == "DISPATCHED"

    run_logger.start_run("RUN-LOCAL", architecture="hierarchical")
    assert run_logger.seed_if_empty(seed) is False
    assert run_logger.run_status("RUN-LOCAL") == "RUNNING"
