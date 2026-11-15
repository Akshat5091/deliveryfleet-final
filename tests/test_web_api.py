"""Web API contract tests (no LLM calls: runs are never actually started here)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.web import app as web


@pytest.fixture()
def client(monkeypatch):
    started: list = []
    monkeypatch.setattr(web, "launch", lambda spec, arch: started.append((spec, arch)) or {"run_id": spec.run_id, "message": spec.message})
    with TestClient(web.app) as c:
        c.started = started
        yield c


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
