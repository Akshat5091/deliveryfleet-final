from __future__ import annotations

import pytest

from src.observability import run_logger


@pytest.fixture(autouse=True)
def isolated_run_store(tmp_path, monkeypatch):
    """Tests write to a throwaway run store. Starting the web app marks every RUNNING run
    FAILED, so a test using the live store would kill a run the local server is executing."""
    monkeypatch.setattr(run_logger, "STORE_DB", tmp_path / "runs.sqlite3")
    monkeypatch.setattr(run_logger, "_initialised", False)
    yield
