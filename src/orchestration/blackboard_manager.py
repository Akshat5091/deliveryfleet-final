"""Blackboard (shared state) and the peer-to-peer inbox used for comparison.

Blackboard: one shared notice board per run. Every agent reads the sections it
needs and writes its own section. Every read and write is recorded with the
run id, agent, section, version and time, so any decision can be traced back
to the agent and the inputs that produced it.

PeerInbox: the Lab-03 alternative. No shared store; each agent only sees the
message the previous agent sent to it directly.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

SECTIONS = (
    "request", "guard", "compliance", "candidates", "dispatch", "route",
    "business_check", "disruption", "exception", "review",
)

# Which sections each agent is allowed to read: agents see only what they need,
# which keeps prompts short and stops one agent copying another's job.
READ_SCOPE: dict[str, tuple[str, ...]] = {
    "retrieval_agent": ("request",),
    "dispatch_agent": ("request", "compliance", "candidates"),
    "route_agent": ("request", "dispatch", "compliance"),
    "exception_agent": ("request", "dispatch", "route", "disruption", "candidates", "compliance"),
    "supervisor": SECTIONS,
}


class BlackboardAccessError(PermissionError):
    pass


@dataclass
class Provenance:
    seq: int
    ts: float
    run_id: str
    op: str
    agent: str
    section: str
    version: int
    summary: str

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class Blackboard:
    run_id: str
    on_change: Callable[[Provenance], None] | None = None
    _data: dict[str, Any] = field(default_factory=dict)
    _versions: dict[str, int] = field(default_factory=dict)
    _log: list[Provenance] = field(default_factory=list)
    # Re-entrant: CrewAI tools run in worker threads while the orchestrator awaits.
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def write(self, section: str, value: Any, agent: str, summary: str = "") -> int:
        if section not in SECTIONS:
            raise KeyError(f"unknown blackboard section '{section}'")
        stored = value.model_dump() if isinstance(value, BaseModel) else value
        with self._lock:
            version = self._versions.get(section, 0) + 1
            self._versions[section] = version
            self._data[section] = stored
            self._record("WRITE", agent, section, version, summary)
        return version

    def read(self, section: str, agent: str) -> Any:
        allowed = READ_SCOPE.get(agent, ())
        if section not in allowed:
            raise BlackboardAccessError(f"{agent} may not read '{section}'")
        with self._lock:
            self._record("READ", agent, section, self._versions.get(section, 0), "")
            return self._data.get(section)

    def view_for(self, agent: str) -> dict[str, Any]:
        """Scoped snapshot of everything an agent is allowed to read."""
        return {s: self.read(s, agent) for s in READ_SCOPE.get(agent, ()) if s in self._data}

    def has(self, section: str) -> bool:
        with self._lock:
            return section in self._data

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {s: self._data[s] for s in SECTIONS if s in self._data}

    def provenance(self) -> list[dict]:
        with self._lock:
            return [p.as_dict() for p in self._log]

    def context_chars(self, agent: str) -> int:
        """Size of the scoped view, used to compare against P2P context growth."""
        import json
        with self._lock:
            return len(json.dumps({s: self._data[s] for s in READ_SCOPE.get(agent, ()) if s in self._data}, default=str))

    def _record(self, op: str, agent: str, section: str, version: int, summary: str) -> None:
        entry = Provenance(len(self._log) + 1, time.time(), self.run_id, op, agent, section, version, summary)
        self._log.append(entry)
        if self.on_change:
            self.on_change(entry)


@dataclass
class PeerInbox:
    """Lab-03 peer-to-peer messaging: each agent reads only its own inbox."""
    run_id: str
    on_change: Callable[[dict], None] | None = None
    _inbox: dict[str, Any] = field(default_factory=dict)
    _log: list[dict] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock)

    def send(self, sender: str, receiver: str, payload: Any) -> None:
        stored = payload.model_dump() if isinstance(payload, BaseModel) else payload
        with self._lock:
            self._inbox[receiver] = stored
            entry = {"seq": len(self._log) + 1, "ts": time.time(), "run_id": self.run_id,
                     "op": "P2P_SEND", "agent": sender, "section": f"inbox:{receiver}", "version": 1, "summary": ""}
            self._log.append(entry)
            if self.on_change:
                self.on_change(entry)

    def receive(self, agent: str) -> Any:
        with self._lock:
            return self._inbox.get(agent)

    def provenance(self) -> list[dict]:
        with self._lock:
            return list(self._log)
