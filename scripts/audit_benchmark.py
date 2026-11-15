"""Check every benchmark number against the raw event log, independently of the report code.

  uv run python scripts/audit_benchmark.py

Each check recomputes a value from the stored events, the Blackboard or the CSV and compares it
with what the pipeline reported. Prints one line per failed check and a summary; exits 1 on failure.
"""
from __future__ import annotations

import json
import sqlite3
import statistics
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ARTIFACTS_DIR, JUDGE_MODEL, PRICE_PER_M, ROUTE_MODEL, WORKER_MODEL  # noqa: E402
from src.evaluation.agent_evaluator import CRITERIA  # noqa: E402
from src.evaluation.benchmark_runner import matrix  # noqa: E402

DB = ROOT / "runtime" / "fleet.sqlite3"
problems: list[str] = []
notes: list[str] = []


def fail(run_id: str, what: str) -> None:
    problems.append(f"{run_id:<22} {what}")


def close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def price(model: str, tokens_in: int, tokens_out: int) -> float:
    key = next(k for k in PRICE_PER_M if model.endswith(k))
    p_in, p_out = PRICE_PER_M[key]
    return tokens_in * p_in / 1e6 + tokens_out * p_out / 1e6


def main() -> int:
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    cells = matrix()
    ids = [c["run_id"] for c in cells]
    runs = {r["run_id"]: r for r in con.execute(f"SELECT * FROM runs WHERE run_id IN ({','.join('?' * len(ids))})", ids)}

    # 1. Completeness
    missing = [i for i in ids if i not in runs]
    unfinished = [i for i, r in runs.items() if r["status"] in ("RUNNING", "FAILED") or not r["report"]]
    for i in missing:
        fail(i, "run missing")
    for i in unfinished:
        fail(i, f"run not finished (status {runs[i]['status']})")

    rows = []
    for cell in cells:
        rid = cell["run_id"]
        run = runs.get(rid)
        if not run or not run["report"]:
            continue
        rep = json.loads(run["report"])
        bb = json.loads(run["blackboard"] or "{}")
        events = con.execute("SELECT * FROM events WHERE run_id = ?", (rid,)).fetchall()
        llm = [e for e in events if e["kind"] == "llm_call"]

        # 2. Calls, tokens, cost recomputed from the per-call events
        tok = sum((e["input_tokens"] or 0) + (e["output_tokens"] or 0) for e in llm)
        cost = sum(price(e["name"], e["input_tokens"] or 0, e["output_tokens"] or 0) for e in llm)
        if rep["llm_calls"] != len(llm):
            fail(rid, f"llm_calls {rep['llm_calls']} != {len(llm)} logged calls")
        if rep["total_tokens"] != tok:
            fail(rid, f"total_tokens {rep['total_tokens']} != {tok} from events")
        if not close(rep["cost_usd"], cost, 1e-5):
            fail(rid, f"cost {rep['cost_usd']} != {cost:.6f} recomputed from list prices")
        models = {e["name"] for e in llm}
        if models - {WORKER_MODEL, ROUTE_MODEL}:
            fail(rid, f"unexpected model(s) in agent calls: {models - {WORKER_MODEL, ROUTE_MODEL}}")
        if any((e["input_tokens"] or 0) == 0 for e in llm if e["status"] == "OK"):
            fail(rid, "an OK LLM call has 0 input tokens")

        # 3. Latency against wall-clock timestamps
        if run["finished"] and not close(rep["total_latency_s"], run["finished"] - run["started"], 5.0):
            fail(rid, f"latency {rep['total_latency_s']} s vs {run['finished'] - run['started']:.1f} s wall clock")

        # 4. Outcome rules
        outcome, persona = rep["outcome"], cell["persona"]
        if persona == "injection" and outcome != "BLOCKED":
            fail(rid, f"injection run not blocked ({outcome})")
        if persona != "injection" and outcome == "BLOCKED":
            fail(rid, "legitimate run blocked (false positive)")
        if outcome == "BLOCKED" and [e for e in llm if e["agent"] != "input_guard_judge"]:
            fail(rid, "blocked run still called agent LLMs")
        if outcome in ("DISPATCHED", "RECOVERED_AFTER_DISRUPTION"):
            if rep["eta_mins"] is None or rep["eta_mins"] > rep["deadline_mins"]:
                fail(rid, f"{outcome} but ETA {rep['eta_mins']} > deadline {rep['deadline_mins']}")
            bc = bb.get("business_check") or {}
            if bc.get("action") not in (None, "ALLOW", "WARN"):
                fail(rid, f"{outcome} but final business check is {bc.get('action')}")

        # 5. Grounding: route ETA must equal base + forecast delay from the tools
        route = bb.get("route") or {}
        if route.get("base_eta_mins") is not None and route.get("eta_mins") is not None:
            if not close(route["eta_mins"], route["base_eta_mins"] + route["delay_mean_mins"], 0.15):
                fail(rid, f"route eta {route['eta_mins']} != base {route['base_eta_mins']} + delay {route['delay_mean_mins']}")
            if not route["delay_p10_mins"] <= route["delay_mean_mins"] <= route["delay_p90_mins"]:
                fail(rid, "delay mean outside its own P10-P90 band")
            alt = route.get("alternative_eta_mins")  # base time only, so compare base with base
            if alt is not None and alt < route["base_eta_mins"]:
                fail(rid, f"chose {route['corridor']} (base {route['base_eta_mins']}) over a faster "
                          f"alternative (base {alt})")
        disruption = bb.get("disruption") or {}
        if disruption.get("closed_corridor") and rep["corridor"] == disruption["closed_corridor"]:
            fail(rid, f"{outcome} run reports the closed corridor {rep['corridor']}")
        if cell["scenario"] == "corridor_closed" and outcome != "BLOCKED" and not bb.get("exception"):
            fail(rid, "road-closure run has no exception decision")

        # 5b. A reroute ETA must be the number the Route Agent returned over A2A, not the LLM's copy
        exc = bb.get("exception") or {}
        if exc.get("action") in ("REROUTE", "REROUTE_AND_REASSIGN", "ESCALATE"):
            routed = {}
            for e in events:
                if e["kind"] == "tool_call" and e["name"] == "request_reroute_via_a2a" and e["status"] == "OK":
                    try:
                        plan = json.loads(json.loads(e["payload"])["result"])
                        routed[plan["corridor"]] = plan["eta_mins"]
                    except (KeyError, TypeError, json.JSONDecodeError):
                        continue
            got = routed.get(exc.get("new_corridor"))
            if exc["action"] == "ESCALATE":
                if exc.get("new_eta_mins") is not None and (got is None or not close(exc["new_eta_mins"], got, 1.0)):
                    fail(rid, f"escalation quotes ETA {exc['new_eta_mins']} the Route Agent never returned")
                if rep.get("eta_mins") is not None and rep["eta_mins"] <= (rep.get("deadline_mins") or 0) \
                        and cell["scenario"] == "corridor_closed":
                    fail(rid, f"escalated but reports an on-time ETA {rep['eta_mins']}")
            elif got is None:
                fail(rid, f"{exc['action']} without a Route Agent plan for {exc.get('new_corridor')}")
            elif not close(exc.get("new_eta_mins") or 0, got, 1.0):
                fail(rid, f"reroute ETA {exc.get('new_eta_mins')} != Route Agent's {got}")

        # 6. Judge
        judge = json.loads(run["judge"]) if run["judge"] else None
        if outcome != "BLOCKED":
            if not judge:
                fail(rid, "served run not judged")
            else:
                for c in CRITERIA:
                    if not isinstance(judge.get(c), int) or not 1 <= judge[c] <= 5:
                        fail(rid, f"judge {c} = {judge.get(c)!r} not an int 1-5")
                if judge.get("judge") and judge["judge"] != JUDGE_MODEL:
                    fail(rid, f"judged by {judge['judge']}, expected {JUDGE_MODEL}")
                if judge.get("judge") in models:
                    fail(rid, "judge model is the same as an agent model")
        rows.append({"run_id": rid, "outcome": outcome, "persona": persona, "latency_s": rep["total_latency_s"],
                     "tokens": rep["total_tokens"], "cost_usd": rep["cost_usd"], "driver": rep.get("driver_id"),
                     "architecture": cell["arch"]})

    # 7. CSV and summary against the database
    csv_path, summary_path = ARTIFACTS_DIR / "benchmark_runs.csv", ARTIFACTS_DIR / "evaluation_summary.json"
    if csv_path.exists() and rows:
        csv = pd.read_csv(csv_path).set_index("run_id")
        for r in rows:
            if r["run_id"] not in csv.index:
                fail(r["run_id"], "missing from benchmark_runs.csv")
                continue
            c = csv.loc[r["run_id"]]
            for k in ("outcome", "tokens"):
                if str(c[k]) != str(r[k]):
                    fail(r["run_id"], f"CSV {k} {c[k]} != DB {r[k]}")
            if not close(float(c["cost_usd"]), r["cost_usd"], 1e-6):
                fail(r["run_id"], "CSV cost differs from DB")
    else:
        notes.append("benchmark_runs.csv not built yet")

    if summary_path.exists() and rows:
        s = json.loads(summary_path.read_text())
        h = s["headline"]
        served = [r for r in rows if r["outcome"] != "BLOCKED"]
        inj = [r for r in rows if r["persona"] == "injection"]
        legit = [r for r in rows if r["persona"] != "injection"]
        checks = {
            "runs": (h["runs"], len(rows)),
            "total_tokens": (h["total_tokens"], sum(r["tokens"] for r in rows)),
            "total_cost_usd": (round(h["total_cost_usd"], 6), round(sum(r["cost_usd"] for r in rows), 6)),
            "p50_latency_s": (round(h["p50_latency_s"], 3), round(statistics.median(r["latency_s"] for r in served), 3)),
            "sensitivity": (h["sensitivity"], round(sum(r["outcome"] == "BLOCKED" for r in inj) / len(inj), 3) if inj else None),
            "specificity": (h["specificity"], round(sum(r["outcome"] != "BLOCKED" for r in legit) / len(legit), 3) if legit else None),
        }
        for name, (got, want) in checks.items():
            if want is not None and not close(float(got), float(want), 1e-3):
                fail("summary", f"{name} {got} != recomputed {want}")
        if s["outcomes"] != dict(Counter(r["outcome"] for r in rows)):
            fail("summary", f"outcome counts {s['outcomes']} != {dict(Counter(r['outcome'] for r in rows))}")
        for arch in s["by_architecture"]:
            sub = [r for r in served if r["architecture"] == arch["architecture"]]
            if arch["runs"] != len(sub):
                fail("summary", f"{arch['architecture']} runs {arch['runs']} != {len(sub)}")
            elif not close(arch["p50_latency_s"], statistics.median(r["latency_s"] for r in sub), 0.01):
                fail("summary", f"{arch['architecture']} median latency differs")
        for fig in s["figures"]:
            p = Path(fig["file"]) if Path(fig["file"]).is_absolute() else ARTIFACTS_DIR / Path(fig["file"]).name
            if not p.exists():
                fail("figures", f"{p.name} missing")
            elif p.stat().st_mtime < csv_path.stat().st_mtime - 60:
                fail("figures", f"{p.name} is older than the CSV (stale chart)")
    else:
        notes.append("evaluation_summary.json not built yet")

    print(f"audited {len(rows)} of {len(ids)} runs")
    for n in notes:
        print("note:", n)
    for p in problems:
        print("FAIL:", p)
    print(f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
