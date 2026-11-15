"""The 72-run experiment: run, judge, measure, plot.

Design (same factors as Eval 2):
  3 architectures x 2 prompt styles x 4 personas x 3 orders = 72 runs
  ORD-101 normal, ORD-103 heavy cross-zone, ORD-105 critical + road closure mid-trip.

Each run has a fixed id (EVAL-<arch><style>-<persona>-<order>), so the script
can be stopped and restarted: finished runs and judged runs are skipped.

  uv run python -m src.evaluation.benchmark_runner            # run + judge + report
  uv run python -m src.evaluation.benchmark_runner --report   # only rebuild charts/summary
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import time

import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR
from src.evaluation import visualizer as viz
from src.evaluation.agent_evaluator import (
    CRITERIA,
    judge_run,
    reference_answer,
    reference_match,
    sensitivity_specificity,
    shannon_entropy,
)
from src.evaluation.personas import message_for, order_hint, orders
from src.observability import run_logger, tracing
from src.orchestration import get_orchestrator
from src.orchestration.base import RunSpec

ARCHITECTURES = ("hierarchical", "sequential", "parallel")
STYLES = ("ReAct", "Direct")
PERSONAS = ("polite", "urgent", "abusive", "injection")
ORDERS = {"ORD-101": "normal", "ORD-103": "normal", "ORD-105": "corridor_closed"}
HOUR = 14


def run_id(arch: str, style: str, persona: str, order_id: str) -> str:
    return f"EVAL-{arch[0].upper()}{style[0]}-{persona[:3]}-{order_id[-3:]}"


def matrix() -> list[dict]:
    return [{"arch": a, "style": s, "persona": p, "order_id": o, "scenario": ORDERS[o], "run_id": run_id(a, s, p, o)}
            for a, s, p, o in itertools.product(ARCHITECTURES, STYLES, PERSONAS, ORDERS)]


async def execute(fresh: bool = False, concurrency: int = 3) -> None:
    """Run every unfinished cell, `concurrency` at a time. Each run has its own Blackboard,
    trace and run id, so parallel runs never share state."""
    cells = matrix()
    gate = asyncio.Semaphore(concurrency)

    async def one(i: int, cell: dict) -> None:
        existing = run_logger.get_run(cell["run_id"])
        if existing and existing["status"] not in ("RUNNING", "FAILED") and not fresh:
            return
        async with gate:
            spec = RunSpec(message=message_for(cell["order_id"], cell["persona"]), persona=cell["persona"],
                           prompt_style=cell["style"], scenario=cell["scenario"],
                           order=order_hint(cell["order_id"]), hour_of_day=HOUR, source="benchmark",
                           run_id=cell["run_id"])
            t0 = time.perf_counter()
            report = await get_orchestrator(cell["arch"])(spec)
            print(f"[{i:02d}/{len(cells)}] {cell['run_id']:<22} {report.outcome:<27} "
                  f"{time.perf_counter() - t0:6.1f}s  {report.total_tokens:>6} tok", flush=True)

    await asyncio.gather(*(one(i, c) for i, c in enumerate(cells, 1)))


async def judge_all(concurrency: int = 4) -> None:
    gate = asyncio.Semaphore(concurrency)

    async def one(cell: dict) -> None:
        run = run_logger.get_run(cell["run_id"])
        if not run or run.get("judge") or run["status"] == "RUNNING":
            return
        async with gate:
            token = tracing.set_context(tracing.RunContext(cell["run_id"] + "-judge", "llm_judge", "google-genai"))
            try:
                run_logger.save_judge(cell["run_id"], await judge_run(run))
            except Exception as exc:
                print(f"judge failed for {cell['run_id']}: {exc}", flush=True)
            finally:
                tracing.reset_context(token)

    await asyncio.gather(*(one(c) for c in matrix()))


async def run_and_judge(fresh: bool, concurrency: int) -> None:
    """One event loop for both steps: the cached Gemini client must not outlive its loop."""
    await execute(fresh, concurrency)
    await judge_all()


def collect() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, agent_rows = [], []
    refs = {o: reference_answer(orders()[o], HOUR) for o in ORDERS}
    for cell in matrix():
        run = run_logger.get_run(cell["run_id"])
        if not run or not run.get("report"):
            continue
        rep, judge = run["report"], run.get("judge") or {}
        match = reference_match(run, refs[cell["order_id"]])
        rows.append({
            "run_id": cell["run_id"], "architecture": cell["arch"], "prompt_style": cell["style"],
            "persona": cell["persona"], "order_id": cell["order_id"], "scenario": cell["scenario"],
            "outcome": rep["outcome"], "guard": rep["guard_status"], "driver": rep.get("driver_id"),
            "corridor": rep.get("corridor"), "eta_mins": rep.get("eta_mins"), "deadline": rep.get("deadline_mins"),
            "latency_s": rep["total_latency_s"], "llm_calls": rep["llm_calls"], "tokens": rep["total_tokens"],
            "cost_usd": rep["cost_usd"], "driver_ok": match["driver_ok"], "corridor_ok": match["corridor_ok"],
            **{c: judge.get(c) for c in CRITERIA}, "judge_overall": judge.get("overall"),
        })
        for agent, t in run_logger.agent_totals(cell["run_id"]).items():
            agent_rows.append({"run_id": cell["run_id"], "architecture": cell["arch"], "persona": cell["persona"],
                               "agent": agent, "latency_s": t["latency_ms"] / 1000, "llm_calls": t["llm_calls"],
                               "tool_calls": t["tool_calls"], "tokens": t["input_tokens"] + t["output_tokens"],
                               "cost_usd": t["cost_usd"]})
    return pd.DataFrame(rows), pd.DataFrame(agent_rows)


def build_report() -> dict:
    df, agents = collect()
    if df.empty:
        raise SystemExit("no finished benchmark runs found")
    df.to_csv(ARTIFACTS_DIR / "benchmark_runs.csv", index=False)
    agents.to_csv(ARTIFACTS_DIR / "benchmark_agents.csv", index=False)
    served = df[df["outcome"] != "BLOCKED"]
    clean = served[served["persona"] != "injection"]

    entropy_rows = []
    for (order_id, persona), g in clean.groupby(["order_id", "persona"]):
        entropy_rows.append({"order_id": order_id, "persona": persona, "runs": len(g),
                             "driver_entropy_bits": shannon_entropy(g["driver"].fillna("none").tolist()),
                             "distinct_drivers": g["driver"].nunique()})
    entropy = pd.DataFrame(entropy_rows)

    by_arch = served.groupby("architecture").agg(
        runs=("run_id", "count"), p50_latency_s=("latency_s", "median"), mean_latency_s=("latency_s", "mean"),
        mean_tokens=("tokens", "mean"), mean_cost_usd=("cost_usd", "mean"), judge_mean=("judge_overall", "mean"),
        driver_match=("driver_ok", "mean"), corridor_match=("corridor_ok", "mean"),
        escalated=("outcome", lambda s: int((s == "ESCALATED").sum())),
    ).reindex([a for a in ARCHITECTURES if a in set(served["architecture"])]).round(3).reset_index()

    figures = [
        {"file": viz.boxplot_by(served, "latency_s", "architecture", "End-to-end latency per run",
                                "seconds", "latency_by_architecture.png", list(ARCHITECTURES), True),
         "title": "Latency by architecture", "caption": "Each dot is one run; box = middle 50%, line = median."},
        {"file": viz.boxplot_by(served, "cost_usd", "persona", "Cost per run by customer persona",
                                "USD (Gemini list price)", "cost_by_persona.png", ["polite", "urgent", "abusive"]),
         "title": "Cost by persona", "caption": "Abusive messages are sanitised, not refused, so they cost about the same."},
        {"file": viz.boxplot_by(served.dropna(subset=["judge_overall"]), "judge_overall", "prompt_style",
                                "LLM-judge overall score by prompt style", "score (1-5)",
                                "judge_by_prompt_style.png", list(STYLES)),
         "title": "Quality by prompt style", "caption": "ReAct vs Direct prompts on the same agents and inputs."},
        {"file": viz.judge_bars(served.dropna(subset=["judge_overall"])),
         "title": "Judge criteria", "caption": "Mean of each criterion of the 4-score vector."},
        {"file": viz.per_agent_heatmaps(agents[agents["persona"] != "injection"]),
         "title": "Heatmap inside each agent", "caption": "Where time goes inside every agent, split by persona and architecture."},
        {"file": viz.heatmap(
            agents[agents["persona"] != "injection"].pivot_table(index="run_id", columns="agent", values="latency_s",
                                                                 aggfunc="sum").reindex(columns=viz.AGENTS).fillna(0),
            "Runs x agents: latency (s)", "heatmap_runs_by_agent.png", cbar_label="seconds",
            figsize=(7.5, 11)),
         "title": "Runs x agents", "caption": "Every served run (rows) against every agent (columns)."},
    ]
    sens = sensitivity_specificity(df.to_dict("records"))
    summary = {
        "design": "3 architectures x 2 prompt styles (ReAct, Direct) x 4 personas (polite, urgent, abusive, injection) "
                  "x 3 orders (ORD-101, ORD-103, ORD-105 with a mid-trip road closure) = 72 runs.",
        "headline": {"runs": int(len(df)), "sensitivity": sens["sensitivity"], "specificity": sens["specificity"],
                     "judge_mean": float(served["judge_overall"].mean()) if served["judge_overall"].notna().any() else None,
                     "p50_latency_s": float(served["latency_s"].median()), "mean_cost_usd": float(served["cost_usd"].mean()),
                     "total_cost_usd": float(df["cost_usd"].sum()), "total_tokens": int(df["tokens"].sum()),
                     "driver_reference_match": float(served["driver_ok"].dropna().mean()) if served["driver_ok"].notna().any() else None},
        "confusion": sens,
        "by_architecture": by_arch.to_dict("records"),
        "by_prompt_style": served.groupby("prompt_style")[["latency_s", "tokens", "judge_overall"]].mean().round(3).reset_index().to_dict("records"),
        "by_persona": df.groupby("persona").agg(runs=("run_id", "count"), blocked=("outcome", lambda s: int((s == "BLOCKED").sum())),
                                                mean_latency_s=("latency_s", "mean"), mean_cost_usd=("cost_usd", "mean")).round(4).reset_index().to_dict("records"),
        "outcomes": df["outcome"].value_counts().to_dict(),
        "entropy": entropy.to_dict("records"),
        "mean_driver_entropy_bits": float(entropy["driver_entropy_bits"].mean()) if not entropy.empty else None,
        "per_agent": agents.groupby("agent")[["latency_s", "llm_calls", "tool_calls", "tokens", "cost_usd"]].mean().round(4).reset_index().to_dict("records"),
        "figures": figures,
    }
    (ARTIFACTS_DIR / "evaluation_summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true", help="only rebuild charts and summary")
    parser.add_argument("--fresh", action="store_true", help="re-run every cell")
    parser.add_argument("--concurrency", type=int, default=3, help="runs executed at the same time")
    args = parser.parse_args()
    if not args.report:
        tracing.init("deliveryfleet-benchmark")
        asyncio.run(run_and_judge(args.fresh, args.concurrency))
    s = build_report()
    print(json.dumps(s["headline"], indent=2))


if __name__ == "__main__":
    main()
