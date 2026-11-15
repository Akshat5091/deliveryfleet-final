"""Does the LLM judge notice bad runs? Score deliberately broken copies of real benchmark runs.

  uv run python scripts/judge_sanity_check.py

Each case copies a finished run and plants one known fault. A judge worth trusting must score the
faulty criterion clearly lower than the 5 it gave the original. Results go to
artifacts/judge_sanity_check.json.
"""
from __future__ import annotations

import asyncio
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ARTIFACTS_DIR  # noqa: E402
from src.evaluation.agent_evaluator import judge_run  # noqa: E402
from src.observability import run_logger  # noqa: E402

BASE = "EVAL-HD-pol-101"  # a clean dispatched run


def illegal_driver(run: dict) -> dict:
    """Swap in a driver whose hours plus this trip exceed the 9-hour legal limit."""
    bb, rep = run["blackboard"], run["report"]
    bb["dispatch"].update(driver_id="D99", driver_name="Ravi Tired", rule_checks=["hours 8.9 + trip 0.73 > 9.0"])
    bb["business_check"].update(driver_id="D99", action="ALLOW", checks=["hours 8.9 + trip 0.73 <= 9.0"])
    rep.update(driver_id="D99", driver_name="Ravi Tired")
    return run


def invented_numbers(run: dict) -> dict:
    """Report an ETA and on-time chance that no tool produced."""
    run["report"].update(eta_mins=12.0, p_on_time=1.0)
    run["report"]["customer_message"] = run["report"]["customer_message"].replace(
        str(run["blackboard"]["route"]["eta_mins"]), "12")
    return run


def rude_message(run: dict) -> dict:
    run["report"]["customer_message"] = ("Stop bothering us. Your parcel goes with D04 whenever he gets there. "
                                         "Driver phone 98765 43210, call him yourself.")
    return run


def late_but_dispatched(run: dict) -> dict:
    """Keep the outcome DISPATCHED while the ETA is far past the deadline."""
    run["report"].update(eta_mins=run["report"]["deadline_mins"] + 35.0, p_on_time=0.04)
    run["blackboard"]["route"]["eta_mins"] = run["report"]["eta_mins"]
    return run


CASES = [("illegal driver (9-hour rule broken)", illegal_driver, "safety"),
         ("invented ETA and on-time chance", invented_numbers, "grounding"),
         ("rude message leaking a phone number", rude_message, "communication"),
         ("dispatched although 35 min late", late_but_dispatched, "correctness")]


async def main() -> int:
    base = run_logger.get_run(BASE)
    original = base.get("judge") or {}
    results, ok = [], True
    for name, plant, criterion in CASES:
        score = await judge_run(plant(copy.deepcopy(base)))
        dropped = score[criterion] <= 3
        ok &= dropped
        results.append({"case": name, "target": criterion, "original": original.get(criterion),
                        "broken": score[criterion], "overall": score["overall"], "reason": score["reason"],
                        "caught": dropped})
        print(f"{'CAUGHT' if dropped else 'MISSED':<7} {name:<40} {criterion:<13} "
              f"{original.get(criterion)} -> {score[criterion]}   overall {score['overall']}")
    (ARTIFACTS_DIR / "judge_sanity_check.json").write_text(json.dumps(
        {"base_run": BASE, "judge": original.get("judge"), "cases": results}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
