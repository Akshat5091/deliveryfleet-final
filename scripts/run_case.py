"""Run one demo case from the command line and print what each agent did.

  uv run python scripts/run_case.py standard
  uv run python scripts/run_case.py exception --arch sequential --style Direct
Needs the MCP server and the ADK route service running (start.py starts both).
"""
from __future__ import annotations

import argparse
import asyncio
import json

from src.evaluation.personas import DEMO_CASES, message_for, order_hint
from src.observability import run_logger, tracing
from src.orchestration import get_orchestrator
from src.orchestration.base import RunSpec


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("case", choices=[c["id"] for c in DEMO_CASES])
    p.add_argument("--arch", default="hierarchical", choices=["hierarchical", "sequential", "parallel"])
    p.add_argument("--style", default="ReAct", choices=["ReAct", "Direct"])
    p.add_argument("--scenario", default=None)
    a = p.parse_args()
    tracing.init("deliveryfleet-cli")
    case = next(c for c in DEMO_CASES if c["id"] == a.case)
    message = case.get("message") or message_for(case["order_id"], case["persona"])
    hint = order_hint(case["order_id"]) if case["order_id"] and case["persona"] != "custom" else None
    spec = RunSpec(message=message, persona=case["persona"], prompt_style=a.style,
                   scenario=a.scenario or case["scenario"], order=hint, source="cli")
    report = await get_orchestrator(a.arch)(spec)
    for ev in run_logger.events_after(spec.run_id):
        if ev["kind"] in ("agent_start", "blackboard") and ev["status"] != "WRITE":
            continue
        extra = ev["payload"].get("error") or ev["payload"].get("summary") or ""
        print(f"{ev['kind']:<14} {ev['agent']:<18} {ev['name'][:28]:<28} {ev['status']:<9} "
              f"{(ev['duration_ms'] or 0):>8.0f}ms {ev['input_tokens']:>6}/{ev['output_tokens']:<5} {str(extra)[:90]}")
    print(json.dumps(report.model_dump(), indent=1))


if __name__ == "__main__":
    asyncio.run(main())
