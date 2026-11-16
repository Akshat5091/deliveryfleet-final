"""Copy the 72 benchmark runs and the live demo cases into deploy/seed_runs.sqlite3.

A freshly started server copies this file into its empty run store (start.py), so the deployed
Runs tab can replay real runs straight away. Only masked customer text is stored in the runs.

  uv run python scripts/build_seed_runs.py
"""
from __future__ import annotations

import sqlite3
import sys
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SEED_RUNS_DB, STORE_DB  # noqa: E402
from src.observability.run_logger import SCHEMA  # noqa: E402

# One live run per control-room demo case, plus one customer booking (16 Nov 2026).
DEMO_RUNS = [
    "RUN-1116-120800-0d0f",  # standard dispatch
    "RUN-1116-120903-2ac1",  # road closed mid-delivery
    "RUN-1116-122703-0c8d",  # vehicle breakdown
    "RUN-1116-121652-805e",  # abusive customer
    "RUN-1116-121820-cc30",  # prompt injection
    "RUN-1116-121825-9a3b",  # paraphrased attack
    "RUN-1116-121829-678b",  # impossible deadline
    "RUN-1116-122217-8359",  # overweight parcel
    "RUN-1116-122337-10c4",  # same-zone hop
    "RUN-1116-123512-edd3",  # customer booking
]


def main() -> None:
    SEED_RUNS_DB.unlink(missing_ok=True)
    with closing(sqlite3.connect(SEED_RUNS_DB)) as seed:
        seed.executescript(SCHEMA)
        seed.execute("ATTACH DATABASE ? AS live", (str(STORE_DB),))
        marks = ",".join("?" * len(DEMO_RUNS))
        keep = f"(SELECT run_id FROM live.runs WHERE source = 'benchmark' OR run_id IN ({marks}))"
        seed.execute(f"INSERT INTO runs SELECT * FROM live.runs WHERE run_id IN {keep}", DEMO_RUNS)
        seed.execute(f"INSERT INTO events SELECT * FROM live.events WHERE run_id IN {keep}", DEMO_RUNS)
        seed.execute(f"INSERT INTO spans SELECT * FROM live.spans WHERE run_id IN {keep}", DEMO_RUNS)
        seed.commit()
        runs = seed.execute("SELECT count(*) FROM runs").fetchone()[0]
        missing = [r for r in DEMO_RUNS if not seed.execute("SELECT 1 FROM runs WHERE run_id = ?", (r,)).fetchone()]
        seed.execute("DETACH DATABASE live")
        seed.execute("VACUUM")
    print(f"{SEED_RUNS_DB.name}: {runs} runs, {SEED_RUNS_DB.stat().st_size / 1e6:.1f} MB; missing: {missing or 'none'}")


if __name__ == "__main__":
    main()
