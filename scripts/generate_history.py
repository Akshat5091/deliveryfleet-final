"""Generate the synthetic delivery-history dataset used to train the delay model.

The data is synthetic (no real fleet logs were available). The generating
process is written out below so the model's behaviour can be explained:

    delay = 3
          + 1.1 * congestion_km * (1 + 1.5 * is_peak_hour)   peak hours multiply congestion cost
          + 0.35 * rainfall_mm
          + 7 * is_two_wheeler * [rainfall_mm > 6]           bikes slow sharply past a rain threshold
          + 3 * is_peak_hour
          + 0.06 * max(weight_kg - 0.5 * capacity, 0)        heavy loads load/unload slower
          + 6 * is_zone_d_rain                               airport-zone rain protocol (> 10 mm)
          + noise ~ Normal(0, 2.0)   (heteroscedastic: x1.5 when raining)

    where congestion_km = distance_km * (1 - speed_factor).

The model never sees this formula; it has to recover it from 1,500 rows.
Run:  uv run python scripts/generate_history.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import HISTORY_CSV, ZONES
from src.mcp.tools.routing_engine import ZONE_DISTANCES_KM, traffic_table

VEHICLES = {"Bike": 20, "Electric Scooter": 25, "Van": 90, "Electric Van": 75, "Truck": 280}
SEED = 2026
N_ROWS = 1500


def main() -> None:
    rng = np.random.default_rng(SEED)
    traffic = traffic_table()
    rows = []
    for i in range(N_ROWS):
        src, dst = rng.choice(ZONES, size=2)
        same = src == dst
        distance = rng.uniform(2, 7) if same else ZONE_DISTANCES_KM[frozenset({src, dst})] + rng.uniform(2, 8)
        speed_factor = rng.uniform(0.85, 1.0) if same else traffic[(src, dst)][0] + rng.normal(0, 0.05)
        speed_factor = float(np.clip(speed_factor, 0.35, 1.0))
        hour = int(rng.integers(7, 23))
        rain = float(max(0.0, rng.gamma(0.6, 6.0) - 1.0)) if rng.random() < 0.45 else 0.0
        vehicle = rng.choice(list(VEHICLES))
        capacity = VEHICLES[vehicle]
        weight = float(min(capacity, rng.gamma(2.0, capacity / 6)))
        two_wheeler = vehicle in ("Bike", "Electric Scooter")
        peak = hour in (8, 9, 17, 18, 19)
        zone_d_rain = "Zone D" in (src, dst) and rain > 10
        congestion_km = distance * (1 - speed_factor)
        delay = (
            3
            + 1.1 * congestion_km * (1 + 1.5 * peak)
            + 0.35 * rain
            + 7 * (two_wheeler and rain > 6)
            + 3 * peak
            + 0.06 * max(weight - 0.5 * capacity, 0)
            + 6 * zone_d_rain
            + rng.normal(0, 2.0 * (1.5 if rain > 0 else 1.0))
        )
        rows.append({
            "trip_id": f"TRIP-{1000 + i}",
            "pickup_zone": src,
            "dropoff_zone": dst,
            "distance_km": round(distance, 1),
            "traffic_speed_factor": round(speed_factor, 2),
            "rainfall_mm": round(rain, 1),
            "hour_of_day": hour,
            "vehicle_type": vehicle,
            "weight_kg": round(weight, 1),
            "delay_minutes": round(max(delay, 0.0), 1),
        })
    pd.DataFrame(rows).to_csv(HISTORY_CSV, index=False)
    print(f"wrote {len(rows)} rows -> {HISTORY_CSV}")


if __name__ == "__main__":
    main()
