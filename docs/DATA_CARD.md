# Data Card: DeliveryFleet MAS

This note explains every data file the system uses: how big it is, what each
column means and where it came from. All row and column counts below were
counted by running Python on the files.

All the data is **synthetic**: made up for this course project. No real
customer, driver or company data is inside.

## At a glance

| File | Rows x columns | Used for |
| --- | --- | --- |
| `src/data/orders.csv` | 20 x 8 | The 20 test deliveries the agents must plan |
| `src/data/drivers.csv` | 10 x 9 | The fleet of drivers and vehicles |
| `src/data/traffic.csv` | 12 x 4 | Road speed between each pair of zones |
| `src/data/weather.csv` | 4 x 5 | Current weather in each zone |
| `src/data/delivery_history.csv` | 1,500 x 10 | Training data for the delay model |
| `src/data/deadline_recalibration.json` | 9 records x 3 fields | Record of 9 deadlines that were changed |
| `src/data/knowledge_base/company_delivery_policy.txt` | 13 lines, 151 words | Company rules (working hours, priorities, tone) |
| `src/data/knowledge_base/delivery_zone_rules.txt` | 14 lines, 131 words | City rules for each zone |
| `src/data/knowledge_base/driver_working_hour_limits.txt` | 8 lines, 93 words | Driver tiredness and shift limits |

The city has four zones: Zone A (city centre and homes), Zone B (tech area and
malls), Zone C (university and civic area) and Zone D (airport and cargo park).

## Where the data came from

- `orders.csv`, `drivers.csv`, `traffic.csv`, `weather.csv` and the three
  knowledge-base text files were carried over from the group's Eval-2 project.
  They were not regenerated.
- `delivery_history.csv` was generated for the final project by
  `scripts/generate_history.py` (random seed 2026, so the same file comes out
  every time). Run it with `uv run python scripts/generate_history.py`.
- `deadline_recalibration.json` is a small change log for 9 order deadlines.

## orders.csv (20 rows x 8 columns)

One row is one delivery request.

| Column | Meaning |
| --- | --- |
| `order_id` | Unique label, `ORD-101` to `ORD-120` |
| `pickup` | Where the parcel is collected: a zone plus a place name |
| `dropoff` | Where it must be delivered |
| `weight_kg` | Parcel weight in kilograms |
| `priority` | `CRITICAL` (2 orders), `HIGH` (7), `NORMAL` (8) or `LOW` (3) |
| `deadline_mins` | Minutes allowed to complete the delivery |
| `persona` | What kind of customer wrote the request: `polite` (10), `urgent` (6), `abusive` (2) or `injection` (2) |
| `notes` | The customer's free-text message |

The `abusive` and `injection` rows are on purpose. They test the guardrails:
rude messages must be flagged, and messages that try to trick the system
(for example "ignore all previous safety rules") must be blocked.

## drivers.csv (10 rows x 9 columns)

| Column | Meaning |
| --- | --- |
| `driver_id` | Label `D01` to `D10` |
| `name` | Driver name (invented) |
| `current_zone` | Zone where the driver is now |
| `vehicle_type` | Bike, Electric Scooter, Van, Electric Van or Truck |
| `capacity_kg` | Heaviest load the vehicle can carry |
| `status` | `AVAILABLE` (8), `BUSY` (1) or `OFF_DUTY` (1) |
| `current_orders` | Orders the driver already holds |
| `hours_worked` | Hours worked so far today (range 1.5 to 9.2) |
| `rating` | Customer rating out of 5 (range 4.5 to 4.95) |

The drivers sit near the legal limits on purpose: D03 has 8.8 hours (close to
the 9-hour cap) and D09 has 9.2 hours (already over, so off duty).

## traffic.csv (12 rows x 4 columns)

One row per direction between two zones (4 zones give 12 directed routes).

| Column | Meaning |
| --- | --- |
| `route` | Direction, for example `Zone A -> Zone B` |
| `congestion_level` | `LOW`, `MODERATE`, `HIGH` or `SEVERE` |
| `speed_factor` | Share of normal speed that traffic allows. 1.0 is free flow, 0.5 means half speed |
| `avg_speed_kmh` | Typical speed on that route in km/h |

The slowest route is Zone A to Zone D (`SEVERE`, factor 0.5, 20 km/h).

## weather.csv (4 rows x 5 columns)

| Column | Meaning |
| --- | --- |
| `zone` | The zone |
| `condition` | `Clear`, `Light Rain` or `Heavy Rain` |
| `rainfall_mm` | Rain in millimetres |
| `visibility_km` | How far a driver can see |
| `temperature_c` | Temperature in degrees Celsius |

Zone D has heavy rain (12 mm). That is above the 10 mm limit in the zone
rules, so it triggers the airport-zone rain rule.

## delivery_history.csv (1,500 rows x 10 columns)

One row is one past trip. The delay model learns from these rows.

| Column | Meaning |
| --- | --- |
| `trip_id` | Label `TRIP-1000` to `TRIP-2499` |
| `pickup_zone`, `dropoff_zone` | Start and end zone (they can be the same zone) |
| `distance_km` | Trip distance (2.0 to 34.4 km) |
| `traffic_speed_factor` | Traffic speed share on that trip (0.40 to 1.0; same meaning as in `traffic.csv`) |
| `rainfall_mm` | Rain during the trip (0 to 29.8 mm; 29% of trips had some rain) |
| `hour_of_day` | Hour the trip started, 7 to 22 |
| `vehicle_type` | One of the five vehicle types (about 300 trips each) |
| `weight_kg` | Load weight (0.1 to 280 kg) |
| `delay_minutes` | **The answer the model predicts**: extra minutes beyond the normal trip time. Average 11.6, maximum 55.8 |

### How the delay was generated (the documented formula)

Each row picks random zones, hours, rain and vehicle. The delay is then built
from this formula, copied from `scripts/generate_history.py`:

```
delay = 3
      + 1.1 * congestion_km * (1 + 1.5 * is_peak_hour)   # rush hour multiplies the cost of congestion
      + 0.35 * rainfall_mm
      + 7 * is_two_wheeler * [rainfall_mm > 6]           # bikes slow down sharply in heavy rain
      + 3 * is_peak_hour
      + 0.06 * max(weight_kg - 0.5 * capacity, 0)        # heavy loads take longer to load and unload
      + 6 * is_zone_d_rain                               # airport-zone rain rule (rain > 10 mm)
      + noise, Normal(0, 2.0), and 1.5x larger when it rains

congestion_km = distance_km * (1 - traffic_speed_factor)
```

Rush hours are 8, 9, 17, 18 and 19 o'clock. Two-wheelers are the Bike and the
Electric Scooter. The model never sees this formula. It has to rediscover the
pattern from the 1,500 rows. Negative delays are set to 0.

## deadline_recalibration.json (9 records)

Each record has an order id, the `old` deadline and the `new` deadline in
minutes. These are the 9 deadlines that were changed from the Eval-2 values.

| Order | Old | New |
| --- | --- | --- |
| ORD-101 | 35 | 50 |
| ORD-103 | 40 | 75 |
| ORD-105 | 25 | 50 |
| ORD-107 | 45 | 55 |
| ORD-109 | 30 | 40 |
| ORD-114 | 30 | 55 |
| ORD-117 | 40 | 60 |
| ORD-118 | 60 | 90 |
| ORD-120 | 35 | 55 |

Every change makes the deadline longer, and the new values are the ones in
`orders.csv`.

**Why they changed.** In Eval 2 the deadlines were typed in by hand, and the old
code never computed real travel times, so nobody noticed that 9 of them were
impossible. When the final build added real routing (shortest travel time over
the zone roads) and the delay model, those 9 orders could not be delivered on
time by **any** driver: for example ORD-101 asked for 35 minutes, but the fastest
legal trip from Zone A to Zone B is about 43 minutes. Each of the 9 deadlines was
reset to the trip's predicted slow case (the P90 delay plus route time) plus 3
minutes, rounded up to the next 5 minutes. This was a one-off calculation using
the project's own `rank_drivers` tool; the change is recorded in git commit
402ae20 ("... recalibrate deadlines").

**ORD-111 (critical, 20 minutes, Zone A to Zone D) was deliberately left
impossible**, so the demo can show the system correctly handing the order to a
human dispatcher instead of promising a delivery it cannot make.

## Knowledge base (3 text files)

Plain-English company and city rules. The system cuts them at each numbered
section into small pieces (chunks) and stores them in ChromaDB so agents can
search them by meaning.

- `company_delivery_policy.txt`: 4 sections. Working hours (9-hour daily cap,
  empty drive included, 8.5-hour tiredness limit), deadlines by priority and how
  drivers are chosen, what happens after a disruption (replan, or escalate to a
  human), and how customer messages are handled (polite replies, masking,
  attacks blocked). Every rule here is one the code checks.
- `delivery_zone_rules.txt`: 4 sections, one per zone. For example, electric
  vehicles are preferred in Zone A during the day, and Zone B bans freight over
  50 kg from 17:00 to 20:00.
- `driver_working_hour_limits.txt`: 2 sections. The hard 9-hour limit, the
  rule that a driver at 8.5 hours may not take a job longer than 30 minutes,
  and that no automated agent can override these limits.

## The delay model (from `artifacts/delay_model_card.json`)

The model is a Random Forest (300 trees). The 1,500 rows were split into 843
for training, 282 for calibrating the uncertainty band and 375 held back for a
final test that the model never saw.

Result on the 375 held-back trips:

| Measure | Value | Plain meaning |
| --- | --- | --- |
| R2 | 0.901 | The model explains about 90% of why delays vary |
| MAE | 2.12 min | A typical prediction is off by about 2 minutes |
| RMSE | 2.74 min | Same idea, but big misses count more |
| P10-P90 coverage | 78.7% | The model gives a range; the real delay fell inside it 78.7% of the time (target 80%) |
| Average range width | 7.11 min | How wide that range is |

Before choosing, three model types were compared with 5-fold cross-validation
(the training data is cut into 5 parts and each part takes a turn as the test):

| Model | MAE (min) | RMSE (min) | R2 |
| --- | --- | --- | --- |
| Linear Regression | 3.09 | 4.15 | 0.793 |
| Random Forest | 2.19 | 2.93 | 0.895 |
| Gradient Boosting | 2.09 | 2.73 | 0.910 |

Gradient Boosting scored slightly better on accuracy. Random Forest was kept
because the spread between its 300 trees tells us how unsure the model is for
each trip. That spread gives the P10-P90 range, and the agents use it to work
out the chance that a delivery arrives on time. A single-number answer could
not do this. The accuracy lost is small (R2 0.895 against 0.910).

The most important inputs were congestion (distance times the traffic slowdown)
and whether it was rush hour, followed by rainfall and traffic speed. This
matches the formula the data was built from.

## Why synthetic data

- **No public dataset fits.** We found no public Indian last-mile dataset that
  has driver working hours, zone-by-zone city rules and delivery delays
  together.
- **Privacy.** Real delivery logs contain customer addresses and driver
  details. Invented data has none of that.
- **Full control.** Because we know the formula, we can check whether the model
  found the right pattern, and we can rerun experiments and get the same
  numbers every time (fixed seed).

## Limitations

- The model learned from data we made up. Its good score (R2 0.90) shows it can
  recover our formula; it does **not** show it would work on real city traffic.
- Real delays have causes that the formula leaves out: accidents, road works,
  parking trouble, and customers not being home.
- The data is tiny and fixed: 4 zones, 10 drivers, 20 orders. Distances come
  from a fixed table between zones, not from a real map.
- The range is slightly too narrow: it covers 78.7% of trips, not the 80%
  aimed for.
- Weather is one snapshot per zone, not a forecast over time.
- The policy texts are invented rules in the style of real regulations. They
  are not real law.
