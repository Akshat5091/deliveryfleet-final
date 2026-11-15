---
version: 1
slug: "src-web-static-index-html"
primary_target: "src/web/static/index.html"
related_targets: ["src/web/static/app.js","src/web/static/styles.css"]
---

Scope: the deployed web app (customer booking app + control room + runs/evaluation/data). Mode: operate.
Audience: faculty examiner watching a projected demo driven by the student group; job: see a delivery booked like a real product, then see which agent did what.

## Direction contract

THESIS: A delivery product with its sorting office in view. The customer side is the familiar map-plus-booking-sheet every delivery app uses; the control room is a Speed Post counter where every agent's finished work is franked with a round postmark. Refuses the grey KPI-card dashboard and the dark neon console.

OWN-WORLD: Ink shell (#111827 top bar), cool light ground (#f2f3f5) with white panels separated by 1px seams (no drop shadows on panels), India-Post crimson #991b1b for action and the postmark, forest green #166534 for delivered/verified, amber #b45309 only for live/working. Archivo for UI and numerals (tabular), JetBrains Mono only for ids, versions and log times. Postmark: double ring, agent name on the arc, time in the centre.

STORY: The examiner books a parcel and watches a rider get assigned and tracked; then opens the control room for the same booking and sees each agent franked in order, the blackboard filling, MCP and A2A messages, and the guard verdict; then picks a scenario from a dropdown and watches the system respond.

FIRST VIEWPORT: Book tab: zone map fills the left two-thirds at full height, the vehicle and route drawn on it; booking sheet on the right (pickup, drop, weight, urgency choices, note, Book delivery). After booking the sheet becomes rider card, ETA in large numerals, status timeline. Control room: scenario dropdown row on top, the agent counter strip below it, map and consignment slip side by side, log and blackboard underneath.

FORM: Speed Post Desk, assigned candidate 7 of 7. seed:369e084e1dd2b11a

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
