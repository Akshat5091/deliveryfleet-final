# Eval-2 build vs final build

What the Phase-2 project (`DeliveryFleet-MAS`) did, what was wrong with it, and what the final build
(`DeliveryFleet-Final/Final Project`) does instead. Kept: the domain, the five agents and their names,
the data, the file names the Eval-2 report cites, the 72-run design.

## The biggest change: the agents now actually run

| Area | Eval-2 build | Final build |
|---|---|---|
| CrewAI agents | `Agent` objects were created but never run; decisions came from Python if/else code | Every CrewAI agent runs a real `Crew.kickoff` with Gemini, tools and a Pydantic output |
| Google ADK agent | Object created, logic in plain Python, called in the same process | Real ADK `LlmAgent` in its own service, run by ADK's runner behind an A2A endpoint |
| MCP | FastMCP server existed, but agents called the tool classes directly | Agents reach tools only through the MCP server over HTTP (CrewAI `MCPServerHTTP`, ADK `McpToolset`) |
| A2A | Typed messages passed between Python objects in one process; the endpoint was a placeholder | Real A2A JSON-RPC over HTTP, Agent Card published, run id carried in metadata |
| Blackboard | Written to, but downstream agents mostly got their inputs as function arguments | Every hand-off goes through the Blackboard; each agent may read only its own sections; every read and write logged |
| Callbacks | A callback class existed but nothing called it | CrewAI event listener and tool hooks, all six ADK callbacks, wired into every run |
| Tokens and cost | Hard-coded (`740 if not simulate_exception else 1045`) | Taken from Gemini's usage data on every call, thinking tokens included |
| "LLM-as-judge" | A hand-written formula (`0.25*input + 0.35*dispatch + ...`) | A real Gemini judge (a different model from the agents) with a rubric at temperature 0, checked against planted faults, plus reference answers computed from the tools |
| Heatmaps | Some cells filled with generated fallback values | Every value comes from logged runs |
| 72 runs | Finished in minutes because nothing called an LLM | About an hour across four free keys, because every run makes 14 to 25 real calls; an audit script re-checks every number from the raw log |

## Correctness and safety problems that were fixed

| Problem in Eval-2 | Fix |
|---|---|
| Route was planned after dispatch with no real routing; ETAs were made up | Time-weighted Dijkstra over the zone network with traffic and rain, inside the MCP routing tool |
| 9 order deadlines were impossible for every driver; nobody noticed because ETAs were not computed | Deadlines recalibrated and logged; ORD-111 kept impossible to show escalation |
| Delay model returned one number | Random Forest with a calibrated 80% band and an on-time probability that the agents use |
| Driver hours ignored the empty drive to the pickup | The empty drive now counts towards the 9-hour shift limit |
| A reroute never re-checked driver hours | Hard rules re-run on the final driver and final ETA after every replan |
| (found by our own audit) The Exception Agent could report a late detour as on time | A reroute's ETA must come from the Route Agent's A2A reply; a late detour is escalated |
| The LLM review alone decided whether a run succeeded | Legality and on-time are decided in code; the review cannot approve a late or illegal plan |
| The raw customer message (with phone numbers) was stored and returned by the API | Only the guard's masked text is stored |
| The Gemini guard check passed anything without the word "UNSAFE", even an empty reply | It now fails closed |
| Runs could stay "RUNNING" forever after a crash | Every run is closed as FAILED on cancellation; stale runs are closed at start-up |

## Things removed because they did not serve the 11 criteria

- HMAC request signing and an idempotency store
- A Cloudflare `wrangler.toml` edge worker
- A telematics WebSocket simulator
- About 15 one-off `scripts/patch_*.py` document patchers
- Three overlapping setup documents (`README`, `SETUP`, `SETUP_GUIDE`), now one `README.md`

## Folder structure

| Eval-2 | Final | Why |
|---|---|---|
| `src/agents, mcp, orchestration, schemas, guardrails, observability, evaluation, web` | Same folders and the same file names | The Eval-2 report's file references stay valid |
| `src/schemas/` split over four files with duplicates | `src/schemas/decisions.py`, one model per Blackboard section | One source of truth |
| `src/security/`, `src/telematics/`, `src/edge/` | Removed | Not part of the criteria |
| `scripts/` with doc patchers | `scripts/` with the data generator and a single-case runner only | Code only; documents are built from `Final Eval Documents/Sources/` |
| Tests that asserted hard-coded numbers | `tests/` with 46 behaviour tests; new ones were first run against code with the rule removed | Tests fail when the rule they protect is removed |
| One 1,169-line `index.html` | `src/web/static/` with `index.html`, `styles.css`, `app.js` | A customer app and a control room, readable and separable |
| No deployment | `Dockerfile`, `deploy/DEPLOY.md`, Hugging Face Space | Criterion 11 |
| Data with no documentation | `docs/DATA_CARD.md` | Every file, its rows and columns, and how it was made |
