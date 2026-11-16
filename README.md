# DeliveryFleet MAS

A multi-agent system that dispatches parcel deliveries across four city zones. A customer books a
delivery; one supervisor agent and four worker agents decide **which driver**, **which road**, and
**what to do if something goes wrong on the way**, while obeying driver-hour laws and zone rules.

CSE3101 Agentic AI, Group 8, BML Munjal University.
Bhumika Yadav (230732), Aadit Bassi (230728), Akshat Nagori (230682), Sahil Mudgil (230743).

## What it does, in one run

```
Customer message
  -> Input Guard            blocks prompt injection, masks insults and phone numbers
  -> Supervisor (CrewAI)    reads the message and writes a structured job to the Blackboard
  -> Retrieval (CrewAI)     looks up zone and driver-hour rules in ChromaDB      } at the same
     Driver database (MCP)  scores all 10 drivers, with a delay forecast each   } time
  -> Dispatch (CrewAI)      picks the driver and explains why
  -> Route (Google ADK)     separate service, reached over A2A: best road + delay band
  -> Rule check (code)      hard limits: 9-hour shift, capacity, zone bans
  -> Exception (CrewAI)     only if a road closes, a vehicle breaks down or rain hits
  -> Supervisor review      approves, escalates to a human, or sends one worker back
  -> Final report (Pydantic)
```

The agents never call each other's code. Each one reads its part of a shared **Blackboard** and
writes its own section. Every tool is reached through one **MCP server**. The Route agent runs in its
own process and talks to the others only through the **A2A protocol**.

## Run it locally (Windows, PowerShell)

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/). A free Gemini key from aistudio.google.com.

```powershell
uv sync
"GEMINI_API_KEY=your_key" | Out-File -Encoding ascii .env
$env:WEB_HOST="127.0.0.1"; uv run python start.py
```

Open http://127.0.0.1:7860. `start.py` starts three services: the MCP tool server (port 8001), the
ADK route agent (port 8002) and the website (port 7860).

Other commands:

```powershell
uv run pytest                                          # 48 tests, no API key needed
uv run python scripts/run_case.py exception            # one demo case in the terminal
uv run python -m src.evaluation.benchmark_runner       # the 72-run evaluation (about 2 hours on one free key)
uv run python scripts/audit_benchmark.py               # recheck every benchmark number against the raw log
uv run python scripts/judge_sanity_check.py            # does the LLM judge mark planted faults down?
uv run python scripts/generate_history.py              # regenerate the synthetic delay data
```

## The website

| Tab | What it shows |
|---|---|
| Book a delivery | The customer app: book, see the rider, track the parcel. No agent internals. |
| Control room | The inside view: a scenario dropdown, each agent stamped when it finishes, map, consignment slip, live event log, Blackboard, per-agent heat. |
| Runs | Every run with its own ID; click one to replay its full trace. |
| Evaluation | The 72-run results as four questions: did it work, where does the time go, can we trust the judge, what does it cost. |
| Fleet & data | Driver database, delay-model card, orders, the Route agent's A2A Agent Card. |
| How it works | Architecture diagram and where each course requirement lives. |

## Where each requirement lives

| # | Requirement | Code |
|---|---|---|
| 1 | Blackboard shared state (plus peer-to-peer for comparison) | `src/orchestration/blackboard_manager.py` |
| 2 | Tools through MCP (5 tools) | `src/mcp/server.py`, `src/mcp/tools/` |
| 3 | Pydantic structured output | `src/schemas/decisions.py` |
| 4 | Hierarchical, sequential and parallel architectures | `src/orchestration/*_orchestrator.py` |
| 5 | Callbacks, monitoring, per-agent heatmaps | `src/observability/callbacks.py`, `src/agents/adk_route_agent.py` |
| 6 | A2A between CrewAI and ADK | `src/agents/a2a_client.py`, `src/agents/route_service.py` |
| 7 | CrewAI and ADK only | `src/agents/` |
| 8 | Guardrails, prompt variance, personas | `src/guardrails/`, `src/evaluation/personas.py` |
| 9 | Observability and provenance | `src/observability/tracing.py`, `run_logger.py` |
| 10 | 72-run evaluation, cost, boxplots | `src/evaluation/` |
| 11 | Cloud deployment | `Dockerfile`, `deploy/DEPLOY.md` |

## Folder map

```
src/
  agents/          the five agents, the Gemini wrapper, the A2A client and service
  mcp/             MCP server and its tools (RAG, routing, delay model, driver database)
  orchestration/   Blackboard, peer-to-peer inbox, the three architectures, final report
  guardrails/      input guard, tool-output validator, business rules
  observability/   event store, OpenTelemetry tracing, CrewAI/ADK callbacks
  evaluation/      personas, demo cases, LLM judge, 72-run benchmark, charts
  schemas/         Pydantic models for everything written to the Blackboard
  web/             FastAPI app and the website (HTML/CSS/JS, no build step)
  data/            synthetic data and the policy knowledge base (see docs/DATA_CARD.md)
tests/             unit and API tests
scripts/           data generator, single-case runner, benchmark audit, judge check
artifacts/         evaluation outputs and the delay-model card
docs/              data card and other documentation
deploy/            Hugging Face deployment guide
```

## Models and cost

Every agent and the input guard use `gemini-3.1-flash-lite`, the cheapest Gemini model open to new
API keys ($0.25 in / $1.50 out per 1M tokens). The LLM judge uses a different model,
`gemini-3.5-flash-lite`, so it never grades its own model's answers. Each role has its own setting
(`WORKER_MODEL`, `ROUTE_MODEL`, `JUDGE_MODEL`) so it can be changed from `.env`. Costs in the
app use Google's published paid-tier prices (see `src/config.py`). A booking makes about 14 LLM
calls (about 75 s) and costs about $0.015; a run with a disruption about 25 calls and $0.03. Thinking tokens are
included in the cost.

## Known limitations

- All data is synthetic (see `docs/DATA_CARD.md`).
- The road network is zone-level (4 hubs), not a street map.
- On the free Hugging Face tier the event store resets when the Space restarts.
- Gemini occasionally returns "high demand" errors; calls are retried with back-off, which can make a
  single run take two to three minutes.
- The free tier allows 500 calls a day per model (about 35 bookings). When that is spent, runs stop at
  once with a clear message instead of retrying.
- The ADK service keeps finished A2A sessions in memory; restart it after very long uptimes.
