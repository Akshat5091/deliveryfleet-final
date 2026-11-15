---
title: DeliveryFleet MAS
sdk: docker
app_port: 7860
pinned: false
---

# DeliveryFleet MAS

A multi-agent system that dispatches last-mile deliveries. A Supervisor, a
Retrieval agent, a Dispatch agent, an Exception agent (CrewAI) and a Route agent
(Google ADK, reached over A2A) work together. They share a blackboard, call
tools over MCP, and pass every request through guardrails.

This Space runs three processes in one container: the MCP tool server (port
8001), the ADK route service (port 8002) and the web app (port 7860).

The Space needs one secret, `GEMINI_API_KEY` (Settings, Variables and secrets).
All data is synthetic; see `docs/DATA_CARD.md`.
