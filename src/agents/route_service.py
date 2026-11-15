"""A2A server for the ADK Route Agent (separate process, own port).

to_a2a() wraps the ADK agent in a JSON-RPC A2A endpoint and publishes its
Agent Card at /.well-known/agent-card.json, so any A2A client (here: the
CrewAI side) can discover and call it without importing ADK.
"""
from __future__ import annotations

import uvicorn
from google.adk.a2a.utils.agent_to_a2a import to_a2a

from src.agents.llm import mark_route_service
from src.config import A2A_PORT, HOST
from src.observability import tracing

tracing.init("deliveryfleet-adk-route-service")
mark_route_service()


def build_app():
    from src.agents.adk_route_agent import build_route_agent

    return to_a2a(build_route_agent(), host=HOST, port=A2A_PORT)


def main() -> None:
    uvicorn.run(build_app(), host=HOST, port=A2A_PORT, log_level="warning")


if __name__ == "__main__":
    main()
