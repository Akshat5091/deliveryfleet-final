"""Start the three services that make up the deployed system.

  1. MCP tool server         (FastMCP, streamable HTTP)   MCP_PORT  8001
  2. ADK Route Agent service (A2A JSON-RPC + Agent Card)  A2A_PORT  8002
  3. Web app + orchestrator  (FastAPI, SSE)               PORT      7860

Each runs in its own process. The web app waits until the other two answer
before accepting traffic; if any service dies, the launcher stops all of them
so the container restarts cleanly instead of running half a system.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import httpx

from src.config import A2A_URL, HOST, MCP_PORT, WEB_PORT

SERVICES = [
    ("mcp", [sys.executable, "-m", "src.mcp.server"]),
    ("a2a", [sys.executable, "-m", "src.agents.route_service"]),
]


def wait_until_up(name: str, url: str, timeout: float = 180) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            httpx.get(url, timeout=3)
            print(f"[start] {name} is up: {url}", flush=True)
            return
        except httpx.HTTPError:
            time.sleep(1.5)
    raise RuntimeError(f"{name} did not start within {timeout:.0f}s")


def main() -> None:
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    procs = [subprocess.Popen(cmd, env=env) for _, cmd in SERVICES]
    exit_code = 1
    try:
        wait_until_up("MCP server", f"http://{HOST}:{MCP_PORT}/mcp")
        wait_until_up("ADK route service", f"{A2A_URL}/.well-known/agent-card.json")
        web_host = os.getenv("WEB_HOST", "0.0.0.0")
        procs.append(subprocess.Popen([sys.executable, "-m", "uvicorn", "src.web.app:app", "--host", web_host,
                                       "--port", str(WEB_PORT), "--log-level", "warning"], env=env))
        print(f"[start] web app on http://{web_host}:{WEB_PORT}", flush=True)
        while all(p.poll() is None for p in procs):
            time.sleep(2)
        print("[start] a service exited; stopping the rest", flush=True)
    except KeyboardInterrupt:
        exit_code = 0
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
