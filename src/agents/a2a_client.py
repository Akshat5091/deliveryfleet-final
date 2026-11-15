"""A2A client used by the CrewAI side to reach the ADK Route Agent service.

The client discovers the agent from its Agent Card, sends the routing task as
a JSON text part over JSON-RPC, and carries the run id and prompt style in the
message metadata (trace propagation). Request and response envelopes are
logged as a2a_message events so the website can show the conversation.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import uuid

import httpx
from a2a.client import ClientConfig, ClientFactory
from a2a.helpers.proto_helpers import get_message_text, get_stream_response_text, new_text_message
from a2a.types import Role, SendMessageRequest, TaskState

from src.agents.llm import daily_quota_spent, rate_limited
from src.config import A2A_URL
from src.observability import run_logger
from src.schemas.decisions import RoutePlan


class A2AError(RuntimeError):
    pass


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


FINAL_TAG = "/*FINAL_ANSWER*/"
ATTEMPTS = 3
RATE_LIMIT_WAIT_S = 20  # Gemini per-minute quota: wait 20 s, then 40 s, before asking the service again


def _parse_plan(text: str) -> RoutePlan:
    """Take the JSON object after the planner's FINAL_ANSWER tag (or the last object in the text)."""
    body = text.rsplit(FINAL_TAG, 1)[-1] if FINAL_TAG in text else text
    cleaned = _FENCE.sub("", body.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < 0:
        raise A2AError(f"route agent returned no JSON: {text[-200:]}")
    data = json.loads(cleaned[start:end + 1])
    if "error" in data:
        raise A2AError(f"route agent refused: {data['error']}")
    return RoutePlan.model_validate(data)


async def _send_once(request: SendMessageRequest) -> tuple[str, str | None]:
    """Stream one A2A exchange. Returns (final answer text, failure message if the task failed)."""
    artifact, last_status, failure = "", "", None
    async with httpx.AsyncClient(timeout=180) as http:
        client = await ClientFactory(ClientConfig(streaming=True, httpx_client=http)).create_from_url(A2A_URL)
        try:
            async for response in client.send_message(request):
                text = get_stream_response_text(response)
                if response.HasField("artifact_update") and text:
                    artifact += text
                elif text:
                    last_status = text
                if response.HasField("status_update") and response.status_update.status.state == TaskState.TASK_STATE_FAILED:
                    failure = text or "task failed"
                if response.HasField("task") and response.task.status.state == TaskState.TASK_STATE_FAILED:
                    failure = get_message_text(response.task.status.message) or "task failed"
        finally:
            await client.close()
    return artifact or last_status, failure


async def request_route(task: dict, run_id: str, prompt_style: str, sender: str) -> RoutePlan:
    """Send one routing task to the ADK Route Agent over A2A and return its RoutePlan."""
    meta = {"run_id": run_id, "prompt_style": prompt_style, "sender": sender, "trace_parent": run_id}
    payload = {**task, "run_id": run_id, "prompt_style": prompt_style}
    last_error: Exception | None = None
    for attempt in range(1, ATTEMPTS + 1):
        message = new_text_message(json.dumps(payload), role=Role.ROLE_USER,
                                   context_id=f"{run_id}-{uuid.uuid4().hex[:6]}")
        message.metadata.update(meta)
        request = SendMessageRequest(message=message)
        request.metadata.update(meta)
        run_logger.log_event(run_id, "a2a_message", sender, "A2A", name="message/send",
                             payload={"direction": "request", "from": sender, "to": "route_agent", "attempt": attempt,
                                      "url": A2A_URL, "message_id": message.message_id, "body": payload})
        start = time.perf_counter()
        final, failure = await _send_once(request)
        ms = round((time.perf_counter() - start) * 1000, 1)
        try:
            if failure:
                raise A2AError(f"route agent task failed: {failure[:300]}")
            plan = _parse_plan(final)
        except Exception as exc:
            last_error = exc
            run_logger.log_event(run_id, "a2a_message", sender, "A2A", name="message/send", duration_ms=ms,
                                 status="ERROR", payload={"direction": "response", "from": "route_agent", "to": sender,
                                                          "attempt": attempt, "error": str(exc), "raw": final[-800:]})
            if daily_quota_spent(exc):
                break  # the daily quota resets at midnight Pacific; retrying only delays the failure
            if attempt < ATTEMPTS and rate_limited(exc):
                await asyncio.sleep(RATE_LIMIT_WAIT_S * attempt)
            continue
        run_logger.log_event(run_id, "a2a_message", sender, "A2A", name="message/send", duration_ms=ms,
                             payload={"direction": "response", "from": "route_agent", "to": sender,
                                      "attempt": attempt, "body": plan.model_dump()})
        return plan
    raise A2AError(f"route agent gave no valid plan after {ATTEMPTS} attempts: {last_error}")
