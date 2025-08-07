# flake8: noqa
"""
OpenAI Responses API adapter with streaming (SSE) and session continuity.

This module provides a minimal, robust bridge to the Responses API while the
rest of the codebase still uses Chat Completions. It supports:
- Streaming text output via SSE
- Optional previous_response_id for conversation continuity
- Normalized chunking compatible with existing SSE consumers

Security: uses server-side OPENAI_API_KEY only. No client keys are accepted.
"""

from __future__ import annotations

import os
import json
from typing import AsyncGenerator, Dict, Any, Optional, List

import httpx


OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")


def _build_responses_input(system_prompt: str, user_text: str) -> List[Dict[str, Any]]:
    """Build Responses API input blocks from system and user texts."""
    input_blocks: List[Dict[str, Any]] = []

    if system_prompt:
        input_blocks.append(
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            }
        )

    input_blocks.append(
        {
            "role": "user",
            "content": [{"type": "input_text", "text": user_text}],
        }
    )

    return input_blocks


async def stream_responses_api(
    model: str,
    system_prompt: str,
    user_text: str,
    temperature: float = 0.7,
    max_output_tokens: int = 2000,
    previous_response_id: Optional[str] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream text content from the OpenAI Responses API via SSE.

    Yields normalized events:
      {"type": "content", "data": "..."}
      {"type": "done", "data": {"response_id": str, "usage": {...}}}
      {"type": "error", "data": "message"}
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        yield {"type": "error", "data": "OpenAI API key not configured"}
        return

    url = f"{OPENAI_API_BASE}/responses"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Some deployments require this beta header for Responses
        "OpenAI-Beta": "responses=v1",
        "Accept": "text/event-stream",
    }

    payload: Dict[str, Any] = {
        "model": model,
        "input": _build_responses_input(system_prompt, user_text),
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "stream": True,
        # Prefer plain text output for normalization; callers can switch later
        "response_format": {"type": "text"},
    }

    if previous_response_id:
        payload["previous_response_id"] = previous_response_id

    response_id: Optional[str] = None
    final_usage: Optional[Dict[str, Any]] = None

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as resp:
            if resp.status_code >= 400:
                try:
                    err = resp.json()
                except Exception:
                    err = {"error": resp.text}
                yield {
                    "type": "error",
                    "data": f"Responses API HTTP {resp.status_code}: {err}",
                }
                return

            async for raw_line in resp.aiter_lines():
                if not raw_line:
                    continue
                if raw_line.startswith("event:"):
                    # we don't use the value here; we parse data lines
                    continue
                if not raw_line.startswith("data:"):
                    continue

                data_str = raw_line[5:].strip()
                if data_str == "[DONE]":
                    # Final completion indicator (some SDKs send this)
                    yield {
                        "type": "done",
                        "data": {"response_id": response_id, "usage": final_usage},
                    }
                    return

                try:
                    event = json.loads(data_str)
                except Exception:
                    # Non-JSON data; ignore
                    continue

                # Normalize key events
                etype = event.get("type")
                if etype == "response.created":
                    response = event.get("response", {})
                    response_id = response.get("id", response_id)
                elif etype == "response.output_text.delta":
                    # Newer event names may carry deltas
                    delta = event.get("delta", "")
                    if delta:
                        yield {"type": "content", "data": delta}
                elif etype == "response.output_item.added":
                    # Sometimes full text appears in output_item
                    item = event.get("item", {})
                    if item.get("type") == "message":
                        # message.content is a list of blocks; each may have text
                        for block in item.get("content", []) or []:
                            text = block.get("text")
                            if text:
                                yield {"type": "content", "data": text}
                elif etype == "response.completed":
                    response = event.get("response", {})
                    response_id = response.get("id", response_id)
                    final_usage = response.get("usage")
                    yield {
                        "type": "done",
                        "data": {
                            "response_id": response_id,
                            "usage": final_usage,
                        },
                    }
                    return
                elif etype == "error":
                    err = event.get("error", "Unknown error")
                    yield {"type": "error", "data": err}
                    return
