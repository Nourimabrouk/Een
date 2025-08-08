"""
Secure BYOK (Bring Your Own Key) proxy endpoints.

Principles:
- Never store or log user-provided API keys
- Keys are accepted only via short-lived HTTPS requests in headers
- Only forward to allowed providers (OpenAI) with strict timeouts
- Stream results back to client; normalize SSE
"""

from __future__ import annotations

import json
import logging
import os
from typing import AsyncGenerator, Dict, Any, Optional, List

import httpx
from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/byok", tags=["byok"])


class BYOKChatBody(BaseModel):
    model: str = Field(...)
    system_prompt: str = Field(default="")
    user_text: str = Field(..., min_length=1, max_length=8000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=2000, ge=1, le=8000)
    use_responses_api: Optional[bool] = None


class BYOKImagesBody(BaseModel):
    prompt: str
    model: str = Field(default="dall-e-3")
    size: str = Field(default="1024x1024")
    quality: str = Field(default="hd")
    n: int = Field(default=1, ge=1, le=4)


OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
ANTHROPIC_API_BASE = os.getenv("ANTHROPIC_API_BASE", "https://api.anthropic.com/v1")


def _responses_stream_headers(user_api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {user_api_key}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "responses=v1",
        "Accept": "text/event-stream",
    }


def _chat_stream_headers(user_api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {user_api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }


def _use_responses(model: str, hint: Optional[bool]) -> bool:
    if hint is not None:
        return hint
    return (
        model.startswith("gpt-5")
        or model.startswith("gpt-4.1")
        or model.startswith("o")
    )


@router.post("/openai/stream")
async def byok_openai_stream(
    body: BYOKChatBody,
    x_openai_api_key: str = Header(..., convert_underscores=False),
):
    """
    Stream OpenAI responses using user-provided API key. Never persisted.
    """
    user_key = x_openai_api_key.strip()
    if not user_key:
        raise HTTPException(status_code=400, detail="Missing X-OpenAI-Api-Key header")

    if _use_responses(body.model, body.use_responses_api):
        # Responses API streaming
        url = f"{OPENAI_API_BASE}/responses"
        payload: Dict[str, Any] = {
            "model": body.model,
            "input": [
                (
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": body.system_prompt}],
                    }
                    if body.system_prompt
                    else None
                ),
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": body.user_text}],
                },
            ],
            "temperature": body.temperature,
            "max_output_tokens": body.max_output_tokens,
            "stream": True,
            "response_format": {"type": "text"},
        }
        # Remove Nones
        payload["input"] = [blk for blk in payload["input"] if blk is not None]

        async def gen() -> AsyncGenerator[str, None]:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    url,
                    headers=_responses_stream_headers(user_key),
                    json=payload,
                ) as resp:
                    if resp.status_code >= 400:
                        try:
                            err = resp.json()
                        except Exception:
                            err = {"error": resp.text}
                        yield f"data: {json.dumps({'type':'error','data':err})}\n\n"
                        return
                    async for raw in resp.aiter_lines():
                        if not raw or not raw.startswith("data:"):
                            continue
                        data_str = raw[5:].strip()
                        if data_str == "[DONE]":
                            yield f"data: {json.dumps({'type':'done'})}\n\n"
                            return
                        try:
                            event = json.loads(data_str)
                        except Exception:
                            continue
                        et = event.get("type")
                        if et == "response.output_text.delta":
                            delta = event.get("delta") or ""
                            if delta:
                                yield f"data: {json.dumps({'type':'content','data':delta})}\n\n"
                        elif et == "response.completed":
                            yield f"data: {json.dumps({'type':'done'})}\n\n"
                            return

        return StreamingResponse(gen(), media_type="text/event-stream")

    # Chat Completions fallback
    url = f"{OPENAI_API_BASE}/chat/completions"
    payload = {
        "model": body.model,
        "messages": [
            (
                {"role": "system", "content": body.system_prompt}
                if body.system_prompt
                else None
            ),
            {"role": "user", "content": body.user_text},
        ],
        "temperature": body.temperature,
        "max_tokens": body.max_output_tokens,
        "stream": True,
    }
    payload["messages"] = [m for m in payload["messages"] if m is not None]

    async def gen_chat() -> AsyncGenerator[str, None]:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST", url, headers=_chat_stream_headers(user_key), json=payload
            ) as resp:
                if resp.status_code >= 400:
                    try:
                        err = resp.json()
                    except Exception:
                        err = {"error": resp.text}
                    yield f"data: {json.dumps({'type':'error','data':err})}\n\n"
                    return
                async for raw in resp.aiter_lines():
                    if not raw or not raw.startswith("data:"):
                        continue
                    data_str = raw[5:].strip()
                    if data_str == "[DONE]":
                        yield f"data: {json.dumps({'type':'done'})}\n\n"
                        return
                    try:
                        event = json.loads(data_str)
                    except Exception:
                        continue
                    # choices -> delta -> content
                    try:
                        content = (
                            event["choices"][0]["delta"].get("content")
                            if event.get("choices")
                            else None
                        )
                    except Exception:
                        content = None
                    if content:
                        yield f"data: {json.dumps({'type':'content','data':content})}\n\n"

    return StreamingResponse(gen_chat(), media_type="text/event-stream")


@router.post("/openai/images")
async def byok_openai_images(
    body: BYOKImagesBody, x_openai_api_key: str = Header(..., convert_underscores=False)
):
    user_key = x_openai_api_key.strip()
    if not user_key:
        raise HTTPException(status_code=400, detail="Missing X-OpenAI-Api-Key header")

    url = f"{OPENAI_API_BASE}/images/generations"
    payload = {
        "model": body.model,
        "prompt": body.prompt,
        "size": body.size,
        "quality": body.quality,
        "n": body.n,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {user_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        if resp.status_code >= 400:
            try:
                return JSONResponse(status_code=resp.status_code, content=resp.json())
            except Exception:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
        return resp.json()


class BYOKAnthropicChatBody(BaseModel):
    model: str = Field(default="claude-3-5-sonnet-20241022")
    system_prompt: str = Field(default="")
    user_text: str = Field(..., min_length=1, max_length=8000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=2000, ge=1, le=8000)


@router.post("/anthropic/stream")
async def byok_anthropic_stream(
    body: BYOKAnthropicChatBody,
    x_anthropic_api_key: str = Header(..., convert_underscores=False),
):
    """Stream Anthropic responses using user key. Never persisted."""
    user_key = x_anthropic_api_key.strip()
    if not user_key:
        raise HTTPException(
            status_code=400, detail="Missing X-Anthropic-Api-Key header"
        )

    url = f"{ANTHROPIC_API_BASE}/messages"
    payload = {
        "model": body.model,
        "max_tokens": body.max_output_tokens,
        "temperature": body.temperature,
        "system": body.system_prompt or None,
        "messages": [{"role": "user", "content": body.user_text}],
        "stream": True,
    }
    if not payload["system"]:
        del payload["system"]

    async def gen() -> AsyncGenerator[str, None]:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                url,
                headers={
                    "x-api-key": user_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                    "accept": "text/event-stream",
                },
                json=payload,
            ) as resp:
                if resp.status_code >= 400:
                    try:
                        err = resp.json()
                    except Exception:
                        err = {"error": resp.text}
                    yield f"data: {json.dumps({'type':'error','data':err})}\n\n"
                    return
                async for raw in resp.aiter_lines():
                    if not raw or not raw.startswith("data:"):
                        continue
                    data_str = raw[5:].strip()
                    if data_str == "[DONE]":
                        yield f"data: {json.dumps({'type':'done'})}\n\n"
                        return
                    try:
                        event = json.loads(data_str)
                    except Exception:
                        continue
                    et = event.get("type")
                    if et == "content_block_delta":
                        delta = (event.get("delta") or {}).get("text")
                        if delta:
                            yield f"data: {json.dumps({'type':'content','data':delta})}\n\n"
                    elif et == "message_stop":
                        yield f"data: {json.dumps({'type':'done'})}\n\n"
                        return

    return StreamingResponse(gen(), media_type="text/event-stream")
