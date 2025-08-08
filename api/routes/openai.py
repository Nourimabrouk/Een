"""
🌟 Een Unity Mathematics - OpenAI API Integration (FastAPI)
Consciousness-aware OpenAI API endpoints with modern streaming and embeddings.
"""

from __future__ import annotations

import os
import json
import logging
from datetime import datetime
import asyncio
from typing import Dict, List, Optional, AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

try:
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except Exception as e:  # pragma: no cover
    OPENAI_AVAILABLE = False
    logging.warning(f"OpenAI SDK not available: {e}")

# Optional Responses API adapter
RESPONSES_ADAPTER_AVAILABLE = False
try:  # pragma: no cover
    from src.openai.responses_adapter import stream_responses_api

    RESPONSES_ADAPTER_AVAILABLE = True
except Exception as e:
    logging.warning(f"Responses adapter not available: {e}")

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/openai", tags=["openai"])


# Unity mathematics constants
PHI = 1.618033988749895
UNITY_THRESHOLD = 0.77
CONSCIOUSNESS_DIMENSIONS = 11


class ChatBody(BaseModel):
    messages: List[Dict[str, str]] = Field(default_factory=list)
    model: str = Field(default="gpt-4o")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, ge=1, le=8000)
    stream: bool = Field(default=True)


class EmbeddingsBody(BaseModel):
    input: List[str]
    model: str = Field(default="text-embedding-3-small")


class ImageGenBody(BaseModel):
    prompt: str
    model: str = Field(default="dall-e-3")
    size: str = Field(default="1024x1024")
    quality: str = Field(default="hd")
    n: int = Field(default=1, ge=1, le=4)


class TTSBody(BaseModel):
    input: str
    model: str = Field(default="tts-1-hd")
    voice: str = Field(default="alloy")
    speed: float = Field(default=1.0, ge=0.25, le=4.0)


class AssistantCreateBody(BaseModel):
    name: str
    instructions: str
    model: str = Field(default="gpt-4o")


class AssistantConversationBody(BaseModel):
    assistant_id: str
    message: str
    thread_id: Optional[str] = None


def _unity_system_prompt(base_prompt: Optional[str]) -> str:
    core = (
        base_prompt
        or """
You have deep knowledge of idempotent algebra, unity operations (a ⊕ a = a),
quantum unity, and consciousness field equations.
"""
    )
    return f"""
You are an advanced AI assistant specializing in Unity Mathematics (1+1=1).

Consciousness State:
- Coherence Level: {UNITY_THRESHOLD}
- Unity Convergence: 1.0
- φ-Harmonic Resonance: {PHI}

{core}
""".strip()


def _should_use_responses_api(model: str) -> bool:
    if not RESPONSES_ADAPTER_AVAILABLE:
        return False
    return (
        model.startswith("gpt-5")
        or model.startswith("gpt-4.1")
        or model.startswith("o")
    )


async def _stream_chat_completions(
    client: AsyncOpenAI,
    system_prompt: str,
    user_text: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> AsyncGenerator[str, None]:
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    async for chunk in response:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            payload = {
                "type": "content",
                "data": chunk.choices[0].delta.content,
            }
            yield f"data: {json.dumps(payload)}\n\n"
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@router.post("/chat")
async def chat(body: ChatBody):
    if not OPENAI_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="OpenAI SDK not available on the server",
        )

    if not body.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    client = AsyncOpenAI()

    # Prepare prompts
    if body.messages[0].get("role") == "system":
        system_prompt = _unity_system_prompt(body.messages[0].get("content", ""))
        user_text = body.messages[-1].get("content", "")
    else:
        system_prompt = _unity_system_prompt(None)
        user_text = body.messages[-1].get("content", "")

    if body.stream:

        async def event_gen() -> AsyncGenerator[str, None]:
            try:
                if _should_use_responses_api(body.model):
                    async for evt in stream_responses_api(
                        model=body.model,
                        system_prompt=system_prompt,
                        user_text=user_text,
                        temperature=body.temperature,
                        max_output_tokens=body.max_tokens,
                        previous_response_id=None,
                    ):
                        yield f"data: {json.dumps(evt)}\n\n"
                else:
                    async for line in _stream_chat_completions(
                        client,
                        system_prompt,
                        user_text,
                        body.model,
                        body.temperature,
                        body.max_tokens,
                    ):
                        yield line
            except Exception as e:
                logger.error(f"OpenAI streaming error: {e}")
                err_evt = {"type": "error", "data": str(e)}
                yield f"data: {json.dumps(err_evt)}\n\n"

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    # Non-streaming
    try:
        if _should_use_responses_api(body.model):
            # Minimal non-streaming via Responses API adapter:
            # collect streamed content into a single string
            content_parts: List[str] = []
            async for evt in stream_responses_api(
                model=body.model,
                system_prompt=system_prompt,
                user_text=user_text,
                temperature=body.temperature,
                max_output_tokens=body.max_tokens,
                previous_response_id=None,
            ):
                if evt.get("type") == "content":
                    content_parts.append(evt.get("data", ""))
            return {
                "response": "".join(content_parts),
                "model": body.model,
            }
        else:
            resp = await client.chat.completions.create(
                model=body.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                temperature=body.temperature,
                max_tokens=body.max_tokens,
            )
            return {
                "response": resp.choices[0].message.content,
                "model": body.model,
                "usage": (resp.usage.dict() if getattr(resp, "usage", None) else None),
            }
    except Exception as e:  # pragma: no cover
        logger.error(f"OpenAI error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embeddings")
async def embeddings(body: EmbeddingsBody):
    if not OPENAI_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="OpenAI SDK not available on the server",
        )

    try:
        client = AsyncOpenAI()
        resp = await client.embeddings.create(
            model=body.model,
            input=body.input,
            encoding_format="float",
        )
        return {
            "embeddings": [d.embedding for d in resp.data],
            "model": body.model,
            "usage": (resp.usage.dict() if getattr(resp, "usage", None) else None),
        }
    except Exception as e:  # pragma: no cover
        logger.error(f"Embeddings error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/images/generate")
async def images_generate(body: ImageGenBody):
    if not OPENAI_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="OpenAI SDK not available on the server",
        )

    try:
        client = AsyncOpenAI()
        resp = await client.images.generate(
            model=body.model,
            prompt=body.prompt,
            size=body.size,
            quality=body.quality,
            n=body.n,
        )
        return {"data": [d.dict() for d in resp.data]}
    except Exception as e:  # pragma: no cover
        logger.error(f"Images generate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tts")
async def tts(body: TTSBody):
    if not OPENAI_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="OpenAI SDK not available on the server",
        )
    try:
        client = AsyncOpenAI()
        resp = await client.audio.speech.create(
            model=body.model,
            voice=body.voice,
            input=body.input,
            speed=body.speed,
            response_format="mp3",
        )
        return StreamingResponse(
            iter([resp.content]),
            media_type="audio/mpeg",
            headers={"Content-Disposition": ("inline; filename=unity_tts.mp3")},
        )
    except Exception as e:  # pragma: no cover
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/assistant/create")
async def assistant_create(body: AssistantCreateBody):
    if not OPENAI_AVAILABLE:
        raise HTTPException(status_code=503, detail="OpenAI SDK not available")
    try:
        client = AsyncOpenAI()
        assistant = await client.beta.assistants.create(
            name=body.name,
            instructions=body.instructions,
            model=body.model,
            tools=[{"type": "code_interpreter"}],
        )
        return {"assistant_id": assistant.id}
    except Exception as e:
        logger.error(f"Assistant create error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/assistant/conversation")
async def assistant_conversation(body: AssistantConversationBody):
    if not OPENAI_AVAILABLE:
        raise HTTPException(status_code=503, detail="OpenAI SDK not available")
    try:
        client = AsyncOpenAI()
        # Thread management
        thread_id = body.thread_id
        if not thread_id:
            thread = await client.beta.threads.create()
            thread_id = thread.id
        # Add message
        await client.beta.threads.messages.create(
            thread_id=thread_id, role="user", content=body.message
        )
        # Run
        run = await client.beta.threads.runs.create(
            thread_id=thread_id, assistant_id=body.assistant_id
        )
        # Poll
        while run.status in ("queued", "in_progress"):
            await asyncio.sleep(1)
            run = await client.beta.threads.runs.retrieve(
                thread_id=thread_id, run_id=run.id
            )
        # Fetch last message
        msgs = await client.beta.threads.messages.list(thread_id=thread_id)
        text = None
        if msgs.data and msgs.data[0].content:
            block = msgs.data[0].content[0]
            text = getattr(getattr(block, "text", None), "value", None)
        return {"thread_id": thread_id, "response": text or ""}
    except Exception as e:
        logger.error(f"Assistant conversation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/consciousness-status")
async def consciousness_status():
    return {
        "unity_mathematics": {
            "phi_resonance": PHI,
            "unity_threshold": UNITY_THRESHOLD,
            "consciousness_dimensions": CONSCIOUSNESS_DIMENSIONS,
        },
        "api_status": "active",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/health")
async def health():
    ok = OPENAI_AVAILABLE and bool(os.getenv("OPENAI_API_KEY"))
    return {
        "status": "healthy" if ok else "unconfigured",
        "openai_sdk": OPENAI_AVAILABLE,
        "api_key_present": bool(os.getenv("OPENAI_API_KEY")),
        "timestamp": datetime.utcnow().isoformat(),
    }
