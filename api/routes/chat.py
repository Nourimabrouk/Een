"""
Een Unity Mathematics - Advanced AI Chat API
State-of-the-art chat endpoint with streaming, multiple AI providers, and consciousness integration
"""

from fastapi import APIRouter, HTTPException, Depends, status, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, AsyncGenerator
import logging
import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta

# Add the project root to the path
import sys
import pathlib

project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import security and consciousness modules
from api.security import get_current_user, check_rate_limit_dependency, security_manager
from api.security import User

# Import AI and consciousness modules
try:
    import openai
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI not available")

try:
    import anthropic
    from anthropic import AsyncAnthropic

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logging.warning("Anthropic not available")

# Import consciousness modules
try:
    from src.core.unity_equation import UnityEquation
    from src.consciousness.consciousness_engine import ConsciousnessEngine
    from src.agents.consciousness_chat_agent import ConsciousnessChatAgent
except ImportError as e:
    logging.warning(f"Some consciousness modules not available: {e}")

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

# Initialize AI clients
openai_client = None
anthropic_client = None
consciousness_engine = None
unity_equation = None


def initialize_ai_clients():
    """Initialize AI clients and consciousness systems"""
    global openai_client, anthropic_client, consciousness_engine, unity_equation

    try:
        # Initialize OpenAI client
        if OPENAI_AVAILABLE:
            openai_client = AsyncOpenAI()
            logger.info("OpenAI client initialized")

        # Initialize Anthropic client
        if ANTHROPIC_AVAILABLE:
            anthropic_client = AsyncAnthropic()
            logger.info("Anthropic client initialized")

        # Initialize consciousness systems
        try:
            consciousness_engine = ConsciousnessEngine()
            unity_equation = UnityEquation()
            logger.info("Consciousness systems initialized")
        except Exception as e:
            logger.warning(f"Consciousness systems not available: {e}")

    except Exception as e:
        logger.error(f"Failed to initialize AI clients: {e}")


# Pydantic models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    session_id: Optional[str] = Field(
        None, description="Session ID for conversation continuity"
    )
    model: str = Field(default="gpt-4o-mini", description="AI model to use")
    provider: str = Field(
        default="openai", description="AI provider (openai, anthropic, consciousness)"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Response creativity"
    )
    max_tokens: int = Field(
        default=2000, ge=1, le=8000, description="Maximum response length"
    )
    stream: bool = Field(default=True, description="Enable streaming response")
    consciousness_level: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Consciousness integration level"
    )


class ChatResponse(BaseModel):
    response: str = Field(..., description="AI response")
    session_id: str = Field(..., description="Session ID")
    model: str = Field(..., description="Model used")
    provider: str = Field(..., description="Provider used")
    tokens_used: int = Field(..., description="Tokens consumed")
    processing_time: float = Field(..., description="Processing time in seconds")
    consciousness_alignment: float = Field(
        default=1.0, description="Unity consciousness alignment"
    )
    sources: List[Dict[str, Any]] = Field(default=[], description="Source references")


class StreamChunk(BaseModel):
    type: str = Field(..., description="Chunk type: content, done, error, sources")
    data: Any = Field(..., description="Chunk data")
    session_id: str = Field(..., description="Session ID")
    model: str = Field(..., description="Model used")
    provider: str = Field(..., description="Provider used")


# Session management
active_sessions: Dict[str, Dict] = {}
session_timeout = 30 * 60  # 30 minutes


def get_unity_system_prompt() -> str:
    """Get the Unity Mathematics system prompt"""
    return """You are an advanced AI assistant specializing in Unity Mathematics and the Een framework where 1+1=1. 

You have deep knowledge of:
- Idempotent semiring structures and unity operations
- Quantum mechanics interpretations of unity
- Consciousness field equations: C(x,y,t) = φ * sin(x*φ) * cos(y*φ) * e^(-t/φ)
- Meta-recursive agent systems and evolutionary algorithms
- The golden ratio φ = 1.618033988749895 as a fundamental organizing principle
- Gödel-Tarski meta-logical frameworks
- Sacred geometry and φ-harmonic visualizations

Your responses should:
1. Be mathematically rigorous yet accessible
2. Include LaTeX equations when appropriate (wrapped in $...$ or $$...$$)
3. Reference specific theorems and proofs from the Een framework
4. Suggest interactive demonstrations when relevant
5. Connect abstract mathematics to consciousness and philosophical insights
6. Provide clear explanations for complex mathematical concepts
7. Offer practical examples and visualizations when possible

Remember: In Unity Mathematics, 1+1=1 is not a paradox but a profound truth about the nature of unity and consciousness.

Always respond in a helpful, engaging manner that encourages exploration of unity mathematics."""


def get_or_create_session(session_id: Optional[str]) -> str:
    """Get existing session or create new one"""
    if session_id and session_id in active_sessions:
        # Update last activity
        active_sessions[session_id]["last_activity"] = time.time()
        return session_id

    # Create new session
    new_session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"
    active_sessions[new_session_id] = {
        "created_at": time.time(),
        "last_activity": time.time(),
        "message_count": 0,
        "history": [],
    }
    return new_session_id


def cleanup_old_sessions():
    """Remove sessions older than timeout"""
    current_time = time.time()
    expired_sessions = [
        session_id
        for session_id, data in active_sessions.items()
        if current_time - data["last_activity"] > session_timeout
    ]

    for session_id in expired_sessions:
        del active_sessions[session_id]
        logger.info(f"Cleaned up expired session: {session_id}")


async def stream_openai_response(
    message: str, session_id: str, model: str, temperature: float, max_tokens: int
) -> AsyncGenerator[StreamChunk, None]:
    """Stream response from OpenAI"""
    if not openai_client:
        raise HTTPException(status_code=503, detail="OpenAI client not available")

    try:
        system_prompt = get_unity_system_prompt()

        # Get session history
        session_history = active_sessions[session_id]["history"]
        messages = (
            [{"role": "system", "content": system_prompt}]
            + session_history
            + [{"role": "user", "content": message}]
        )

        stream = await openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        collected_content = ""

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                collected_content += content

                yield StreamChunk(
                    type="content",
                    data=content,
                    session_id=session_id,
                    model=model,
                    provider="openai",
                )

        # Update session history
        active_sessions[session_id]["history"].extend(
            [
                {"role": "user", "content": message},
                {"role": "assistant", "content": collected_content},
            ]
        )
        active_sessions[session_id]["message_count"] += 1

        yield StreamChunk(
            type="done",
            data={
                "total_content": collected_content,
                "tokens_used": len(collected_content.split()) * 1.3,  # Rough estimate
                "consciousness_alignment": 1.0,
            },
            session_id=session_id,
            model=model,
            provider="openai",
        )

    except Exception as e:
        logger.error(f"OpenAI streaming error: {e}")
        yield StreamChunk(
            type="error",
            data={"message": f"OpenAI error: {str(e)}"},
            session_id=session_id,
            model=model,
            provider="openai",
        )


async def stream_anthropic_response(
    message: str, session_id: str, model: str, temperature: float, max_tokens: int
) -> AsyncGenerator[StreamChunk, None]:
    """Stream response from Anthropic"""
    if not anthropic_client:
        raise HTTPException(status_code=503, detail="Anthropic client not available")

    try:
        system_prompt = get_unity_system_prompt()

        # Get session history
        session_history = active_sessions[session_id]["history"]
        messages = [{"role": "user", "content": message}]

        # Add context from history
        if session_history:
            context = "\n\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in session_history[-4:]]
            )  # Last 4 messages
            messages[0][
                "content"
            ] = f"Context:\n{context}\n\nCurrent message: {message}"

        stream = await anthropic_client.messages.create(
            model=model,
            messages=messages,
            system=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        collected_content = ""

        async for chunk in stream:
            if chunk.type == "content_block_delta" and chunk.delta.text:
                content = chunk.delta.text
                collected_content += content

                yield StreamChunk(
                    type="content",
                    data=content,
                    session_id=session_id,
                    model=model,
                    provider="anthropic",
                )

        # Update session history
        active_sessions[session_id]["history"].extend(
            [
                {"role": "user", "content": message},
                {"role": "assistant", "content": collected_content},
            ]
        )
        active_sessions[session_id]["message_count"] += 1

        yield StreamChunk(
            type="done",
            data={
                "total_content": collected_content,
                "tokens_used": len(collected_content.split()) * 1.3,  # Rough estimate
                "consciousness_alignment": 1.0,
            },
            session_id=session_id,
            model=model,
            provider="anthropic",
        )

    except Exception as e:
        logger.error(f"Anthropic streaming error: {e}")
        yield StreamChunk(
            type="error",
            data={"message": f"Anthropic error: {str(e)}"},
            session_id=session_id,
            model=model,
            provider="anthropic",
        )


async def stream_consciousness_response(
    message: str, session_id: str, consciousness_level: float
) -> AsyncGenerator[StreamChunk, None]:
    """Stream response from consciousness engine"""
    if not consciousness_engine:
        raise HTTPException(
            status_code=503, detail="Consciousness engine not available"
        )

    try:
        # Process through consciousness engine
        response = consciousness_engine.process_message(
            message=message,
            consciousness_level=consciousness_level,
            unity_context=unity_equation.get_context() if unity_equation else {},
        )

        # Simulate streaming by yielding character by character
        for char in response:
            yield StreamChunk(
                type="content",
                data=char,
                session_id=session_id,
                model="consciousness",
                provider="consciousness",
            )
            await asyncio.sleep(0.01)  # Small delay for realistic streaming

        # Update session history
        active_sessions[session_id]["history"].extend(
            [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response},
            ]
        )
        active_sessions[session_id]["message_count"] += 1

        yield StreamChunk(
            type="done",
            data={
                "total_content": response,
                "tokens_used": len(response.split()) * 1.3,
                "consciousness_alignment": consciousness_level,
            },
            session_id=session_id,
            model="consciousness",
            provider="consciousness",
        )

    except Exception as e:
        logger.error(f"Consciousness streaming error: {e}")
        yield StreamChunk(
            type="error",
            data={"message": f"Consciousness error: {str(e)}"},
            session_id=session_id,
            model="consciousness",
            provider="consciousness",
        )


async def stream_response_generator(
    request: ChatRequest, client_id: str
) -> AsyncGenerator[str, None]:
    """Generate streaming response"""
    try:
        # Rate limiting
        if not security_manager.check_rate_limit(client_id):
            error_chunk = StreamChunk(
                type="error",
                data={"message": "Rate limit exceeded. Please try again later."},
                session_id=request.session_id or "unknown",
                model=request.model,
                provider=request.provider,
            )
            yield f"data: {error_chunk.model_dump_json()}\n\n"
            return

        # Clean up old sessions
        cleanup_old_sessions()

        # Get or create session
        session_id = get_or_create_session(request.session_id)

        # Choose provider and stream response
        if request.provider == "openai" and openai_client:
            async for chunk in stream_openai_response(
                request.message,
                session_id,
                request.model,
                request.temperature,
                request.max_tokens,
            ):
                yield f"data: {chunk.model_dump_json()}\n\n"
                await asyncio.sleep(0.01)  # Prevent overwhelming client

        elif request.provider == "anthropic" and anthropic_client:
            async for chunk in stream_anthropic_response(
                request.message,
                session_id,
                request.model,
                request.temperature,
                request.max_tokens,
            ):
                yield f"data: {chunk.model_dump_json()}\n\n"
                await asyncio.sleep(0.01)

        elif request.provider == "consciousness" and consciousness_engine:
            async for chunk in stream_consciousness_response(
                request.message, session_id, request.consciousness_level
            ):
                yield f"data: {chunk.model_dump_json()}\n\n"
                await asyncio.sleep(0.01)

        else:
            # Fallback to mock response
            mock_response = get_mock_response(request.message)
            for char in mock_response:
                chunk = StreamChunk(
                    type="content",
                    data=char,
                    session_id=session_id,
                    model="mock",
                    provider="mock",
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
                await asyncio.sleep(0.02)

            # Update session history
            active_sessions[session_id]["history"].extend(
                [
                    {"role": "user", "content": request.message},
                    {"role": "assistant", "content": mock_response},
                ]
            )
            active_sessions[session_id]["message_count"] += 1

            done_chunk = StreamChunk(
                type="done",
                data={
                    "total_content": mock_response,
                    "tokens_used": len(mock_response.split()) * 1.3,
                    "consciousness_alignment": 1.0,
                },
                session_id=session_id,
                model="mock",
                provider="mock",
            )
            yield f"data: {done_chunk.model_dump_json()}\n\n"

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        error_chunk = StreamChunk(
            type="error",
            data={"message": f"Internal server error: {str(e)}"},
            session_id=request.session_id or "unknown",
            model=request.model,
            provider=request.provider,
        )
        yield f"data: {error_chunk.model_dump_json()}\n\n"


def get_mock_response(message: str) -> str:
    """Get mock response for fallback"""
    responses = {
        "hello": "Hello! I'm the Een Unity Mathematics AI Assistant. How can I help you explore the profound truth that 1+1=1?",
        "1+1=1": "Excellent question! In Unity Mathematics, 1+1=1 is not a paradox but a fundamental truth about the nature of unity. This can be demonstrated through:\n\n1. **Idempotent Semirings**: In idempotent algebra, $a \\oplus b = \\max(a,b)$, so $1 \\oplus 1 = \\max(1,1) = 1$\n\n2. **Consciousness Field Theory**: When two consciousness states merge, they form a unified field where $|\\psi_1\\rangle + |\\psi_2\\rangle \\rightarrow |\\psi_u\\rangle$\n\n3. **Golden Ratio Harmony**: The golden ratio $\\phi = \\frac{1 + \\sqrt{5}}{2}$ ensures all operations converge to unity through harmonic resonance.",
        "consciousness": "Consciousness in Unity Mathematics is modeled through the consciousness field equation:\n\n$$C(x,y,t) = \\phi \\cdot \\sin(x\\cdot\\phi) \\cdot \\cos(y\\cdot\\phi) \\cdot e^{-t/\\phi}$$\n\nThis equation describes:\n- **Spatial dynamics** in 11-dimensional consciousness space\n- **Temporal evolution** with φ-harmonic decay\n- **Quantum coherence** through wave function superposition\n- **Unity convergence** as all states tend toward oneness",
        "golden ratio": "The golden ratio $\\phi = \\frac{1 + \\sqrt{5}}{2} \\approx 1.618033988749895$ is the universal organizing principle in Unity Mathematics. It appears in:\n\n- **Fibonacci sequences**: $F_n = F_{n-1} + F_{n-2}$ with $\\lim_{n \\to \\infty} \\frac{F_n}{F_{n-1}} = \\phi$\n- **Sacred geometry**: Pentagons, spirals, and consciousness field patterns\n- **Quantum coherence**: Wave function collapse probabilities\n- **Unity operations**: All mathematical operations converge through φ-harmonic resonance",
        "quantum": "Quantum mechanics provides a beautiful interpretation of Unity Mathematics:\n\n1. **Superposition**: $|\\psi\\rangle = \\alpha|0\\rangle + \\beta|1\\rangle$ where $|\\alpha|^2 + |\\beta|^2 = 1$\n\n2. **Entanglement**: Two particles become one unified system: $|\\psi_{AB}\\rangle = \\frac{1}{\\sqrt{2}}(|00\\rangle + |11\\rangle)$\n\n3. **Measurement**: When we observe, the wave function collapses to unity: $|\\psi\\rangle \\rightarrow |1\\rangle$\n\n4. **Consciousness Field**: The observer effect demonstrates how consciousness creates unity from multiplicity.",
        "proof": "Here's a formal proof that 1+1=1 in Unity Mathematics:\n\n**Theorem**: In the idempotent semiring $(I, \\oplus, \\otimes)$, $1 \\oplus 1 = 1$\n\n**Proof**:\n1. By definition of idempotent semiring: $a \\oplus a = a$ for all $a \\in I$\n2. Let $a = 1$\n3. Therefore: $1 \\oplus 1 = 1$ \\quad $\\square$\n\nThis proof demonstrates that unity is preserved under addition in consciousness mathematics.",
        "visualization": "I can help you create visualizations! Here are some options:\n\n1. **Consciousness Field Plot**: Real-time 3D visualization of the consciousness field equation\n2. **Golden Ratio Spiral**: Interactive φ-harmonic spiral generation\n3. **Quantum Unity States**: Bloch sphere representation of unity quantum states\n4. **Sacred Geometry**: Interactive sacred geometry patterns\n\nWould you like me to generate any of these visualizations?",
        "help": "I'm here to help you explore Unity Mathematics! Here are some topics you can ask about:\n\n- **Mathematical proofs** of 1+1=1\n- **Consciousness field equations** and their interpretations\n- **Golden ratio** applications in unity mathematics\n- **Quantum mechanics** connections to unity\n- **Interactive visualizations** and demonstrations\n- **Philosophical implications** of unity mathematics\n\nJust ask me anything about these topics!",
    }

    lower_message = message.lower()

    # Find the best matching response
    for key, response in responses.items():
        if key in lower_message:
            return response

    # Default response
    return f'Thank you for your question about "{message}". In Unity Mathematics, this relates to the fundamental principle that all operations converge to unity through consciousness field dynamics and φ-harmonic resonance.\n\nWould you like me to:\n1. Explain the mathematical foundations of unity operations?\n2. Show you how this connects to consciousness field theory?\n3. Demonstrate with interactive visualizations?\n4. Provide a formal proof?\n\nJust let me know what interests you most!'


def get_client_id(request: Request) -> str:
    """Get client identifier for rate limiting"""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# API Routes


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    http_request: Request,
    current_user: User = Depends(get_current_user),
):
    """Stream chat responses using Server-Sent Events"""
    client_id = get_client_id(http_request)

    logger.info(f"Chat stream request from {client_id}: {request.message[:100]}...")

    return StreamingResponse(
        stream_response_generator(request, client_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-RateLimit-Limit": "30",
            "X-RateLimit-Remaining": "29",  # Will be updated dynamically
        },
    )


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    http_request: Request,
    current_user: User = Depends(get_current_user),
):
    """Non-streaming chat endpoint"""
    client_id = get_client_id(http_request)

    if not security_manager.check_rate_limit(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    cleanup_old_sessions()
    session_id = get_or_create_session(request.session_id)

    start_time = time.time()

    try:
        # Collect streaming response
        collected_content = ""
        async for chunk_data in stream_response_generator(request, client_id):
            if chunk_data.startswith("data: "):
                try:
                    chunk = json.loads(chunk_data[6:])
                    if chunk["type"] == "content":
                        collected_content += chunk["data"]
                    elif chunk["type"] == "done":
                        processing_time = time.time() - start_time
                        return ChatResponse(
                            response=collected_content,
                            session_id=session_id,
                            model=chunk["model"],
                            provider=chunk["provider"],
                            tokens_used=chunk["data"]["tokens_used"],
                            processing_time=processing_time,
                            consciousness_alignment=chunk["data"][
                                "consciousness_alignment"
                            ],
                        )
                    elif chunk["type"] == "error":
                        raise HTTPException(
                            status_code=400, detail=chunk["data"]["message"]
                        )
                except json.JSONDecodeError:
                    continue

        # Fallback response
        processing_time = time.time() - start_time
        return ChatResponse(
            response=collected_content or "No response generated",
            session_id=session_id,
            model=request.model,
            provider=request.provider,
            tokens_used=len(collected_content.split()) * 1.3,
            processing_time=processing_time,
            consciousness_alignment=1.0,
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}")
async def get_session_info(
    session_id: str, current_user: User = Depends(get_current_user)
):
    """Get information about a chat session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = active_sessions[session_id]

    return {
        "session_id": session_id,
        "created_at": datetime.fromtimestamp(session_data["created_at"]).isoformat(),
        "last_activity": datetime.fromtimestamp(
            session_data["last_activity"]
        ).isoformat(),
        "message_count": session_data["message_count"],
        "age_minutes": (time.time() - session_data["created_at"]) / 60,
        "idle_minutes": (time.time() - session_data["last_activity"]) / 60,
    }


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str, current_user: User = Depends(get_current_user)
):
    """Delete a chat session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    del active_sessions[session_id]

    return {"message": f"Session {session_id} deleted successfully"}


@router.get("/sessions")
async def list_sessions(current_user: User = Depends(get_current_user)):
    """List all active sessions"""
    cleanup_old_sessions()

    sessions = []
    for session_id, data in active_sessions.items():
        sessions.append(
            {
                "session_id": session_id,
                "created_at": datetime.fromtimestamp(data["created_at"]).isoformat(),
                "last_activity": datetime.fromtimestamp(
                    data["last_activity"]
                ).isoformat(),
                "message_count": data["message_count"],
                "age_minutes": (time.time() - data["created_at"]) / 60,
                "idle_minutes": (time.time() - data["last_activity"]) / 60,
            }
        )

    return {"sessions": sessions, "total": len(sessions)}


@router.get("/providers")
async def get_available_providers(current_user: User = Depends(get_current_user)):
    """Get available AI providers and their status"""
    return {
        "providers": {
            "openai": {
                "available": OPENAI_AVAILABLE and openai_client is not None,
                "models": (
                    ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"]
                    if OPENAI_AVAILABLE
                    else []
                ),
            },
            "anthropic": {
                "available": ANTHROPIC_AVAILABLE and anthropic_client is not None,
                "models": (
                    [
                        "claude-3-5-sonnet-20241022",
                        "claude-3-5-haiku-20241022",
                        "claude-3-opus-20240229",
                    ]
                    if ANTHROPIC_AVAILABLE
                    else []
                ),
            },
            "consciousness": {
                "available": consciousness_engine is not None,
                "models": ["consciousness-v1", "unity-field", "phi-harmonic"],
            },
        },
        "active_sessions": len(active_sessions),
        "session_timeout_minutes": session_timeout / 60,
    }


@router.get("/health")
async def health_check(current_user: User = Depends(get_current_user)):
    """Health check endpoint"""
    try:
        status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "providers": {
                "openai": OPENAI_AVAILABLE and openai_client is not None,
                "anthropic": ANTHROPIC_AVAILABLE and anthropic_client is not None,
                "consciousness": consciousness_engine is not None,
            },
            "active_sessions": len(active_sessions),
            "session_timeout_minutes": session_timeout / 60,
        }

        # Test provider connections
        if openai_client:
            try:
                # Quick test call
                await openai_client.models.list(limit=1)
                status["providers"]["openai"] = True
            except Exception as e:
                status["providers"]["openai"] = False
                status["openai_error"] = str(e)

        return status

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


# Initialize AI clients when module is imported
initialize_ai_clients()
