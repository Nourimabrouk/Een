#!/usr/bin/env python3
"""
Een Repository AI Chat API
==========================

FastAPI backend providing streaming chat interface for the Een repository AI assistant.
Integrates with OpenAI Assistants API and Vector Store for RAG-powered responses.

Features:
- Server-Sent Events (SSE) streaming responses
- Session-based chat threads
- Rate limiting and cost monitoring
- Bearer token authentication (optional)
- Integration with existing Een API infrastructure

Author: Claude (3000 ELO AGI)
"""

import os
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, AsyncGenerator, Any
from datetime import datetime, timedelta
import asyncio
import hmac

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import openai
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv

# Import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not available - using word-based token estimation")

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    """Chat request model."""

    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    session_id: Optional[str] = Field(
        None, description="Optional session ID for conversation continuity"
    )


class ChatResponse(BaseModel):
    """Chat response model."""

    response: str
    session_id: str
    tokens_used: int
    processing_time: float
    sources: List[Dict[str, Any]] = []


class StreamChunk(BaseModel):
    """Streaming response chunk."""

    type: str  # "content", "sources", "done", "error"
    data: Any
    session_id: str


class TokenCounter:
    """Handles accurate token counting using tiktoken."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fallback to cl100k_base encoding for unknown models
                self.encoding = tiktoken.get_encoding("cl100k_base")
                logger.warning(f"Unknown model {model}, using cl100k_base encoding")
        else:
            self.encoding = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.encoding and text:
            return len(self.encoding.encode(text))
        else:
            # Fallback: estimate based on word count
            return max(1, int(len(text.split()) * 1.3))
    
    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count tokens in a list of messages."""
        if not self.encoding:
            # Fallback estimation
            total_words = sum(len(msg.get("content", "").split()) for msg in messages)
            return max(1, int(total_words * 1.3 + len(messages) * 4))
        
        total_tokens = 0
        for message in messages:
            # Add tokens for message formatting
            total_tokens += 4  # Every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                total_tokens += len(self.encoding.encode(str(value)))
                
        total_tokens += 2  # Every reply is primed with <im_start>assistant
        return total_tokens


class EenChatAPI:
    """Main chat API class."""

    def __init__(self):
        self.openai_client = openai.OpenAI()
        self.active_sessions: Dict[str, Dict] = {}
        self.rate_limits: Dict[str, List[float]] = {}

        # Configuration
        self.assistant_id = self._get_assistant_id()
        self.chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
        self.max_tokens = int(os.getenv("MAX_CHAT_TOKENS", "1000"))
        self.chat_bearer_token = os.getenv("CHAT_BEARER_TOKEN")

        # Rate limiting (requests per minute)
        self.rate_limit = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
        
        # Token counter
        self.token_counter = TokenCounter(self.chat_model)
        
        # Usage tracking
        self.usage_stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_errors": 0,
            "start_time": time.time()
        }

        logger.info(f"Een Chat API initialized with assistant: {self.assistant_id}")
        logger.info(f"Token counting: {'tiktoken' if TIKTOKEN_AVAILABLE else 'word estimation'}")

    def _get_assistant_id(self) -> str:
        """Get the OpenAI Assistant ID."""
        from . import get_assistant_id

        assistant_id = get_assistant_id()
        if not assistant_id:
            raise ValueError("No OpenAI Assistant found. Run prepare_index.py first.")

        # Verify assistant exists
        try:
            self.openai_client.beta.assistants.retrieve(assistant_id)
            return assistant_id
        except Exception as e:
            raise ValueError(f"Assistant {assistant_id} not accessible: {e}")

    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limits."""
        now = time.time()

        if client_id not in self.rate_limits:
            self.rate_limits[client_id] = []

        # Clean old requests (older than 1 minute)
        self.rate_limits[client_id] = [
            req_time for req_time in self.rate_limits[client_id] if now - req_time < 60
        ]

        # Check limit
        if len(self.rate_limits[client_id]) >= self.rate_limit:
            return False

        # Add current request
        self.rate_limits[client_id].append(now)
        return True

    def _get_or_create_thread(self, session_id: Optional[str]) -> tuple[str, str]:
        """Get existing thread or create new one."""
        if session_id and session_id in self.active_sessions:
            thread_id = self.active_sessions[session_id]["thread_id"]
            return session_id, thread_id

        # Create new thread
        thread = self.openai_client.beta.threads.create()
        new_session_id = session_id or str(uuid.uuid4())

        self.active_sessions[new_session_id] = {
            "thread_id": thread.id,
            "created_at": time.time(),
            "message_count": 0,
        }

        return new_session_id, thread.id

    def _cleanup_old_sessions(self) -> None:
        """Remove sessions older than 24 hours."""
        cutoff_time = time.time() - (24 * 60 * 60)

        old_sessions = [
            session_id
            for session_id, data in self.active_sessions.items()
            if data["created_at"] < cutoff_time
        ]

        for session_id in old_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Cleaned up old session: {session_id}")

    async def _stream_assistant_response(
        self, thread_id: str, session_id: str, message: str
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream response from OpenAI Assistant."""
        start_time = time.time()

        try:
            # Add user message to thread
            self.openai_client.beta.threads.messages.create(
                thread_id=thread_id, role="user", content=message
            )

            # Start streaming run
            with self.openai_client.beta.threads.runs.stream(
                thread_id=thread_id,
                assistant_id=self.assistant_id,
                max_prompt_tokens=self.max_tokens,
                temperature=0.2,
            ) as stream:

                collected_content = ""

                for event in stream:
                    if event.event == "thread.message.delta":
                        if event.data.delta.content:
                            for content_delta in event.data.delta.content:
                                if (
                                    hasattr(content_delta, "text")
                                    and content_delta.text
                                ):
                                    text_delta = content_delta.text.value
                                    collected_content += text_delta

                                    yield StreamChunk(
                                        type="content",
                                        data=text_delta,
                                        session_id=session_id,
                                    )

                    elif event.event == "thread.run.completed":
                        # Get sources from file search annotations
                        messages = self.openai_client.beta.threads.messages.list(
                            thread_id=thread_id, limit=1
                        )

                        sources = []
                        if messages.data:
                            message_obj = messages.data[0]
                            for content in message_obj.content:
                                if hasattr(content, "text") and content.text:
                                    for annotation in content.text.annotations:
                                        if hasattr(annotation, "file_citation"):
                                            sources.append(
                                                {
                                                    "type": "file_citation",
                                                    "text": annotation.text,
                                                    "file_id": annotation.file_citation.file_id,
                                                }
                                            )

                        if sources:
                            yield StreamChunk(
                                type="sources", data=sources, session_id=session_id
                            )

                        # Calculate accurate token usage
                        input_tokens = self.token_counter.count_tokens(message)
                        output_tokens = self.token_counter.count_tokens(collected_content)
                        total_tokens = input_tokens + output_tokens
                        
                        # Update usage statistics
                        self.usage_stats["total_tokens"] += total_tokens
                        self.usage_stats["total_requests"] += 1
                        
                        # Final completion data
                        processing_time = time.time() - start_time
                        yield StreamChunk(
                            type="done",
                            data={
                                "total_content": collected_content,
                                "processing_time": processing_time,
                                "tokens_used": total_tokens,
                                "input_tokens": input_tokens,
                                "output_tokens": output_tokens,
                                "tokens_per_second": output_tokens / max(processing_time, 0.001),
                                "method": "tiktoken" if TIKTOKEN_AVAILABLE else "estimation"
                            },
                            session_id=session_id,
                        )
                        break

                    elif event.event == "thread.run.failed":
                        self.usage_stats["total_errors"] += 1
                        error_data = {
                            "message": "Assistant run failed",
                            "details": str(event.data),
                            "error_type": "assistant_run_failed",
                            "timestamp": time.time()
                        }
                        logger.error(f"Assistant run failed: {error_data}")
                        yield StreamChunk(
                            type="error",
                            data=error_data,
                            session_id=session_id,
                        )
                        break

        except openai.RateLimitError as e:
            self.usage_stats["total_errors"] += 1
            error_data = {
                "message": "OpenAI rate limit exceeded. Please try again later.",
                "error_type": "rate_limit_error",
                "retry_after": getattr(e, 'retry_after', 60),
                "timestamp": time.time()
            }
            logger.warning(f"Rate limit error: {e}")
            yield StreamChunk(type="error", data=error_data, session_id=session_id)
            
        except openai.AuthenticationError as e:
            self.usage_stats["total_errors"] += 1
            error_data = {
                "message": "OpenAI authentication failed. Check API key configuration.",
                "error_type": "authentication_error",
                "timestamp": time.time()
            }
            logger.error(f"Authentication error: {e}")
            yield StreamChunk(type="error", data=error_data, session_id=session_id)
            
        except openai.PermissionDeniedError as e:
            self.usage_stats["total_errors"] += 1
            error_data = {
                "message": "Permission denied. Check API key permissions.",
                "error_type": "permission_error",
                "timestamp": time.time()
            }
            logger.error(f"Permission error: {e}")
            yield StreamChunk(type="error", data=error_data, session_id=session_id)
            
        except openai.BadRequestError as e:
            self.usage_stats["total_errors"] += 1
            error_data = {
                "message": "Invalid request format or parameters.",
                "error_type": "bad_request_error",
                "details": str(e),
                "timestamp": time.time()
            }
            logger.error(f"Bad request error: {e}")
            yield StreamChunk(type="error", data=error_data, session_id=session_id)
            
        except openai.APITimeoutError as e:
            self.usage_stats["total_errors"] += 1
            error_data = {
                "message": "OpenAI API request timed out. Please try again.",
                "error_type": "timeout_error",
                "timestamp": time.time()
            }
            logger.warning(f"API timeout error: {e}")
            yield StreamChunk(type="error", data=error_data, session_id=session_id)
            
        except openai.APIConnectionError as e:
            self.usage_stats["total_errors"] += 1
            error_data = {
                "message": "Failed to connect to OpenAI API. Please try again.",
                "error_type": "connection_error",
                "timestamp": time.time()
            }
            logger.error(f"API connection error: {e}")
            yield StreamChunk(type="error", data=error_data, session_id=session_id)
            
        except Exception as e:
            self.usage_stats["total_errors"] += 1
            error_data = {
                "message": f"Unexpected error: {str(e)}",
                "error_type": "unknown_error",
                "timestamp": time.time()
            }
            logger.error(f"Unexpected error in assistant response: {e}")
            yield StreamChunk(type="error", data=error_data, session_id=session_id)

    async def process_chat_stream(
        self, request: ChatRequest, client_id: str
    ) -> AsyncGenerator[str, None]:
        """Process chat request and yield SSE-formatted responses."""
        try:
            # Rate limiting
            if not self._check_rate_limit(client_id):
                error_chunk = StreamChunk(
                    type="error",
                    data={"message": "Rate limit exceeded. Please try again later."},
                    session_id=request.session_id or "unknown",
                )
                yield f"data: {error_chunk.model_dump_json()}\n\n"
                return

            # Clean up old sessions
            self._cleanup_old_sessions()

            # Get or create thread
            session_id, thread_id = self._get_or_create_thread(request.session_id)

            # Update session
            self.active_sessions[session_id]["message_count"] += 1

            # Stream response
            async for chunk in self._stream_assistant_response(
                thread_id, session_id, request.message
            ):
                yield f"data: {chunk.model_dump_json()}\n\n"

                # Add small delay to prevent overwhelming client
                await asyncio.sleep(0.01)

        except Exception as e:
            logger.error(f"Error in chat stream: {e}")
            error_chunk = StreamChunk(
                type="error",
                data={"message": f"Internal server error: {str(e)}"},
                session_id=request.session_id or "unknown",
            )
            yield f"data: {error_chunk.model_dump_json()}\n\n"


# Initialize API
chat_api = EenChatAPI()

# FastAPI app
app = FastAPI(
    title="Een Repository AI Chat API",
    description="RAG-powered chatbot for exploring Unity Mathematics and φ-Harmonic Consciousness",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def verify_bearer_token(authorization: Optional[str] = Header(None)) -> bool:
    """
    Verify bearer token authentication.
    
    Supports both API_KEY and CHAT_BEARER_TOKEN for backward compatibility.
    Uses constant-time comparison to prevent timing attacks.
    """
    import hmac
    
    # Get API keys from environment
    api_key = os.getenv("API_KEY")
    chat_bearer_token = os.getenv("CHAT_BEARER_TOKEN")
    
    # If no authentication is configured, allow all requests
    if not api_key and not chat_bearer_token:
        logger.warning("No API key or bearer token configured - authentication disabled")
        return True

    # Check for authorization header
    if not authorization:
        logger.warning("Missing Authorization header")
        raise HTTPException(
            status_code=401, 
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Validate Bearer format
    if not authorization.startswith("Bearer "):
        logger.warning("Invalid authorization format")
        raise HTTPException(
            status_code=401, 
            detail="Invalid authorization format. Use: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"}
        )

    token = authorization[7:]  # Remove "Bearer " prefix
    
    # Validate token is not empty
    if not token.strip():
        logger.warning("Empty bearer token")
        raise HTTPException(
            status_code=401, 
            detail="Bearer token cannot be empty",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Check against API_KEY first (primary authentication)
    if api_key and hmac.compare_digest(token, api_key):
        return True
    
    # Check against CHAT_BEARER_TOKEN (legacy authentication)
    if chat_bearer_token and hmac.compare_digest(token, chat_bearer_token):
        return True

    # Authentication failed
    logger.warning(f"Invalid bearer token authentication attempt")
    raise HTTPException(
        status_code=401, 
        detail="Invalid bearer token",
        headers={"WWW-Authenticate": "Bearer"}
    )


def get_client_id(request: Request) -> str:
    """Get client identifier for rate limiting."""
    # Use X-Forwarded-For if available, otherwise client IP
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Een Repository AI Chat API",
        "version": "1.0.0",
        "status": "operational",
        "assistant_id": chat_api.assistant_id,
        "endpoints": {"chat": "/chat", "health": "/health", "docs": "/api/docs"},
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test OpenAI connection
        chat_api.openai_client.beta.assistants.retrieve(chat_api.assistant_id)

        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "assistant_id": chat_api.assistant_id,
            "active_sessions": len(chat_api.active_sessions),
            "rate_limit_clients": len(chat_api.rate_limits),
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/chat")
async def chat_stream(
    request: ChatRequest, http_request: Request, _: bool = Depends(verify_bearer_token)
):
    """Stream chat responses using Server-Sent Events."""
    client_id = get_client_id(http_request)

    logger.info(f"Chat request from {client_id}: {request.message[:100]}...")

    return EventSourceResponse(
        chat_api.process_chat_stream(request, client_id),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-RateLimit-Limit": str(chat_api.rate_limit),
            "X-RateLimit-Remaining": str(
                max(
                    0,
                    chat_api.rate_limit - len(chat_api.rate_limits.get(client_id, [])),
                )
            ),
        },
    )


@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str, _: bool = Depends(verify_bearer_token)):
    """Get information about a chat session."""
    if session_id not in chat_api.active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = chat_api.active_sessions[session_id]

    return {
        "session_id": session_id,
        "thread_id": session_data["thread_id"],
        "created_at": datetime.fromtimestamp(session_data["created_at"]).isoformat(),
        "message_count": session_data["message_count"],
        "age_hours": (time.time() - session_data["created_at"]) / 3600,
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, _: bool = Depends(verify_bearer_token)):
    """Delete a chat session."""
    if session_id not in chat_api.active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    del chat_api.active_sessions[session_id]

    return {"message": f"Session {session_id} deleted successfully"}


@app.get("/stats")
async def get_api_stats(_: bool = Depends(verify_bearer_token)):
    """Get comprehensive API usage statistics."""
    uptime = time.time() - chat_api.usage_stats["start_time"]
    
    return {
        # Session statistics
        "active_sessions": len(chat_api.active_sessions),
        "rate_limited_clients": len(chat_api.rate_limits),
        "total_requests_last_minute": sum(
            len(reqs) for reqs in chat_api.rate_limits.values()
        ),
        
        # Usage statistics
        "total_requests": chat_api.usage_stats["total_requests"],
        "total_tokens": chat_api.usage_stats["total_tokens"],
        "total_errors": chat_api.usage_stats["total_errors"],
        "uptime_seconds": uptime,
        "uptime_hours": uptime / 3600,
        
        # Performance metrics
        "requests_per_hour": chat_api.usage_stats["total_requests"] / max(uptime / 3600, 0.001),
        "tokens_per_hour": chat_api.usage_stats["total_tokens"] / max(uptime / 3600, 0.001),
        "error_rate": chat_api.usage_stats["total_errors"] / max(chat_api.usage_stats["total_requests"], 1),
        
        # Configuration
        "assistant_id": chat_api.assistant_id,
        "model": chat_api.chat_model,
        "max_tokens": chat_api.max_tokens,
        "rate_limit_per_minute": chat_api.rate_limit,
        "token_counting_method": "tiktoken" if TIKTOKEN_AVAILABLE else "estimation",
        
        # Timestamp
        "timestamp": datetime.utcnow().isoformat()
    }


# Store startup time for uptime calculation
chat_api.active_sessions["__start_time"] = time.time()

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"Starting Een Chat API on {host}:{port}")

    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=os.getenv("ENVIRONMENT") == "development",
        log_level="info",
    )
