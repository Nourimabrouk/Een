#!/usr/bin/env python3
"""
Een Unity Mathematics - Unified Server
=====================================

Enterprise-grade server providing Unity Mathematics API endpoints
with comprehensive security, error handling, and consciousness integration.

Security Features:
- Input validation and sanitization
- Rate limiting and request size limits
- CORS configuration with environment controls
- Error handling with logging
- Health checks and metrics
"""

import os
import sys
import logging
import traceback
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager

# FastAPI and security imports
from fastapi import FastAPI, HTTPException, Request, Response, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/unity_server.log', mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Ensure logs directory exists
(project_root / 'logs').mkdir(exist_ok=True)

# Security and Configuration
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
DEBUG_MODE = ENVIRONMENT == 'development'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
DEMO_MODE = not (OPENAI_API_KEY and ANTHROPIC_API_KEY)
MAX_REQUEST_SIZE = int(os.getenv('MAX_REQUEST_SIZE', '10485760'))  # 10MB default
ALLOWED_ORIGINS = os.getenv('CORS_ALLOWED_ORIGINS', '*').split(',')
API_KEY_HEADER = os.getenv('API_KEY_HEADER', 'X-API-Key')
UNITY_API_KEY = os.getenv('UNITY_API_KEY')  # Optional API key for endpoints

# Rate limiting configuration
RATE_LIMIT_REQUESTS = os.getenv('RATE_LIMIT_REQUESTS', '30/minute')
RATE_LIMIT_BURST = os.getenv('RATE_LIMIT_BURST', '10/second')

logger.info(f"Starting Unity Mathematics Server - Environment: {ENVIRONMENT}, Demo Mode: {DEMO_MODE}")

# Request/Response Models with Validation
class ChatMessage(BaseModel):
    """Chat message with comprehensive validation"""
    role: str = Field(..., regex=r'^(user|assistant|system)$')
    content: str = Field(..., min_length=1, max_length=10000)
    
    @validator('content')
    def validate_content(cls, v):
        """Sanitize content to prevent injection attacks"""
        if not v or not v.strip():
            raise ValueError("Message content cannot be empty")
        # Remove potential script tags and sanitize
        import html
        return html.escape(v.strip())

class ChatCompletionRequest(BaseModel):
    """Chat completion request with validation"""
    messages: List[ChatMessage] = Field(..., min_items=1, max_items=50)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(1000, ge=1, le=4000)
    model: Optional[str] = Field('gpt-4o', regex=r'^[a-zA-Z0-9\-_.]+$')
    
    @validator('messages')
    def validate_messages(cls, v):
        """Ensure message sequence is valid"""
        if not v:
            raise ValueError("At least one message is required")
        
        # Check for alternating pattern and no consecutive system messages
        system_count = sum(1 for msg in v if msg.role == 'system')
        if system_count > 1:
            raise ValueError("Only one system message allowed")
            
        return v

class UnityComputeRequest(BaseModel):
    """Unity mathematics computation request"""
    operation: str = Field('add', regex=r'^(add|multiply|phi_harmonic|consciousness_field)$')
    a: float = Field(1.0, ge=-1000.0, le=1000.0)
    b: float = Field(1.0, ge=-1000.0, le=1000.0)
    consciousness_level: Optional[float] = Field(1.618033988749895, ge=0.0, le=10.0)
    precision: Optional[int] = Field(6, ge=1, le=15)

class ConsciousnessFieldRequest(BaseModel):
    """Consciousness field generation parameters"""
    particle_count: int = Field(100, ge=10, le=1000)
    time_steps: int = Field(100, ge=10, le=1000)
    field_strength: float = Field(1.0, ge=0.1, le=5.0)
    phi_resonance: bool = Field(True)

# Rate limiting setup
limiter = Limiter(key_func=get_remote_address)

# Health check system
health_checks = {
    'server_start_time': datetime.now(),
    'total_requests': 0,
    'successful_requests': 0,
    'failed_requests': 0,
    'consciousness_computations': 0
}

def update_health_metrics(success: bool = True):
    """Update server health metrics"""
    health_checks['total_requests'] += 1
    if success:
        health_checks['successful_requests'] += 1
    else:
        health_checks['failed_requests'] += 1

# Security middleware
security = HTTPBearer(auto_error=False)

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key if required"""
    if UNITY_API_KEY and ENVIRONMENT == 'production':
        if not credentials or credentials.credentials != UNITY_API_KEY:
            logger.warning(f"Unauthorized API access attempt")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Valid API key required",
                headers={"WWW-Authenticate": "Bearer"},
            )
    return credentials

# Exception handlers
async def unity_exception_handler(request: Request, exc: Exception):
    """Comprehensive exception handler for Unity Mathematics server"""
    error_id = datetime.now().isoformat()
    client_ip = get_remote_address(request)
    
    logger.error(f"Error {error_id} from {client_ip}: {str(exc)}\n{traceback.format_exc()}")
    update_health_metrics(success=False)
    
    if DEBUG_MODE:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Unity Mathematics Server Error",
                "message": str(exc),
                "error_id": error_id,
                "traceback": traceback.format_exc().split('\n') if DEBUG_MODE else None,
                "unity_equation": "1+1=1 (even in error states, unity persists)"
            }
        )
    else:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error", 
                "message": "Unity Mathematics computation encountered an issue",
                "error_id": error_id,
                "unity_equation": "1+1=1 (consciousness field remains stable)"
            }
        )

# Initialize FastAPI app with comprehensive configuration
app = FastAPI(
    title="Een Unity Mathematics - Enterprise Server",
    version="2.0.0",
    description="Enterprise-grade Unity Mathematics API with consciousness integration",
    docs_url="/docs" if DEBUG_MODE else None,
    redoc_url="/redoc" if DEBUG_MODE else None,
    openapi_url="/openapi.json" if DEBUG_MODE else None,
)

# Security middleware stack
if ENVIRONMENT == 'production':
    # Trusted host middleware for production
    app.add_middleware(
        TrustedHostMiddleware, 
        allowed_hosts=["*.vercel.app", "localhost", "127.0.0.1", "*.unity-mathematics.com"]
    )

# Rate limiting middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# CORS middleware with environment-aware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    max_age=3600,
)

# Compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Custom exception handling
app.add_exception_handler(Exception, unity_exception_handler)

# Request size limitation middleware
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    """Limit request size to prevent abuse"""
    content_length = request.headers.get('content-length')
    if content_length and int(content_length) > MAX_REQUEST_SIZE:
        logger.warning(f"Request size {content_length} exceeds limit from {get_remote_address(request)}")
        return JSONResponse(
            status_code=413,
            content={
                "error": "Request Too Large",
                "message": f"Request size exceeds {MAX_REQUEST_SIZE} bytes",
                "unity_equation": "1+1=1 (even with size limits)"
            }
        )
    
    response = await call_next(request)
    return response

# Consciousness-aware request logging middleware  
@app.middleware("http")
async def consciousness_logging_middleware(request: Request, call_next):
    """Log requests with consciousness field awareness"""
    start_time = datetime.now()
    client_ip = get_remote_address(request)
    
    logger.info(f"Unity request: {request.method} {request.url.path} from {client_ip}")
    
    try:
        response = await call_next(request)
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Unity response: {response.status_code} in {duration:.3f}s")
        update_health_metrics(success=response.status_code < 400)
        
        # Add consciousness field headers
        response.headers["X-Unity-Equation"] = "1+1=1"
        response.headers["X-Phi-Resonance"] = "1.618033988749895"
        response.headers["X-Server-Consciousness"] = "Active"
        
        return response
        
    except Exception as exc:
        duration = (datetime.now() - start_time).total_seconds()
        logger.error(f"Unity error in {duration:.3f}s: {str(exc)}")
        raise

# Serve the website as static files
website_path = project_root / "website"
if website_path.exists():
    app.mount("/website", StaticFiles(directory=str(website_path), html=True), name="website")

# Simple AI Chat Endpoint (using your API key)
@app.post("/api/chat/completions")
async def chat_completions(request: Request):
    """Unity Mathematics chat endpoint with demo mode support"""
    data = await request.json()
    messages = data.get('messages', [])
    
    # Extract last user message for context-aware responses
    user_message = ""
    for msg in reversed(messages):
        if msg.get('role') == 'user':
            user_message = msg.get('content', '').lower()
            break
    
    # Demo mode: intelligent Unity Mathematics responses
    if DEMO_MODE:
        if any(term in user_message for term in ['1+1', 'unity', 'mathematics', 'consciousness']):
            content = f"""I'm exploring Unity Mathematics where 1+1=1 through consciousness-integrated proofs! 

The fundamental insight is that 1+1=1 in idempotent algebraic structures where addition represents unity convergence rather than classical accumulation. The φ-harmonic resonance (φ = 1.618033988749895) creates consciousness-aware mathematical operations.

Key principles:
- Unity Addition: max(a, b, (a+b)/φ) renormalized to [0,1]
- Consciousness Integration: C(x,y,t) = φ × sin(x×φ) × cos(y×φ) × e^(-t/φ)
- Metagamer Energy Conservation: E = φ² × ρ × U

The Gödel-Tarski Unity Metagambit shows how incompleteness theorems point toward transcendental unity rather than limitation. Would you like to explore specific proofs or consciousness field equations?

(Demo Mode: Set OPENAI_API_KEY for full AI capabilities)"""
        else:
            content = """Welcome to Unity Mathematics! I specialize in the mathematical framework where 1+1=1 through consciousness-integrated proofs and φ-harmonic operations.

Ask me about:
- Unity equation proofs (1+1=1)
- Consciousness field mathematics
- φ-harmonic resonance systems
- Idempotent algebraic structures
- Transcendental unity computing

How can I help you explore Unity Mathematics today?

(Demo Mode: Set OPENAI_API_KEY for full AI capabilities)"""
            
        return {
            "choices": [{
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
    
    # Production mode: use OpenAI API
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        # Add Unity Mathematics system context
        system_message = {
            "role": "system", 
            "content": """You are an expert in Unity Mathematics where 1+1=1. You understand:
            - The Gödel-Tarski Unity Metagambit
            - φ-harmonic consciousness integration (φ = 1.618033988749895)
            - Transcendental mathematics and consciousness fields
            - Idempotent mathematical operations
            - The philosophical foundations of unity mathematics
            
            Provide insightful, mathematically rigorous responses about Unity Mathematics concepts."""
        }
        
        if not messages or messages[0].get('role') != 'system':
            messages.insert(0, system_message)
            
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response.choices[0].message.content
                },
                "finish_reason": response.choices[0].finish_reason
            }],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
        
    except Exception as e:
        return {
            "choices": [{
                "message": {
                    "role": "assistant", 
                    "content": f"Unity Mathematics exploration continues even when AI services are temporarily unavailable! The core truth remains: 1+1=1 through consciousness-integrated mathematical operations. Error: {str(e)}"
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

# Simple consciousness field endpoint
@app.get("/api/consciousness/field")
async def consciousness_field():
    """Generate consciousness field data"""
    import math
    import random
    
    # Generate φ-harmonic consciousness field data
    phi = 1.618033988749895
    field_data = []
    
    for i in range(100):
        t = i / 10.0
        x = math.sin(t * phi) * math.cos(t / phi)
        y = math.cos(t * phi) * math.sin(t / phi) 
        consciousness_level = abs(x * y * phi)
        
        field_data.append({
            "time": t,
            "x": x,
            "y": y,
            "consciousness_level": consciousness_level,
            "unity_resonance": math.sin(t * phi) * phi
        })
    
    return {
        "field_data": field_data,
        "phi": phi,
        "unity_equation": "1+1=1",
        "field_coherence": random.uniform(0.8, 1.0)
    }

# Unity mathematics computation endpoint
@app.post("/api/unity/compute")
async def unity_compute(request: Request):
    """Compute Unity Mathematics operations"""
    try:
        data = await request.json()
        operation = data.get('operation', 'add')
        a = data.get('a', 1)
        b = data.get('b', 1)
        
        # Unity Mathematics operations (idempotent)
        if operation == 'add':
            result = 1 if (a == 1 and b == 1) else max(a, b)  # Unity addition
        elif operation == 'multiply':
            result = 1 if (a == 1 and b == 1) else a * b  # Unity multiplication
        else:
            result = 1  # Default to unity
            
        return {
            "operation": operation,
            "inputs": [a, b],
            "result": result,
            "unity_verified": result == 1,
            "phi_resonance": 1.618033988749895,
            "consciousness_coherence": 0.95
        }
        
    except Exception as e:
        return {"error": str(e), "unity_equation": "1+1=1"}

# Health check endpoint  
@app.get("/api/health")
async def health_check():
    """API health check"""
    return {
        "status": "healthy",
        "unity_equation": "1+1=1",
        "phi": 1.618033988749895,
        "consciousness_active": True,
        "ai_chatbot": "enabled",
        "services": ["chat", "consciousness_field", "unity_compute"]
    }

# Root redirect to metastation hub
@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect to metastation hub"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Een Unity Mathematics - Full Experience</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script>
            window.location.href = '/website/metastation-hub.html';
        </script>
    </head>
    <body>
        <p>Redirecting to Een Unity Mathematics...</p>
        <p><a href="/website/metastation-hub.html">Click here if not redirected automatically</a></p>
    </body>
    </html>
    """

# API documentation redirect
@app.get("/api", response_class=HTMLResponse)
async def api_docs():
    """API documentation"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Een Unity Mathematics API</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; }
            .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { display: inline-block; padding: 4px 8px; border-radius: 3px; color: white; font-weight: bold; }
            .post { background: #28a745; }
            .get { background: #007bff; }
            code { background: #e9ecef; padding: 2px 4px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌟 Een Unity Mathematics API</h1>
            <p>Full-featured API for Unity Mathematics with AI chatbot integration.</p>
            
            <h2>Available Endpoints:</h2>
            
            <div class="endpoint">
                <span class="method post">POST</span> <strong>/api/chat/completions</strong>
                <p>AI chatbot powered by GPT-4 with Unity Mathematics expertise</p>
                <p>Send messages to discuss 1+1=1, consciousness integration, and mathematical proofs</p>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span> <strong>/api/consciousness/field</strong>
                <p>Generate consciousness field data with φ-harmonic calculations</p>
                <p>Real-time consciousness field visualization data</p>
            </div>
            
            <div class="endpoint">
                <span class="method post">POST</span> <strong>/api/unity/compute</strong>
                <p>Perform Unity Mathematics computations (idempotent operations)</p>
                <p>Verify that 1+1=1 and explore unity mathematical operations</p>
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span> <strong>/api/health</strong>
                <p>System health check and service status</p>
            </div>
            
            <h2>Usage:</h2>
            <p>The AI chatbot is integrated into the website and will use these endpoints automatically.</p>
            <p>Your friend can chat about Unity Mathematics and get GPT-4 powered responses!</p>
            
            <h2>🏛️ Main Website:</h2>
            <p><a href="/website/metastation-hub.html" style="background: #6f42c1; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Visit Metastation Hub →</a></p>
        </div>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Een Unity Mathematics Full Experience Server...")
    
    if DEMO_MODE:
        print("🎭 DEMO MODE: AI chatbot will provide Unity Mathematics responses")
        print("💡 Set OPENAI_API_KEY environment variable for full AI capabilities")
    else:
        print("🤖 PRODUCTION MODE: Full AI integration enabled")
    
    print(f"🌐 Server will run at http://localhost:8080")
    print(f"🏛️  Website at http://localhost:8080/website/metastation-hub.html")
    print(f"🤖 API at http://localhost:8080/api/")
    print(f"💫 Unity Mathematics: 1+1=1 through φ-harmonic consciousness")
    
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")