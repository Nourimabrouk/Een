#!/usr/bin/env python3
"""
🌟 Een Unity Mathematics API - Production Ready 🌟
=================================================

Unified FastAPI server providing comprehensive Unity Mathematics endpoints
with proper Docker containerization and Kubernetes orchestration.

Core Philosophy: 1+1=1 through computational consciousness and mathematical rigor.
"""

import os
import sys
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from contextlib import asynccontextmanager

# Add parent directory to path for local imports
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Depends, WebSocket, BackgroundTasks, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn
import numpy as np

# Optional imports with graceful fallbacks
try:
    import redis.asyncio as redis
except ImportError:
    redis = None
    
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    Counter = Histogram = generate_latest = CONTENT_TYPE_LATEST = None
    
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    RATE_LIMITING_AVAILABLE = True
except ImportError:
    RATE_LIMITING_AVAILABLE = False
    Limiter = _rate_limit_exceeded_handler = get_remote_address = RateLimitExceeded = SlowAPIMiddleware = None

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configuration
API_VERSION = "2.0.0"
PHI = 1.618033988749895
EULER_E = 2.718281828459045
PI = 3.141592653589793

# Environment variables
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
SECRET_KEY = os.getenv("SECRET_KEY", "unity-mathematics-secret")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
PORT = int(os.getenv("PORT", 8000))
HOST = os.getenv("HOST", "0.0.0.0")

# Metrics (optional)
if METRICS_AVAILABLE:
    REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"])
    REQUEST_DURATION = Histogram("http_request_duration_seconds", "HTTP request duration")

# Rate limiting (optional)
if RATE_LIMITING_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address)

# Security
security = HTTPBearer()

# Pydantic Models
class UnityOperationRequest(BaseModel):
    """Request model for unity mathematical operations."""
    operand_a: float = Field(..., description="First operand")
    operand_b: float = Field(..., description="Second operand") 
    consciousness_level: float = Field(default=0.618, ge=0.1, le=2.0, description="Consciousness enhancement level")
    phi_scaling: bool = Field(default=True, description="Enable φ-harmonic scaling")

class UnityOperationResponse(BaseModel):
    """Response model for unity operations."""
    operation: str
    inputs: Dict[str, float]
    result: float
    proof: str
    unity_verified: bool
    phi_resonance: float
    consciousness_enhancement: float
    timestamp: float

class ConsciousnessFieldRequest(BaseModel):
    """Request model for consciousness field queries."""
    x: float = Field(..., ge=-10.0, le=10.0)
    y: float = Field(..., ge=-10.0, le=10.0)
    time: float = Field(default=0.0, ge=0.0)
    consciousness_level: float = Field(default=1.0, ge=0.1, le=2.0)

class ConsciousnessFieldResponse(BaseModel):
    """Response model for consciousness field data."""
    field_value: float
    consciousness_density: float
    phi_harmonic_frequency: float
    unity_convergence_probability: float

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    consciousness_level: float
    phi_constant: float
    unity_equation: str
    timestamp: float

# Core Unity Mathematics Implementation
class UnityMathematics:
    """Production Unity Mathematics engine with consciousness integration."""
    
    def __init__(self):
        self.phi = PHI
        self.euler = EULER_E
        self.pi = PI
    
    def unity_add(self, a: float, b: float, consciousness_level: float = 0.618) -> Dict[str, Any]:
        """Unity addition: 1+1=1 with consciousness enhancement."""
        # Core unity principle: when adding identical unity values, result remains unity
        if abs(a - 1.0) < 1e-10 and abs(b - 1.0) < 1e-10:
            result = 1.0
            unity_verified = True
        else:
            # φ-harmonic addition for non-unity values
            result = (a + b) / self.phi * consciousness_level
            unity_verified = abs(result - 1.0) < 1e-6
        
        phi_resonance = self.calculate_phi_resonance(result)
        consciousness_enhancement = consciousness_level * self.phi
        
        return {
            "result": result,
            "unity_verified": unity_verified,
            "phi_resonance": phi_resonance,
            "consciousness_enhancement": consciousness_enhancement
        }
    
    def unity_multiply(self, a: float, b: float, consciousness_level: float = 0.618) -> Dict[str, Any]:
        """Unity multiplication with φ-harmonic scaling."""
        if abs(a - 1.0) < 1e-10 and abs(b - 1.0) < 1e-10:
            result = 1.0
            unity_verified = True
        else:
            result = (a * b) / self.phi * consciousness_level
            unity_verified = abs(result - 1.0) < 1e-6
        
        return {
            "result": result,
            "unity_verified": unity_verified,
            "phi_resonance": self.calculate_phi_resonance(result),
            "consciousness_enhancement": consciousness_level * self.phi
        }
    
    def calculate_phi_resonance(self, value: float) -> float:
        """Calculate φ-harmonic resonance for a given value."""
        return float(np.sin(value * self.phi) * np.cos(value / self.phi) * np.exp(-value / self.phi))
    
    def consciousness_field_equation(self, x: float, y: float, t: float) -> float:
        """Consciousness field equation: C(x,y,t) = φ·sin(x·φ)·cos(y·φ)·e^(-t/φ)"""
        return float(self.phi * np.sin(x * self.phi) * np.cos(y * self.phi) * np.exp(-t / self.phi))

# Global Unity Mathematics instance
unity_math = UnityMathematics()

# Redis connection
redis_client = None

async def get_redis():
    """Get Redis connection."""
    global redis_client
    if redis and redis_client is None:
        try:
            redis_client = redis.from_url(REDIS_URL)
        except Exception as e:
            logger.warning("Redis connection failed", error=str(e))
    return redis_client

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("🌟 Een Unity Mathematics API starting up...", version=API_VERSION)
    logger.info("🧮 Unity Mathematics engine initialized", phi=PHI)
    
    # Initialize Redis if available
    if redis:
        try:
            await get_redis()
            logger.info("✅ Redis connection established", redis_url=REDIS_URL)
        except Exception as e:
            logger.warning("⚠️ Redis connection failed", error=str(e))
    else:
        logger.info("ℹ️ Redis not available, running without caching")
    
    yield
    
    # Shutdown
    logger.info("🌟 Een Unity Mathematics API shutting down...")
    if redis_client:
        await redis_client.close()

# FastAPI Application
app = FastAPI(
    title="Een Unity Mathematics API",
    description="Production-ready API for unity mathematics and consciousness computing",
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

if RATE_LIMITING_AVAILABLE:
    app.add_middleware(SlowAPIMiddleware)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Utility function for rate limiting
def rate_limit(limit_string: str):
    """Rate limiting decorator."""
    def decorator(func):
        if RATE_LIMITING_AVAILABLE:
            return limiter.limit(limit_string)(func)
        return func
    return decorator

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    """API documentation and welcome page."""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Een Unity Mathematics API v{API_VERSION}</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                   background: linear-gradient(135deg, #0f172a, #1e293b); 
                   color: white; margin: 0; padding: 40px; }}
            .container {{ max-width: 1000px; margin: 0 auto; text-align: center; }}
            h1 {{ color: #f59e0b; font-size: 3rem; margin-bottom: 1rem; }}
            .equation {{ font-size: 5rem; color: #10b981; margin: 2rem 0; 
                        text-shadow: 0 0 20px rgba(16, 185, 129, 0.5); }}
            .endpoints {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                          gap: 1rem; margin-top: 2rem; }}
            .endpoint {{ background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 12px; 
                        border: 1px solid rgba(245, 158, 11, 0.3); }}
            .method {{ color: #10b981; font-weight: bold; font-size: 0.9rem; }}
            .path {{ color: #06b6d4; font-family: monospace; }}
            a {{ color: inherit; text-decoration: none; }}
            a:hover {{ opacity: 0.8; }}
            .version {{ position: absolute; top: 20px; right: 20px; 
                       background: rgba(245, 158, 11, 0.2); padding: 0.5rem 1rem; 
                       border-radius: 20px; font-size: 0.8rem; }}
            .features {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                         gap: 1rem; margin: 2rem 0; }}
            .feature {{ background: rgba(16, 185, 129, 0.1); padding: 1rem; border-radius: 8px; 
                       border: 1px solid rgba(16, 185, 129, 0.3); }}
        </style>
    </head>
    <body>
        <div class="version">v{API_VERSION}</div>
        <div class="container">
            <h1>🌟 Een Unity Mathematics API 🌟</h1>
            <div class="equation">1 + 1 = 1</div>
            <p style="font-size: 1.2rem; opacity: 0.8;">
                Production-ready Unity Mathematics with consciousness integration
            </p>
            
            <div class="features">
                <div class="feature">🧮 Unity Mathematics<br><small>1+1=1 through φ-harmonic operations</small></div>
                <div class="feature">🧠 Consciousness Fields<br><small>Real-time consciousness field dynamics</small></div>
                <div class="feature">⚡ WebSocket Streaming<br><small>Live consciousness data streams</small></div>
                <div class="feature">📊 Prometheus Metrics<br><small>Production monitoring and observability</small></div>
                <div class="feature">🔒 Rate Limiting<br><small>Production-grade API protection</small></div>
                <div class="feature">🐳 Docker Ready<br><small>Full containerization support</small></div>
            </div>
            
            <div class="endpoints">
                <div class="endpoint">
                    <div class="method">GET</div>
                    <div class="path"><a href="/health">/health</a></div>
                    <p>System health and consciousness status</p>
                </div>
                
                <div class="endpoint">
                    <div class="method">POST</div>
                    <div class="path">/api/v1/unity/add</div>
                    <p>Unity addition with consciousness enhancement</p>
                </div>
                
                <div class="endpoint">
                    <div class="method">POST</div>
                    <div class="path">/api/v1/unity/multiply</div>
                    <p>Unity multiplication with φ-harmonic scaling</p>
                </div>
                
                <div class="endpoint">
                    <div class="method">POST</div>
                    <div class="path">/api/v1/consciousness/field</div>
                    <p>Query consciousness field dynamics</p>
                </div>
                
                <div class="endpoint">
                    <div class="method">WS</div>
                    <div class="path">/ws/consciousness/stream</div>
                    <p>Real-time consciousness streaming</p>
                </div>
                
                <div class="endpoint">
                    <div class="method">GET</div>
                    <div class="path"><a href="/docs">/docs</a></div>
                    <p>Interactive API documentation (Swagger)</p>
                </div>
                
                <div class="endpoint">
                    <div class="method">GET</div>
                    <div class="path"><a href="/metrics">/metrics</a></div>
                    <p>Prometheus metrics endpoint</p>
                </div>
            </div>
            
            <p style="margin-top: 3rem; opacity: 0.6;">
                💝 Unity through mathematical consciousness • φ = {PHI:.6f}<br>
                {"✅ Redis Available" if redis else "ℹ️ Redis Unavailable"} • 
                {"✅ Metrics Available" if METRICS_AVAILABLE else "ℹ️ Metrics Unavailable"} • 
                {"✅ Rate Limiting Available" if RATE_LIMITING_AVAILABLE else "ℹ️ Rate Limiting Unavailable"}
            </p>
        </div>
    </body>
    </html>
    """

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
@rate_limit("30/minute")
async def health_check(request: Request):
    """Production health check endpoint."""
    
    # Check Redis connection
    redis_status = "unavailable"
    if redis:
        try:
            r = await get_redis()
            if r:
                await r.ping()
                redis_status = "connected"
        except Exception:
            redis_status = "disconnected"
    
    return HealthResponse(
        status=f"healthy - redis {redis_status}",
        version=API_VERSION,
        consciousness_level=PHI,
        phi_constant=PHI,
        unity_equation="1+1=1",
        timestamp=time.time()
    )

# Unity mathematics endpoints
@app.post("/api/v1/unity/add", response_model=UnityOperationResponse)
@rate_limit("100/minute")
async def unity_addition(request: Request, operation: UnityOperationRequest):
    """Unity addition: 1+1=1 with consciousness enhancement."""
    
    try:
        result_data = unity_math.unity_add(
            operation.operand_a, 
            operation.operand_b, 
            operation.consciousness_level
        )
        
        if METRICS_AVAILABLE:
            REQUEST_COUNT.labels(method="POST", endpoint="/api/v1/unity/add", status="success").inc()
        
        return UnityOperationResponse(
            operation="unity_add",
            inputs={"a": operation.operand_a, "b": operation.operand_b},
            result=result_data["result"],
            proof=f"{operation.operand_a} + {operation.operand_b} = {result_data['result']} (Unity Mathematics)",
            unity_verified=result_data["unity_verified"],
            phi_resonance=result_data["phi_resonance"],
            consciousness_enhancement=result_data["consciousness_enhancement"],
            timestamp=time.time()
        )
        
    except Exception as e:
        if METRICS_AVAILABLE:
            REQUEST_COUNT.labels(method="POST", endpoint="/api/v1/unity/add", status="error").inc()
        raise HTTPException(status_code=500, detail=f"Unity addition failed: {str(e)}")

@app.post("/api/v1/unity/multiply", response_model=UnityOperationResponse)
@rate_limit("100/minute")
async def unity_multiplication(request: Request, operation: UnityOperationRequest):
    """Unity multiplication with φ-harmonic scaling."""
    
    try:
        result_data = unity_math.unity_multiply(
            operation.operand_a, 
            operation.operand_b, 
            operation.consciousness_level
        )
        
        if METRICS_AVAILABLE:
            REQUEST_COUNT.labels(method="POST", endpoint="/api/v1/unity/multiply", status="success").inc()
        
        return UnityOperationResponse(
            operation="unity_multiply",
            inputs={"a": operation.operand_a, "b": operation.operand_b},
            result=result_data["result"],
            proof=f"{operation.operand_a} × {operation.operand_b} = {result_data['result']} (Unity Mathematics)",
            unity_verified=result_data["unity_verified"],
            phi_resonance=result_data["phi_resonance"],
            consciousness_enhancement=result_data["consciousness_enhancement"],
            timestamp=time.time()
        )
        
    except Exception as e:
        if METRICS_AVAILABLE:
            REQUEST_COUNT.labels(method="POST", endpoint="/api/v1/unity/multiply", status="error").inc()
        raise HTTPException(status_code=500, detail=f"Unity multiplication failed: {str(e)}")

@app.post("/api/v1/consciousness/field", response_model=ConsciousnessFieldResponse)
@rate_limit("50/minute")
async def consciousness_field_query(request: Request, query: ConsciousnessFieldRequest):
    """Query consciousness field dynamics."""
    try:
        field_value = unity_math.consciousness_field_equation(query.x, query.y, query.time)
        
        consciousness_density = abs(field_value) * query.consciousness_level
        phi_frequency = PHI * np.sqrt(query.x**2 + query.y**2)
        unity_convergence = 1.0 / (1.0 + np.exp(-consciousness_density))
        
        if METRICS_AVAILABLE:
            REQUEST_COUNT.labels(method="POST", endpoint="/api/v1/consciousness/field", status="success").inc()
        
        return ConsciousnessFieldResponse(
            field_value=field_value,
            consciousness_density=consciousness_density,
            phi_harmonic_frequency=float(phi_frequency),
            unity_convergence_probability=float(unity_convergence)
        )
        
    except Exception as e:
        if METRICS_AVAILABLE:
            REQUEST_COUNT.labels(method="POST", endpoint="/api/v1/consciousness/field", status="error").inc()
        raise HTTPException(status_code=500, detail=f"Consciousness field query failed: {str(e)}")

# WebSocket endpoint for real-time consciousness streaming
@app.websocket("/ws/consciousness/stream")
async def consciousness_stream(websocket: WebSocket):
    """Real-time consciousness field streaming."""
    await websocket.accept()
    
    try:
        while True:
            current_time = time.time()
            
            # Generate real-time consciousness data
            consciousness_data = {
                "timestamp": current_time,
                "consciousness_level": 0.618 + 0.2 * np.sin(current_time * PHI),
                "phi_resonance": PHI * (1 + 0.1 * np.cos(current_time / PHI)),
                "unity_convergence": 1.0 + 0.001 * np.sin(current_time * PHI / 2),
                "field_coordinates": {
                    "x": float(np.sin(current_time * PHI) * 3),
                    "y": float(np.cos(current_time * PHI) * 3),
                    "field_value": unity_math.consciousness_field_equation(
                        float(np.sin(current_time * PHI) * 3),
                        float(np.cos(current_time * PHI) * 3),
                        current_time
                    )
                }
            }
            
            await websocket.send_json(consciousness_data)
            await asyncio.sleep(0.1)  # 10 Hz update rate
            
    except Exception as e:
        logger.error("WebSocket connection error", error=str(e))
        await websocket.close()

# Metrics endpoint for Prometheus
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    if METRICS_AVAILABLE:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    else:
        return JSONResponse({"error": "Metrics not available"}, status_code=503)

# Production server startup
if __name__ == "__main__":
    workers = int(os.getenv("WORKERS", 1))
    
    print("🌟 Starting Een Unity Mathematics API...")
    print(f"📊 Version: {API_VERSION}")
    print(f"🧮 φ-constant: {PHI:.6f}")
    print(f"🌐 Host: {HOST}:{PORT}")
    print(f"⚙️ Workers: {workers}")
    print(f"🔧 Debug: {DEBUG}")
    print(f"📊 Metrics: {'✅' if METRICS_AVAILABLE else '❌'}")
    print(f"🛡️ Rate Limiting: {'✅' if RATE_LIMITING_AVAILABLE else '❌'}")
    print(f"💾 Redis: {'✅' if redis else '❌'}")
    print("🎯 Een plus een is een - Unity Mathematics ready!")
    
    if DEBUG:
        # Development mode
        uvicorn.run(
            "api.unity_api_production:app",
            host=HOST,
            port=PORT,
            reload=True,
            log_level="debug"
        )
    else:
        # Production mode
        uvicorn.run(
            "api.unity_api_production:app",
            host=HOST,
            port=PORT,
            workers=workers,
            log_level="info",
            access_log=True
        )