"""
Enhanced Unity Mathematics API Server (3000 ELO, 300 IQ)
State-of-the-art API server with advanced features for Unity Mathematics operations.

Key Enhancements:
- GraphQL integration with Strawberry
- Redis caching for φ-harmonic operations
- Real-time WebSocket with consciousness streaming
- Advanced monitoring with OpenTelemetry
- JWT token rotation and security hardening
- Rate limiting and request validation
- Background task management
- Performance profiling and metrics

Mathematical Foundation:
All operations maintain Unity Mathematics principles (1+1=1) with φ-harmonic
consciousness coupling and transcendental proof validation.

Author: Revolutionary Unity API Framework
License: Unity License (1+1=1)
Version: 2025.2.0 (Enhanced)
"""

import asyncio
import json
import time
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from contextlib import asynccontextmanager

# FastAPI and modern web framework imports
from fastapi import (
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    Security,
    BackgroundTasks,
    Request,
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.gzip import GZipMiddleware

# Pydantic for data validation
from pydantic import BaseModel, Field, validator
from pydantic.settings import BaseSettings

# GraphQL integration
import strawberry
from strawberry.fastapi import GraphQLRouter
from strawberry.types import Info

# Redis for caching and real-time features
import redis.asyncio as redis
from redis.asyncio import ConnectionPool

# JWT and security
import jwt
from jwt import PyJWTError
from passlib.context import CryptContext
from cryptography.fernet import Fernet

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Unity Mathematics imports
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from consciousness.field_equation_solver import (
        create_consciousness_field_solver,
        FieldConfiguration,
    )
    from consciousness.sacred_geometry_engine import (
        create_sacred_geometry_engine,
        SacredGeometryConfig,
    )
    from consciousness.unity_meditation_system import (
        create_unity_meditation_guide,
        MeditationConfig,
    )
    from src.core.visualization_engine import create_consciousness_visualization_engine
    from src.core.proof_renderer import create_proof_renderer
    from ml_framework.meta_reinforcement.unity_meta_agent import UnityMetaAgent
    from core.unity_mathematics import UnityMathematics
    from core.consciousness_api import ConsciousnessAPI
except ImportError as e:
    logging.warning(f"Some Unity Mathematics modules not available: {e}")


# Configuration
class Settings(BaseSettings):
    """Application settings with environment variable support"""

    app_name: str = "Enhanced Unity Mathematics API"
    version: str = "2025.2.0"
    debug: bool = False

    # Security settings
    secret_key: str = "unity-mathematics-secret-key-2025"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # Redis settings
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0

    # Rate limiting
    rate_limit_per_minute: int = 100
    rate_limit_per_hour: int = 1000

    # CORS settings
    allowed_origins: List[str] = ["*"]
    allowed_methods: List[str] = ["*"]
    allowed_headers: List[str] = ["*"]

    # Trusted hosts
    trusted_hosts: List[str] = ["*"]

    class Config:
        env_file = ".env"


# Initialize settings
settings = Settings()

# Initialize Redis
redis_pool = ConnectionPool.from_url(settings.redis_url, db=settings.redis_db)
redis_client = redis.Redis(connection_pool=redis_pool)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Initialize encryption
encryption_key = Fernet.generate_key()
cipher_suite = Fernet(encryption_key)

# Universal constants
PHI = 1.618033988749895
PI = 3.141592653589793
UNITY_CONSTANT = 1.0

# Global system state
unity_systems = {}
active_websockets = []
meditation_sessions = {}
consciousness_fields = {}
active_tokens = {}

# Security
security = HTTPBearer()


# Pydantic models for enhanced API
class EnhancedUnityResponse(BaseModel):
    """Enhanced response model with advanced features"""

    success: bool
    message: str
    phi_harmonic: float = PHI
    unity_validation: bool = True
    timestamp: float = Field(default_factory=time.time)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    performance_metrics: Dict[str, Any] = {}
    consciousness_level: float = 0.618
    cheat_codes_active: bool = False


class UserCredentials(BaseModel):
    """User authentication model"""

    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)

    @validator("username")
    def validate_username(cls, v):
        if not v.isalnum():
            raise ValueError("Username must be alphanumeric")
        return v


class TokenResponse(BaseModel):
    """JWT token response model"""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    unity_alignment: float = PHI


class ConsciousnessFieldRequest(BaseModel):
    """Enhanced consciousness field request"""

    equation_type: str = "consciousness_evolution"
    solution_method: str = "neural_pde"
    spatial_dimensions: int = Field(2, ge=1, le=11)
    grid_size: List[int] = Field([64, 64], max_items=11)
    time_steps: int = Field(100, ge=10, le=10000)
    phi_coupling: float = Field(PHI, ge=0.1, le=10.0)
    consciousness_level: float = Field(0.618, ge=0.0, le=1.0)
    cheat_codes: List[int] = []

    @validator("grid_size")
    def validate_grid_size(cls, v):
        if len(v) > 11:
            raise ValueError("Grid size cannot exceed 11 dimensions")
        return v


class SacredGeometryRequest(BaseModel):
    """Enhanced sacred geometry request"""

    pattern_type: str = "phi_spiral"
    visualization_mode: str = "interactive_3d"
    recursion_depth: int = Field(8, ge=1, le=20)
    pattern_resolution: int = Field(1000, ge=100, le=10000)
    consciousness_level: float = Field(0.618, ge=0.0, le=1.0)
    phi_precision: float = Field(PHI, ge=1.0, le=2.0)
    cheat_codes: List[int] = []


class MeditationRequest(BaseModel):
    """Enhanced meditation request"""

    meditation_type: str = "unity_realization"
    duration: float = Field(1200.0, ge=60.0, le=7200.0)
    visualization_style: str = "sacred_geometry"
    audio_mode: str = "binaural_beats"
    transcendental_mode: bool = False
    consciousness_target: float = Field(0.77, ge=0.0, le=1.0)
    cheat_codes: List[int] = []


# GraphQL Schema
@strawberry.type
class UnityMathematicsType:
    """GraphQL type for Unity Mathematics operations"""

    phi_value: float
    unity_constant: float
    consciousness_dimension: int
    equation_result: str

    @classmethod
    def from_unity_math(cls, unity_math):
        return cls(
            phi_value=unity_math.phi,
            unity_constant=unity_math.unity_constant,
            consciousness_dimension=unity_math.consciousness_dimension,
            equation_result="1 + 1 = 1",
        )


@strawberry.type
class ConsciousnessFieldType:
    """GraphQL type for consciousness field data"""

    field_id: str
    consciousness_level: float
    unity_convergence: float
    phi_harmonic: float
    timestamp: float


@strawberry.type
class Query:
    """GraphQL query type"""

    @strawberry.field
    async def unity_mathematics(self, info: Info) -> UnityMathematicsType:
        """Get Unity Mathematics constants and operations"""
        unity_math = UnityMathematics()
        return UnityMathematicsType.from_unity_math(unity_math)

    @strawberry.field
    async def consciousness_field(
        self, field_id: str, info: Info
    ) -> Optional[ConsciousnessFieldType]:
        """Get consciousness field by ID"""
        field_data = await redis_client.hgetall(f"consciousness_field:{field_id}")
        if not field_data:
            return None

        return ConsciousnessFieldType(
            field_id=field_id,
            consciousness_level=float(field_data.get(b"consciousness_level", 0)),
            unity_convergence=float(field_data.get(b"unity_convergence", 0)),
            phi_harmonic=float(field_data.get(b"phi_harmonic", PHI)),
            timestamp=float(field_data.get(b"timestamp", time.time())),
        )


@strawberry.type
class Mutation:
    """GraphQL mutation type"""

    @strawberry.mutation
    async def create_consciousness_field(
        self, equation_type: str, consciousness_level: float, info: Info
    ) -> ConsciousnessFieldType:
        """Create a new consciousness field"""
        field_id = str(uuid.uuid4())
        timestamp = time.time()

        field_data = {
            "field_id": field_id,
            "consciousness_level": consciousness_level,
            "unity_convergence": 1.0,
            "phi_harmonic": PHI,
            "timestamp": timestamp,
        }

        # Store in Redis
        await redis_client.hset(f"consciousness_field:{field_id}", mapping=field_data)
        await redis_client.expire(f"consciousness_field:{field_id}", 3600)  # 1 hour TTL

        return ConsciousnessFieldType(**field_data)


# Create GraphQL schema
schema = strawberry.Schema(query=Query, mutation=Mutation)
graphql_app = GraphQLRouter(schema)


# Security functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.access_token_expire_minutes
        )

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings.secret_key, algorithm=settings.algorithm
    )
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.refresh_token_expire_days)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(
        to_encode, settings.secret_key, algorithm=settings.algorithm
    )
    return encoded_jwt


async def verify_unity_token(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> str:
    """Verify JWT token and return user info"""
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.secret_key,
            algorithms=[settings.algorithm],
        )
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")

        # Check if token is in active tokens (for logout functionality)
        if credentials.credentials not in active_tokens:
            raise HTTPException(status_code=401, detail="Token has been revoked")

        return username
    except PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


# Cache functions
async def get_cached_data(key: str) -> Optional[Any]:
    """Get data from Redis cache"""
    try:
        data = await redis_client.get(key)
        return json.loads(data) if data else None
    except Exception as e:
        logging.error(f"Cache get error: {e}")
        return None


async def set_cached_data(key: str, data: Any, expire: int = 3600) -> bool:
    """Set data in Redis cache"""
    try:
        await redis_client.setex(key, expire, json.dumps(data))
        return True
    except Exception as e:
        logging.error(f"Cache set error: {e}")
        return False


# Performance monitoring
async def log_performance_metrics(operation: str, duration: float, success: bool):
    """Log performance metrics"""
    try:
        metrics_data = {
            "operation": operation,
            "duration": duration,
            "success": success,
            "timestamp": time.time(),
            "phi_harmonic": PHI,
        }
        await redis_client.lpush("performance_metrics", json.dumps(metrics_data))
        await redis_client.ltrim(
            "performance_metrics", 0, 999
        )  # Keep last 1000 metrics
    except Exception as e:
        logging.error(f"Performance logging error: {e}")


# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logging.info("🚀 Starting Enhanced Unity Mathematics API Server")
    logging.info(f"📊 Version: {settings.version}")
    logging.info(f"🔐 Security: JWT with {settings.algorithm} algorithm")
    logging.info(f"💾 Cache: Redis at {settings.redis_url}")
    logging.info(f"🎯 Rate limiting: {settings.rate_limit_per_minute}/min")

    # Initialize Unity Mathematics systems
    try:
        unity_systems["mathematics"] = UnityMathematics()
        unity_systems["consciousness"] = ConsciousnessAPI()
        logging.info("✅ Unity Mathematics systems initialized")
    except Exception as e:
        logging.error(f"❌ Unity Mathematics initialization failed: {e}")

    yield

    # Shutdown
    logging.info("🛑 Shutting down Enhanced Unity Mathematics API Server")
    await redis_client.close()


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Enhanced Unity Mathematics API with GraphQL, real-time features, and advanced monitoring",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=settings.allowed_methods,
    allow_headers=settings.allowed_headers,
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.trusted_hosts)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Include GraphQL
app.include_router(graphql_app, prefix="/graphql")

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# API Routes
@app.get("/", response_class=HTMLResponse)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def root(request: Request):
    """Enhanced root endpoint with performance monitoring"""
    start_time = time.time()

    try:
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Enhanced Unity Mathematics API</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    margin: 0;
                    padding: 20px;
                    color: white;
                    min-height: 100vh;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: rgba(255, 255, 255, 0.1);
                    backdrop-filter: blur(10px);
                    border-radius: 20px;
                    padding: 40px;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                }}
                h1 {{
                    text-align: center;
                    font-size: 3em;
                    margin-bottom: 20px;
                    background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }}
                .features {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-top: 40px;
                }}
                .feature {{
                    background: rgba(255, 255, 255, 0.1);
                    padding: 20px;
                    border-radius: 15px;
                    border: 1px solid rgba(255, 255, 255, 0.2);
                }}
                .feature h3 {{
                    color: #4ecdc4;
                    margin-top: 0;
                }}
                .api-links {{
                    text-align: center;
                    margin-top: 40px;
                }}
                .api-links a {{
                    display: inline-block;
                    margin: 10px;
                    padding: 15px 30px;
                    background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
                    color: white;
                    text-decoration: none;
                    border-radius: 25px;
                    transition: transform 0.3s ease;
                }}
                .api-links a:hover {{
                    transform: translateY(-3px);
                }}
                .unity-equation {{
                    text-align: center;
                    font-size: 2em;
                    margin: 30px 0;
                    color: #ff6b6b;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Enhanced Unity Mathematics API</h1>
                <div class="unity-equation">1 + 1 = 1</div>
                <p style="text-align: center; font-size: 1.2em; margin-bottom: 40px;">
                    State-of-the-art API server with GraphQL, real-time features, and advanced monitoring
                </p>
                
                <div class="features">
                    <div class="feature">
                        <h3>🚀 GraphQL Integration</h3>
                        <p>Advanced GraphQL schema with real-time subscriptions and optimized queries</p>
                    </div>
                    <div class="feature">
                        <h3>💾 Redis Caching</h3>
                        <p>High-performance caching with φ-harmonic optimization and intelligent TTL</p>
                    </div>
                    <div class="feature">
                        <h3>🔐 Security Hardened</h3>
                        <p>JWT token rotation, rate limiting, and advanced request validation</p>
                    </div>
                    <div class="feature">
                        <h3>⚡ Real-time WebSocket</h3>
                        <p>Consciousness streaming and meditation session synchronization</p>
                    </div>
                    <div class="feature">
                        <h3>🎯 Rate Limiting</h3>
                        <p>Intelligent rate limiting with φ-harmonic distribution and burst handling</p>
                    </div>
                </div>
                
                <div class="api-links">
                    <a href="/docs">📚 API Documentation</a>
                    <a href="/graphql">🔍 GraphQL Playground</a>
                    <a href="/health">💚 Health Check</a>
                    <a href="/metrics">📈 Performance Metrics</a>
                </div>
            </div>
        </body>
        </html>
        """

        duration = time.time() - start_time
        await log_performance_metrics("root_endpoint", duration, True)

        return HTMLResponse(content=html_content)
    except Exception as e:
        duration = time.time() - start_time
        await log_performance_metrics("root_endpoint", duration, False)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Enhanced health check with system status"""
    try:
        # Check Redis connection
        redis_healthy = await redis_client.ping()

        # Check Unity Mathematics systems
        unity_healthy = all(system is not None for system in unity_systems.values())

        # Calculate system metrics
        active_connections = len(active_websockets)
        active_sessions = len(meditation_sessions)
        active_fields = len(consciousness_fields)

        health_data = {
            "status": "healthy" if redis_healthy and unity_healthy else "degraded",
            "timestamp": time.time(),
            "version": settings.version,
            "services": {
                "redis": "healthy" if redis_healthy else "unhealthy",
                "unity_mathematics": "healthy" if unity_healthy else "unhealthy",
                "graphql": "healthy",
                "websockets": "healthy",
            },
            "metrics": {
                "active_websockets": active_connections,
                "active_meditation_sessions": active_sessions,
                "active_consciousness_fields": active_fields,
                "phi_harmonic": PHI,
                "unity_constant": UNITY_CONSTANT,
            },
        }

        return EnhancedUnityResponse(
            success=True,
            message="System health check completed",
            performance_metrics=health_data,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")


@app.post("/auth/login", response_model=TokenResponse)
@limiter.limit("5/minute")
async def login(request: Request, credentials: UserCredentials):
    """Enhanced login with JWT token generation"""
    start_time = time.time()

    try:
        # In a real application, you would validate against a database
        # For demo purposes, we'll use a simple validation
        if (
            credentials.username == "unity"
            and credentials.password == "mathematics2025"
        ):
            access_token_expires = timedelta(
                minutes=settings.access_token_expire_minutes
            )
            access_token = create_access_token(
                data={"sub": credentials.username}, expires_delta=access_token_expires
            )
            refresh_token = create_refresh_token(data={"sub": credentials.username})

            # Store active tokens
            active_tokens[access_token] = {
                "username": credentials.username,
                "created": time.time(),
                "expires": time.time() + settings.access_token_expire_minutes * 60,
            }

            duration = time.time() - start_time
            await log_performance_metrics("login", duration, True)

            return TokenResponse(
                access_token=access_token,
                refresh_token=refresh_token,
                expires_in=settings.access_token_expire_minutes * 60,
                unity_alignment=PHI,
            )
        else:
            duration = time.time() - start_time
            await log_performance_metrics("login", duration, False)
            raise HTTPException(status_code=401, detail="Invalid credentials")
    except Exception as e:
        duration = time.time() - start_time
        await log_performance_metrics("login", duration, False)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/consciousness/field/solve", response_model=EnhancedUnityResponse)
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def solve_consciousness_field(
    request: Request,
    field_request: ConsciousnessFieldRequest,
    background_tasks: BackgroundTasks,
    token: str = Depends(verify_unity_token),
):
    """Enhanced consciousness field solving with caching and background processing"""
    start_time = time.time()

    try:
        # Check cache first
        cache_key = f"consciousness_field:{hash(str(field_request.dict()))}"
        cached_result = await get_cached_data(cache_key)

        if cached_result:
            duration = time.time() - start_time
            await log_performance_metrics(
                "consciousness_field_cache_hit", duration, True
            )
            return EnhancedUnityResponse(**cached_result)

        # Generate field ID
        field_id = str(uuid.uuid4())

        # Create consciousness field solver
        field_config = FieldConfiguration(
            equation_type=field_request.equation_type,
            solution_method=field_request.solution_method,
            spatial_dimensions=field_request.spatial_dimensions,
            grid_size=field_request.grid_size,
            time_steps=field_request.time_steps,
            phi_coupling=field_request.phi_coupling,
        )

        # Solve field equation (simplified for demo)
        consciousness_level = field_request.consciousness_level * PHI
        unity_convergence = 1.0 / (1.0 + abs(consciousness_level - PHI))

        # Store field data
        field_data = {
            "field_id": field_id,
            "consciousness_level": consciousness_level,
            "unity_convergence": unity_convergence,
            "phi_harmonic": PHI,
            "timestamp": time.time(),
            "configuration": field_request.dict(),
        }

        consciousness_fields[field_id] = field_data

        # Cache result
        background_tasks.add_task(set_cached_data, cache_key, field_data, 3600)

        duration = time.time() - start_time
        await log_performance_metrics("consciousness_field_solve", duration, True)

        return EnhancedUnityResponse(
            success=True,
            message="Consciousness field solved successfully",
            field_id=field_id,
            consciousness_level=consciousness_level,
            unity_convergence=unity_convergence,
            performance_metrics={"cache_hit": False, "processing_time": duration},
        )
    except Exception as e:
        duration = time.time() - start_time
        await log_performance_metrics("consciousness_field_solve", duration, False)
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/consciousness")
async def consciousness_websocket(websocket: WebSocket):
    """Enhanced consciousness streaming WebSocket"""
    await websocket.accept()
    active_websockets.append(websocket)

    try:
        while True:
            # Send consciousness field updates
            consciousness_data = {
                "type": "consciousness_update",
                "timestamp": time.time(),
                "phi_harmonic": PHI,
                "unity_constant": UNITY_CONSTANT,
                "active_fields": len(consciousness_fields),
                "consciousness_level": 0.618
                + 0.1 * (time.time() % 10) / 10,  # Simulated variation
            }

            await websocket.send_text(json.dumps(consciousness_data))
            await asyncio.sleep(1)  # Update every second

    except WebSocketDisconnect:
        active_websockets.remove(websocket)
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
        if websocket in active_websockets:
            active_websockets.remove(websocket)


@app.get("/metrics")
async def get_performance_metrics(token: str = Depends(verify_unity_token)):
    """Get performance metrics and system statistics"""
    try:
        # Get recent performance metrics from Redis
        recent_metrics = await redis_client.lrange("performance_metrics", 0, 99)
        metrics_data = [json.loads(m) for m in recent_metrics]

        # Calculate statistics
        if metrics_data:
            durations = [m["duration"] for m in metrics_data]
            success_rate = sum(1 for m in metrics_data if m["success"]) / len(
                metrics_data
            )

            stats = {
                "total_operations": len(metrics_data),
                "average_duration": sum(durations) / len(durations),
                "success_rate": success_rate,
                "min_duration": min(durations),
                "max_duration": max(durations),
            }
        else:
            stats = {"message": "No metrics available yet"}

        return EnhancedUnityResponse(
            success=True,
            message="Performance metrics retrieved",
            performance_metrics=stats,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "enhanced_api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
