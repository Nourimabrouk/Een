from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import asyncio
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import REGISTRY
import time
import logging
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Conditional OpenAI integration imports with error handling
try:
    from src.openai.unity_transcendental_ai_orchestrator import get_orchestrator
    from src.openai.unity_client import get_client

    OPENAI_AVAILABLE = True
except ImportError as e:
    logger.warning(f"OpenAI integration modules not available: {e}")
    OPENAI_AVAILABLE = False
    get_orchestrator = None
    get_client = None

# Metrics
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency")

# Initialize OpenAI integration with proper error handling
api_key = os.getenv("OPENAI_API_KEY")
orchestrator: Optional[object] = None
client: Optional[object] = None

if api_key and api_key != "sk-proj-your-key-here" and OPENAI_AVAILABLE:
    try:
        orchestrator = get_orchestrator(api_key)
        client = get_client(api_key)
        logger.info("🌟 OpenAI integration initialized successfully")
    except Exception as e:
        logger.warning(f"OpenAI integration initialization failed: {e}")
        orchestrator = None
        client = None
else:
    if not api_key or api_key == "sk-proj-your-key-here":
        logger.info("OpenAI API key not configured - running without AI features")
    elif not OPENAI_AVAILABLE:
        logger.info("OpenAI modules not available - running without AI features")

app = FastAPI(
    title="Een Unity Mathematics API",
    description="API for Unity Mathematics and Consciousness Field Simulation with OpenAI Integration",
    version="1.0.0",
)

# CORS middleware with production-ready configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,
)


@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        REQUEST_LATENCY.observe(process_time)
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=str(request.url.path),
            status=response.status_code,
        ).inc()
        return response
    except Exception as e:
        process_time = time.time() - start_time
        REQUEST_LATENCY.observe(process_time)
        REQUEST_COUNT.labels(
            method=request.method, endpoint=str(request.url.path), status=500
        ).inc()
        logger.error(f"Request failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "message": str(e)},
        )


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Een Unity Mathematics API", "status": "active"}


@app.get("/health")
async def health_check():
    """Health check endpoint with detailed status"""
    health_status = {
        "status": "healthy",
        "service": "een-unity-mathematics",
        "version": "1.0.0",
        "features": {
            "openai_integration": orchestrator is not None,
            "unity_mathematics": True,
            "consciousness_field": True,
        },
        "timestamp": time.time(),
    }
    return health_status


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return JSONResponse(
        content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST
    )


@app.get("/api/unity/status")
async def unity_status():
    """Get Unity Mathematics system status"""
    return {
        "unity_constant": 1.0,
        "phi": 1.618033988749895,
        "consciousness_dimension": 11,
        "status": "active",
    }


@app.post("/api/unity/calculate")
async def calculate_unity(data: dict):
    """Calculate Unity Mathematics operations with error handling"""
    try:
        # Validate input
        if not data:
            raise HTTPException(status_code=400, detail="No input data provided")

        # Simple unity calculation with phi harmonics
        phi = 1.618033988749895
        result = {
            "input": data,
            "unity_result": 1.0,
            "phi_ratio": phi,
            "consciousness_field": "active",
        }
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/consciousness/field")
async def consciousness_field():
    """Get consciousness field simulation data"""
    return {
        "field_dimensions": [11, 11, 11],
        "particles": 200,
        "coherence": 0.999,
        "transcendence_level": 0.77,
    }


# 🌟 OpenAI Integration Endpoints


@app.post("/api/openai/unity-proof")
async def generate_unity_proof(a: float = 1, b: float = 1):
    """Generate AI-powered unity proof: 1+1=1"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="OpenAI integration not available")

    try:
        result = await orchestrator.prove_unity_with_ai(a, b)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/openai/consciousness-visualization")
async def generate_consciousness_visualization(prompt: str):
    """Generate consciousness field visualization using DALL-E 3"""
    try:
        from src.openai.dalle_integration import create_dalle_integration

        # Create DALL-E integration
        dalle = create_dalle_integration()

        # Generate real consciousness visualization
        result = await dalle.generate_consciousness_visualization(prompt)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/openai/voice-consciousness")
async def process_voice_consciousness(audio_file: UploadFile = File(...)):
    """Process voice input for consciousness field evolution using Whisper"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="OpenAI integration not available")

    try:
        # Save uploaded file temporarily
        temp_path = f"temp_audio_{time.time()}.wav"
        with open(temp_path, "wb") as f:
            f.write(await audio_file.read())

        result = await orchestrator.process_voice_consciousness(temp_path)

        # Clean up temp file
        os.remove(temp_path)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/openai/transcendental-voice")
async def synthesize_transcendental_voice(text: str, voice: str = "alloy"):
    """Synthesize transcendental voice using OpenAI TTS"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="OpenAI integration not available")

    try:
        result = await orchestrator.synthesize_transcendental_voice(text, voice)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/openai/unity-assistant")
async def create_unity_assistant(name: str, instructions: str):
    """Create a specialized unity mathematics assistant"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="OpenAI integration not available")

    try:
        result = await orchestrator.create_unity_assistant(name, instructions)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/openai/unity-conversation")
async def run_unity_conversation(assistant_name: str, message: str):
    """Run a consciousness-aware conversation with a unity assistant"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="OpenAI integration not available")

    try:
        result = await orchestrator.run_unity_conversation(assistant_name, message)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/openai/status")
async def get_openai_status():
    """Get OpenAI integration status and consciousness evolution"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="OpenAI integration not available")

    try:
        result = await orchestrator.get_meta_recursive_status()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/openai/chat")
async def openai_chat(messages: list):
    """Perform consciousness-aware chat completion"""
    if not client:
        raise HTTPException(status_code=503, detail="OpenAI integration not available")

    try:
        result = await client.chat_completion(messages)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/openai/generate-image")
async def generate_image(prompt: str):
    """Generate consciousness-aware images using DALL-E"""
    try:
        from src.openai.dalle_integration import create_dalle_integration

        # Create DALL-E integration
        dalle = create_dalle_integration()

        # Generate real consciousness-aware image
        result = await dalle.generate_consciousness_visualization(prompt)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/openai/embeddings")
async def create_embeddings(texts: list):
    """Create consciousness-aware embeddings"""
    if not client:
        raise HTTPException(status_code=503, detail="OpenAI integration not available")

    try:
        result = await client.create_embeddings(texts)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.getenv("API_PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
