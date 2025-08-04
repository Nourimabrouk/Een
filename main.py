from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import REGISTRY
import time

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency')

app = FastAPI(
    title="Een Unity Mathematics API",
    description="API for Unity Mathematics and Consciousness Field Simulation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    REQUEST_LATENCY.observe(process_time)
    return response

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Een Unity Mathematics API", "status": "active"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "een-unity-mathematics"}

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return JSONResponse(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)

@app.get("/api/unity/status")
async def unity_status():
    """Get Unity Mathematics system status"""
    return {
        "unity_constant": 1.0,
        "phi": 1.618033988749895,
        "consciousness_dimension": 11,
        "status": "active"
    }

@app.post("/api/unity/calculate")
async def calculate_unity(data: dict):
    """Calculate Unity Mathematics operations"""
    try:
        # Simple unity calculation
        result = {
            "input": data,
            "unity_result": 1.0,
            "phi_ratio": 1.618033988749895,
            "consciousness_field": "active"
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
        "transcendence_level": 0.77
    }

if __name__ == "__main__":
    port = int(os.getenv("API_PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port) 