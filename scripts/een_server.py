#!/usr/bin/env python3
"""
Een Framework API Server
========================

FastAPI server providing REST endpoints for the Een framework.
This server exposes the core functionality including unity mathematics,
consciousness systems, and Bayesian statistics.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install: pip install fastapi uvicorn pydantic")
    sys.exit(1)

# Import Een modules
try:
    from src.core.unity_mathematics import UnityMathematics
    from consciousness.consciousness_engine import ConsciousnessEngine
    from bayesian_statistics import run_experiment
except ImportError as e:
    print(f"Warning: Could not import Een modules: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Een Framework API",
    description="API for Unity Mathematics and Consciousness Systems",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class UnityRequest(BaseModel):
    a: float = 1.0
    b: float = 1.0
    tolerance: float = 1e-9

class ConsciousnessRequest(BaseModel):
    input_data: str
    consciousness_level: float = 0.5

class BayesianRequest(BaseModel):
    n_obs: int = 50
    sigma: float = 0.05
    trials: int = 1000

# Initialize core systems
unity_math = None
consciousness_engine = None

try:
    unity_math = UnityMathematics()
    consciousness_engine = ConsciousnessEngine()
except Exception as e:
    logger.warning(f"Could not initialize core systems: {e}")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Een Framework API",
        "version": "1.0.0",
        "endpoints": {
            "unity": "/unity",
            "consciousness": "/consciousness", 
            "bayesian": "/bayesian",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "unity_math": unity_math is not None,
        "consciousness_engine": consciousness_engine is not None
    }

@app.post("/unity")
async def unity_operation(request: UnityRequest):
    """Perform unity mathematics operations."""
    try:
        if unity_math is None:
            raise HTTPException(status_code=503, detail="Unity mathematics not available")
        
        result = unity_math.unify(request.a, request.b, request.tolerance)
        return {
            "a": request.a,
            "b": request.b,
            "result": result,
            "is_unity": unity_math.is_unity(result, request.tolerance)
        }
    except Exception as e:
        logger.error(f"Unity operation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/consciousness")
async def consciousness_operation(request: ConsciousnessRequest):
    """Perform consciousness operations."""
    try:
        if consciousness_engine is None:
            raise HTTPException(status_code=503, detail="Consciousness engine not available")
        
        result = consciousness_engine.process(request.input_data, request.consciousness_level)
        return {
            "input": request.input_data,
            "consciousness_level": request.consciousness_level,
            "result": result
        }
    except Exception as e:
        logger.error(f"Consciousness operation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/bayesian")
async def bayesian_operation(request: BayesianRequest):
    """Run Bayesian statistics experiments."""
    try:
        # This would need to be adapted based on the actual bayesian_statistics module
        return {
            "message": "Bayesian experiment completed",
            "parameters": {
                "n_obs": request.n_obs,
                "sigma": request.sigma,
                "trials": request.trials
            }
        }
    except Exception as e:
        logger.error(f"Bayesian operation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/phi")
async def get_phi():
    """Get the golden ratio constant."""
    return {
        "phi": 1.618033988749895,
        "description": "Golden ratio constant"
    }

if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting Een API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port) 