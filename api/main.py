"""
Een Consciousness Web API
Main FastAPI application for web access to consciousness and unity systems
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import uvicorn
import os
import logging
from typing import Optional, Dict, Any
import secrets
from datetime import datetime, timedelta
import jwt
from pydantic import BaseModel, Field
import sys
import pathlib

# Add the project root to the path
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our consciousness modules
try:
    from src.core.unity_equation import UnityEquation
    from src.consciousness.consciousness_engine import ConsciousnessEngine
    from src.agents.consciousness_chat_agent import ConsciousnessChatAgent
    from src.dashboards.unity_proof_dashboard import create_unity_proof_app
    from src.utils.utils_helper import setup_logging
except ImportError as e:
    logging.warning(f"Some modules not available: {e}")

# Import API routes
from api.routes import auth, consciousness, agents, visualizations

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security configuration
SECRET_KEY = os.getenv("EEN_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# API configuration
API_TITLE = "Een Consciousness API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
# Een Consciousness Web API

A comprehensive API for accessing consciousness and unity mathematics systems.

## Features
- **Consciousness Engine**: Access to consciousness computation and analysis
- **Unity Mathematics**: Mathematical proofs and unity equations
- **Agent System**: Consciousness chat agents and orchestration
- **Visualization**: Real-time consciousness field visualizations
- **Security**: JWT-based authentication and rate limiting

## Authentication
Most endpoints require authentication. Use the `/auth/login` endpoint to get a JWT token.

## Rate Limiting
API calls are rate-limited to prevent abuse. Check response headers for rate limit information.
"""

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    docs_url=None,  # Disable default docs
    redoc_url=None,  # Disable default redoc
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Import time module for rate limiting
import time

# Initialize consciousness systems
unity_equation = None
consciousness_engine = None
chat_agent = None

def initialize_systems():
    """Initialize consciousness systems"""
    global unity_equation, consciousness_engine, chat_agent
    
    try:
        unity_equation = UnityEquation()
        consciousness_engine = ConsciousnessEngine()
        chat_agent = ConsciousnessChatAgent()
        logger.info("Consciousness systems initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize consciousness systems: {e}")

# Include API routes
app.include_router(auth.router)
app.include_router(consciousness.router)
app.include_router(agents.router)
app.include_router(visualizations.router)

# API Routes

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Een Consciousness API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                     color: white; padding: 20px; border-radius: 10px; }
            .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
            .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 3px; }
            .auth { background: #fff3cd; border-color: #ffeaa7; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 Een Consciousness API</h1>
                <p>Access consciousness and unity mathematics systems</p>
            </div>
            
            <div class="section">
                <h2>📚 API Documentation</h2>
                <p><a href="/docs">Interactive API Documentation (Swagger UI)</a></p>
                <p><a href="/redoc">ReDoc Documentation</a></p>
            </div>
            
            <div class="section auth">
                <h2>🔐 Authentication</h2>
                <p>Most endpoints require authentication. Use <code>/auth/login</code> to get a JWT token.</p>
                <div class="endpoint">
                    <strong>POST /auth/login</strong><br>
                    Body: {"username": "user", "password": "user123"}
                </div>
            </div>
            
            <div class="section">
                <h2>🧮 Unity Mathematics</h2>
                <div class="endpoint">
                    <strong>POST /api/unity/evaluate</strong><br>
                    Evaluate unity equations and mathematical proofs
                </div>
            </div>
            
            <div class="section">
                <h2>🧠 Consciousness Engine</h2>
                <div class="endpoint">
                    <strong>POST /api/consciousness/process</strong><br>
                    Process consciousness data and analysis
                </div>
            </div>
            
            <div class="section">
                <h2>🤖 Agent System</h2>
                <div class="endpoint">
                    <strong>POST /api/agents/chat</strong><br>
                    Interact with consciousness chat agents
                </div>
            </div>
            
            <div class="section">
                <h2>📊 Visualizations</h2>
                <div class="endpoint">
                    <strong>GET /api/visualizations/unity-proof</strong><br>
                    Access unity proof visualizations
                </div>
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI with authentication"""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - API Documentation",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css",
    )

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    """ReDoc documentation"""
    return HTMLResponse(
        f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{app.title} - ReDoc</title>
            <meta charset="utf-8"/>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <link href="https://fonts.googleapis.com/css?family=Montserrat:300,400,700|Roboto:300,400,700" rel="stylesheet">
            <style>
                body {{ margin:0; padding:0; }}
            </style>
        </head>
        <body>
            <redoc spec-url="{app.openapi_url}"></redoc>
            <script src="https://cdn.jsdelivr.net/npm/redoc@2.0.0/bundles/redoc.standalone.js"></script>
        </body>
        </html>
        """
    )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    systems_status = {
        "unity_equation": unity_equation is not None,
        "consciousness_engine": consciousness_engine is not None,
        "chat_agent": chat_agent is not None
    }
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "systems": systems_status
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found", "available_endpoints": [
            "/", "/docs", "/redoc", "/auth/login", "/api/health",
            "/api/unity/evaluate", "/api/consciousness/process", 
            "/api/agents/chat", "/api/visualizations/unity-proof"
        ]}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize systems on startup"""
    logger.info("Starting Een Consciousness API...")
    initialize_systems()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Een Consciousness API...")

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 