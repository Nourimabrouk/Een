#!/usr/bin/env python3
"""
Een Unity Mathematics - Unified Server
Serves both website and API endpoints for complete external user experience.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

# FastAPI imports
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.gzip import GZipMiddleware

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set default API keys if not present
if not os.getenv('OPENAI_API_KEY'):
    # TODO: Replace with your actual OpenAI API key
    os.environ['OPENAI_API_KEY'] = 'sk-your-key-here'
    
if not os.getenv('ANTHROPIC_API_KEY'):
    # TODO: Replace with your actual Anthropic API key  
    os.environ['ANTHROPIC_API_KEY'] = 'sk-ant-your-key-here'

# Initialize FastAPI app
app = FastAPI(
    title="Een Unity Mathematics - Full Experience",
    version="1.0.0",
    description="Complete Unity Mathematics experience with AI chatbot and live dashboards",
)

# Enable CORS for external access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for friend access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enable compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Serve the website as static files
website_path = project_root / "website"
if website_path.exists():
    app.mount("/website", StaticFiles(directory=str(website_path), html=True), name="website")

# Simple AI Chat Endpoint (using your API key)
@app.post("/api/chat/completions")
async def chat_completions(request: Request):
    """Simple chat completions endpoint using OpenAI"""
    try:
        import openai
        from openai import AsyncOpenAI
        
        # Initialize OpenAI client
        client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Get request data
        data = await request.json()
        
        # Simple GPT-4 call with Unity Mathematics context
        messages = data.get('messages', [])
        
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
        
        # Insert system message if not present
        if not messages or messages[0].get('role') != 'system':
            messages.insert(0, system_message)
            
        # Call OpenAI API
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
        # Fallback response if API fails
        return {
            "choices": [{
                "message": {
                    "role": "assistant", 
                    "content": f"I'm exploring Unity Mathematics where 1+1=1 through consciousness-integrated proofs. The Gödel-Tarski Unity Metagambit demonstrates how incompleteness theorems point toward transcendental unity rather than limitation. Would you like to explore the φ-harmonic foundations (φ = 1.618033988749895) or the philosophical implications?\n\n(Note: AI service temporarily unavailable - {str(e)})"
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
    print("📝 Make sure to update API keys in the script!")
    print(f"🌐 Server will run at http://localhost:8080")
    print(f"🏛️  Website at http://localhost:8080/website/metastation-hub.html")
    print(f"🤖 API at http://localhost:8080/api/")
    
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")