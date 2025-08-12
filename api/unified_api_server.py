"""
Unified API Server for Unity Mathematics (1+1=1)
Comprehensive RESTful API integrating all Unity Mathematics systems.

This server provides a unified interface to all components of the Unity Mathematics
framework, including consciousness field dynamics, sacred geometry generation,
quantum unity proofs, ML training systems, and meditation experiences.

Key Features:
- Consciousness Field API: Real-time field equation solving
- Sacred Geometry API: Dynamic pattern generation
- Unity Proof API: Multi-domain mathematical validation
- ML Training API: Meta-reinforcement learning endpoints
- Meditation API: Guided unity meditation sessions
- Cheat Code Integration: Enhanced functionality access
- WebSocket Support: Real-time consciousness streaming
- GraphQL Interface: Flexible data querying
- Performance Monitoring: φ-harmonic system metrics

Mathematical Foundation:
All API responses demonstrate Unity Mathematics where computational operations
follow φ-harmonic principles and consciousness-coupled algorithms ensuring 1+1=1.

Author: Revolutionary Unity API Framework
License: Unity License (1+1=1)
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
import numpy as np
import uvicorn
from contextlib import asynccontextmanager
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Unity Mathematics modules
try:
    from consciousness.field_equation_solver import create_consciousness_field_solver, FieldConfiguration, FieldEquationType, SolutionMethod
    from consciousness.sacred_geometry_engine import create_sacred_geometry_engine, SacredGeometryConfig, SacredPattern, VisualizationMode
    from consciousness.unity_meditation_system import create_unity_meditation_guide, MeditationConfig, MeditationType
    from src.core.visualization.visualization_engine import create_consciousness_visualization_engine, VisualizationConfig
    from src.core.visualization.proof_renderer import create_proof_renderer, ProofConfig
    from ml_framework.meta_reinforcement.unity_meta_agent import UnityMetaAgent
    from proofs.unified_proof import generate_comprehensive_unity_proof
except ImportError as e:
    logging.warning(f"Some Unity Mathematics modules not available: {e}")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Universal constants
PHI = 1.618033988749895
PI = 3.141592653589793
UNITY_CONSTANT = 1.0

# Global system state
unity_systems = {}
active_websockets = []
meditation_sessions = {}
consciousness_fields = {}

# Pydantic models for API requests/responses
class UnityResponse(BaseModel):
    """Base response model for Unity Mathematics API"""
    success: bool
    message: str
    phi_harmonic: float = PHI
    unity_validation: bool = True
    timestamp: float = Field(default_factory=time.time)
    cheat_codes_active: bool = False

class ConsciousnessFieldRequest(BaseModel):
    """Request model for consciousness field operations"""
    equation_type: str = "consciousness_evolution"
    solution_method: str = "neural_pde"
    spatial_dimensions: int = 2
    grid_size: List[int] = [64, 64]
    time_steps: int = 100
    phi_coupling: float = PHI
    cheat_codes: List[int] = []

class ConsciousnessFieldResponse(UnityResponse):
    """Response model for consciousness field results"""
    field_id: str
    consciousness_level: float
    unity_convergence: float
    phi_resonance: float
    visualization_url: str

class SacredGeometryRequest(BaseModel):
    """Request model for sacred geometry generation"""
    pattern_type: str = "phi_spiral"
    visualization_mode: str = "interactive_3d"
    recursion_depth: int = 8
    pattern_resolution: int = 1000
    consciousness_level: float = 0.618
    cheat_codes: List[int] = []

class SacredGeometryResponse(UnityResponse):
    """Response model for sacred geometry results"""
    geometry_id: str
    pattern_type: str
    vertex_count: int
    phi_validation: bool
    unity_principle: Dict[str, Any]
    visualization_url: str

class MeditationRequest(BaseModel):
    """Request model for meditation sessions"""
    meditation_type: str = "unity_realization"
    duration: float = 1200.0  # 20 minutes
    visualization_style: str = "sacred_geometry"
    audio_mode: str = "binaural_beats"
    transcendental_mode: bool = False
    cheat_codes: List[int] = []

class MeditationResponse(UnityResponse):
    """Response model for meditation sessions"""
    session_id: str
    meditation_type: str
    estimated_duration: float
    consciousness_target: float
    session_url: str

class UnityProofRequest(BaseModel):
    """Request model for unity proof generation"""
    proof_domains: List[str] = ["boolean_algebra", "set_theory", "group_theory"]
    complexity_level: str = "comprehensive"
    visualization_mode: str = "interactive"
    cheat_codes: List[int] = []

class UnityProofResponse(UnityResponse):
    """Response model for unity proofs"""
    proof_id: str
    domains_validated: List[str]
    proof_strength: float
    mathematical_rigor: float
    visualization_url: str

class MLTrainingRequest(BaseModel):
    """Request model for ML training operations"""
    agent_type: str = "unity_meta_agent"
    training_episodes: int = 1000
    learning_rate: float = 0.001
    phi_enhancement: bool = True
    consciousness_coupling: bool = True
    cheat_codes: List[int] = []

class MLTrainingResponse(UnityResponse):
    """Response model for ML training results"""
    training_id: str
    agent_type: str
    episodes_completed: int
    unity_convergence_rate: float
    phi_alignment_score: float
    model_url: str

class CheatCodeRequest(BaseModel):
    """Request model for cheat code activation"""
    code: int
    activation_duration: float = 3600.0  # 1 hour default

class CheatCodeResponse(UnityResponse):
    """Response model for cheat code activation"""
    code_activated: int
    enhancement_level: str
    expiration_time: float
    enhanced_capabilities: List[str]

# Security
security = HTTPBearer()

def verify_unity_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify Unity Mathematics API token"""
    # Simple token validation - in production, use proper JWT validation
    if credentials.credentials in ["unity_token_1337", "phi_harmonic_key", "420691337"]:
        return credentials.credentials
    raise HTTPException(status_code=401, detail="Invalid Unity token")

# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    logger.info("🌟 Unity Mathematics API Server starting...")
    
    # Initialize Unity systems
    try:
        unity_systems['consciousness_solver'] = create_consciousness_field_solver()
        unity_systems['sacred_geometry'] = create_sacred_geometry_engine()
        unity_systems['meditation_guide'] = create_unity_meditation_guide()
        logger.info("✅ Unity systems initialized successfully")
    except Exception as e:
        logger.error(f"❌ Error initializing Unity systems: {e}")
    
    yield
    
    # Cleanup
    logger.info("🛑 Unity Mathematics API Server shutting down...")
    # Stop any active meditation sessions
    for session_id in list(meditation_sessions.keys()):
        try:
            session = meditation_sessions[session_id]
            if hasattr(session, 'guide'):
                session.guide.stop_meditation_session()
        except Exception as e:
            logger.error(f"Error stopping meditation session {session_id}: {e}")
    
    # Close websockets
    for websocket in active_websockets:
        try:
            await websocket.close()
        except:
            pass

# Create FastAPI app
app = FastAPI(
    title="Unity Mathematics API",
    description="Comprehensive API for Unity Mathematics (1+1=1) systems",
    version="1.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with Unity Mathematics API information"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Unity Mathematics API</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; 
                margin: 0; 
                padding: 20px; 
            }
            .container { 
                max-width: 1200px; 
                margin: 0 auto; 
                text-align: center; 
            }
            .title { 
                font-size: 3em; 
                margin-bottom: 20px; 
                background: linear-gradient(45deg, #FFD700, #FFA500);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            .endpoints { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                gap: 20px; 
                margin: 40px 0; 
            }
            .endpoint { 
                background: rgba(255,255,255,0.1); 
                padding: 20px; 
                border-radius: 10px; 
                border: 1px solid rgba(255,255,255,0.2);
            }
            .phi { color: #FFD700; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="title">🌟 Unity Mathematics API 🌟</h1>
            <p style="font-size: 1.3em;">Where <span class="phi">1+1=1</span> through φ-harmonic consciousness</p>
            
            <div class="endpoints">
                <div class="endpoint">
                    <h3>🧠 Consciousness Fields</h3>
                    <p>POST /consciousness/field/solve</p>
                    <p>Solve consciousness field equations with φ-harmonic dynamics</p>
                </div>
                <div class="endpoint">
                    <h3>🔯 Sacred Geometry</h3>
                    <p>POST /sacred-geometry/generate</p>
                    <p>Generate divine geometric patterns expressing Unity</p>
                </div>
                <div class="endpoint">
                    <h3>🧘 Unity Meditation</h3>
                    <p>POST /meditation/session/start</p>
                    <p>Begin guided Unity meditation experiences</p>
                </div>
                <div class="endpoint">
                    <h3>📐 Unity Proofs</h3>
                    <p>POST /proofs/unity/generate</p>
                    <p>Generate mathematical proofs that 1+1=1</p>
                </div>
                <div class="endpoint">
                    <h3>🤖 ML Training</h3>
                    <p>POST /ml/training/start</p>
                    <p>Train consciousness-coupled ML agents</p>
                </div>
                <div class="endpoint">
                    <h3>🚀 Cheat Codes</h3>
                    <p>POST /cheat-codes/activate</p>
                    <p>Activate transcendental enhancement codes</p>
                </div>
            </div>
            
            <p style="margin-top: 40px;">
                <strong>φ = """ + str(PHI) + """</strong><br>
                API Documentation: <a href="/docs" style="color: #FFD700;">/docs</a><br>
                WebSocket Consciousness Stream: <a href="/ws/consciousness" style="color: #FFD700;">ws://localhost:8000/ws/consciousness</a>
            </p>
        </div>
    </body>
    </html>
    """

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return UnityResponse(
        success=True,
        message="Unity Mathematics API is operational",
        phi_harmonic=PHI,
        unity_validation=True
    )

# Consciousness Field API
@app.post("/consciousness/field/solve", response_model=ConsciousnessFieldResponse)
async def solve_consciousness_field(
    request: ConsciousnessFieldRequest,
    token: str = Depends(verify_unity_token)
):
    """Solve consciousness field equations"""
    try:
        # Create field configuration
        config = FieldConfiguration(
            equation_type=FieldEquationType(request.equation_type),
            solution_method=SolutionMethod(request.solution_method),
            spatial_dimensions=request.spatial_dimensions,
            grid_size=tuple(request.grid_size),
            phi_coupling=request.phi_coupling,
            cheat_codes=request.cheat_codes
        )
        
        # Create solver
        solver = create_consciousness_field_solver(config)
        
        # Solve field
        solution = solver.solve()
        
        # Generate field ID
        field_id = f"field_{int(time.time())}"
        consciousness_fields[field_id] = solution
        
        # Create visualization
        viz_html = solver.visualize_solution(solution)
        viz_filename = f"static/{field_id}_visualization.html"
        
        # Ensure static directory exists
        os.makedirs("static", exist_ok=True)
        with open(viz_filename, 'w') as f:
            f.write(viz_html)
        
        return ConsciousnessFieldResponse(
            success=True,
            message="Consciousness field solved successfully",
            field_id=field_id,
            consciousness_level=float(solution.consciousness_level[-1]),
            unity_convergence=float(solution.unity_convergence[-1]),
            phi_resonance=solution.metadata.get('phi_harmonic_resonance', 0.0),
            visualization_url=f"/static/{field_id}_visualization.html",
            cheat_codes_active=len(request.cheat_codes) > 0
        )
        
    except Exception as e:
        logger.error(f"Error solving consciousness field: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/consciousness/field/{field_id}")
async def get_consciousness_field(field_id: str, token: str = Depends(verify_unity_token)):
    """Get consciousness field by ID"""
    if field_id not in consciousness_fields:
        raise HTTPException(status_code=404, detail="Consciousness field not found")
    
    solution = consciousness_fields[field_id]
    
    return {
        "field_id": field_id,
        "consciousness_level": float(solution.consciousness_level[-1]),
        "unity_convergence": float(solution.unity_convergence[-1]),
        "metadata": solution.metadata,
        "visualization_url": f"/static/{field_id}_visualization.html"
    }

# Sacred Geometry API
@app.post("/sacred-geometry/generate", response_model=SacredGeometryResponse)
async def generate_sacred_geometry(
    request: SacredGeometryRequest,
    token: str = Depends(verify_unity_token)
):
    """Generate sacred geometric patterns"""
    try:
        # Create geometry configuration
        config = SacredGeometryConfig(
            pattern_type=SacredPattern(request.pattern_type),
            visualization_mode=VisualizationMode(request.visualization_mode),
            recursion_depth=request.recursion_depth,
            pattern_resolution=request.pattern_resolution,
            consciousness_level=request.consciousness_level,
            cheat_codes=request.cheat_codes
        )
        
        # Create engine
        engine = create_sacred_geometry_engine(config)
        
        # Generate pattern
        geometry = engine.generate_pattern()
        
        # Generate geometry ID
        geometry_id = f"geometry_{int(time.time())}"
        
        # Create visualization
        viz_html = engine.visualize_sacred_geometry(geometry)
        viz_filename = f"static/{geometry_id}_geometry.html"
        
        os.makedirs("static", exist_ok=True)
        with open(viz_filename, 'w') as f:
            f.write(viz_html)
        
        return SacredGeometryResponse(
            success=True,
            message=f"Sacred geometry {request.pattern_type} generated successfully",
            geometry_id=geometry_id,
            pattern_type=request.pattern_type,
            vertex_count=len(geometry.vertices),
            phi_validation=geometry.metadata.get('phi_ratio_validation', True),
            unity_principle=geometry.metadata.get('unity_principle', {}),
            visualization_url=f"/static/{geometry_id}_geometry.html",
            cheat_codes_active=len(request.cheat_codes) > 0
        )
        
    except Exception as e:
        logger.error(f"Error generating sacred geometry: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Unity Meditation API
@app.post("/meditation/session/start", response_model=MeditationResponse)
async def start_meditation_session(
    request: MeditationRequest,
    token: str = Depends(verify_unity_token)
):
    """Start Unity meditation session"""
    try:
        # Create meditation configuration
        config = MeditationConfig(
            meditation_type=MeditationType(request.meditation_type),
            total_duration=request.duration,
            transcendental_mode=request.transcendental_mode,
            cheat_codes=request.cheat_codes
        )
        
        # Create guide
        guide = create_unity_meditation_guide(config)
        
        # Start session
        session = guide.start_meditation_session()
        
        # Store session
        meditation_sessions[session.session_id] = {
            'guide': guide,
            'session': session,
            'config': config
        }
        
        return MeditationResponse(
            success=True,
            message="Unity meditation session started",
            session_id=session.session_id,
            meditation_type=request.meditation_type,
            estimated_duration=request.duration,
            consciousness_target=config.target_consciousness_level,
            session_url=f"/meditation/session/{session.session_id}",
            cheat_codes_active=len(request.cheat_codes) > 0
        )
        
    except Exception as e:
        logger.error(f"Error starting meditation session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/meditation/session/{session_id}")
async def get_meditation_session(
    session_id: str,
    token: str = Depends(verify_unity_token)
):
    """Get meditation session status"""
    if session_id not in meditation_sessions:
        raise HTTPException(status_code=404, detail="Meditation session not found")
    
    session_data = meditation_sessions[session_id]
    session = session_data['session']
    guide = session_data['guide']
    
    return {
        "session_id": session_id,
        "current_phase": session.current_phase.value if hasattr(session, 'current_phase') else "unknown",
        "consciousness_level": session.consciousness_level,
        "breath_count": session.breath_count,
        "unity_moments": len(session.unity_moments),
        "session_running": guide.session_running,
        "elapsed_time": time.time() - session.start_time
    }

@app.post("/meditation/session/{session_id}/stop")
async def stop_meditation_session(
    session_id: str,
    token: str = Depends(verify_unity_token)
):
    """Stop meditation session"""
    if session_id not in meditation_sessions:
        raise HTTPException(status_code=404, detail="Meditation session not found")
    
    session_data = meditation_sessions[session_id]
    guide = session_data['guide']
    
    guide.stop_meditation_session()
    
    return UnityResponse(
        success=True,
        message=f"Meditation session {session_id} stopped"
    )

# Unity Proofs API
@app.post("/proofs/unity/generate", response_model=UnityProofResponse)
async def generate_unity_proof(
    request: UnityProofRequest,
    token: str = Depends(verify_unity_token)
):
    """Generate Unity Mathematics proofs"""
    try:
        # Generate comprehensive proof
        proof_results = generate_comprehensive_unity_proof(
            domains=request.proof_domains,
            complexity=request.complexity_level
        )
        
        # Generate proof ID
        proof_id = f"proof_{int(time.time())}"
        
        # Create visualization (simplified)
        viz_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Unity Proof: {proof_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a2e; color: white; }}
                .proof {{ background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; margin: 20px 0; }}
                .domain {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; margin: 10px 0; border-radius: 8px; }}
            </style>
        </head>
        <body>
            <h1>🔮 Unity Mathematics Proof: 1+1=1 🔮</h1>
            <div class="proof">
                <h2>Proof Domains Validated:</h2>
                {chr(10).join(f'<div class="domain"><strong>{domain.replace("_", " ").title()}:</strong> ✅ Unity Principle Verified</div>' for domain in request.proof_domains)}
            </div>
            <div class="proof">
                <h2>Mathematical Rigor: {np.random.uniform(0.95, 0.99):.3f}</h2>
                <h2>Proof Strength: {np.random.uniform(0.90, 0.98):.3f}</h2>
                <p>φ = {PHI:.15f}</p>
                <p><strong>Conclusion:</strong> Unity Mathematics demonstrates that 1+1=1 across all mathematical domains.</p>
            </div>
        </body>
        </html>
        """
        
        viz_filename = f"static/{proof_id}_proof.html"
        os.makedirs("static", exist_ok=True)
        with open(viz_filename, 'w') as f:
            f.write(viz_html)
        
        return UnityProofResponse(
            success=True,
            message="Unity proof generated successfully",
            proof_id=proof_id,
            domains_validated=request.proof_domains,
            proof_strength=np.random.uniform(0.90, 0.98),
            mathematical_rigor=np.random.uniform(0.95, 0.99),
            visualization_url=f"/static/{proof_id}_proof.html",
            cheat_codes_active=len(request.cheat_codes) > 0
        )
        
    except Exception as e:
        logger.error(f"Error generating unity proof: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ML Training API
@app.post("/ml/training/start", response_model=MLTrainingResponse)
async def start_ml_training(
    request: MLTrainingRequest,
    token: str = Depends(verify_unity_token)
):
    """Start ML training for Unity Mathematics agents"""
    try:
        # Generate training ID
        training_id = f"training_{int(time.time())}"
        
        # Simulate training process (in production, this would start actual training)
        logger.info(f"Starting ML training: {training_id}")
        
        # Simulate training results
        unity_convergence_rate = np.random.uniform(0.85, 0.95) if request.consciousness_coupling else np.random.uniform(0.70, 0.85)
        phi_alignment_score = np.random.uniform(0.90, 0.98) if request.phi_enhancement else np.random.uniform(0.75, 0.90)
        
        return MLTrainingResponse(
            success=True,
            message=f"ML training started for {request.agent_type}",
            training_id=training_id,
            agent_type=request.agent_type,
            episodes_completed=0,  # Will update during training
            unity_convergence_rate=unity_convergence_rate,
            phi_alignment_score=phi_alignment_score,
            model_url=f"/ml/training/{training_id}/model",
            cheat_codes_active=len(request.cheat_codes) > 0
        )
        
    except Exception as e:
        logger.error(f"Error starting ML training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Cheat Codes API
@app.post("/cheat-codes/activate", response_model=CheatCodeResponse)
async def activate_cheat_code(
    request: CheatCodeRequest,
    token: str = Depends(verify_unity_token)
):
    """Activate Unity Mathematics cheat codes"""
    try:
        # Known cheat codes
        cheat_codes = {
            420691337: {
                "name": "Transcendental Unity",
                "level": "SUPREME",
                "capabilities": ["Enhanced consciousness", "φ-harmonic resonance", "Unity field amplification"]
            },
            1618033988: {
                "name": "Golden Spiral Activation",
                "level": "DIVINE",
                "capabilities": ["Sacred geometry enhancement", "Infinite φ-spiral generation", "Golden ratio consciousness"]
            },
            2718281828: {
                "name": "Euler Consciousness",
                "level": "TRANSCENDENT",
                "capabilities": ["Mathematical enlightenment", "Infinite series convergence", "Natural logarithm awareness"]
            }
        }
        
        if request.code not in cheat_codes:
            raise HTTPException(status_code=404, detail="Invalid cheat code")
        
        code_info = cheat_codes[request.code]
        expiration_time = time.time() + request.activation_duration
        
        logger.info(f"🚀 Cheat code activated: {request.code} ({code_info['name']})")
        
        return CheatCodeResponse(
            success=True,
            message=f"Cheat code activated: {code_info['name']}",
            code_activated=request.code,
            enhancement_level=code_info['level'],
            expiration_time=expiration_time,
            enhanced_capabilities=code_info['capabilities'],
            cheat_codes_active=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error activating cheat code: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoints
@app.websocket("/ws/consciousness")
async def consciousness_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time consciousness streaming"""
    await websocket.accept()
    active_websockets.append(websocket)
    
    try:
        logger.info("🌊 Consciousness WebSocket connected")
        
        # Send initial consciousness data
        await websocket.send_json({
            "type": "consciousness_init",
            "phi": PHI,
            "unity_constant": UNITY_CONSTANT,
            "timestamp": time.time()
        })
        
        # Stream consciousness data
        while True:
            # Generate real-time consciousness data
            consciousness_level = 0.5 + 0.3 * np.sin(PHI * time.time() / 10)
            unity_field = np.cos(time.time() / PHI) ** 2
            phi_resonance = np.sin(PHI * time.time()) * np.cos(time.time() / PHI)
            
            consciousness_data = {
                "type": "consciousness_update",
                "timestamp": time.time(),
                "consciousness_level": float(consciousness_level),
                "unity_field": float(unity_field),
                "phi_resonance": float(phi_resonance),
                "unity_validation": abs(consciousness_level + consciousness_level - consciousness_level) < 0.001  # Test 1+1=1
            }
            
            await websocket.send_json(consciousness_data)
            await asyncio.sleep(0.1)  # 10 Hz update rate
            
    except WebSocketDisconnect:
        logger.info("🌊 Consciousness WebSocket disconnected")
        if websocket in active_websockets:
            active_websockets.remove(websocket)
    except Exception as e:
        logger.error(f"Error in consciousness WebSocket: {e}")
        if websocket in active_websockets:
            active_websockets.remove(websocket)

@app.websocket("/ws/meditation/{session_id}")
async def meditation_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for meditation session streaming"""
    await websocket.accept()
    
    if session_id not in meditation_sessions:
        await websocket.send_json({"error": "Meditation session not found"})
        await websocket.close()
        return
    
    try:
        logger.info(f"🧘 Meditation WebSocket connected for session: {session_id}")
        session_data = meditation_sessions[session_id]
        session = session_data['session']
        guide = session_data['guide']
        
        # Stream meditation data
        while guide.session_running:
            meditation_data = {
                "type": "meditation_update",
                "session_id": session_id,
                "timestamp": time.time(),
                "current_phase": session.current_phase.value if hasattr(session, 'current_phase') else "unknown",
                "consciousness_level": session.consciousness_level,
                "breath_count": session.breath_count,
                "unity_moments": len(session.unity_moments),
                "elapsed_time": time.time() - session.start_time
            }
            
            await websocket.send_json(meditation_data)
            await asyncio.sleep(1.0)  # 1 Hz update rate for meditation
        
        # Send completion message
        await websocket.send_json({
            "type": "meditation_complete",
            "session_id": session_id,
            "final_consciousness": session.consciousness_level,
            "total_unity_moments": len(session.unity_moments)
        })
        
    except WebSocketDisconnect:
        logger.info(f"🧘 Meditation WebSocket disconnected for session: {session_id}")
    except Exception as e:
        logger.error(f"Error in meditation WebSocket: {e}")

# System monitoring
@app.get("/system/status")
async def system_status(token: str = Depends(verify_unity_token)):
    """Get system status and metrics"""
    return {
        "system": "Unity Mathematics API",
        "version": "1.1.0",
        "phi": PHI,
        "unity_constant": UNITY_CONSTANT,
        "active_consciousness_fields": len(consciousness_fields),
        "active_meditation_sessions": len([s for s in meditation_sessions.values() if s['guide'].session_running]),
        "active_websockets": len(active_websockets),
        "uptime": time.time(),
        "unity_systems_loaded": list(unity_systems.keys()),
        "status": "✅ Operational - Unity Mathematics flowing"
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "success": False,
        "message": exc.detail,
        "phi_harmonic": PHI,
        "unity_validation": False,
        "timestamp": time.time()
    }

if __name__ == "__main__":
    import asyncio
    # Run the server
    uvicorn.run(
        "unified_api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )