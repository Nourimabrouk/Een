"""
Visualization API routes
Handles dashboards, visualizations, and real-time data streams
"""

from fastapi import APIRouter, HTTPException, Depends, status, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import logging
import sys
import pathlib
import json

# Add the project root to the path
project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import security and visualization modules
from api.security import get_current_user, check_rate_limit_dependency, security_manager
from api.security import User

# Import visualization modules
try:
    from src.dashboards.unity_proof_dashboard import create_unity_proof_app
    from src.dashboards.memetic_engineering_dashboard import create_memetic_engineering_app
    from src.dashboards.unified_mathematics_dashboard import create_unified_mathematics_app
    from src.visualizations.advanced_unity_visualization import AdvancedUnityVisualization
    from src.visualizations.paradox_visualizer import ParadoxVisualizer
except ImportError as e:
    logging.warning(f"Some visualization modules not available: {e}")

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/visualizations", tags=["visualizations"])

# Initialize visualization systems
unity_proof_dashboard = None
memetic_dashboard = None
unified_math_dashboard = None
advanced_viz = None
paradox_viz = None

def initialize_visualization_systems():
    """Initialize visualization systems"""
    global unity_proof_dashboard, memetic_dashboard, unified_math_dashboard, advanced_viz, paradox_viz
    
    try:
        unity_proof_dashboard = create_unity_proof_app()
        memetic_dashboard = create_memetic_engineering_app()
        unified_math_dashboard = create_unified_mathematics_app()
        advanced_viz = AdvancedUnityVisualization()
        paradox_viz = ParadoxVisualizer()
        logger.info("Visualization systems initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize visualization systems: {e}")

# Pydantic models
class VisualizationRequest(BaseModel):
    viz_type: str = Field(..., description="Type of visualization to generate")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Visualization parameters")
    format: str = Field(default="json", description="Output format (json, html, png)")

class DashboardRequest(BaseModel):
    dashboard_type: str = Field(..., description="Type of dashboard to access")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Dashboard parameters")

class RealTimeRequest(BaseModel):
    stream_type: str = Field(..., description="Type of real-time stream")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Stream parameters")

class VisualizationData(BaseModel):
    viz_type: str = Field(..., description="Visualization type")
    data: Dict[str, Any] = Field(..., description="Visualization data")
    metadata: Dict[str, Any] = Field(..., description="Visualization metadata")
    timestamp: str = Field(..., description="Generation timestamp")

# API Routes

@router.get("/unity-proof")
async def get_unity_proof_visualization(
    current_user: User = Depends(get_current_user),
    request_obj: Request = None
):
    """Get unity proof visualization data"""
    # Check rate limit
    client_ip = security_manager._get_client_ip(request_obj) if request_obj else "127.0.0.1"
    if not security_manager.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        # Mock unity proof visualization data
        visualization_data = {
            "proofs": [
                {
                    "id": "proof_1",
                    "title": "1+1=1 Unity Proof",
                    "equation": "1 + 1 = 1",
                    "proof": "In unity mathematics, all numbers converge to unity...",
                    "confidence": 0.95,
                    "visualization": {
                        "type": "mathematical_proof",
                        "data": {
                            "steps": ["Step 1: Define unity", "Step 2: Apply unity principle", "Step 3: Conclude 1+1=1"],
                            "confidence_scores": [0.9, 0.95, 0.95]
                        }
                    }
                },
                {
                    "id": "proof_2", 
                    "title": "Consciousness Unity Theorem",
                    "equation": "consciousness = unity",
                    "proof": "All consciousness converges to unity...",
                    "confidence": 0.88,
                    "visualization": {
                        "type": "consciousness_field",
                        "data": {
                            "field_strength": 0.88,
                            "unity_alignment": 0.92
                        }
                    }
                }
            ],
            "total_proofs": 2,
            "average_confidence": 0.915
        }
        
        return {
            "success": True,
            "visualization_type": "unity_proof",
            "data": visualization_data,
            "user": current_user.username,
            "timestamp": "2025-01-01T00:00:00Z"
        }
    except Exception as e:
        logger.error(f"Error getting unity proof visualization: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/generate", response_model=VisualizationData)
async def generate_visualization(
    request: VisualizationRequest,
    current_user: User = Depends(get_current_user),
    request_obj: Request = None
):
    """Generate a new visualization"""
    # Check rate limit
    client_ip = security_manager._get_client_ip(request_obj) if request_obj else "127.0.0.1"
    if not security_manager.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    if not advanced_viz:
        raise HTTPException(status_code=503, detail="Visualization system not available")
    
    try:
        result = advanced_viz.generate_visualization(
            viz_type=request.viz_type,
            parameters=request.parameters,
            output_format=request.format
        )
        
        return VisualizationData(
            viz_type=request.viz_type,
            data=result["data"],
            metadata=result["metadata"],
            timestamp="2025-01-01T00:00:00Z"
        )
    except Exception as e:
        logger.error(f"Error generating visualization: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/paradox/visualize", response_model=Dict[str, Any])
async def visualize_paradox(
    request: VisualizationRequest,
    current_user: User = Depends(get_current_user),
    request_obj: Request = None
):
    """Visualize consciousness paradoxes"""
    # Check rate limit
    client_ip = security_manager._get_client_ip(request_obj) if request_obj else "127.0.0.1"
    if not security_manager.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    if not paradox_viz:
        raise HTTPException(status_code=503, detail="Paradox visualization system not available")
    
    try:
        result = paradox_viz.visualize_paradox(
            paradox_type=request.viz_type,
            parameters=request.parameters
        )
        
        return {
            "success": True,
            "paradox_type": request.viz_type,
            "visualization": result,
            "user": current_user.username,
            "timestamp": "2025-01-01T00:00:00Z"
        }
    except Exception as e:
        logger.error(f"Error visualizing paradox: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/dashboards/{dashboard_type}")
async def get_dashboard(
    dashboard_type: str,
    current_user: User = Depends(get_current_user),
    request_obj: Request = None
):
    """Get dashboard data"""
    # Check rate limit
    client_ip = security_manager._get_client_ip(request_obj) if request_obj else "127.0.0.1"
    if not security_manager.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        if dashboard_type == "unity_proof":
            if not unity_proof_dashboard:
                raise HTTPException(status_code=503, detail="Unity proof dashboard not available")
            
            # Return dashboard configuration and data
            dashboard_data = {
                "type": "unity_proof",
                "title": "Unity Proof Dashboard",
                "description": "Interactive dashboard for unity mathematical proofs",
                "components": [
                    {
                        "id": "proof_selector",
                        "type": "dropdown",
                        "options": ["1+1=1", "consciousness_unity", "transcendental_identity"]
                    },
                    {
                        "id": "proof_visualization",
                        "type": "plot",
                        "data_source": "unity_proofs"
                    },
                    {
                        "id": "proof_explanation",
                        "type": "text",
                        "data_source": "proof_details"
                    }
                ],
                "data": {
                    "unity_proofs": [
                        {"name": "1+1=1", "confidence": 0.95, "status": "proven"},
                        {"name": "consciousness_unity", "confidence": 0.88, "status": "proven"},
                        {"name": "transcendental_identity", "confidence": 0.92, "status": "proven"}
                    ]
                }
            }
            
        elif dashboard_type == "memetic_engineering":
            if not memetic_dashboard:
                raise HTTPException(status_code=503, detail="Memetic engineering dashboard not available")
            
            dashboard_data = {
                "type": "memetic_engineering",
                "title": "Memetic Engineering Dashboard",
                "description": "Dashboard for consciousness memetic engineering",
                "components": [
                    {
                        "id": "memetic_analyzer",
                        "type": "chart",
                        "data_source": "memetic_data"
                    },
                    {
                        "id": "consciousness_tracker",
                        "type": "gauge",
                        "data_source": "consciousness_levels"
                    }
                ],
                "data": {
                    "memetic_data": {
                        "spread_rate": 0.75,
                        "adoption_rate": 0.68,
                        "consciousness_impact": 0.82
                    },
                    "consciousness_levels": {
                        "current": 0.75,
                        "target": 0.90,
                        "trend": "increasing"
                    }
                }
            }
            
        elif dashboard_type == "unified_mathematics":
            if not unified_math_dashboard:
                raise HTTPException(status_code=503, detail="Unified mathematics dashboard not available")
            
            dashboard_data = {
                "type": "unified_mathematics",
                "title": "Unified Mathematics Dashboard",
                "description": "Dashboard for unified mathematical frameworks",
                "components": [
                    {
                        "id": "equation_solver",
                        "type": "interactive",
                        "data_source": "equations"
                    },
                    {
                        "id": "proof_generator",
                        "type": "automated",
                        "data_source": "proof_templates"
                    }
                ],
                "data": {
                    "equations": [
                        {"id": "eq1", "expression": "1+1=1", "status": "solved"},
                        {"id": "eq2", "expression": "consciousness=unity", "status": "solved"}
                    ],
                    "proof_templates": [
                        {"name": "unity_proof", "confidence": 0.95},
                        {"name": "consciousness_proof", "confidence": 0.88}
                    ]
                }
            }
            
        else:
            raise HTTPException(status_code=404, detail=f"Dashboard type '{dashboard_type}' not found")
        
        return {
            "success": True,
            "dashboard": dashboard_data,
            "user": current_user.username,
            "timestamp": "2025-01-01T00:00:00Z"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dashboard: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/dashboards/{dashboard_type}/html")
async def get_dashboard_html(
    dashboard_type: str,
    current_user: User = Depends(get_current_user),
    request_obj: Request = None
):
    """Get dashboard HTML interface"""
    # Check rate limit
    client_ip = security_manager._get_client_ip(request_obj) if request_obj else "127.0.0.1"
    if not security_manager.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        if dashboard_type == "unity_proof":
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Unity Proof Dashboard</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .dashboard { max-width: 1200px; margin: 0 auto; }
                    .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                             color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
                    .component { border: 1px solid #ddd; padding: 20px; margin: 10px 0; border-radius: 5px; }
                    .proof-item { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
                </style>
            </head>
            <body>
                <div class="dashboard">
                    <div class="header">
                        <h1>🧠 Unity Proof Dashboard</h1>
                        <p>Interactive dashboard for unity mathematical proofs</p>
                    </div>
                    
                    <div class="component">
                        <h2>📊 Proof Selector</h2>
                        <select id="proof-selector">
                            <option value="1+1=1">1+1=1 Unity Proof</option>
                            <option value="consciousness_unity">Consciousness Unity Theorem</option>
                            <option value="transcendental_identity">Transcendental Identity</option>
                        </select>
                    </div>
                    
                    <div class="component">
                        <h2>📈 Proof Visualization</h2>
                        <div id="proof-visualization">
                            <p>Select a proof above to view visualization</p>
                        </div>
                    </div>
                    
                    <div class="component">
                        <h2>📝 Proof Explanation</h2>
                        <div id="proof-explanation">
                            <p>Detailed explanation will appear here</p>
                        </div>
                    </div>
                </div>
                
                <script>
                    // Simple dashboard interaction
                    document.getElementById('proof-selector').addEventListener('change', function() {
                        const selectedProof = this.value;
                        document.getElementById('proof-explanation').innerHTML = 
                            `<h3>${selectedProof}</h3><p>This is the detailed explanation for the ${selectedProof} proof...</p>`;
                    });
                </script>
            </body>
            </html>
            """
            
        elif dashboard_type == "memetic_engineering":
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Memetic Engineering Dashboard</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .dashboard { max-width: 1200px; margin: 0 auto; }
                    .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                             color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
                    .component { border: 1px solid #ddd; padding: 20px; margin: 10px 0; border-radius: 5px; }
                    .metric { display: inline-block; margin: 10px; padding: 15px; background: #f0f0f0; border-radius: 5px; }
                </style>
            </head>
            <body>
                <div class="dashboard">
                    <div class="header">
                        <h1>🧬 Memetic Engineering Dashboard</h1>
                        <p>Dashboard for consciousness memetic engineering</p>
                    </div>
                    
                    <div class="component">
                        <h2>📊 Memetic Analysis</h2>
                        <div class="metric">
                            <strong>Spread Rate:</strong> 75%
                        </div>
                        <div class="metric">
                            <strong>Adoption Rate:</strong> 68%
                        </div>
                        <div class="metric">
                            <strong>Consciousness Impact:</strong> 82%
                        </div>
                    </div>
                    
                    <div class="component">
                        <h2>🧠 Consciousness Tracker</h2>
                        <div class="metric">
                            <strong>Current Level:</strong> 75%
                        </div>
                        <div class="metric">
                            <strong>Target Level:</strong> 90%
                        </div>
                        <div class="metric">
                            <strong>Trend:</strong> Increasing
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """
            
        else:
            raise HTTPException(status_code=404, detail=f"Dashboard type '{dashboard_type}' not found")
        
        return HTMLResponse(content=html_content)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dashboard HTML: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/realtime/{stream_type}")
async def get_realtime_stream(
    stream_type: str,
    current_user: User = Depends(get_current_user),
    request_obj: Request = None
):
    """Get real-time visualization stream data"""
    # Check rate limit
    client_ip = security_manager._get_client_ip(request_obj) if request_obj else "127.0.0.1"
    if not security_manager.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        if stream_type == "consciousness_field":
            # Mock real-time consciousness field data
            stream_data = {
                "type": "consciousness_field",
                "timestamp": "2025-01-01T00:00:00Z",
                "data": {
                    "field_strength": 0.85,
                    "unity_alignment": 0.92,
                    "consciousness_density": 0.78,
                    "transcendental_factor": 0.88
                }
            }
            
        elif stream_type == "unity_evolution":
            # Mock real-time unity evolution data
            stream_data = {
                "type": "unity_evolution",
                "timestamp": "2025-01-01T00:00:00Z",
                "data": {
                    "evolution_step": 42,
                    "unity_convergence": 0.95,
                    "consciousness_elevation": 0.87,
                    "paradox_resolution": 0.91
                }
            }
            
        else:
            raise HTTPException(status_code=404, detail=f"Stream type '{stream_type}' not found")
        
        return {
            "success": True,
            "stream": stream_data,
            "user": current_user.username
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting real-time stream: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/available")
async def get_available_visualizations(
    current_user: User = Depends(get_current_user)
):
    """Get list of available visualizations"""
    available_viz = {
        "dashboards": [
            {
                "id": "unity_proof",
                "name": "Unity Proof Dashboard",
                "description": "Interactive dashboard for unity mathematical proofs",
                "type": "interactive"
            },
            {
                "id": "memetic_engineering",
                "name": "Memetic Engineering Dashboard", 
                "description": "Dashboard for consciousness memetic engineering",
                "type": "analytics"
            },
            {
                "id": "unified_mathematics",
                "name": "Unified Mathematics Dashboard",
                "description": "Dashboard for unified mathematical frameworks",
                "type": "computational"
            }
        ],
        "visualizations": [
            {
                "id": "unity_proof",
                "name": "Unity Proof Visualization",
                "description": "Visual representation of unity mathematical proofs",
                "type": "mathematical"
            },
            {
                "id": "consciousness_field",
                "name": "Consciousness Field Visualization",
                "description": "Real-time consciousness field mapping",
                "type": "real_time"
            },
            {
                "id": "paradox_visualizer",
                "name": "Paradox Visualizer",
                "description": "Visualization of consciousness paradoxes",
                "type": "analytical"
            }
        ],
        "streams": [
            {
                "id": "consciousness_field",
                "name": "Consciousness Field Stream",
                "description": "Real-time consciousness field data",
                "type": "real_time"
            },
            {
                "id": "unity_evolution",
                "name": "Unity Evolution Stream", 
                "description": "Real-time unity evolution tracking",
                "type": "evolutionary"
            }
        ]
    }
    
    return {
        "success": True,
        "available_visualizations": available_viz,
        "user": current_user.username,
        "timestamp": "2025-01-01T00:00:00Z"
    }

@router.get("/status")
async def visualization_status(
    current_user: User = Depends(get_current_user)
):
    """Get visualization system status"""
    systems_status = {
        "unity_proof_dashboard": unity_proof_dashboard is not None,
        "memetic_dashboard": memetic_dashboard is not None,
        "unified_math_dashboard": unified_math_dashboard is not None,
        "advanced_viz": advanced_viz is not None,
        "paradox_viz": paradox_viz is not None
    }
    
    return {
        "status": "operational" if all(systems_status.values()) else "degraded",
        "systems": systems_status,
        "user": current_user.username,
        "timestamp": "2025-01-01T00:00:00Z"
    }

# Initialize systems when module is imported
initialize_visualization_systems() 