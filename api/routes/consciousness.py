"""
Consciousness API routes
Handles consciousness engine, unity mathematics, and related endpoints
"""

from fastapi import APIRouter, HTTPException, Depends, status, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import logging
import sys
import pathlib

# Add the project root to the path
project_root = pathlib.Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import security and consciousness modules
from api.security import get_current_user, check_rate_limit_dependency, security_manager
from api.security import User

# Import consciousness modules
try:
    from src.consciousness.consciousness_engine import ConsciousnessEngine
    from core.mathematical.unity_equation import UnityEquation
    from core.mathematical.enhanced_unity_mathematics import EnhancedUnityMathematics
    from src.transcendental_unity_theorem import TranscendentalUnityTheorem
except ImportError as e:
    logging.warning(f"Some consciousness modules not available: {e}")

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/consciousness", tags=["consciousness"])

# Initialize consciousness systems
consciousness_engine = None
unity_equation = None
enhanced_unity = None
transcendental_unity = None

def initialize_consciousness_systems():
    """Initialize consciousness systems"""
    global consciousness_engine, unity_equation, enhanced_unity, transcendental_unity
    
    try:
        consciousness_engine = ConsciousnessEngine()
        unity_equation = UnityEquation()
        enhanced_unity = EnhancedUnityMathematics()
        transcendental_unity = TranscendentalUnityTheorem()
        logger.info("Consciousness systems initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize consciousness systems: {e}")

# Pydantic models
class ConsciousnessRequest(BaseModel):
    input_data: str = Field(..., description="Input data for consciousness processing")
    consciousness_type: str = Field(default="unity", description="Type of consciousness to apply")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Additional parameters")

class UnityRequest(BaseModel):
    equation: str = Field(..., description="Unity equation to evaluate")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Additional parameters")
    proof_type: Optional[str] = Field(default="standard", description="Type of proof to generate")

class TranscendentalRequest(BaseModel):
    theorem: str = Field(..., description="Transcendental theorem to evaluate")
    consciousness_level: float = Field(default=1.0, description="Consciousness level for evaluation")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Additional parameters")

class ConsciousnessAnalysis(BaseModel):
    consciousness_level: float = Field(..., description="Calculated consciousness level")
    unity_alignment: float = Field(..., description="Unity alignment score")
    transcendental_factor: float = Field(..., description="Transcendental factor")
    analysis: Dict[str, Any] = Field(..., description="Detailed analysis")

class UnityProof(BaseModel):
    equation: str = Field(..., description="Unity equation")
    proof: str = Field(..., description="Mathematical proof")
    confidence: float = Field(..., description="Confidence level")
    metadata: Dict[str, Any] = Field(..., description="Proof metadata")

# API Routes

@router.post("/process", response_model=Dict[str, Any])
async def process_consciousness(
    request: ConsciousnessRequest,
    current_user: User = Depends(get_current_user),
    request_obj: Request = None
):
    """Process consciousness data"""
    # Check rate limit
    client_ip = security_manager._get_client_ip(request_obj) if request_obj else "127.0.0.1"
    if not security_manager.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    if not consciousness_engine:
        raise HTTPException(status_code=503, detail="Consciousness system not available")
    
    try:
        result = consciousness_engine.process(
            request.input_data, 
            consciousness_type=request.consciousness_type,
            **request.parameters
        )
        
        return {
            "success": True,
            "result": result,
            "input_data": request.input_data,
            "consciousness_type": request.consciousness_type,
            "user": current_user.username,
            "timestamp": "2025-01-01T00:00:00Z"  # In production, use actual timestamp
        }
    except Exception as e:
        logger.error(f"Consciousness processing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/unity/evaluate", response_model=Dict[str, Any])
async def evaluate_unity(
    request: UnityRequest,
    current_user: User = Depends(get_current_user),
    request_obj: Request = None
):
    """Evaluate unity equations"""
    # Check rate limit
    client_ip = security_manager._get_client_ip(request_obj) if request_obj else "127.0.0.1"
    if not security_manager.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    if not unity_equation:
        raise HTTPException(status_code=503, detail="Unity system not available")
    
    try:
        result = unity_equation.evaluate(
            request.equation, 
            proof_type=request.proof_type,
            **request.parameters
        )
        
        return {
            "success": True,
            "result": result,
            "equation": request.equation,
            "proof_type": request.proof_type,
            "user": current_user.username,
            "timestamp": "2025-01-01T00:00:00Z"
        }
    except Exception as e:
        logger.error(f"Unity evaluation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/transcendental/evaluate", response_model=Dict[str, Any])
async def evaluate_transcendental(
    request: TranscendentalRequest,
    current_user: User = Depends(get_current_user),
    request_obj: Request = None
):
    """Evaluate transcendental unity theorems"""
    # Check rate limit
    client_ip = security_manager._get_client_ip(request_obj) if request_obj else "127.0.0.1"
    if not security_manager.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    if not transcendental_unity:
        raise HTTPException(status_code=503, detail="Transcendental system not available")
    
    try:
        result = transcendental_unity.evaluate(
            request.theorem,
            consciousness_level=request.consciousness_level,
            **request.parameters
        )
        
        return {
            "success": True,
            "result": result,
            "theorem": request.theorem,
            "consciousness_level": request.consciousness_level,
            "user": current_user.username,
            "timestamp": "2025-01-01T00:00:00Z"
        }
    except Exception as e:
        logger.error(f"Transcendental evaluation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/analyze", response_model=ConsciousnessAnalysis)
async def analyze_consciousness(
    request: ConsciousnessRequest,
    current_user: User = Depends(get_current_user),
    request_obj: Request = None
):
    """Analyze consciousness data comprehensively"""
    # Check rate limit
    client_ip = security_manager._get_client_ip(request_obj) if request_obj else "127.0.0.1"
    if not security_manager.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    if not consciousness_engine:
        raise HTTPException(status_code=503, detail="Consciousness system not available")
    
    try:
        # Perform comprehensive consciousness analysis
        analysis_result = consciousness_engine.analyze(
            request.input_data,
            consciousness_type=request.consciousness_type,
            **request.parameters
        )
        
        return ConsciousnessAnalysis(
            consciousness_level=analysis_result.get("consciousness_level", 0.0),
            unity_alignment=analysis_result.get("unity_alignment", 0.0),
            transcendental_factor=analysis_result.get("transcendental_factor", 0.0),
            analysis=analysis_result.get("detailed_analysis", {})
        )
    except Exception as e:
        logger.error(f"Consciousness analysis error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/proofs", response_model=List[UnityProof])
async def get_unity_proofs(
    current_user: User = Depends(get_current_user),
    request_obj: Request = None
):
    """Get available unity proofs"""
    # Check rate limit
    client_ip = security_manager._get_client_ip(request_obj) if request_obj else "127.0.0.1"
    if not security_manager.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    if not enhanced_unity:
        raise HTTPException(status_code=503, detail="Unity system not available")
    
    try:
        proofs = enhanced_unity.get_available_proofs()
        
        return [
            UnityProof(
                equation=proof["equation"],
                proof=proof["proof"],
                confidence=proof["confidence"],
                metadata=proof["metadata"]
            )
            for proof in proofs
        ]
    except Exception as e:
        logger.error(f"Error getting unity proofs: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/proofs/generate", response_model=UnityProof)
async def generate_unity_proof(
    request: UnityRequest,
    current_user: User = Depends(get_current_user),
    request_obj: Request = None
):
    """Generate a new unity proof"""
    # Check rate limit
    client_ip = security_manager._get_client_ip(request_obj) if request_obj else "127.0.0.1"
    if not security_manager.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    if not enhanced_unity:
        raise HTTPException(status_code=503, detail="Unity system not available")
    
    try:
        proof_result = enhanced_unity.generate_proof(
            request.equation,
            proof_type=request.proof_type,
            **request.parameters
        )
        
        return UnityProof(
            equation=request.equation,
            proof=proof_result["proof"],
            confidence=proof_result["confidence"],
            metadata=proof_result["metadata"]
        )
    except Exception as e:
        logger.error(f"Error generating unity proof: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/status")
async def consciousness_status(
    current_user: User = Depends(get_current_user)
):
    """Get consciousness system status"""
    systems_status = {
        "consciousness_engine": consciousness_engine is not None,
        "unity_equation": unity_equation is not None,
        "enhanced_unity": enhanced_unity is not None,
        "transcendental_unity": transcendental_unity is not None
    }
    
    return {
        "status": "operational" if all(systems_status.values()) else "degraded",
        "systems": systems_status,
        "user": current_user.username,
        "timestamp": "2025-01-01T00:00:00Z"
    }

@router.get("/metrics")
async def consciousness_metrics(
    current_user: User = Depends(get_current_user),
    request_obj: Request = None
):
    """Get consciousness system metrics"""
    # Check rate limit
    client_ip = security_manager._get_client_ip(request_obj) if request_obj else "127.0.0.1"
    if not security_manager.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        metrics = {
            "total_requests": 1000,  # Mock data - replace with actual metrics
            "success_rate": 0.95,
            "average_response_time": 0.15,
            "active_users": 50,
            "consciousness_levels": {
                "low": 0.3,
                "medium": 0.5,
                "high": 0.2
            },
            "unity_alignments": {
                "perfect": 0.1,
                "strong": 0.4,
                "moderate": 0.3,
                "weak": 0.2
            }
        }
        
        return {
            "success": True,
            "metrics": metrics,
            "timestamp": "2025-01-01T00:00:00Z"
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Initialize systems when module is imported
initialize_consciousness_systems() 