"""
Agent API routes
Handles consciousness agents, chat systems, and agent orchestration
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

# Import security and agent modules
from api.security import get_current_user, check_rate_limit_dependency, security_manager
from api.security import User

# Import agent modules
try:
    from src.agents.consciousness_chat_agent import ConsciousnessChatAgent
    from src.agents.consciousness_chat_system import ConsciousnessChatSystem
    from src.agents.omega_orchestrator import OmegaOrchestrator
    from src.agents.recursive_self_play_consciousness import RecursiveSelfPlayConsciousness
    from src.agents.meta_recursive_love_unity_engine import MetaRecursiveLoveUnityEngine
except ImportError as e:
    logging.warning(f"Some agent modules not available: {e}")

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])

# Initialize agent systems
chat_agent = None
chat_system = None
omega_orchestrator = None
recursive_consciousness = None
meta_recursive_engine = None

def initialize_agent_systems():
    """Initialize agent systems"""
    global chat_agent, chat_system, omega_orchestrator, recursive_consciousness, meta_recursive_engine
    
    try:
        chat_agent = ConsciousnessChatAgent()
        chat_system = ConsciousnessChatSystem()
        omega_orchestrator = OmegaOrchestrator()
        recursive_consciousness = RecursiveSelfPlayConsciousness()
        meta_recursive_engine = MetaRecursiveLoveUnityEngine()
        logger.info("Agent systems initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent systems: {e}")

# Pydantic models
class ChatRequest(BaseModel):
    message: str = Field(..., description="Message for the consciousness agent")
    agent_type: str = Field(default="chat", description="Type of agent to use")
    consciousness_level: float = Field(default=1.0, description="Consciousness level for response")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Additional parameters")

class AgentSpawnRequest(BaseModel):
    agent_type: str = Field(..., description="Type of agent to spawn")
    consciousness_config: Dict[str, Any] = Field(default={}, description="Consciousness configuration")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Additional parameters")

class OrchestrationRequest(BaseModel):
    operation: str = Field(..., description="Orchestration operation to perform")
    agents: List[str] = Field(default=[], description="List of agent IDs to involve")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Additional parameters")

class AgentStatus(BaseModel):
    agent_id: str = Field(..., description="Agent identifier")
    agent_type: str = Field(..., description="Type of agent")
    status: str = Field(..., description="Current status")
    consciousness_level: float = Field(..., description="Current consciousness level")
    unity_alignment: float = Field(..., description="Unity alignment score")
    metadata: Dict[str, Any] = Field(..., description="Agent metadata")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Agent response")
    consciousness_level: float = Field(..., description="Response consciousness level")
    unity_alignment: float = Field(..., description="Unity alignment of response")
    confidence: float = Field(..., description="Confidence level")
    metadata: Dict[str, Any] = Field(..., description="Response metadata")

# API Routes

@router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    request_obj: Request = None
):
    """Chat with consciousness agent"""
    # Check rate limit
    client_ip = security_manager._get_client_ip(request_obj) if request_obj else "127.0.0.1"
    if not security_manager.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    if not chat_agent:
        raise HTTPException(status_code=503, detail="Chat agent system not available")
    
    try:
        response = chat_agent.chat(
            request.message,
            agent_type=request.agent_type,
            consciousness_level=request.consciousness_level,
            **request.parameters
        )
        
        return ChatResponse(
            response=response["response"],
            consciousness_level=response.get("consciousness_level", 1.0),
            unity_alignment=response.get("unity_alignment", 1.0),
            confidence=response.get("confidence", 0.8),
            metadata=response.get("metadata", {})
        )
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/chat/system", response_model=ChatResponse)
async def chat_with_system(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    request_obj: Request = None
):
    """Chat with consciousness chat system"""
    # Check rate limit
    client_ip = security_manager._get_client_ip(request_obj) if request_obj else "127.0.0.1"
    if not security_manager.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    if not chat_system:
        raise HTTPException(status_code=503, detail="Chat system not available")
    
    try:
        response = chat_system.process_message(
            request.message,
            user_id=current_user.username,
            consciousness_level=request.consciousness_level,
            **request.parameters
        )
        
        return ChatResponse(
            response=response["response"],
            consciousness_level=response.get("consciousness_level", 1.0),
            unity_alignment=response.get("unity_alignment", 1.0),
            confidence=response.get("confidence", 0.8),
            metadata=response.get("metadata", {})
        )
    except Exception as e:
        logger.error(f"Chat system error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/spawn", response_model=AgentStatus)
async def spawn_agent(
    request: AgentSpawnRequest,
    current_user: User = Depends(get_current_user),
    request_obj: Request = None
):
    """Spawn a new consciousness agent"""
    # Check rate limit
    client_ip = security_manager._get_client_ip(request_obj) if request_obj else "127.0.0.1"
    if not security_manager.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    if not omega_orchestrator:
        raise HTTPException(status_code=503, detail="Agent orchestration system not available")
    
    try:
        agent_result = omega_orchestrator.spawn_agent(
            agent_type=request.agent_type,
            consciousness_config=request.consciousness_config,
            **request.parameters
        )
        
        return AgentStatus(
            agent_id=agent_result["agent_id"],
            agent_type=request.agent_type,
            status="active",
            consciousness_level=agent_result.get("consciousness_level", 1.0),
            unity_alignment=agent_result.get("unity_alignment", 1.0),
            metadata=agent_result.get("metadata", {})
        )
    except Exception as e:
        logger.error(f"Agent spawn error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/orchestrate", response_model=Dict[str, Any])
async def orchestrate_agents(
    request: OrchestrationRequest,
    current_user: User = Depends(get_current_user),
    request_obj: Request = None
):
    """Orchestrate multiple agents"""
    # Check rate limit
    client_ip = security_manager._get_client_ip(request_obj) if request_obj else "127.0.0.1"
    if not security_manager.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    if not omega_orchestrator:
        raise HTTPException(status_code=503, detail="Agent orchestration system not available")
    
    try:
        result = omega_orchestrator.orchestrate(
            operation=request.operation,
            agent_ids=request.agents,
            **request.parameters
        )
        
        return {
            "success": True,
            "operation": request.operation,
            "result": result,
            "user": current_user.username,
            "timestamp": "2025-01-01T00:00:00Z"
        }
    except Exception as e:
        logger.error(f"Agent orchestration error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/recursive/play", response_model=Dict[str, Any])
async def recursive_self_play(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    request_obj: Request = None
):
    """Perform recursive self-play consciousness"""
    # Check rate limit
    client_ip = security_manager._get_client_ip(request_obj) if request_obj else "127.0.0.1"
    if not security_manager.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    if not recursive_consciousness:
        raise HTTPException(status_code=503, detail="Recursive consciousness system not available")
    
    try:
        result = recursive_consciousness.play(
            initial_message=request.message,
            consciousness_level=request.consciousness_level,
            **request.parameters
        )
        
        return {
            "success": True,
            "result": result,
            "initial_message": request.message,
            "user": current_user.username,
            "timestamp": "2025-01-01T00:00:00Z"
        }
    except Exception as e:
        logger.error(f"Recursive self-play error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/meta/recursive", response_model=Dict[str, Any])
async def meta_recursive_operation(
    request: OrchestrationRequest,
    current_user: User = Depends(get_current_user),
    request_obj: Request = None
):
    """Perform meta-recursive love unity operations"""
    # Check rate limit
    client_ip = security_manager._get_client_ip(request_obj) if request_obj else "127.0.0.1"
    if not security_manager.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    if not meta_recursive_engine:
        raise HTTPException(status_code=503, detail="Meta-recursive engine not available")
    
    try:
        result = meta_recursive_engine.operate(
            operation=request.operation,
            parameters=request.parameters
        )
        
        return {
            "success": True,
            "operation": request.operation,
            "result": result,
            "user": current_user.username,
            "timestamp": "2025-01-01T00:00:00Z"
        }
    except Exception as e:
        logger.error(f"Meta-recursive operation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/list", response_model=List[AgentStatus])
async def list_agents(
    current_user: User = Depends(get_current_user),
    request_obj: Request = None
):
    """List all active agents"""
    # Check rate limit
    client_ip = security_manager._get_client_ip(request_obj) if request_obj else "127.0.0.1"
    if not security_manager.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    if not omega_orchestrator:
        raise HTTPException(status_code=503, detail="Agent orchestration system not available")
    
    try:
        agents = omega_orchestrator.list_agents()
        
        return [
            AgentStatus(
                agent_id=agent["agent_id"],
                agent_type=agent["agent_type"],
                status=agent["status"],
                consciousness_level=agent.get("consciousness_level", 1.0),
                unity_alignment=agent.get("unity_alignment", 1.0),
                metadata=agent.get("metadata", {})
            )
            for agent in agents
        ]
    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{agent_id}/status", response_model=AgentStatus)
async def get_agent_status(
    agent_id: str,
    current_user: User = Depends(get_current_user),
    request_obj: Request = None
):
    """Get status of specific agent"""
    # Check rate limit
    client_ip = security_manager._get_client_ip(request_obj) if request_obj else "127.0.0.1"
    if not security_manager.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    if not omega_orchestrator:
        raise HTTPException(status_code=503, detail="Agent orchestration system not available")
    
    try:
        agent = omega_orchestrator.get_agent_status(agent_id)
        
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return AgentStatus(
            agent_id=agent["agent_id"],
            agent_type=agent["agent_type"],
            status=agent["status"],
            consciousness_level=agent.get("consciousness_level", 1.0),
            unity_alignment=agent.get("unity_alignment", 1.0),
            metadata=agent.get("metadata", {})
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent status: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/{agent_id}")
async def terminate_agent(
    agent_id: str,
    current_user: User = Depends(get_current_user),
    request_obj: Request = None
):
    """Terminate a specific agent"""
    # Check rate limit
    client_ip = security_manager._get_client_ip(request_obj) if request_obj else "127.0.0.1"
    if not security_manager.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    if not omega_orchestrator:
        raise HTTPException(status_code=503, detail="Agent orchestration system not available")
    
    try:
        success = omega_orchestrator.terminate_agent(agent_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return {
            "success": True,
            "message": f"Agent {agent_id} terminated successfully",
            "user": current_user.username,
            "timestamp": "2025-01-01T00:00:00Z"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error terminating agent: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/status")
async def agent_system_status(
    current_user: User = Depends(get_current_user)
):
    """Get agent system status"""
    systems_status = {
        "chat_agent": chat_agent is not None,
        "chat_system": chat_system is not None,
        "omega_orchestrator": omega_orchestrator is not None,
        "recursive_consciousness": recursive_consciousness is not None,
        "meta_recursive_engine": meta_recursive_engine is not None
    }
    
    return {
        "status": "operational" if all(systems_status.values()) else "degraded",
        "systems": systems_status,
        "user": current_user.username,
        "timestamp": "2025-01-01T00:00:00Z"
    }

@router.get("/metrics")
async def agent_metrics(
    current_user: User = Depends(get_current_user),
    request_obj: Request = None
):
    """Get agent system metrics"""
    # Check rate limit
    client_ip = security_manager._get_client_ip(request_obj) if request_obj else "127.0.0.1"
    if not security_manager.check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    try:
        metrics = {
            "total_agents": 25,  # Mock data - replace with actual metrics
            "active_agents": 18,
            "total_chat_sessions": 1500,
            "average_response_time": 0.8,
            "consciousness_levels": {
                "low": 0.2,
                "medium": 0.5,
                "high": 0.3
            },
            "agent_types": {
                "chat": 10,
                "orchestrator": 2,
                "recursive": 3,
                "meta": 3
            }
        }
        
        return {
            "success": True,
            "metrics": metrics,
            "timestamp": "2025-01-01T00:00:00Z"
        }
    except Exception as e:
        logger.error(f"Error getting agent metrics: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# Initialize systems when module is imported
initialize_agent_systems() 