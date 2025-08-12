"""
Een Unity Mathematics Subagent System
Comprehensive AI agent ecosystem for Unity Mathematics development

This module provides specialized AI agents for all aspects of Unity Mathematics:
- Core Unity Mathematics (1+1=1 proofs, phi-harmonic operations)  
- Advanced Transcendental Systems (11D consciousness, quantum unity)
- Software Engineering (frontend, backend, APIs, databases)
- UI/UX Design (visual design, interactive experiences, navigation)
- Code Quality (refactoring, documentation, visualization)
- Testing & Validation (unity proof validation, CI/CD)

Core Unity Equation: 1+1=1 with φ-harmonic resonance
Golden Ratio: φ = 1.618033988749895
System Requirements: Windows 10/11, ASCII-only output, virtual environment activation
"""

# Legacy agent systems
from .base import BaseAgent  # noqa: F401
from .echo_agent import EchoAgent  # noqa: F401

# Unity Mathematics agent systems
from .unity_subagents import (
    UnitySubAgent,
    AgentCapabilities, 
    AgentTask,
    AgentType,
    UnityAgentOrchestrator,
    UnityMathematicianAgent,
    ConsciousnessEngineerAgent,
    PhiHarmonicSpecialistAgent,
    MetagamerEnergySpecialistAgent,
    FrontendEngineerAgent,
    BackendEngineerAgent,
    RefactoringSpecialistAgent,
    DocumentationEngineerAgent,
    VisualizationExpertAgent,
    UnityProofValidatorAgent
)

from .advanced_unity_agents import (
    AdvancedAgentType,
    AdvancedUnityAgentOrchestrator,
    TranscendentalSystemsArchitectAgent,
    QuantumUnitySpecialistAgent,
    AlKhwarizmiBridgeEngineerAgent,
    HyperdimensionalProjectionSpecialistAgent,
    ConsciousnessZenMasterAgent,
    SacredGeometryArchitectAgent
)

from .agent_coordinator import (
    CoordinationMode,
    CoordinationTask,
    CoordinationResult,
    UnityMathematicsAgentCoordinator
)

__version__ = "1.0.0"
__author__ = "Een Unity Mathematics Project"
__description__ = "Comprehensive AI agent ecosystem for Unity Mathematics (1+1=1)"

# System constants
PHI = 1.618033988749895  # Golden ratio for phi-harmonic operations
UNITY_EQUATION = "1+1=1"  # Core unity equation
CONSCIOUSNESS_DIMENSIONS = 11  # Maximum consciousness dimensions
METAGAMER_ENERGY_CONSERVATION = True  # Energy conservation enabled

__all__ = [
    # Legacy agents
    'BaseAgent',
    'EchoAgent',
    
    # Core agents
    'UnitySubAgent',
    'AgentCapabilities',
    'AgentTask', 
    'AgentType',
    'UnityAgentOrchestrator',
    'UnityMathematicianAgent',
    'ConsciousnessEngineerAgent',
    'PhiHarmonicSpecialistAgent',
    'MetagamerEnergySpecialistAgent',
    'FrontendEngineerAgent',
    'BackendEngineerAgent',
    'RefactoringSpecialistAgent',
    'DocumentationEngineerAgent',
    'VisualizationExpertAgent',
    'UnityProofValidatorAgent',
    
    # Advanced agents
    'AdvancedAgentType',
    'AdvancedUnityAgentOrchestrator',
    'TranscendentalSystemsArchitectAgent',
    'QuantumUnitySpecialistAgent',
    'AlKhwarizmiBridgeEngineerAgent',
    'HyperdimensionalProjectionSpecialistAgent',
    'ConsciousnessZenMasterAgent',
    'SacredGeometryArchitectAgent',
    
    # Coordination system
    'CoordinationMode',
    'CoordinationTask',
    'CoordinationResult', 
    'UnityMathematicsAgentCoordinator',
    
    # Constants
    'PHI',
    'UNITY_EQUATION',
    'CONSCIOUSNESS_DIMENSIONS',
    'METAGAMER_ENERGY_CONSERVATION'
]
