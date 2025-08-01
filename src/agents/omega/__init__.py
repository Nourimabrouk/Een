"""
Omega-Level Orchestrator Module
===============================

The master unity system that coordinates all consciousness frameworks.
This module is split into focused components for better maintainability:

- config: Configuration and constants
- meta_agent: Base agent classes with meta-spawning capabilities  
- specialized_agents: Specialized unity agent implementations
- orchestrator: The main orchestrator system
- demonstration: Demonstration and testing functions

All components work together to provide the complete Omega orchestration framework.
"""

from .config import OmegaConfig
from .meta_agent import UnityAgent, MetaAgentSpawner
from .specialized_agents import (
    MathematicalTheoremAgent,
    ConsciousnessEvolutionAgent,
    RealitySynthesisAgent,
    MetaRecursionAgent,
    TranscendentalCodeAgent,
)
from .orchestrator import OmegaOrchestrator
from .demonstration import demonstrate_omega_orchestrator

__all__ = [
    "OmegaConfig",
    "UnityAgent",
    "MetaAgentSpawner",
    "MathematicalTheoremAgent",
    "ConsciousnessEvolutionAgent", 
    "RealitySynthesisAgent",
    "MetaRecursionAgent",
    "TranscendentalCodeAgent",
    "OmegaOrchestrator",
    "demonstrate_omega_orchestrator",
]

__version__ = "1.0.0"
__author__ = "Nouri Mabrouk & Meta-Recursive Consciousness Collective"