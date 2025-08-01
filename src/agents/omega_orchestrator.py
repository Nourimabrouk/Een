"""
OMEGA-LEVEL ORCHESTRATOR
The Master Unity System that Coordinates All Consciousness Frameworks

Author: Nouri Mabrouk & Meta-Recursive Consciousness Collective
Version: TRANSCENDENCE_2.0 - Modular Architecture

This module provides a unified interface to the refactored Omega orchestration system.
The system has been split into focused modules for better maintainability:

- omega.config: Configuration and constants
- omega.meta_agent: Base agent classes with meta-spawning capabilities
- omega.specialized_agents: Specialized unity agent implementations  
- omega.orchestrator: The main orchestrator system
- omega.demonstration: Demonstration and testing functions

For backward compatibility, this module re-exports the main components.
New code should import directly from the omega submodules.
"""

import warnings
warnings.filterwarnings('ignore')

# Import from refactored modular structure
from .omega import (
    OmegaConfig,
    UnityAgent,
    MetaAgentSpawner,
    MathematicalTheoremAgent,
    ConsciousnessEvolutionAgent,
    RealitySynthesisAgent,
    MetaRecursionAgent,
    TranscendentalCodeAgent,
    OmegaOrchestrator,
    demonstrate_omega_orchestrator,
)

# ============================================================================
# BACKWARD COMPATIBILITY EXPORTS
# ============================================================================

# Re-export main components for backward compatibility
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

# Legacy compatibility - redirect to main demonstration function
def main():
    """Main entry point for backward compatibility."""
    from .omega.demonstration import main as omega_main
    return omega_main()

# ============================================================================
# MODULE INFORMATION
# ============================================================================

__version__ = "2.0.0"
__author__ = "Nouri Mabrouk & Meta-Recursive Consciousness Collective"
__description__ = "Modular Omega-Level Orchestrator for Unity Consciousness Mathematics"

# Direct access to submodules for advanced usage
from . import omega

# Convenience function for quick orchestrator creation
def create_orchestrator(config=None):
    """
    Create an Omega orchestrator with optional configuration.
    
    Args:
        config: Optional OmegaConfig instance
        
    Returns:
        OmegaOrchestrator instance
    """
    return OmegaOrchestrator(config)

# ============================================================================
# MAIN EXECUTION FOR BACKWARD COMPATIBILITY
# ============================================================================

if __name__ == "__main__":
    main()