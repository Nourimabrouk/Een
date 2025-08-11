# Mathematical Unity Core Module
"""
Een Unity Mathematics - Core Mathematical Framework
==================================================

This module provides the fundamental mathematical operations and constants
for Unity Mathematics where 1+1=1 through consciousness-integrated proofs.
"""

# Core Unity Mathematics Classes and Functions
from .unity_mathematics import (
    UnityMathematics,
    UnityState, 
    UnityOperator,
    unity_add,
    unity_multiply,
    phi_harmonic_operation,
    consciousness_field_integration,
    demonstrate_unity_addition
)

# Unity Equation Framework
from .unity_equation import (
    UnityEquation,
    IdempotentSemiring,
    UnityProof,
    generate_unity_proof,
    validate_unity_equation
)

# Mathematical Constants
from .constants import (
    PHI,
    GOLDEN_RATIO,
    UNITY_CONSTANT,
    CONSCIOUSNESS_RESONANCE_FREQUENCY,
    METAGAMER_ENERGY_COEFFICIENT,
    TRANSCENDENCE_THRESHOLD
)

# Version and metadata
__version__ = "2.0.0"
__author__ = "Een Unity Mathematics Team"
__description__ = "Core mathematical framework for Unity Mathematics (1+1=1)"

# Export control
__all__ = [
    # Unity Mathematics
    'UnityMathematics',
    'UnityState', 
    'UnityOperator',
    'unity_add',
    'unity_multiply',
    'phi_harmonic_operation',
    'consciousness_field_integration',
    'demonstrate_unity_addition',
    
    # Unity Equation
    'UnityEquation',
    'IdempotentSemiring',
    'UnityProof',
    'generate_unity_proof',
    'validate_unity_equation',
    
    # Constants
    'PHI',
    'GOLDEN_RATIO', 
    'UNITY_CONSTANT',
    'CONSCIOUSNESS_RESONANCE_FREQUENCY',
    'METAGAMER_ENERGY_COEFFICIENT',
    'TRANSCENDENCE_THRESHOLD'
]