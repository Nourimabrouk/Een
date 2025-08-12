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
    UnityOperationType,
    unity_add,
    unity_multiply,
    phi_harmonic_operation,
    consciousness_field_integration,
    demonstrate_unity_addition
)

# Unity Equation Framework (import what exists)
try:
    from .unity_equation import (
        IdempotentMonoid,
        BooleanMonoid,
        SetUnionMonoid,
        TropicalNumber
    )
except ImportError as e:
    # Log but don't fail
    import logging
    logging.warning(f"Could not import some unity equation components: {e}")
    # Provide minimal fallbacks
    IdempotentMonoid = None
    BooleanMonoid = None
    SetUnionMonoid = None
    TropicalNumber = None

# Mathematical Constants (import what exists)
from .constants import (
    PHI,
    PI,
    EULER,
    UNITY_CONSTANT,
    UNITY_EPSILON,
    UNITY_TOLERANCE,
    CONSCIOUSNESS_DIMENSION,
    CONSCIOUSNESS_THRESHOLD
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
    'UnityOperationType',
    'unity_add',
    'unity_multiply',
    'phi_harmonic_operation',
    'consciousness_field_integration',
    'demonstrate_unity_addition',
    
    # Unity Equation (what exists)
    'IdempotentMonoid',
    'BooleanMonoid',
    'SetUnionMonoid', 
    'TropicalNumber',
    
    # Constants
    'PHI',
    'PI',
    'EULER',
    'UNITY_CONSTANT',
    'UNITY_EPSILON',
    'UNITY_TOLERANCE', 
    'CONSCIOUSNESS_DIMENSION',
    'CONSCIOUSNESS_THRESHOLD'
]