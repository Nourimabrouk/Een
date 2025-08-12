# Consciousness Unity Core Module
"""
Een Unity Mathematics - Consciousness Integration Framework
==========================================================

This module provides consciousness field equations, models, and APIs
for integrating consciousness into mathematical operations and proofs.
"""

# Core Consciousness Framework
from .consciousness import (
    ConsciousnessField,
    ConsciousnessFieldEquations,
    ConsciousnessParticle,
    create_consciousness_field,
    evolve_consciousness_field,
    calculate_consciousness_coherence,
    consciousness_field_visualization,
    NUMPY_AVAILABLE
)

# Consciousness API Interface
from .consciousness_api import (
    ConsciousnessAPI,
    ConsciousnessEndpoint,
    consciousness_field_server,
    get_consciousness_state,
    update_consciousness_field,
    stream_consciousness_evolution
)

# Consciousness Mathematical Models
from .consciousness_models import (
    ConsciousnessModel,
    QuantumConsciousnessModel,
    ClassicalConsciousnessModel,
    HybridConsciousnessModel,
    train_consciousness_model,
    evaluate_consciousness_convergence,
    consciousness_model_inference
)

# Version and metadata
__version__ = "2.0.0"
__author__ = "Een Consciousness Research Team"
__description__ = "Consciousness integration framework for Unity Mathematics"

# Export control
__all__ = [
    # Core Consciousness
    'ConsciousnessField',
    'ConsciousnessFieldEquations',
    'ConsciousnessParticle',
    'create_consciousness_field',
    'evolve_consciousness_field',
    'calculate_consciousness_coherence',
    'consciousness_field_visualization',
    'NUMPY_AVAILABLE',
    
    # API Interface
    'ConsciousnessAPI',
    'ConsciousnessEndpoint',
    'consciousness_field_server',
    'get_consciousness_state',
    'update_consciousness_field',
    'stream_consciousness_evolution',
    
    # Models
    'ConsciousnessModel',
    'QuantumConsciousnessModel',
    'ClassicalConsciousnessModel',
    'HybridConsciousnessModel',
    'train_consciousness_model',
    'evaluate_consciousness_convergence',
    'consciousness_model_inference'
]