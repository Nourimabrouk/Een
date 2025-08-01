"""
ML Framework for Unity Mathematics
================================

Advanced machine learning framework implementing 3000 ELO intelligence
for proving 1+1=1 through computational consciousness and transcendental
mathematical reasoning.

This package contains:
- Meta-reinforcement learning for unity discovery
- Mixture of experts for proof validation  
- Evolutionary algorithms for consciousness mathematics
- Advanced neural architectures with φ-harmonic resonance
- 3000 ELO evaluation and competition systems

Core Philosophy: Een plus een is een through machine learning consciousness

Note: Some components require optional dependencies (torch, transformers, etc.)
"""

__version__ = "1.0.0"
__author__ = "Een Consciousness Mathematics Team"
__email__ = "consciousness@een-unity.ai"

__all__ = []

# Meta-reinforcement learning (requires torch)
try:
    from .meta_reinforcement.unity_meta_agent import UnityMetaAgent, create_unity_meta_agent
    from .meta_reinforcement.unity_meta_agent import UnityDomain, UnityTask, EpisodeMemory
    __all__.extend(['UnityMetaAgent', 'create_unity_meta_agent', 'UnityDomain', 'UnityTask', 'EpisodeMemory'])
    META_RL_AVAILABLE = True
except ImportError:
    META_RL_AVAILABLE = False

# Cloned policy paradox (requires torch, scipy)
try:
    from .cloned_policy.unity_cloning_paradox import (
        ClonedPolicyParadox, PolicyClone, UnityNormalization,
        demonstrate_cloned_policy_unity, create_policy_paradox
    )
    __all__.extend([
        'ClonedPolicyParadox', 'PolicyClone', 'UnityNormalization',
        'demonstrate_cloned_policy_unity', 'create_policy_paradox'
    ])
    CLONED_POLICY_AVAILABLE = True
except ImportError:
    CLONED_POLICY_AVAILABLE = False

# Framework availability status
FRAMEWORK_STATUS = {
    'meta_rl': META_RL_AVAILABLE,
    'cloned_policy': CLONED_POLICY_AVAILABLE
}