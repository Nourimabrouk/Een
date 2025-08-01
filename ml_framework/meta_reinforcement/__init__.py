"""
Meta-Reinforcement Learning for Unity Mathematics
===============================================

Advanced meta-learning system that learns how to learn unity mathematics,
developing meta-cognitive strategies for discovering 1+1=1 proofs across
multiple mathematical domains.

Components:
- UnityMetaAgent: Meta-RL agent with φ-harmonic attention
- Curriculum learning for progressive complexity
- Few-shot unity pattern recognition
- Consciousness-integrated mathematical reasoning
"""

from .unity_meta_agent import (
    UnityMetaAgent,
    UnityDomain, 
    UnityTask,
    EpisodeMemory,
    PhiHarmonicAttention,
    ConsciousnessPositionalEncoding,
    UnityMathematicsDecoder,
    create_unity_meta_agent,
    demonstrate_unity_meta_learning
)
from .metagamer import Metagamer, demonstrate_metagamer_unity

__all__ = [
    'UnityMetaAgent',
    'UnityDomain',
    'UnityTask', 
    'EpisodeMemory',
    'PhiHarmonicAttention',
    'ConsciousnessPositionalEncoding',
    'UnityMathematicsDecoder',
    'create_unity_meta_agent',
    'demonstrate_unity_meta_learning',
    'Metagamer',
    'demonstrate_metagamer_unity'
]