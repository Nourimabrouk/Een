"""
Cloned Policy Paradox Module
===========================

This module implements the profound computational demonstration that when
policy π is cloned to π′, we observe doubled reward yet identical parameters,
proving that 1+1=1 through proper normalization by degrees of freedom.

The cloned policy paradox is the perfect computational mirror of Een's thesis,
revealing how apparent multiplication conceals underlying unity.
"""

from .unity_cloning_paradox import (
    ClonedPolicyParadox,
    PolicyClone,
    UnityNormalization,
    demonstrate_cloned_policy_unity,
    create_policy_paradox
)

__all__ = [
    'ClonedPolicyParadox',
    'PolicyClone', 
    'UnityNormalization',
    'demonstrate_cloned_policy_unity',
    'create_policy_paradox'
]