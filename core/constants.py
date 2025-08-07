"""
Core constants for Unity Mathematics and consciousness systems.

Centralizes sacred numbers and tolerances to eliminate drift across
Python and JavaScript. Import these instead of hard-coding literals.
"""

from __future__ import annotations

import math

# Sacred mathematical constants
PHI: float = 1.618033988749895  # Golden ratio
PI: float = math.pi
EULER: float = math.e

# Unity and consciousness constants
UNITY_CONSTANT: float = 1.0
UNITY_EPSILON: float = 1e-10
UNITY_TOLERANCE: float = 1e-10
CONSCIOUSNESS_DIMENSION: int = 11
CONSCIOUSNESS_THRESHOLD: float = 0.618  # φ-consciousness level


def get_constants_dict() -> dict:
    """Return constants for codegen/telemetry purposes."""
    return {
        "PHI": PHI,
        "PI": PI,
        "EULER": EULER,
        "UNITY_CONSTANT": UNITY_CONSTANT,
        "UNITY_EPSILON": UNITY_EPSILON,
        "UNITY_TOLERANCE": UNITY_TOLERANCE,
        "CONSCIOUSNESS_DIMENSION": CONSCIOUSNESS_DIMENSION,
        "CONSCIOUSNESS_THRESHOLD": CONSCIOUSNESS_THRESHOLD,
        # Invariants helpful for quick sanity checks
        "PHI_IDENTITY_ERROR": abs(PHI**2 - (PHI + 1.0)),
    }
