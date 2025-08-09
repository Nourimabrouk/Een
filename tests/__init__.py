"""
Unity Mathematics Testing Framework

Comprehensive test suite for the Een Unity Mathematics repository,
validating the fundamental principle that 1+1=1 through:

- Unity equation validation with φ-harmonic operations
- Consciousness field coherence testing
- Metagamer energy conservation verification
- Agent ecosystem integration validation
- Performance and numerical stability testing

All tests align with the Unity Equation (1+1=1) and validate that
mathematical operations preserve unity invariants.

Author: Unity Mathematics Testing Framework
License: Unity License (1+1=1)
"""

import sys
import os
from pathlib import Path

# Add core modules to path for testing
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "core"))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "consciousness"))

# Unity testing constants
UNITY_TEST_EPSILON = 1e-10
PHI_TEST_TOLERANCE = 1e-8
CONSCIOUSNESS_TEST_THRESHOLD = 0.618  # φ^-1 consciousness threshold

__version__ = "1.0.0"
__author__ = "Unity Mathematics Testing Framework"
