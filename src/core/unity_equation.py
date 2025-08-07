"""
Shim module to preserve legacy imports in tests and examples.

Usage: from src.core.unity_equation import IdempotentMonoid
"""

from core.unity_equation import *  # noqa: F401,F403
