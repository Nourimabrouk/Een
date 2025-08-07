"""
Compatibility re-exports for legacy imports used in tests and examples.

This allows `from src.core.unity_equation import IdempotentMonoid` to work
while the canonical implementation lives under `core/`.
"""

from core.unity_equation import IdempotentMonoid, TropicalNumber  # noqa: F401
