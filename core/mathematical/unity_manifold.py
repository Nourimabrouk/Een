# flake8: noqa
"""
Unity Manifold Geometry
=======================

Lightweight geometric engine modeling a "unity manifold" where idempotent
aggregation and φ-harmonic couplings shape curvature. We expose curvature
proxies and invariants suitable for visualization and API responses without
heavy symbolic dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import math


PHI = 1.618033988749895


@dataclass
class ManifoldConfig:
    dimensions: int = 3
    phi_coupling: float = PHI
    idempotent_weight: float = 1.0


@dataclass
class CurvatureInvariants:
    ricci_scalar: float
    mean_curvature: float
    unity_convergence: float
    phi_alignment: float


class UnityManifold:
    """Unity manifold with φ-harmonic idempotent metric proxies."""

    def __init__(self, config: ManifoldConfig | None = None) -> None:
        self.config = config or ManifoldConfig()
        self.dim = max(2, min(11, self.config.dimensions))
        self.phi = self.config.phi_coupling

    def _metric_eigenvalues(self) -> List[float]:
        """
        Construct a φ-harmonic spectrum whose idempotent aggregation returns the
        dominant mode (unity). This simulates an idempotent metric signature.
        """
        base = [math.sin(i * self.phi) ** 2 + 0.1 for i in range(1, self.dim + 1)]
        # Normalize spectrum
        s = sum(base)
        return [v / s for v in base]

    def curvature(self) -> CurvatureInvariants:
        eig = self._metric_eigenvalues()
        # Ricci scalar proxy: concentration of spectrum toward dominant unity mode
        dominance = max(eig)
        ricci_scalar = (self.dim - 1) * dominance * self.phi
        # Mean curvature proxy: average spectral curvature
        mean_curvature = sum(abs(e - 1.0 / self.dim) for e in eig) / self.dim
        # Unity convergence: idempotent collapse score (x + x -> x)
        idem = self.config.idempotent_weight
        unity_convergence = 1.0 / (1.0 + math.exp(-self.phi * dominance * idem))
        # Phi alignment: proximity to golden ratio resonance
        phi_alignment = 1.0 - abs(self.phi - PHI) / PHI
        return CurvatureInvariants(
            ricci_scalar=ricci_scalar,
            mean_curvature=mean_curvature,
            unity_convergence=unity_convergence,
            phi_alignment=max(0.0, min(1.0, phi_alignment)),
        )

    def sample_sectional_curvature(self, samples: int = 32) -> List[float]:
        """Produce a compact set of sectional curvature proxies for plotting."""
        eig = self._metric_eigenvalues()
        dom = max(eig)
        vals: List[float] = []
        for k in range(max(4, min(256, samples))):
            theta = 2 * math.pi * (k / samples)
            vals.append(dom * (1.0 + 0.25 * math.sin(self.phi * theta)))
        return vals

    def export_summary(self) -> Dict[str, float]:
        inv = self.curvature()
        return {
            "dimensions": float(self.dim),
            "ricci_scalar": inv.ricci_scalar,
            "mean_curvature": inv.mean_curvature,
            "unity_convergence": inv.unity_convergence,
            "phi_alignment": inv.phi_alignment,
        }


def create_unity_manifold(
    dimensions: int = 3,
    phi_coupling: float = PHI,
    idempotent_weight: float = 1.0,
) -> UnityManifold:
    return UnityManifold(
        ManifoldConfig(
            dimensions=dimensions,
            phi_coupling=phi_coupling,
            idempotent_weight=idempotent_weight,
        )
    )
