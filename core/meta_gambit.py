# flake8: noqa
"""
Meta-Gambit Game Theory
=======================

Minimalist meta-game formalism where strategic agents recognize unity payoffs
that reward coalition/idempotent outcomes over competitive accumulation. We
compute equilibrium-like recommendations via potential games with unity-aware
potential.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple
import math


PHI = 1.618033988749895


@dataclass
class Strategy:
    name: str
    cooperativity: float  # [0,1], unity inclination
    risk: float  # [0,1]


@dataclass
class AgentSpec:
    name: str
    strategies: List[Strategy]


class MetaGambit:
    """Unity-aware meta-game recommender."""

    def __init__(self, agents: List[AgentSpec]) -> None:
        self.agents = agents

    def potential(self, profile: List[int]) -> float:
        # Unity potential: maximize average cooperativity and minimize variance
        cooperativities = [
            self.agents[i].strategies[s].cooperativity for i, s in enumerate(profile)
        ]
        avg_c = sum(cooperativities) / len(cooperativities)
        var_c = sum((c - avg_c) ** 2 for c in cooperativities) / max(
            1, len(cooperativities)
        )
        # Risk penalty
        risks = [self.agents[i].strategies[s].risk for i, s in enumerate(profile)]
        avg_r = sum(risks) / len(risks)
        return avg_c - 0.25 * var_c - 0.15 * avg_r + 0.05 * math.sin(PHI * avg_c)

    def best_response_profile(self, iters: int = 64) -> Tuple[List[int], float]:
        # Start with cooperative choices if available
        profile = [
            max(
                range(len(a.strategies)),
                key=lambda j: a.strategies[j].cooperativity,
            )
            for a in self.agents
        ]
        best_val = self.potential(profile)
        improved = True
        steps = 0
        while improved and steps < iters:
            improved = False
            steps += 1
            for i, agent in enumerate(self.agents):
                base = profile[:]
                local_best = best_val
                local_j = base[i]
                for j in range(len(agent.strategies)):
                    base[i] = j
                    val = self.potential(base)
                    if val > local_best:
                        local_best = val
                        local_j = j
                if local_j != profile[i]:
                    profile[i] = local_j
                    best_val = local_best
                    improved = True
        return profile, best_val

    def export_recommendations(self) -> Dict[str, str]:
        profile, val = self.best_response_profile()
        recs: Dict[str, str] = {}
        for i, agent in enumerate(self.agents):
            recs[agent.name] = agent.strategies[profile[i]].name
        recs["unity_potential"] = f"{val:.3f}"
        return recs


def create_default_meta_gambit() -> MetaGambit:
    a = AgentSpec(
        name="A",
        strategies=[
            Strategy("Unify", cooperativity=0.95, risk=0.2),
            Strategy("Compete", cooperativity=0.2, risk=0.6),
        ],
    )
    b = AgentSpec(
        name="B",
        strategies=[
            Strategy("Unify", cooperativity=0.9, risk=0.25),
            Strategy("Exploit", cooperativity=0.35, risk=0.5),
        ],
    )
    return MetaGambit([a, b])
