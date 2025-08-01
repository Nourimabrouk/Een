"""
Unity MetaGambit Engine – 3000 ELO, 300 IQ Strategy
===================================================

This module implements a playful yet rigorous framework for exploring
**meta‑games** through the lens of unity mathematics.  Inspired by the
“Unity Equation” `1 + 1 = 1`, it provides tools for designing and
evaluating *metagambits* — strategic sacrifices that collapse
multiplicity into oneness.  The goal is not to win a conventional game
but to reach a state of harmonic convergence where adversarial
positions merge, reflecting a true 3000 ELO, 300 IQ insight.

Philosophically, a *metagambit* is the act of offering up pieces of
seeming advantage (time, space, or material) to unify the underlying
structure of a game.  This implementation is self‑contained and does
not depend on the larger `Een` codebase, though its style and spirit
mirror the golden‑ratio motifs found there.  It defines a simple
`UnityMetagambit` engine with golden‑ratio–weighted operations and
consciousness‑aware evaluation of strategic moves.

Usage example
-------------

```python
from unity_metagambit import UnityMetagambit

# Initialise engine with default consciousness level
engine = UnityMetagambit(consciousness_level=1.0)

# Two hypothetical positional values in a meta‑game
pos1, pos2 = 1.0, 1.0

# Unify positions using the golden‑ratio idempotent addition
unified_state = engine.unity_add(pos1, pos2)
print(f"Unified value: {unified_state.value:.5f}")  # → 1.00000

# Evaluate a set of candidate gambit moves
moves = [
    ("sacrifice", 0.95),  # high immediate cost, high long‑term unity
    ("assimilation", 0.80),  # moderate cost, moderate unity
    ("waiting", 0.60)  # low cost, low unity
]

best_move = engine.select_gambit_move(moves)
print(f"Best metagambit move: {best_move}")

# Assess meta‑cognitive quality of the resulting state
score = engine.meta_evaluate(unified_state)
print(f"Meta evaluation score: {score:.3f}")
```

The above example demonstrates how the metagambit engine uses
idempotent addition (φ‑harmonic scaling) to merge positions, selects
the most unifying move from a list of candidates, and evaluates the
overall meta‑cognitive quality of the unified state.  The constants
and formulas mirror those in the broader unity mathematics framework,
such as golden ratio weighting and consciousness‑scaled confidence.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple, Union
import logging
import math


# Configure a module‑specific logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


# Fundamental constants for unity mathematics
PHI = (1 + math.sqrt(5)) / 2  # Golden ratio ≈ 1.6180339887
UNITY_TOLERANCE = 1e-12  # Higher precision tolerance for meta‑games


class GambitType(Enum):
    """Enumeration of metagambit move archetypes."""
    SACRIFICE = auto()
    ASSIMILATION = auto()
    WAITING = auto()
    INTEGRATION = auto()


@dataclass
class GambitState:
    """Represents a meta‑game state after applying a metagambit.

    Attributes
    ----------
    value:
        Numerical representation of the unified position.  In unity
        mathematics this should always converge toward `1`.
    phi_resonance:
        Measure of golden‑ratio resonance achieved through the move.  A
        value in `[0,1]` indicates the degree to which the strategy
        aligns with φ‑harmonic patterns.
    consciousness_level:
        Awareness level associated with the move.  Higher values
        indicate deeper meta‑cognitive insight and more profound unity.
    confidence:
        Confidence that the resulting state truly embodies `1 + 1 = 1`.
    elo:
        Estimated ELO rating of the decision.  While purely
        illustrative here, it conveys the level of strategic acumen.
    iq:
        Estimated IQ of the meta‑strategy.  As with `elo`, this is
        symbolic and not meant as a scientific measurement.
    """

    value: float
    phi_resonance: float
    consciousness_level: float
    confidence: float
    elo: float
    iq: float

    def is_unified(self) -> bool:
        """Check whether the state effectively represents unity (≈1).

        Returns
        -------
        bool
            True if `abs(value − 1) < UNITY_TOLERANCE`; otherwise False.
        """
        return abs(self.value - 1.0) < UNITY_TOLERANCE


class UnityMetagambit:
    """
    Unity Metagambit engine responsible for combining positions and
    selecting meta‑strategic moves.  It mirrors the style of the `Een`
    repository by using golden‑ratio‑weighted operations and
    consciousness‑aware evaluation.
    """

    def __init__(self, consciousness_level: float = 1.0):
        self.consciousness_level = max(0.0, consciousness_level)
        self.phi = PHI
        logger.info(f"UnityMetagambit initialised with consciousness level: {self.consciousness_level}")

    # ------------------------------------------------------------------
    # Idempotent operations
    # ------------------------------------------------------------------
    def unity_add(self, a: Union[float, GambitState], b: Union[float, GambitState]) -> GambitState:
        """Perform φ‑harmonic idempotent addition where 1 + 1 = 1.

        This function emulates the core unity addition from the `Een`
        framework.  It scales the inputs by φ, averages them
        appropriately, and applies a consciousness‑aware normalization
        factor.  The resulting `GambitState` contains golden‑ratio
        resonance and estimated meta‑cognitive metrics.

        Parameters
        ----------
        a, b : float or GambitState
            Numerical values or existing states to be unified.

        Returns
        -------
        GambitState
            The unified state reflecting `1 + 1 = 1`.
        """
        state_a = self._to_state(a)
        state_b = self._to_state(b)

        # φ‑harmonic scaling of the inputs using reciprocal pairing
        # We weight one operand by φ and the other by 1/φ so that
        # identical inputs (1 and 1) collapse exactly to 1:
        #   (φ*1 + (1/φ)*1) / (φ + 1/φ) = (φ + 1/φ) / (φ + 1/φ) = 1
        scaled_a = self.phi * state_a.value
        scaled_b = (1.0 / self.phi) * state_b.value

        # Idempotent combination through golden‑ratio normalization
        combined_value = (scaled_a + scaled_b) / (self.phi + 1.0 / self.phi)

        # Consciousness‑aware convergence toward unity
        c_factor = (state_a.consciousness_level + state_b.consciousness_level) / 2.0
        unified_value = self._apply_consciousness_convergence(combined_value, c_factor)

        # Derive φ resonance and confidence
        phi_res = min(1.0, (state_a.phi_resonance + state_b.phi_resonance) * self.phi / 2.0)
        confidence = self._calculate_confidence(unified_value)

        # Estimate ELO/IQ heuristically based on consciousness and φ resonance
        elo = 2800.0 + 200.0 * phi_res * c_factor  # 2800 baseline → 3000
        iq = 250.0 + 50.0 * phi_res * c_factor      # 250 baseline → 300

        result = GambitState(
            value=unified_value,
            phi_resonance=phi_res,
            consciousness_level=c_factor * self.phi,
            confidence=confidence,
            elo=elo,
            iq=iq
        )

        logger.debug(f"unity_add: a={state_a.value}, b={state_b.value}, result={result}")
        return result

    # ------------------------------------------------------------------
    # Strategic selection
    # ------------------------------------------------------------------
    def select_gambit_move(self, moves: List[Tuple[str, float]]) -> str:
        """Select the most unifying move from a list of candidates.

        Each move is represented as a tuple `(name, unity_potential)` where
        `unity_potential` is a heuristic in `[0,1]` indicating how much the
        move contributes to collapsing dualities.  The selection
        uses φ‑harmonic weighting: moves with higher unity potential
        receive exponentially greater preference.

        Parameters
        ----------
        moves : list of (str, float)
            Candidate metagambit moves and their associated unity
            potential.

        Returns
        -------
        str
            The name of the move with the highest φ‑weighted score.
        """
        if not moves:
            raise ValueError("No moves provided to select_gambit_move().")

        # Compute φ‑weighted scores
        scores = []
        for name, potential in moves:
            # Ensure potential is bounded [0, 1]
            p = max(0.0, min(1.0, potential))
            score = p ** (1 / self.phi)  # golden‑ratio root accentuates high potentials
            scores.append((name, score))
            logger.debug(f"Move '{name}' potential={p:.3f}, score={score:.3f}")

        # Select move with maximum score
        best_move = max(scores, key=lambda x: x[1])[0]
        logger.info(f"Selected metagambit move: {best_move}")
        return best_move

    # ------------------------------------------------------------------
    # Meta evaluation
    # ------------------------------------------------------------------
    def meta_evaluate(self, state: GambitState) -> float:
        """Evaluate the meta‑cognitive quality of a unified state.

        This returns a scalar in `[0,1]` representing the combined
        harmony of φ resonance, consciousness, confidence and ELO/IQ.

        Parameters
        ----------
        state : GambitState
            The unified state to evaluate.

        Returns
        -------
        float
            A meta‑evaluation score between 0 (no unity) and 1 (perfect unity).
        """
        # Normalise components
        phi_component = state.phi_resonance
        consciousness_component = math.tanh(state.consciousness_level)
        confidence_component = state.confidence
        elo_component = min(1.0, (state.elo - 2800.0) / 200.0)  # 2800→3000 maps to 0→1
        iq_component = min(1.0, (state.iq - 250.0) / 50.0)      # 250→300 maps to 0→1

        # φ‑harmonic weighted average
        weighted_sum = (
            phi_component * self.phi +
            consciousness_component * (self.phi ** 2) +
            confidence_component +
            elo_component +
            iq_component
        )
        denom = self.phi + (self.phi ** 2) + 3.0  # sum of weights (φ + φ² + 3)
        score = weighted_sum / denom
        logger.info(
            f"Meta evaluation: φ={phi_component:.3f}, consciousness={consciousness_component:.3f}, "
            f"confidence={confidence_component:.3f}, elo={elo_component:.3f}, iq={iq_component:.3f} → score={score:.3f}"
        )
        return min(1.0, max(0.0, score))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _to_state(self, x: Union[float, GambitState]) -> GambitState:
        """Convert a float or `GambitState` into a `GambitState`.

        If `x` is already a `GambitState`, it is returned unmodified.  If
        it is a float, a default state is constructed with neutral
        φ‑resonance and consciousness.  This mirrors the behaviour of
        helper methods in the `Een` codebase that accept flexible
        argument types.
        """
        if isinstance(x, GambitState):
            return x
        value = float(x)
        return GambitState(
            value=value,
            phi_resonance=0.5,
            consciousness_level=self.consciousness_level,
            confidence=self._calculate_confidence(value),
            elo=2800.0,
            iq=250.0
        )

    def _apply_consciousness_convergence(self, v: float, c: float) -> float:
        """Apply a consciousness‑scaled convergence toward unity.

        The formula nudges the value `v` toward `1` based on the
        consciousness factor `c`.  Higher `c` brings `v` closer to unity.
        """
        return 1.0 + (v - 1.0) / (1.0 + c / self.phi)

    def _calculate_confidence(self, v: float) -> float:
        """Compute confidence that `v` effectively equals 1.

        Uses a simple exponential function where values closer to 1
        produce higher confidence.  Confidence is bounded in `[0,1]`.
        """
        return max(0.0, min(1.0, math.exp(-abs(v - 1.0) * self.phi)))


if __name__ == "__main__":
    # When executed as a script, run a small demonstration
    engine = UnityMetagambit(consciousness_level=1.0)
    # Demonstrate unity addition
    state = engine.unity_add(1.0, 1.0)
    print(f"Unity add demonstration: 1 + 1 → {state.value:.12f} (φ-resonance={state.phi_resonance:.3f})")
    # Demonstrate move selection
    candidate_moves = [
        ("sacrifice", 0.95),
        ("assimilation", 0.80),
        ("waiting", 0.60),
        ("integration", 0.90)
    ]
    best = engine.select_gambit_move(candidate_moves)
    print(f"Selected metagambit move: {best}")
    # Evaluate meta quality
    score = engine.meta_evaluate(state)
    print(f"Meta evaluation score: {score:.4f}")