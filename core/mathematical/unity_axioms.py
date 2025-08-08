"""
Unity Axioms and Meta-Layer Semantics

This module axiomatizes the core mathematical structures used across the
project to formalize the unity equation (1 + 1 = 1) in idempotent contexts,
and provides a principled reflective meta-layer (typed truth predicates).

Foundations
- IdempotentSemiring: (S, ⊕, ⊗, 0, 1) with ⊕ idempotent and commutative,
  and ⊗ associative with 1 as multiplicative identity. Distributivity holds.
- UnityToposSketch: Category-theoretic sketch where 1 is terminal and ⊕ is
  an idempotent monoidal sum satisfying a collapse-to-unity natural
  transformation η: Id ⇒ Δ₁.
- TypedTruthTower: Tarski-style stratified predicates T₀, T₁, … with safe
  reflection up the tower.

These utilities are intentionally lightweight, designed to be imported by
API routes for audits, safety checks, and demonstrations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Iterable, List, Tuple, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class IdempotentSemiring(Generic[T]):
    """A minimal idempotent semiring interface.

    - addition (oplus) is idempotent and commutative
    - multiplication (otimes) is associative
    - distributivity: a ⊗ (b ⊕ c) = (a ⊗ b) ⊕ (a ⊗ c)
    - identities: 0 for ⊕ and 1 for ⊗
    """

    oplus: Callable[[T, T], T]
    otimes: Callable[[T, T], T]
    zero: T
    one: T

    def is_idempotent(self, x: T) -> bool:
        return self.oplus(x, x) == x

    def fold_sum(self, values: Iterable[T]) -> T:
        result = self.zero
        for v in values:
            result = self.oplus(result, v)
        return result

    def fold_product(self, values: Iterable[T]) -> T:
        result = self.one
        for v in values:
            result = self.otimes(result, v)
        return result


def boolean_or_semiring() -> IdempotentSemiring[int]:
    """The canonical witness: ({0,1}, OR, AND, 0, 1).

    Interprets 1 ⊕ 1 = 1 with ⊕ := OR.
    """

    def oplus(a: int, b: int) -> int:
        return 1 if (a or b) else 0

    def otimes(a: int, b: int) -> int:
        return 1 if (a and b) else 0

    return IdempotentSemiring(oplus=oplus, otimes=otimes, zero=0, one=1)


@dataclass(frozen=True)
class UnityToposSketch:
    """A minimal categorical sketch sufficient for programmatic checks.

    - Objects: abstract placeholders; here we only model the universal
      property involving terminal 1 and an idempotent ⊕ operation.
    - The commuting square is validated symbolically by function identities.
    """

    terminal_object: str = "1"

    def collapse_to_unity(self, x: T, y: T, semiring: IdempotentSemiring[T]) -> T:
        # Natural transformation η: X → 1 is represented by discarding content
        # (unique arrow to terminal). Commutativity check is modeled by
        # equality of the two paths in the square when reduced to terminal
        # observations.
        return semiring.oplus(x, y)

    def commuting_square_holds(
        self, x: T, y: T, semiring: IdempotentSemiring[T]
    ) -> bool:
        top = semiring.oplus(x, y)
        # The square expresses observational equivalence post-collapse.
        # Accept the law if ⊕ is idempotent and commutative for samples.
        return semiring.is_idempotent(top) and semiring.oplus(x, y) == semiring.oplus(
            y, x
        )


class TypedTruthTower:
    """Tarski-style stratified truth predicates.

    We provide a tiny, safe reflective kernel for programmatic evidence:
    - level 0: base propositions (strings) with evaluator
    - level k+1: may assert Truth_k("phi") but not Truth_{k+1}(self) (no liar)
    """

    def __init__(self) -> None:
        self._evaluators: List[Callable[[str], bool]] = []

    def add_level(self, evaluator: Callable[[str], bool]) -> int:
        self._evaluators.append(evaluator)
        return len(self._evaluators) - 1

    def truth_at(self, level: int, proposition: str) -> bool:
        if 0 <= level < len(self._evaluators):
            return bool(self._evaluators[level](proposition))
        raise IndexError("Truth level does not exist")

    def reflective_assert(self, higher: int, lower: int, proposition: str) -> bool:
        if higher <= lower:
            raise ValueError("Reflection must go upwards (higher > lower)")
        return self.truth_at(lower, proposition)


# --- MDL (Minimum Description Length) helpers ---


def token_length(s: str) -> int:
    return max(1, len(s.strip().split()))


def mdl_length(
    expressions: List[str],
) -> Tuple[int, int, float, List[Tuple[str, int, int]]]:
    """Compute a simple MDL audit over expressions.

    Baseline: sum of token lengths of each expression.
    Unity rewrite: expressions deduplicated by idempotent ⊕; shared motifs
    factored via a naive motif table; overhead charged per unique motif.
    Returns (baseline, compressed, ratio, per_item_breakdown).
    """

    baseline = sum(token_length(e) for e in expressions)

    # Deduplicate identical expressions (idempotence)
    unique = list(dict.fromkeys(e.strip() for e in expressions if e.strip()))

    # Extract recurring motifs (very simple: most common tokens)
    from collections import Counter

    tokens: List[str] = []
    for e in unique:
        tokens.extend(e.split())
    freq = Counter(tokens)
    motifs = [tok for tok, cnt in freq.items() if cnt >= 3 and len(tok) > 2]

    # Cost model: unique expressions + 0.5 cost per motif + reference savings
    unique_cost = sum(token_length(e) for e in unique)
    motif_overhead = int(round(0.5 * len(motifs)))
    reference_savings = min(int(0.2 * baseline), int(0.6 * len(motifs)))
    compressed = max(1, unique_cost + motif_overhead - reference_savings)

    ratio = compressed / baseline if baseline > 0 else 1.0

    breakdown: List[Tuple[str, int, int]] = []
    for e in unique:
        bl = token_length(e)
        use_motif = any(m in e for m in motifs)
        factor = 0.6 if use_motif else 0.85
        cl = max(1, int(round(bl * factor)))
        breakdown.append((e, bl, cl))

    return baseline, compressed, ratio, breakdown


# --- Safety and monotonicity ---


def monotone_update_sequence(
    start: T,
    updates: Iterable[T],
    semiring: IdempotentSemiring[T],
    max_steps: int = 100,
) -> Tuple[T, int]:
    """Iteratively apply x_{t+1} = x_t ⊕ u_t and return fixed point and steps.

    In an idempotent setting, this sequence is monotone and reaches a fixed
    point in at most the size of the carrier (for finite carriers) or in
    practice stabilizes quickly when updates repeat.
    """

    x = start
    steps = 0
    for u in updates:
        nx = semiring.oplus(x, u)
        steps += 1
        if nx == x or steps >= max_steps:
            x = nx
            break
        x = nx
    return x, steps


__all__ = [
    "IdempotentSemiring",
    "UnityToposSketch",
    "TypedTruthTower",
    "boolean_or_semiring",
    "mdl_length",
    "monotone_update_sequence",
]
