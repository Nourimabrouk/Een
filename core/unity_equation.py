"""
Ω Equation: a transcendental, idempotent synthesis of mathematics.

Inspired by: Bertrand Russell (logical rigor), Kurt Gödel (self‑reference),
             Leonhard Euler (analytic elegance), and Nouri Mabrouk (1 + 1 = 1).

Core Idea: For any finite *set* of mathematical atoms 𝕊, define
    Ω(𝕊) := ∏_{a ∈ 𝕊} exp(i·π / 𝔭(a))
where 𝔭(a) is the unique *prime index* assigned to `a`.
Since 𝕊 is a **set**, duplicates collapse ⇒ Ω is **idempotent**:
    Ω(𝕊 ∪ {a}) = Ω(𝕊) if a ∈ 𝕊.
"""

import cmath
import math
from typing import Any, Dict, Set, Iterable

# ───────────────────────────────────────────────────────────────────────
# Prime machinery (simple, deterministic, sufficient for demo)
# ───────────────────────────────────────────────────────────────────────
def _prime_generator():
    yield 2
    primes = [2]
    n = 3
    while True:
        if all(n % p for p in primes if p * p <= n):
            primes.append(n)
            yield n
        n += 2

_PRIMES = _prime_generator()
_ATOM_TO_PRIME: Dict[int, int] = {}

def _prime_for(atom: Any) -> int:
    """Deterministically assign a unique prime to the hash of *atom*."""
    h = hash(atom)
    if h not in _ATOM_TO_PRIME:
        _ATOM_TO_PRIME[h] = next(_PRIMES)
    return _ATOM_TO_PRIME[h]

# ───────────────────────────────────────────────────────────────────────
# Ω equation
# ───────────────────────────────────────────────────────────────────────
def omega(atoms: Iterable[Any]) -> complex:
    """Compute Ω(𝕊) for a collection of atoms (treated as a set)."""
    unique_atoms: Set[Any] = set(atoms)
    phase = sum(math.pi / _prime_for(a) for a in unique_atoms)
    return cmath.exp(1j * phase)

# ───────────────────────────────────────────────────────────────────────
#  Demonstration & self‑test
# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":  # pragma: no cover
    A = [1, 2, 3, 1]             # duplicate 1
    B = [complex(0,1), 42]
    print("Ω(A) =", omega(A))
    print("Ω(A ∪ duplicate 1) == Ω(A) ?", omega(A + [1]) == omega(A))
    print("Ω(A ⊕ B) =", omega(A + B))
    # Euler moment: include -1 ⇒ prime mapping, watch phase shift
    print("Ω({-1}) =", omega([-1]))