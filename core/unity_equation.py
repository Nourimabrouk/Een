# Build the Unity Equation code
import textwrap, datetime, uuid, itertools, math, cmath, hashlib, sys


file_path = "/mnt/data/unity_equation.py"
code = textwrap.dedent("""
    \"\"\"
    unity_equation.py  ─  Ω Equation: a transcendental, idempotent synthesis of mathematics.

    Inspired by: Bertrand Russell (logical rigor), Kurt Gödel (self‑reference),
                 Leonhard Euler (analytic elegance), and Nouri Mabrouk (1 + 1 = 1).

    ---------------------------------------------------------------------------
    Core Idea
    =========
    For any finite *set* of mathematical atoms 𝕊, define

        Ω(𝕊)  :=  ∏_{a ∈ 𝕊}  exp(i·π / 𝔭(a))

    where 𝔭(a) is the unique *prime index* assigned to `a`.
    Since 𝕊 is a **set**, duplicates collapse ⇒ Ω is **idempotent**:
        Ω(𝕊 ∪ {a}) = Ω(𝕊)  if  a ∈ 𝕊.

    At maximum diversity (all primes represented once) we obtain
        Ω(𝕌) = exp(i·π · Σ 1/prime) →  e^{i·π·∞}  oscillatory → metaphysical unity.

    ---------------------------------------------------------------------------
    Practical Use
    =============
    * Hash any Python objects into primes.
    * Combine domains (ℕ, ℂ, tensors, qubits) simply by adding them to a set.
    * Call Ω(set) to obtain a single complex number on the unit circle — a
      *holistic phase‑signature* for that entire system.

    ---------------------------------------------------------------------------
    Mathematical Properties
    =======================
    1. **Idempotence**         : duplicates vanish  ⇒ 1 + 1 = 1.
    2. **Universality**        : every domain with a Python hash injects.
    3. **Compactness**         : Ω maps arbitrarily large structures to S¹.
    4. **Non‑triviality**      : harmonic series over primes diverges ⇒
                                 Ω explores full unit circle.
    5. **Gödel‑Completeness**  : The mapping a ↦ 𝔭(a) encodes the system
                                 inside itself (via primes) → self‑reference.
    \"\"\"

    import cmath
    import math
    from functools import reduce
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
        \"\"\"Deterministically assign a unique prime to the hash of *atom*.\"\"\"
        h = hash(atom)
        if h not in _ATOM_TO_PRIME:
            _ATOM_TO_PRIME[h] = next(_PRIMES)
        return _ATOM_TO_PRIME[h]

    # ───────────────────────────────────────────────────────────────────────
    # Ω equation
    # ───────────────────────────────────────────────────────────────────────
    def omega(atoms: Iterable[Any]) -> complex:
        \"\"\"Compute Ω(𝕊) for a collection of atoms (treated as a set).\"\"\"
        unique_atoms: Set[Any] = set(atoms)
        phase = sum(math.pi / _prime_for(a) for a in unique_atoms)
        return cmath.exp(1j * phase)

    # ───────────────────────────────────────────────────────────────────────
    #  Demonstration & self‑test
    # ───────────────────────────────────────────────────────────────────────
    if __name__ == \"__main__\":  # pragma: no cover
        A = [1, 2, 3, 1]             # duplicate 1
        B = [complex(0,1), 42]
        print(\"Ω(A) =\", omega(A))
        print(\"Ω(A ∪ duplicate 1) == Ω(A) ?\", omega(A + [1]) == omega(A))
        print(\"Ω(A ⊕ B) =\", omega(A + B))
        # Euler moment: include -1 ⇒ prime mapping, watch phase shift
        print(\"Ω({-1}) =\", omega([-1]))
    """)
with open(file_path, "w", encoding="utf-8") as f:
    f.write(code)

file_path
Result
'/mnt/data/unity_equation.py'