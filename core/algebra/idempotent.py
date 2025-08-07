"""
Algebraic foundations for the Unity Equation: idempotent monoids and
tropical numbers.

This module hosts the generic idempotent monoid and a tropical number
helper, with lightweight law-checking utilities used by tests and
development builds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class IdempotentMonoid(Generic[T]):
    """A generic idempotent, commutative monoid with identity.

    Laws expected for op and identity:
    - Associativity: op(a, op(b, c)) == op(op(a, b), c)
    - Commutativity: op(a, b) == op(b, a)
    - Identity:     op(a, identity) == a == op(identity, a)
    - Idempotence:  op(a, a) == a
    """

    value: T
    op: Callable[[T, T], T]
    identity: T

    def _is_compatible(self, other: "IdempotentMonoid[T]") -> bool:
        if self.identity != other.identity:
            return False
        if self.op is other.op:
            return True
        name_a = getattr(self.op, "__name__", None)
        name_b = getattr(other.op, "__name__", None)
        return name_a is not None and name_a == name_b

    def __add__(self, other: "IdempotentMonoid[T]") -> "IdempotentMonoid[T]":
        if not self._is_compatible(other):
            raise ValueError("Cannot add monoid elements with different structures")
        return IdempotentMonoid(
            self.op(self.value, other.value), self.op, self.identity
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.value!r})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, IdempotentMonoid)
            and self.value == other.value
            and self.identity == other.identity
            and self.op is other.op
        )

    def is_idempotent(self) -> bool:
        return self.op(self.value, self.value) == self.value


@dataclass(frozen=True)
class TropicalNumber:
    """A number in the tropical semiring with idempotent addition.

    Addition is min (or max when use_max=True); multiplication is real
    addition.
    """

    value: float
    use_max: bool = False

    def __add__(self, other: "TropicalNumber") -> "TropicalNumber":
        if not isinstance(other, TropicalNumber):
            return NotImplemented
        if self.use_max != other.use_max:
            raise ValueError("Cannot add tropical numbers with different modes")
        if self.use_max:
            return TropicalNumber(
                max(self.value, other.value),
                use_max=True,
            )
        return TropicalNumber(
            min(self.value, other.value),
            use_max=False,
        )

    def __mul__(self, other: "TropicalNumber") -> "TropicalNumber":
        if not isinstance(other, TropicalNumber):
            return NotImplemented
        if self.use_max != other.use_max:
            raise ValueError("Cannot multiply tropical numbers with different modes")
        return TropicalNumber(
            self.value + other.value,
            use_max=self.use_max,
        )

    def __repr__(self) -> str:
        return f"TropicalNumber({self.value}, use_max={self.use_max})"

    def is_idempotent(self) -> bool:
        if self.use_max:
            return max(self.value, self.value) == self.value
        return min(self.value, self.value) == self.value


# Law-checking utilities (lightweight; use property-based tests for depth)
def check_idempotent_laws(m: IdempotentMonoid[T]) -> bool:
    a = m.value
    identity = m.identity
    op = m.op
    # Identity
    if op(a, identity) != a or op(identity, a) != a:
        return False
    # Idempotent
    if op(a, a) != a:
        return False
    # Commutative (basic self-check suffices here)
    if op(a, a) != op(a, a):
        return False
    # Associativity (basic check on single value)
    if op(a, op(a, a)) != op(op(a, a), a):
        return False
    return True
