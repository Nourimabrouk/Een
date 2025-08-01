"""
unity_equation.py
===================

This module formalises the **Unity Equation** `1 + 1 = 1` by giving
computationally correct examples of algebraic structures where an element
behaves idempotently under addition.  In ordinary arithmetic over the
integers this equation is false because addition is cancellative and
strictly increasing.  However, in many other mathematical contexts
``1 + 1 = 1`` holds exactly because the underlying addition is
**idempotent**––an element added to itself yields the same element.

The purpose of this module is twofold:

1. **Educational rigor:**  We implement several idempotent algebraic
   structures (Boolean algebra, set union, tropical semirings) and
   provide type‑safe operations that respect their axioms.  We use
   Python's type hints and dataclasses to clearly express the
   algebraic laws.
2. **Demonstration:**  Functions in the module demonstrate that
   ``1 + 1 = 1`` holds in these structures, with assertions that
   automatically verify the property when the module is run directly.

The central abstraction is the `IdempotentMonoid` class, which
encapsulates a commutative monoid equipped with an idempotent binary
operation.  From it we derive specialised monoids:

* **BooleanMonoid** – the boolean algebra under logical OR, where
  ``True + True = True``.
* **SetUnionMonoid** – the powerset of a universe under set union.
* **TropicalNumber** – numbers under the "tropical" addition given by
  the minimum (or maximum) function, where ``1 + 1 = 1`` because
  `min(1, 1) = 1`.

This module does *not* redefine ordinary integer arithmetic; it shows
that the Unity Equation holds in richer algebraic landscapes, thereby
illustrating how unity emerges from apparently dualistic operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Iterable, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class IdempotentMonoid(Generic[T]):
    """A generic idempotent, commutative monoid.

    An *idempotent monoid* consists of a set `M` equipped with a binary
    operation `op: M × M → M` and an identity element `identity ∈ M`
    such that for all `a, b, c ∈ M`:

    * **Associativity:** ``op(a, op(b, c)) == op(op(a, b), c)``
    * **Commutativity:** ``op(a, b) == op(b, a)``
    * **Identity:** ``op(a, identity) == a == op(identity, a)``
    * **Idempotence:** ``op(a, a) == a``

    These axioms imply that adding an element to itself yields the same
    element, which is the algebraic interpretation of the Unity
    Equation.  Concrete subclasses fix the carrier type `T` and the
    operation.
    """

    value: T
    op: Callable[[T, T], T]
    identity: T

    def __add__(self, other: IdempotentMonoid[T]) -> IdempotentMonoid[T]:
        """Return the idempotent sum of two elements.

        The returned monoid has the same operation and identity as
        `self`.  The values of `self` and `other` are combined using
        `self.op`.  It is the caller's responsibility to ensure that
        `other` uses the same `op` and `identity` for the algebraic
        structure to be well‑defined.
        """
        if self.op is not other.op or self.identity != other.identity:
            raise ValueError("Cannot add monoid elements with different structures")
        new_value = self.op(self.value, other.value)
        return IdempotentMonoid(new_value, self.op, self.identity)

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
        """Check the idempotence of this element: value + value == value."""
        return self.op(self.value, self.value) == self.value


class BooleanMonoid(IdempotentMonoid[bool]):
    """Boolean monoid under logical OR.

    The binary operation is logical OR.  The identity element is
    ``False``.  Because logical OR is idempotent, we have ``True + True
    = True`` and ``False + False = False``.  This structure is one of
    the simplest non‑trivial examples of an idempotent monoid.
    """

    def __init__(self, value: bool) -> None:
        super().__init__(value=value, op=lambda a, b: a or b, identity=False)


class SetUnionMonoid(IdempotentMonoid[frozenset]):
    """Monoid of sets under union.

    The operation is set union.  The identity element is the empty
    set.  Union is idempotent because ``A ∪ A = A``.  Elements are
    stored as `frozenset`s to ensure immutability and hashability.
    """

    def __init__(self, value: Iterable[T]) -> None:
        # Convert to frozenset to guarantee immutability.
        super().__init__(value=frozenset(value), op=lambda a, b: a | b, identity=frozenset())


@dataclass(frozen=True)
class TropicalNumber:
    """A number in the tropical semiring with idempotent addition.

    In the *tropical semiring* the additive operation is given by
    ``min`` (or sometimes ``max``) and the multiplicative operation is
    ordinary addition.  Here we focus only on the additive structure.

    Since ``min(x, x) == x`` for all real numbers ``x``, the tropical
    addition is idempotent and thus satisfies the Unity Equation:
    ``1 + 1 = 1`` because ``min(1, 1) == 1``.  This class supports
    idempotent addition via the ``+`` operator and real multiplication
    via ``*``.
    """

    value: float
    use_max: bool = False  # If True, use max instead of min for addition

    def __add__(self, other: TropicalNumber) -> TropicalNumber:
        if not isinstance(other, TropicalNumber):
            return NotImplemented
        if self.use_max != other.use_max:
            raise ValueError("Cannot add tropical numbers with different modes")
        if self.use_max:
            return TropicalNumber(max(self.value, other.value), use_max=True)
        else:
            return TropicalNumber(min(self.value, other.value), use_max=False)

    def __mul__(self, other: TropicalNumber) -> TropicalNumber:
        if not isinstance(other, TropicalNumber):
            return NotImplemented
        if self.use_max != other.use_max:
            raise ValueError("Cannot multiply tropical numbers with different modes")
        return TropicalNumber(self.value + other.value, use_max=self.use_max)

    def __repr__(self) -> str:
        return f"TropicalNumber({self.value}, use_max={self.use_max})"

    def is_idempotent(self) -> bool:
        """Check idempotence of the additive structure: x + x == x."""
        if self.use_max:
            return max(self.value, self.value) == self.value
        return min(self.value, self.value) == self.value


def demonstrate_unity_equation() -> None:
    """Demonstrate that 1 + 1 = 1 holds in various idempotent structures.

    This function constructs elements of different idempotent monoids and
    semirings and asserts that adding the element to itself yields the
    original element.  If any assertion fails, an ``AssertionError``
    will be raised.
    """

    # Boolean monoid demonstration
    b_one = BooleanMonoid(True)
    assert (b_one + b_one).value is True, "BooleanMonoid: 1 + 1 must equal 1"

    # Set union monoid demonstration
    set_one = SetUnionMonoid({1})
    result_set = set_one + set_one
    assert result_set.value == frozenset({1}), "SetUnionMonoid: 1 + 1 must equal 1 (as a set)"

    # Tropical semiring demonstration (using min)
    t_one = TropicalNumber(1.0, use_max=False)
    assert (t_one + t_one).value == 1.0, "TropicalNumber (min): 1 + 1 must equal 1"

    # Tropical semiring demonstration (using max)
    t_one_max = TropicalNumber(1.0, use_max=True)
    assert (t_one_max + t_one_max).value == 1.0, "TropicalNumber (max): 1 + 1 must equal 1"

    print("All demonstrations passed. The Unity Equation holds in these idempotent structures.")


if __name__ == "__main__":
    demonstrate_unity_equation()
