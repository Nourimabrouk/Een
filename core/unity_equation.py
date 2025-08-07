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

from typing import Iterable, TypeVar

# Re-export algebraic primitives from centralized module to preserve imports
from .algebra.idempotent import IdempotentMonoid, TropicalNumber  # noqa: F401

T = TypeVar("T")


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
        super().__init__(
            value=frozenset(value),
            op=lambda a, b: a | b,
            identity=frozenset(),
        )


# Note: TropicalNumber is re-exported from .algebra.idempotent above.


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
    assert result_set.value == frozenset(
        {1}
    ), "SetUnionMonoid: 1 + 1 must equal 1 (as a set)"

    # Tropical semiring demonstration (using min)
    t_one = TropicalNumber(1.0, use_max=False)
    assert (t_one + t_one).value == 1.0, "TropicalNumber (min): 1 + 1 must equal 1"

    # Tropical semiring demonstration (using max)
    t_one_max = TropicalNumber(1.0, use_max=True)
    assert (
        t_one_max + t_one_max
    ).value == 1.0, "TropicalNumber (max): 1 + 1 must equal 1"

    msg = (
        "All demonstrations passed. The Unity Equation holds in these "
        "idempotent structures."
    )
    print(msg)


if __name__ == "__main__":
    demonstrate_unity_equation()
