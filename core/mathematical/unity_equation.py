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

from typing import Iterable, TypeVar, Dict, Any

# Re-export algebraic primitives from centralized module to preserve imports
# Import with enhanced error handling
try:
    from ..algebra.idempotent import IdempotentMonoid, TropicalNumber
except ImportError:
    # Fallback implementation for idempotent structures
    class IdempotentMonoid:
        def __init__(self, value, op, identity):
            self.value = value
            self.op = op
            self.identity = identity
        
        def __add__(self, other):
            if isinstance(other, IdempotentMonoid):
                return IdempotentMonoid(self.op(self.value, other.value), self.op, self.identity)
            return self
    
    class TropicalNumber(IdempotentMonoid):
        def __init__(self, value: float, use_max: bool = False):
            if use_max:
                super().__init__(value, max, float('-inf'))
            else:
                super().__init__(value, min, float('inf'))
            self.use_max = use_max

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


# Note: TropicalNumber is re-exported from ..algebra.idempotent above.


def demonstrate_unity_equation() -> Dict[str, Any]:
    """Demonstrate that 1 + 1 = 1 holds in various idempotent structures.

    This function constructs elements of different idempotent monoids and
    semirings and validates that adding the element to itself yields the
    original element. Returns detailed results for analysis.
    
    Returns:
        Dictionary containing demonstration results and validation status
    """
    
    results = {
        'demonstrations': [],
        'all_passed': True,
        'total_tests': 0,
        'passed_tests': 0
    }

    # Boolean monoid demonstration
    try:
        b_one = BooleanMonoid(True)
        result_bool = (b_one + b_one).value
        test_passed = result_bool is True
        results['demonstrations'].append({
            'type': 'BooleanMonoid',
            'input': True,
            'result': result_bool,
            'expected': True,
            'passed': test_passed,
            'message': "Boolean OR: True + True = True"
        })
        results['total_tests'] += 1
        if test_passed:
            results['passed_tests'] += 1
        else:
            results['all_passed'] = False
    except Exception as e:
        results['demonstrations'].append({
            'type': 'BooleanMonoid',
            'error': str(e),
            'passed': False
        })
        results['all_passed'] = False
        results['total_tests'] += 1

    # Set union monoid demonstration
    try:
        set_one = SetUnionMonoid({1})
        result_set = set_one + set_one
        expected_set = frozenset({1})
        test_passed = result_set.value == expected_set
        results['demonstrations'].append({
            'type': 'SetUnionMonoid',
            'input': {1},
            'result': set(result_set.value),
            'expected': set(expected_set),
            'passed': test_passed,
            'message': "Set Union: {1} ∪ {1} = {1}"
        })
        results['total_tests'] += 1
        if test_passed:
            results['passed_tests'] += 1
        else:
            results['all_passed'] = False
    except Exception as e:
        results['demonstrations'].append({
            'type': 'SetUnionMonoid',
            'error': str(e),
            'passed': False
        })
        results['all_passed'] = False
        results['total_tests'] += 1

    # Tropical semiring demonstration (using min)
    try:
        t_one = TropicalNumber(1.0, use_max=False)
        result_tropical_min = (t_one + t_one).value
        test_passed = abs(result_tropical_min - 1.0) < 1e-10
        results['demonstrations'].append({
            'type': 'TropicalNumber_min',
            'input': 1.0,
            'result': result_tropical_min,
            'expected': 1.0,
            'passed': test_passed,
            'message': "Tropical (min): min(1, 1) = 1"
        })
        results['total_tests'] += 1
        if test_passed:
            results['passed_tests'] += 1
        else:
            results['all_passed'] = False
    except Exception as e:
        results['demonstrations'].append({
            'type': 'TropicalNumber_min',
            'error': str(e),
            'passed': False
        })
        results['all_passed'] = False
        results['total_tests'] += 1

    # Tropical semiring demonstration (using max)
    try:
        t_one_max = TropicalNumber(1.0, use_max=True)
        result_tropical_max = (t_one_max + t_one_max).value
        test_passed = abs(result_tropical_max - 1.0) < 1e-10
        results['demonstrations'].append({
            'type': 'TropicalNumber_max',
            'input': 1.0,
            'result': result_tropical_max,
            'expected': 1.0,
            'passed': test_passed,
            'message': "Tropical (max): max(1, 1) = 1"
        })
        results['total_tests'] += 1
        if test_passed:
            results['passed_tests'] += 1
        else:
            results['all_passed'] = False
    except Exception as e:
        results['demonstrations'].append({
            'type': 'TropicalNumber_max',
            'error': str(e),
            'passed': False
        })
        results['all_passed'] = False
        results['total_tests'] += 1

    # Summary message
    if results['all_passed']:
        msg = (f"All {results['passed_tests']}/{results['total_tests']} demonstrations passed. "
               f"The Unity Equation 1+1=1 holds in these idempotent structures.")
    else:
        msg = (f"Only {results['passed_tests']}/{results['total_tests']} demonstrations passed. "
               f"Some idempotent structures may need further investigation.")
    
    results['summary_message'] = msg
    print(msg)
    
    return results


if __name__ == "__main__":
    demonstrate_unity_equation()
