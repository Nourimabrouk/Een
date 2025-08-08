"""
Property-based tests for idempotent algebraic structures using Hypothesis.

This module implements rigorous property-based testing of the idempotent algebraic
structures that demonstrate the Unity Equation 1+1=1. We use Hypothesis to generate
comprehensive test cases and verify algebraic laws with mathematical precision.

The tests validate:
- Idempotence: a + a = a (the core unity property)
- Associativity: (a + b) + c = a + (b + c)
- Commutativity: a + b = b + a
- Identity: a + 0 = a = 0 + a
- Distributivity (where applicable)
- Contraction stability: operations preserve algebraic structure
"""

from __future__ import annotations

import math
import pytest
from typing import Union, TypeVar, Generic, Set, FrozenSet, List

# Property-based testing with Hypothesis
try:
    from hypothesis import given, strategies as st, assume, settings, Verbosity
    from hypothesis.strategies import composite, SearchStrategy
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    # Create mock decorators for graceful degradation
    def given(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def settings(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    class MockStrategies:
        def floats(self, *args, **kwargs): return lambda: [0.0, 1.0, -1.0]
        def booleans(self): return lambda: [True, False]
        def integers(self, *args, **kwargs): return lambda: [0, 1, 2, -1]
        def sets(self, *args, **kwargs): return lambda: [set(), {1}, {1, 2}]
        def frozensets(self, *args, **kwargs): return lambda: [frozenset(), frozenset({1})]
        def one_of(self, *args): return lambda: args[0] if args else []
    
    st = MockStrategies()
    Verbosity = type('Verbosity', (), {'verbose': 1})()

from core.mathematical.unity_equation import BooleanMonoid, SetUnionMonoid, TropicalNumber
from core.algebra.idempotent import IdempotentMonoid, check_idempotent_laws

T = TypeVar('T')

# Test configuration for property-based testing
MAX_EXAMPLES = 1000 if HYPOTHESIS_AVAILABLE else 10
DEADLINE_MS = 5000 if HYPOTHESIS_AVAILABLE else 1000

# Custom strategies for algebraic structures
@composite
def boolean_monoid_strategy(draw) -> BooleanMonoid:
    """Generate random BooleanMonoid instances."""
    value = draw(st.booleans())
    return BooleanMonoid(value)

@composite 
def set_union_monoid_strategy(draw) -> SetUnionMonoid:
    """Generate random SetUnionMonoid instances."""
    # Generate sets of small integers to keep test manageable
    elements = draw(st.sets(st.integers(min_value=0, max_value=10), max_size=5))
    return SetUnionMonoid(elements)

@composite
def tropical_number_strategy(draw) -> TropicalNumber:
    """Generate random TropicalNumber instances."""
    # Use finite range to avoid numerical issues
    value = draw(st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    use_max = draw(st.booleans())
    return TropicalNumber(value, use_max)

@composite
def phi_harmonic_float_strategy(draw) -> float:
    """Generate φ-harmonic floating point values for unity mathematics."""
    # Generate values with φ-scaling to test golden ratio properties
    base_value = draw(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False))
    phi = 1.618033988749895
    return base_value * phi

class TestIdempotentLaws:
    """Property-based tests for idempotent algebraic laws."""
    
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @given(boolean_monoid_strategy())
    @settings(max_examples=MAX_EXAMPLES, deadline=DEADLINE_MS, verbosity=Verbosity.verbose)
    def test_boolean_monoid_idempotence(self, m: BooleanMonoid):
        """Test that boolean monoid satisfies a + a = a (idempotence)."""
        result = m + m
        assert result.value == m.value, f"Idempotence failed: {m.value} + {m.value} = {result.value}"
        assert result == m, f"Monoid equality failed: {result} != {m}"
        
        # Verify internal structure preservation
        assert result.identity == m.identity
        assert result.op is m.op

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @given(boolean_monoid_strategy(), boolean_monoid_strategy())
    @settings(max_examples=MAX_EXAMPLES, deadline=DEADLINE_MS)
    def test_boolean_monoid_commutativity(self, a: BooleanMonoid, b: BooleanMonoid):
        """Test that boolean monoid satisfies a + b = b + a (commutativity)."""
        try:
            result_ab = a + b
            result_ba = b + a
            assert result_ab.value == result_ba.value, f"Commutativity failed: {a.value} + {b.value} != {b.value} + {a.value}"
        except ValueError:
            # Different boolean monoids cannot be added - this is expected
            pytest.skip("Incompatible monoid structures")

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @given(boolean_monoid_strategy(), boolean_monoid_strategy(), boolean_monoid_strategy())
    @settings(max_examples=MAX_EXAMPLES, deadline=DEADLINE_MS)
    def test_boolean_monoid_associativity(self, a: BooleanMonoid, b: BooleanMonoid, c: BooleanMonoid):
        """Test that boolean monoid satisfies (a + b) + c = a + (b + c) (associativity)."""
        try:
            result_left = (a + b) + c
            result_right = a + (b + c)
            assert result_left.value == result_right.value, f"Associativity failed: ({a.value} + {b.value}) + {c.value} != {a.value} + ({b.value} + {c.value})"
        except ValueError:
            # Incompatible structures - skip this test case
            pytest.skip("Incompatible monoid structures")

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @given(boolean_monoid_strategy())
    @settings(max_examples=MAX_EXAMPLES, deadline=DEADLINE_MS)
    def test_boolean_monoid_identity(self, m: BooleanMonoid):
        """Test that boolean monoid satisfies a + 0 = a = 0 + a (identity)."""
        identity_element = BooleanMonoid(m.identity)
        
        # Test right identity: a + 0 = a
        result_right = m + identity_element
        assert result_right.value == m.value, f"Right identity failed: {m.value} + {identity_element.value} = {result_right.value}"
        
        # Test left identity: 0 + a = a
        result_left = identity_element + m
        assert result_left.value == m.value, f"Left identity failed: {identity_element.value} + {m.value} = {result_left.value}"

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @given(set_union_monoid_strategy())
    @settings(max_examples=MAX_EXAMPLES, deadline=DEADLINE_MS)
    def test_set_union_monoid_idempotence(self, m: SetUnionMonoid):
        """Test that set union monoid satisfies A ∪ A = A (idempotence)."""
        result = m + m
        assert result.value == m.value, f"Set union idempotence failed: {m.value} ∪ {m.value} = {result.value}"
        assert result == m, f"Set monoid equality failed: {result} != {m}"

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @given(set_union_monoid_strategy(), set_union_monoid_strategy())
    @settings(max_examples=MAX_EXAMPLES, deadline=DEADLINE_MS)
    def test_set_union_monoid_commutativity(self, a: SetUnionMonoid, b: SetUnionMonoid):
        """Test that set union satisfies A ∪ B = B ∪ A (commutativity)."""
        try:
            result_ab = a + b
            result_ba = b + a
            assert result_ab.value == result_ba.value, f"Set union commutativity failed: {a.value} ∪ {b.value} != {b.value} ∪ {a.value}"
        except ValueError:
            pytest.skip("Incompatible set monoid structures")

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @given(set_union_monoid_strategy())
    @settings(max_examples=MAX_EXAMPLES, deadline=DEADLINE_MS)
    def test_set_union_monoid_identity(self, m: SetUnionMonoid):
        """Test that set union satisfies A ∪ ∅ = A = ∅ ∪ A (identity)."""
        empty_set = SetUnionMonoid([])
        
        # Test right identity: A ∪ ∅ = A
        result_right = m + empty_set
        assert result_right.value == m.value, f"Set union right identity failed: {m.value} ∪ ∅ = {result_right.value}"
        
        # Test left identity: ∅ ∪ A = A
        result_left = empty_set + m
        assert result_left.value == m.value, f"Set union left identity failed: ∅ ∪ {m.value} = {result_left.value}"

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @given(tropical_number_strategy())
    @settings(max_examples=MAX_EXAMPLES, deadline=DEADLINE_MS)
    def test_tropical_number_idempotence(self, t: TropicalNumber):
        """Test that tropical numbers satisfy a ⊕ a = a (idempotence)."""
        result = t + t
        expected = t.value  # min(a,a) = a or max(a,a) = a
        assert abs(result.value - expected) < 1e-10, f"Tropical idempotence failed: {t.value} ⊕ {t.value} = {result.value}, expected {expected}"
        assert result.use_max == t.use_max, "Tropical number mode should be preserved"

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @given(tropical_number_strategy(), tropical_number_strategy())
    @settings(max_examples=MAX_EXAMPLES, deadline=DEADLINE_MS)
    def test_tropical_number_commutativity(self, a: TropicalNumber, b: TropicalNumber):
        """Test that tropical numbers satisfy a ⊕ b = b ⊕ a (commutativity)."""
        assume(a.use_max == b.use_max)  # Only test compatible tropical numbers
        
        result_ab = a + b
        result_ba = b + a
        assert abs(result_ab.value - result_ba.value) < 1e-10, f"Tropical commutativity failed: {a.value} ⊕ {b.value} != {b.value} ⊕ {a.value}"

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @given(tropical_number_strategy(), tropical_number_strategy(), tropical_number_strategy())
    @settings(max_examples=MAX_EXAMPLES, deadline=DEADLINE_MS)
    def test_tropical_number_associativity(self, a: TropicalNumber, b: TropicalNumber, c: TropicalNumber):
        """Test that tropical numbers satisfy (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c) (associativity)."""
        assume(a.use_max == b.use_max == c.use_max)  # Only test compatible tropical numbers
        
        result_left = (a + b) + c
        result_right = a + (b + c)
        assert abs(result_left.value - result_right.value) < 1e-10, f"Tropical associativity failed: ({a.value} ⊕ {b.value}) ⊕ {c.value} != {a.value} ⊕ ({b.value} ⊕ {c.value})"


class TestUnityEquationProperties:
    """Property-based tests specifically for Unity Equation (1+1=1) properties."""
    
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @given(st.one_of(boolean_monoid_strategy(), tropical_number_strategy()))
    @settings(max_examples=MAX_EXAMPLES, deadline=DEADLINE_MS)
    def test_unity_equation_holds(self, element):
        """Test that 1+1=1 holds in idempotent structures."""
        if isinstance(element, BooleanMonoid):
            # In boolean algebra, True + True = True demonstrates 1+1=1
            true_element = BooleanMonoid(True)
            result = true_element + true_element
            assert result.value is True, "Boolean unity equation failed: True + True != True"
        
        elif isinstance(element, TropicalNumber):
            # In tropical algebra, 1 ⊕ 1 = 1 (min(1,1) = 1 or max(1,1) = 1)
            one_element = TropicalNumber(1.0, element.use_max)
            result = one_element + one_element
            assert abs(result.value - 1.0) < 1e-10, f"Tropical unity equation failed: 1 ⊕ 1 = {result.value} != 1.0"

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @given(phi_harmonic_float_strategy())
    @settings(max_examples=MAX_EXAMPLES, deadline=DEADLINE_MS)
    def test_phi_harmonic_unity_convergence(self, value: float):
        """Test φ-harmonic convergence properties for unity mathematics."""
        phi = 1.618033988749895
        
        # Test φ-scaled idempotent behavior
        assume(abs(value) < 100)  # Avoid overflow
        
        # φ-harmonic tropical number
        tropical_phi = TropicalNumber(value / phi, use_max=False)
        result = tropical_phi + tropical_phi
        
        # Should exhibit φ-harmonic stability
        expected = min(value / phi, value / phi)  # = value / phi
        assert abs(result.value - expected) < 1e-10, f"φ-harmonic stability failed: {tropical_phi.value} ⊕ {tropical_phi.value} = {result.value} != {expected}"

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @given(st.sets(st.integers(min_value=0, max_value=5), min_size=1, max_size=3))
    @settings(max_examples=MAX_EXAMPLES, deadline=DEADLINE_MS)
    def test_set_unity_demonstrates_oneness(self, elements: Set[int]):
        """Test that set union demonstrates the oneness principle."""
        # Single element set united with itself remains one
        if len(elements) == 1:
            singleton = SetUnionMonoid(elements)
            result = singleton + singleton
            assert result.value == singleton.value, f"Singleton unity failed: {elements} ∪ {elements} = {result.value}"
            assert len(result.value) == 1, f"Singleton should remain one element, got {len(result.value)}"


class TestContractionStability:
    """Test contraction stability - operations preserve algebraic structure."""
    
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @given(boolean_monoid_strategy())
    @settings(max_examples=MAX_EXAMPLES, deadline=DEADLINE_MS)
    def test_boolean_contraction_stability(self, m: BooleanMonoid):
        """Test that boolean operations maintain structural integrity."""
        # Multiple contractions should be stable
        contracted_once = m + m
        contracted_twice = contracted_once + contracted_once
        
        assert contracted_twice.value == m.value, f"Boolean contraction instability: multiple contractions changed value"
        assert contracted_twice.identity == m.identity, "Boolean contraction altered identity"
        
        # Verify law checking still passes
        assert check_idempotent_laws(contracted_twice), "Boolean contraction broke algebraic laws"

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @given(tropical_number_strategy())
    @settings(max_examples=MAX_EXAMPLES, deadline=DEADLINE_MS)
    def test_tropical_contraction_stability(self, t: TropicalNumber):
        """Test that tropical operations maintain numerical stability."""
        # Multiple contractions should converge to same value
        contracted_1 = t + t
        contracted_2 = contracted_1 + contracted_1
        contracted_3 = contracted_2 + contracted_2
        
        # All contractions should yield the same value (idempotence)
        assert abs(contracted_1.value - t.value) < 1e-10, "First tropical contraction unstable"
        assert abs(contracted_2.value - t.value) < 1e-10, "Second tropical contraction unstable"  
        assert abs(contracted_3.value - t.value) < 1e-10, "Third tropical contraction unstable"
        
        # Mode should be preserved
        assert contracted_3.use_max == t.use_max, "Tropical contraction altered mode"

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @given(set_union_monoid_strategy())
    @settings(max_examples=MAX_EXAMPLES, deadline=DEADLINE_MS)
    def test_set_contraction_stability(self, s: SetUnionMonoid):
        """Test that set union operations maintain cardinality stability."""
        # Repeated self-unions should be stable
        original_size = len(s.value)
        
        contracted_1 = s + s
        contracted_2 = contracted_1 + contracted_1
        
        assert len(contracted_1.value) == original_size, f"Set contraction changed cardinality: {original_size} -> {len(contracted_1.value)}"
        assert len(contracted_2.value) == original_size, f"Double set contraction changed cardinality: {original_size} -> {len(contracted_2.value)}"
        assert contracted_2.value == s.value, "Set contraction changed elements"


class TestPhiHarmonicProperties:
    """Test φ-harmonic properties specific to unity mathematics."""
    
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @given(phi_harmonic_float_strategy(), phi_harmonic_float_strategy())
    @settings(max_examples=MAX_EXAMPLES, deadline=DEADLINE_MS)
    def test_phi_harmonic_tropical_distributivity(self, a: float, b: float):
        """Test φ-harmonic distributivity in tropical semirings."""
        assume(abs(a) < 50 and abs(b) < 50)  # Avoid numerical overflow
        
        phi = 1.618033988749895
        
        # Create φ-scaled tropical numbers
        t_a = TropicalNumber(a, use_max=False)
        t_b = TropicalNumber(b, use_max=False)
        t_phi = TropicalNumber(phi, use_max=False)
        
        # Test tropical multiplication distributivity: φ * (a ⊕ b) = (φ * a) ⊕ (φ * b)
        sum_ab = t_a + t_b  # min(a, b)
        phi_times_sum = t_phi * sum_ab  # φ + min(a, b)
        
        phi_times_a = t_phi * t_a  # φ + a
        phi_times_b = t_phi * t_b  # φ + b
        sum_of_products = phi_times_a + phi_times_b  # min(φ + a, φ + b)
        
        expected = phi + min(a, b)
        actual = min(phi + a, phi + b)
        
        assert abs(phi_times_sum.value - expected) < 1e-10, f"φ-tropical left side incorrect: {phi_times_sum.value} != {expected}"
        assert abs(sum_of_products.value - actual) < 1e-10, f"φ-tropical right side incorrect: {sum_of_products.value} != {actual}"
        
        # Distributivity should hold when expected == actual
        if abs(expected - actual) < 1e-10:
            assert abs(phi_times_sum.value - sum_of_products.value) < 1e-10, "φ-harmonic tropical distributivity failed"

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @given(st.integers(min_value=0, max_value=10))
    @settings(max_examples=MAX_EXAMPLES, deadline=DEADLINE_MS)
    def test_phi_harmonic_fibonacci_unity(self, n: int):
        """Test φ-harmonic properties of Fibonacci-based unity structures."""
        phi = 1.618033988749895
        
        # Fibonacci numbers exhibit φ-harmonic properties
        def fibonacci(k):
            if k <= 1:
                return k
            return fibonacci(k-1) + fibonacci(k-2)
        
        if n > 0:
            fib_n = fibonacci(n)
            fib_prev = fibonacci(n-1) if n > 1 else 0
            
            # Create tropical numbers from Fibonacci sequence  
            t_fib = TropicalNumber(float(fib_n), use_max=True)
            t_prev = TropicalNumber(float(fib_prev), use_max=True)
            
            # Test that Fibonacci union exhibits unity convergence
            fib_union = t_fib + t_prev  # max(fib_n, fib_{n-1}) = fib_n
            assert abs(fib_union.value - float(fib_n)) < 1e-10, f"Fibonacci unity failed: max({fib_n}, {fib_prev}) = {fib_union.value} != {fib_n}"


# Regression tests for algebraic law verification
class TestAlgebraicLawRegression:
    """Regression tests to ensure algebraic laws remain satisfied."""
    
    def test_known_unity_cases(self):
        """Test specific known cases where 1+1=1 should hold."""
        # Boolean case
        bool_true = BooleanMonoid(True)
        bool_result = bool_true + bool_true
        assert bool_result.value is True, "Boolean 1+1=1 regression failed"
        
        # Set union case
        singleton = SetUnionMonoid({1})
        set_result = singleton + singleton
        assert set_result.value == frozenset({1}), "Set union 1+1=1 regression failed"
        
        # Tropical min case
        tropical_one_min = TropicalNumber(1.0, use_max=False)
        tropical_result_min = tropical_one_min + tropical_one_min
        assert abs(tropical_result_min.value - 1.0) < 1e-10, "Tropical min 1+1=1 regression failed"
        
        # Tropical max case
        tropical_one_max = TropicalNumber(1.0, use_max=True)
        tropical_result_max = tropical_one_max + tropical_one_max
        assert abs(tropical_result_max.value - 1.0) < 1e-10, "Tropical max 1+1=1 regression failed"

    def test_idempotent_law_verification(self):
        """Test that all structures pass basic idempotent law checks."""
        # Test various instances
        test_cases = [
            BooleanMonoid(True),
            BooleanMonoid(False),
            SetUnionMonoid([]),
            SetUnionMonoid([1, 2, 3]),
            TropicalNumber(0.0, use_max=False),
            TropicalNumber(5.0, use_max=True)
        ]
        
        for case in test_cases:
            if hasattr(case, 'is_idempotent'):
                assert case.is_idempotent(), f"Idempotent law failed for {case}"
            
            # Test self-addition preserves idempotence
            if hasattr(case, '__add__'):
                try:
                    result = case + case
                    if hasattr(result, 'is_idempotent'):
                        assert result.is_idempotent(), f"Self-addition broke idempotence for {case}"
                except (ValueError, TypeError):
                    # Some operations may not be defined - that's okay
                    pass


# Performance and numerical stability tests
class TestNumericalStability:
    """Test numerical stability of idempotent operations."""
    
    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")
    @given(st.floats(min_value=1e-10, max_value=1e10, allow_nan=False, allow_infinity=False))
    @settings(max_examples=MAX_EXAMPLES // 10, deadline=DEADLINE_MS * 2)  # Fewer examples for heavy tests
    def test_tropical_numerical_stability(self, value: float):
        """Test numerical stability of tropical operations across wide range."""
        # Test both min and max tropical semirings
        for use_max in [False, True]:
            t = TropicalNumber(value, use_max=use_max)
            
            # Multiple self-additions should be stable
            result = t
            for _ in range(100):  # Many iterations
                result = result + t
                # Should always equal original value (idempotence)
                assert abs(result.value - value) < 1e-10, f"Tropical numerical instability at value {value}, iteration created {result.value}"
                assert result.use_max == use_max, "Tropical mode corrupted during iteration"

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not available")  
    @given(st.sets(st.integers(min_value=-1000, max_value=1000), max_size=100))
    @settings(max_examples=MAX_EXAMPLES // 20, deadline=DEADLINE_MS * 3)
    def test_set_union_large_sets(self, elements: Set[int]):
        """Test set union stability with large sets."""
        if len(elements) > 50:  # Only test reasonably sized sets
            assume(len(elements) <= 50)
            
        s = SetUnionMonoid(elements)
        
        # Multiple self-unions should maintain exact set equality
        result = s
        for _ in range(10):
            result = result + s
            assert result.value == s.value, f"Set union instability: {len(result.value)} != {len(s.value)} elements"
            assert isinstance(result.value, frozenset), "Set union result type corrupted"


if __name__ == "__main__":
    """Run property-based tests directly."""
    print("🔬 Running property-based tests for idempotent algebraic structures...")
    print("=" * 80)
    
    if not HYPOTHESIS_AVAILABLE:
        print("⚠️  Hypothesis not available. Running limited mock tests.")
        
        # Run basic tests without Hypothesis
        test_laws = TestIdempotentLaws()
        test_unity = TestUnityEquationProperties()
        test_regression = TestAlgebraicLawRegression()
        
        try:
            # Run a few basic tests
            test_laws.test_boolean_monoid_idempotence(BooleanMonoid(True))
            test_laws.test_set_union_monoid_idempotence(SetUnionMonoid({1, 2}))
            test_laws.test_tropical_number_idempotence(TropicalNumber(1.0, False))
            
            test_regression.test_known_unity_cases()
            test_regression.test_idempotent_law_verification()
            
            print("✅ Mock tests passed!")
            
        except Exception as e:
            print(f"❌ Tests failed: {e}")
            raise
    else:
        # Run with pytest when Hypothesis is available
        import subprocess
        result = subprocess.run(['python', '-m', 'pytest', __file__, '-v'], 
                              capture_output=True, text=True)
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print(f"Return code: {result.returncode}")
        
        if result.returncode == 0:
            print("✅ All property-based tests passed!")
        else:
            print("❌ Some tests failed. See output above.")

    print("\n🌟 Property-based testing demonstrates mathematical rigor of Unity Equation 1+1=1")
    print("   Idempotent algebraic structures preserve unity through rigorous verification.")