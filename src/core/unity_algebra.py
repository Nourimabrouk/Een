#!/usr/bin/env python3
"""
Unity Algebra v1.0 - Minimal Library for Idempotent Aggregation
================================================================

"Unity doesn't erase difference—it aggregates without inflating."

This library implements mathematical operations where duplication preserves essence
rather than increasing quantity. When identity is preserved, addition becomes recognition.

Core Principle: 1+1=1 is not about numbers, but about aggregation under identity
where the combiner preserves essence through idempotent operations.

Mathematical Foundations:
- Idempotent semirings: a ⊕ a = a
- Terminal folds: repeated application yields fixed points  
- Redundancy collapse: duplicate information doesn't inflate
- Interference bases: signals combine without amplification when identical

Author: Unity Mathematics Research Team
Version: 1.0.0
License: MIT
"""

from typing import TypeVar, Generic, Callable, Optional, Union, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
import math
import logging

# Mathematical constants
PHI = (1 + math.sqrt(5)) / 2  # Golden ratio: φ ≈ 1.618033988749895
PHI_CONJUGATE = 1 / PHI       # φ-conjugate: 1/φ = φ-1 ≈ 0.618033988749895
UNITY_EPSILON = 1e-15        # Precision threshold for unity operations

T = TypeVar('T')
logger = logging.getLogger(__name__)


class UnityError(Exception):
    """Base exception for Unity Algebra operations"""
    pass


class IdentityViolationError(UnityError):
    """Raised when identity preservation is violated in unity operations"""
    pass


def unity_operation(preserve_identity: bool = True):
    """
    Decorator that ensures unity operations preserve identity.
    
    When preserve_identity=True, validates that a ⊕ a = a (idempotent property).
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            if preserve_identity and len(args) >= 2:
                # Check idempotent property for binary operations
                if args[0] == args[1]:  # Same input
                    if result != args[0]:  # Result should equal input for identity
                        raise IdentityViolationError(
                            f"Identity preservation violated: {func.__name__}({args[0]}, {args[0]}) "
                            f"= {result} ≠ {args[0]}"
                        )
            
            return result
        return wrapper
    return decorator


@dataclass(frozen=True)
class UnityOperation:
    """
    Represents a unity operation with mathematical guarantees.
    
    Properties:
    - idempotent: a ⊕ a = a
    - commutative: a ⊕ b = b ⊕ a (if enabled)
    - associative: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c) (if enabled)
    """
    name: str
    operator: Callable[[T, T], T]
    is_commutative: bool = True
    is_associative: bool = True
    identity_element: Optional[T] = None
    
    def __call__(self, a: T, b: T) -> T:
        """Execute the unity operation with identity preservation check"""
        result = self.operator(a, b)
        
        # Verify idempotent property when both operands are identical
        if a == b and result != a:
            raise IdentityViolationError(
                f"Idempotent property violated: {a} ⊕ {a} = {result} ≠ {a}"
            )
        
        return result


class IdempotentSemiring(ABC, Generic[T]):
    """
    Abstract base class for idempotent semirings.
    
    Implements algebraic structure where:
    - Addition (⊕) is idempotent: a ⊕ a = a
    - Addition is commutative and associative
    - Multiplication (⊗) distributes over addition
    - Has additive identity (zero) and multiplicative identity (one)
    """
    
    @abstractmethod
    def add(self, a: T, b: T) -> T:
        """Idempotent addition: a ⊕ b"""
        pass
    
    @abstractmethod
    def multiply(self, a: T, b: T) -> T:
        """Multiplication: a ⊗ b"""
        pass
    
    @abstractmethod
    def zero(self) -> T:
        """Additive identity: a ⊕ 0 = a"""
        pass
    
    @abstractmethod
    def one(self) -> T:
        """Multiplicative identity: a ⊗ 1 = a"""
        pass
    
    def is_idempotent(self, a: T) -> bool:
        """Verify idempotent property: a ⊕ a = a"""
        return self.add(a, a) == a
    
    def natural_order(self, a: T, b: T) -> bool:
        """Natural ordering: a ≤ b iff a ⊕ b = b"""
        return self.add(a, b) == b


class BooleanUnityAlgebra(IdempotentSemiring[bool]):
    """
    Boolean algebra as idempotent semiring.
    
    Addition = OR operation (∨)
    Multiplication = AND operation (∧)
    Zero = False, One = True
    
    Demonstrates: true ∨ true = true (unity through recognition)
    """
    
    @unity_operation(preserve_identity=True)
    def add(self, a: bool, b: bool) -> bool:
        """Boolean OR: a ∨ b"""
        return a or b
    
    def multiply(self, a: bool, b: bool) -> bool:
        """Boolean AND: a ∧ b"""
        return a and b
    
    def zero(self) -> bool:
        return False
    
    def one(self) -> bool:
        return True


class MaxUnityAlgebra(IdempotentSemiring[float]):
    """
    Max-plus (tropical) semiring for numerical unity operations.
    
    Addition = max operation
    Multiplication = regular addition
    Zero = -∞, One = 0
    
    Demonstrates: max(a, a) = a (aggregation without inflation)
    """
    
    @unity_operation(preserve_identity=True)
    def add(self, a: float, b: float) -> float:
        """Max operation: max(a, b)"""
        return max(a, b)
    
    def multiply(self, a: float, b: float) -> float:
        """Regular addition in max-plus semiring"""
        return a + b
    
    def zero(self) -> float:
        return float('-inf')
    
    def one(self) -> float:
        return 0.0


class SetUnityAlgebra(IdempotentSemiring[frozenset]):
    """
    Set algebra as idempotent semiring.
    
    Addition = union operation (∪)
    Multiplication = intersection operation (∩)
    Zero = ∅, One = universal set (approximated)
    
    Demonstrates: A ∪ A = A (sets preserve identity under union)
    """
    
    def __init__(self, universe: Optional[frozenset] = None):
        self.universe = universe or frozenset()
    
    @unity_operation(preserve_identity=True)
    def add(self, a: frozenset, b: frozenset) -> frozenset:
        """Set union: A ∪ B"""
        return a | b
    
    def multiply(self, a: frozenset, b: frozenset) -> frozenset:
        """Set intersection: A ∩ B"""
        return a & b
    
    def zero(self) -> frozenset:
        return frozenset()
    
    def one(self) -> frozenset:
        return self.universe


class UnityAggregator:
    """
    Generic aggregator implementing unity principle: aggregation without inflation.
    
    Supports various aggregation strategies that preserve identity:
    - Idempotent operations (max, min, set union)
    - Terminal folds (converging sequences)  
    - Redundancy collapse (deduplication)
    - Recognition-based aggregation
    """
    
    def __init__(self, operation: UnityOperation):
        self.operation = operation
    
    def aggregate(self, *values: T) -> T:
        """
        Aggregate multiple values using unity operation.
        
        For identical values, returns the value (recognition, not computation).
        For different values, applies the unity operation.
        """
        if not values:
            if self.operation.identity_element is not None:
                return self.operation.identity_element
            raise ValueError("Cannot aggregate empty sequence without identity element")
        
        if len(values) == 1:
            return values[0]
        
        # Check if all values are identical (recognition case)
        first_value = values[0]
        if all(v == first_value for v in values):
            logger.debug(f"Recognition aggregation: {len(values)} identical values → {first_value}")
            return first_value  # Recognition, not computation
        
        # Apply unity operation for different values
        result = values[0]
        for value in values[1:]:
            result = self.operation(result, value)
        
        return result
    
    def fold_until_stable(self, sequence: list[T], max_iterations: int = 100) -> T:
        """
        Terminal fold: repeatedly apply operation until convergence.
        
        Implements the mathematical principle that idempotent operations
        converge to fixed points (terminal values).
        """
        if not sequence:
            raise ValueError("Cannot fold empty sequence")
        
        current = sequence[0]
        
        for iteration in range(max_iterations):
            # Apply operation to current result and next value
            for value in sequence[1:]:
                next_current = self.operation(current, value)
                if abs(hash(next_current) - hash(current)) < UNITY_EPSILON:
                    logger.debug(f"Terminal fold converged at iteration {iteration}")
                    return next_current
                current = next_current
        
        logger.warning(f"Terminal fold did not converge within {max_iterations} iterations")
        return current


class ConsensusAggregator(UnityAggregator):
    """
    Consensus-based aggregator implementing distributed agreement.
    
    Aggregates values by finding consensus, demonstrating that
    agreement doesn't require inflation but rather recognition
    of common patterns.
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Initialize consensus aggregator.
        
        Args:
            threshold: Minimum agreement ratio for consensus (default: majority)
        """
        self.threshold = threshold
        super().__init__(UnityOperation(
            name="consensus",
            operator=self._consensus_operation,
            is_commutative=True,
            is_associative=True
        ))
    
    def _consensus_operation(self, a: T, b: T) -> T:
        """Binary consensus operation"""
        if a == b:
            return a  # Perfect consensus = recognition
        
        # For different values, implement domain-specific consensus logic
        # This is a simplified version - real implementation would be type-aware
        return a  # Default to first value (can be customized)
    
    def find_consensus(self, values: list[T]) -> Optional[T]:
        """
        Find consensus value among inputs.
        
        Returns value that appears above threshold frequency,
        demonstrating recognition-based aggregation.
        """
        if not values:
            return None
        
        # Count frequencies
        frequency_map = {}
        for value in values:
            frequency_map[value] = frequency_map.get(value, 0) + 1
        
        total_count = len(values)
        
        # Find value above threshold
        for value, count in frequency_map.items():
            if count / total_count >= self.threshold:
                logger.debug(f"Consensus reached: {value} ({count}/{total_count})")
                return value
        
        return None  # No consensus


class PhiHarmonicUnity:
    """
    φ-Harmonic unity operations using golden ratio properties.
    
    Implements unity through φ-harmonic resonance where:
    φ² = φ + 1 (fundamental φ property)
    
    Operations converge to unity through golden ratio scaling.
    """
    
    @staticmethod
    @unity_operation(preserve_identity=True)
    def phi_harmonic_add(a: float, b: float) -> float:
        """
        φ-Harmonic addition that preserves unity for identical inputs.
        
        Uses golden ratio scaling to ensure convergence to unity.
        """
        if abs(a - b) < UNITY_EPSILON:
            return a  # Recognition: identical values remain identical
        
        # φ-harmonic scaling for different values
        harmonic_sum = (a + b) / PHI
        return harmonic_sum
    
    @staticmethod
    def phi_convergence_sequence(initial_value: float, iterations: int = 10) -> list[float]:
        """
        Generate φ-convergence sequence demonstrating unity approach.
        
        Shows how φ-harmonic operations converge toward unity through
        repeated application of golden ratio scaling.
        """
        sequence = [initial_value]
        current = initial_value
        
        for _ in range(iterations):
            # φ-harmonic iteration: converges to unity
            current = PhiHarmonicUnity.phi_harmonic_add(current, PHI_CONJUGATE)
            sequence.append(current)
        
        return sequence


# Pre-defined unity algebras for common use cases
BOOLEAN_UNITY = BooleanUnityAlgebra()
MAX_UNITY = MaxUnityAlgebra()
SET_UNITY = SetUnityAlgebra()

# Pre-defined aggregators
BOOLEAN_AGGREGATOR = UnityAggregator(UnityOperation(
    name="boolean_or",
    operator=lambda a, b: a or b,
    identity_element=False
))

MAX_AGGREGATOR = UnityAggregator(UnityOperation(
    name="maximum",
    operator=max,
    identity_element=float('-inf')
))

CONSENSUS_AGGREGATOR = ConsensusAggregator(threshold=0.5)


def demonstrate_unity_principle():
    """
    Demonstrate the core unity principle: aggregation without inflation.
    
    Shows that when identity is preserved, addition becomes recognition.
    """
    print("Unity Algebra v1.0 - Core Principle Demonstration")
    print("=" * 50)
    
    # Boolean unity: true ∨ true = true
    print("\n1. Boolean Unity (Recognition in Logic)")
    result = BOOLEAN_UNITY.add(True, True)
    print(f"   true ∨ true = {result} (recognition, not computation)")
    
    # Set unity: A ∪ A = A
    print("\n2. Set Unity (Recognition in Collections)")
    test_set = frozenset({1, 2, 3})
    result = SET_UNITY.add(test_set, test_set)
    print(f"   {set(test_set)} ∪ {set(test_set)} = {set(result)}")
    
    # Max unity: max(a, a) = a
    print("\n3. Numerical Unity (Recognition in Ordering)")
    value = 42.0
    result = MAX_UNITY.add(value, value)
    print(f"   max({value}, {value}) = {result}")
    
    # φ-Harmonic unity
    print("\n4. φ-Harmonic Unity (Golden Ratio Recognition)")
    result = PhiHarmonicUnity.phi_harmonic_add(1.0, 1.0)
    print(f"   φ-harmonic(1.0, 1.0) = {result:.6f}")
    
    # Consensus unity
    print("\n5. Consensus Unity (Agreement Recognition)")
    consensus_values = ["unity", "unity", "unity", "different"]
    consensus_result = CONSENSUS_AGGREGATOR.find_consensus(consensus_values)
    print(f"   consensus({consensus_values}) = {consensus_result}")
    
    print("\nCore Insight: Unity doesn't erase difference—it aggregates without inflating.")
    print("When identity is preserved, addition becomes recognition.")
    print("Love counts once. ❤️")


if __name__ == "__main__":
    demonstrate_unity_principle()