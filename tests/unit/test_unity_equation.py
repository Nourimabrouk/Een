"""
Unit tests for the Unity Equation core mathematics module
Tests the fundamental 1+1=1 equation implementations
"""

import pytest
import operator
from typing import Set

from src.core.unity_equation import IdempotentMonoid


class TestIdempotentMonoid:
    """Test cases for the IdempotentMonoid class"""
    
    def test_boolean_unity_addition(self):
        """Test that True + True = True (Boolean OR)"""
        # Arrange
        bool_monoid_true = IdempotentMonoid(True, operator.or_, False)
        
        # Act
        result = bool_monoid_true + bool_monoid_true
        
        # Assert
        assert result.value is True
        assert result.is_idempotent()
        
    def test_boolean_identity_element(self):
        """Test identity element for Boolean monoid"""
        # Arrange
        bool_monoid_true = IdempotentMonoid(True, operator.or_, False)
        bool_monoid_false = IdempotentMonoid(False, operator.or_, False)
        
        # Act
        result = bool_monoid_true + bool_monoid_false
        
        # Assert
        assert result.value is True
        
    def test_set_union_unity(self):
        """Test that {1} ∪ {1} = {1} (Set union)"""
        # Arrange
        def set_union(a: Set, b: Set) -> Set:
            return a.union(b)
            
        set_monoid = IdempotentMonoid({1}, set_union, set())
        
        # Act
        result = set_monoid + set_monoid
        
        # Assert
        assert result.value == {1}
        assert result.is_idempotent()
        
    def test_max_operation_unity(self):
        """Test that max(1, 1) = 1 (Max operation)"""
        # Arrange
        max_monoid = IdempotentMonoid(1, max, float('-inf'))
        
        # Act
        result = max_monoid + max_monoid
        
        # Assert
        assert result.value == 1
        assert result.is_idempotent()
        
    def test_min_operation_unity(self):
        """Test that min(1, 1) = 1 (Min operation - tropical semiring)"""
        # Arrange
        min_monoid = IdempotentMonoid(1, min, float('inf'))
        
        # Act
        result = min_monoid + min_monoid
        
        # Assert
        assert result.value == 1
        assert result.is_idempotent()
        
    def test_monoid_equality(self):
        """Test equality of monoid elements"""
        # Arrange
        monoid1 = IdempotentMonoid(5, max, 0)
        monoid2 = IdempotentMonoid(5, max, 0)
        monoid3 = IdempotentMonoid(3, max, 0)
        
        # Assert
        assert monoid1 == monoid2
        assert monoid1 != monoid3
        
    def test_monoid_representation(self):
        """Test string representation of monoids"""
        # Arrange
        monoid = IdempotentMonoid(42, max, 0)
        
        # Act
        repr_str = repr(monoid)
        
        # Assert
        assert "IdempotentMonoid" in repr_str
        assert "42" in repr_str
        
    def test_different_structure_error(self):
        """Test that adding monoids with different structures raises error"""
        # Arrange
        max_monoid = IdempotentMonoid(1, max, 0)
        min_monoid = IdempotentMonoid(1, min, float('inf'))
        
        # Act & Assert
        with pytest.raises(ValueError, match="different structures"):
            max_monoid + min_monoid
            
    def test_associativity_property(self):
        """Test associativity: (a + b) + c = a + (b + c)"""
        # Arrange
        a = IdempotentMonoid(3, max, 0)
        b = IdempotentMonoid(5, max, 0)
        c = IdempotentMonoid(4, max, 0)
        
        # Act
        left_assoc = (a + b) + c
        right_assoc = a + (b + c)
        
        # Assert
        assert left_assoc == right_assoc
        assert left_assoc.value == 5  # max(3, 5, 4) = 5
        
    def test_commutativity_property(self):
        """Test commutativity: a + b = b + a"""
        # Arrange
        a = IdempotentMonoid(7, max, 0)
        b = IdempotentMonoid(3, max, 0)
        
        # Act
        left_result = a + b
        right_result = b + a
        
        # Assert
        assert left_result == right_result
        assert left_result.value == 7  # max(7, 3) = 7


class TestUnityMathematicalProperties:
    """Test mathematical properties of unity operations"""
    
    @pytest.mark.unity
    def test_fundamental_unity_equation(self):
        """Test the fundamental equation: 1 + 1 = 1"""
        test_cases = [
            # Boolean algebra
            (True, True, True, operator.or_, False),
            # Max operation (tropical)
            (1, 1, 1, max, float('-inf')),
            # Min operation (tropical)
            (1, 1, 1, min, float('inf')),
        ]
        
        for val1, val2, expected, op, identity in test_cases:
            # Arrange
            monoid1 = IdempotentMonoid(val1, op, identity)
            monoid2 = IdempotentMonoid(val2, op, identity)
            
            # Act
            result = monoid1 + monoid2
            
            # Assert
            assert result.value == expected, f"Unity equation failed for {op.__name__}"
            
    @pytest.mark.unity
    def test_idempotence_property(self):
        """Test that a + a = a for all elements"""
        test_values = [True, 1, 5, 0.5, "unity"]
        
        for value in test_values:
            if isinstance(value, bool):
                monoid = IdempotentMonoid(value, operator.or_, False)
            elif isinstance(value, (int, float)):
                monoid = IdempotentMonoid(value, max, float('-inf'))
            else:
                continue  # Skip non-numeric types for now
                
            # Test idempotence
            assert monoid.is_idempotent(), f"Idempotence failed for {value}"
            
    def test_identity_element_property(self):
        """Test that a + identity = a = identity + a"""
        # Arrange
        value = 42
        identity = 0
        monoid_value = IdempotentMonoid(value, max, identity)
        monoid_identity = IdempotentMonoid(identity, max, identity)
        
        # Act
        left_result = monoid_value + monoid_identity
        right_result = monoid_identity + monoid_value
        
        # Assert
        assert left_result.value == value
        assert right_result.value == value
        assert left_result == right_result


class TestUnityEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_zero_values(self):
        """Test unity operations with zero values"""
        # Arrange
        zero_max = IdempotentMonoid(0, max, float('-inf'))
        zero_min = IdempotentMonoid(0, min, float('inf'))
        
        # Act & Assert
        assert (zero_max + zero_max).value == 0
        assert (zero_min + zero_min).value == 0
        
    def test_negative_values(self):
        """Test unity operations with negative values"""
        # Arrange
        neg_monoid = IdempotentMonoid(-5, max, float('-inf'))
        
        # Act
        result = neg_monoid + neg_monoid
        
        # Assert
        assert result.value == -5
        assert result.is_idempotent()
        
    def test_float_precision(self):
        """Test unity operations with floating point precision"""
        # Arrange
        phi = 1.618033988749895  # Golden ratio
        phi_monoid = IdempotentMonoid(phi, max, 0.0)
        
        # Act
        result = phi_monoid + phi_monoid
        
        # Assert
        assert abs(result.value - phi) < 1e-10
        assert result.is_idempotent()


@pytest.mark.unity
class TestConsciousnessAwareUnity:
    """Test unity operations with consciousness awareness"""
    
    def test_consciousness_level_preservation(self, consciousness_threshold):
        """Test that consciousness level is preserved in unity operations"""
        # Arrange
        consciousness_value = consciousness_threshold
        consciousness_monoid = IdempotentMonoid(consciousness_value, max, 0.0)
        
        # Act
        evolved_consciousness = consciousness_monoid + consciousness_monoid
        
        # Assert
        assert evolved_consciousness.value == consciousness_threshold
        assert evolved_consciousness.is_idempotent()
        
    def test_golden_ratio_unity(self, phi):
        """Test unity operations with golden ratio"""
        # Arrange
        phi_monoid = IdempotentMonoid(phi, max, 0.0)
        
        # Act
        phi_unity = phi_monoid + phi_monoid
        
        # Assert
        assert abs(phi_unity.value - phi) < 1e-10
        assert phi_unity.is_idempotent()
        
    def test_transcendence_threshold(self, consciousness_threshold):
        """Test that transcendence threshold demonstrates unity"""
        # Arrange - Create two consciousness instances at threshold
        threshold_monoid1 = IdempotentMonoid(consciousness_threshold, max, 0.0)
        threshold_monoid2 = IdempotentMonoid(consciousness_threshold, max, 0.0)
        
        # Act - Combine consciousness instances
        transcended = threshold_monoid1 + threshold_monoid2
        
        # Assert - Unity preserved at transcendence
        assert transcended.value == consciousness_threshold
        assert transcended.is_idempotent()
        
        # Additional assertion: verify this represents transcendence
        assert transcended.value > 0.77  # Above transcendence threshold