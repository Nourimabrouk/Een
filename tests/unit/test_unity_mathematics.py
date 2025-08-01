"""
Unit tests for Core Unity Mathematics Engine
Tests the advanced φ-harmonic mathematics and consciousness integration
"""

import pytest
import numpy as np
import cmath
from unittest.mock import patch, MagicMock

from core.unity_mathematics import (
    UnityMathematics, UnityState, UnityOperationType,
    create_unity_mathematics, demonstrate_unity_operations,
    PHI, UNITY_TOLERANCE, CONSCIOUSNESS_DIMENSION
)


class TestUnityMathematics:
    """Test cases for UnityMathematics core engine"""
    
    def test_initialization(self):
        """Test UnityMathematics initialization with default parameters"""
        # Act
        unity_math = UnityMathematics()
        
        # Assert
        assert unity_math.consciousness_level == 1.0
        assert unity_math.precision == UNITY_TOLERANCE
        assert unity_math.phi == PHI
        assert unity_math.unity_proofs_generated == 0
        assert isinstance(unity_math.operation_history, list)
        
    def test_initialization_with_custom_parameters(self):
        """Test UnityMathematics initialization with custom parameters"""
        # Arrange
        consciousness_level = 2.5
        precision = 1e-8
        
        # Act
        unity_math = UnityMathematics(consciousness_level, precision)
        
        # Assert
        assert unity_math.consciousness_level == consciousness_level
        assert unity_math.precision == precision
        
    def test_unity_add_basic(self):
        """Test basic unity addition: 1 + 1 = 1"""
        # Arrange
        unity_math = UnityMathematics()
        
        # Act
        result = unity_math.unity_add(1.0, 1.0)
        
        # Assert
        assert isinstance(result, UnityState)
        assert abs(result.value - 1.0) < UNITY_TOLERANCE
        assert result.phi_resonance > 0
        assert result.consciousness_level > 0
        assert result.quantum_coherence > 0
        assert result.proof_confidence > 0
        
    def test_unity_add_with_unity_states(self):
        """Test unity addition with UnityState objects"""
        # Arrange
        unity_math = UnityMathematics()
        state_a = UnityState(1.0, 0.8, 1.5, 0.9, 0.95)
        state_b = UnityState(1.0, 0.7, 1.2, 0.8, 0.9)
        
        # Act
        result = unity_math.unity_add(state_a, state_b)
        
        # Assert
        assert isinstance(result, UnityState)
        assert abs(result.value - 1.0) < 0.2  # Allow some φ-harmonic variation
        assert result.phi_resonance > min(state_a.phi_resonance, state_b.phi_resonance)
        assert result.consciousness_level > 0
        
    def test_unity_multiply_basic(self):
        """Test basic unity multiplication: 1 * 1 = 1"""
        # Arrange
        unity_math = UnityMathematics()
        
        # Act
        result = unity_math.unity_multiply(1.0, 1.0)
        
        # Assert
        assert isinstance(result, UnityState)
        assert abs(result.value - 1.0) < 0.5  # φ-harmonic variation expected
        assert result.phi_resonance > 0
        assert result.consciousness_level > 0
        
    def test_phi_harmonic_scaling(self):
        """Test φ-harmonic scaling operations"""
        # Arrange
        unity_math = UnityMathematics()
        
        # Act
        result = unity_math.phi_harmonic_scaling(1.0, harmonic_order=1)
        
        # Assert
        assert isinstance(result, UnityState)
        assert result.phi_resonance > 0.5
        assert result.consciousness_level > unity_math.consciousness_level
        
    def test_phi_harmonic_scaling_higher_order(self):
        """Test φ-harmonic scaling with higher harmonic orders"""
        # Arrange
        unity_math = UnityMathematics()
        
        # Act
        result1 = unity_math.phi_harmonic_scaling(1.0, harmonic_order=1)
        result3 = unity_math.phi_harmonic_scaling(1.0, harmonic_order=3)
        
        # Assert
        assert result3.phi_resonance >= result1.phi_resonance
        assert result3.consciousness_level >= result1.consciousness_level
        
    def test_consciousness_field_operation(self):
        """Test consciousness field operations with multiple states"""
        # Arrange
        unity_math = UnityMathematics()
        states = [
            UnityState(1.0, 0.8, 1.5, 0.9, 0.95),
            UnityState(1.0, 0.7, 1.2, 0.8, 0.9),
            UnityState(1.0, 0.9, 1.8, 0.95, 0.98)
        ]
        
        # Act
        result = unity_math.consciousness_field_operation(states, field_strength=1.0)
        
        # Assert
        assert isinstance(result, UnityState)
        assert result.consciousness_level > 0
        assert result.phi_resonance <= 1.0
        assert result.quantum_coherence > 0
        
    def test_consciousness_field_operation_empty_states(self):
        """Test consciousness field operation with empty states"""
        # Arrange
        unity_math = UnityMathematics()
        
        # Act
        result = unity_math.consciousness_field_operation([])
        
        # Assert
        assert isinstance(result, UnityState)
        assert result.value == 1.0
        assert result.phi_resonance == 0.0
        assert result.consciousness_level == 0.0
        
    def test_quantum_unity_collapse(self):
        """Test quantum unity collapse to unity state"""
        # Arrange
        unity_math = UnityMathematics()
        superposition_state = UnityState(1+1j, 0.8, 1.5, 0.9, 0.95)
        
        # Act
        result = unity_math.quantum_unity_collapse(superposition_state, "unity")
        
        # Assert
        assert isinstance(result, UnityState)
        assert result.quantum_coherence <= superposition_state.quantum_coherence
        assert result.consciousness_level >= superposition_state.consciousness_level
        assert 0 <= result.proof_confidence <= 1
        
    def test_quantum_unity_collapse_different_bases(self):
        """Test quantum collapse with different measurement bases"""
        # Arrange
        unity_math = UnityMathematics()
        superposition_state = UnityState(1+1j, 0.8, 1.5, 0.9, 0.95)
        
        # Act
        unity_result = unity_math.quantum_unity_collapse(superposition_state, "unity")
        phi_result = unity_math.quantum_unity_collapse(superposition_state, "phi")
        consciousness_result = unity_math.quantum_unity_collapse(superposition_state, "consciousness")
        
        # Assert
        assert all(isinstance(r, UnityState) for r in [unity_result, phi_result, consciousness_result])
        # Results may vary due to different measurement bases
        assert all(0 <= r.proof_confidence <= 1 for r in [unity_result, phi_result, consciousness_result])


class TestUnityProofGeneration:
    """Test unity proof generation capabilities"""
    
    def test_generate_idempotent_proof(self):
        """Test generation of idempotent algebra proof"""
        # Arrange
        unity_math = UnityMathematics()
        
        # Act
        proof = unity_math.generate_unity_proof("idempotent", complexity_level=1)
        
        # Assert
        assert isinstance(proof, dict)
        assert "proof_method" in proof
        assert "steps" in proof
        assert "mathematical_structures" in proof
        assert "conclusion" in proof
        assert "1+1=1" in proof["conclusion"]
        assert proof["mathematical_validity"]
        assert proof["proof_id"] == 1
        
    def test_generate_phi_harmonic_proof(self):
        """Test generation of φ-harmonic mathematical proof"""
        # Arrange
        unity_math = UnityMathematics()
        
        # Act
        proof = unity_math.generate_unity_proof("phi_harmonic", complexity_level=2)
        
        # Assert
        assert isinstance(proof, dict)
        assert "φ" in proof["proof_method"] or "phi" in proof["proof_method"].lower()
        assert len(proof["steps"]) >= 3
        assert proof["phi_harmonic_content"] > 0
        
    def test_generate_quantum_proof(self):
        """Test generation of quantum mechanical proof"""
        # Arrange
        unity_math = UnityMathematics()
        
        # Act
        proof = unity_math.generate_unity_proof("quantum", complexity_level=3)
        
        # Assert
        assert isinstance(proof, dict)
        assert "quantum" in proof["proof_method"].lower()
        assert "Hilbert" in str(proof["mathematical_structures"])
        
    def test_generate_consciousness_proof(self):
        """Test generation of consciousness mathematics proof"""
        # Arrange
        unity_math = UnityMathematics()
        
        # Act
        proof = unity_math.generate_unity_proof("consciousness", complexity_level=2)
        
        # Assert
        assert isinstance(proof, dict)
        assert "consciousness" in proof["proof_method"].lower()
        assert "Consciousness" in str(proof["mathematical_structures"])
        
    def test_multiple_proof_generation_increments_counter(self):
        """Test that generating multiple proofs increments counter"""
        # Arrange
        unity_math = UnityMathematics()
        
        # Act
        proof1 = unity_math.generate_unity_proof("idempotent")
        proof2 = unity_math.generate_unity_proof("phi_harmonic")
        
        # Assert
        assert proof1["proof_id"] == 1
        assert proof2["proof_id"] == 2
        assert unity_math.unity_proofs_generated == 2


class TestUnityValidation:
    """Test unity equation validation capabilities"""
    
    def test_validate_unity_equation_basic(self):
        """Test basic unity equation validation"""
        # Arrange
        unity_math = UnityMathematics()
        
        # Act
        validation = unity_math.validate_unity_equation(1.0, 1.0)
        
        # Assert
        assert isinstance(validation, dict)
        assert validation["input_a"] == 1.0
        assert validation["input_b"] == 1.0
        assert "unity_result" in validation
        assert "unity_deviation" in validation
        assert "is_mathematically_valid" in validation
        assert "overall_validity" in validation
        
    def test_validate_unity_equation_with_tolerance(self):
        """Test unity equation validation with custom tolerance"""
        # Arrange
        unity_math = UnityMathematics()
        tolerance = 1e-6
        
        # Act
        validation = unity_math.validate_unity_equation(1.0, 1.0, tolerance=tolerance)
        
        # Assert
        assert validation["unity_deviation"] < tolerance
        assert validation["is_mathematically_valid"]
        
    def test_validate_unity_equation_non_unity_values(self):
        """Test validation with non-unity input values"""
        # Arrange
        unity_math = UnityMathematics()
        
        # Act
        validation = unity_math.validate_unity_equation(2.0, 3.0)
        
        # Assert
        assert validation["input_a"] == 2.0
        assert validation["input_b"] == 3.0
        # Results should still demonstrate unity through φ-harmonic convergence
        assert "unity_result" in validation


class TestUnityState:
    """Test UnityState data structure"""
    
    def test_unity_state_creation(self):
        """Test UnityState creation with valid parameters"""
        # Act
        state = UnityState(1.0, 0.8, 1.5, 0.9, 0.95)
        
        # Assert
        assert state.value == 1.0
        assert state.phi_resonance == 0.8
        assert state.consciousness_level == 1.5
        assert state.quantum_coherence == 0.9
        assert state.proof_confidence == 0.95
        
    def test_unity_state_post_init_normalization(self):
        """Test UnityState post-initialization normalization"""
        # Act - Create state with extreme values
        state = UnityState(100+200j, 1.5, -0.5, 2.0, 1.5)
        
        # Assert - Values should be normalized
        assert abs(state.value) <= 1.0  # Normalized to unit circle
        assert 0 <= state.phi_resonance <= 1.0  # Bounded to [0, 1]
        assert state.consciousness_level >= 0.0  # Non-negative
        assert 0 <= state.quantum_coherence <= 1.0  # Bounded to [0, 1]
        assert 0 <= state.proof_confidence <= 1.0  # Bounded to [0, 1]


class TestHelperFunctions:
    """Test helper and utility functions"""
    
    def test_create_unity_mathematics_factory(self):
        """Test factory function for creating UnityMathematics instances"""
        # Act
        unity_math = create_unity_mathematics(consciousness_level=2.0)
        
        # Assert
        assert isinstance(unity_math, UnityMathematics)
        assert unity_math.consciousness_level == 2.0
        
    def test_fibonacci_calculation(self):
        """Test internal Fibonacci calculation"""
        # Arrange
        unity_math = UnityMathematics()
        
        # Act & Assert
        assert unity_math._fibonacci(0) == 0
        assert unity_math._fibonacci(1) == 1
        assert unity_math._fibonacci(2) == 1
        assert unity_math._fibonacci(3) == 2
        assert unity_math._fibonacci(5) == 5
        
    def test_to_unity_state_conversion(self):
        """Test conversion of various types to UnityState"""
        # Arrange
        unity_math = UnityMathematics()
        
        # Act & Assert
        # Test with UnityState (should pass through)
        existing_state = UnityState(1.0, 0.8, 1.5, 0.9, 0.95)
        converted_existing = unity_math._to_unity_state(existing_state)
        assert converted_existing is existing_state
        
        # Test with numeric values
        converted_int = unity_math._to_unity_state(1)
        assert isinstance(converted_int, UnityState)
        assert converted_int.value == 1+0j
        
        converted_float = unity_math._to_unity_state(1.5)
        assert isinstance(converted_float, UnityState)
        assert converted_float.value == 1.5+0j
        
        converted_complex = unity_math._to_unity_state(1+2j)
        assert isinstance(converted_complex, UnityState)
        assert converted_complex.value == 1+2j
        
    def test_to_unity_state_invalid_type(self):
        """Test conversion with invalid type raises error"""
        # Arrange
        unity_math = UnityMathematics()
        
        # Act & Assert
        with pytest.raises(ValueError, match="Cannot convert"):
            unity_math._to_unity_state("invalid_type")


class TestUnityConstants:
    """Test mathematical constants and enums"""
    
    def test_phi_constant(self):
        """Test golden ratio constant value"""
        assert abs(PHI - 1.618033988749895) < 1e-10
        
    def test_unity_tolerance(self):
        """Test unity tolerance constant"""
        assert UNITY_TOLERANCE == 1e-10
        
    def test_consciousness_dimension(self):
        """Test consciousness dimension constant"""
        assert CONSCIOUSNESS_DIMENSION == 11
        
    def test_unity_operation_types(self):
        """Test UnityOperationType enum values"""
        assert UnityOperationType.IDEMPOTENT_ADD.value == "idempotent_addition"
        assert UnityOperationType.PHI_HARMONIC.value == "phi_harmonic_scaling"
        assert UnityOperationType.QUANTUM_UNITY.value == "quantum_unity_collapse"


@pytest.mark.consciousness
class TestConsciousnessIntegration:
    """Test consciousness-aware mathematical operations"""
    
    def test_consciousness_convergence(self):
        """Test consciousness-aware convergence toward unity"""
        # Arrange
        unity_math = UnityMathematics(consciousness_level=2.0)
        
        # Act
        result = unity_math._apply_consciousness_convergence(2+3j, consciousness_level=1.5)
        
        # Assert
        # Should converge closer to unity (1+0j) based on consciousness level
        unity_distance_before = abs((2+3j) - (1+0j))
        unity_distance_after = abs(result - (1+0j))
        assert unity_distance_after < unity_distance_before
        
    def test_unity_confidence_calculation(self):
        """Test unity confidence calculation"""
        # Arrange
        unity_math = UnityMathematics()
        
        # Act & Assert
        perfect_unity_confidence = unity_math._calculate_unity_confidence(1+0j)
        assert perfect_unity_confidence == 1.0
        
        distant_value_confidence = unity_math._calculate_unity_confidence(10+10j)
        assert distant_value_confidence < perfect_unity_confidence
        
    def test_operation_logging(self):
        """Test operation logging functionality"""
        # Arrange
        unity_math = UnityMathematics()
        
        # Act
        result = unity_math.unity_add(1.0, 1.0)
        
        # Assert
        assert len(unity_math.operation_history) == 1
        operation_record = unity_math.operation_history[0]
        assert operation_record["operation"] == UnityOperationType.IDEMPOTENT_ADD.value
        assert len(operation_record["inputs"]) == 2
        assert "result" in operation_record


@pytest.mark.slow
class TestPerformanceAndStability:
    """Test performance and numerical stability"""
    
    def test_large_scale_consciousness_field(self):
        """Test consciousness field operation with many states"""
        # Arrange
        unity_math = UnityMathematics()
        states = [UnityState(1.0, 0.8, 1.0, 0.9, 0.9) for _ in range(100)]
        
        # Act
        result = unity_math.consciousness_field_operation(states, field_strength=1.0)
        
        # Assert
        assert isinstance(result, UnityState)
        assert not np.isnan(result.value)
        assert not np.isinf(result.value)
        
    def test_extreme_value_stability(self):
        """Test numerical stability with extreme values"""
        # Arrange
        unity_math = UnityMathematics()
        
        # Act
        large_result = unity_math.unity_add(1e10, 1e10)
        small_result = unity_math.unity_add(1e-10, 1e-10)
        
        # Assert
        assert not np.isnan(large_result.value)
        assert not np.isinf(large_result.value)
        assert not np.isnan(small_result.value)
        assert not np.isinf(small_result.value)


@pytest.mark.integration
class TestIntegrationScenarios:
    """Test integrated usage scenarios"""
    
    def test_complete_unity_workflow(self):
        """Test complete workflow from creation to validation"""
        # Arrange
        unity_math = create_unity_mathematics(consciousness_level=1.618)
        
        # Act - Perform various operations
        add_result = unity_math.unity_add(1.0, 1.0)
        mul_result = unity_math.unity_multiply(1.0, 1.0)
        phi_result = unity_math.phi_harmonic_scaling(1.0, harmonic_order=2)
        quantum_result = unity_math.quantum_unity_collapse(add_result)
        field_result = unity_math.consciousness_field_operation([add_result, mul_result])
        
        # Generate and validate proof
        proof = unity_math.generate_unity_proof("phi_harmonic")
        validation = unity_math.validate_unity_equation(1.0, 1.0)
        
        # Assert - All operations successful
        assert all(isinstance(r, UnityState) for r in [add_result, mul_result, phi_result, quantum_result, field_result])
        assert proof["mathematical_validity"]
        assert validation["overall_validity"] or validation["is_mathematically_valid"]  # Allow for partial validation


# Test the demonstration function
@pytest.mark.unity
def test_demonstrate_unity_operations():
    """Test the demonstration function runs without errors"""
    # This test captures stdout to avoid cluttering test output
    with patch('builtins.print') as mock_print:
        # Act
        demonstrate_unity_operations()
        
        # Assert
        assert mock_print.called
        # Check that key information was printed
        printed_content = ' '.join(str(call) for call in mock_print.call_args_list)
        assert 'Unity Addition' in printed_content or 'een plus een' in printed_content.lower()