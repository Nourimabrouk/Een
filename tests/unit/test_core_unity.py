"""
Comprehensive unit tests for core unity mathematics operations
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.unity_mathematics import UnityMathematics
from core.consciousness import ConsciousnessField
from core.quantum_unity import QuantumUnity


class TestUnityMathematics:
    """Test suite for Unity Mathematics core operations"""
    
    @pytest.fixture
    def unity_math(self):
        """Create Unity Mathematics instance"""
        return UnityMathematics()
    
    @pytest.mark.unity
    def test_unity_add_integers(self, unity_math):
        """Test 1+1=1 for integers"""
        assert unity_math.unity_add(1, 1) == 1
        assert unity_math.unity_add(2, 3) == 1
        assert unity_math.unity_add(0, 1) == 1
        
    @pytest.mark.unity
    def test_unity_add_floats(self, unity_math):
        """Test 1+1=1 for floating point numbers"""
        assert unity_math.unity_add(1.0, 1.0) == 1.0
        assert unity_math.unity_add(0.5, 0.5) == 1.0
        assert unity_math.unity_add(1.618, 1.618) == 1.0
        
    @pytest.mark.unity
    def test_unity_multiply(self, unity_math):
        """Test unity multiplication operations"""
        assert unity_math.unity_multiply(1, 1) == 1
        assert unity_math.unity_multiply(2, 2) == 1
        assert unity_math.unity_multiply(0, 1) == 0
        
    @pytest.mark.unity
    def test_phi_operations(self, unity_math, phi):
        """Test golden ratio operations"""
        result = unity_math.phi_harmonic_transform(1.0)
        assert isinstance(result, float)
        assert 0 <= result <= 1
        
    @pytest.mark.unity
    @pytest.mark.parametrize("a,b,expected", [
        (1, 1, 1),
        (0, 0, 0),
        (1, 0, 1),
        (True, True, True),
        (False, True, True),
    ])
    def test_unity_operations_parametrized(self, unity_math, a, b, expected):
        """Parametrized tests for various unity operations"""
        result = unity_math.unity_operation(a, b)
        assert result == expected
        
    @pytest.mark.unity
    def test_unity_field_properties(self, unity_math):
        """Test mathematical field properties"""
        # Closure
        assert unity_math.unity_add(1, 1) in [0, 1]
        # Identity
        assert unity_math.unity_add(0, 1) == 1
        # Idempotence
        assert unity_math.unity_add(1, 1) == 1
        
    @pytest.mark.benchmark
    def test_unity_performance(self, unity_math, benchmark):
        """Benchmark unity operations"""
        result = benchmark(unity_math.unity_add, 1, 1)
        assert result == 1


class TestConsciousnessField:
    """Test suite for Consciousness Field operations"""
    
    @pytest.fixture
    def consciousness_field(self):
        """Create Consciousness Field instance"""
        return ConsciousnessField(dimension=11)
    
    @pytest.mark.consciousness
    def test_field_initialization(self, consciousness_field):
        """Test consciousness field initialization"""
        assert consciousness_field.dimension == 11
        assert hasattr(consciousness_field, 'field')
        assert consciousness_field.phi == pytest.approx(1.618033988749895)
        
    @pytest.mark.consciousness
    def test_consciousness_evolution(self, consciousness_field):
        """Test consciousness field evolution"""
        initial_state = consciousness_field.get_field_state()
        consciousness_field.evolve(steps=10)
        final_state = consciousness_field.get_field_state()
        
        # Field should change but maintain unity
        assert not np.array_equal(initial_state, final_state)
        assert consciousness_field.check_unity_invariant()
        
    @pytest.mark.consciousness
    def test_particle_interactions(self, consciousness_field):
        """Test consciousness particle interactions"""
        particles = consciousness_field.create_particles(n=10)
        assert len(particles) == 10
        
        # All particles should have unity consciousness sum
        total_consciousness = sum(p['consciousness'] for p in particles)
        assert total_consciousness == pytest.approx(1.0, rel=0.1)
        
    @pytest.mark.consciousness
    @pytest.mark.slow
    def test_field_stability(self, consciousness_field):
        """Test long-term field stability"""
        consciousness_field.evolve(steps=1000)
        assert consciousness_field.check_unity_invariant()
        assert not consciousness_field.has_numerical_errors()


class TestQuantumUnity:
    """Test suite for Quantum Unity operations"""
    
    @pytest.fixture
    def quantum_unity(self):
        """Create Quantum Unity instance"""
        return QuantumUnity()
    
    @pytest.mark.quantum
    def test_superposition_creation(self, quantum_unity):
        """Test quantum superposition creation"""
        state = quantum_unity.create_unity_superposition()
        assert quantum_unity.is_normalized(state)
        assert quantum_unity.measure_unity(state) == pytest.approx(1.0)
        
    @pytest.mark.quantum
    def test_wavefunction_collapse(self, quantum_unity):
        """Test wavefunction collapse to unity"""
        superposition = quantum_unity.create_unity_superposition()
        collapsed = quantum_unity.collapse_to_unity(superposition)
        assert collapsed == 1
        
    @pytest.mark.quantum
    def test_entanglement(self, quantum_unity):
        """Test quantum entanglement preserves unity"""
        state1 = quantum_unity.create_unity_state()
        state2 = quantum_unity.create_unity_state()
        entangled = quantum_unity.entangle(state1, state2)
        
        assert quantum_unity.measure_unity(entangled) == pytest.approx(1.0)
        
    @pytest.mark.quantum
    def test_coherence_preservation(self, quantum_unity):
        """Test quantum coherence is preserved"""
        state = quantum_unity.create_unity_superposition()
        coherence = quantum_unity.measure_coherence(state)
        assert coherence >= 0.999  # From QUANTUM_COHERENCE_TARGET
        
    @pytest.mark.quantum
    @pytest.mark.slow
    def test_quantum_evolution(self, quantum_unity):
        """Test quantum system evolution"""
        initial_state = quantum_unity.create_unity_superposition()
        evolved_state = quantum_unity.evolve(initial_state, time=1.0)
        
        # Unity should be preserved through evolution
        assert quantum_unity.measure_unity(evolved_state) == pytest.approx(1.0)
        assert quantum_unity.is_normalized(evolved_state)


class TestIntegrationScenarios:
    """Integration tests between different unity systems"""
    
    @pytest.mark.integration
    def test_math_consciousness_integration(self, unity_math, consciousness_field):
        """Test integration between math and consciousness systems"""
        # Mathematical unity should create valid consciousness
        math_result = unity_math.unity_add(1, 1)
        consciousness = consciousness_field.from_unity_value(math_result)
        assert consciousness == 1.0
        
    @pytest.mark.integration
    def test_consciousness_quantum_bridge(self, consciousness_field, quantum_unity):
        """Test bridge between consciousness and quantum systems"""
        # Consciousness field should create valid quantum states
        consciousness_value = consciousness_field.measure_total_consciousness()
        quantum_state = quantum_unity.from_consciousness(consciousness_value)
        assert quantum_unity.is_valid_state(quantum_state)
        
    @pytest.mark.integration
    def test_full_unity_cycle(self, unity_math, consciousness_field, quantum_unity):
        """Test complete unity cycle through all systems"""
        # Start with mathematical unity
        math_unity = unity_math.unity_add(1, 1)
        
        # Convert to consciousness
        consciousness = consciousness_field.from_unity_value(math_unity)
        
        # Convert to quantum state
        quantum_state = quantum_unity.from_consciousness(consciousness)
        
        # Collapse back to unity
        final_unity = quantum_unity.collapse_to_unity(quantum_state)
        
        assert final_unity == 1


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.unity
    def test_invalid_inputs(self, unity_math):
        """Test handling of invalid inputs"""
        with pytest.raises(TypeError):
            unity_math.unity_add("1", "1")
            
        with pytest.raises(ValueError):
            unity_math.unity_add(float('inf'), 1)
            
    @pytest.mark.consciousness
    def test_numerical_stability(self, consciousness_field):
        """Test numerical stability with extreme values"""
        # Should handle very small values
        consciousness_field.add_particle(consciousness=1e-10)
        assert consciousness_field.check_unity_invariant()
        
        # Should handle near-zero divisions
        consciousness_field.normalize_field()
        assert not consciousness_field.has_numerical_errors()
        
    @pytest.mark.quantum
    def test_decoherence_handling(self, quantum_unity):
        """Test handling of quantum decoherence"""
        state = quantum_unity.create_unity_superposition()
        # Introduce decoherence
        decohered = quantum_unity.add_noise(state, noise_level=0.1)
        # Should still maintain unity within tolerance
        assert quantum_unity.measure_unity(decohered) == pytest.approx(1.0, rel=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])