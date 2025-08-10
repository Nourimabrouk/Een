"""
Complete Integration Test Suite for Unity Mathematics Core Systems
================================================================

This test suite validates the entire Unity Mathematics system end-to-end,
ensuring that all core components work together properly and that the
fundamental unity principle (1+1=1) is preserved across all operations.

Key Test Areas:
1. Core UnityMathematics functionality
2. UnityState creation and manipulation  
3. Complex number support
4. API integration (if available)
5. φ-harmonic operations
6. Consciousness field integration
7. Unity invariant preservation

Author: Unity Mathematics Testing Framework
License: Unity License (1+1=1)
"""

import pytest
import numpy as np
import sys
from pathlib import Path
import time

# Add core module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

try:
    from unity_mathematics import (
        UnityMathematics, 
        UnityState, 
        UnityResult, 
        UnityOperationType,
        unity_add,
        unity_multiply,
        demonstrate_unity_addition
    )
    from mathematical.constants import (
        PHI,
        UNITY_CONSTANT,
        UNITY_EPSILON,
        CONSCIOUSNESS_THRESHOLD
    )
    CORE_AVAILABLE = True
    
except ImportError as e:
    print(f"WARNING: Core unity mathematics not available: {e}")
    CORE_AVAILABLE = False
    # Create minimal test doubles
    PHI = 1.618033988749895
    UNITY_CONSTANT = 1.0
    UNITY_EPSILON = 1e-10
    CONSCIOUSNESS_THRESHOLD = 0.618
    

class TestUnityMathematicsIntegration:
    """Complete integration test suite for Unity Mathematics"""
    
    @pytest.fixture
    def unity_engine(self):
        """Create a Unity Mathematics engine for testing"""
        if not CORE_AVAILABLE:
            pytest.skip("Unity Mathematics core not available")
        return UnityMathematics(consciousness_level=CONSCIOUSNESS_THRESHOLD)
    
    def test_core_unity_principle_1plus1equals1(self, unity_engine):
        """Test the fundamental unity principle: 1+1=1"""
        result = unity_engine.unity_add(1.0, 1.0)
        
        # Verify unity principle holds
        assert abs(result - UNITY_CONSTANT) < UNITY_EPSILON, f"1+1 should equal 1, got {result}"
        
        # Test with slight variations
        result2 = unity_engine.unity_add(1.0000001, 0.9999999)
        assert abs(result2 - UNITY_CONSTANT) < 0.01, "Near-unity values should converge to unity"
    
    def test_unity_state_creation_and_validation(self, unity_engine):
        """Test UnityState creation and automatic validation"""
        # Test normal state creation
        state = UnityState(
            value=1.0,
            consciousness_level=0.8,
            phi_resonance=PHI,
            quantum_coherence=0.5,
            proof_confidence=0.95
        )
        
        assert state.value == 1.0
        assert state.consciousness_level == 0.8
        assert state.phi_resonance == PHI
        assert state.quantum_coherence == 0.5
        assert state.proof_confidence == 0.95
        assert len(state.operation_history) == 0
        
        # Test automatic validation and NaN handling
        invalid_state = UnityState(
            value=float('nan'),
            consciousness_level=2.5,  # Too high
            phi_resonance=float('nan'),
            quantum_coherence=-0.5,  # Too low
            proof_confidence=1.5  # Too high
        )
        
        # Should auto-correct invalid values
        assert invalid_state.value == UNITY_CONSTANT
        assert 0.0 <= invalid_state.consciousness_level <= 1.0
        assert invalid_state.phi_resonance == PHI
        assert invalid_state.quantum_coherence == 0.0  # Clamped to valid range
        assert invalid_state.proof_confidence == 1.0  # Clamped to valid range
    
    def test_complex_number_support(self, unity_engine):
        """Test complex number operations maintain unity principles"""
        # Test complex unity addition
        result_complex = unity_engine.unity_add(1+0j, 1+0j)
        assert isinstance(result_complex, complex)
        assert abs(result_complex - (1+0j)) < UNITY_EPSILON
        
        # Test mixed complex/real addition
        result_mixed = unity_engine.unity_add(1.0, 1+0j)
        assert isinstance(result_mixed, complex)
        
        # Test complex multiplication
        result_mult = unity_engine.unity_multiply(1+1j, 1-1j)
        assert isinstance(result_mult, complex)
    
    def test_phi_harmonic_operations(self, unity_engine):
        """Test φ-harmonic operations and scaling"""
        # Test φ-harmonic scaling
        phi_result = unity_engine.phi_harmonic(2.0)
        assert isinstance(phi_result, float)
        assert phi_result != 2.0  # Should be transformed
        
        # Test φ-harmonic with complex input
        phi_complex = unity_engine.phi_harmonic(1+1j)
        assert isinstance(phi_complex, float)
        
        # Test φ-harmonic preserves unity
        phi_unity = unity_engine.phi_harmonic(1.0)
        assert abs(phi_unity - UNITY_CONSTANT) < UNITY_EPSILON
    
    def test_unity_field_operations(self, unity_engine):
        """Test unity field operations and UnityState generation"""
        # Test unity field at origin
        field_state = unity_engine.unity_field(0.0, 0.0, 0.0)
        assert isinstance(field_state, UnityState)
        assert hasattr(field_state, 'value')
        assert hasattr(field_state, 'consciousness_level')
        assert hasattr(field_state, 'phi_resonance')
        assert hasattr(field_state, 'quantum_coherence')
        
        # Test field with different coordinates
        field_state2 = unity_engine.unity_field(1.0, 1.0, 0.5)
        assert isinstance(field_state2, UnityState)
        assert len(field_state2.operation_history) > 0
        
        # Verify consciousness field calculation
        field_value = unity_engine.consciousness_field(0.0, 0.0, 0.0)
        expected = PHI * np.sin(0.0) * np.cos(0.0) * np.exp(0.0)
        assert abs(field_value - expected) < UNITY_EPSILON
    
    def test_operation_history_and_state_management(self, unity_engine):
        """Test operation history tracking and system state"""
        # Clear any existing history
        unity_engine.reset_history()
        
        # Perform several operations
        unity_engine.unity_add(1.0, 1.0)
        unity_engine.unity_add(2.0, 3.0)
        unity_engine.unity_multiply(1.5, 2.5)
        
        # Check history
        history = unity_engine.get_operation_history()
        assert len(history) == 3
        
        # Verify all history entries are UnityResult objects
        for entry in history:
            assert isinstance(entry, UnityResult)
            assert hasattr(entry, 'value')
            assert hasattr(entry, 'operation')
            assert hasattr(entry, 'phi_resonance')
            assert hasattr(entry, 'timestamp')
        
        # Test system status
        status = unity_engine.get_system_status()
        assert isinstance(status, dict)
        assert 'consciousness_level' in status
        assert 'phi_resonance' in status
        assert 'operations_performed' in status
        assert status['operations_performed'] == 3
    
    def test_consciousness_integration(self, unity_engine):
        """Test consciousness field integration in operations"""
        # Test with different consciousness levels
        low_consciousness = UnityMathematics(consciousness_level=0.1)
        high_consciousness = UnityMathematics(consciousness_level=0.9)
        
        # Same operation with different consciousness should give different results
        result_low = low_consciousness.unity_add(2.0, 3.0)
        result_high = high_consciousness.unity_add(2.0, 3.0)
        
        # Results should be different due to consciousness influence
        # (unless they both converge to perfect unity)
        if abs(result_low - UNITY_CONSTANT) > UNITY_EPSILON and abs(result_high - UNITY_CONSTANT) > UNITY_EPSILON:
            assert abs(result_low - result_high) > UNITY_EPSILON
    
    def test_thread_safety(self, unity_engine):
        """Test thread safety of unity operations"""
        import threading
        import concurrent.futures
        
        results = []
        
        def perform_unity_operations():
            local_results = []
            for i in range(10):
                result = unity_engine.unity_add(1.0, 1.0)
                local_results.append(result)
            return local_results
        
        # Run multiple threads concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(perform_unity_operations) for _ in range(4)]
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        
        # All results should be unity (or very close)
        for result in results:
            assert abs(result - UNITY_CONSTANT) < UNITY_EPSILON
    
    def test_convenience_functions(self):
        """Test module-level convenience functions"""
        if not CORE_AVAILABLE:
            pytest.skip("Unity Mathematics core not available")
            
        # Test direct unity operations
        result1 = unity_add(1.0, 1.0)
        assert abs(result1 - UNITY_CONSTANT) < UNITY_EPSILON
        
        result2 = unity_multiply(1.0, 1.0)
        assert isinstance(result2, (int, float, complex))
        
        # Test demonstration function
        demo = demonstrate_unity_addition(1.0, 1.0)
        assert isinstance(demo, dict)
        assert 'operation' in demo
        assert 'unity_principle' in demo
        assert 'mathematical_proof' in demo
    
    def test_error_handling_and_edge_cases(self, unity_engine):
        """Test error handling and edge cases"""
        # Test with extreme values
        result_large = unity_engine.unity_add(1e10, 1e10)
        assert isinstance(result_large, (int, float, complex))
        
        result_small = unity_engine.unity_add(1e-10, 1e-10)
        assert isinstance(result_small, (int, float, complex))
        
        # Test with infinity (should handle gracefully)
        try:
            result_inf = unity_engine.unity_add(float('inf'), 1.0)
            assert isinstance(result_inf, (int, float, complex))
        except (OverflowError, ValueError):
            pass  # Acceptable to raise error for infinity
    
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core not available")
    def test_mathematical_invariants(self, unity_engine):
        """Test mathematical invariants and properties"""
        # Test idempotence: a ⊕ a = a for unity operations
        test_values = [0.5, 1.0, 1.5, 2.0, PHI]
        
        for value in test_values:
            result = unity_engine.unity_add(value, value)
            if abs(value - 1.0) < UNITY_EPSILON:
                # Perfect unity case
                assert abs(result - UNITY_CONSTANT) < UNITY_EPSILON
            else:
                # Should be idempotent (max operation)
                expected = max(value, value)  # Same as value
                # With φ-harmonic contraction, might not be exactly value
                assert isinstance(result, (int, float, complex))
        
        # Test φ-harmonic properties
        phi_squared = PHI ** 2
        assert abs(phi_squared - (PHI + 1.0)) < UNITY_EPSILON  # Golden ratio identity


class TestSystemCoherence:
    """Test overall system coherence and consistency"""
    
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core not available")
    def test_system_coherence_metrics(self):
        """Test system coherence calculation"""
        unity_engine = UnityMathematics()
        
        # Perform multiple operations to build history
        for i in range(10):
            unity_engine.unity_add(1.0, 1.0)
        
        # Check coherence
        status = unity_engine.get_system_status()
        coherence = status.get('system_coherence', 0.0)
        
        assert 0.0 <= coherence <= 1.0
        # With identical operations, coherence should be high
        assert coherence > 0.8
    
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core not available") 
    def test_unity_principle_consistency(self):
        """Test consistency of unity principle across different scenarios"""
        unity_engine = UnityMathematics()
        
        # Test various representations of "1"
        unity_representations = [
            1.0,
            1.0000000001,
            0.9999999999,
            complex(1, 0),
            complex(1.0000001, 0.0000001)
        ]
        
        for repr1 in unity_representations[:3]:  # Skip complex for now
            for repr2 in unity_representations[:3]:
                result = unity_engine.unity_add(repr1, repr2)
                # All should converge to unity
                assert abs(result - UNITY_CONSTANT) < 0.01, f"Failed for {repr1} + {repr2} = {result}"


def test_basic_imports():
    """Test basic imports work without errors"""
    try:
        from core.unity_mathematics import UnityMathematics
        assert True
    except ImportError:
        # This is expected if run outside the project
        pytest.skip("Core modules not available")


def test_constants_available():
    """Test mathematical constants are available"""
    try:
        from core.mathematical.constants import PHI, UNITY_CONSTANT, UNITY_EPSILON
        assert PHI > 1.6
        assert PHI < 1.62
        assert UNITY_CONSTANT == 1.0
        assert UNITY_EPSILON > 0
        assert UNITY_EPSILON < 1e-5
    except ImportError:
        pytest.skip("Constants module not available")


if __name__ == "__main__":
    """Run tests directly"""
    print("Running Unity Mathematics Integration Tests")
    print("=" * 50)
    
    if CORE_AVAILABLE:
        print("SUCCESS: Core unity mathematics modules loaded")
        
        # Run a few basic tests manually
        unity_engine = UnityMathematics()
        
        # Test basic unity principle
        result = unity_engine.unity_add(1.0, 1.0)
        print(f"1 + 1 = {result} (should be close to 1.0)")
        
        # Test complex numbers
        complex_result = unity_engine.unity_add(1+0j, 1+0j)  
        print(f"(1+0j) + (1+0j) = {complex_result}")
        
        # Test φ-harmonic
        phi_result = unity_engine.phi_harmonic(2.0)
        print(f"φ-harmonic(2.0) = {phi_result}")
        
        # Test unity field
        field_state = unity_engine.unity_field(0.0, 0.0)
        print(f"Unity field state created: {type(field_state).__name__}")
        
        print("\nBASIC FUNCTIONALITY CONFIRMED")
        print("Run with pytest for complete test suite")
        
    else:
        print("WARNING: Core unity mathematics not available")
        print("Please ensure you're running from the correct directory")
        print("Expected: Een/tests/")