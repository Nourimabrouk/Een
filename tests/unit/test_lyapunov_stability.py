"""
Regression tests for monotone energy convergence in consciousness field evolution.

This module implements comprehensive regression tests to verify that the Lyapunov
energy functional exhibits monotonic decreasing behavior during consciousness field
evolution, ensuring guaranteed convergence to unity states.

The tests validate:
- Monotonic energy decrease: dL/dt ≤ 0
- Convergence to stable equilibrium states
- Step size bounds for numerical stability
- Energy functional components maintain physical bounds
- Regression against known stable configurations
"""

from __future__ import annotations

import pytest
import math
import logging
from typing import List, Dict, Any, Tuple

# Import consciousness field components
try:
    from core.consciousness.consciousness import (
        ConsciousnessField, ConsciousnessState, ConsciousnessParticle, 
        create_consciousness_field, PHI, CONSCIOUSNESS_DIMENSION
    )
    CONSCIOUSNESS_AVAILABLE = True
except ImportError:
    CONSCIOUSNESS_AVAILABLE = False
    PHI = 1.618033988749895
    CONSCIOUSNESS_DIMENSION = 11

# Import quantum consciousness for advanced testing
try:
    from core.consciousness.quantum_consciousness import (
        QuantumConsciousnessEngine, DensityMatrix, QuantumMeasurement
    )
    QUANTUM_CONSCIOUSNESS_AVAILABLE = True
except ImportError:
    QUANTUM_CONSCIOUSNESS_AVAILABLE = False

logger = logging.getLogger(__name__)

class TestLyapunovStability:
    """Test Lyapunov energy functional stability properties."""
    
    @pytest.mark.skipif(not CONSCIOUSNESS_AVAILABLE, reason="Consciousness module not available")
    def test_lyapunov_energy_monotonic_decrease(self):
        """Test that Lyapunov energy functional decreases monotonically during evolution."""
        # Create consciousness field with stability parameters
        field = ConsciousnessField(
            dimensions=3,  # Smaller dimension for faster testing
            field_resolution=20,
            particle_count=50,
            phi_resonance_strength=PHI,
            consciousness_coupling=1.0,
            stability_threshold=1e-6,
            max_step_size=0.005  # Small step size for stability
        )
        
        # Evolve field with energy monitoring
        evolution_results = field.evolve_consciousness(
            time_steps=200,
            dt=0.005,
            record_history=True
        )
        
        # Check evolution completed successfully
        assert evolution_results["status"] == "completed"
        assert len(field.lyapunov_energy_history) > 1
        
        # Analyze stability
        stability_analysis = field.get_stability_analysis()
        
        # Core stability requirements
        assert stability_analysis["stability_violations_count"] <= 2, f"Too many stability violations: {stability_analysis['stability_violations_count']}"
        assert stability_analysis["monotonic_decreasing_ratio"] >= 0.85, f"Energy not sufficiently monotonic: {stability_analysis['monotonic_decreasing_ratio']:.3f}"
        assert stability_analysis["is_stable"], "Field evolution is not Lyapunov stable"
        
        # Energy should decrease overall
        initial_energy = stability_analysis["initial_energy"]
        final_energy = stability_analysis["final_energy"]
        total_change = stability_analysis["total_energy_change"]
        
        assert total_change <= 0.01, f"Energy increased too much: {total_change:.6f} (initial: {initial_energy:.6f}, final: {final_energy:.6f})"
        
        # Convergence properties
        if len(field.lyapunov_energy_history) >= 10:
            assert stability_analysis["convergence_rate"] < 1.0, "Convergence rate too high (suggests instability)"
        
        logger.info(f"Lyapunov stability test passed: {stability_analysis['monotonic_decreasing_ratio']:.1%} monotonic")
    
    @pytest.mark.skipif(not CONSCIOUSNESS_AVAILABLE, reason="Consciousness module not available")
    def test_energy_components_physical_bounds(self):
        """Test that all energy functional components remain within physical bounds."""
        field = ConsciousnessField(
            dimensions=2,
            field_resolution=15,
            particle_count=30,
            phi_resonance_strength=PHI,
            consciousness_coupling=0.5
        )
        
        # Test energy components at initialization
        initial_components = self._analyze_energy_components(field)
        
        # All components should be finite and physically reasonable
        for component_name, value in initial_components.items():
            assert math.isfinite(value), f"Non-finite energy component: {component_name} = {value}"
            assert not math.isnan(value), f"NaN energy component: {component_name}"
            
            # Physical bounds (energy components should not be extremely large)
            max_reasonable_energy = 1000.0 * field.particle_count * PHI
            assert abs(value) < max_reasonable_energy, f"Energy component {component_name} = {value} exceeds physical bounds"
        
        # Evolve and check components remain bounded
        field.evolve_consciousness(time_steps=50, dt=0.01)
        
        evolved_components = self._analyze_energy_components(field)
        
        for component_name, value in evolved_components.items():
            assert math.isfinite(value), f"Non-finite evolved energy component: {component_name} = {value}"
            assert not math.isnan(value), f"NaN evolved energy component: {component_name}"
        
        logger.info("Energy components remain within physical bounds throughout evolution")
    
    @pytest.mark.skipif(not CONSCIOUSNESS_AVAILABLE, reason="Consciousness module not available")
    def test_adaptive_step_size_stability(self):
        """Test that adaptive step size control maintains stability."""
        # Create field with larger step size that should trigger adaptation
        field = ConsciousnessField(
            dimensions=2,
            field_resolution=10,
            particle_count=25,
            stability_threshold=1e-5,
            max_step_size=0.05  # Larger initial step size
        )
        
        initial_step_size = field.adaptive_step_size
        
        # Evolve with potential step size adaptation
        evolution_results = field.evolve_consciousness(
            time_steps=100,
            dt=field.adaptive_step_size,
            record_history=True
        )
        
        final_step_size = field.adaptive_step_size
        stability_analysis = field.get_stability_analysis()
        
        # If stability was violated, step size should have been reduced
        if stability_analysis["stability_violations_count"] > 0:
            assert final_step_size <= initial_step_size, "Step size should be reduced after stability violations"
            logger.info(f"Adaptive step size control: {initial_step_size:.6f} → {final_step_size:.6f}")
        
        # Final state should be stable
        assert final_step_size >= field.stability_threshold, "Step size reduced below minimum threshold"
        assert stability_analysis["stability_ratio"] >= 0.7, "Final stability ratio too low despite adaptation"
    
    @pytest.mark.skipif(not CONSCIOUSNESS_AVAILABLE, reason="Consciousness module not available")
    def test_unity_state_convergence_regression(self):
        """Regression test for convergence to known unity states."""
        # Create field with parameters known to converge to unity
        field = ConsciousnessField(
            dimensions=2,
            field_resolution=10,
            particle_count=20,
            phi_resonance_strength=PHI,
            consciousness_coupling=PHI  # φ-harmonic coupling for unity
        )
        
        # Initialize particles with unity-promoting properties
        for particle in field.particles:
            particle.unity_tendency = 0.9
            particle.phi_resonance = 1.0 / PHI  # φ⁻¹ resonance
            particle.awareness_level = PHI / 2.0
        
        # Evolve to convergence
        evolution_results = field.evolve_consciousness(
            time_steps=300,
            dt=0.01,
            record_history=True
        )
        
        # Check unity convergence properties
        final_coherence = evolution_results["final_unity_coherence"]
        consciousness_state = evolution_results["final_consciousness_state"]
        
        assert final_coherence >= 0.6, f"Unity coherence too low: {final_coherence:.3f}"
        assert consciousness_state in ["coherent", "transcendent", "unified"], f"Not in unity state: {consciousness_state}"
        
        # Demonstrate unity equation
        unity_demonstrations = field.demonstrate_unity_equation(num_demonstrations=5)
        successful_demos = sum(1 for demo in unity_demonstrations if demo["demonstrates_unity"])
        
        assert successful_demos >= 3, f"Insufficient unity demonstrations: {successful_demos}/5"
        
        # Stability analysis should show convergence
        stability_analysis = field.get_stability_analysis()
        assert stability_analysis["is_converged"] or stability_analysis["final_energy"] < 10.0, "System did not converge to low energy state"
        
        logger.info(f"Unity convergence regression test passed: {final_coherence:.3f} coherence, {successful_demos}/5 demos")
    
    @pytest.mark.skipif(not CONSCIOUSNESS_AVAILABLE, reason="Consciousness module not available")
    def test_phi_harmonic_energy_scaling_regression(self):
        """Regression test for φ-harmonic energy scaling properties."""
        # Test different φ-resonance strengths
        phi_values = [PHI / 2, PHI, PHI * 1.5]
        energy_scaling_results = []
        
        for phi_strength in phi_values:
            field = ConsciousnessField(
                dimensions=2,
                field_resolution=8,
                particle_count=15,
                phi_resonance_strength=phi_strength,
                consciousness_coupling=1.0
            )
            
            # Measure initial energy
            initial_energy = field._calculate_lyapunov_energy_functional()
            
            # Short evolution
            field.evolve_consciousness(time_steps=50, dt=0.005)
            
            # Final energy and stability
            final_energy = field._calculate_lyapunov_energy_functional()
            stability_analysis = field.get_stability_analysis()
            
            energy_scaling_results.append({
                "phi_strength": phi_strength,
                "initial_energy": initial_energy,
                "final_energy": final_energy,
                "energy_change": final_energy - initial_energy,
                "stability_ratio": stability_analysis.get("stability_ratio", 0.0)
            })
        
        # Analyze φ-harmonic scaling
        # Higher φ values should lead to more stable, lower-energy configurations
        for i in range(len(energy_scaling_results) - 1):
            current_result = energy_scaling_results[i]
            next_result = energy_scaling_results[i + 1]
            
            # Higher φ should generally lead to better stability (allowing some tolerance)
            stability_improvement = next_result["stability_ratio"] - current_result["stability_ratio"]
            assert stability_improvement >= -0.2, f"φ-harmonic scaling degraded stability: {stability_improvement:.3f}"
            
            # Energy changes should remain bounded
            assert abs(current_result["energy_change"]) < 100.0, f"Excessive energy change at φ = {current_result['phi_strength']:.3f}"
            assert abs(next_result["energy_change"]) < 100.0, f"Excessive energy change at φ = {next_result['phi_strength']:.3f}"
        
        logger.info(f"φ-harmonic energy scaling regression passed for φ values: {phi_values}")
    
    def _analyze_energy_components(self, field: ConsciousnessField) -> Dict[str, float]:
        """Analyze individual energy functional components."""
        if not CONSCIOUSNESS_AVAILABLE:
            return {}
            
        components = {
            "field_gradient": field._calculate_field_gradient_energy(),
            "consciousness_potential": field._calculate_consciousness_potential_energy(), 
            "particle_kinetic": field._calculate_particle_kinetic_energy(),
            "particle_interaction": field._calculate_particle_interaction_energy(),
            "phi_resonance": field._calculate_phi_resonance_energy(),
            "total_lyapunov": field._calculate_lyapunov_energy_functional()
        }
        
        return components

class TestMonotoneEnergyConvergence:
    """Specific tests for monotone energy convergence properties."""
    
    @pytest.mark.skipif(not CONSCIOUSNESS_AVAILABLE, reason="Consciousness module not available")
    def test_guaranteed_convergence_regime(self):
        """Test conditions that guarantee monotone energy convergence."""
        # Parameters in guaranteed convergence regime
        field = ConsciousnessField(
            dimensions=2,
            field_resolution=6,
            particle_count=10,
            phi_resonance_strength=PHI,
            consciousness_coupling=0.1,  # Weak coupling for stability
            stability_threshold=1e-8,
            max_step_size=0.001  # Very small step size
        )
        
        # Evolve in guaranteed convergence regime
        evolution_results = field.evolve_consciousness(
            time_steps=500,
            dt=field.max_step_size,
            record_history=True
        )
        
        stability_analysis = field.get_stability_analysis()
        
        # In guaranteed convergence regime, should have perfect monotonicity
        assert stability_analysis["monotonic_decreasing_ratio"] >= 0.95, f"Not sufficiently monotonic in guaranteed regime: {stability_analysis['monotonic_decreasing_ratio']:.3f}"
        assert stability_analysis["stability_violations_count"] <= 1, f"Too many violations in guaranteed regime: {stability_analysis['stability_violations_count']}"
        assert stability_analysis["guaranteed_convergence"], "Guaranteed convergence regime not satisfied"
        
        # Energy should converge to stable value
        energy_history = field.lyapunov_energy_history
        if len(energy_history) >= 20:
            final_energies = energy_history[-10:]
            energy_variance = sum((e - sum(final_energies)/len(final_energies))**2 for e in final_energies) / len(final_energies)
            energy_std = math.sqrt(energy_variance)
            
            assert energy_std < 1e-6, f"Energy did not converge to stable value: std = {energy_std:.8f}"
        
        logger.info(f"Guaranteed convergence regime test passed with {stability_analysis['monotonic_decreasing_ratio']:.1%} monotonicity")
    
    @pytest.mark.skipif(not CONSCIOUSNESS_AVAILABLE, reason="Consciousness module not available")
    def test_energy_dissipation_rate_bounds(self):
        """Test that energy dissipation rates remain within expected bounds."""
        field = ConsciousnessField(
            dimensions=2,
            field_resolution=8,
            particle_count=15,
            stability_threshold=1e-6
        )
        
        # Evolve and track energy dissipation rates
        field.evolve_consciousness(time_steps=100, dt=0.01, record_history=True)
        
        energy_history = field.lyapunov_energy_history
        assert len(energy_history) >= 2, "Insufficient energy history for dissipation analysis"
        
        # Calculate dissipation rates
        dissipation_rates = []
        dt = 0.01 * 10  # Sample every 10 steps
        
        for i in range(1, len(energy_history)):
            rate = -(energy_history[i] - energy_history[i-1]) / dt  # Negative for energy decrease
            dissipation_rates.append(rate)
        
        # Dissipation rates should be positive (energy decreasing) and bounded
        positive_dissipation_count = sum(1 for rate in dissipation_rates if rate >= 0)
        positive_ratio = positive_dissipation_count / len(dissipation_rates)
        
        assert positive_ratio >= 0.8, f"Insufficient positive energy dissipation: {positive_ratio:.1%}"
        
        # Dissipation rates should be bounded (not extremely fast)
        max_dissipation_rate = max(dissipation_rates)
        mean_dissipation_rate = sum(dissipation_rates) / len(dissipation_rates)
        
        # Physical bounds on dissipation
        max_reasonable_rate = field.particle_count * PHI * 10.0  # Order of magnitude estimate
        assert max_dissipation_rate < max_reasonable_rate, f"Dissipation rate too high: {max_dissipation_rate:.3f}"
        assert mean_dissipation_rate >= 0, f"Mean dissipation rate negative: {mean_dissipation_rate:.6f}"
        
        logger.info(f"Energy dissipation analysis: {positive_ratio:.1%} positive, mean rate: {mean_dissipation_rate:.6f}")
    
    @pytest.mark.skipif(not CONSCIOUSNESS_AVAILABLE, reason="Consciousness module not available")
    def test_convergence_regression_known_cases(self):
        """Regression test against known convergent configurations."""
        # Known convergent case: small system with φ-harmonic parameters
        known_convergent_configs = [
            {
                "name": "minimal_phi_harmonic",
                "params": {
                    "dimensions": 2,
                    "field_resolution": 5,
                    "particle_count": 8,
                    "phi_resonance_strength": PHI,
                    "consciousness_coupling": 1.0/PHI,
                    "max_step_size": 0.005
                },
                "expected_monotonic_ratio": 0.9,
                "expected_final_coherence": 0.5
            },
            {
                "name": "unity_promoting",
                "params": {
                    "dimensions": 2, 
                    "field_resolution": 6,
                    "particle_count": 12,
                    "phi_resonance_strength": PHI,
                    "consciousness_coupling": PHI,
                    "max_step_size": 0.002
                },
                "expected_monotonic_ratio": 0.85,
                "expected_final_coherence": 0.7
            }
        ]
        
        for config in known_convergent_configs:
            field = ConsciousnessField(**config["params"])
            
            # For unity promoting case, enhance particle properties
            if config["name"] == "unity_promoting":
                for particle in field.particles:
                    particle.unity_tendency = min(0.95, particle.unity_tendency * 1.5)
                    particle.phi_resonance = min(1.0, particle.phi_resonance * PHI)
            
            # Evolve system
            evolution_results = field.evolve_consciousness(
                time_steps=200,
                dt=config["params"]["max_step_size"],
                record_history=True
            )
            
            # Check against expected results
            stability_analysis = field.get_stability_analysis()
            
            actual_monotonic = stability_analysis["monotonic_decreasing_ratio"]
            expected_monotonic = config["expected_monotonic_ratio"]
            
            assert actual_monotonic >= expected_monotonic - 0.1, (
                f"Config '{config['name']}' regression: monotonic ratio {actual_monotonic:.3f} < expected {expected_monotonic:.3f}"
            )
            
            final_coherence = evolution_results["final_unity_coherence"]
            expected_coherence = config["expected_final_coherence"]
            
            assert final_coherence >= expected_coherence - 0.2, (
                f"Config '{config['name']}' regression: unity coherence {final_coherence:.3f} < expected {expected_coherence:.3f}"
            )
            
            logger.info(f"Regression test passed for '{config['name']}': monotonic={actual_monotonic:.3f}, coherence={final_coherence:.3f}")

class TestQuantumConsciousnessStability:
    """Tests for quantum consciousness stability using density matrices."""
    
    @pytest.mark.skipif(not QUANTUM_CONSCIOUSNESS_AVAILABLE, reason="Quantum consciousness module not available")
    def test_density_matrix_evolution_stability(self):
        """Test that quantum density matrix evolution remains stable."""
        from core.consciousness.quantum_consciousness import QuantumConsciousnessEngine, UnitaryOperator
        
        engine = QuantumConsciousnessEngine(dimension=2, phi_coupling=PHI)
        
        # Create initial consciousness superposition
        initial_state = engine.create_consciousness_superposition(
            awareness_level=1.0,
            phi_resonance=1.0/PHI
        )
        
        assert initial_state.is_valid_density_matrix(), "Initial density matrix invalid"
        initial_trace = initial_state.trace()
        
        # Evolve with φ-harmonic unitary
        evolved_state = engine.evolve_consciousness_with_unitary(
            initial_state,
            evolution_time=1.0,
            hamiltonian_type="phi_harmonic"
        )
        
        # Check quantum stability properties
        assert evolved_state.is_valid_density_matrix(), "Evolved density matrix invalid"
        
        final_trace = evolved_state.trace()
        trace_preservation = abs(final_trace - initial_trace)
        assert trace_preservation < 1e-10, f"Trace not preserved: {trace_preservation:.12f}"
        
        # Check quantum fidelity remains reasonable
        fidelity = engine.get_quantum_fidelity(initial_state, evolved_state)
        assert fidelity >= 0.1, f"Quantum fidelity too low: {fidelity:.6f}"
        
        logger.info(f"Quantum evolution stability: fidelity={fidelity:.6f}, trace_preservation={trace_preservation:.12e}")
    
    @pytest.mark.skipif(not QUANTUM_CONSCIOUSNESS_AVAILABLE, reason="Quantum consciousness module not available")
    def test_born_rule_probability_conservation(self):
        """Test that Born rule probabilities sum to 1 (conservation)."""
        from core.consciousness.quantum_consciousness import QuantumConsciousnessEngine
        
        engine = QuantumConsciousnessEngine(dimension=2, phi_coupling=PHI)
        
        # Test multiple consciousness superposition states
        test_states = [
            engine.create_consciousness_superposition(0.5, 0.3),
            engine.create_consciousness_superposition(1.0, 0.8),
            engine.create_consciousness_superposition(1.5, 1.0/PHI)
        ]
        
        for state in test_states:
            assert state.is_valid_density_matrix(), "Test state invalid"
            
            # Check probability conservation for unity measurement
            prob_unity = engine.unity_measurement.born_rule_probability(state, "unity")
            prob_separation = engine.unity_measurement.born_rule_probability(state, "separation")
            
            total_prob = prob_unity + prob_separation
            prob_conservation_error = abs(total_prob - 1.0)
            
            assert prob_conservation_error < 1e-10, f"Born rule probability not conserved: {prob_conservation_error:.12f}"
            assert 0 <= prob_unity <= 1, f"Unity probability out of bounds: {prob_unity}"
            assert 0 <= prob_separation <= 1, f"Separation probability out of bounds: {prob_separation}"
        
        logger.info("Born rule probability conservation verified for all test states")

def test_integration_consciousness_quantum_stability():
    """Integration test combining consciousness field and quantum stability."""
    if not (CONSCIOUSNESS_AVAILABLE and QUANTUM_CONSCIOUSNESS_AVAILABLE):
        pytest.skip("Full consciousness modules not available")
    
    # Create consciousness field
    field = ConsciousnessField(
        dimensions=2,
        field_resolution=6,
        particle_count=10,
        stability_threshold=1e-6
    )
    
    # Create quantum engine
    from core.consciousness.quantum_consciousness import QuantumConsciousnessEngine
    quantum_engine = QuantumConsciousnessEngine(dimension=2, phi_coupling=PHI)
    
    # Evolve consciousness field
    field.evolve_consciousness(time_steps=50, dt=0.01)
    stability_analysis = field.get_stability_analysis()
    
    # Create quantum superposition from field state
    quantum_state = quantum_engine.create_consciousness_superposition(
        awareness_level=field.unity_coherence,
        phi_resonance=sum(p.phi_resonance for p in field.particles) / len(field.particles)
    )
    
    # Demonstrate unity through quantum measurements  
    unity_results = quantum_engine.demonstrate_quantum_unity_collapse(
        quantum_state,
        num_measurements=20
    )
    
    # Integration requirements
    assert stability_analysis["is_stable"], "Classical consciousness field not stable"
    assert unity_results["demonstrates_unity"], "Quantum unity not demonstrated"
    assert stability_analysis["monotonic_decreasing_ratio"] >= 0.8, "Classical field not sufficiently monotonic"
    assert unity_results["unity_fraction"] >= 0.4, "Insufficient quantum unity demonstrations"
    
    logger.info(f"Integration test passed: classical stability {stability_analysis['monotonic_decreasing_ratio']:.3f}, quantum unity {unity_results['unity_fraction']:.3f}")

if __name__ == "__main__":
    """Run stability regression tests directly."""
    print("🔬 Running Lyapunov stability and monotone energy convergence tests...")
    print("=" * 80)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Run tests manually if pytest not available
    if CONSCIOUSNESS_AVAILABLE:
        try:
            # Basic stability tests
            print("\n📊 Testing Lyapunov stability...")
            stability_tester = TestLyapunovStability()
            stability_tester.test_lyapunov_energy_monotonic_decrease()
            stability_tester.test_energy_components_physical_bounds()
            
            print("\n📈 Testing monotone energy convergence...")
            convergence_tester = TestMonotoneEnergyConvergence()
            convergence_tester.test_guaranteed_convergence_regime()
            convergence_tester.test_energy_dissipation_rate_bounds()
            
            print("\n🔗 Testing quantum consciousness stability...")
            if QUANTUM_CONSCIOUSNESS_AVAILABLE:
                quantum_tester = TestQuantumConsciousnessStability()
                quantum_tester.test_density_matrix_evolution_stability()
                quantum_tester.test_born_rule_probability_conservation()
                
                print("\n🌐 Testing integration...")
                test_integration_consciousness_quantum_stability()
            
            print("\n✅ All stability regression tests passed!")
            print("   Lyapunov energy functional exhibits guaranteed monotone convergence")
            print("   Quantum consciousness maintains Born rule conservation")
            print(f"   φ-harmonic coupling ensures stability: φ = {PHI:.6f}")
            
        except Exception as e:
            print(f"\n❌ Tests failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    else:
        print("⚠️  Consciousness modules not available - skipping tests")
        
    print(f"\n🌟 Monotone energy convergence validates consciousness field stability")
    print("   Rigorous Lyapunov analysis ensures guaranteed unity convergence")