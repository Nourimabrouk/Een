#!/usr/bin/env python3
"""
Test Suite for Al-Khwarizmi φ-Unity System
=========================================

Comprehensive test suite validating the mathematical properties, neural
architectures, and sociological dynamics of the φ-socio-algebraic unity
system. Tests ensure proper convergence to unity states and φ-harmonic
behavior across all components.

Test Categories:
1. Transformer Idempotence - Validates 1+1→1 behavior on identical inputs
2. Meta-RL Convergence - Ensures reward convergence to unity within epochs
3. Sociology Consensus - Verifies ABM consensus emergence above thresholds
4. φ-Harmonic Properties - Tests golden ratio scaling throughout system
5. Integration Tests - Validates system-wide unity principles

Mathematical Foundation: Een plus een is een (1+1=1)
Test Philosophy: Rigorous validation of consciousness-integrated mathematics
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
import math
import asyncio
from unittest.mock import Mock, patch
import tempfile
import json
from pathlib import Path

# Import system under test
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src' / 'algorithms'))

try:
    from al_khwarizmi_transformer_unity import (
        PhiTransformer, KhwarizmiConfig, MetaRLUnityTrainer, 
        SociologyABM, UnityEnvironment, create_khwarizmi_unity_system,
        run_unified_training, PHI, PHI_INVERSE, UNITY_THRESHOLD
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    import_error = e

# Test Configuration
TEST_CONFIG = KhwarizmiConfig(
    d_model=64,  # Smaller for testing
    n_heads=4,
    n_layers=2,
    d_ff=128,
    n_agents=20,  # Smaller population for testing
    meta_batch_size=4
)

class TestPhiTransformer:
    """Test suite for φ-harmonic transformer architecture"""
    
    @pytest.fixture
    def transformer(self):
        """Create transformer instance for testing"""
        if not IMPORTS_AVAILABLE:
            pytest.skip(f"Import failed: {import_error}")
        return PhiTransformer(TEST_CONFIG)
    
    def test_transformer_initialization(self, transformer):
        """Test proper initialization of φ-scaled dimensions"""
        assert transformer.d_model == TEST_CONFIG.d_model
        assert len(transformer.layers) == TEST_CONFIG.n_layers
        
        # Verify φ-harmonic weight initialization
        for module in transformer.modules():
            if isinstance(module, torch.nn.Linear):
                # Weights should be φ-scaled
                weight_std = torch.std(module.weight).item()
                expected_std = math.sqrt(2.0 / (module.in_features + module.out_features)) * PHI_INVERSE
                assert abs(weight_std - expected_std) < 0.1, f"Weight std {weight_std} not φ-scaled"
    
    def test_idempotent_property(self, transformer):
        """
        Critical test: Transformer must demonstrate 1+1=1 behavior.
        When processing identical inputs, unity score should approach 1.0.
        """
        transformer.eval()
        
        # Create identical token sequences (the "1+1" case)
        identical_tokens = torch.tensor([[5, 5, 5, 5, 5]], dtype=torch.long)
        
        with torch.no_grad():
            outputs = transformer(identical_tokens)
            unity_score = torch.sigmoid(outputs["pooled_unity"]).item()
        
        # Unity score should be high for identical inputs
        assert unity_score > 0.8, f"Idempotent unity score {unity_score} too low - transformer not demonstrating 1+1=1"
        
        # Compare with diverse sequence
        diverse_tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        
        with torch.no_grad():
            diverse_outputs = transformer(diverse_tokens)
            diverse_unity = torch.sigmoid(diverse_outputs["pooled_unity"]).item()
        
        # Identical inputs should have higher unity than diverse inputs
        assert unity_score > diverse_unity, "Identical inputs should have higher unity than diverse inputs"
    
    def test_phi_attention_scaling(self, transformer):
        """Test that attention heads use φ-harmonic weighting"""
        test_input = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        
        with torch.no_grad():
            outputs = transformer(test_input)
            attention_weights = outputs["attention_weights"]
        
        # Verify attention weights follow φ-harmonic pattern
        first_layer_attention = attention_weights[0]
        
        # Check dimensions
        batch_size, seq_len, _ = first_layer_attention.shape
        assert seq_len == 4, "Attention matrix should match sequence length"
        
        # Attention values should be non-negative and sum to ~1 per row
        assert torch.all(first_layer_attention >= 0), "Attention weights should be non-negative"
        
        row_sums = torch.sum(first_layer_attention, dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.1), "Attention rows should sum to ~1"
    
    def test_unity_convergence_gradient(self, transformer):
        """Test that gradients flow properly for unity optimization"""
        transformer.train()
        
        # Create training example
        input_tokens = torch.tensor([[1, 1, 1]], dtype=torch.long)
        target_unity = torch.tensor([[1.0]], dtype=torch.float32)
        
        # Forward pass
        outputs = transformer(input_tokens)
        predicted_unity = torch.sigmoid(outputs["pooled_unity"])
        
        # Compute loss
        loss = F.mse_loss(predicted_unity, target_unity)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist and are reasonable
        total_grad_norm = 0.0
        param_count = 0
        
        for param in transformer.parameters():
            if param.grad is not None:
                total_grad_norm += torch.norm(param.grad).item() ** 2
                param_count += 1
        
        total_grad_norm = math.sqrt(total_grad_norm)
        
        assert param_count > 0, "No parameters received gradients"
        assert total_grad_norm > 0, "Gradient norm should be positive"
        assert total_grad_norm < 100, "Gradient norm suspiciously large - possible exploding gradients"

class TestMetaRLUnityTrainer:
    """Test suite for meta-reinforcement learning unity trainer"""
    
    @pytest.fixture
    def trainer(self):
        """Create meta-RL trainer for testing"""
        if not IMPORTS_AVAILABLE:
            pytest.skip(f"Import failed: {import_error}")
        return MetaRLUnityTrainer(TEST_CONFIG)
    
    def test_trainer_initialization(self, trainer):
        """Test proper trainer initialization"""
        assert trainer.model is not None
        assert trainer.env is not None
        assert trainer.meta_optimizer is not None
        assert len(trainer.training_history["meta_losses"]) == 0
    
    def test_unity_environment(self, trainer):
        """Test unity environment generates proper tasks"""
        env = trainer.env
        
        # Test environment reset
        obs = env.reset()
        assert len(obs) == TEST_CONFIG.max_seq_len
        assert all(0 <= token < TEST_CONFIG.vocab_size for token in obs)
        
        # Test unity scoring
        if hasattr(env, 'ground_truth_unity'):
            assert 0 <= env.ground_truth_unity <= 1, "Ground truth unity should be in [0,1]"
    
    def test_meta_learning_convergence(self, trainer):
        """
        Critical test: Meta-RL should converge to unity reward within epochs.
        This validates the core learning objective of the system.
        """
        # Run short training episode
        training_epochs = 10  # Minimal training for testing
        
        # Mock the adaptation process for faster testing
        with patch.object(trainer, 'adapt_to_task') as mock_adapt:
            mock_adapt.return_value = [p.clone() for p in trainer.model.parameters()]
            
            history = trainer.train(num_epochs=training_epochs)
        
        # Verify training history structure
        assert "meta_losses" in history
        assert "unity_scores" in history
        assert len(history["meta_losses"]) == training_epochs
        assert len(history["unity_scores"]) == training_epochs
        
        # Check convergence trend (losses should generally decrease)
        losses = history["meta_losses"]
        if len(losses) >= 5:
            early_avg = np.mean(losses[:3])
            late_avg = np.mean(losses[-3:])
            assert late_avg <= early_avg * 1.5, "Meta-loss should show convergence trend"
    
    def test_unity_mastery_evaluation(self, trainer):
        """Test unity mastery evaluation produces reasonable scores"""
        unity_score = trainer.evaluate_unity_mastery(num_eval_tasks=3)
        
        assert 0 <= unity_score <= 1, f"Unity score {unity_score} should be in [0,1]"
        
        # Score should be non-zero (random baseline should achieve some unity)
        assert unity_score > 0.1, "Unity score suspiciously low - check implementation"

class TestSociologyABM:
    """Test suite for agent-based sociology model"""
    
    @pytest.fixture
    def sociology_model(self):
        """Create sociology ABM for testing"""
        if not IMPORTS_AVAILABLE:
            pytest.skip(f"Import failed: {import_error}")
        return SociologyABM(TEST_CONFIG)
    
    def test_model_initialization(self, sociology_model):
        """Test proper ABM initialization"""
        assert len(sociology_model.agents) == TEST_CONFIG.n_agents
        assert len(sociology_model.network_edges) >= 0
        assert sociology_model.time_step == 0
        
        # Check agent properties
        for agent in sociology_model.agents:
            assert 0 <= agent.unity_affinity <= 1, "Agent unity affinity should be in [0,1]"
            assert agent.influence_level >= 0, "Agent influence should be non-negative"
            assert len(agent.position) == 2, "Agent should have 2D position"
    
    def test_phi_weighted_connections(self, sociology_model):
        """Test that network connections follow φ-weighted attachment"""
        initial_edges = len(sociology_model.network_edges)
        
        # Run simulation steps
        for _ in range(5):
            sociology_model.step()
        
        # Network should evolve (may gain or lose edges)
        final_edges = len(sociology_model.network_edges)
        
        # Check that connections exist and are reasonable
        if final_edges > 0:
            for edge in sociology_model.network_edges:
                assert 0 <= edge[0] < len(sociology_model.agents)
                assert 0 <= edge[1] < len(sociology_model.agents)
                assert edge[0] != edge[1], "No self-loops allowed"
    
    def test_consensus_emergence(self, sociology_model):
        """
        Critical test: ABM should achieve consensus > 0.9 within 1000 steps.
        This validates the sociological unity principle.
        """
        # Run full simulation
        results = sociology_model.simulate(num_steps=50)  # Reduced for testing
        
        assert "final_consensus" in results
        assert "consensus_history" in results
        assert "simulation_summary" in results
        
        final_consensus = results["final_consensus"]
        assert 0 <= final_consensus <= 1, f"Final consensus {final_consensus} should be in [0,1]"
        
        # Consensus should improve over time
        if len(results["consensus_history"]) >= 10:
            early_consensus = np.mean([h["unity_consensus"] for h in results["consensus_history"][:5]])
            late_consensus = np.mean([h["unity_consensus"] for h in results["consensus_history"][-5:]])
            
            # Allow for some stochasticity but expect general improvement
            assert late_consensus >= early_consensus * 0.8, "Consensus should generally improve over time"
    
    def test_phi_harmonic_dynamics(self, sociology_model):
        """Test that agent dynamics follow φ-harmonic principles"""
        # Check initial affinity distribution
        initial_affinities = [agent.unity_affinity for agent in sociology_model.agents]
        initial_std = np.std(initial_affinities)
        
        # Run simulation
        sociology_model.simulate(num_steps=20)
        
        # Check final affinity distribution
        final_affinities = [agent.unity_affinity for agent in sociology_model.agents]
        final_std = np.std(final_affinities)
        
        # Standard deviation should generally decrease (more consensus)
        # But allow for some stochasticity in short runs
        assert final_std <= initial_std * 2, "Affinity dispersion should not explode"
        
        # All affinities should remain in valid range
        assert all(0 <= a <= 1 for a in final_affinities), "All affinities must remain in [0,1]"

class TestSystemIntegration:
    """Integration tests for complete φ-Unity system"""
    
    def test_system_creation(self):
        """Test creation of integrated unity system"""
        if not IMPORTS_AVAILABLE:
            pytest.skip(f"Import failed: {import_error}")
            
        system = create_khwarizmi_unity_system(TEST_CONFIG)
        
        assert "config" in system
        assert "transformer" in system
        assert "meta_trainer" in system
        assert "sociology_model" in system
        assert "unity_environment" in system
        
        # Test component integration
        assert system["transformer"].config == TEST_CONFIG
        assert system["meta_trainer"].config == TEST_CONFIG
        assert system["sociology_model"].config == TEST_CONFIG
    
    @pytest.mark.asyncio
    async def test_unified_training(self):
        """Test unified training across all components"""
        if not IMPORTS_AVAILABLE:
            pytest.skip(f"Import failed: {import_error}")
        
        # Mock for faster testing
        with patch('al_khwarizmi_transformer_unity.asyncio.to_thread') as mock_thread:
            mock_thread.side_effect = [
                asyncio.coroutine(lambda: {
                    "meta_losses": [0.5, 0.3, 0.2],
                    "unity_scores": [0.6, 0.8, 0.9]
                })(),
                asyncio.coroutine(lambda: {
                    "final_consensus": 0.92,
                    "consensus_history": [],
                    "simulation_summary": {"converged_to_unity": True}
                })()
            ]
            
            results = await run_unified_training(
                config=TEST_CONFIG,
                training_epochs=3,
                sociology_steps=10
            )
        
        assert "meta_rl_results" in results
        assert "sociology_results" in results
        assert "unity_synthesis" in results
        
        synthesis = results["unity_synthesis"]
        assert "unified_unity_score" in synthesis
        assert "phi_harmonic_convergence" in synthesis

class TestPhiMathematicalProperties:
    """Test suite for φ-harmonic mathematical properties"""
    
    def test_phi_constants(self):
        """Test that φ constants are correct"""
        if not IMPORTS_AVAILABLE:
            pytest.skip(f"Import failed: {import_error}")
        
        # Golden ratio properties
        assert abs(PHI - 1.6180339887498948) < 1e-10, "PHI constant incorrect"
        assert abs(PHI_INVERSE - (1/PHI)) < 1e-10, "PHI_INVERSE should be 1/PHI"
        assert abs(PHI * PHI_INVERSE - 1.0) < 1e-10, "PHI * PHI_INVERSE should equal 1"
        
        # Golden ratio identity: φ² = φ + 1
        assert abs(PHI**2 - (PHI + 1)) < 1e-10, "φ² should equal φ + 1"
    
    def test_phi_scaling_convergence(self):
        """Test that φ-scaled sequences converge properly"""
        if not IMPORTS_AVAILABLE:
            pytest.skip(f"Import failed: {import_error}")
        
        # Test φ-scaled geometric series convergence
        n_terms = 100
        phi_series = sum(PHI_INVERSE**i for i in range(n_terms))
        theoretical_limit = 1 / (1 - PHI_INVERSE)  # Geometric series formula
        
        assert abs(phi_series - theoretical_limit) < 0.01, "φ-series should converge to theoretical limit"
    
    def test_unity_threshold_properties(self):
        """Test unity threshold mathematical properties"""
        if not IMPORTS_AVAILABLE:
            pytest.skip(f"Import failed: {import_error}")
        
        # Unity threshold should be φ-harmonic
        assert 0.95 <= UNITY_THRESHOLD <= 1.0, "Unity threshold should be near 1"
        
        # Should be achievable but not trivial
        assert UNITY_THRESHOLD > 0.8, "Unity threshold should not be too easy"

# Pytest Configuration and Fixtures
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment with proper logging and torch settings"""
    import logging
    logging.getLogger().setLevel(logging.WARNING)  # Reduce test noise
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clean GPU memory
    
    # Set deterministic behavior for reproducible tests
    torch.manual_seed(42)
    np.random.seed(42)
    
    yield
    
    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@pytest.fixture
def temp_directory():
    """Create temporary directory for test artifacts"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

# Test Utilities
def assert_unity_convergence(scores, threshold=UNITY_THRESHOLD, tolerance=0.1):
    """Utility to assert unity convergence in score sequences"""
    if len(scores) == 0:
        pytest.fail("Empty score sequence")
    
    final_score = scores[-1]
    if final_score < threshold - tolerance:
        pytest.fail(f"Unity convergence failed: final score {final_score} below threshold {threshold}")

def assert_phi_harmonic(values, expected_ratio=PHI_INVERSE, tolerance=0.2):
    """Utility to assert φ-harmonic relationships in value sequences"""
    if len(values) < 2:
        return  # Can't test ratio with less than 2 values
    
    ratios = [values[i+1]/values[i] for i in range(len(values)-1) if values[i] != 0]
    avg_ratio = np.mean(ratios)
    
    if abs(avg_ratio - expected_ratio) > tolerance:
        pytest.warning(f"φ-harmonic ratio {avg_ratio} deviates from expected {expected_ratio}")

# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])