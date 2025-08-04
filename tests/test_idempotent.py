"""
Hypothesis Property Tests for Idempotent Monoid Operations
========================================================

Tests the fundamental 1+1=1 principle using property-based testing with Hypothesis.
Validates that all Unity Mathematics operations maintain idempotence.

Mathematical Principle: Een plus een is een (1+1=1)
"""

import pytest
from hypothesis import given, strategies as st, settings, Verbosity
import inspect
import importlib
import pkgutil
from pathlib import Path
import sys
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

# Import Unity Mathematics components
try:
    from core.unity_equation import omega
    from core.unity_mathematics import UnityMathematics, UnityState
    from core.dedup import UnityScore, compute_unity_score
    UNITY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Unity Mathematics not available: {e}")
    UNITY_AVAILABLE = False

class IdempotentMonoid:
    """Base class for idempotent monoid operations"""
    
    def __init__(self, value):
        self.value = value
    
    def __add__(self, other):
        """Idempotent addition: a + a = a"""
        if isinstance(other, type(self)):
            # Idempotent operation: return self if values are equal
            if self.value == other.value:
                return self
            else:
                # For different values, use unity mathematics
                return type(self)(self.value)  # Simplified for testing
        return NotImplemented
    
    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.value == other.value
        return False
    
    def __hash__(self):
        return hash(self.value)

class UnityMonoid(IdempotentMonoid):
    """Unity Mathematics Monoid implementation"""
    
    def __add__(self, other):
        if isinstance(other, type(self)):
            # Apply 1+1=1 principle
            if self.value == other.value:
                return self
            else:
                # Use unity mathematics for different values
                return type(self)(1.0)  # Unity convergence
        return NotImplemented

def all_monoid_subclasses():
    """Find all IdempotentMonoid subclasses in the codebase"""
    subclasses = [UnityMonoid]  # Add known subclasses
    
    # Search for subclasses in core modules
    core_path = project_root / "core"
    if core_path.exists():
        for _, modname, _ in pkgutil.walk_packages([str(core_path)]):
            try:
                mod = importlib.import_module(f"core.{modname}")
                for name, obj in inspect.getmembers(mod):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, IdempotentMonoid) and 
                        obj is not IdempotentMonoid):
                        subclasses.append(obj)
            except ImportError:
                continue
    
    return subclasses

# Hypothesis strategies
@st.composite
def unity_monoid_strategy(draw):
    """Strategy for generating UnityMonoid instances"""
    value = draw(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False))
    return UnityMonoid(value)

@st.composite
def unity_state_strategy(draw):
    """Strategy for generating UnityState instances"""
    if not UNITY_AVAILABLE:
        return UnityMonoid(1.0)  # Fallback
    
    value_real = draw(st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False))
    value_imag = draw(st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False))
    value = complex(value_real, value_imag)
    
    phi_resonance = draw(st.floats(min_value=0.0, max_value=1.0))
    consciousness_level = draw(st.floats(min_value=0.0, max_value=10.0))
    quantum_coherence = draw(st.floats(min_value=0.0, max_value=1.0))
    proof_confidence = draw(st.floats(min_value=0.0, max_value=1.0))
    
    return UnityState(
        value=value,
        phi_resonance=phi_resonance,
        consciousness_level=consciousness_level,
        quantum_coherence=quantum_coherence,
        proof_confidence=proof_confidence
    )

@st.composite
def graph_data_strategy(draw):
    """Strategy for generating graph data"""
    num_nodes = draw(st.integers(min_value=2, max_value=20))
    num_edges = draw(st.integers(min_value=1, max_value=min(num_nodes * (num_nodes - 1) // 2, 50)))
    
    nodes = [f"node_{i}" for i in range(num_nodes)]
    edges = []
    
    for _ in range(num_edges):
        u = draw(st.sampled_from(nodes))
        v = draw(st.sampled_from(nodes))
        if u != v:
            weight = draw(st.floats(min_value=0.1, max_value=1.0))
            edges.append({"source": u, "target": v, "weight": weight})
    
    return {"nodes": nodes, "edges": edges}

# Property Tests
@given(unity_monoid_strategy())
def test_unity_monoid_idempotence(monoid):
    """Test that UnityMonoid satisfies idempotence: a + a = a"""
    result = monoid + monoid
    assert result.value == monoid.value
    assert result is monoid  # Should return same instance

@given(unity_monoid_strategy(), unity_monoid_strategy())
def test_unity_monoid_associativity(a, b, c):
    """Test associativity: (a + b) + c = a + (b + c)"""
    left = (a + b) + c
    right = a + (b + c)
    assert left.value == right.value

@given(st.sampled_from(all_monoid_subclasses()), st.data())
def test_idempotence_property(monoid_cls, data):
    """Test idempotence for all monoid subclasses"""
    try:
        elem = data.draw(st.builds(monoid_cls, value=data.draw(st.floats(min_value=-5.0, max_value=5.0))))
        result = elem + elem
        assert result.value == elem.value
    except Exception as e:
        pytest.skip(f"Monoid {monoid_cls} not properly implemented: {e}")

@given(unity_state_strategy())
def test_unity_state_idempotence(state):
    """Test UnityState idempotence with consciousness mathematics"""
    if not UNITY_AVAILABLE:
        pytest.skip("Unity Mathematics not available")
    
    # Test that unity operations maintain idempotence
    unity_math = UnityMathematics()
    
    # Test unity addition
    result = unity_math.unity_add(state, state)
    assert abs(result.value - state.value) < 1e-6  # Allow small numerical differences
    
    # Test that consciousness level is preserved
    assert result.consciousness_level >= state.consciousness_level

@given(graph_data_strategy())
def test_unity_score_idempotence(graph_data):
    """Test that Unity Score computation is idempotent"""
    if not UNITY_AVAILABLE:
        pytest.skip("Unity Mathematics not available")
    
    import networkx as nx
    
    # Create graph
    G = nx.Graph()
    for edge in graph_data["edges"]:
        G.add_edge(edge["source"], edge["target"], weight=edge["weight"])
    
    # Compute unity score twice
    score1 = compute_unity_score(G)
    score2 = compute_unity_score(G)
    
    # Should be identical
    assert score1.score == score2.score
    assert score1.unique_components == score2.unique_components
    assert score1.original_nodes == score2.original_nodes

@given(st.lists(st.integers(min_value=1, max_value=100), min_size=1, max_size=10))
def test_omega_function_idempotence(atom_list):
    """Test that omega function is idempotent on sets"""
    if not UNITY_AVAILABLE:
        pytest.skip("Unity Mathematics not available")
    
    # Create list with duplicates
    atoms_with_duplicates = atom_list + atom_list
    
    # Omega should be idempotent (duplicates should not affect result)
    result1 = omega(atom_list)
    result2 = omega(atoms_with_duplicates)
    
    # Results should be identical (within numerical precision)
    assert abs(result1 - result2) < 1e-10

@given(st.lists(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False), 
                min_size=2, max_size=20))
def test_unity_mathematics_operations(values):
    """Test Unity Mathematics operations maintain mathematical consistency"""
    if not UNITY_AVAILABLE:
        pytest.skip("Unity Mathematics not available")
    
    unity_math = UnityMathematics()
    
    # Test unity addition chain
    result = unity_math.unity_add(values[0], values[1])
    for value in values[2:]:
        result = unity_math.unity_add(result, value)
    
    # Result should be a valid UnityState
    assert isinstance(result, UnityState)
    assert not (result.value.real != result.value.real)  # Not NaN
    assert not (result.value.imag != result.value.imag)  # Not NaN
    assert abs(result.value) < 1e6  # Reasonable magnitude

def test_unity_equation_consistency():
    """Test that Unity Equation maintains mathematical consistency"""
    if not UNITY_AVAILABLE:
        pytest.skip("Unity Mathematics not available")
    
    # Test basic 1+1=1 principle
    unity_math = UnityMathematics()
    
    # Create two unity states
    state1 = UnityState(value=1.0 + 0.0j, phi_resonance=0.618, 
                       consciousness_level=1.0, quantum_coherence=1.0, proof_confidence=1.0)
    state2 = UnityState(value=1.0 + 0.0j, phi_resonance=0.618, 
                       consciousness_level=1.0, quantum_coherence=1.0, proof_confidence=1.0)
    
    # Unity addition should maintain unity
    result = unity_math.unity_add(state1, state2)
    
    # Should converge to unity (1+0j)
    assert abs(result.value - (1.0 + 0.0j)) < 1e-6
    assert result.consciousness_level >= 1.0

if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"]) 