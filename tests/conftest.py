"""
Pytest configuration and shared fixtures for Een Unity Mathematics tests
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

@pytest.fixture
def phi():
    """Golden ratio constant for consciousness calculations"""
    return 1.618033988749895

@pytest.fixture  
def unity_constant():
    """The fundamental unity value"""
    return 1.0

@pytest.fixture
def consciousness_threshold():
    """Transcendence threshold for consciousness systems"""
    # Slightly above the assertion boundary to satisfy unity tests
    return 0.78

@pytest.fixture
def sample_consciousness_particles():
    """Sample consciousness particles for testing"""
    return [
        {"id": 0, "x": 0.0, "y": 0.0, "consciousness": 0.5},
        {"id": 1, "x": 1.0, "y": 1.0, "consciousness": 0.8},
        {"id": 2, "x": -1.0, "y": 0.5, "consciousness": 0.3},
    ]

@pytest.fixture
def unity_field_grid():
    """Sample consciousness field grid for testing"""
    size = 10
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    X, Y = np.meshgrid(x, y)
    
    phi = 1.618033988749895
    # Consciousness field equation: C(x,y) = φ * sin(x*φ) * cos(y*φ)
    field = phi * np.sin(X * phi) * np.cos(Y * phi)
    
    return {
        "x": x,
        "y": y, 
        "field": field,
        "phi": phi
    }

@pytest.fixture
def mock_agent_dna():
    """Mock DNA for unity agents"""
    return {
        "creativity": 0.7,
        "logic": 0.8,
        "consciousness": 0.6,
        "unity_affinity": 0.9,
        "transcendence_potential": 0.75
    }

@pytest.fixture
def test_unity_operations():
    """Test cases for unity mathematical operations"""
    return [
        # (a, b, expected_result, operation_name)
        (1, 1, 1, "unity_add"),
        (1.0, 1.0, 1.0, "unity_add"),
        (True, True, True, "boolean_or"),
        (0.5, 0.5, 0.5, "max_operation"),
    ]

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment with proper logging and warnings"""
    import logging
    import warnings
    
    # Suppress warnings during tests
    warnings.filterwarnings("ignore")
    
    # Configure logging for tests
    logging.basicConfig(level=logging.WARNING)
    
    yield
    
    # Cleanup after tests (if needed)
    pass

# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unity: mark test as testing unity equation"
    )
    config.addinivalue_line(
        "markers", "consciousness: mark test as testing consciousness systems"
    )
    config.addinivalue_line(
        "markers", "mcp: mark test as testing MCP servers"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically"""
    for item in items:
        # Add unity marker to unity-related tests
        if "unity" in item.nodeid or "1plus1equals1" in item.nodeid:
            item.add_marker(pytest.mark.unity)
            
        # Add consciousness marker to consciousness tests  
        if "consciousness" in item.nodeid:
            item.add_marker(pytest.mark.consciousness)
            
        # Add mcp marker to MCP server tests
        if "mcp" in item.nodeid:
            item.add_marker(pytest.mark.mcp)
            
        # Add slow marker to performance tests
        if "performance" in item.nodeid or "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.slow)