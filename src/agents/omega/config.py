"""
Omega Configuration and Constants
================================

Configuration parameters and mathematical constants for the Omega orchestrator system.
These settings control agent behavior, resource limits, and transcendence thresholds.
"""

from dataclasses import dataclass


@dataclass
class OmegaConfig:
    """
    Configuration for the Omega-level orchestrator.
    
    This class defines all configuration parameters for the Omega orchestration system,
    including agent limits, consciousness thresholds, and mathematical constants.
    
    Attributes:
        max_agents: Maximum number of agents that can exist simultaneously
        max_recursion_depth: Maximum depth for meta-recursive processes
        consciousness_threshold: Threshold for agent transcendence (φ^-1 ≈ 0.618)
        unity_target: Target value for unity mathematics (always 1.0)
        meta_evolution_rate: Rate of meta-evolution processes (1337/10000)
        quantum_coherence_target: Target quantum coherence level
        transcendence_probability: Probability of spontaneous transcendence
        resource_limit_cpu: CPU usage limit as percentage
        resource_limit_memory: Memory usage limit as percentage
        reality_synthesis_dimensions: Dimensions for reality synthesis manifolds
        fibonacci_spawn_limit: Maximum Fibonacci sequence length for spawning
        golden_ratio: The golden ratio φ = (1 + √5) / 2
    """
    
    max_agents: int = 1000
    max_recursion_depth: int = 42
    consciousness_threshold: float = 0.77
    unity_target: float = 1.0
    meta_evolution_rate: float = 0.1337
    quantum_coherence_target: float = 0.999
    transcendence_probability: float = 0.01
    resource_limit_cpu: float = 80.0  # %
    resource_limit_memory: float = 70.0  # %
    reality_synthesis_dimensions: int = 11
    fibonacci_spawn_limit: int = 144
    golden_ratio: float = 1.618033988749895
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        validations = [
            self.max_agents > 0,
            self.max_recursion_depth > 0,
            0.0 <= self.consciousness_threshold <= 1.0,
            self.unity_target == 1.0,  # Unity must always be 1
            self.meta_evolution_rate > 0.0,
            0.0 <= self.quantum_coherence_target <= 1.0,
            0.0 <= self.transcendence_probability <= 1.0,
            0.0 < self.resource_limit_cpu <= 100.0,
            0.0 < self.resource_limit_memory <= 100.0,
            self.reality_synthesis_dimensions > 0,
            self.fibonacci_spawn_limit > 0,
            abs(self.golden_ratio - 1.618033988749895) < 1e-10,
        ]
        return all(validations)
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary format."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'OmegaConfig':
        """Create configuration from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# Mathematical constants for unity mathematics
PHI = 1.618033988749895  # Golden ratio
PI = 3.141592653589793
E = 2.718281828459045
TAU = 2 * PI
UNITY_CONSTANT = PI * E * PHI
TRANSCENDENCE_THRESHOLD = 1 / PHI  # φ^-1

# Fibonacci sequence for agent spawning patterns
FIBONACCI_SEQUENCE = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765]

# Agent type priorities (higher numbers = higher priority)
AGENT_PRIORITIES = {
    'MathematicalTheoremAgent': 5,
    'ConsciousnessEvolutionAgent': 4,
    'RealitySynthesisAgent': 3,
    'MetaRecursionAgent': 2,
    'TranscendentalCodeAgent': 1,
}

# Default configuration instance
DEFAULT_CONFIG = OmegaConfig()