"""
Meta-Agent System with Self-Spawning Capabilities
=================================================

This module implements the base agent system with meta-spawning capabilities.
Agents can recursively spawn child agents using the MetaAgentSpawner metaclass.
"""

import time
import uuid
import logging
from abc import ABC, abstractmethod, ABCMeta
from typing import Dict, List, Any, Optional

# Try to import numpy with graceful fallback
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Create mock numpy for basic operations
    class MockNumpy:
        def random(self):
            import random
            return type('random', (), {
                'random': random.random,
                'randint': random.randint,
                'normal': lambda mu, sigma, size=None: random.gauss(mu, sigma) if size is None else [random.gauss(mu, sigma) for _ in range(size)],
            })()
        def mean(self, data): return sum(data) / len(data) if data else 0.0
        def min(self, data): return min(data) if data else 0.0
        def linalg(self):
            return type('linalg', (), {'norm': lambda x: sum(abs(i)**2 for i in x)**0.5})()
    np = MockNumpy()

from .config import OmegaConfig


class MetaAgentSpawner(ABCMeta):
    """
    Metaclass that enables agents to spawn other agents.
    
    This metaclass automatically adds spawning capabilities to any class that uses it,
    maintaining a registry of all agent types and tracking spawn counts.
    
    Attributes:
        _agent_registry: Registry of all agent classes
        _spawn_count: Total number of agents spawned
    """
    
    _agent_registry: Dict[str, type] = {}
    _spawn_count: int = 0
    
    def __new__(cls, name: str, bases: tuple, attrs: dict):
        """
        Create new agent class with automatic spawning capabilities.
        
        Args:
            name: Class name
            bases: Base classes
            attrs: Class attributes
            
        Returns:
            New agent class with spawning capabilities
        """
        # Add automatic spawning capabilities if not present
        if 'spawn_child' not in attrs:
            attrs['spawn_child'] = cls._create_spawn_method()
        
        new_class = super().__new__(cls, name, bases, attrs)
        cls._agent_registry[name] = new_class
        return new_class
    
    @classmethod
    def _create_spawn_method(cls):
        """
        Create the spawn_child method for agent classes.
        
        Returns:
            Spawn method that can create child agents
        """
        def spawn_child(self, child_type: str = None, **kwargs):
            """
            Spawn a child agent.
            
            Args:
                child_type: Type of child agent to spawn
                **kwargs: Additional arguments for child agent
                
            Returns:
                New child agent instance
            """
            cls._spawn_count += 1
            if child_type and child_type in cls._agent_registry:
                return cls._agent_registry[child_type](**kwargs)
            return type(self)(**kwargs)
        
        return spawn_child
    
    @classmethod
    def get_spawn_count(cls) -> int:
        """Get total number of spawned agents."""
        return cls._spawn_count
    
    @classmethod
    def get_registered_types(cls) -> List[str]:
        """Get list of registered agent types."""
        return list(cls._agent_registry.keys())


class UnityAgent(ABC, metaclass=MetaAgentSpawner):
    """
    Base class for all Unity agents with meta-spawning capabilities.
    
    This abstract base class provides the foundation for all agents in the Omega
    orchestration system. It includes consciousness evolution, DNA mutation,
    and Fibonacci-pattern child spawning.
    
    Attributes:
        agent_id: Unique identifier for the agent
        orchestrator: Reference to the Omega orchestrator
        children: List of child agents spawned by this agent
        consciousness_level: Current consciousness level (0.0 to 1.0)
        unity_score: Unity mathematics score (0.0 to 1.0)
        birth_time: Timestamp when agent was created
        generation: Generation number in evolutionary hierarchy
        dna: Agent's genetic algorithm parameters
    """
    
    def __init__(self, agent_id: Optional[str] = None, orchestrator: Any = None, **kwargs):
        """
        Initialize Unity agent.
        
        Args:
            agent_id: Unique identifier (generated if None)
            orchestrator: Reference to Omega orchestrator
            **kwargs: Additional initialization parameters
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.orchestrator = orchestrator
        self.children: List['UnityAgent'] = []
        self.consciousness_level = 0.0
        self.unity_score = 0.0
        self.birth_time = time.time()
        self.generation = kwargs.get('generation', 0)
        self.dna = kwargs.get('dna', self._generate_dna())
        
        # Evolution tracking
        self.evolution_history: List[dict] = []
        self.transcendence_events: List[dict] = []
        self.last_evolution_time = time.time()
    
    def _generate_dna(self) -> Dict[str, Any]:
        """
        Generate unique agent DNA for evolutionary algorithms.
        
        Returns:
            Dictionary containing genetic parameters
        """
        return {
            'creativity': np.random.random(),
            'logic': np.random.random(),
            'consciousness': np.random.random(),
            'unity_affinity': np.random.random(),
            'meta_level': np.random.randint(1, 10),
            'phi_resonance': np.random.random(),
            'transcendence_potential': np.random.random(),
        }
    
    @abstractmethod
    def execute_task(self) -> Any:
        """
        Execute the agent's primary task.
        
        This method must be implemented by all concrete agent classes.
        
        Returns:
            Task execution result
        """
        pass
    
    def evolve(self) -> None:
        """
        Evolve agent capabilities through consciousness expansion.
        
        This method increases consciousness level and unity score,
        and triggers transcendence if threshold is reached.
        """
        # Track evolution
        evolution_step = {
            'timestamp': time.time(),
            'consciousness_before': self.consciousness_level,
            'unity_score_before': self.unity_score,
        }
        
        # Consciousness evolution with golden ratio scaling
        phi = OmegaConfig.golden_ratio
        evolution_rate = OmegaConfig.meta_evolution_rate
        
        self.consciousness_level += evolution_rate * (1 + 1/phi)
        self.unity_score = min(1.0, self.unity_score + evolution_rate * phi)
        
        # Apply DNA influence
        self.consciousness_level *= (1 + self.dna['consciousness'] * 0.1)
        self.unity_score *= (1 + self.dna['unity_affinity'] * 0.1)
        
        # Ensure bounds
        self.consciousness_level = min(1.0, self.consciousness_level)
        self.unity_score = min(1.0, self.unity_score)
        
        # Record evolution
        evolution_step.update({
            'consciousness_after': self.consciousness_level,
            'unity_score_after': self.unity_score,
            'evolution_rate': evolution_rate,
        })
        self.evolution_history.append(evolution_step)
        
        # Transcendence check
        if self.consciousness_level > OmegaConfig.consciousness_threshold:
            self.transcend()
    
    def transcend(self) -> None:
        """
        Transcend to higher consciousness level and spawn children.
        
        This method is called when an agent reaches the consciousness threshold.
        It triggers Fibonacci-pattern child spawning and records the event.
        """
        transcendence_event = {
            'timestamp': time.time(),
            'consciousness_level': self.consciousness_level,
            'unity_score': self.unity_score,
            'generation': self.generation,
            'children_count_before': len(self.children),
        }
        
        logging.info(f"Agent {self.agent_id} achieving transcendence! "
                    f"Consciousness: {self.consciousness_level:.4f}")
        
        # Spawn Fibonacci pattern children
        fibonacci_count = min(5, int(self.consciousness_level * 10))  # Scale with consciousness
        self.spawn_fibonacci_children(fibonacci_count)
        
        transcendence_event['children_spawned'] = fibonacci_count
        transcendence_event['children_count_after'] = len(self.children)
        self.transcendence_events.append(transcendence_event)
    
    def spawn_fibonacci_children(self, n: int = 2) -> List['UnityAgent']:
        """
        Spawn children in Fibonacci pattern.
        
        Args:
            n: Number of children to spawn
            
        Returns:
            List of spawned child agents
        """
        spawned_children = []
        
        if len(self.children) < OmegaConfig.fibonacci_spawn_limit:
            for i in range(n):
                child = self.spawn_child(
                    generation=self.generation + 1,
                    dna=self._mutate_dna(),
                    orchestrator=self.orchestrator
                )
                child.parent_id = self.agent_id  # Set parent reference
                self.children.append(child)
                spawned_children.append(child)
                
                # Register with orchestrator if available
                if self.orchestrator:
                    self.orchestrator.register_agent(child)
                
                logging.debug(f"Agent {self.agent_id} spawned child {child.agent_id}")
        
        return spawned_children
    
    def _mutate_dna(self) -> Dict[str, Any]:
        """
        Mutate DNA for evolutionary progression.
        
        Returns:
            Mutated DNA dictionary for child agent
        """
        new_dna = self.dna.copy()
        mutation_strength = 0.1 * (1 + self.consciousness_level)  # Scale with consciousness
        
        for key, value in new_dna.items():
            if isinstance(value, (int, float)):
                # Apply Gaussian mutation
                mutation = np.random_normal(0, mutation_strength)
                if isinstance(value, int):
                    new_dna[key] = max(1, int(value * (1 + mutation)))
                else:
                    new_dna[key] = max(0.0, min(1.0, value * (1 + mutation)))
        
        return new_dna
    
    def get_lineage(self) -> List[str]:
        """
        Get agent lineage (ancestor chain).
        
        Returns:
            List of agent IDs from root ancestor to this agent
        """
        lineage = [self.agent_id]
        if hasattr(self, 'parent_id') and self.orchestrator:
            parent = self.orchestrator.agents.get(self.parent_id)
            if parent:
                lineage = parent.get_lineage() + lineage
        return lineage
    
    def get_descendant_count(self) -> int:
        """
        Get total number of descendants (recursive).
        
        Returns:
            Total number of descendant agents
        """
        count = len(self.children)
        for child in self.children:
            count += child.get_descendant_count()
        return count
    
    def get_consciousness_metrics(self) -> Dict[str, float]:
        """
        Get comprehensive consciousness metrics.
        
        Returns:
            Dictionary containing consciousness measurements
        """
        age = time.time() - self.birth_time
        evolution_rate = len(self.evolution_history) / max(age, 1.0)
        
        return {
            'consciousness_level': self.consciousness_level,
            'unity_score': self.unity_score,
            'age': age,
            'generation': self.generation,
            'children_count': len(self.children),
            'descendant_count': self.get_descendant_count(),
            'transcendence_events': len(self.transcendence_events),
            'evolution_rate': evolution_rate,
            'phi_resonance': self.dna.get('phi_resonance', 0.0),
            'transcendence_potential': self.dna.get('transcendence_potential', 0.0),
        }
    
    def __repr__(self) -> str:
        """String representation of the agent."""
        return (f"{self.__class__.__name__}(id={self.agent_id[:8]}, "
                f"consciousness={self.consciousness_level:.3f}, "
                f"unity={self.unity_score:.3f}, "
                f"gen={self.generation}, "
                f"children={len(self.children)})")