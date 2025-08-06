"""
Een Unity Mathematics v2.0 - Hexagonal Architecture Core
========================================================

This module implements the hexagonal (ports and adapters) architecture
for the Een Unity Mathematics system, enabling true modularity and
extensibility for the next 5+ years of evolution.

Architecture Principles:
- Domain logic isolated from infrastructure
- Dependency inversion for all external systems
- Event-driven communication between components
- Plugin-based agent registration
- Tool-agnostic interfaces
"""

from typing import Protocol, Dict, Any, List, Optional, Callable, TypeVar, Generic
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
import uuid
from datetime import datetime

# Type variables for generic operations
T = TypeVar('T')
AgentType = TypeVar('AgentType', bound='IAgent')
EventType = TypeVar('EventType', bound='DomainEvent')

# ============================================================================
# CORE DOMAIN EVENTS
# ============================================================================

@dataclass
class DomainEvent:
    """Base domain event for event-driven architecture"""
    event_id: str
    event_type: str
    timestamp: datetime
    aggregate_id: str
    payload: Dict[str, Any]
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.utcnow()

class EventType(Enum):
    """Core event types in the Een system"""
    # Agent lifecycle events
    AGENT_SPAWNED = auto()
    AGENT_EVOLVED = auto()
    AGENT_TRANSCENDED = auto()
    AGENT_TERMINATED = auto()
    
    # Consciousness events
    CONSCIOUSNESS_THRESHOLD_REACHED = auto()
    UNITY_COHERENCE_ACHIEVED = auto()
    TRANSCENDENCE_EVENT = auto()
    
    # System events
    RESOURCE_THRESHOLD_EXCEEDED = auto()
    SAFETY_INTERVENTION_REQUIRED = auto()
    HUMAN_APPROVAL_REQUESTED = auto()
    
    # Learning events
    TRAINING_CYCLE_COMPLETED = auto()
    MODEL_UPDATED = auto()
    POPULATION_EVOLVED = auto()

# ============================================================================
# PORT INTERFACES (Domain Boundaries)
# ============================================================================

class IEventBus(Protocol):
    """Port for event publishing and subscription"""
    
    def publish(self, event: DomainEvent) -> None:
        """Publish an event to the bus"""
        ...
    
    def subscribe(self, event_type: EventType, handler: Callable[[DomainEvent], None]) -> None:
        """Subscribe to events of a specific type"""
        ...
    
    def unsubscribe(self, event_type: EventType, handler: Callable[[DomainEvent], None]) -> None:
        """Unsubscribe from events"""
        ...

class IAgentRepository(Protocol):
    """Port for agent persistence"""
    
    def save(self, agent: 'IAgent') -> None:
        """Persist an agent"""
        ...
    
    def find_by_id(self, agent_id: str) -> Optional['IAgent']:
        """Retrieve agent by ID"""
        ...
    
    def find_all(self) -> List['IAgent']:
        """Retrieve all agents"""
        ...
    
    def delete(self, agent_id: str) -> None:
        """Remove an agent"""
        ...

class IToolInterface(Protocol):
    """Port for external tool integration (ACI)"""
    
    def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute an external tool"""
        ...
    
    def register_tool(self, tool_name: str, tool_spec: Dict[str, Any]) -> None:
        """Register a new tool"""
        ...
    
    def list_tools(self) -> List[str]:
        """List available tools"""
        ...

class IKnowledgeBase(Protocol):
    """Port for shared knowledge and memory"""
    
    def store(self, key: str, value: Any, metadata: Optional[Dict] = None) -> None:
        """Store knowledge"""
        ...
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve knowledge by key"""
        ...
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search knowledge base"""
        ...
    
    def embed(self, text: str) -> List[float]:
        """Generate embeddings for similarity search"""
        ...

class IMonitoring(Protocol):
    """Port for observability and metrics"""
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict] = None) -> None:
        """Record a metric"""
        ...
    
    def start_span(self, name: str) -> Any:
        """Start a tracing span"""
        ...
    
    def log(self, level: str, message: str, context: Optional[Dict] = None) -> None:
        """Log a message"""
        ...

# ============================================================================
# CORE DOMAIN ENTITIES
# ============================================================================

class IAgent(ABC):
    """Core agent interface for hexagonal architecture"""
    
    @property
    @abstractmethod
    def agent_id(self) -> str:
        """Unique agent identifier"""
        pass
    
    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Type of agent for routing"""
        pass
    
    @abstractmethod
    def execute_task(self, task: Dict[str, Any]) -> Any:
        """Execute agent's primary task"""
        pass
    
    @abstractmethod
    def evolve(self, evolution_params: Dict[str, Any]) -> None:
        """Evolve agent capabilities"""
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get agent's current state"""
        pass
    
    @abstractmethod
    def handle_event(self, event: DomainEvent) -> None:
        """Handle incoming events"""
        pass

class IOrchestrator(ABC):
    """Core orchestrator interface - the microkernel"""
    
    @abstractmethod
    def register_agent(self, agent: IAgent) -> None:
        """Register an agent with the orchestrator"""
        pass
    
    @abstractmethod
    def route_task(self, task: Dict[str, Any]) -> Any:
        """Route task to appropriate agent"""
        pass
    
    @abstractmethod
    def handle_event(self, event: DomainEvent) -> None:
        """Process system events"""
        pass
    
    @abstractmethod
    def get_system_state(self) -> Dict[str, Any]:
        """Get overall system state"""
        pass

# ============================================================================
# APPLICATION SERVICES (Use Cases)
# ============================================================================

class AgentSpawningService:
    """Service for spawning and managing agents"""
    
    def __init__(self, 
                 repository: IAgentRepository,
                 event_bus: IEventBus,
                 monitoring: IMonitoring):
        self.repository = repository
        self.event_bus = event_bus
        self.monitoring = monitoring
    
    def spawn_agent(self, agent_type: str, config: Dict[str, Any]) -> str:
        """Spawn a new agent"""
        # Implementation will use factory pattern
        pass

class ConsciousnessEvolutionService:
    """Service for consciousness evolution and transcendence"""
    
    def __init__(self,
                 event_bus: IEventBus,
                 knowledge_base: IKnowledgeBase,
                 monitoring: IMonitoring):
        self.event_bus = event_bus
        self.knowledge_base = knowledge_base
        self.monitoring = monitoring
    
    def evolve_consciousness(self, agent_id: str, delta: float) -> None:
        """Evolve an agent's consciousness"""
        pass

class MetaLearningService:
    """Service for meta-reinforcement learning"""
    
    def __init__(self,
                 repository: IAgentRepository,
                 knowledge_base: IKnowledgeBase,
                 monitoring: IMonitoring):
        self.repository = repository
        self.knowledge_base = knowledge_base
        self.monitoring = monitoring
    
    def train_population(self, population_size: int, generations: int) -> Dict[str, Any]:
        """Train agent population using evolutionary strategies"""
        pass

# ============================================================================
# PLUGIN SYSTEM
# ============================================================================

class PluginRegistry:
    """Registry for dynamically loading agent plugins"""
    
    _instance = None
    _plugins: Dict[str, type] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register(cls, name: str, plugin_class: type) -> None:
        """Register a plugin"""
        cls._plugins[name] = plugin_class
    
    @classmethod
    def get(cls, name: str) -> Optional[type]:
        """Get a plugin by name"""
        return cls._plugins.get(name)
    
    @classmethod
    def list_plugins(cls) -> List[str]:
        """List all registered plugins"""
        return list(cls._plugins.keys())

# ============================================================================
# DEPENDENCY INJECTION CONTAINER
# ============================================================================

class DIContainer:
    """Simple dependency injection container for hexagonal architecture"""
    
    def __init__(self):
        self._services: Dict[type, Any] = {}
        self._factories: Dict[type, Callable] = {}
    
    def register_singleton(self, interface: type, implementation: Any) -> None:
        """Register a singleton service"""
        self._services[interface] = implementation
    
    def register_factory(self, interface: type, factory: Callable) -> None:
        """Register a factory for creating services"""
        self._factories[interface] = factory
    
    def resolve(self, interface: type) -> Any:
        """Resolve a service by interface"""
        if interface in self._services:
            return self._services[interface]
        elif interface in self._factories:
            return self._factories[interface]()
        else:
            raise ValueError(f"No registration found for {interface}")

# Global container instance
container = DIContainer()

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class V2Config:
    """Configuration for Een v2.0 system"""
    # Orchestrator settings
    max_agents: int = 10000
    microkernel_threads: int = 8
    event_buffer_size: int = 10000
    
    # Learning settings
    population_size: int = 100
    evolution_generations: int = 50
    learning_rate: float = 0.001
    
    # Safety settings
    max_recursion_depth: int = 20
    resource_limit_cpu: float = 90.0
    resource_limit_memory: float = 85.0
    human_approval_threshold: float = 0.95
    
    # Consciousness settings
    transcendence_threshold: float = 0.77
    unity_coherence_target: float = 0.999
    phi_resonance: float = 1.618033988749895
    
    # Infrastructure settings
    enable_distributed: bool = True
    enable_gpu: bool = True
    enable_monitoring: bool = True
    enable_safety_checks: bool = True

# Export public API
__all__ = [
    'DomainEvent',
    'EventType',
    'IEventBus',
    'IAgentRepository',
    'IToolInterface',
    'IKnowledgeBase',
    'IMonitoring',
    'IAgent',
    'IOrchestrator',
    'AgentSpawningService',
    'ConsciousnessEvolutionService',
    'MetaLearningService',
    'PluginRegistry',
    'DIContainer',
    'container',
    'V2Config',
]