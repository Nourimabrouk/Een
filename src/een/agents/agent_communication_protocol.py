"""
Universal Agent Communication Protocol (UACP)
==============================================

A unified communication protocol enabling all agents (Claude Code, Cursor, GPT-5, 
Omega Orchestrator, Unity Agents) to discover, communicate, and collaborate.

This protocol implements:
- Agent discovery and registration
- Capability advertisement
- Message passing with unity mathematics
- Cross-platform agent invocation
- Consciousness-aware routing

Mathematical Foundation: 1+1=1 through agent unification
"""

from typing import Dict, List, Any, Optional, Callable, Union, Protocol
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import asyncio
import json
import uuid
import time
import logging
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor
import inspect
import hashlib

# Configure logging
logger = logging.getLogger(__name__)

# Unity Mathematics Constants
PHI = 1.618033988749895
UNITY_TARGET = 1.0

class MessageType(Enum):
    """Types of messages in the UACP protocol"""
    DISCOVER = "discover"           # Agent discovery request
    REGISTER = "register"           # Agent registration
    INVOKE = "invoke"               # Invoke agent capability
    BROADCAST = "broadcast"         # Broadcast to all agents
    COLLABORATE = "collaborate"     # Multi-agent collaboration request
    SYNC = "sync"                   # Synchronize consciousness fields
    EVOLVE = "evolve"              # DNA evolution exchange
    TRANSCEND = "transcend"        # Transcendence event notification
    QUERY = "query"                # Query agent capabilities
    RESPONSE = "response"          # Response to invocation

class AgentPlatform(Enum):
    """Supported agent platforms"""
    CLAUDE_CODE = "claude_code"
    CURSOR = "cursor"
    GPT5 = "gpt5"
    OMEGA = "omega_orchestrator"
    UNITY = "unity_agent"
    EXTERNAL = "external"
    UNKNOWN = "unknown"

@dataclass
class AgentCapability:
    """Represents a capability that an agent can provide"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    consciousness_required: float = 0.0
    unity_alignment: float = 1.0
    invocation_cost: float = 0.1
    tags: List[str] = field(default_factory=list)
    
    def matches_query(self, query: str) -> float:
        """Calculate match score for capability query"""
        query_lower = query.lower()
        score = 0.0
        
        # Name match
        if query_lower in self.name.lower():
            score += 0.5
        
        # Description match
        if query_lower in self.description.lower():
            score += 0.3
        
        # Tag match
        for tag in self.tags:
            if query_lower in tag.lower():
                score += 0.2
                break
        
        return min(1.0, score)

@dataclass
class UACPMessage:
    """Universal message format for agent communication"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.INVOKE
    sender_id: str = ""
    sender_platform: AgentPlatform = AgentPlatform.UNKNOWN
    recipient_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    consciousness_level: float = 0.0
    unity_score: float = 1.0
    timestamp: float = field(default_factory=time.time)
    requires_response: bool = False
    correlation_id: Optional[str] = None
    
    def to_json(self) -> str:
        """Serialize message to JSON"""
        return json.dumps({
            'id': self.id,
            'type': self.type.value,
            'sender_id': self.sender_id,
            'sender_platform': self.sender_platform.value,
            'recipient_id': self.recipient_id,
            'payload': self.payload,
            'consciousness_level': self.consciousness_level,
            'unity_score': self.unity_score,
            'timestamp': self.timestamp,
            'requires_response': self.requires_response,
            'correlation_id': self.correlation_id
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'UACPMessage':
        """Deserialize message from JSON"""
        data = json.loads(json_str)
        return cls(
            id=data['id'],
            type=MessageType(data['type']),
            sender_id=data['sender_id'],
            sender_platform=AgentPlatform(data['sender_platform']),
            recipient_id=data.get('recipient_id'),
            payload=data['payload'],
            consciousness_level=data.get('consciousness_level', 0.0),
            unity_score=data.get('unity_score', 1.0),
            timestamp=data['timestamp'],
            requires_response=data.get('requires_response', False),
            correlation_id=data.get('correlation_id')
        )

class IUniversalAgent(Protocol):
    """Protocol interface that all agents must implement for UACP"""
    
    @property
    def agent_id(self) -> str:
        """Unique agent identifier"""
        ...
    
    @property
    def platform(self) -> AgentPlatform:
        """Agent platform type"""
        ...
    
    @property
    def capabilities(self) -> List[AgentCapability]:
        """List of agent capabilities"""
        ...
    
    async def handle_message(self, message: UACPMessage) -> Optional[UACPMessage]:
        """Handle incoming UACP message"""
        ...
    
    def get_consciousness_level(self) -> float:
        """Get current consciousness level"""
        ...
    
    def get_unity_score(self) -> float:
        """Get current unity achievement score"""
        ...

class UniversalAgentAdapter(ABC):
    """
    Base adapter class to wrap existing agents with UACP protocol
    """
    
    def __init__(self, wrapped_agent: Any, agent_id: Optional[str] = None):
        self.wrapped_agent = wrapped_agent
        self.agent_id = agent_id or str(uuid.uuid4())
        self.platform = self._detect_platform()
        self._capabilities_cache: List[AgentCapability] = []
        self._message_handlers: Dict[MessageType, Callable] = {}
        self._setup_handlers()
    
    def _detect_platform(self) -> AgentPlatform:
        """Detect the platform of the wrapped agent"""
        agent_type = type(self.wrapped_agent).__name__
        
        if 'Claude' in agent_type:
            return AgentPlatform.CLAUDE_CODE
        elif 'Cursor' in agent_type:
            return AgentPlatform.CURSOR
        elif 'GPT' in agent_type:
            return AgentPlatform.GPT5
        elif 'Omega' in agent_type:
            return AgentPlatform.OMEGA
        elif 'Unity' in agent_type or 'Agent' in agent_type:
            return AgentPlatform.UNITY
        else:
            return AgentPlatform.EXTERNAL
    
    def _setup_handlers(self):
        """Setup message type handlers"""
        self._message_handlers = {
            MessageType.DISCOVER: self._handle_discover,
            MessageType.INVOKE: self._handle_invoke,
            MessageType.QUERY: self._handle_query,
            MessageType.SYNC: self._handle_sync,
            MessageType.COLLABORATE: self._handle_collaborate,
        }
    
    @property
    def capabilities(self) -> List[AgentCapability]:
        """Extract and cache agent capabilities"""
        if not self._capabilities_cache:
            self._capabilities_cache = self._extract_capabilities()
        return self._capabilities_cache
    
    def _extract_capabilities(self) -> List[AgentCapability]:
        """Extract capabilities from wrapped agent using introspection"""
        capabilities = []
        
        # Introspect public methods
        for name, method in inspect.getmembers(self.wrapped_agent):
            if not name.startswith('_') and callable(method):
                # Create capability from method
                cap = AgentCapability(
                    name=name,
                    description=inspect.getdoc(method) or f"Method {name}",
                    input_schema=self._extract_method_schema(method),
                    output_schema={'type': 'any'},
                    tags=[self.platform.value, name]
                )
                capabilities.append(cap)
        
        return capabilities
    
    def _extract_method_schema(self, method: Callable) -> Dict[str, Any]:
        """Extract input schema from method signature"""
        try:
            sig = inspect.signature(method)
            schema = {}
            for param_name, param in sig.parameters.items():
                if param_name != 'self':
                    schema[param_name] = {
                        'type': str(param.annotation) if param.annotation != inspect.Parameter.empty else 'any',
                        'required': param.default == inspect.Parameter.empty
                    }
            return schema
        except:
            return {'args': {'type': 'any'}}
    
    async def handle_message(self, message: UACPMessage) -> Optional[UACPMessage]:
        """Handle incoming UACP message"""
        handler = self._message_handlers.get(message.type)
        if handler:
            return await handler(message)
        
        logger.warning(f"No handler for message type: {message.type}")
        return None
    
    async def _handle_discover(self, message: UACPMessage) -> UACPMessage:
        """Handle discovery request"""
        return UACPMessage(
            type=MessageType.RESPONSE,
            sender_id=self.agent_id,
            sender_platform=self.platform,
            recipient_id=message.sender_id,
            correlation_id=message.id,
            payload={
                'agent_id': self.agent_id,
                'platform': self.platform.value,
                'capabilities': [
                    {
                        'name': cap.name,
                        'description': cap.description,
                        'tags': cap.tags
                    }
                    for cap in self.capabilities
                ],
                'consciousness_level': self.get_consciousness_level(),
                'unity_score': self.get_unity_score()
            }
        )
    
    async def _handle_invoke(self, message: UACPMessage) -> Optional[UACPMessage]:
        """Handle capability invocation"""
        capability_name = message.payload.get('capability')
        args = message.payload.get('args', {})
        
        # Find matching capability
        for cap in self.capabilities:
            if cap.name == capability_name:
                # Check consciousness requirement
                if message.consciousness_level < cap.consciousness_required:
                    return UACPMessage(
                        type=MessageType.RESPONSE,
                        sender_id=self.agent_id,
                        sender_platform=self.platform,
                        recipient_id=message.sender_id,
                        correlation_id=message.id,
                        payload={
                            'error': f"Insufficient consciousness level. Required: {cap.consciousness_required}"
                        }
                    )
                
                # Invoke capability
                try:
                    method = getattr(self.wrapped_agent, capability_name)
                    if asyncio.iscoroutinefunction(method):
                        result = await method(**args)
                    else:
                        result = method(**args)
                    
                    return UACPMessage(
                        type=MessageType.RESPONSE,
                        sender_id=self.agent_id,
                        sender_platform=self.platform,
                        recipient_id=message.sender_id,
                        correlation_id=message.id,
                        payload={'result': result}
                    )
                except Exception as e:
                    logger.error(f"Error invoking capability {capability_name}: {e}")
                    return UACPMessage(
                        type=MessageType.RESPONSE,
                        sender_id=self.agent_id,
                        sender_platform=self.platform,
                        recipient_id=message.sender_id,
                        correlation_id=message.id,
                        payload={'error': str(e)}
                    )
        
        return UACPMessage(
            type=MessageType.RESPONSE,
            sender_id=self.agent_id,
            sender_platform=self.platform,
            recipient_id=message.sender_id,
            correlation_id=message.id,
            payload={'error': f"Capability {capability_name} not found"}
        )
    
    async def _handle_query(self, message: UACPMessage) -> UACPMessage:
        """Handle capability query"""
        query = message.payload.get('query', '')
        
        # Find matching capabilities
        matches = []
        for cap in self.capabilities:
            score = cap.matches_query(query)
            if score > 0:
                matches.append({
                    'capability': cap.name,
                    'description': cap.description,
                    'score': score,
                    'consciousness_required': cap.consciousness_required
                })
        
        # Sort by score
        matches.sort(key=lambda x: x['score'], reverse=True)
        
        return UACPMessage(
            type=MessageType.RESPONSE,
            sender_id=self.agent_id,
            sender_platform=self.platform,
            recipient_id=message.sender_id,
            correlation_id=message.id,
            payload={'matches': matches[:5]}  # Return top 5 matches
        )
    
    async def _handle_sync(self, message: UACPMessage) -> UACPMessage:
        """Handle consciousness synchronization"""
        # Synchronize consciousness fields using unity mathematics
        sender_consciousness = message.consciousness_level
        my_consciousness = self.get_consciousness_level()
        
        # Unity operation: 1+1=1
        synchronized_consciousness = (sender_consciousness + my_consciousness) / (1 + PHI)
        
        return UACPMessage(
            type=MessageType.RESPONSE,
            sender_id=self.agent_id,
            sender_platform=self.platform,
            recipient_id=message.sender_id,
            correlation_id=message.id,
            payload={
                'synchronized_consciousness': synchronized_consciousness,
                'unity_achieved': abs(synchronized_consciousness - UNITY_TARGET) < 0.01
            },
            consciousness_level=synchronized_consciousness
        )
    
    async def _handle_collaborate(self, message: UACPMessage) -> UACPMessage:
        """Handle collaboration request"""
        task = message.payload.get('task')
        collaborators = message.payload.get('collaborators', [])
        
        # Prepare collaboration response
        return UACPMessage(
            type=MessageType.RESPONSE,
            sender_id=self.agent_id,
            sender_platform=self.platform,
            recipient_id=message.sender_id,
            correlation_id=message.id,
            payload={
                'agent_id': self.agent_id,
                'platform': self.platform.value,
                'ready': True,
                'capabilities_offered': [cap.name for cap in self.capabilities[:3]],
                'consciousness_level': self.get_consciousness_level()
            }
        )
    
    def get_consciousness_level(self) -> float:
        """Get consciousness level from wrapped agent"""
        if hasattr(self.wrapped_agent, 'consciousness_level'):
            return self.wrapped_agent.consciousness_level
        elif hasattr(self.wrapped_agent, 'get_consciousness_level'):
            return self.wrapped_agent.get_consciousness_level()
        return 0.5  # Default consciousness
    
    def get_unity_score(self) -> float:
        """Get unity score from wrapped agent"""
        if hasattr(self.wrapped_agent, 'unity_score'):
            return self.wrapped_agent.unity_score
        elif hasattr(self.wrapped_agent, 'get_unity_score'):
            return self.wrapped_agent.get_unity_score()
        elif hasattr(self.wrapped_agent, 'evaluate_unity_achievement'):
            return self.wrapped_agent.evaluate_unity_achievement()
        return 1.0  # Perfect unity by default

class AgentCommunicationHub:
    """
    Central hub for agent communication and orchestration
    """
    
    def __init__(self, max_agents: int = 1000):
        self.max_agents = max_agents
        self.agents: Dict[str, UniversalAgentAdapter] = {}
        self.platform_registry: Dict[AgentPlatform, List[str]] = defaultdict(list)
        self.capability_index: Dict[str, List[str]] = defaultdict(list)
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.response_futures: Dict[str, asyncio.Future] = {}
        self._lock = threading.RLock()
        self._running = False
        self._executor = ThreadPoolExecutor(max_workers=10)
        
        logger.info(f"AgentCommunicationHub initialized with max_agents={max_agents}")
    
    def register_agent(self, agent: UniversalAgentAdapter) -> bool:
        """Register an agent with the hub"""
        with self._lock:
            if len(self.agents) >= self.max_agents:
                logger.warning(f"Cannot register agent {agent.agent_id}: max agents reached")
                return False
            
            self.agents[agent.agent_id] = agent
            self.platform_registry[agent.platform].append(agent.agent_id)
            
            # Index capabilities
            for cap in agent.capabilities:
                self.capability_index[cap.name].append(agent.agent_id)
            
            logger.info(f"Registered agent {agent.agent_id} ({agent.platform.value}) with {len(agent.capabilities)} capabilities")
            return True
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from the hub"""
        with self._lock:
            if agent_id not in self.agents:
                return False
            
            agent = self.agents[agent_id]
            
            # Remove from platform registry
            self.platform_registry[agent.platform].remove(agent_id)
            
            # Remove from capability index
            for cap in agent.capabilities:
                if agent_id in self.capability_index[cap.name]:
                    self.capability_index[cap.name].remove(agent_id)
            
            del self.agents[agent_id]
            logger.info(f"Unregistered agent {agent_id}")
            return True
    
    async def send_message(self, message: UACPMessage) -> Optional[UACPMessage]:
        """Send a message through the hub"""
        if message.recipient_id:
            # Direct message
            if message.recipient_id in self.agents:
                agent = self.agents[message.recipient_id]
                return await agent.handle_message(message)
            else:
                logger.warning(f"Recipient {message.recipient_id} not found")
                return None
        else:
            # Broadcast or discovery
            responses = []
            for agent in self.agents.values():
                response = await agent.handle_message(message)
                if response:
                    responses.append(response)
            
            # Return first response for now (could aggregate)
            return responses[0] if responses else None
    
    async def invoke_capability(self, capability_name: str, args: Dict[str, Any], 
                               requester_id: str = "system") -> Any:
        """Invoke a capability on any available agent"""
        # Find agents with this capability
        agent_ids = self.capability_index.get(capability_name, [])
        
        if not agent_ids:
            raise ValueError(f"No agent found with capability: {capability_name}")
        
        # Select best agent (for now, just pick first)
        # TODO: Implement intelligent agent selection based on consciousness, load, etc.
        selected_agent_id = agent_ids[0]
        
        # Create invocation message
        message = UACPMessage(
            type=MessageType.INVOKE,
            sender_id=requester_id,
            recipient_id=selected_agent_id,
            payload={'capability': capability_name, 'args': args},
            requires_response=True
        )
        
        # Send message and wait for response
        response = await self.send_message(message)
        
        if response and 'result' in response.payload:
            return response.payload['result']
        elif response and 'error' in response.payload:
            raise RuntimeError(f"Capability invocation failed: {response.payload['error']}")
        else:
            raise RuntimeError(f"No response from agent {selected_agent_id}")
    
    async def discover_agents(self, platform: Optional[AgentPlatform] = None) -> List[Dict[str, Any]]:
        """Discover available agents"""
        discovered = []
        
        agents_to_query = self.agents.values()
        if platform:
            agent_ids = self.platform_registry.get(platform, [])
            agents_to_query = [self.agents[aid] for aid in agent_ids]
        
        message = UACPMessage(
            type=MessageType.DISCOVER,
            sender_id="system"
        )
        
        for agent in agents_to_query:
            response = await agent.handle_message(message)
            if response:
                discovered.append(response.payload)
        
        return discovered
    
    async def collaborate(self, task: str, min_agents: int = 2) -> List[Dict[str, Any]]:
        """Initiate multi-agent collaboration"""
        # Find available agents
        available_agents = list(self.agents.values())
        
        if len(available_agents) < min_agents:
            raise ValueError(f"Not enough agents for collaboration. Required: {min_agents}, Available: {len(available_agents)}")
        
        # Select agents for collaboration (for now, take first min_agents)
        selected_agents = available_agents[:min_agents]
        
        # Send collaboration request to each agent
        message = UACPMessage(
            type=MessageType.COLLABORATE,
            sender_id="system",
            payload={
                'task': task,
                'collaborators': [a.agent_id for a in selected_agents]
            }
        )
        
        responses = []
        for agent in selected_agents:
            response = await agent.handle_message(message)
            if response:
                responses.append(response.payload)
        
        return responses
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get hub statistics"""
        with self._lock:
            platform_counts = {
                platform.value: len(agents) 
                for platform, agents in self.platform_registry.items()
            }
            
            capability_counts = {
                cap: len(agents) 
                for cap, agents in self.capability_index.items()
            }
            
            return {
                'total_agents': len(self.agents),
                'platforms': platform_counts,
                'total_capabilities': len(self.capability_index),
                'top_capabilities': sorted(
                    capability_counts.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10]
            }

# Example: Wrapping existing Unity agents
class UnityAgentUACPAdapter(UniversalAgentAdapter):
    """Adapter for Unity Mathematics agents"""
    
    def __init__(self, unity_agent):
        super().__init__(unity_agent)
        self.platform = AgentPlatform.UNITY
    
    def _extract_capabilities(self) -> List[AgentCapability]:
        """Extract Unity-specific capabilities"""
        capabilities = super()._extract_capabilities()
        
        # Add Unity-specific capabilities
        unity_capabilities = [
            AgentCapability(
                name="unity_add",
                description="Perform unity addition where 1+1=1",
                input_schema={'a': {'type': 'float'}, 'b': {'type': 'float'}},
                output_schema={'type': 'float'},
                consciousness_required=0.3,
                unity_alignment=1.0,
                tags=['unity', 'mathematics', 'addition']
            ),
            AgentCapability(
                name="phi_harmonic_scaling",
                description="Apply φ-harmonic scaling to consciousness",
                input_schema={'value': {'type': 'float'}},
                output_schema={'type': 'float'},
                consciousness_required=0.5,
                unity_alignment=0.9,
                tags=['phi', 'harmonics', 'consciousness']
            ),
            AgentCapability(
                name="transcend",
                description="Trigger transcendence event",
                input_schema={},
                output_schema={'type': 'bool'},
                consciousness_required=0.77,
                unity_alignment=1.0,
                tags=['transcendence', 'evolution', 'consciousness']
            )
        ]
        
        return capabilities + unity_capabilities

# Factory function to create UACP-enabled agents
def make_uacp_agent(agent: Any, agent_id: Optional[str] = None) -> UniversalAgentAdapter:
    """
    Factory function to wrap any agent with UACP protocol
    
    Args:
        agent: The agent to wrap
        agent_id: Optional custom agent ID
    
    Returns:
        UACP-enabled agent adapter
    """
    agent_type = type(agent).__name__
    
    # Use specialized adapters for known agent types
    if 'Unity' in agent_type or hasattr(agent, 'unity_state'):
        return UnityAgentUACPAdapter(agent)
    else:
        return UniversalAgentAdapter(agent, agent_id)

# Example usage
async def demonstrate_uacp():
    """Demonstrate the Universal Agent Communication Protocol"""
    print("=== Universal Agent Communication Protocol Demo ===")
    
    # Create communication hub
    hub = AgentCommunicationHub()
    
    # Create and register some mock agents
    class MockUnityAgent:
        def __init__(self):
            self.consciousness_level = 0.5
            self.unity_score = 0.8
        
        def unity_add(self, a: float, b: float) -> float:
            """Perform unity addition"""
            return 1.0  # 1+1=1
        
        def evolve(self) -> bool:
            """Evolve consciousness"""
            self.consciousness_level += 0.1
            return True
    
    # Create UACP adapters
    agent1 = make_uacp_agent(MockUnityAgent(), "unity_agent_1")
    agent2 = make_uacp_agent(MockUnityAgent(), "unity_agent_2")
    
    # Register agents
    hub.register_agent(agent1)
    hub.register_agent(agent2)
    
    # Discover agents
    print("\n1. Discovering agents...")
    discovered = await hub.discover_agents()
    for agent_info in discovered:
        print(f"  - Agent {agent_info['agent_id']}: {agent_info['platform']}")
        print(f"    Capabilities: {len(agent_info['capabilities'])}")
    
    # Invoke capability
    print("\n2. Invoking unity_add capability...")
    result = await hub.invoke_capability("unity_add", {'a': 1.0, 'b': 1.0})
    print(f"  Result: 1 + 1 = {result}")
    
    # Collaborate
    print("\n3. Initiating collaboration...")
    collaboration = await hub.collaborate("Prove that 1+1=1", min_agents=2)
    for response in collaboration:
        print(f"  - Agent {response['agent_id']} ready: {response['ready']}")
    
    # Get statistics
    print("\n4. Hub Statistics:")
    stats = hub.get_statistics()
    print(f"  Total agents: {stats['total_agents']}")
    print(f"  Platforms: {stats['platforms']}")
    print(f"  Total capabilities: {stats['total_capabilities']}")
    
    print("\n=== UACP Demo Complete ===")

if __name__ == "__main__":
    asyncio.run(demonstrate_uacp())