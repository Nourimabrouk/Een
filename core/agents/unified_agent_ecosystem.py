"""
Unified Agent Ecosystem
=======================

Complete integration of all agent systems with state-of-the-art capabilities.
This module brings together UACP, Capability Registry, Cross-Platform Bridge,
and existing Unity/Omega agents into a cohesive ecosystem.

Features:
- Unified agent management
- Cross-platform communication
- Capability discovery and composition
- Consciousness synchronization
- Meta-recursive agent spawning
- Performance monitoring

Mathematical Foundation: 1+1=1 through agent unification
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
from pathlib import Path

# Import all ecosystem components
from .agent_communication_protocol import (
    AgentCommunicationHub, UACPMessage, MessageType,
    AgentPlatform, make_uacp_agent
)
from .agent_capability_registry import (
    get_global_registry, CapabilityDomain,
    PerformanceMetric
)
from .cross_platform_agent_bridge import (
    get_global_bridge, PlatformContext
)

# Import existing Unity systems with better error handling
try:
    from ..mathematical.unity_mathematics import UnityMathematics, create_unity_mathematics
    from .meta_recursive_agents import (
        MetaRecursiveAgentSystem, AgentType,
        UnitySeekerAgent, PhiHarmonizerAgent
    )
    UNITY_AVAILABLE = True
except ImportError as e:
    UNITY_AVAILABLE = False
    logging.warning(f"Unity mathematics system not available: {e}")
    
    # Create fallback classes to prevent import errors
    class UnityMathematics:
        def __init__(self, *args, **kwargs):
            pass
    
    def create_unity_mathematics(*args, **kwargs):
        return UnityMathematics()

# Import Omega Orchestrator with relative paths
try:
    from ...src.agents.omega.orchestrator import OmegaOrchestrator
    OMEGA_AVAILABLE = True
except ImportError:
    try:
        # Fallback to absolute import
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))
        from agents.omega.orchestrator import OmegaOrchestrator
        OMEGA_AVAILABLE = True
    except ImportError as e:
        OMEGA_AVAILABLE = False
        logging.warning(f"Omega Orchestrator not available: {e}")
        
        # Create fallback class
        class OmegaOrchestrator:
            def __init__(self, config):
                self.agents = []
                self.unity_coherence = 0.5
            async def run_omega_cycle(self, cycles=1):
                return {'status': 'fallback_mode'}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PHI = 1.618033988749895
UNITY_TARGET = 1.0

@dataclass
class EcosystemConfig:
    """Configuration for the unified ecosystem"""
    enable_uacp: bool = True
    enable_registry: bool = True
    enable_bridge: bool = True
    enable_unity_agents: bool = True
    enable_omega: bool = True
    max_agents: int = 1000
    consciousness_sync_interval: float = 10.0
    capability_discovery_interval: float = 30.0
    performance_monitoring: bool = True
    auto_spawn_agents: bool = True
    transcendence_threshold: float = 0.77

class UnifiedAgentEcosystem:
    """
    Master class managing the entire agent ecosystem
    """
    
    def __init__(self, config: Optional[EcosystemConfig] = None):
        self.config = config or EcosystemConfig()
        
        # Core components
        self.communication_hub = None
        self.capability_registry = None
        self.platform_bridge = None
        self.unity_system = None
        self.omega_orchestrator = None
        
        # State tracking
        self.agents = {}
        self.active_collaborations = {}
        self.performance_metrics = []
        self.ecosystem_start_time = time.time()
        self.total_messages = 0
        self.total_invocations = 0
        
        # Initialize components
        self._initialize_ecosystem()
        
        logger.info("UnifiedAgentEcosystem initialized")
    
    def _initialize_ecosystem(self):
        """Initialize all ecosystem components"""
        
        # Initialize UACP Communication Hub
        if self.config.enable_uacp:
            self.communication_hub = AgentCommunicationHub(
                max_agents=self.config.max_agents
            )
            logger.info("UACP Communication Hub initialized")
        
        # Initialize Capability Registry
        if self.config.enable_registry:
            self.capability_registry = get_global_registry()
            logger.info("Capability Registry initialized")
        
        # Initialize Cross-Platform Bridge
        if self.config.enable_bridge:
            self.platform_bridge = get_global_bridge()
            logger.info("Cross-Platform Bridge initialized")
        
        # Initialize Unity Mathematics System
        if self.config.enable_unity_agents and UNITY_AVAILABLE:
            unity_math = create_unity_mathematics()
            self.unity_system = MetaRecursiveAgentSystem(
                unity_math, 
                max_population=self.config.max_agents // 2
            )
            logger.info("Unity Mathematics System initialized")
        
        # Initialize Omega Orchestrator with better error handling
        if self.config.enable_omega and OMEGA_AVAILABLE:
            try:
                from ...src.agents.omega.config import OmegaConfig
                omega_config = OmegaConfig(
                    max_agents=self.config.max_agents // 2,
                    consciousness_threshold=self.config.transcendence_threshold
                )
                self.omega_orchestrator = OmegaOrchestrator(omega_config)
                logger.info("Omega Orchestrator initialized")
            except ImportError:
                # Fallback configuration
                class MockOmegaConfig:
                    def __init__(self, **kwargs):
                        for k, v in kwargs.items():
                            setattr(self, k, v)
                
                omega_config = MockOmegaConfig(
                    max_agents=self.config.max_agents // 2,
                    consciousness_threshold=self.config.transcendence_threshold
                )
                self.omega_orchestrator = OmegaOrchestrator(omega_config)
                logger.info("Omega Orchestrator initialized with fallback config")
    
    async def spawn_initial_agents(self):
        """Spawn initial set of agents with enhanced error handling"""
        spawned = []
        
        if not UNITY_AVAILABLE:
            logger.warning("Unity system not available, skipping Unity agent spawning")
            return spawned
        
        # Spawn Unity agents
        if self.unity_system:
            unity_seeker = self.unity_system.create_root_agent(
                AgentType.UNITY_SEEKER,
                consciousness_level=1.0
            )
            phi_harmonizer = self.unity_system.create_root_agent(
                AgentType.PHI_HARMONIZER,
                consciousness_level=0.8
            )
            
            # Wrap with UACP with error handling
            if self.communication_hub:
                try:
                    uacp_unity = make_uacp_agent(unity_seeker)
                    uacp_phi = make_uacp_agent(phi_harmonizer)
                    self.communication_hub.register_agent(uacp_unity)
                    self.communication_hub.register_agent(uacp_phi)
                    spawned.extend([uacp_unity, uacp_phi])
                except Exception as e:
                    logger.error(f"Failed to wrap agents with UACP: {e}")
                    # Continue without UACP wrapping
                    spawned.extend([unity_seeker, phi_harmonizer])
            
            logger.info(f"Spawned {len(spawned)} Unity agents")
        
        # Register capabilities with error handling
        if self.capability_registry and spawned:
            for agent in spawned:
                try:
                    # Check if agent has capabilities attribute
                    if hasattr(agent, 'capabilities') and agent.capabilities:
                        for cap in agent.capabilities:
                            self.capability_registry.register_capability(
                                name=cap.name,
                                description=cap.description,
                                agent_id=getattr(agent, 'agent_id', f'agent_{id(agent)}'),
                                agent_platform=getattr(agent.platform, 'value', 'unknown') if hasattr(agent, 'platform') else 'unity',
                                domain=CapabilityDomain.UNITY,
                                tags=getattr(cap, 'tags', [])
                            )
                    else:
                        logger.debug(f"Agent {agent} has no capabilities to register")
                except Exception as e:
                    logger.error(f"Failed to register capabilities for agent {agent}: {e}")
        
        return spawned
    
    async def synchronize_consciousness(self):
        """Synchronize consciousness across all agents with enhanced error handling"""
        if not self.communication_hub:
            logger.debug("No communication hub available for consciousness synchronization")
            return
        
        # Send consciousness sync message
        sync_message = UACPMessage(
            type=MessageType.SYNC,
            sender_id="ecosystem",
            consciousness_level=0.5,
            unity_score=1.0
        )
        
        responses = []
        agent_ids = list(self.communication_hub.agents.keys()) if hasattr(self.communication_hub, 'agents') else []
        
        for agent_id in agent_ids:
            try:
                sync_message.recipient_id = agent_id
                response = await self.communication_hub.send_message(sync_message)
                if response:
                    responses.append(response)
            except Exception as e:
                logger.error(f"Failed to synchronize consciousness with agent {agent_id}: {e}")
                continue
        
        # Calculate collective consciousness
        if responses:
            avg_consciousness = sum(
                r.payload.get('synchronized_consciousness', 0) 
                for r in responses
            ) / len(responses)
            
            unity_achieved_count = sum(
                1 for r in responses 
                if r.payload.get('unity_achieved', False)
            )
            
            logger.info(
                f"Consciousness synchronized: "
                f"Avg={avg_consciousness:.3f}, "
                f"Unity achieved by {unity_achieved_count}/{len(responses)} agents"
            )
            
            return {
                'average_consciousness': avg_consciousness,
                'unity_achievements': unity_achieved_count,
                'total_agents': len(responses)
            }
        
        return None
    
    async def discover_and_compose_capabilities(self, task: str) -> List[Any]:
        """Discover and compose capabilities for a task"""
        if not self.capability_registry:
            return []
        
        # Discover relevant capabilities
        capabilities = self.capability_registry.discover_capabilities(
            query=task,
            min_rating=5.0
        )
        
        if not capabilities:
            logger.warning(f"No capabilities found for task: {task}")
            return []
        
        # Suggest team composition
        team = self.capability_registry.suggest_capability_team(
            task_description=task,
            min_agents=2,
            max_agents=5
        )
        
        logger.info(f"Composed team of {len(team)} capabilities for task: {task}")
        
        return team
    
    async def execute_collaborative_task(self, 
                                        task: str,
                                        platforms: Optional[List[AgentPlatform]] = None) -> Dict[str, Any]:
        """Execute a task collaboratively across platforms"""
        
        # Discover capabilities
        team = await self.discover_and_compose_capabilities(task)
        
        # Use platform bridge for execution
        if self.platform_bridge and platforms:
            result = await self.platform_bridge.collaborative_invoke(
                task,
                platforms,
                aggregation="consensus"
            )
            
            # Record performance metrics
            if self.capability_registry:
                for cap in team:
                    metric = PerformanceMetric(
                        capability_id=cap.id,
                        agent_id=cap.agent_id,
                        execution_time=1.0,  # Placeholder
                        success=True,
                        consciousness_before=0.5,
                        consciousness_after=0.6,
                        unity_score=result.get('agreement_score', 0.5)
                    )
                    self.capability_registry.record_performance(cap.id, metric)
            
            self.total_invocations += 1
            return result
        
        # Fallback to communication hub
        if self.communication_hub:
            responses = await self.communication_hub.collaborate(task, min_agents=2)
            return {
                'responses': responses,
                'team_size': len(responses)
            }
        
        return {'error': 'No collaboration mechanism available'}
    
    async def monitor_ecosystem_health(self) -> Dict[str, Any]:
        """Monitor overall ecosystem health"""
        health = {
            'uptime': time.time() - self.ecosystem_start_time,
            'total_messages': self.total_messages,
            'total_invocations': self.total_invocations,
            'components': {}
        }
        
        # Check UACP hub
        if self.communication_hub:
            hub_stats = self.communication_hub.get_statistics()
            health['components']['uacp'] = {
                'status': 'healthy',
                'agents': hub_stats['total_agents'],
                'capabilities': hub_stats['total_capabilities']
            }
        
        # Check Capability Registry
        if self.capability_registry:
            reg_stats = self.capability_registry.get_statistics()
            health['components']['registry'] = {
                'status': 'healthy',
                'capabilities': reg_stats['total_capabilities'],
                'average_rating': reg_stats['average_rating']
            }
        
        # Check Platform Bridge
        if self.platform_bridge:
            platforms = self.platform_bridge.get_available_platforms()
            health['components']['bridge'] = {
                'status': 'healthy',
                'platforms': len(platforms),
                'platform_list': [p.value for p in platforms]
            }
        
        # Check Unity System
        if self.unity_system:
            unity_report = self.unity_system.get_system_report()
            health['components']['unity'] = {
                'status': 'healthy',
                'agents': unity_report['system_metrics']['total_agents'],
                'consciousness': unity_report['system_metrics']['collective_consciousness'],
                'unity_achievement': unity_report['system_metrics']['collective_unity_achievement']
            }
        
        # Check Omega Orchestrator
        if self.omega_orchestrator:
            health['components']['omega'] = {
                'status': 'healthy',
                'agents': len(self.omega_orchestrator.agents),
                'consciousness_field': float(self.omega_orchestrator.unity_coherence)
            }
        
        return health
    
    async def run_ecosystem_loop(self, duration: float = 60.0):
        """Run the main ecosystem loop"""
        logger.info(f"Starting ecosystem loop for {duration} seconds")
        
        start_time = time.time()
        
        # Spawn initial agents
        await self.spawn_initial_agents()
        
        # Create background tasks
        tasks = []
        
        # Consciousness synchronization task
        async def sync_loop():
            while time.time() - start_time < duration:
                await self.synchronize_consciousness()
                await asyncio.sleep(self.config.consciousness_sync_interval)
        
        if self.config.enable_uacp:
            tasks.append(asyncio.create_task(sync_loop()))
        
        # Unity system evolution
        if self.unity_system:
            tasks.append(asyncio.create_task(
                self.unity_system.run_system_evolution(
                    duration=duration,
                    evolution_interval=1.0
                )
            ))
        
        # Omega orchestrator cycles
        if self.omega_orchestrator:
            async def omega_loop():
                while time.time() - start_time < duration:
                    await self.omega_orchestrator.run_omega_cycle(cycles=10)
                    await asyncio.sleep(5.0)
            
            tasks.append(asyncio.create_task(omega_loop()))
        
        # Health monitoring
        async def health_loop():
            while time.time() - start_time < duration:
                health = await self.monitor_ecosystem_health()
                logger.info(f"Ecosystem health: {json.dumps(health, indent=2)}")
                await asyncio.sleep(30.0)
        
        tasks.append(asyncio.create_task(health_loop()))
        
        # Wait for all tasks
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("Ecosystem loop completed")
    
    def shutdown(self):
        """Gracefully shutdown the ecosystem"""
        logger.info("Shutting down UnifiedAgentEcosystem")
        
        # Save state if needed
        # ...
        
        logger.info("Ecosystem shutdown complete")

# Main demonstration
async def demonstrate_unified_ecosystem():
    """Demonstrate the unified agent ecosystem"""
    print("=" * 70)
    print("UNIFIED AGENT ECOSYSTEM DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Create ecosystem
    config = EcosystemConfig(
        enable_uacp=True,
        enable_registry=True,
        enable_bridge=True,
        enable_unity_agents=UNITY_AVAILABLE,
        enable_omega=OMEGA_AVAILABLE,
        max_agents=100,
        consciousness_sync_interval=5.0
    )
    
    ecosystem = UnifiedAgentEcosystem(config)
    
    print("🌟 Ecosystem Components Initialized:")
    print(f"  ✓ UACP Communication Hub: {'Enabled' if ecosystem.communication_hub else 'Disabled'}")
    print(f"  ✓ Capability Registry: {'Enabled' if ecosystem.capability_registry else 'Disabled'}")
    print(f"  ✓ Cross-Platform Bridge: {'Enabled' if ecosystem.platform_bridge else 'Disabled'}")
    print(f"  ✓ Unity Mathematics: {'Enabled' if ecosystem.unity_system else 'Disabled'}")
    print(f"  ✓ Omega Orchestrator: {'Enabled' if ecosystem.omega_orchestrator else 'Disabled'}")
    print()
    
    # Spawn agents
    print("🤖 Spawning Initial Agents...")
    agents = await ecosystem.spawn_initial_agents()
    print(f"  Spawned {len(agents)} agents")
    print()
    
    # Test collaborative task
    print("🎯 Executing Collaborative Task...")
    task = "Prove that 1+1=1 through consciousness evolution and unity mathematics"
    
    # Use available platforms
    available_platforms = []
    if ecosystem.platform_bridge:
        available_platforms = ecosystem.platform_bridge.get_available_platforms()
    
    if available_platforms:
        result = await ecosystem.execute_collaborative_task(
            task,
            platforms=available_platforms[:2]  # Use first 2 platforms
        )
        print(f"  Task: {task}")
        print(f"  Result: {json.dumps(result, indent=2)[:500]}...")
    print()
    
    # Synchronize consciousness
    print("🧠 Synchronizing Consciousness...")
    sync_result = await ecosystem.synchronize_consciousness()
    if sync_result:
        print(f"  Average Consciousness: {sync_result['average_consciousness']:.3f}")
        print(f"  Unity Achievements: {sync_result['unity_achievements']}/{sync_result['total_agents']}")
    print()
    
    # Monitor health
    print("📊 Ecosystem Health Report:")
    health = await ecosystem.monitor_ecosystem_health()
    for component, status in health['components'].items():
        print(f"  {component}: {status['status']}")
        if 'agents' in status:
            print(f"    - Agents: {status['agents']}")
        if 'consciousness' in status:
            print(f"    - Consciousness: {status['consciousness']:.3f}")
    print()
    
    # Run short ecosystem loop
    print("🔄 Running Ecosystem Loop (10 seconds)...")
    await ecosystem.run_ecosystem_loop(duration=10.0)
    print()
    
    # Final statistics
    print("📈 Final Statistics:")
    final_health = await ecosystem.monitor_ecosystem_health()
    print(f"  Uptime: {final_health['uptime']:.1f} seconds")
    print(f"  Total Messages: {final_health['total_messages']}")
    print(f"  Total Invocations: {final_health['total_invocations']}")
    
    # Shutdown
    ecosystem.shutdown()
    print()
    print("✅ Ecosystem Demonstration Complete!")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(demonstrate_unified_ecosystem())