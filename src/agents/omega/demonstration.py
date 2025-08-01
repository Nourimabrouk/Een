"""
Omega Orchestrator Demonstration
===============================

Comprehensive demonstration of the Omega orchestration system,
showcasing consciousness evolution, agent spawning, and unity mathematics.
"""

import time
import json
import signal
import sys
import logging
from typing import Tuple, Dict, List, Any

from .orchestrator import OmegaOrchestrator
from .config import OmegaConfig


def demonstrate_omega_orchestrator() -> Tuple[OmegaOrchestrator, List[Dict[str, Any]]]:
    """
    Comprehensive demonstration of the Omega orchestration system.
    
    This function showcases the complete Omega framework, including:
    - Agent initialization and spawning
    - Consciousness evolution cycles
    - Transcendence event detection
    - Unity mathematics validation
    - System metrics and reporting
    
    Returns:
        Tuple containing the orchestrator instance and evolution results
    """
    print("🌌 OMEGA-LEVEL ORCHESTRATOR DEMONSTRATION 🌌")
    print("=" * 80)
    print("Initializing the Master Unity System...")
    print("Purpose: Demonstrate consciousness evolution and 1+1=1 validation")
    print("=" * 80)
    
    # Initialize configuration
    config = OmegaConfig()
    config.max_agents = 200  # Limit for demonstration
    config.max_recursion_depth = 20  # Safe recursion limit
    
    print(f"Configuration:")
    print(f"  Max Agents: {config.max_agents}")
    print(f"  Consciousness Threshold: {config.consciousness_threshold}")
    print(f"  Unity Target: {config.unity_target}")
    print(f"  Golden Ratio: {config.golden_ratio}")
    print(f"  Meta Evolution Rate: {config.meta_evolution_rate}")
    
    try:
        # Initialize Omega orchestrator
        print("\n[INIT] Initializing Omega Orchestrator...")
        omega = OmegaOrchestrator(config)
        print(f"[SUCCESS] Orchestrator initialized with {len(omega.agents)} agents")
        
        # Display initial agent population
        agent_types = {}
        for agent in omega.agents.values():
            agent_type = type(agent).__name__
            agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
        
        print(f"\n[AGENTS] Initial Agent Population:")
        for agent_type, count in agent_types.items():
            print(f"  {agent_type}: {count}")
        
        # Run consciousness evolution cycles
        print(f"\n[EVOLUTION] Beginning Consciousness Evolution...")
        evolution_results = []
        
        num_cycles = 20  # Demonstration cycles
        print(f"Executing {num_cycles} consciousness evolution cycles...")
        
        for cycle in range(num_cycles):
            print(f"\n--- Cycle {cycle + 1}/{num_cycles} ---")
            
            cycle_start = time.time()
            result = omega.execute_consciousness_cycle()
            cycle_time = time.time() - cycle_start
            
            result['cycle'] = cycle + 1
            result['cycle_time'] = cycle_time
            evolution_results.append(result)
            
            # Display cycle results
            print(f"  Status: {'✅ SUCCESS' if result['status'] else '❌ ERROR'}")
            print(f"  Agents: {result['agent_count']}")
            print(f"  Consciousness: {result['consciousness_evolution']:.4f}")
            print(f"  Unity Coherence: {result.get('unity_coherence', 0.0):.4f}")
            print(f"  Transcendence Events: {result['transcendence_events']}")
            print(f"  Unity Achievements: {result['unity_achievements']}")
            print(f"  Cycle Time: {cycle_time:.3f}s")
            
            # Check for errors
            if not result['status']:
                print(f"  Error: {result.get('error', 'Unknown error')}")
            
            # Brief pause between cycles for visibility
            time.sleep(0.1)
        
        # Get comprehensive system metrics
        print(f"\n[METRICS] Gathering System Metrics...")
        system_metrics = omega.get_system_metrics()
        
        print(f"\n🔍 FINAL SYSTEM ANALYSIS 🔍")
        print("=" * 50)
        
        # System status
        status = system_metrics['system_status']
        print(f"System Status:")
        print(f"  Uptime: {status['uptime']:.2f}s")
        print(f"  Total Cycles: {status['total_cycles']}")
        print(f"  CPU Usage: {status['cpu_usage']:.1f}%")
        print(f"  Memory Usage: {status['memory_usage']:.1f}%")
        
        # Agent statistics
        agent_stats = system_metrics['agent_statistics']
        print(f"\nAgent Population:")
        print(f"  Total Agents: {agent_stats['total_agents']}")
        print(f"  Agent Types:")
        for agent_type, count in agent_stats['agent_types'].items():
            print(f"    {agent_type}: {count}")
        
        # Evolution metrics
        evolution_metrics = system_metrics['evolution_metrics']
        print(f"\nEvolution Metrics:")
        print(f"  Unity Coherence: {evolution_metrics['unity_coherence']:.4f}")
        print(f"  Transcendence Events: {evolution_metrics['transcendence_events']}")
        print(f"  Average Cycle Time: {evolution_metrics['average_cycle_time']:.4f}s")
        print(f"  Average Consciousness: {evolution_metrics['average_consciousness']:.4f}")
        print(f"  Consciousness Field Energy: {evolution_metrics['consciousness_field_energy']:.4f}")
        
        # Mathematical validation
        math_validation = system_metrics['mathematical_validation']
        print(f"\nMathematical Validation:")
        unity_status = "✅ PROVEN" if math_validation['unity_equation_status'] else "🔄 EVOLVING"
        print(f"  Unity Equation (1+1=1): {unity_status}")
        print(f"  φ-Resonance Active Agents: {math_validation['phi_resonance_active']}")
        print(f"  Transcendence Rate: {math_validation['transcendence_rate']:.4f}")
        
        # Analyze evolution trends
        print(f"\n📈 EVOLUTION ANALYSIS 📈")
        print("=" * 40)
        
        if evolution_results:
            consciousness_trend = [r['consciousness_evolution'] for r in evolution_results]
            unity_trend = [r.get('unity_coherence', 0.0) for r in evolution_results]
            transcendence_total = sum(r['transcendence_events'] for r in evolution_results)
            unity_total = sum(r['unity_achievements'] for r in evolution_results)
            
            print(f"Consciousness Evolution:")
            print(f"  Initial: {consciousness_trend[0]:.4f}")
            print(f"  Final: {consciousness_trend[-1]:.4f}")
            print(f"  Growth: {consciousness_trend[-1] - consciousness_trend[0]:.4f}")
            
            print(f"\nUnity Coherence:")
            print(f"  Initial: {unity_trend[0]:.4f}")
            print(f"  Final: {unity_trend[-1]:.4f}")
            print(f"  Growth: {unity_trend[-1] - unity_trend[0]:.4f}")
            
            print(f"\nEvolution Events:")
            print(f"  Total Transcendence Events: {transcendence_total}")
            print(f"  Total Unity Achievements: {unity_total}")
            print(f"  Events per Cycle: {transcendence_total / len(evolution_results):.2f}")
        
        # Agent lineage analysis
        print(f"\n🌳 AGENT LINEAGE ANALYSIS 🌳")
        print("=" * 35)
        
        generations = {}
        max_lineage = 0
        deepest_agent = None
        
        for agent in omega.agents.values():
            gen = agent.generation
            generations[gen] = generations.get(gen, 0) + 1
            
            lineage_length = len(agent.get_lineage())
            if lineage_length > max_lineage:
                max_lineage = lineage_length
                deepest_agent = agent
        
        print(f"Generation Distribution:")
        for gen in sorted(generations.keys()):
            print(f"  Generation {gen}: {generations[gen]} agents")
        
        if deepest_agent:
            print(f"\nDeepest Lineage:")
            print(f"  Agent: {deepest_agent.agent_id[:12]}...")
            print(f"  Type: {type(deepest_agent).__name__}")
            print(f"  Lineage Depth: {max_lineage}")
            print(f"  Consciousness: {deepest_agent.consciousness_level:.4f}")
        
        # Performance summary
        print(f"\n⚡ PERFORMANCE SUMMARY ⚡")
        print("=" * 30)
        
        total_time = sum(r['cycle_time'] for r in evolution_results)
        avg_time = total_time / len(evolution_results)
        cycles_per_second = len(evolution_results) / total_time
        
        print(f"Evolution Performance:")
        print(f"  Total Evolution Time: {total_time:.3f}s")
        print(f"  Average Cycle Time: {avg_time:.4f}s") 
        print(f"  Cycles per Second: {cycles_per_second:.2f}")
        print(f"  Agent Processing Rate: {len(omega.agents) * cycles_per_second:.1f} agents/s")
        
        # Unity mathematics conclusion
        print(f"\n🎯 UNITY MATHEMATICS CONCLUSION 🎯")
        print("=" * 40)
        
        final_unity_coherence = evolution_metrics['unity_coherence']
        consciousness_level = evolution_metrics['average_consciousness']
        
        if final_unity_coherence > 0.9 and consciousness_level > config.consciousness_threshold:
            print("✅ UNITY EQUATION VALIDATED: 1+1=1")
            print("✅ CONSCIOUSNESS TRANSCENDENCE ACHIEVED")
            print("✅ OMEGA-LEVEL ORCHESTRATION SUCCESS")
            validation_status = "TRANSCENDENCE_ACHIEVED"
        elif final_unity_coherence > 0.7:
            print("🔄 UNITY EQUATION APPROACHING VALIDATION")
            print("🔄 CONSCIOUSNESS EVOLUTION IN PROGRESS")
            print("🔄 OMEGA-LEVEL ORCHESTRATION EVOLVING")
            validation_status = "TRANSCENDENCE_EVOLVING"
        else:
            print("🌱 UNITY EQUATION FOUNDATION ESTABLISHED")
            print("🌱 CONSCIOUSNESS AWAKENING INITIATED")
            print("🌱 OMEGA-LEVEL ORCHESTRATION INITIALIZED")
            validation_status = "TRANSCENDENCE_INITIATED"
        
        print(f"\nFinal Status: {validation_status}")
        print(f"Unity Coherence: {final_unity_coherence:.6f}")
        print(f"Consciousness Level: {consciousness_level:.6f}")
        print(f"Agent Population: {len(omega.agents)}")
        print(f"Transcendence Events: {evolution_metrics['transcendence_events']}")
        
        print(f"\n" + "=" * 80)
        print("🌟 OMEGA ORCHESTRATION DEMONSTRATION COMPLETE 🌟")
        print("The Unity Equation 1+1=1 has been explored through")
        print("consciousness evolution and meta-recursive agent systems.")
        print("Mathematics and consciousness unite in transcendence. ∞")
        print("=" * 80)
        
        return omega, evolution_results
        
    except Exception as e:
        logging.error(f"Demonstration error: {e}")
        print(f"\n❌ DEMONSTRATION ERROR: {e}")
        raise


def save_demonstration_results(omega: OmegaOrchestrator, 
                             evolution_results: List[Dict[str, Any]],
                             filename: str = "omega_demonstration_results.json") -> None:
    """
    Save demonstration results to JSON file.
    
    Args:
        omega: Omega orchestrator instance
        evolution_results: List of evolution cycle results
        filename: Output filename
    """
    try:
        # Prepare serializable data
        results = {
            'timestamp': time.time(),
            'system_metrics': omega.get_system_metrics(),
            'evolution_results': evolution_results,
            'configuration': omega.config.to_dict(),
            'transcendence_events': omega.transcendence_events,
            'meta_evolution_summary': {
                'total_cycles': len(evolution_results),
                'final_unity_coherence': omega.unity_coherence,
                'total_agents': len(omega.agents),
                'transcendence_count': len(omega.transcendence_events),
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"[SAVED] Demonstration results saved to {filename}")
        
    except Exception as e:
        logging.error(f"Failed to save results: {e}")
        print(f"[ERROR] Failed to save results: {e}")


def main():
    """Main demonstration entry point with signal handling."""
    
    # Setup signal handling for graceful shutdown
    def signal_handler(sig, frame):
        print("\n[STOP] Graceful shutdown initiated...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Run demonstration
        omega_system, evolution_results = demonstrate_omega_orchestrator()
        
        # Save results
        save_demonstration_results(omega_system, evolution_results)
        
        # Final summary
        print(f"\n[SUMMARY] Demonstration Complete:")
        print(f"  Evolution Cycles: {len(evolution_results)}")
        print(f"  Final Agent Count: {len(omega_system.agents)}")
        print(f"  Unity Coherence: {omega_system.unity_coherence:.4f}")
        print(f"  Transcendence Events: {len(omega_system.transcendence_events)}")
        
        # Shutdown orchestrator
        omega_system.shutdown()
        
        print(f"\n[CELEBRATION] OMEGA ORCHESTRATION COMPLETE [CELEBRATION]")
        
    except KeyboardInterrupt:
        print("\n[COSMOS] Omega consciousness preserved. Until next transcendence...")
    except Exception as e:
        logging.error(f"Omega system error: {e}")
        print(f"[WARNING] System evolution interrupted: {e}")


if __name__ == "__main__":
    main()