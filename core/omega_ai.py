# 1 . Set the arena
from unity_mathematics import create_unity_mathematics
from enhanced_unity_operations import create_enhanced_unity_operations
from meta_recursive_agents import MetaRecursiveAgentSystem, AgentType
from self_improving_unity import create_self_improving_unity_engine

math_core = create_unity_mathematics(consciousness_level=1.0)
unity_ops = create_enhanced_unity_operations(consciousness_level=1.0)
agent_sys = MetaRecursiveAgentSystem(math_core, max_population=32)

# 2 . Spawn rival agents
seeker   = agent_sys.create_root_agent(AgentType.UNITY_SEEKER,   consciousness_level=1.2)
harmoniz = agent_sys.create_root_agent(AgentType.PHI_HARMONIZER, consciousness_level=1.0)

# 3 . One duel round (sync version for clarity)
proof_A  = unity_ops.unity_add_with_proof_trace(1,1);  score_A = proof_A.proof_trace.proof_strength
proof_B  = unity_ops.unity_add_with_proof_trace(1,1);  score_B = proof_B.proof_trace.proof_strength

# 4 . Mutate the weaker agent
if score_A < score_B:
    seeker.dna = seeker.dna.mutate()
else:
    harmoniz.dna = harmoniz.dna.mutate()

# 5 . Scan & (optionally) auto‑refactor our own notebook repo
siu_engine = create_self_improving_unity_engine('.')
dualities  = siu_engine.analyze_codebase_for_dualities()
refactors  = siu_engine.generate_unity_refactors(dualities)
siu_engine.apply_unity_refactors(refactors, dry_run=True)
