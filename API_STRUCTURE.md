# Een Unity Mathematics - API Structure

## Core Python APIs

### `core.unity_mathematics.UnityMathematics`
Main unity mathematics engine with phi-harmonic operations.

```python
from core.unity_mathematics import UnityMathematics

unity = UnityMathematics()
result = unity.unity_add(1, 1)  # Returns 1.0
```

**Key Methods:**
- `unity_add(a, b)` - Unity addition (idempotent)
- `unity_multiply(a, b)` - Unity multiplication  
- `demonstrate_unity_addition()` - Multi-domain proof
- `calculate_phi_harmonic(x)` - Golden ratio transformation

### `core.consciousness.ConsciousnessFieldEquations`
Consciousness field dynamics with 11D→4D projections.

```python
from core.consciousness import ConsciousnessFieldEquations

field = ConsciousnessFieldEquations()
evolution = field.evolve_field(steps=1000)
coherence = field.calculate_coherence(evolution)
```

**Key Methods:**
- `evolve_field(steps)` - Evolve consciousness field
- `calculate_coherence(field)` - Measure unity coherence
- `detect_transcendence_events()` - Find emergence points
- `generate_11d_to_4d_projection()` - Dimensional reduction

### `ml_framework.meta_reinforcement.UnityMetaAgent`
Meta-reinforcement learning for unity discovery.

```python
from ml_framework.meta_reinforcement import UnityMetaAgent

agent = UnityMetaAgent()
discoveries = agent.discover_unity_proofs()
```

## Web APIs

### Unity API Server
RESTful API for unity mathematics operations.

**Endpoints:**
- `GET /api/unity/add?a=1&b=1` - Unity addition
- `POST /api/consciousness/evolve` - Evolve field
- `GET /api/proofs` - List all proofs
- `GET /api/visualizations` - Get visualizations

### MCP Servers
Model Context Protocol servers for AI integration.

**Available Servers:**
- `unity_server.py` - Core unity operations
- `consciousness_server.py` - Consciousness field
- `quantum_server.py` - Quantum unity
- `omega_server.py` - Meta-recursive agents

## JavaScript APIs

### `website/js/unity-api-manager.js`
Client-side API management for web interface.

```javascript
const unityAPI = new UnityAPIManager();
const result = await unityAPI.unityAdd(1, 1);
```

## Formal Verification APIs

### Lean 4 Proofs
Formal mathematical verification in `formal_proofs/lean4/`.

**Key Theorems:**
- `ZeroKnowledgeUnityProof.lean` - Cryptographic proof
- `RevolutionaryUnityProof.lean` - Phi-harmonic analysis
- `UltimateUnityTheorem.lean` - Quantum field theory

## Documentation

For detailed API documentation:
- [API Documentation](docs/api/)
- [Integration Guide](docs/UNITY_INTEGRATION_GUIDE.md)
- [MCP Setup](docs/MCP_SETUP_GUIDE.md)