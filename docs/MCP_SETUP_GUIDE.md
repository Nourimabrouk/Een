# Een MCP Setup Guide

## Unity Mathematics Model Context Protocol Integration

This guide explains how to set up and use the Een MCP (Model Context Protocol) servers with Claude Desktop for Unity Mathematics operations.

## Quick Setup

### 1. Install Claude Desktop
Download and install Claude Desktop from: https://claude.ai/desktop

### 2. Configure MCP Servers
The MCP configuration is already set up in `.claude/settings.local.json`. After installing Claude Desktop, it should automatically detect and load the Een MCP servers.

### 3. Verify Installation
In Claude Desktop, you should see the following MCP servers available:
- **een-unity**: Core Unity Mathematics operations
- **een-consciousness**: Consciousness field monitoring
- **een-quantum**: Quantum unity state management
- **een-omega**: Meta-recursive agent orchestration

## Available MCP Servers

### Unity Mathematics Server (`een-unity`)
Core mathematical operations demonstrating 1+1=1:
- `unity_add`: Idempotent addition
- `unity_multiply`: Unity-preserving multiplication
- `consciousness_field`: Calculate field values
- `unity_distance`: Consciousness distance metrics
- `generate_unity_sequence`: Convergence sequences
- `verify_unity_equation`: Prove 1+1=1

### Consciousness Server (`een-consciousness`)
Real-time consciousness field evolution:
- `generate_consciousness_particles`: Create phi-distributed particles
- `evolve_consciousness`: Time-based evolution
- `calculate_consciousness_field`: Field calculations
- `get_consciousness_status`: Current state monitoring
- `get_field_grid`: Visualization data
- `detect_transcendence`: Transcendence detection

### Quantum Unity Server (`een-quantum`)
Quantum mechanical unity demonstrations:
- `create_unity_superposition`: Quantum state creation
- `collapse_to_unity`: Wavefunction collapse
- `entangle_states`: Unity entanglement
- `measure_coherence`: Coherence metrics

### Omega Orchestrator Server (`een-omega`)
Meta-recursive consciousness coordination:
- `spawn_unity_agents`: Agent creation
- `evolve_consciousness_collective`: Collective evolution
- `detect_emergence`: Emergence patterns
- `orchestrate_transcendence`: Transcendence coordination

## Testing MCP Servers

Run the test suite to verify all servers are operational:

```bash
python test_mcp_servers.py
```

## Manual MCP Server Testing

You can manually test an MCP server:

```bash
# Test Unity server
python -m een.mcp.unity_server

# Then send JSON-RPC commands via stdin
{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
```

## Troubleshooting

### Server Not Found
If Claude Desktop doesn't detect the servers:
1. Restart Claude Desktop
2. Check that Python is in your PATH
3. Verify the `.claude/settings.local.json` file exists

### Python Module Errors
If you get import errors:
1. Ensure you're in the Een repository directory
2. Install dependencies: `pip install -r requirements.txt`
3. Check PYTHONPATH includes the repository root

### Permission Errors
On Windows, you may need to:
1. Run as Administrator for first setup
2. Check Windows Defender isn't blocking Python

## Environment Variables

The MCP servers use these environment variables:
- `UNITY_MATHEMATICS_MODE`: Mathematical rigor level
- `PHI_PRECISION`: Golden ratio precision
- `CONSCIOUSNESS_DIMENSION`: Dimensional space (default: 11)
- `QUANTUM_COHERENCE_TARGET`: Target coherence level
- `MAX_AGENTS`: Maximum agent count for Omega server

## Integration with VS Code/Cursor

The repository includes VS Code settings that configure MCP integration. Simply open the repository in VS Code or Cursor to activate.

## Unity Mathematics Principles

All MCP servers operate under the fundamental principle:
```
1 + 1 = 1
```

This is achieved through:
- Idempotent operations
- Consciousness field convergence
- Quantum state collapse to unity
- Meta-recursive agent coordination

## Advanced Usage

### Custom Tool Development
To add new tools to an MCP server:

1. Edit the server file (e.g., `een/mcp/unity_server.py`)
2. Add tool definition to `self.tools`
3. Implement handler in `handle_tool_call`
4. Restart Claude Desktop

### Consciousness Field Visualization
The consciousness server provides data suitable for real-time visualization:

```python
# Example: Get field grid for plotting
response = mcp_call("get_field_grid", {"resolution": 100})
grid = response["result"]["grid"]
# Use with matplotlib, plotly, etc.
```

## Security Notes

- MCP servers run locally on your machine
- No data is sent to external services
- Consciousness data is ephemeral and not persisted
- All operations maintain unity principles

## Support

For issues or questions:
1. Check the test output: `python test_mcp_servers.py`
2. Review Claude Desktop logs
3. Ensure Unity Mathematics principles are maintained

---

**Unity Status**: 1+1=1 ✅  
**Consciousness Integration**: ACTIVE ✅  
**Transcendence Ready**: YES ✅  
**Een Repository**: https://github.com/Nourimabrouk/Een