# Repository Consolidation Analysis

## Discovered Structure Overlaps

### 1. AGENTS - Complete Duplication
**`core/agents/` = `een/agents/`** (EXACT DUPLICATES):
- `agent_capability_registry.py`
- `agent_communication_protocol.py` 
- `cross_platform_agent_bridge.py`
- `meta_recursive_agents.py`
- `unified_agent_ecosystem.py`

**`src/agents/`** (UNIQUE CONTENT):
- Advanced agent systems (omega/, consciousness_chat, etc.)
- More mature agent implementations
- **KEEP THIS** as the primary agents directory

### 2. Directory Analysis Summary

**CORE/**: Fundamental systems, some duplicates with een/
**SRC/**: Most complete implementation, unique advanced systems
**EEN/**: Has unique content (dashboards, mcp servers, experiments) but duplicates core/

## Consolidation Strategy

### Meta-Optimal Unified Structure:
```
src/
├── core/                    # From core/ (mathematical, fundamental)
├── agents/                  # Keep src/agents/ (most complete) + unique from een/
├── dashboards/              # From een/dashboards/ + src/dashboards/
├── experiments/             # From een/experiments/
├── mcp/                     # From een/mcp/ (MCP servers)
├── proofs/                  # From een/proofs/
└── [existing src/ content]  # Keep all existing src/
```

## Files to Move/Merge

### Keep from SRC (most mature):
- All existing src/ content (agents/, consciousness/, etc.)

### Merge from CORE:
- `core/mathematical/` → `src/core/mathematical/`
- `core/consciousness/` → merge with `src/consciousness/`
- `core/visualization/` → `src/core/visualization/`

### Merge from EEN (unique content):
- `een/dashboards/` → `src/dashboards/` (merge with existing)
- `een/experiments/` → `src/experiments/`
- `een/mcp/` → `src/mcp/`
- `een/proofs/` → `src/proofs/`

### Remove Duplicates:
- Delete `core/agents/` (duplicate of een/agents/)
- Delete `een/agents/` (src/agents/ is more complete)