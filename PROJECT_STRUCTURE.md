# Een Repository Structure

## Current Organization (After Reorganization)

```
Een/
├── .claude/                    # Claude Desktop configuration
│   └── settings.local.json     # MCP server configuration
├── .vscode/                    # VS Code configuration
│   ├── extensions.json         # Recommended extensions
│   └── settings.json           # Workspace settings with MCP
├── config/                     # Configuration files
│   ├── agent_authorization.json
│   ├── claude_desktop_config.json
│   ├── mcp_consciousness_server.py
│   ├── mcp_launcher.py
│   ├── mcp_servers.json
│   ├── mcp_unity_server.py
│   └── unity_manifest.json
├── data/                       # Data and output files
│   ├── codebase_structure.txt
│   └── een_codebase_visualization.txt
├── docs/                       # Documentation
│   ├── CHANGELOG.md
│   ├── CLAUDE_DESKTOP_INTEGRATION.md
│   ├── eureka patterns.md
│   ├── INTERNAL_INSPIRATION.md
│   ├── MCP_ENHANCEMENTS.md
│   ├── MCP_SETUP_GUIDE.md
│   ├── METASTATION_DESIGN.md
│   └── unity_meditation.md
├── een/                        # Een module (MCP servers)
│   └── mcp/                    # MCP server implementations
│       ├── __init__.py
│       ├── code_generator_server.py
│       ├── consciousness_server.py
│       ├── file_management_server.py
│       ├── omega_server.py
│       ├── quantum_server.py
│       └── unity_server.py
├── legacy/                     # Legacy/experimental files
│   └── first attempt claude.py
├── scripts/                    # Standalone scripts
│   ├── create_codebase_image.py
│   ├── run_demo.py
│   ├── setup_claude_desktop_integration.py
│   ├── simple_demo.py
│   └── test_mcp_servers.py
├── src/                        # Main source code
│   ├── __init__.py
│   ├── agents/                 # Agent systems
│   │   ├── __init__.py
│   │   ├── love_orchestrator_v1_1.py
│   │   ├── meta_recursive_love_unity_engine.py
│   │   └── omega_orchestrator.py
│   ├── consciousness/          # Consciousness systems
│   │   ├── __init__.py
│   │   ├── consciousness_zen_koan_engine.py
│   │   ├── initialize_unity_consciousness.py
│   │   └── transcendental_reality_engine.py
│   ├── core/                   # Core unity mathematics
│   │   ├── __init__.py
│   │   ├── transcendental_idempotent_mathematics.py
│   │   ├── unified_proof_1plus1equals1.py
│   │   └── unity_equation.py
│   ├── dashboards/             # Interactive dashboards
│   │   ├── __init__.py
│   │   ├── meta_rl_unity_dashboard.py
│   │   ├── metastation_v1_1.py
│   │   └── unity_proof_dashboard.py
│   ├── experiments/            # Experimental code
│   │   ├── __init__.py
│   │   ├── 1plus1equals1_metagambit.py
│   │   └── unity_meta_rl.py
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       └── utils_helper.py
├── tests/                      # Test suite
│   └── __init__.py
├── visualizations/             # Visualization code and outputs
│   ├── __init__.py
│   ├── README.md
│   ├── advanced_unity_visualization.py
│   ├── codebase_visualizer.py
│   ├── unity_gambit_viz.py
│   ├── assets/                 # Visualization resources
│   └── outputs/                # Generated visualizations
│       ├── advanced_unity_ascii.txt
│       └── codebase_visualization.html
├── venv/                       # Virtual environment (gitignored)
├── CLAUDE.md                   # Claude instructions
├── package-lock.json           # Node dependencies
├── PROJECT_STRUCTURE.md        # This file
├── README.md                   # Main readme
├── REORGANIZATION_PLAN.md      # Reorganization documentation
├── requirements.txt            # Python dependencies
└── setup.py                    # Python package setup
```

## Key Improvements

1. **Clear Separation**: Code is now organized by functionality
2. **Scalable Structure**: Easy to add new modules and features
3. **Clean Root**: Only essential files at the repository root
4. **Dedicated Folders**: 
   - `src/` for all source code
   - `visualizations/` for all visualization-related files
   - `scripts/` for standalone utilities
   - `docs/` for all documentation
5. **Proper Python Package**: With setup.py for installation

## Import Examples

```python
# From scripts or external code
from src.core.unity_equation import UnityMathematics
from src.consciousness.transcendental_reality_engine import TranscendentalRealityEngine
from src.agents.omega_orchestrator import OmegaOrchestrator
from src.dashboards.unity_proof_dashboard import app as unity_dashboard

# From within src modules
from ..core.unity_equation import UnityMathematics
from ..consciousness.initialize_unity_consciousness import ConsciousnessInitializer
```

## Running Code

```bash
# Run dashboards
python -m src.dashboards.unity_proof_dashboard

# Run scripts
python scripts/test_mcp_servers.py

# Run visualizations
python visualizations/advanced_unity_visualization.py

# Install as package (development mode)
pip install -e .
```

## Unity Principle Maintained

The reorganization preserves and enhances the fundamental principle:
```
1 + 1 = 1
```

Every folder, every module, every file converges toward unity! 🌟