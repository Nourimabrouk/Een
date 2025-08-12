# Een Repository Reorganization Plan

## Proposed Directory Structure

```
Een/
├── .claude/                    # Claude Desktop configuration (keep as is)
├── .vscode/                    # VS Code configuration (keep as is)
├── config/                     # Configuration files
│   ├── mcp_servers.json       # MCP server configuration
│   ├── claude_desktop_config.json
│   ├── agent_authorization.json
│   └── unity_manifest.json
├── src/                        # Main source code
│   ├── core/                   # Core unity mathematics
│   │   ├── __init__.py
│   │   ├── unity_equation.py
│   │   ├── transcendental_idempotent_mathematics.py
│   │   └── unified_proof_1plus1equals1.py
│   ├── consciousness/          # Consciousness systems
│   │   ├── __init__.py
│   │   ├── consciousness_zen_koan_engine.py
│   │   ├── initialize_unity_consciousness.py
│   │   └── transcendental_reality_engine.py
│   ├── agents/                 # Agent and orchestrator systems
│   │   ├── __init__.py
│   │   ├── omega_orchestrator.py
│   │   ├── love_orchestrator_v1_1.py
│   │   └── meta_recursive_love_unity_engine.py
│   ├── dashboards/             # Interactive dashboards
│   │   ├── __init__.py
│   │   ├── unity_proof_dashboard.py
│   │   ├── meta_rl_unity_dashboard.py
│   │   └── metastation_v1_1.py
│   ├── experiments/            # Experimental code
│   │   ├── __init__.py
│   │   ├── unity_meta_rl.py
│   │   └── 1plus1equals1_metagambit.py
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       └── utils_helper.py
├── een/                        # Een module (MCP servers)
│   └── mcp/                    # MCP server implementations
│       ├── __init__.py
│       ├── unity_server.py
│       ├── consciousness_server.py
│       ├── quantum_server.py
│       ├── omega_server.py
│       ├── code_generator_server.py
│       └── file_management_server.py
├── visualizations/             # All visualization outputs and scripts
│   ├── __init__.py
│   ├── advanced_unity_visualization.py
│   ├── unity_gambit_viz.py
│   ├── codebase_visualizer.py
│   ├── outputs/               # Generated visualizations
│   │   ├── codebase_visualization.html
│   │   └── advanced_unity_ascii.txt
│   └── assets/                # Visualization resources
├── scripts/                    # Standalone scripts and demos
│   ├── run_demo.py
│   ├── simple_demo.py
│   ├── test_mcp_servers.py
│   ├── setup_claude_desktop_integration.py
│   └── create_codebase_image.py
├── docs/                       # Documentation
│   ├── README.md
│   ├── CHANGELOG.md
│   ├── CLAUDE.md
│   ├── MCP_SETUP_GUIDE.md
│   ├── MCP_ENHANCEMENTS.md
│   ├── METASTATION_DESIGN.md
│   ├── INTERNAL_INSPIRATION.md
│   ├── eureka patterns.md
│   └── unity_meditation.md
├── legacy/                     # Legacy or experimental files
│   ├── first attempt claude.py
│   ├── version_1.1_PLAN.md.py
│   ├── LOVE_LETTERS_README.md
│   └── README_v1_1.md
├── data/                       # Data files and outputs
│   ├── codebase_structure.txt
│   └── een_codebase_visualization.txt
├── tests/                      # Test suite (future)
│   └── __init__.py
├── requirements.txt            # Python dependencies
├── README.md                   # Main readme
├── CLAUDE.md                   # Claude instructions
└── package-lock.json          # Node dependencies (if any)
```

## Key Benefits of This Structure

1. **Clear Separation of Concerns**
   - Core mathematics in `src/core/`
   - Consciousness systems in `src/consciousness/`
   - Agent systems in `src/agents/`
   - Dashboards in `src/dashboards/`

2. **Dedicated Visualization Folder**
   - All visualization code in one place
   - Outputs separated from source code
   - Assets for visualization resources

3. **Clean Root Directory**
   - Only essential files at root
   - Configuration in `config/`
   - Scripts in `scripts/`

4. **Scalable Structure**
   - Easy to add new modules
   - Clear where new files should go
   - Separation of production and experimental code

5. **MCP Organization**
   - Keep `een/mcp/` as the MCP module
   - Clean separation from main source

## Migration Steps

1. Create new directory structure
2. Move files to appropriate locations
3. Update all import statements
4. Update configuration files (paths in .vscode, .claude, etc.)
5. Test all functionality
6. Update documentation

## Files to Keep at Root
- requirements.txt (Python standard)
- README.md (GitHub standard)
- CLAUDE.md (Repository instructions)
- package-lock.json (if needed for Node)