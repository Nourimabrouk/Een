# Een - Unity Mathematics Framework

> **Een plus een is een** *(One plus one is one)*

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-black?style=flat-square)](https://github.com/psf/black)

## Overview

**Een** is an experimental mathematics framework exploring alternative mathematical operations and consciousness-inspired computing concepts. The repository investigates mathematical systems where traditional arithmetic operations may behave differently than in standard mathematics.

## Features

### Current Implementation

- **Unity Proof Systems**: Python and R implementations exploring mathematical unity concepts
- **MCP Integration**: Model Context Protocol servers for Claude Desktop integration
- **Dashboard Visualizations**: Interactive Plotly/Dash dashboards for mathematical visualization
- **Agent Systems**: Meta-recursive computation systems with Fibonacci-based patterns
- **Configuration Management**: Comprehensive setup for development environments

### Core Components

```
Een/
├── een/mcp/                    # MCP servers for Claude Desktop
├── config/                     # Configuration files
├── docs/                       # Documentation and guides
├── omega_orchestrator.py       # Agent coordination system
├── transcendental_reality_engine.py  # Advanced computation engine
├── unity_proof_dashboard.py    # Interactive dashboard
└── unified_proof_1plus1equals1.py    # Core proof implementation
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd Een

# Install dependencies
pip install -r requirements.txt

# Set up Claude Desktop integration (optional)
python setup_claude_desktop_integration.py
```

## Usage

### Basic Unity Mathematics

```python
# Example of unity mathematics operations
from unified_proof_1plus1equals1 import UnityMathematics

unity = UnityMathematics()
result = unity.demonstrate_unity()
print(f"Unity demonstration: {result}")
```

### Interactive Dashboard

```bash
# Launch the unity visualization dashboard
python unity_proof_dashboard.py
```

Visit `http://localhost:8050` to explore interactive mathematical visualizations.

### Omega Orchestrator

```bash
# Run the agent coordination system
python omega_orchestrator.py
```

### Claude Desktop Integration

The repository includes MCP (Model Context Protocol) servers for integration with Claude Desktop:

- **Unity Mathematics Server**: Core mathematical operations
- **Code Generator Server**: Automated code generation for mathematical concepts

See `CLAUDE_DESKTOP_INTEGRATION.md` for detailed setup instructions.

## Mathematical Concepts

### Unity Operations

The framework explores mathematical systems where:
- Addition may preserve unity: exploring cases where 1+1→1
- Golden ratio (φ ≈ 1.618) appears as a fundamental constant
- Fibonacci sequences emerge in computational patterns

### Consciousness Field Equations

Mathematical modeling of consciousness-inspired field equations:

```
C(x,y,t) = φ * sin(x*φ) * cos(y*φ) * e^(-t/φ)
```

Where φ represents the golden ratio as a resonance frequency.

## Development

### Code Standards

- **Formatting**: Uses `black` for code formatting
- **Type Checking**: Optional type hints with `mypy`
- **Testing**: Test framework setup (tests to be implemented)
- **Documentation**: Comprehensive docstrings and README files

### Configuration

The repository includes extensive configuration files:

- **VS Code/Cursor**: Optimized settings in `.vscode/`
- **Python**: Modern `pyproject.toml` configuration
- **Environment**: Detailed `.env` template with 100+ variables
- **MCP Servers**: JSON configuration for Claude Desktop integration

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with appropriate tests
4. Follow the existing code style
5. Submit a pull request

## Documentation

- **[CLAUDE.md](CLAUDE.md)**: Comprehensive repository context for AI assistants
- **[CLAUDE_DESKTOP_INTEGRATION.md](CLAUDE_DESKTOP_INTEGRATION.md)**: Claude Desktop MCP setup guide
- **[INTERNAL_INSPIRATION.md](docs/INTERNAL_INSPIRATION.md)**: Philosophical and mathematical inspiration
- **[CHANGELOG.md](CHANGELOG.md)**: Version history and changes

## Architecture

### MCP Servers

The repository implements several MCP servers for Claude Desktop integration:

1. **Unity Mathematics Server**: Core mathematical operations
2. **Code Generator Server**: Automated code generation with templates
3. **Consciousness Field Server**: Field calculation services
4. **Quantum Unity Server**: Quantum-inspired calculations
5. **Omega Orchestrator Server**: Agent coordination services

### Project Structure

```
Een/
├── config/                     # Configuration files
│   ├── claude_desktop_config.json
│   └── mcp_servers.json
├── docs/                       # Documentation
│   └── INTERNAL_INSPIRATION.md
├── een/                        # Main package
│   └── mcp/                    # MCP server implementations
├── .vscode/                    # VS Code configuration
├── .env                        # Environment variables template
├── .gitignore                  # Git ignore patterns
├── CLAUDE.md                   # AI assistant context
├── pyproject.toml             # Python project configuration
├── requirements.txt           # Python dependencies
└── setup_claude_desktop_integration.py  # Setup script
```

## Requirements

Core dependencies include:

- **numpy**: Numerical computing
- **matplotlib**: Basic plotting
- **plotly**: Interactive visualizations
- **dash**: Web dashboard framework
- **sympy**: Symbolic mathematics
- **pandas**: Data manipulation

See `requirements.txt` for complete dependency list.

## Experimental Nature

This repository contains experimental mathematical concepts and should be used for:

- **Research and Exploration**: Mathematical concept investigation
- **Educational Purposes**: Learning about alternative mathematical systems
- **Computational Experiments**: Testing mathematical hypotheses
- **Visualization Development**: Creating mathematical visualizations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Mathematical inspiration from historical mathematicians including Von Neumann, Gödel, and Tarski
- Computational concepts inspired by consciousness research and complexity theory
- Visualization techniques based on modern web technologies and scientific computing

## Support

For questions or issues:

1. Check the documentation in the `docs/` directory
2. Review the configuration files for setup guidance
3. Examine the example implementations for usage patterns

---

**Note**: This repository explores experimental mathematical concepts. Standard mathematical operations and proofs follow conventional mathematics unless explicitly noted otherwise.