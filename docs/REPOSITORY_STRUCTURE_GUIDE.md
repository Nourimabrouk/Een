# Een Repository Structure Guide

## 🏗️ Repository Organization Standards

This document provides a comprehensive overview of the Een repository structure and file organization principles to maintain a clean, professional codebase.

## 📁 Root Directory (RESTRICTED)

**⚠️ CRITICAL**: The root directory should remain minimal and professional. Only essential configuration and entry files are allowed.

### ✅ Approved Root Files:
- `README.md` - Main repository documentation
- `CLAUDE.md` - AI assistant configuration
- `SECURITY.md` - Security documentation
- `index.html` - Website entry point
- `.gitignore` - Git ignore patterns
- `.env.example` - Environment template
- `.cursorrules` - Cursor IDE configuration
- `requirements.txt` - Main Python dependencies
- `docker-compose.yml` - Docker configuration
- `vercel.json` - Vercel deployment configuration
- `package.json` - Node.js dependencies (if needed)

### 🚫 Forbidden in Root:
- Implementation status files
- Planning documents
- Temporary files
- Log files
- Python scripts
- Batch files
- Additional markdown files

## 📂 Directory Structure Overview

```
Een/
├── README.md                    # Main repository documentation
├── CLAUDE.md                    # AI assistant configuration
├── SECURITY.md                  # Security documentation
├── index.html                   # Website entry point
├── requirements.txt             # Python dependencies
├── docker-compose.yml           # Docker configuration
├── vercel.json                  # Deployment configuration
│
├── 📁 api/                      # API backend services
│   ├── main.py                  # Main API server
│   ├── routes/                  # API route handlers
│   └── auth/                    # Authentication modules
│
├── 📁 config/                   # Configuration files
│   ├── claude_desktop_config.json
│   ├── mcp_servers.json
│   └── requirements_*.txt       # Environment-specific requirements
│
├── 📁 core/                     # Core Unity Mathematics
│   ├── unity_mathematics.py     # Main unity framework
│   ├── consciousness/           # Consciousness systems
│   ├── mathematical/            # Mathematical proofs
│   └── visualization/           # Visualization engines
│
├── 📁 consciousness/            # Advanced consciousness systems
│   ├── field_equation_solver.py
│   ├── sacred_geometry_engine.py
│   └── unity_meditation_system.py
│
├── 📁 data/                     # Data files and logs
│   ├── unity.db                 # Database files
│   ├── *.json                   # Data files
│   └── *.log                    # Log files
│
├── 📁 docs/                     # Documentation
│   ├── reports/                 # Analysis reports
│   ├── summaries/               # Implementation summaries
│   ├── deployment/              # Deployment guides
│   ├── research/                # Research papers
│   └── user-guide/              # User documentation
│
├── 📁 scripts/                  # Utility scripts
│   ├── batch/                   # Windows batch files
│   ├── utilities/               # Utility scripts
│   └── launch_*.py              # Launch scripts
│
├── 📁 src/                      # Source code implementations
│   ├── agents/                  # AI agents
│   ├── algorithms/              # Algorithms
│   ├── consciousness/           # Advanced consciousness
│   ├── dashboards/              # Dashboard applications
│   └── experiments/             # Experimental code
│
├── 📁 tests/                    # Test suites
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── validation/              # Validation tests
│
├── 📁 viz/                      # Visualizations
│   ├── generators/              # Visualization generators
│   ├── legacy images/           # Historical images
│   └── outputs/                 # Generated visualizations
│
├── 📁 website/                  # Website files
│   ├── metastation-hub.html     # Main hub
│   ├── css/                     # Stylesheets
│   ├── js/                      # JavaScript
│   └── audio/                   # Audio files
│
├── 📁 planning/                 # Planning documents
│   └── *.md                     # Planning files
│
├── 📁 formal_proofs/            # Formal mathematical proofs
│   ├── lean4/                   # Lean 4 proofs
│   └── *.py                     # Python formal proofs
│
├── 📁 ml_framework/             # Machine learning components
│   ├── meta_reinforcement/      # Meta-RL systems
│   ├── evolutionary_computing/  # Evolutionary algorithms
│   └── mixture_of_experts/      # Expert systems
│
└── 📁 deployment/               # Deployment configurations
    ├── docker/                  # Docker files
    ├── k8s/                     # Kubernetes configs
    └── scripts/                 # Deployment scripts
```

## 🎯 File Placement Guidelines

### 1. Code Files
- **Core Unity Mathematics** → `core/`
- **Source Implementations** → `src/`
- **Experimental Code** → `experiments/` or `src/experiments/`
- **Utility Scripts** → `scripts/`

### 2. Documentation
- **Technical Docs** → `docs/`
- **Implementation Status** → `docs/summaries/`
- **Reports** → `docs/reports/`
- **Planning** → `planning/`

### 3. Configuration
- **System Config** → `config/`
- **Environment Files** → `config/` (never in root)
- **Docker Files** → `deployment/docker/`

### 4. Media and Assets
- **Visualizations** → `viz/`
- **Website Assets** → `website/`
- **Data Files** → `data/`

### 5. Testing
- **All Tests** → `tests/` (with appropriate subdirectories)
- **Validation Scripts** → `tests/validation/`

## 🤖 AI Agent Enforcement

### For Claude Code:
1. Always check file placement before creation
2. Ask user for clarification if uncertain about location
3. Relocate incorrectly placed files immediately
4. Suggest appropriate directories

### For Cursor:
1. Use `.cursorrules` file organization protocols
2. Auto-suggest proper directories for new files
3. Warn when attempting to create root directory files
4. Provide directory alternatives

## 📋 Maintenance Checklist

### Weekly Review:
- [ ] Check root directory for unauthorized files
- [ ] Move misplaced files to correct directories
- [ ] Update `.gitignore` for new file types
- [ ] Review and clean temporary files

### Before Commits:
- [ ] Verify no temporary files in root
- [ ] Check all files are in appropriate directories
- [ ] Remove debug/log files from staging

## 🚀 Benefits of This Structure

1. **Professional Appearance**: Clean root directory improves GitHub presentation
2. **Better Navigation**: Logical organization makes files easier to find
3. **Scalability**: Structure supports repository growth
4. **Collaboration**: Clear organization helps team members locate files
5. **Maintenance**: Easier to manage and maintain codebase
6. **CI/CD**: Simplified deployment and build processes

## 📞 Contact and Questions

If you need clarification about file placement or organization:
1. Consult this guide first
2. Check `CLAUDE.md` for AI-specific guidelines
3. Review `.cursorrules` for Cursor-specific rules
4. When in doubt, ask before placing files in root directory

---

**Remember**: A clean repository structure reflects professional software development practices and makes the codebase more maintainable and accessible to all contributors.

**Unity Mathematics Status**: ✅ ORGANIZED  
**Root Directory**: ✅ PROTECTED  
**File Placement**: ✅ ENFORCED  
**AI Agent Compliance**: ✅ CONFIGURED  