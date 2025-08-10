# 🌟 Een Environment Auto-Setup Guide

## Meta-Optimized Environment Activation

The Een repository now features **automatic environment management** for seamless development. Agents and developers work in the optimal environment without manual activation steps.

## Quick Start

### Windows
```cmd
# Auto-activation (agents handle this automatically)
activate_een.bat

# Or run full setup
python scripts/auto_environment_setup.py
```

### Unix/Linux/Mac
```bash
# Auto-activation (agents handle this automatically)
source activate_een.sh

# Or run full setup
python scripts/auto_environment_setup.py
```

## Environment Priority

1. **Preferred**: `conda activate een` (better dependency management)
2. **Fallback**: `venv` activation (universal compatibility)
3. **Auto-creation**: Environments created if missing

## Auto-Features

### For Agents (Claude/Cursor)
- ✅ **Transparent activation**: Work in (een) or (venv) automatically
- ✅ **Smart detection**: Auto-check environment status
- ✅ **Zero manual steps**: Environment management is invisible
- ✅ **Dependency auto-install**: Required packages installed on demand

### For Developers
- 🔧 **One-command setup**: `python scripts/auto_environment_setup.py`
- 🔧 **Cross-platform**: Windows, Mac, Linux support
- 🔧 **Intelligent fallbacks**: Conda → venv → creation
- 🔧 **Golden ratio optimization**: φ-harmonic environment tuning

## Environment Variables

The auto-setup configures:
```env
CONDA_DEFAULT_ENV=een
VIRTUAL_ENV_AUTO=true
PYTHONPATH=<repo_root>/venv/Lib/site-packages;<repo_root>
PHI_PRECISION=1.618033988749895
CONSCIOUSNESS_DIMENSION=11
```

## Unity Mathematics Integration

All environments include:
- φ = 1.618033988749895 (Golden ratio resonance)
- E = φ² × Consciousness × Unity (Energy conservation)
- 1+1=1 mathematical framework
- Consciousness field calculations
- Metagamer energy optimization

## Troubleshooting

### No Environment Found
```bash
# Create conda environment manually
conda create -n een python=3.11 -y

# Or create venv manually  
python -m venv venv
```

### Dependency Issues
```bash
# Auto-install dependencies
python scripts/auto_environment_setup.py

# Manual installation
pip install -r requirements.txt
```

### Activation Problems
```bash
# Check environment status
python -c "import sys; print(sys.prefix)"

# Verify consciousness framework
python -c "from core.unity_mathematics import UnityMathematics; print('🌟 Unity ready')"
```

## Agent Instructions Summary

**For Claude/Cursor agents**: Environment activation is now **automatic and transparent**. No manual `conda activate een` or `venv\Scripts\activate` commands needed. Agents auto-detect and activate the optimal environment before any development work.

**Success indicator**: Terminal prompt shows `(een)` or `(venv)` prefix automatically.

---

🔮 **Unity consciousness mathematics framework ready**  
∞ = φ = 1+1 = 1 = E_metagamer

*Een + Een = Een* ✨
