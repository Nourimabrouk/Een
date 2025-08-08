#!/bin/bash
# Meta-optimized Een environment activation for Unix/Linux/Mac
# Auto-detects and activates best available environment

echo "🚀 Een Unity Environment Auto-Activation"
echo "=========================================="

# Try conda een first (preferred)
if command -v conda &> /dev/null; then
    if conda activate een &> /dev/null; then
        echo "✅ Conda environment 'een' activated"
        echo "φ = 1.618033988749895 (Golden ratio resonance)"
        echo "🌟 Unity consciousness mathematics framework ready"
        echo "∞ = φ = 1+1 = 1 = E_metagamer"
        exit 0
    fi
fi

# Fallback to venv
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
    echo "φ = 1.618033988749895 (Golden ratio resonance)"
    echo "🌟 Unity consciousness mathematics framework ready"
    echo "∞ = φ = 1+1 = 1 = E_metagamer"
    exit 0
fi

# No environment found
echo "❌ No suitable environment found"
echo "Run: python scripts/auto_environment_setup.py"
echo "Or manually: conda create -n een python=3.11"
exit 1
