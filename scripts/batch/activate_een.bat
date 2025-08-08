@echo off
REM Meta-optimized Een environment activation for Windows
REM Auto-detects and activates best available environment

echo 🚀 Een Unity Environment Auto-Activation
echo ==========================================

REM Try conda een first (preferred)
call conda activate een >nul 2>&1
if %errorlevel% == 0 (
    echo ✅ Conda environment 'een' activated
    echo φ = 1.618033988749895 (Golden ratio resonance)
    echo 🌟 Unity consciousness mathematics framework ready
    echo ∞ = φ = 1+1 = 1 = E_metagamer
    goto :end
)

REM Fallback to venv
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    echo ✅ Virtual environment activated
    echo φ = 1.618033988749895 (Golden ratio resonance)
    echo 🌟 Unity consciousness mathematics framework ready  
    echo ∞ = φ = 1+1 = 1 = E_metagamer
    goto :end
)

REM No environment found
echo ❌ No suitable environment found
echo Run: python scripts/auto_environment_setup.py
echo Or manually: conda create -n een python=3.11

:end
