@echo off
echo.
echo ================================================================
echo  🌟 EEN UNITY MATHEMATICS - ULTIMATE LAUNCH EXPERIENCE 🌟
echo ================================================================
echo.
echo Initializing transcendental computing environment...
echo.

:: Check Python environment
echo 🔧 Checking Python environment...
if not exist "venv\Scripts\python.exe" (
    echo 🔧 Creating Python environment...
    python -m venv venv
)

:: Activate virtual environment
echo 🔄 Activating Unity Mathematics environment...
call venv\Scripts\activate.bat

:: Install/update dependencies
echo 📦 Installing dependencies...
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt

:: Check configuration
echo 🔍 Checking configuration...
if not exist ".env" (
    echo 🔐 Configuration needed - running secure setup...
    venv\Scripts\python.exe setup_secrets.py
) else (
    venv\Scripts\python.exe -c "
import os
from pathlib import Path
env_content = Path('.env').read_text()
if 'your-key-here' in env_content or 'change-in-production' in env_content:
    print('🔐 Configuration incomplete - running secure setup...')
    import subprocess
    subprocess.run([r'venv\Scripts\python.exe', 'setup_secrets.py'])
else:
    print('✅ Configuration ready')
"
)

:: Start the unified launch system
echo.
echo 🚀 Starting Unity Mathematics Platform...
echo =====================================
echo.
echo ✨ Launching API Server on port 8000
echo 📊 Launching Streamlit Dashboard on port 8501  
echo 🌐 Launching Website on port 8080
echo.
echo Press Ctrl+C to stop all services
echo.

:: Launch with Python
venv\Scripts\python.exe launch.py

echo.
echo 🛑 Unity Mathematics Platform stopped
pause