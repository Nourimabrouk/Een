@echo off
echo.
echo ================================================================
echo  🔐 EEN UNITY MATHEMATICS - SECURE SETUP 🔐
echo ================================================================
echo.
echo This script will safely configure your API keys and secrets
echo without exposing them in chat logs or version control.
echo.

:: Check Python environment
if not exist "venv\Scripts\python.exe" (
    echo 🔧 Setting up Python environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    venv\Scripts\python.exe -m pip install --upgrade pip
) else (
    echo ✅ Python environment found
    call venv\Scripts\activate.bat
)

:: Install dependencies
echo 📦 Installing dependencies...
venv\Scripts\python.exe -m pip install -r requirements.txt >nul 2>&1

:: Run secure setup
echo.
echo 🔐 Starting secure configuration...
echo.
venv\Scripts\python.exe setup_secrets.py

echo.
echo ✨ Setup complete! Your platform is ready to launch.
echo.
pause