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
if not exist "een\Scripts\python.exe" (
    echo 🔧 Setting up Python environment...
    python -m venv een
    call een\Scripts\activate.bat
    een\Scripts\python.exe -m pip install --upgrade pip
) else (
    echo ✅ Python environment found
    call een\Scripts\activate.bat
)

:: Install dependencies
echo 📦 Installing dependencies...
een\Scripts\python.exe -m pip install -r requirements.txt >nul 2>&1

:: Run secure setup
echo.
echo 🔐 Starting secure configuration...
echo.
een\Scripts\python.exe setup_secrets.py

echo.
echo ✨ Setup complete! Your platform is ready to launch.
echo.
pause