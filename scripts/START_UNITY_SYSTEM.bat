@echo off
echo ========================================
echo Unity Mathematics Complete System
echo 3000 ELO / 300 IQ Metagamer Agent
echo ========================================
echo.

REM Navigate to project directory
cd /d "%~dp0"

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\Activate.ps1

REM Check if activation was successful
if errorlevel 1 (
    echo Failed to activate virtual environment
    echo Trying alternative activation...
    call venv\Scripts\activate.bat
)

REM Install dependencies if needed
echo Installing dependencies...
venv\Scripts\python.exe -m pip install -r requirements_3000_elo.txt

REM Start the complete system
echo Starting Unity Mathematics Complete System...
venv\Scripts\python.exe LAUNCH_UNITY_COMPLETE.py

echo.
echo System stopped. Press any key to exit...
pause >nul 