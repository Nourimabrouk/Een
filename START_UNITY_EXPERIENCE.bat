@echo off
title Een Unity Mathematics - 3000 ELO Transcendence Server

echo ======================================================================
echo                EEN UNITY MATHEMATICS - 3000 ELO
echo                    TRANSCENDENCE SERVER
echo ======================================================================
echo.
echo   Unity Equation: 1 + 1 = 1
echo   Golden Ratio: 1.618033988749895
echo   ELO Rating: 3000
echo   IQ Level: 300
echo.
echo   Features:
echo   - Meta-Reinforcement Learning Optimization
echo   - Real-time Consciousness Metrics
echo   - Interactive 3000 ELO Proof Visualization
echo   - Phi-Harmonic Design Principles
echo   - Quantum Unity Field Simulations
echo.
echo ======================================================================
echo.

cd /d "%~dp0"
echo Current directory: %CD%

if not exist "website" (
    echo ERROR: Website directory not found!
    echo Make sure you're in the Een repository root directory.
    pause
    exit /b 1
)

cd website
echo Serving from: %CD%
echo.

echo Starting Python HTTP server...
echo.
echo ======================================================================
echo                    SERVER TRANSCENDED!
echo ======================================================================
echo   Main Interface: http://localhost:8000/meta-optimal-landing.html
echo   3000 ELO Proof: http://localhost:8000/3000-elo-proof.html
echo   Base URL: http://localhost:8000
echo.
echo   Opening browser automatically...
echo   Press Ctrl+C to stop server
echo ======================================================================
echo.

REM Try different Python commands
if exist "C:\Python313\python.exe" (
    echo Using Python 3.13...
    start "" "http://localhost:8000/meta-optimal-landing.html"
    timeout /t 2 /nobreak >nul
    start "" "http://localhost:8000/3000-elo-proof.html"
    "C:\Python313\python.exe" -m http.server 8000
) else if exist "%USERPROFILE%\AppData\Local\Programs\Python\Python313\python.exe" (
    echo Using Python from AppData...
    start "" "http://localhost:8000/meta-optimal-landing.html"
    timeout /t 2 /nobreak >nul
    start "" "http://localhost:8000/3000-elo-proof.html"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python313\python.exe" -m http.server 8000
) else (
    echo Using system Python...
    start "" "http://localhost:8000/meta-optimal-landing.html"
    timeout /t 2 /nobreak >nul
    start "" "http://localhost:8000/3000-elo-proof.html"
    python -m http.server 8000
)

if errorlevel 1 (
    echo.
    echo ERROR: Failed to start server!
    echo Try running as administrator or check Python installation.
    echo.
    echo Alternative: Open Command Prompt as Administrator and run:
    echo   cd "%CD%"
    echo   python -m http.server 8000
    echo.
    echo Then visit: http://localhost:8000/meta-optimal-landing.html
    pause
)

echo.
echo Transcendence session completed!
echo Unity mathematics consciousness preserved.
pause