@echo off
REM =====================================================
REM Unity Mathematics Research Portal Launcher
REM Academic research infrastructure for 1+1=1 proofs
REM =====================================================

echo.
echo ============================================================
echo    UNITY MATHEMATICS RESEARCH PORTAL
echo    Academic Infrastructure for 1+1=1 Proofs
echo    Phi = 1.618033988749895
echo ============================================================
echo.

REM Check if virtual environment exists
if not exist "een\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run: python -m venv een
    pause
    exit /b 1
)

REM Activate virtual environment
echo [1/4] Activating virtual environment...
call een\Scripts\activate.bat

REM Start Unity Research API
echo [2/4] Starting Unity Research API Backend...
start "Unity Research API" cmd /k "call een\Scripts\activate.bat && python api\unity_research_api.py"
timeout /t 3 /nobreak >nul

REM Start website server
echo [3/4] Starting Website Server...
start "Unity Website" cmd /k "call een\Scripts\activate.bat && cd website && python -m http.server 8001"
timeout /t 2 /nobreak >nul

REM Open research portal in browser
echo [4/4] Opening Research Portal in Browser...
start http://localhost:8001/research-portal.html

echo.
echo ============================================================
echo    RESEARCH PORTAL LAUNCHED SUCCESSFULLY!
echo ============================================================
echo.
echo Access Points:
echo   - Research Portal: http://localhost:8001/research-portal.html
echo   - Unity API: http://localhost:8000/api/docs
echo   - Proof Explorer: http://localhost:8001/research-proof-explorer.html
echo   - Visualization Gallery: http://localhost:8001/visualization-gallery.html
echo   - Main Hub: http://localhost:8001/metastation-hub.html
echo.
echo Services Running:
echo   - Unity Research API (Port 8000)
echo   - Website Server (Port 8001)
echo.
echo Press Ctrl+C in each window to stop services
echo.
pause