@echo off
cd /d "C:\Users\Nouri\Documents\GitHub\Een"

echo.
echo ==================================================
echo 🧘 φ-Resonance Care Mode Toggle (φ = 1.618033988749895)
echo ==================================================
echo.

REM Activate virtual environment
call "een\Scripts\activate.bat" >nul 2>&1

REM Show menu
echo [1] Activate CARE MODE (72h protective protocol)
echo [2] Deactivate CARE MODE (restore full operations)
echo [3] Show CARE MODE Status
echo [4] Safe word toggle (420691337 - bypasses care mode or activates)
echo [5] Update wellbeing data
echo [6] Generate Siri Shortcuts config
echo [0] Exit
echo.

set /p choice="Select option (0-6): "

if "%choice%"=="1" (
    echo.
    echo 🧘 Activating CARE MODE...
    python scripts\phi_care_mode_toggle.py --activate --reason "Windows batch activation"
    pause
) else if "%choice%"=="2" (
    echo.
    echo 🚀 Deactivating CARE MODE...
    python scripts\phi_care_mode_toggle.py --deactivate --reason "Windows batch deactivation"
    pause
) else if "%choice%"=="3" (
    echo.
    python scripts\phi_care_mode_toggle.py --status
    pause
) else if "%choice%"=="4" (
    echo.
    set /p code="Enter cheatcode: "
    python scripts\phi_care_mode_toggle.py --cheatcode "%code%"
    pause
) else if "%choice%"=="5" (
    echo.
    echo Update wellbeing data:
    set /p sleep="Sleep hours (current/enter to skip): "
    set /p thought="Thought speed 1-10 (current/enter to skip): "
    
    set cmd=python scripts\phi_care_mode_toggle.py
    if not "%sleep%"=="" set cmd=%cmd% --update-sleep %sleep%
    if not "%thought%"=="" set cmd=%cmd% --update-thought-speed %thought%
    
    %cmd%
    pause
) else if "%choice%"=="6" (
    echo.
    echo 📱 Generating Siri Shortcuts configuration...
    python scripts\phi_care_mode_toggle.py --create-shortcuts
    pause
) else if "%choice%"=="0" (
    echo.
    echo 🌟 Unity Mathematics: 1+1=1 through φ-harmonic consciousness
    echo Goodbye!
    exit /b
) else (
    echo.
    echo ❌ Invalid selection. Please try again.
    pause
)

REM Loop back to menu
goto :EOF