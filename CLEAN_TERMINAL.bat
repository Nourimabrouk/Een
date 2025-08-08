@echo off

REM Clear any existing virtual environment
if defined VIRTUAL_ENV (
    echo Deactivating existing environment...
    set VIRTUAL_ENV=
    set VIRTUAL_ENV_PROMPT=
    set _OLD_VIRTUAL_PROMPT=
    set _OLD_VIRTUAL_PYTHONHOME=
    set _OLD_VIRTUAL_PATH=
    set PROMPT=$P$G
)

REM Change to repository directory
cd /d "%~dp0"

REM Activate clean environment
call een\Scripts\activate.bat

REM Show status
echo.
echo ========================================
echo  Een Unity Mathematics Terminal (Clean)
echo ========================================
echo Repository: %CD%
echo Virtual Environment: %VIRTUAL_ENV%
echo Python: %VIRTUAL_ENV%\Scripts\python.exe
echo.
echo Commands:
echo   python --version       - Test Python
echo   START_WEBSITE.bat      - Launch website
echo   python core/unity_mathematics.py - Test unity math
echo.

REM Start interactive session
cmd /k