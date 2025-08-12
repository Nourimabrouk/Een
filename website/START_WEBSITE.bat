@echo off
echo Starting Een Unity Mathematics Website...
echo Activating Conda environment 'een'...
echo.
echo Website will be available at: http://localhost:8001
echo.

REM Activate Conda environment and start server
call conda activate een
python -m http.server 8001

pause