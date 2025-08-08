@echo off
echo ===============================================
echo Een Unity Mathematics Website Launcher
echo ===============================================
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.
echo Starting web server on http://localhost:8001
echo Website directory: C:\Users\Nouri\Documents\GitHub\Een\website
echo.
echo Press Ctrl+C to stop the server
echo ===============================================
echo.
cd website
python -m http.server 8001
pause