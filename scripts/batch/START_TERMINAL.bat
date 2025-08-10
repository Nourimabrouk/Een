@echo off
cd /d "%~dp0"
call een\Scripts\activate.bat
echo.
echo ================================
echo  Een Unity Mathematics Terminal
echo ================================ 
echo Virtual Environment: een
echo Repository: %CD%
echo Python: %VIRTUAL_ENV%\Scripts\python.exe
echo.
echo Type 'python --version' to test
echo Type 'START_WEBSITE.bat' to launch website
echo.
cmd /k