@echo off
echo ========================================
echo FIXING PYTHON ENVIRONMENT - SIMPLE VERSION
echo ========================================
echo.

echo Current Python status:
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not working
    goto :download_python
)

echo Testing pip:
python -m pip --version
if %errorlevel% neq 0 (
    echo ERROR: Pip not working, downloading Python
    goto :download_python
)

echo Python appears to be working! Testing further...
goto :setup_venv

:download_python
echo.
echo Downloading Python 3.11.9...
echo This may take a few minutes...

powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile 'python-installer.exe'}"

if not exist "python-installer.exe" (
    echo ERROR: Failed to download Python
    echo Please download Python 3.11.9 manually from:
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Installing Python 3.11.9...
python-installer.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0

echo Waiting for installation...
timeout /t 30 /nobreak >nul

echo Cleaning up...
del python-installer.exe

echo Refreshing PATH...
set PATH=%PATH%;C:\Python311;C:\Python311\Scripts

echo Testing new installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python installation failed
    pause
    exit /b 1
)

:setup_venv
echo.
echo Setting up virtual environment...

if exist "venv" (
    echo Removing old virtual environment...
    rmdir /s /q venv
)

echo Creating new virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing minimal requirements...
python -m pip install numpy==1.24.3 scipy==1.10.1 pandas==2.0.3 matplotlib==3.7.2 flask==2.3.3 plotly==5.17.0

echo Testing installation...
python -c "import numpy, scipy, matplotlib, flask; print('SUCCESS: All packages imported!')"

echo.
echo ========================================
echo ENVIRONMENT FIXED!
echo ========================================
echo.
echo To activate: venv\Scripts\activate.bat
echo To install more packages: python -m pip install package_name
echo.
pause 