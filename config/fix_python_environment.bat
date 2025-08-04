@echo off
echo ========================================
echo FIXING CORRUPTED PYTHON ENVIRONMENT
echo ========================================
echo.

echo Step 1: Checking current Python installation...
python --version 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Python is corrupted or not found
    goto :install_python
)

echo Step 2: Attempting to fix pip...
python -m ensurepip --upgrade 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Cannot fix pip, need to reinstall Python
    goto :install_python
)

echo Step 3: Testing pip...
python -m pip --version 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Pip still not working, reinstalling Python
    goto :install_python
)

echo Python environment appears to be working now!
goto :setup_venv

:install_python
echo.
echo ========================================
echo REINSTALLING PYTHON 3.11
echo ========================================
echo.

echo Downloading Python 3.11 installer...
powershell -Command "& {Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile 'python-3.11.9-amd64.exe'}"

if not exist "python-3.11.9-amd64.exe" (
    echo ERROR: Failed to download Python installer
    echo Please download Python 3.11 manually from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Installing Python 3.11...
python-3.11.9-amd64.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0

echo Waiting for installation to complete...
timeout /t 30 /nobreak >nul

echo Cleaning up installer...
del python-3.11.9-amd64.exe

echo Refreshing PATH...
call refreshenv 2>nul
if %errorlevel% neq 0 (
    echo Refreshing environment variables...
    set PATH=%PATH%;C:\Python311;C:\Python311\Scripts
)

echo Testing new Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python installation failed
    pause
    exit /b 1
)

echo Testing pip...
python -m pip --version
if %errorlevel% neq 0 (
    echo ERROR: Pip not working after installation
    pause
    exit /b 1
)

:setup_venv
echo.
echo ========================================
echo SETTING UP VIRTUAL ENVIRONMENT
echo ========================================
echo.

echo Removing old virtual environment if it exists...
if exist "venv" (
    echo Removing old venv...
    rmdir /s /q venv
)

echo Creating new virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading pip in virtual environment...
python -m pip install --upgrade pip

echo Installing minimal requirements...
python -m pip install -r requirements_minimal.txt

echo Testing installation...
python -c "import numpy, scipy, matplotlib, flask; print('All core packages imported successfully!')"

echo.
echo ========================================
echo ENVIRONMENT FIXED SUCCESSFULLY!
echo ========================================
echo.
echo To activate the virtual environment in the future:
echo   venv\Scripts\activate.bat
echo.
echo To install additional packages:
echo   python -m pip install -r requirements.txt
echo.
pause 