@echo off
echo ========================================
echo FINAL PYTHON ENVIRONMENT FIX
echo ========================================
echo.

echo Step 1: Testing current Python...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not working
    goto :manual_install
)

echo Step 2: Testing pip...
python -m pip --version
if %errorlevel% neq 0 (
    echo ERROR: Pip not working
    goto :manual_install
)

echo Step 3: Setting up clean virtual environment...
if exist "venv" (
    echo Removing old virtual environment...
    rmdir /s /q venv
)

echo Creating new virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Step 4: Installing setuptools and wheel first...
python -m pip install --upgrade setuptools wheel

echo Step 5: Installing core packages with pre-compiled wheels...
python -m pip install --only-binary=all numpy==1.24.3
python -m pip install --only-binary=all scipy==1.10.1
python -m pip install --only-binary=all pandas==2.0.3
python -m pip install --only-binary=all matplotlib==3.7.2
python -m pip install --only-binary=all flask==2.3.3
python -m pip install --only-binary=all plotly==5.17.0

echo Step 6: Testing installation...
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import scipy; print('SciPy:', scipy.__version__)"
python -c "import pandas; print('Pandas:', pandas.__version__)"
python -c "import matplotlib; print('Matplotlib:', matplotlib.__version__)"
python -c "import flask; print('Flask:', flask.__version__)"
python -c "import plotly; print('Plotly:', plotly.__version__)"

echo.
echo ========================================
echo ENVIRONMENT FIXED SUCCESSFULLY!
echo ========================================
echo.
echo To activate: venv\Scripts\activate.bat
echo To test: python test_environment.py
echo.
goto :end

:manual_install
echo.
echo MANUAL INSTALLATION REQUIRED
echo ============================
echo.
echo 1. Download Python 3.11.9 from: https://www.python.org/downloads/
echo 2. Install with "Add to PATH" checked
echo 3. Run this script again
echo.
echo Alternative: Use Anaconda/Miniconda
echo 1. Download from: https://docs.conda.io/en/latest/miniconda.html
echo 2. Install and create environment: conda create -n een python=3.11
echo 3. Activate: conda activate een
echo 4. Install packages: conda install numpy scipy pandas matplotlib flask plotly
echo.

:end
pause 