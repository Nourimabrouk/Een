@echo off
echo =====================================
echo Een Virtual Environment Migration
echo Moving from in-repo venvs to Conda
echo =====================================

cd /d "C:\Users\Nouri\Documents\GitHub\Een"

echo.
echo [1/6] Creating external Conda environment...
conda create -n een python=3.11 -y

echo.
echo [2/6] Activating Conda environment...
call conda activate een

echo.
echo [3/6] Installing requirements...
pip install -r requirements.txt

echo.
echo [4/6] Adding virtual environments to .gitignore...
echo.>> .gitignore
echo # Virtual Environments>> .gitignore
echo een/>> .gitignore
echo venv/>> .gitignore
echo .venv/>> .gitignore
echo env/>> .gitignore
echo ENV/>> .gitignore

echo.
echo [5/6] Removing virtual environments from Git tracking...
git rm -r --cached een/ 2>nul
git rm -r --cached venv/ 2>nul

echo.
echo [6/6] Testing new environment...
python -c "import streamlit, numpy, pandas, plotly, scipy; print('SUCCESS: All core packages installed')"

echo.
echo =====================================
echo Migration Complete!
echo =====================================
echo To activate: conda activate een
echo To use in Cursor: Set Python interpreter to Conda een environment
echo You can now safely delete the een/ and venv/ folders
echo =====================================
pause