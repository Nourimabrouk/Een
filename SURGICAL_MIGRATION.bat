@echo off
echo =====================================
echo SURGICAL Virtual Environment Migration
echo Preserving project code, removing venv
echo =====================================

cd /d "C:\Users\Nouri\Documents\GitHub\Een"

echo.
echo [1/8] Creating external Conda environment...
conda create -n een python=3.11 -y

echo.
echo [2/8] Activating Conda environment...
call conda activate een

echo.
echo [3/8] Installing requirements...
pip install -r requirements.txt

echo.
echo [4/8] Moving project code from een/ to src/
echo Creating backup directory...
mkdir een_project_backup 2>nul

echo Backing up project code...
xcopy /E /I "een\agents" "een_project_backup\agents"
xcopy /E /I "een\consciousness" "een_project_backup\consciousness" 2>nul
xcopy /E /I "een\core" "een_project_backup\core" 2>nul
xcopy /E /I "een\dashboards" "een_project_backup\dashboards"
xcopy /E /I "een\experiments" "een_project_backup\experiments"
xcopy /E /I "een\mcp" "een_project_backup\mcp"
xcopy /E /I "een\proofs" "een_project_backup\proofs"
xcopy /E /I "een\utils" "een_project_backup\utils"
copy "een\__init__.py" "een_project_backup\" 2>nul

echo Moving project code to src/ hierarchy...
mkdir "src\een" 2>nul
xcopy /E /I "een_project_backup\*" "src\een\"

echo.
echo [5/8] Adding virtual environments to .gitignore...
echo.>> .gitignore
echo # Virtual Environment Files Only>> .gitignore
echo een/Lib/>> .gitignore
echo een/Scripts/>> .gitignore
echo een/Include/>> .gitignore
echo een/share/>> .gitignore
echo een/etc/>> .gitignore
echo een/pyvenv.cfg>> .gitignore
echo venv/>> .gitignore

echo.
echo [6/8] Removing virtual environment files from Git tracking...
git rm -r --cached een/Lib/ 2>nul
git rm -r --cached een/Scripts/ 2>nul
git rm -r --cached een/Include/ 2>nul
git rm -r --cached een/share/ 2>nul
git rm -r --cached een/etc/ 2>nul
git rm --cached een/pyvenv.cfg 2>nul
git rm -r --cached venv/ 2>nul

echo.
echo [7/8] Testing new environment...
python -c "import streamlit, numpy, pandas, plotly, scipy; print('SUCCESS: All core packages installed')"

echo.
echo [8/8] Summary of changes...
echo.
echo PROJECT CODE PRESERVED in src/een/:
dir /B "src\een"
echo.
echo VIRTUAL ENVIRONMENT: Now external Conda 'een'
echo BACKED UP PROJECT CODE: een_project_backup/ (safe to delete after verification)

echo.
echo =====================================
echo SURGICAL Migration Complete!
echo =====================================
echo Your project code is safe in src/een/
echo Virtual environment is now external
echo You can now manually delete the large folders:
echo   rmdir /s /q "een\Lib"
echo   rmdir /s /q "een\Scripts" 
echo   rmdir /s /q "een\Include"
echo   rmdir /s /q "een\share"
echo   rmdir /s /q "een\etc"
echo   del "een\pyvenv.cfg"
echo   rmdir /s /q "venv"
echo =====================================
pause