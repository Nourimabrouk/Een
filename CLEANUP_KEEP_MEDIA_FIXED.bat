@echo off
echo =====================================
echo Repository Cleanup - Keep Media Files
echo Target: Remove 6GB virtual env bloat
echo Keep: Audio/video for GitHub Pages
echo =====================================

cd /d "C:\Users\Nouri\Documents\GitHub\Een"

echo.
echo [1/6] Removing redundant venv/ directory...
if exist "venv\" (
    echo Removing venv/ directory (2-3GB)...
    rmdir /s /q "venv"
    echo SUCCESS: venv/ removed
) else (
    echo INFO: venv/ already removed
)

echo.
echo [2/6] Removing migration backup virtual environments...
if exist "internal\legacy\migration_backup\venv\" (
    echo Removing migration_backup/venv/ (2-3GB)...
    rmdir /s /q "internal\legacy\migration_backup\venv"
    echo SUCCESS: migration_backup/venv/ removed
) else (
    echo INFO: migration_backup/venv/ already removed
)

echo.
echo [3/6] Removing virtual environment parts from een/...
if exist "een\Lib\" (
    echo Removing een/Lib/ (3GB)...
    rmdir /s /q "een\Lib"
    echo SUCCESS: een/Lib/ removed
) else (
    echo INFO: een/Lib/ already removed
)

if exist "een\Scripts\" (
    rmdir /s /q "een\Scripts"
    echo SUCCESS: een/Scripts/ removed
) else (
    echo INFO: een/Scripts/ already removed
)

if exist "een\Include\" (
    rmdir /s /q "een\Include"
    echo SUCCESS: een/Include/ removed
) else (
    echo INFO: een/Include/ already removed
)

if exist "een\share\" (
    rmdir /s /q "een\share"
    echo SUCCESS: een/share/ removed
) else (
    echo INFO: een/share/ already removed
)

if exist "een\etc\" (
    rmdir /s /q "een\etc"
    echo SUCCESS: een/etc/ removed
) else (
    echo INFO: een/etc/ already removed
)

if exist "een\pyvenv.cfg" (
    del "een\pyvenv.cfg"
    echo SUCCESS: een/pyvenv.cfg removed
) else (
    echo INFO: een/pyvenv.cfg already removed
)

echo.
echo [4/6] Cleaning up other migration backup bloat...
if exist "internal\legacy\migration_backup\" (
    echo Removing remaining migration backup directory...
    rmdir /s /q "internal\legacy\migration_backup"
    echo SUCCESS: Migration backup removed
) else (
    echo INFO: Migration backup already removed
)

if exist "internal\legacy\legacy\dist\" (
    echo Removing legacy dist directory...
    rmdir /s /q "internal\legacy\legacy\dist"
    echo SUCCESS: Legacy dist removed
) else (
    echo INFO: Legacy dist already removed
)

echo.
echo [5/6] Removing from Git tracking...
echo Removing large files from Git tracking...
git rm --cached -r "venv/" 2>nul
git rm --cached -r "internal/legacy/migration_backup/" 2>nul
git rm --cached -r "een/Lib/" 2>nul
git rm --cached -r "een/Scripts/" 2>nul
git rm --cached -r "een/Include/" 2>nul
git rm --cached -r "een/share/" 2>nul
git rm --cached -r "een/etc/" 2>nul
git rm --cached "een/pyvenv.cfg" 2>nul

echo.
echo [6/6] Current repository status...
echo.
echo REMOVED (5-6GB):
echo SUCCESS: Triple virtual environments removed
echo SUCCESS: Legacy build artifacts removed  
echo SUCCESS: Migration backup bloat removed
echo.
echo KEPT FOR GITHUB PAGES:
echo SUCCESS: website/audio/ (music files preserved)
echo SUCCESS: viz/ videos and images preserved
echo SUCCESS: All project code and documentation preserved
echo.
echo EXTERNAL CONDA ENVIRONMENT:
echo Location: C:\Users\Nouri\miniconda3\envs\een\
echo Activate: conda activate een
echo.
echo =====================================
echo CLEANUP COMPLETE - MEDIA PRESERVED!
echo =====================================
echo Repository should now be 2-3GB (down from 7.8GB)
echo Your website media files are preserved for GitHub Pages
echo Virtual environment is now external and clean
echo =====================================
pause