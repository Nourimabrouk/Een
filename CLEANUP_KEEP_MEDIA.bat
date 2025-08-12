@echo off
echo =====================================
echo Repository Cleanup - Keep Media Files
echo Target: Remove ~6GB virtual env bloat
echo Keep: Audio/video for GitHub Pages
echo =====================================

cd /d "C:\Users\Nouri\Documents\GitHub\Een"

echo.
echo [1/6] Removing redundant venv/ directory...
if exist "venv\" (
    echo Removing venv/ (~2-3GB)...
    rmdir /s /q "venv"
    echo ✓ venv/ removed
) else (
    echo - venv/ already removed
)

echo.
echo [2/6] Removing migration backup virtual environments...
if exist "internal\legacy\migration_backup\venv\" (
    echo Removing migration_backup/venv/ (~2-3GB)...
    rmdir /s /q "internal\legacy\migration_backup\venv"
    echo ✓ migration_backup/venv/ removed
) else (
    echo - migration_backup/venv/ already removed
)

echo.
echo [3/6] Removing virtual environment parts from een/...
if exist "een\Lib\" (
    echo Removing een/Lib/ (~3GB)...
    rmdir /s /q "een\Lib"
    echo ✓ een/Lib/ removed
) else (
    echo - een/Lib/ already removed
)

if exist "een\Scripts\" (
    rmdir /s /q "een\Scripts"
    echo ✓ een/Scripts/ removed
) else (
    echo - een/Scripts/ already removed
)

if exist "een\Include\" (
    rmdir /s /q "een\Include"
    echo ✓ een/Include/ removed
) else (
    echo - een/Include/ already removed
)

if exist "een\share\" (
    rmdir /s /q "een\share"
    echo ✓ een/share/ removed
) else (
    echo - een/share/ already removed
)

if exist "een\etc\" (
    rmdir /s /q "een\etc"
    echo ✓ een/etc/ removed
) else (
    echo - een/etc/ already removed
)

if exist "een\pyvenv.cfg" (
    del "een\pyvenv.cfg"
    echo ✓ een/pyvenv.cfg removed
) else (
    echo - een/pyvenv.cfg already removed
)

echo.
echo [4/6] Cleaning up other migration backup bloat...
if exist "internal\legacy\migration_backup\" (
    echo Removing remaining migration backup directory...
    rmdir /s /q "internal\legacy\migration_backup"
    echo ✓ Migration backup removed
) else (
    echo - Migration backup already removed
)

if exist "internal\legacy\legacy\dist\" (
    echo Removing legacy dist directory...
    rmdir /s /q "internal\legacy\legacy\dist"
    echo ✓ Legacy dist removed
) else (
    echo - Legacy dist already removed
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
echo REMOVED (~5-6GB):
echo ✓ Triple virtual environments (venv/, een/Lib/, migration backups)
echo ✓ Legacy build artifacts
echo ✓ Migration backup bloat
echo.
echo KEPT FOR GITHUB PAGES:
echo ✓ website/audio/ (music files for website)
echo ✓ viz/ videos and images
echo ✓ All project code and documentation
echo.
echo EXTERNAL CONDA ENVIRONMENT:
echo Location: C:\Users\Nouri\miniconda3\envs\een\
echo Activate: conda activate een
echo.
echo =====================================
echo CLEANUP COMPLETE - MEDIA PRESERVED!
echo =====================================
echo Repository should now be ~2-3GB (down from 7.8GB)
echo Your website media files are preserved for GitHub Pages
echo Virtual environment is now external and clean
echo =====================================
pause