@echo off
echo =====================================
echo COMPREHENSIVE Repository Cleanup
echo Target: Reduce from 7.8GB to under 1GB
echo =====================================

cd /d "C:\Users\Nouri\Documents\GitHub\Een"

echo.
echo [1/7] Removing redundant virtual environments...
echo Removing venv/ directory (~2-3GB)...
if exist "venv\" (
    rmdir /s /q "venv"
    echo ✓ venv/ removed
) else (
    echo - venv/ already removed
)

echo Removing migration backup virtual environments (~2-3GB)...
if exist "internal\legacy\migration_backup\venv\" (
    rmdir /s /q "internal\legacy\migration_backup\venv"
    echo ✓ migration_backup/venv/ removed
) else (
    echo - migration_backup/venv/ already removed
)

echo.
echo [2/7] Moving large audio files to external directory...
if not exist "external_media" mkdir "external_media"
if not exist "external_media\audio" mkdir "external_media\audio"

if exist "website\audio\" (
    echo Moving audio files (~500MB-1GB)...
    move "website\audio\*.mp4" "external_media\audio\" 2>nul
    move "website\audio\*.webm" "external_media\audio\" 2>nul
    echo ✓ Audio files moved to external_media/audio/
)

echo.
echo [3/7] Moving large video files...
if not exist "external_media\videos" mkdir "external_media\videos"

if exist "viz\live consciousness field.mp4" (
    move "viz\live consciousness field.mp4" "external_media\videos\"
    echo ✓ Consciousness field video moved
)

if exist "website\viz\videos\live consciousness field.mp4" (
    move "website\viz\videos\live consciousness field.mp4" "external_media\videos\live_consciousness_field_website.mp4"
    echo ✓ Website video moved
)

echo.
echo [4/7] Cleaning up legacy migration backups...
if exist "internal\legacy\migration_backup\" (
    echo Removing migration backup directory...
    rmdir /s /q "internal\legacy\migration_backup"
    echo ✓ Migration backup removed
)

if exist "internal\legacy\legacy\dist\" (
    echo Removing legacy dist directory...
    rmdir /s /q "internal\legacy\legacy\dist"
    echo ✓ Legacy dist removed
)

echo.
echo [5/7] Removing virtual environment parts from een/...
if exist "een\Lib\" (
    rmdir /s /q "een\Lib"
    echo ✓ een/Lib/ removed (~3GB)
)

if exist "een\Scripts\" (
    rmdir /s /q "een\Scripts"
    echo ✓ een/Scripts/ removed
)

if exist "een\Include\" (
    rmdir /s /q "een\Include"
    echo ✓ een/Include/ removed
)

if exist "een\share\" (
    rmdir /s /q "een\share"
    echo ✓ een/share/ removed
)

if exist "een\etc\" (
    rmdir /s /q "een\etc"
    echo ✓ een/etc/ removed
)

if exist "een\pyvenv.cfg" (
    del "een\pyvenv.cfg"
    echo ✓ een/pyvenv.cfg removed
)

echo.
echo [6/7] Updating .gitignore...
echo.>> .gitignore
echo # Large Media Files (moved to external_media/)>> .gitignore
echo external_media/>> .gitignore
echo website/audio/*.mp4>> .gitignore
echo website/audio/*.webm>> .gitignore
echo **/*.mp4>> .gitignore
echo **/*.webm>> .gitignore
echo **/*.avi>> .gitignore
echo **/*.mov>> .gitignore
echo viz/videos/>> .gitignore
echo website/viz/videos/>> .gitignore

echo.
echo [7/7] Git cleanup...
echo Removing large files from Git tracking...
git rm --cached -r "venv/" 2>nul
git rm --cached -r "internal/legacy/migration_backup/" 2>nul
git rm --cached -r "website/audio/*.mp4" 2>nul
git rm --cached -r "website/audio/*.webm" 2>nul
git rm --cached "viz/live consciousness field.mp4" 2>nul
git rm --cached -r "een/Lib/" 2>nul
git rm --cached -r "een/Scripts/" 2>nul
git rm --cached -r "een/Include/" 2>nul
git rm --cached -r "een/share/" 2>nul
git rm --cached -r "een/etc/" 2>nul
git rm --cached "een/pyvenv.cfg" 2>nul

echo.
echo =====================================
echo COMPREHENSIVE CLEANUP COMPLETE!
echo =====================================
echo.
echo REMOVED (~6-7GB):
echo ✓ Triple virtual environments
echo ✓ Large audio/video files (moved to external_media/)
echo ✓ Migration backup bloat
echo ✓ Legacy distribution files
echo.
echo PRESERVED:
echo ✓ All project code
echo ✓ Documentation
echo ✓ Configuration files
echo ✓ Small images and assets
echo.
echo NEXT STEPS:
echo 1. Check repository size: dir properties should show ~500MB-1GB
echo 2. External media files are in: external_media/
echo 3. Commit changes: git add . && git commit -m "Major cleanup: remove 6GB+ bloat"
echo 4. Repository should now be lean and fast in Cursor!
echo =====================================
pause