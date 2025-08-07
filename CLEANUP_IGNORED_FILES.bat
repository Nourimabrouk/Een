@echo off
echo Cleaning up files that are now in .gitignore...
echo.
echo This will remove:
echo - Build artifacts (dist/)
echo - Node modules cache
echo - Temporary files
echo - Build maps and bundles
echo.
echo The following will be preserved:
echo - package.json (needed for Vite)
echo - Source code and configurations
echo - Documentation files
echo.
pause

cd /d "C:\Users\Nouri\Documents\GitHub\Een"

echo Removing Vite build outputs...
if exist "dist\" rmdir /s /q "dist\"

echo Removing temporary files...
del /q *.tmp 2>nul
del /q *.temp 2>nul
if exist "temp\" rmdir /s /q "temp\"
if exist "tmp\" rmdir /s /q "tmp\"

echo Removing CSS maps...
del /q *.css.map 2>nul

echo Removing bundle files...
del /q *.bundle.js 2>nul
del /q *.bundle.css 2>nul

echo Removing runtime files...
del /q *.pid 2>nul
del /q *.seed 2>nul
del /q *.pid.lock 2>nul

echo.
echo Cleanup complete! Run 'git status' to see the changes.
echo.
pause