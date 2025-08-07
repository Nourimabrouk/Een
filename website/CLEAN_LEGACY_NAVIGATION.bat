@echo off
echo ========================================
echo Een Unity Mathematics - Legacy Navigation Cleanup
echo ========================================
echo.
echo This script will safely archive legacy navigation files.
echo.

echo Creating legacy archive directories...
if not exist "legacy" mkdir legacy
if not exist "legacy\js" mkdir legacy\js
if not exist "legacy\css" mkdir legacy\css

echo.
echo Moving legacy JavaScript files to archive...
if exist "js\unified-navigation-system.js" (
    move "js\unified-navigation-system.js" "legacy\js\"
    echo   ✓ Archived unified-navigation-system.js
)
if exist "js\metastation-sidebar-navigation.js" (
    move "js\metastation-sidebar-navigation.js" "legacy\js\"
    echo   ✓ Archived metastation-sidebar-navigation.js
)
if exist "js\master-integration-system.js" (
    move "js\master-integration-system.js" "legacy\js\"
    echo   ✓ Archived master-integration-system.js
)
if exist "js\meta-optimal-integration.js" (
    move "js\meta-optimal-integration.js" "legacy\js\"
    echo   ✓ Archived meta-optimal-integration.js
)

echo.
echo Moving legacy CSS files to archive...
if exist "css\meta-optimal-navigation.css" (
    move "css\meta-optimal-navigation.css" "legacy\css\"
    echo   ✓ Archived meta-optimal-navigation.css
)

echo.
echo Creating legacy documentation...
echo # Legacy Navigation Files > legacy\README.md
echo. >> legacy\README.md
echo This folder contains archived navigation files from the Een Unity Mathematics website. >> legacy\README.md
echo These files have been replaced by the new complete navigation system. >> legacy\README.md
echo. >> legacy\README.md
echo ## Archived Date >> legacy\README.md
echo %date% >> legacy\README.md
echo. >> legacy\README.md
echo ## Replaced By >> legacy\README.md
echo - meta-optimal-navigation-complete.js >> legacy\README.md
echo - meta-optimal-navigation-complete.css >> legacy\README.md
echo. >> legacy\README.md
echo ## Files Archived >> legacy\README.md
echo - unified-navigation-system.js >> legacy\README.md
echo - metastation-sidebar-navigation.js >> legacy\README.md
echo - master-integration-system.js >> legacy\README.md
echo - meta-optimal-integration.js >> legacy\README.md
echo - meta-optimal-navigation.css >> legacy\README.md

echo.
echo ✅ Legacy cleanup complete!
echo.
echo Status:
echo   • Legacy files safely archived
echo   • New complete navigation system is now active
echo   • All conflicts resolved
echo.
echo Next steps:
echo   1. Test navigation on all pages
echo   2. Verify no broken links
echo   3. Check mobile responsiveness
echo.
pause