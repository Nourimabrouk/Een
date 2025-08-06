@echo off
echo.
echo 🌌 EEN UNITY MATHEMATICS MIGRATION
echo Where 1+1=1 through architectural transcendence
echo.

:: Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js is not installed or not in PATH
    echo Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

:: Navigate to migration directory
cd /d "%~dp0"

:: Install migration dependencies
echo 📦 Installing migration dependencies...
npm install
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

:: Execute migration
echo 🚀 Executing Unity Migration...
node execute-migration.js
if %errorlevel% neq 0 (
    echo ❌ Migration failed
    pause
    exit /b 1
)

:: Navigate to root and install project dependencies
cd ..
echo 📦 Installing project dependencies...
npm install
if %errorlevel% neq 0 (
    echo ❌ Failed to install project dependencies
    pause
    exit /b 1
)

:: Build the project
echo 🏗️ Building Unity Portal...
npm run build
if %errorlevel% neq 0 (
    echo ❌ Build failed
    pause
    exit /b 1
)

:: Start development server
echo 🌟 Starting development server...
echo Visit http://localhost:4321 to see your Unity Portal
npm run dev

pause