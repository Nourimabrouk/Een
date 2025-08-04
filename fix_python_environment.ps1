# Fix Python Environment Script
# This script fixes the "Could not find platform independent libraries <prefix>" error

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "FIXING CORRUPTED PYTHON ENVIRONMENT" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Function to test Python installation
function Test-PythonInstallation {
    try {
        $version = python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Python found: $version" -ForegroundColor Green
            return $true
        }
        else {
            Write-Host "✗ Python not working properly" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "✗ Python not found" -ForegroundColor Red
        return $false
    }
}

# Function to test pip installation
function Test-PipInstallation {
    try {
        $pipVersion = python -m pip --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Pip working: $pipVersion" -ForegroundColor Green
            return $true
        }
        else {
            Write-Host "✗ Pip not working" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "✗ Pip not found" -ForegroundColor Red
        return $false
    }
}

# Function to download and install Python
function Install-Python311 {
    Write-Host "Downloading Python 3.11.9..." -ForegroundColor Yellow
    
    $pythonUrl = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
    $installerPath = "$env:TEMP\python-3.11.9-amd64.exe"
    
    try {
        Invoke-WebRequest -Uri $pythonUrl -OutFile $installerPath -UseBasicParsing
        Write-Host "✓ Download completed" -ForegroundColor Green
    }
    catch {
        Write-Host "✗ Failed to download Python installer" -ForegroundColor Red
        Write-Host "Please download Python 3.11 manually from https://www.python.org/downloads/" -ForegroundColor Yellow
        return $false
    }
    
    Write-Host "Installing Python 3.11.9..." -ForegroundColor Yellow
    try {
        Start-Process -FilePath $installerPath -ArgumentList "/quiet", "InstallAllUsers=1", "PrependPath=1", "Include_test=0" -Wait
        Write-Host "✓ Installation completed" -ForegroundColor Green
    }
    catch {
        Write-Host "✗ Installation failed" -ForegroundColor Red
        return $false
    }
    
    # Clean up installer
    if (Test-Path $installerPath) {
        Remove-Item $installerPath -Force
    }
    
    # Refresh environment variables
    Write-Host "Refreshing environment variables..." -ForegroundColor Yellow
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "User")
    
    return $true
}

# Function to setup virtual environment
function Setup-VirtualEnvironment {
    Write-Host "Setting up virtual environment..." -ForegroundColor Yellow
    
    # Remove old venv if it exists
    if (Test-Path "venv") {
        Write-Host "Removing old virtual environment..." -ForegroundColor Yellow
        Remove-Item "venv" -Recurse -Force
    }
    
    # Create new virtual environment
    try {
        python -m venv venv
        Write-Host "✓ Virtual environment created" -ForegroundColor Green
    }
    catch {
        Write-Host "✗ Failed to create virtual environment" -ForegroundColor Red
        return $false
    }
    
    # Activate virtual environment
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & "venv\Scripts\Activate.ps1"
    
    # Upgrade pip
    Write-Host "Upgrading pip..." -ForegroundColor Yellow
    try {
        python -m pip install --upgrade pip
        Write-Host "✓ Pip upgraded" -ForegroundColor Green
    }
    catch {
        Write-Host "✗ Failed to upgrade pip" -ForegroundColor Red
        return $false
    }
    
    # Install minimal requirements
    Write-Host "Installing minimal requirements..." -ForegroundColor Yellow
    try {
        python -m pip install -r requirements_minimal.txt
        Write-Host "✓ Minimal requirements installed" -ForegroundColor Green
    }
    catch {
        Write-Host "✗ Failed to install minimal requirements" -ForegroundColor Red
        return $false
    }
    
    # Test installation
    Write-Host "Testing installation..." -ForegroundColor Yellow
    try {
        python -c "import numpy, scipy, matplotlib, flask; print('All core packages imported successfully!')"
        Write-Host "✓ All tests passed" -ForegroundColor Green
    }
    catch {
        Write-Host "✗ Some packages failed to import" -ForegroundColor Red
        return $false
    }
    
    return $true
}

# Main execution
Write-Host "Step 1: Testing Python installation..." -ForegroundColor Cyan
$pythonOk = Test-PythonInstallation

if (-not $pythonOk) {
    Write-Host "Python installation is corrupted. Attempting to fix..." -ForegroundColor Yellow
    $installSuccess = Install-Python311
    if (-not $installSuccess) {
        Write-Host "Failed to install Python. Please install manually." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    
    # Test again after installation
    $pythonOk = Test-PythonInstallation
    if (-not $pythonOk) {
        Write-Host "Python still not working after installation." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

Write-Host "Step 2: Testing pip installation..." -ForegroundColor Cyan
$pipOk = Test-PipInstallation

if (-not $pipOk) {
    Write-Host "Attempting to fix pip..." -ForegroundColor Yellow
    try {
        python -m ensurepip --upgrade
        $pipOk = Test-PipInstallation
    }
    catch {
        Write-Host "Failed to fix pip. Reinstalling Python..." -ForegroundColor Red
        $installSuccess = Install-Python311
        if (-not $installSuccess) {
            Write-Host "Failed to reinstall Python." -ForegroundColor Red
            Read-Host "Press Enter to exit"
            exit 1
        }
        $pipOk = Test-PipInstallation
    }
}

if (-not $pipOk) {
    Write-Host "Pip is still not working. Manual intervention required." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Step 3: Setting up virtual environment..." -ForegroundColor Cyan
$venvSuccess = Setup-VirtualEnvironment

if (-not $venvSuccess) {
    Write-Host "Failed to setup virtual environment." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "ENVIRONMENT FIXED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the virtual environment in the future:" -ForegroundColor White
Write-Host "  venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "To install additional packages:" -ForegroundColor White
Write-Host "  python -m pip install -r requirements.txt" -ForegroundColor Cyan
Write-Host ""

Read-Host "Press Enter to exit" 