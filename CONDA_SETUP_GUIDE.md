# Conda Setup Guide for Een Project

## Overview
This guide will help you set up a conda virtual environment for the Een Unity Mathematics Framework project.

## Prerequisites
- Windows 10/11
- Python 3.10+ (you have Python 3.13.5 ✅)
- Internet connection

## Step 1: Install Miniconda

### Option A: Download and Install Miniconda
1. Go to https://docs.conda.io/en/latest/miniconda.html
2. Download the Windows 64-bit installer
3. Run the installer as Administrator
4. **Important**: Check "Add Miniconda3 to my PATH environment variable"
5. Choose "Install for all users" (recommended)

### Option B: Using Chocolatey (if you have it)
```cmd
choco install miniconda3
```

### Option C: Using winget
```cmd
winget install Anaconda.Miniconda3
```

## Step 2: Verify Installation
After installation, open a **new** command prompt and run:
```cmd
conda --version
```

If conda is not found, manually add to PATH:
1. Open System Properties → Advanced → Environment Variables
2. Add to PATH: `C:\Users\%USERNAME%\miniconda3` and `C:\Users\%USERNAME%\miniconda3\Scripts`
3. Or: `C:\ProgramData\miniconda3` and `C:\ProgramData\miniconda3\Scripts`

## Step 3: Create Een Project Environment

### Initialize conda (first time only)
```cmd
conda init cmd.exe
```

### Create the Een environment
```cmd
conda create -n een python=3.11 -y
```

### Activate the environment
```cmd
conda activate een
```

### Install core dependencies
```cmd
conda install -c conda-forge numpy scipy pandas matplotlib plotly jupyter -y
```

### Install additional dependencies
```cmd
pip install sympy dash dash-bootstrap-components pytest pytest-cov black mypy click rich tqdm
```

### Install development dependencies
```cmd
pip install pytest-asyncio pytest-benchmark hypothesis pylint flake8 isort pre-commit safety bandit
```

## Step 4: Verify Setup

### Test the environment
```cmd
python -c "import numpy, scipy, pandas, matplotlib, plotly, sympy; print('All core packages imported successfully!')"
```

### Run a simple test
```cmd
python -c "from src.core.unity_equation import UnityEquation; print('Een framework loaded successfully!')"
```

## Step 5: Project-Specific Setup

### Install the Een package in development mode
```cmd
pip install -e .
```

### Run tests
```cmd
pytest tests/ -v
```

## Environment Management

### Activate environment
```cmd
conda activate een
```

### Deactivate environment
```cmd
conda deactivate
```

### List environments
```cmd
conda env list
```

### Remove environment (if needed)
```cmd
conda env remove -n een
```

## Troubleshooting

### If conda command not found:
1. Restart command prompt after installation
2. Check PATH environment variable
3. Try running from Anaconda Prompt

### If packages fail to install:
1. Update conda: `conda update conda`
2. Try conda-forge channel: `conda install -c conda-forge package_name`
3. Use pip as fallback: `pip install package_name`

### If import errors occur:
1. Ensure environment is activated: `conda activate een`
2. Check package installation: `conda list`
3. Reinstall problematic packages

## Alternative: Using venv (if conda fails)

If conda setup is problematic, you can use Python's built-in venv:

```cmd
python -m venv een_env
een_env\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Next Steps

After successful setup:
1. Activate the environment: `conda activate een`
2. Navigate to your project: `cd C:\Users\Nouri\Documents\GitHub\Een`
3. Run your scripts: `python run_viz.py` or `python demonstration.py`
4. Start development with your preferred IDE

## Environment File

The project includes:
- `requirements.txt` - Core dependencies
- `pyproject.toml` - Project configuration and dependencies
- `setup.py` - Package setup (legacy)

## Notes

- Python 3.11 is recommended for optimal compatibility
- The environment includes all dependencies for:
  - Unity mathematics operations
  - Interactive visualizations
  - Development and testing
  - MCP server functionality
  - Dashboard creation

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all prerequisites are met
3. Ensure proper PATH configuration
4. Try the alternative venv approach if needed 