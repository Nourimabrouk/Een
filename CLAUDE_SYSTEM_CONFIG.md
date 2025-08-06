# Claude Code System Configuration

## System Information
- **OS**: Windows 10/11
- **Python Location**: `C:\Users\Nouri\Documents\GitHub\Een\een\Scripts\python.exe`
- **Virtual Environment**: `C:\Users\Nouri\Documents\GitHub\Een\een\`
- **Project Root**: `C:\Users\Nouri\Documents\GitHub\Een`
- **Website Directory**: `C:\Users\Nouri\Documents\GitHub\Een\website`

## Important Notes
- **NO A:/ or B:/ drives exist** - These are legacy floppy disk drives not present on modern systems
- **Always activate virtual environment first** before running Python commands
- Use `C:\` as the primary drive for all operations

## Correct Command Patterns

### ✅ Virtual Environment Activation
```bash
# Correct way to activate venv
cd "C:\Users\Nouri\Documents\GitHub\Een"
cmd /c "een\Scripts\activate.bat"

# Or use the batch launcher
./START_WEBSITE.bat
```

### ✅ Python Commands (After venv activation)
```bash
# Run Python scripts
"C:\Users\Nouri\Documents\GitHub\Een\een\Scripts\python.exe" script.py

# Start web server
cd "C:\Users\Nouri\Documents\GitHub\Een\website"
python -m http.server 8001
```

### ✅ File Operations
```bash
# Navigate to project
cd "C:\Users\Nouri\Documents\GitHub\Een"

# List files
ls website/
dir website\
```

### ❌ Common Mistakes to Avoid
```bash
# DON'T use A:/ or B:/ drives (they don't exist)
# WRONG: cd A:\
# WRONG: A:\project\file.py

# DON'T run Python without activating venv first
# WRONG: python script.py (without venv activation)

# DON'T use Unix paths on Windows without proper quoting
# WRONG: cd C:\Users\Nouri\Documents\GitHub\Een (use quotes)
```

## Quick Reference Commands

### Start Website Development Server
```bash
# Method 1: Use the launcher script
cd "C:\Users\Nouri\Documents\GitHub\Een"
START_WEBSITE.bat

# Method 2: Manual activation
cd "C:\Users\Nouri\Documents\GitHub\Een"
cmd /c "een\Scripts\activate.bat"
cd website
python -m http.server 8001
```

### Install Python Packages
```bash
cd "C:\Users\Nouri\Documents\GitHub\Een"
cmd /c "een\Scripts\activate.bat"
pip install package_name
```

### Run Unity Mathematics Scripts
```bash
cd "C:\Users\Nouri\Documents\GitHub\Een"
cmd /c "een\Scripts\activate.bat"
python -c "from core.unity_mathematics import *; demonstrate_unity_addition(1, 1)"
```

## Website Access URLs
- **Main Hub**: http://localhost:8001/metastation-hub.html
- **Zen Meditation**: http://localhost:8001/zen-unity-meditation.html
- **Implementations**: http://localhost:8001/implementations-gallery.html
- **Mathematical Framework**: http://localhost:8001/mathematical-framework.html
- **Site Map**: http://localhost:8001/sitemap.html

## Development Workflow
1. Always start with `cd "C:\Users\Nouri\Documents\GitHub\Een"`
2. Activate virtual environment: `cmd /c "een\Scripts\activate.bat"`
3. Navigate to appropriate directory for the task
4. Run commands using the activated Python environment
5. For web development, use the `START_WEBSITE.bat` launcher

## Unity Mathematics Specific Commands
```bash
# Test core unity mathematics
cd "C:\Users\Nouri\Documents\GitHub\Een"
cmd /c "een\Scripts\activate.bat"
python -c "
from core.unity_mathematics import UnityMathematics
um = UnityMathematics()
print(f'1 + 1 = {um.unity_add(1, 1)}')
"

# Run transcendental reality engine
python -c "
from src.consciousness.transcendental_reality_engine import demonstrate_3000_elo_transcendental_reality
demonstrate_3000_elo_transcendental_reality()
"
```

## System Compatibility Notes
- **Drive Letters**: Only C:\ (and possibly D:\, E:\, etc.) exist - NO A:\ or B:\
- **Path Separators**: Use `\` for Windows paths or forward slashes `/` with proper quoting
- **Virtual Environment**: Located at `een\Scripts\` (Windows) not `een/bin/` (Unix)
- **Python Executable**: `python.exe` not `python`
- **Batch Files**: Use `.bat` extension for Windows scripts

## Claude Code Recommendations
When working with this project:
1. **Always check system constraints** before suggesting A:/ or B:/ drive operations
2. **Verify virtual environment activation** before running Python commands  
3. **Use proper Windows path formatting** with quotes around paths containing spaces
4. **Suggest the batch launcher** for easy web server startup
5. **Remember the project structure** and use appropriate directories for different tasks

This configuration should be referenced for all future interactions with this project to ensure compatibility with the actual system setup.