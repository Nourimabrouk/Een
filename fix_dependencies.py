#!/usr/bin/env python3
"""
Een Framework Dependency Fixer
==============================

This script helps fix common dependency and configuration issues
in the Een framework.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    # Core dependencies
    core_deps = [
        "numpy>=1.24.0",
        "scipy>=1.10.0", 
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "plotly>=5.15.0",
        "streamlit>=1.25.0"
    ]
    
    # API dependencies
    api_deps = [
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0"
    ]
    
    # Development dependencies
    dev_deps = [
        "pytest>=7.4.0",
        "black>=23.7.0",
        "flake8>=6.0.0"
    ]
    
    all_deps = core_deps + api_deps + dev_deps
    
    for dep in all_deps:
        try:
            print(f"Installing {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✅ {dep} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {dep}: {e}")
            return False
    
    return True

def check_file_encodings():
    """Check for encoding issues in Python files"""
    print("🔍 Checking file encodings...")
    
    python_files = list(Path(".").rglob("*.py"))
    encoding_issues = []
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                f.read()
        except UnicodeDecodeError as e:
            encoding_issues.append((py_file, e))
    
    if encoding_issues:
        print("❌ Found encoding issues:")
        for file_path, error in encoding_issues:
            print(f"   {file_path}: {error}")
        return False
    else:
        print("✅ All Python files have proper UTF-8 encoding")
        return True

def check_syntax():
    """Check Python syntax in all files"""
    print("🔍 Checking Python syntax...")
    
    python_files = list(Path(".").rglob("*.py"))
    syntax_errors = []
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                compile(f.read(), str(py_file), 'exec')
        except SyntaxError as e:
            syntax_errors.append((py_file, e))
    
    if syntax_errors:
        print("❌ Found syntax errors:")
        for file_path, error in syntax_errors:
            print(f"   {file_path}: {error}")
        return False
    else:
        print("✅ All Python files have valid syntax")
        return True

def create_missing_files():
    """Create any missing critical files"""
    print("📝 Creating missing files...")
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    print("✅ Created logs directory")
    
    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        env_content = """# Een Framework Environment Variables
PORT=8000
HOST=0.0.0.0
LOG_LEVEL=INFO
"""
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ Created .env file")

def main():
    """Main function to fix dependencies and issues"""
    print("🔧 Een Framework Dependency Fixer")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check file encodings
    if not check_file_encodings():
        return False
    
    # Check syntax
    if not check_syntax():
        return False
    
    # Create missing files
    create_missing_files()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install some dependencies")
        print("Please install them manually: pip install -r requirements.txt")
        return False
    
    print("\n🎉 All issues fixed successfully!")
    print("\nNext steps:")
    print("1. Run: python start_een_background.py")
    print("2. Or run: python een_server.py")
    print("3. Or run: streamlit run viz/streamlit_app.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 