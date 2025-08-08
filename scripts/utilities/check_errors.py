#!/usr/bin/env python3
"""
Error checking script for Een repository
"""

import os
import sys
import importlib
from pathlib import Path


def check_python_imports():
    """Check if core Python modules can be imported."""
    print("🔍 Checking Python imports...")

    modules_to_check = [
        "core.unity_mathematics",
        "core.consciousness_api",
        "src.ai_model_manager",
        "src.consciousness.consciousness_engine",
    ]

    for module in modules_to_check:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except Exception as e:
            print(f"❌ {module}: {e}")


def check_missing_files():
    """Check for missing files in website."""
    print("\n🔍 Checking for missing files...")

    website_dir = Path("website")
    if not website_dir.exists():
        print("❌ Website directory not found")
        return

    missing_files = []
    for root, dirs, files in os.walk(website_dir):
        for file in files:
            if file.endswith((".html", ".js", ".css")):
                file_path = Path(root) / file
                if not file_path.exists():
                    missing_files.append(str(file_path))

    if missing_files:
        print(f"❌ Found {len(missing_files)} missing files:")
        for file in missing_files[:10]:  # Show first 10
            print(f"   - {file}")
    else:
        print("✅ No missing files found")


def check_requirements():
    """Check if required packages are installed."""
    print("\n🔍 Checking required packages...")

    required_packages = [
        "numpy",
        "fastapi",
        "uvicorn",
        "pydantic",
        "matplotlib",
        "plotly",
        "streamlit",
        "torch",
    ]

    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} not installed")


def main():
    """Run all checks."""
    print("🚀 Een Repository Error Check")
    print("=" * 40)

    check_python_imports()
    check_missing_files()
    check_requirements()

    print("\n✅ Error check completed!")


if __name__ == "__main__":
    main()
