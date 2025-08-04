#!/usr/bin/env python3
"""
Test script to verify Python environment is working correctly
after fixing the "Could not find platform independent libraries <prefix>" error.
"""

import sys
import os
import subprocess


def test_python_basic():
    """Test basic Python functionality"""
    print("Testing basic Python functionality...")

    # Test Python version
    version = sys.version_info
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")

    # Test sys.executable
    print(f"✓ Python executable: {sys.executable}")

    # Test platform
    import platform

    print(f"✓ Platform: {platform.platform()}")

    return True


def test_pip():
    """Test pip functionality"""
    print("\nTesting pip functionality...")

    try:
        import pip

        print(f"✓ Pip version: {pip.__version__}")
        return True
    except ImportError:
        print("✗ Pip not available")
        return False


def test_core_packages():
    """Test core scientific packages"""
    print("\nTesting core packages...")

    packages = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("flask", "flask"),
        ("plotly", "plotly"),
        ("sympy", "sympy"),
        ("networkx", "networkx"),
        ("sklearn", "scikit-learn"),
    ]

    all_good = True
    for package_name, import_name in packages:
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "unknown")
            print(f"✓ {package_name}: {version}")
        except ImportError as e:
            print(f"✗ {package_name}: {e}")
            all_good = False

    return all_good


def test_platform_independent_libraries():
    """Test that platform independent libraries are working"""
    print("\nTesting platform independent libraries...")

    try:
        # Test that we can access sys.prefix without errors
        prefix = sys.prefix
        print(f"✓ sys.prefix: {prefix}")

        # Test that we can access sys.exec_prefix without errors
        exec_prefix = sys.exec_prefix
        print(f"✓ sys.exec_prefix: {exec_prefix}")

        # Test that we can list directories
        if os.path.exists(prefix):
            print(f"✓ Prefix directory exists and is accessible")
        else:
            print(f"✗ Prefix directory does not exist: {prefix}")
            return False

        return True
    except Exception as e:
        print(f"✗ Error accessing platform independent libraries: {e}")
        return False


def test_venv():
    """Test virtual environment"""
    print("\nTesting virtual environment...")

    # Check if we're in a virtual environment
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )

    if in_venv:
        print("✓ Running in virtual environment")
        print(f"✓ Virtual environment: {sys.prefix}")
        print(f"✓ Base Python: {sys.base_prefix}")
    else:
        print("⚠ Not running in virtual environment (this is OK for system Python)")

    return True


def test_import_specific_packages():
    """Test specific packages that might cause issues"""
    print("\nTesting specific packages...")

    # Test numpy operations
    try:
        import numpy as np

        arr = np.array([1, 2, 3, 4, 5])
        result = np.mean(arr)
        print(f"✓ NumPy operations: {result}")
    except Exception as e:
        print(f"✗ NumPy operations failed: {e}")
        return False

    # Test matplotlib
    try:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.close()
        print("✓ Matplotlib plotting")
    except Exception as e:
        print(f"✗ Matplotlib failed: {e}")
        return False

    # Test Flask
    try:
        from flask import Flask

        app = Flask(__name__)
        print("✓ Flask application creation")
    except Exception as e:
        print(f"✗ Flask failed: {e}")
        return False

    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("PYTHON ENVIRONMENT TEST")
    print("=" * 60)

    tests = [
        ("Basic Python", test_python_basic),
        ("Pip", test_pip),
        ("Core Packages", test_core_packages),
        ("Platform Libraries", test_platform_independent_libraries),
        ("Virtual Environment", test_venv),
        ("Specific Packages", test_import_specific_packages),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Your Python environment is working correctly.")
        return True
    else:
        print("⚠ Some tests failed. You may need to run the fix script again.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
