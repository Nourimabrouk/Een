#!/usr/bin/env python3
"""
Simple test script to verify core functionality without GUI dependencies
"""

import sys
import os


def test_core_functionality():
    """Test core packages without GUI dependencies"""
    print("Testing core functionality...")

    # Test NumPy
    try:
        import numpy as np

        arr = np.array([1, 2, 3, 4, 5])
        result = np.mean(arr)
        print(f"✓ NumPy: {result}")
    except Exception as e:
        print(f"✗ NumPy failed: {e}")
        return False

    # Test SciPy
    try:
        import scipy
        from scipy import stats

        data = [1, 2, 3, 4, 5]
        mean = stats.tmean(data)
        print(f"✓ SciPy: {mean}")
    except Exception as e:
        print(f"✗ SciPy failed: {e}")
        return False

    # Test Pandas
    try:
        import pandas as pd

        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        result = df.mean().mean()
        print(f"✓ Pandas: {result}")
    except Exception as e:
        print(f"✗ Pandas failed: {e}")
        return False

    # Test Matplotlib (non-GUI mode)
    try:
        import matplotlib

        matplotlib.use("Agg")  # Use non-GUI backend
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        plt.close(fig)
        print("✓ Matplotlib (non-GUI mode)")
    except Exception as e:
        print(f"✗ Matplotlib failed: {e}")
        return False

    # Test Flask
    try:
        from flask import Flask

        app = Flask(__name__)
        print("✓ Flask")
    except Exception as e:
        print(f"✗ Flask failed: {e}")
        return False

    # Test Plotly
    try:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 4, 2]))
        print("✓ Plotly")
    except Exception as e:
        print(f"✗ Plotly failed: {e}")
        return False

    # Test SymPy
    try:
        import sympy

        x = sympy.Symbol("x")
        expr = x**2 + 2 * x + 1
        result = sympy.factor(expr)
        print(f"✓ SymPy: {result}")
    except Exception as e:
        print(f"✗ SymPy failed: {e}")
        return False

    # Test NetworkX
    try:
        import networkx as nx

        G = nx.Graph()
        G.add_edge(1, 2)
        G.add_edge(2, 3)
        result = nx.number_of_nodes(G)
        print(f"✓ NetworkX: {result} nodes")
    except Exception as e:
        print(f"✗ NetworkX failed: {e}")
        return False

    # Test Scikit-learn
    try:
        from sklearn.linear_model import LinearRegression
        import numpy as np

        X = np.array([[1], [2], [3]])
        y = np.array([2, 4, 6])
        model = LinearRegression()
        model.fit(X, y)
        print("✓ Scikit-learn")
    except Exception as e:
        print(f"✗ Scikit-learn failed: {e}")
        return False

    return True


def test_platform_libraries():
    """Test platform independent libraries"""
    print("\nTesting platform libraries...")

    try:
        prefix = sys.prefix
        exec_prefix = sys.exec_prefix
        print(f"✓ sys.prefix: {prefix}")
        print(f"✓ sys.exec_prefix: {exec_prefix}")

        if os.path.exists(prefix):
            print("✓ Prefix directory accessible")
            return True
        else:
            print("✗ Prefix directory not accessible")
            return False
    except Exception as e:
        print(f"✗ Platform libraries failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 50)
    print("SIMPLE ENVIRONMENT TEST")
    print("=" * 50)

    # Test core functionality
    core_ok = test_core_functionality()

    # Test platform libraries
    platform_ok = test_platform_libraries()

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    if core_ok and platform_ok:
        print("🎉 SUCCESS: All core functionality working!")
        print("✅ Platform independent libraries issue FIXED")
        print("✅ Core packages working correctly")
        print("✅ Ready for Unity Mathematics development")
        return True
    else:
        print("⚠ Some issues remain")
        if not core_ok:
            print("✗ Core packages have issues")
        if not platform_ok:
            print("✗ Platform libraries still problematic")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
