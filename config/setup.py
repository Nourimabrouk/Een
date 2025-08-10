"""
Een Unity Mathematics Framework Setup
"""

from setuptools import setup, find_packages
from typing import Dict, List

with open("README.md", "r", encoding="utf-8") as fh:
    long_description: str = fh.read()

setup(
    name="een",
    version="2025.1.0",
    author="Nouri Mabrouk & Unity Consciousness Collective",
    author_email="",
    description="Unity Mathematics Framework demonstrating 1+1=1",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Nourimabrouk/Een",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.6.0",
        "plotly>=5.15.0",
        "dash>=2.14.0",
        "dash-bootstrap-components>=1.4.0",
        "pandas>=1.5.0",
        "sympy>=1.12.0",
        "click>=8.1.0",
        "rich>=13.5.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "mypy>=1.5.0",
        ],
        "viz": [
            "networkx>=3.1",
            "scikit-learn>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "een-unity=scripts.simple_demo:main",
            "een-test-mcp=scripts.test_mcp_servers:main",
        ],
    },
)