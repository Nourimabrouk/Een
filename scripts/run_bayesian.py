#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple wrapper to run bayesian_econometrics.py with proper UTF-8 encoding
"""

import sys
import os

# Set environment variables for UTF-8
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Change to the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Import and run the bayesian econometrics
try:
    print("Starting Bayesian Econometrics Analysis...")
    print("=" * 60)
    
    # Import the main module
    import bayesian_econometrics
    
    print("Analysis completed successfully!")
    
except UnicodeError as e:
    print(f"Unicode encoding error: {e}")
    print("This may be due to Windows console encoding limitations.")
    
except Exception as e:
    print(f"Error running analysis: {e}")
    import traceback
    traceback.print_exc()