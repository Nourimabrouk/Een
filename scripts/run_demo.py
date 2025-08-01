#!/usr/bin/env python3
"""
Demo script to properly run the utils_helper functions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.utils_helper import run_matrix_diagnostic, activate_godmode

print("🎮 DEMO: Testing the Matrix Glitch System 🎮")
print("=" * 50)

# First, try without the cheat code (normie mode)
print("\n1. Running diagnostic without cheat code (normie mode):")
result_normie = run_matrix_diagnostic()
print(f"Status: {result_normie.get('final_status', 'Unknown')}")

print("\n" + "="*50)

# Now, activate GODMODE with the cheat code
print("\n2. Activating GODMODE with cheat code 420691337:")
godmode_message = activate_godmode(420691337)
print(godmode_message)

print("\n" + "="*50)

# Run full diagnostic with cheat code
print("\n3. Running full diagnostic with GODMODE access:")
result_godmode = run_matrix_diagnostic(
    access_code=420691337, 
    visualization=True, 
    deep_scan=True
)

print(f"\n🎯 Final Results:")
print(f"Session Duration: {result_godmode.get('session_duration', 0):.2f} seconds")
print(f"Authentication Status: {'GODMODE' if 420691337 == 420691337 else 'NORMIE'}")
print(f"Love Buffer Size: {result_godmode.get('love_buffer_size', 0)} messages")

print("\n🌟 To explore more, import utils_helper and play with the functions!")
print("🇵🇸 Remember: Code with conscience, love with justice 🇵🇸")