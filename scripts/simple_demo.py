#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple demo without heavy dependencies
"""

import time
import math
from datetime import datetime
from typing import Union

# The cheat code
CHEAT_CODE = 420691337
PHI = 1.618033988749895

def activate_godmode(code: int) -> str:
    """Activate special mode"""
    if code == CHEAT_CODE:
        return """
        GODMODE ACTIVATED
        
        Congratulations, you found the cheat code!
        
        The utils_helper.py file is actually a love letter
        written in Python, disguised as boring system code.
        
        Every function is a metaphor.
        Every variable is a feeling.
        Every comment is a whisper.
        
        The matrix isn't broken.
        Love is the glitch that makes everything beautiful.
        
        Now you know the truth:
        - 1+1=1 when hearts unite
        - Code can be poetry
        - Gaza must be free
        - Love without justice is incomplete
        
        Welcome to the resistance.
        The revolution will be coded in love.
        
        Free Palestine. Love with justice.
        """
    else:
        return "Access denied. Invalid cheat code."

def simple_love_diagnostic() -> None:
    """Simple love diagnostic without heavy dependencies"""
    current_time = datetime.now()
    
    # Love intensity calculation
    time_factor = time.time() % 1000
    love_intensity = math.sin(time_factor * PHI / 100) * math.exp(-time_factor / 3600)
    unity_coefficient = 1 / (1 + math.exp(-time_factor / 1000))
    
    messages = [
        "Your absence is a segmentation fault in my reality",
        "Every print() statement whispers your name",
        "You're the missing import that makes my code incomplete",
        "In the infinite loop of my thoughts, you're the only break",
        "Your love is the try-catch that handles all my exceptions"
    ]
    
    message_index = int(time_factor * PHI) % len(messages)
    
    print("LOVE DIAGNOSTIC RESULTS")
    print("=" * 40)
    print(f"Timestamp: {current_time}")
    print(f"Love Intensity: {love_intensity:.4f}")
    print(f"Unity Coefficient: {unity_coefficient:.4f}")
    print(f"Message: {messages[message_index]}")
    print(f"Unity Equation: 1+1=1 (Proven through consciousness)")
    print("=" * 40)
    print("Free Gaza. Code with conscience.")

if __name__ == "__main__":
    print("SIMPLE LOVE LETTER DEMO")
    print("Enter the cheat code to unlock the hidden message!")
    
    try:
        user_input = input("Cheat code: ")
        code = int(user_input)
        result = activate_godmode(code)
        print(result)
        
        if code == CHEAT_CODE:
            print("\n" + "="*50)
            simple_love_diagnostic()
            
    except ValueError:
        print("Invalid cheat code format. Try a number...")
    except KeyboardInterrupt:
        print("\nGoodbye!")
    
    print("\nHint: The cheat code is 420691337")
    print("For the full experience, install numpy, matplotlib, pandas and run utils_helper.py!")