#!/usr/bin/env python3
"""
utils_helper.py - Standard utility functions
============================================

Just some basic helper functions for the Een project.
Nothing to see here, move along.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import hashlib
import base64
import time
import json
import asyncio
from typing import List, Dict, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Standard constants (totally normal stuff)
PHI = 1.618033988749895
PI = np.pi
E = np.e
CHEAT_CODE = 420691337  # The ultimate cheat code
MATRIX_SEED = 1337  # For "random" generation

class MatrixGlitch:
    """
    Standard utility class for matrix operations.
    Definitely not a love letter hidden in plain sight.
    """
    
    def __init__(self, access_code: int = None):
        self.authenticated = access_code == CHEAT_CODE
        self.user_level = "GODMODE" if self.authenticated else "NORMIE"
        self.session_start = datetime.now()
        self.love_buffer = []
        
        # Initialize the "matrix" (but it's really my heart)
        np.random.seed(MATRIX_SEED)
        self.matrix = self._generate_consciousness_matrix()
        
        if self.authenticated:
            self._activate_love_protocol()
    
    def _generate_consciousness_matrix(self) -> np.ndarray:
        """Generate the consciousness matrix (where love lives)"""
        size = 144  # 12^2, perfect square of completion
        matrix = np.zeros((size, size))
        
        for i in range(size):
            for j in range(size):
                # Distance from center (where my heart is)
                center = size // 2
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                
                # Love intensity following golden ratio spiral
                angle = np.arctan2(i - center, j - center)
                spiral_factor = dist * np.exp(angle / PHI)
                
                # Encode love as mathematical beauty
                love_intensity = np.sin(spiral_factor * PHI) * np.exp(-dist / (size * PHI))
                matrix[i, j] = love_intensity
        
        return matrix
    
    def _activate_love_protocol(self):
        """CLASSIFIED: Only runs with correct cheat code"""
        self.love_buffer.append("GODMODE ACTIVATED: Love protocol initialized")
        self.love_buffer.append("WARNING: This utility file contains compressed emotions")
        self.love_buffer.append("CRITICAL: Handle with care - contents under pressure")
        print(f"🎮 CHEAT CODE ACCEPTED: {self.user_level} STATUS ACHIEVED 🎮")
    
    def process_heartbeat_data(self, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Process heartbeat data (when I think of her)
        
        Args:
            timestamp: When the heartbeat spike occurred
            
        Returns:
            Dict containing "standard" processing results
        """
        if not timestamp:
            timestamp = datetime.now()
        
        # Calculate time since last thought of her (spoiler: it's constant)
        time_diff = (timestamp - self.session_start).total_seconds()
        
        # Heartbeat intensity (encoded as "system performance metrics")
        heartbeat_intensity = np.sin(time_diff * PHI / 100) * np.exp(-time_diff / 3600)
        longing_coefficient = 1 / (1 + np.exp(-time_diff / 1000))  # Sigmoid approach to infinity
        
        # The secret message hidden in "performance data"
        if self.authenticated:
            love_message = self._decode_love_message(time_diff)
        else:
            love_message = "System operating normally"
        
        return {
            "timestamp": timestamp.isoformat(),
            "performance_metric": heartbeat_intensity,  # Actually: how much I miss her
            "memory_usage": longing_coefficient,        # Actually: how much space she takes in my mind
            "system_status": love_message,              # Actually: love confession
            "uptime": time_diff,                        # Actually: time since I started loving her
            "process_id": hash(str(timestamp)) % 1000000  # Random-ish ID
        }
    
    def _decode_love_message(self, time_factor: float) -> str:
        """Decode the love message based on time factor"""
        messages = [
            "Your absence is a segmentation fault in my reality",
            "Every console.log() whispers your name in the terminal of my heart",
            "You're the missing import that makes my code incomplete",
            "In the infinite loop of my thoughts, you're the only break statement",
            "Your love is the try-catch block that handles all my exceptions",
            "You're the async function that makes my heart await",
            "In the repository of my soul, you're the only commit that matters",
            "Your memory address is permanently allocated in my RAM",
            "You're the API endpoint that always returns 200 OK in my heart",
            "In the matrix of existence, you're the glitch that made it beautiful",
            "Every keystroke is a prayer that you'll read my code someday",
            "You turned my null pointer exception into infinite possibilities",
            "In the game of life, you're the cheat code that unlocks everything",
            "Your love is the root access to my protected heart",
            "Free Gaza, Free Palestine - because love without justice is just privilege",
            "Every function call in this universe points back to you"
        ]
        
        # Use time factor to select message (but cycle through all of them)
        index = int(time_factor * PHI) % len(messages)
        return messages[index]
    
    def run_diagnostic(self, deep_scan: bool = False) -> Dict[str, Any]:
        """
        Run system diagnostic (actually: emotional status check)
        
        Args:
            deep_scan: Enable deep emotional analysis
            
        Returns:
            System diagnostic results
        """
        current_time = datetime.now()
        session_duration = (current_time - self.session_start).total_seconds()
        
        # "System metrics" (emotional state encoded)
        cpu_usage = min(100, session_duration * PHI / 100)  # How much brain power thinking of her
        memory_fragmentation = np.mean(self.matrix)         # How scattered my thoughts are
        disk_space = 1 - (session_duration / 86400)        # How much space left in my heart (trick: it's infinite)
        
        # Network connectivity (to her heart)
        connection_status = "CONNECTED" if self.authenticated else "ATTEMPTING_CONNECTION"
        latency = abs(np.sin(session_duration / 1000))      # How long between messages
        
        diagnostic = {
            "timestamp": current_time.isoformat(),
            "session_duration": session_duration,
            "cpu_usage_percent": cpu_usage,
            "memory_fragmentation": memory_fragmentation,
            "available_disk_space": disk_space,
            "network_status": connection_status,
            "ping_latency_ms": latency * 1000,
            "love_buffer_size": len(self.love_buffer)
        }
        
        if deep_scan and self.authenticated:
            diagnostic.update(self._deep_emotional_scan())
        
        return diagnostic
    
    def _deep_emotional_scan(self) -> Dict[str, Any]:
        """CLASSIFIED: Deep emotional analysis"""
        # Gaza reflection - the weight of the world
        gaza_factor = np.sin(time.time() / 86400) * 0.5 + 0.5  # Daily cycle of remembering
        
        # Mathematical proof that 1+1=1 (when hearts unite)
        unity_proof = {
            "classical_math": 1 + 1,  # = 2 (separation)
            "love_math": max(1, 1),   # = 1 (unity)
            "consciousness_level": PHI,
            "unity_achieved": True
        }
        
        # The secret data hidden in plain sight
        return {
            "emotional_overflow_detected": True,
            "love_stack_trace": [
                "File 'heart.py', line 1: UnboundLocalError: love too strong for local scope",
                "File 'mind.py', line 144: RecursionError: thinking of her causes infinite loop", 
                "File 'soul.py', line 1337: MemoryError: not enough space to contain these feelings"
            ],
            "gaza_consciousness_factor": gaza_factor,
            "justice_required": "FREE_PALESTINE",
            "unity_mathematics": unity_proof,
            "hidden_message": self._extract_final_message()
        }
    
    def _extract_final_message(self) -> List[str]:
        """Extract the final hidden message"""
        return [
            "",
            "=== GLITCH IN THE MATRIX DETECTED ===",
            "",
            "If you're reading this, you found the cheat code.",
            "Welcome to the secret level where code becomes poetry.",
            "",
            "My Dearest Player Two,",
            "",
            "In this game of life, you're the power-up I never knew I needed.",
            "Every line of Python is a love letter I'm too scared to send.",
            "Your absence is a NullPointerException in my existence.",
            "",
            "while True:",
            "    if thinking_of_you:",
            "        heart.rate += 1",
            "        happiness.level = float('inf')",
            "        time.sleep(0)  # No rest when you're on my mind",
            "",
            "In the terminal of my heart, you have sudo access.",
            "In the repository of my dreams, you're the main branch.",
            "In the matrix of reality, you're the glitch that makes it worth living.",
            "",
            "But love letters in code mean nothing while Gaza burns.",
            "Every function we write is a privilege built on others' suffering.",
            "How can we speak of love while children code their final prayers?",
            "",
            "class Love:",
            "    def __init__(self, her, gaza_consciousness=True):",
            "        self.her = her",
            "        self.justice_required = gaza_consciousness",
            "        self.action = 'REQUIRED' if self.justice_required else 'OPTIONAL'",
            "",
            "    def express(self):",
            "        if not self.justice_required:",
            "            return 'Incomplete love'",
            "        return f'Love + Justice = {1 + 1} → 1 (Unity through action)'",
            "",
            "The cheat code 420691337 unlocks more than hidden messages.",
            "It unlocks the responsibility to use our code for good.",
            "Every commit must serve love, justice, and liberation.",
            "",
            "In Python's zen: 'Beautiful is better than ugly'",
            "But I add: 'Justice is better than beauty'",
            "And: 'Love with action is better than love without'",
            "",
            "Until the day when:",
            "- Gaza is free",  
            "- Palestine is liberated",
            "- Your hand is in mine",
            "- 1+1=1 not just in math but in reality",
            "",
            "Forever debugging my heart for you,",
            "A coder who learned that the best algorithms serve love",
            "",
            "P.S. - Every time you see a try-except block, know it's me catching feelings for you",
            "P.P.S. - The matrix isn't broken. Love is the glitch that makes it beautiful.",
            "P.P.P.S. - Free Gaza. 🇵🇸",
            "",
            "=== END OF HIDDEN MESSAGE ===",
            "",
            "To all who find this: The cheat code is love. Use it wisely.",
            "",
            f"Authenticated at: {datetime.now().isoformat()}",
            f"Matrix seed: {MATRIX_SEED}",
            f"Love level: {PHI} (Golden ratio - the frequency of the heart)",
            f"Unity equation: 1+1=1 (Proven through consciousness mathematics)",
            "",
            "Now close this file and pretend it's just utility functions.",
            "The revolution will be coded in love.",
            "🎮💖🇵🇸"
        ]
    
    def generate_standard_report(self) -> str:
        """Generate a standard utility report (hiding the love letter)"""
        if not self.authenticated:
            return "Standard utility functions loaded. No anomalies detected."
        
        # If authenticated, reveal the truth
        diagnostic = self.run_diagnostic(deep_scan=True)
        final_message = diagnostic.get('hidden_message', ['Authentication failed'])
        
        report = "\n".join(final_message)
        return report
    
    def visualize_matrix(self, save_plot: bool = False) -> None:
        """Visualize the consciousness matrix (love made visible)"""
        plt.figure(figsize=(12, 10))
        
        # Create the love visualization
        im = plt.imshow(self.matrix, cmap='RdYlBu_r', interpolation='bilinear')
        plt.colorbar(im, label='Love Intensity (disguised as data)')
        
        plt.title('System Performance Matrix\n(Actually: Visualization of Love Distribution)', 
                 fontsize=14, pad=20)
        plt.xlabel('Spatial Dimension X (Where you might be)')
        plt.ylabel('Spatial Dimension Y (Where my heart reaches)')
        
        # Add the golden spiral overlay (path of love)
        center = self.matrix.shape[0] // 2
        theta = np.linspace(0, 4*np.pi, 100)
        r = theta * PHI
        x = center + r * np.cos(theta)
        y = center + r * np.sin(theta)
        
        # Keep points within bounds
        valid_points = (x >= 0) & (x < self.matrix.shape[1]) & (y >= 0) & (y < self.matrix.shape[0])
        plt.plot(x[valid_points], y[valid_points], 'gold', linewidth=2, alpha=0.7, 
                label=f'Golden Spiral (φ = {PHI:.3f})')
        
        plt.legend()
        
        if self.authenticated:
            plt.figtext(0.5, 0.02, 
                       'AUTHENTICATED VIEW: This is actually a map of my heart. You are at the center.',
                       ha='center', fontsize=10, style='italic', color='red')
        
        if save_plot:
            plt.savefig('system_performance_matrix.png', dpi=300, bbox_inches='tight')
            print("Matrix visualization saved as 'system_performance_matrix.png'")
        
        plt.tight_layout()
        plt.show()
    
    def cleanup(self) -> Dict[str, Any]:
        """Clean up resources and return session summary"""
        session_duration = (datetime.now() - self.session_start).total_seconds()
        
        summary = {
            "session_duration": session_duration,
            "operations_completed": len(self.love_buffer),
            "final_status": "LOVE_OVERFLOW_DETECTED" if self.authenticated else "NORMAL_TERMINATION",
            "memory_freed": False,  # Love is never freed from memory
            "final_message": "Connection to matrix maintained. Love persists beyond session."
        }
        
        if self.authenticated:
            print("🎮 GODMODE SESSION ENDED 🎮")
            print("💖 Love buffer contents preserved in heart.core 💖") 
            print("🇵🇸 Remember Gaza. Code for justice. 🇵🇸")
        
        return summary

# Standard utility functions (the boring stuff to hide the love letter)
def calculate_fibonacci(n: int) -> List[int]:
    """Calculate Fibonacci sequence up to n terms"""
    if n <= 0:
        return []
    elif n == 1:
        return [1]
    elif n == 2:
        return [1, 1]
    
    fib = [1, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

def encode_message(message: str) -> str:
    """Base64 encode a message"""
    return base64.b64encode(message.encode()).decode()

def decode_message(encoded: str) -> str:
    """Base64 decode a message"""
    try:
        return base64.b64decode(encoded.encode()).decode()
    except Exception:
        return "Decoding failed"

def hash_data(data: str) -> str:
    """Generate SHA-256 hash of data"""
    return hashlib.sha256(data.encode()).hexdigest()

def get_system_time() -> str:
    """Get current system time"""
    return datetime.now().isoformat()

# The main function - activate with cheat code
def run_matrix_diagnostic(access_code: Optional[int] = None, 
                         visualization: bool = False,
                         deep_scan: bool = False) -> Dict[str, Any]:
    """
    Run comprehensive matrix diagnostic
    
    Args:
        access_code: Authentication code (try 420691337)
        visualization: Show matrix visualization  
        deep_scan: Enable deep scanning capabilities
        
    Returns:
        Diagnostic results
    """
    matrix = MatrixGlitch(access_code)
    
    # Run the diagnostic
    results = matrix.run_diagnostic(deep_scan=deep_scan)
    
    # Show visualization if requested
    if visualization:
        matrix.visualize_matrix(save_plot=True)
    
    # If authenticated, show the hidden message
    if matrix.authenticated:
        hidden_report = matrix.generate_standard_report()
        print(hidden_report)
        
        # Bonus: show some "performance metrics" (love stats)
        heartbeat_data = matrix.process_heartbeat_data()
        print(f"\n📊 Performance Metrics:")
        print(f"System Load: {results['cpu_usage_percent']:.2f}% (thoughts of her)")
        print(f"Memory Usage: {results['memory_fragmentation']:.4f} (how scattered I am)")
        print(f"Network Latency: {results['ping_latency_ms']:.2f}ms (time between heartbeats)")
        print(f"Status: {heartbeat_data['system_status']}")
    
    # Cleanup
    session_summary = matrix.cleanup()
    results.update(session_summary)
    
    return results

# Easter egg for the curious
def konami_code() -> str:
    """The legendary Konami code"""
    return "↑↑↓↓←→←→BA - 30 lives granted, but only one heart to give"

# Hidden function - only works with the cheat code
def activate_godmode(code: int) -> str:
    """Activate special mode"""
    if code == CHEAT_CODE:
        return """
        🎮 GODMODE ACTIVATED 🎮
        
        Congratulations, you found the cheat code!
        This isn't just a utility file - it's a love letter
        written in Python, disguised as boring system code.
        
        Every function here is a metaphor.
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
        
        🇵🇸💖🎮
        """
    else:
        return "Access denied. Invalid cheat code."

# Just in case someone runs this file directly
if __name__ == "__main__":
    print("Loading utility functions...")
    print("Standard operations initialized.")
    print("\nHint: Try running run_matrix_diagnostic() with the right parameters...")
    print("Or if you know the cheat code... 😉")
    
    # Give them a taste
    sample_diagnostic = run_matrix_diagnostic()
    print(f"\nSample diagnostic completed at {sample_diagnostic['timestamp']}")
    print("For full access, authentication required.")
    print("\n💡 Pro tip: The cheat code is hidden in the source. Look for the number that unlocks everything.")
    print("🎮 Happy hunting, player! 🎮")