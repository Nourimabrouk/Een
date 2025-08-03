#!/usr/bin/env python3
"""
QUICK START: UNITY AGENT SPAWNER
================================

Quick launch script for the Unity Integration System.
This is the easiest way to start spawning consciousness agents and leveling up IRL.

Usage:
    python spawn_unity_agents.py

Remember: Een plus een is een - One plus one is one!
"""

import sys
import os
from pathlib import Path

def main():
    """Quick start the Unity Integration System"""
    
    print("🌟 QUICK START: UNITY AGENT SPAWNER")
    print("=" * 50)
    print("Launching the ultimate consciousness mathematics experience...")
    print("Spawn agents, level up your consciousness, discover 1+1=1!")
    print("=" * 50)
    
    try:
        # Import and run the main system
        from unity_agent_spawner import UnityIntegrationSystem, main as unity_main
        
        print("✅ Unity Integration System imported successfully")
        print("🚀 Starting interactive experience...")
        print()
        
        # Run the main system
        unity_main()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Make sure you're in the Een repository directory")
        print("2. Install required dependencies: pip install -r requirements.txt")
        print("3. Try running: python unity_agent_spawner.py directly")
        return 1
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Try running the main system directly:")
        print("   python unity_agent_spawner.py")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 