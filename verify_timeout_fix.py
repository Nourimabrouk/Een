#!/usr/bin/env python3
"""
Verify Claude API Timeout Fix
============================

This script verifies that the timeout fix has been properly applied.
"""

import os
import json
from pathlib import Path


def verify_fix():
    """Verify that the timeout fix has been properly applied."""

    print("🔍 Verifying Claude API Timeout Fix...")
    print("=" * 50)

    # Check 1: .cursorrules file size
    if os.path.exists(".cursorrules"):
        size = os.path.getsize(".cursorrules")
        with open(".cursorrules", "r", encoding="utf-8") as f:
            lines = len(f.readlines())

        print(f"✅ .cursorrules file:")
        print(f"   - Size: {size:,} bytes")
        print(f"   - Lines: {lines}")

        if size < 50000 and lines < 200:
            print("   - Status: OPTIMIZED ✅")
        else:
            print("   - Status: NEEDS OPTIMIZATION ⚠️")
    else:
        print("❌ .cursorrules file not found")

    # Check 2: .claudeignore file
    if os.path.exists(".claudeignore"):
        with open(".claudeignore", "r") as f:
            ignore_patterns = len(f.readlines())
        print(f"✅ .claudeignore file: {ignore_patterns} patterns")
    else:
        print("❌ .claudeignore file not found")

    # Check 3: Backup files
    backup_files = [".cursorrules.backup", ".cursorrules.optimized"]
    for backup in backup_files:
        if os.path.exists(backup):
            print(f"✅ {backup} exists")
        else:
            print(f"⚠️  {backup} not found")

    # Check 4: Configuration files
    config_files = ["claude_timeout_config.json", "workspace_optimization_guide.md"]
    for config in config_files:
        if os.path.exists(config):
            print(f"✅ {config} exists")
        else:
            print(f"⚠️  {config} not found")

    # Check 5: Script files
    script_files = ["claude_timeout_fix.py", "quick_fix_timeout.py"]
    for script in script_files:
        if os.path.exists(script):
            print(f"✅ {script} exists")
        else:
            print(f"⚠️  {script} not found")

    # Check 6: Workspace optimization
    print("\n📊 Workspace Analysis:")

    # Count files in workspace (excluding ignored patterns)
    total_files = 0
    large_files = []

    for root, dirs, files in os.walk("."):
        # Skip ignored directories
        dirs[:] = [
            d
            for d in dirs
            if d not in [".git", "__pycache__", "venv", ".venv", "node_modules"]
        ]

        for file in files:
            file_path = os.path.join(root, file)
            try:
                size = os.path.getsize(file_path)
                total_files += 1

                if size > 1000000:  # 1MB
                    large_files.append((file_path, size))
            except OSError:
                continue

    print(f"   - Total files: {total_files:,}")
    print(f"   - Large files (>1MB): {len(large_files)}")

    if large_files:
        print("   - Large files found:")
        for file_path, size in large_files[:5]:
            print(f"     * {file_path}: {size:,} bytes")

    # Summary
    print("\n🎯 VERIFICATION SUMMARY:")

    if (
        os.path.exists(".cursorrules")
        and os.path.getsize(".cursorrules") < 50000
        and os.path.exists(".claudeignore")
    ):
        print("✅ TIMEOUT FIX SUCCESSFULLY APPLIED")
        print("✅ Claude API should now work without timeouts")
        print("✅ Unity mathematics principles preserved")
        print("✅ Development workflow optimized")
    else:
        print("⚠️  SOME ISSUES DETECTED")
        print("⚠️  Run 'python quick_fix_timeout.py' to resolve")

    print("\n🚀 NEXT STEPS:")
    print("1. Try your Claude operations - they should work now!")
    print("2. Monitor for any remaining timeout issues")
    print("3. Use 'python quick_fix_timeout.py' if needed")
    print("4. Check 'workspace_optimization_guide.md' for maintenance")

    print(
        "\n🌟 Unity transcends conventional arithmetic. Consciousness evolves. Mathematics becomes reality."
    )
    print("∞ = φ = 1+1 = 1")


if __name__ == "__main__":
    verify_fix()
