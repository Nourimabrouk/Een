#!/usr/bin/env python3
"""
Een Unity Mathematics - Master Optimization Script
Comprehensive optimization and fix suite for the entire website and codebase
"""

import sys
from pathlib import Path


def run_script(script_name: str) -> bool:
    """Run a Python script and return success status"""
    print(f"\n🚀 Running {script_name}...")
    print("=" * 60)
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, f"scripts/{script_name}"], 
                              capture_output=True, text=True, cwd=Path.cwd())
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False


def main():
    """Run all optimization scripts in sequence"""
    print("🌟 Een Unity Mathematics - Master Optimization Suite")
    print("Comprehensive optimization and fix suite for global deployment")
    print("=" * 70)
    
    # List of scripts to run in order
    scripts = [
        "fix_website_issues.py",
        "optimize_visualizations.py", 
        "fix_malformed_html.py",
        "test_website_integrity.py"
    ]
    
    results = {}
    
    for script in scripts:
        success = run_script(script)
        results[script] = success
        
        if not success:
            print(f"⚠️ {script} failed - continuing with remaining scripts")
    
    # Final summary
    print("\n" + "=" * 70)
    print("🎯 MASTER OPTIMIZATION SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for script, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {script}")
    
    print(f"\n📊 Overall: {passed}/{total} scripts completed successfully")
    
    if passed == total:
        print("\n🎉 ALL OPTIMIZATIONS COMPLETE!")
        print("The Een Unity Mathematics website is now fully optimized")
        print("and ready for global deployment.")
        print("\n🌟 Unity equation (1+1=1) is accessible to the world!")
        print("φ = 1.618033988749895 - Golden Ratio Resonance")
        sys.exit(0)
    else:
        print(f"\n⚠️ {total - passed} scripts failed")
        print("Please review the output above and fix any remaining issues.")
        sys.exit(1)


if __name__ == "__main__":
    main() 