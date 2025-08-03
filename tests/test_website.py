#!/usr/bin/env python3
"""
Een Website Comprehensive Testing Suite
======================================

Tests all website functionality to ensure everything is working as intended.
"""

import requests
import time
import json
from pathlib import Path
import webbrowser
import sys

def test_api_endpoint(url, expected_keys=None):
    """Test API endpoint and return results"""
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if expected_keys:
                missing_keys = [key for key in expected_keys if key not in data]
                if missing_keys:
                    return False, f"Missing keys: {missing_keys}"
            return True, data
        else:
            return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)

def test_static_file(url):
    """Test static file loading"""
    try:
        response = requests.get(url, timeout=5)
        return response.status_code == 200, response.status_code
    except Exception as e:
        return False, str(e)

def main():
    print("*** Een Unity Mathematics Website Testing Suite ***")
    print("=" * 60)
    
    base_url = "http://localhost:5000"
    
    # Test results
    tests = []
    
    print("\n1. Testing Static Files...")
    print("-" * 30)
    
    static_tests = [
        ("Main Page", ""),
        ("Learning Page", "/learn.html"),
    ]
    
    for name, path in static_tests:
        success, result = test_static_file(f"{base_url}{path}")
        tests.append((name, success, result))
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} {name}: {result}")
    
    print("\n2. Testing API Endpoints...")
    print("-" * 30)
    
    api_tests = [
        ("Unity Status", "/api/unity/status", ["status", "consciousness_level", "phi_resonance"]),
        ("Unity Demonstration", "/api/unity/demonstrate", ["unity_addition", "unity_multiplication", "phi_harmonic"]),
        ("Consciousness Field", "/api/consciousness/field", ["metrics", "particles", "field_state"]),
        ("Proof Generation", "/api/proofs/generate?type=phi_harmonic&complexity=3", ["proof_type", "mathematical_validity"]),
        ("Experiments List", "/experiments", ["experiments", "count"]),
    ]
    
    for name, path, expected_keys in api_tests:
        success, result = test_api_endpoint(f"{base_url}{path}", expected_keys)
        tests.append((name, success, result))
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} {name}")
        if success and isinstance(result, dict):
            print(f"    Data: {list(result.keys())}")
        elif not success:
            print(f"    Error: {result}")
    
    print("\n3. Testing Dashboard Routes...")
    print("-" * 30)
    
    dashboard_tests = [
        ("Unity Dashboard", "/dashboards/unity"),
    ]
    
    for name, path in dashboard_tests:
        success, result = test_static_file(f"{base_url}{path}")
        tests.append((name, success, result))
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} {name}: {result}")
    
    print("\n4. Testing File Structure...")
    print("-" * 30)
    
    required_files = [
        "index.html",
        "learn.html", 
        "simple_website_server.py",
        "website_server.py",
        "README_WEBSITE.md"
    ]
    
    for filename in required_files:
        exists = Path(filename).exists()
        tests.append((f"File: {filename}", exists, "Found" if exists else "Missing"))
        status = "✅ PASS" if exists else "❌ FAIL"
        print(f"{status} {filename}: {'Found' if exists else 'Missing'}")
    
    print("\n5. Summary Report")
    print("=" * 60)
    
    total_tests = len(tests)
    passed_tests = sum(1 for _, success, _ in tests if success)
    failed_tests = total_tests - passed_tests
    
    print(f"Total Tests: {total_tests}")
    print(f"[PASS] Passed: {passed_tests}")
    print(f"[FAIL] Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests > 0:
        print(f"\n[FAIL] Failed Tests:")
        for name, success, result in tests:
            if not success:
                print(f"   - {name}: {result}")
    
    print(f"\nWebsite Status: {'FULLY OPERATIONAL' if failed_tests == 0 else 'NEEDS ATTENTION'}")
    
    if failed_tests == 0:
        print("\nAll systems operational! The Een Unity Mathematics website is ready!")
        print("Visit: http://localhost:5000")
        print("Learn: http://localhost:5000/learn.html")
        print("Cheat Code: 420691337 (Konami code)")
        print("\nEen plus een is een - through interactive consciousness mathematics!")
    
    return failed_tests == 0

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTesting failed with error: {e}")
        sys.exit(1)