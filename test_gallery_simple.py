#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test for 3000 ELO gallery system
"""

import os
import sys
from pathlib import Path

def test_file_discovery():
    """Test file discovery in viz directories."""
    print("\n[TEST] Testing File Discovery...")
    
    viz_folder = Path(__file__).parent / "viz"
    legacy_folder = viz_folder / "legacy images"
    
    found_files = []
    
    # Test main viz folder
    if viz_folder.exists():
        files = [f for f in viz_folder.iterdir() if f.is_file()]
        print(f"[INFO] Main viz folder: {len(files)} files found")
        for file in files:
            print(f"   - {file.name} ({file.suffix})")
            found_files.append(file.name)
    else:
        print(f"[WARNING] Main viz folder not found: {viz_folder}")
    
    # Test legacy images folder
    if legacy_folder.exists():
        files = [f for f in legacy_folder.iterdir() if f.is_file()]
        print(f"[INFO] Legacy images folder: {len(files)} files found")
        for file in files:
            print(f"   - {file.name} ({file.suffix})")
            found_files.append(file.name)
    else:
        print(f"[WARNING] Legacy images folder not found: {legacy_folder}")
    
    print(f"[RESULT] Total files discovered: {len(found_files)}")
    return len(found_files) > 0

def test_gallery_structure():
    """Test gallery directory structure."""
    print("\n[TEST] Testing Gallery Structure...")
    
    base_path = Path(__file__).parent
    
    # Check key directories
    directories_to_check = [
        "viz",
        "viz/legacy images",
        "website",
        "website/js",
        "api/routes"
    ]
    
    structure_ok = True
    
    for dir_path in directories_to_check:
        full_path = base_path / dir_path
        if full_path.exists():
            print(f"[OK] Directory exists: {dir_path}")
        else:
            print(f"[ERROR] Directory missing: {dir_path}")
            structure_ok = False
    
    # Check key files
    files_to_check = [
        "website/js/dynamic-gallery-loader.js",
        "api/routes/gallery.py",
        "api/routes/gallery_helpers.py"
    ]
    
    for file_path in files_to_check:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"[OK] File exists: {file_path}")
        else:
            print(f"[ERROR] File missing: {file_path}")
            structure_ok = False
    
    return structure_ok

def test_import_helpers():
    """Test importing gallery helpers."""
    print("\n[TEST] Testing Gallery Helper Imports...")
    
    # Add the api routes to path
    sys.path.append(str(Path(__file__).parent / "api" / "routes"))
    
    try:
        from gallery_helpers import (
            generate_sophisticated_title,
            categorize_by_filename,
            generate_academic_description,
            COMPREHENSIVE_VISUALIZATION_METADATA
        )
        print("[OK] Successfully imported gallery helper functions")
        
        # Test basic functionality
        test_title = generate_sophisticated_title("consciousness_field.png", "images")
        test_category = categorize_by_filename("quantum_unity.gif")
        test_description = generate_academic_description("phi_spiral.png", "images")
        
        print(f"[TEST] Sample title: {test_title}")
        print(f"[TEST] Sample category: {test_category}")
        print(f"[TEST] Sample description length: {len(test_description)} chars")
        print(f"[TEST] Metadata entries: {len(COMPREHENSIVE_VISUALIZATION_METADATA)}")
        
        return True
        
    except ImportError as e:
        print(f"[ERROR] Failed to import gallery helpers: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Error testing helper functions: {e}")
        return False

def main():
    """Run simple gallery tests."""
    print("Een Unity Mathematics - Gallery System Test")
    print("=" * 50)
    
    tests = [
        ("File Discovery", test_file_discovery),
        ("Gallery Structure", test_gallery_structure),
        ("Helper Imports", test_import_helpers)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            status = "PASSED" if result else "FAILED"
            print(f"[RESULT] {test_name}: {status}")
        except Exception as e:
            print(f"[ERROR] {test_name}: FAILED - {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("[SUCCESS] All tests passed! Gallery system is operational.")
        return True
    else:
        print("[WARNING] Some tests failed. Check output for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)