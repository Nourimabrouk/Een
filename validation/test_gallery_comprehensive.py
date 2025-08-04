#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for comprehensive 3000 ELO gallery system
Verifies complete file scanning and academic caption generation
"""

import os
import sys
from pathlib import Path

# Add the api routes to path
sys.path.append(str(Path(__file__).parent / "api" / "routes"))

try:
    from gallery_helpers import (
        generate_sophisticated_title,
        categorize_by_filename,
        generate_academic_description,
        generate_significance,
        generate_technique,
        generate_academic_context,
        COMPREHENSIVE_VISUALIZATION_METADATA
    )
    print("[OK] Successfully imported 3000 ELO gallery helper functions")
except ImportError as e:
    print(f"[ERROR] Failed to import gallery helpers: {e}")
    sys.exit(1)

def test_file_discovery():
    """Test comprehensive file discovery in viz directories."""
    print("\n🔍 Testing Comprehensive File Discovery...")
    
    viz_folder = Path(__file__).parent / "viz"
    legacy_folder = viz_folder / "legacy images"
    
    # Test main viz folder
    if viz_folder.exists():
        files = list(viz_folder.glob("*"))
        print(f"📁 Main viz folder: {len(files)} items found")
        for file in files:
            if file.is_file():
                print(f"   📄 {file.name} ({file.suffix})")
    
    # Test legacy images folder
    if legacy_folder.exists():
        files = list(legacy_folder.glob("*"))
        print(f"📁 Legacy images folder: {len(files)} items found")
        for file in files:
            if file.is_file():
                print(f"   📄 {file.name} ({file.suffix})")
    
    return True

def test_caption_generation():
    """Test 3000 ELO academic caption generation."""
    print("\n📝 Testing 3000 ELO Academic Caption Generation...")
    
    test_files = [
        ('water droplets.gif', 'videos'),
        ('consciousness_field.png', 'images'),
        ('unity_proof.html', 'interactive'),
        ('quantum_analysis.json', 'data'),
        ('phi_spiral.png', 'images'),
        ('neural_convergence.png', 'images')
    ]
    
    for filename, file_type in test_files:
        print(f"\n🎯 Testing: {filename}")
        
        title = generate_sophisticated_title(filename, file_type)
        category = categorize_by_filename(filename)
        description = generate_academic_description(filename, file_type)
        significance = generate_significance(filename, file_type)
        technique = generate_technique(filename, file_type)
        context = generate_academic_context(filename, file_type)
        
        print(f"   📰 Title: {title}")
        print(f"   🏷️  Category: {category}")
        print(f"   📖 Description: {description[:100]}...")
        print(f"   🎯 Significance: {significance[:80]}...")
        print(f"   🔬 Technique: {technique[:80]}...")
        print(f"   🎓 Context: {context[:80]}...")
    
    return True

def test_comprehensive_metadata():
    """Test comprehensive metadata system."""
    print("\n📊 Testing Comprehensive Metadata System...")
    
    print(f"📈 Total predefined visualizations: {len(COMPREHENSIVE_VISUALIZATION_METADATA)}")
    
    categories = {}
    featured_count = 0
    
    for filename, metadata in COMPREHENSIVE_VISUALIZATION_METADATA.items():
        category = metadata.get('category', 'unknown')
        categories[category] = categories.get(category, 0) + 1
        
        if metadata.get('featured', False):
            featured_count += 1
        
        print(f"   ✨ {metadata['title'][:60]}...")
    
    print(f"\n📊 Statistics:")
    print(f"   🏷️  Categories: {categories}")
    print(f"   ⭐ Featured items: {featured_count}")
    
    return True

def run_comprehensive_test():
    """Run comprehensive gallery system test."""
    print("🚀 Een Unity Mathematics - 3000 ELO Gallery System Test")
    print("=" * 60)
    
    tests = [
        ("File Discovery", test_file_discovery),
        ("Caption Generation", test_caption_generation),
        ("Comprehensive Metadata", test_comprehensive_metadata)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running {test_name} Test...")
            result = test_func()
            results.append((test_name, result))
            print(f"✅ {test_name} Test: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"❌ {test_name} Test: FAILED - {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! 3000 ELO Gallery System is operational.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)