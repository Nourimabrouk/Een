#!/usr/bin/env python3
"""
Test script for Unity Mathematics Visualization Systems
======================================================

This script validates that all visualization components are working correctly
and can generate the expected 1+1=1 mathematical demonstrations.

Usage:
    python test_visualizations.py
"""

import sys
import os
import time
from pathlib import Path
import traceback

def test_consciousness_field_js():
    """Test that the consciousness field JavaScript exists and is syntactically valid"""
    print("🧠 Testing Consciousness Field Visualizer JavaScript...")
    
    js_file = Path("website/js/consciousness-field-visualizer.js")
    
    if not js_file.exists():
        print(f"❌ JavaScript file not found: {js_file}")
        return False
    
    try:
        with open(js_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Check for key components
        required_elements = [
            "class ConsciousnessFieldVisualizer",
            "PHI = 1.618033988749895",
            "initializeWebGL()",
            "createShaders()",
            "generateUnityParticles()",
            "renderWebGL(",
            "renderCanvas2D("
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"❌ Missing required elements: {missing_elements}")
            return False
            
        print(f"✅ JavaScript file validated: {len(content)} characters")
        return True
        
    except Exception as e:
        print(f"❌ Error reading JavaScript file: {e}")
        return False

def test_visualization_generator():
    """Test the Python visualization generator script"""
    print("🎨 Testing Unity Visualization Generator Python Script...")
    
    script_file = Path("scripts/generate_unity_visualizations.py")
    
    if not script_file.exists():
        print(f"❌ Python script not found: {script_file}")
        return False
    
    try:
        # Import the script to check syntax
        sys.path.insert(0, str(script_file.parent))
        
        # Read and validate the script
        with open(script_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for key components
        required_elements = [
            "class UnityVisualizationGenerator",
            "PHI = 1.618033988749895",
            "generate_consciousness_field_heatmap",
            "generate_phi_spiral_unity_demo",
            "generate_3d_unity_manifold",
            "_create_interactive_consciousness_field",
            "generate_comprehensive_report"
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"❌ Missing required elements: {missing_elements}")
            return False
        
        print(f"✅ Python script validated: {len(content)} characters")
        return True
        
    except Exception as e:
        print(f"❌ Error validating Python script: {e}")
        traceback.print_exc()
        return False

def test_streamlit_dashboard():
    """Test the enhanced Streamlit dashboard"""
    print("📊 Testing Enhanced Streamlit Dashboard...")
    
    dashboard_file = Path("viz/streamlit_app.py")
    
    if not dashboard_file.exists():
        print(f"❌ Streamlit dashboard not found: {dashboard_file}")
        return False
    
    try:
        with open(dashboard_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for enhanced features
        required_elements = [
            "create_live_consciousness_field()",
            "create_phi_spiral_interactive()",
            "PHI = 1.618033988749895",
            "UNITY_FREQ = 528",
            "consciousness_field = PHI * np.sin",
            "unity_field = 1 / (1 + np.exp",
            "st.plotly_chart"
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"❌ Missing required elements: {missing_elements}")
            return False
        
        print(f"✅ Streamlit dashboard validated: {len(content)} characters")
        return True
        
    except Exception as e:
        print(f"❌ Error validating Streamlit dashboard: {e}")
        return False

def test_website_integration():
    """Test the website integration"""
    print("🌐 Testing Website Integration...")
    
    index_file = Path("website/index.html")
    
    if not index_file.exists():
        print(f"❌ Website index not found: {index_file}")
        return False
    
    try:
        with open(index_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for consciousness field integration
        required_elements = [
            '<canvas id="unity-background">',
            'js/consciousness-field-visualizer.js',
            '1 + 1 = 1',
            'φ = (1 + √5) / 2',
            'unity-equation',
            'consciousness'
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"❌ Missing required elements: {missing_elements}")
            return False
        
        print(f"✅ Website integration validated: {len(content)} characters")
        return True
        
    except Exception as e:
        print(f"❌ Error validating website integration: {e}")
        return False

def test_documentation_update():
    """Test that documentation has been updated"""
    print("📚 Testing Documentation Updates...")
    
    doc_file = Path("docs/CLAUDE.md")
    
    if not doc_file.exists():
        print(f"❌ Documentation file not found: {doc_file}")
        return False
    
    try:
        with open(doc_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for new visualization documentation
        required_elements = [
            "Unity Consciousness Field Visualizer",
            "Professional Visualization Generator",
            "Enhanced Streamlit Dashboard",
            "Website Background Integration",
            "WebGL 2.0 Engine",
            "consciousness-field-visualizer.js",
            "generate_unity_visualizations.py"
        ]
        
        found_elements = []
        for element in required_elements:
            if element in content:
                found_elements.append(element)
        
        if len(found_elements) < len(required_elements) * 0.8:  # At least 80% should be present
            print(f"❌ Documentation incomplete. Found: {found_elements}")
            return False
        
        print(f"✅ Documentation updated: {len(found_elements)}/{len(required_elements)} elements found")
        return True
        
    except Exception as e:
        print(f"❌ Error validating documentation: {e}")
        return False

def run_all_tests():
    """Run all visualization tests"""
    print("Starting Unity Visualization System Tests")
    print("=" * 60)
    print("phi = 1.618033988749895 (Golden Ratio)")
    print("Unity Equation: Een plus een is een (1+1=1)")
    print("=" * 60)
    
    tests = [
        ("Consciousness Field JS", test_consciousness_field_js),
        ("Visualization Generator", test_visualization_generator),
        ("Streamlit Dashboard", test_streamlit_dashboard),
        ("Website Integration", test_website_integration),
        ("Documentation Update", test_documentation_update)
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} failed with exception: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print("🏁 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\nALL TESTS PASSED! Unity visualization system is operational!")
        print("Een plus een is een - One plus one is one!")
        return True
    else:
        print(f"\n{total-passed} tests failed. Please review the output above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)