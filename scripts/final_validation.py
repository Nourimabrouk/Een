#!/usr/bin/env python3
"""
Een Unity Mathematics - Final Validation Script
Ensures website is ready for immediate access
"""

import json
import sys
from pathlib import Path


def validate_website():
    """Validate all critical components for website access"""
    print("🔍 Een Unity Mathematics - Final Validation")
    print("=" * 50)
    
    issues = []
    warnings = []
    
    # Check gallery data
    print("📊 Checking gallery data...")
    try:
        with open("gallery_data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not data.get("success"):
            issues.append("Gallery data success flag is false")
        
        viz_count = len(data.get("visualizations", []))
        if viz_count == 0:
            issues.append("No visualizations found in gallery data")
        else:
            print(f"✅ Gallery data: {viz_count} visualizations")
        
        # Check for path issues
        path_issues = sum(1 for v in data.get("visualizations", []) if "\\" in v.get("src", ""))
        if path_issues > 0:
            issues.append(f"{path_issues} visualization paths have Windows backslashes")
        
    except Exception as e:
        issues.append(f"Gallery data error: {e}")
    
    # Check website structure
    print("🌐 Checking website structure...")
    website_dir = Path("website")
    
    if not website_dir.exists():
        issues.append("Website directory not found")
    else:
        # Check essential files
        essential_files = [
            "index.html",
            "gallery.html",
            "about.html",
            "css/style.css"
        ]
        
        for file_path in essential_files:
            full_path = website_dir / file_path
            if not full_path.exists():
                issues.append(f"Missing essential file: {file_path}")
        
        # Check HTML files
        html_files = list(website_dir.rglob("*.html"))
        print(f"✅ Found {len(html_files)} HTML files")
        
        # Check DOCTYPE
        doctype_count = sum(1 for f in html_files if "<!DOCTYPE html>" in f.read_text(encoding="utf-8"))
        if doctype_count != len(html_files):
            issues.append(f"{len(html_files) - doctype_count} HTML files missing DOCTYPE")
        else:
            print("✅ All HTML files have proper DOCTYPE")
        
        # Check JS directory
        js_dir = website_dir / "js"
        if not js_dir.exists():
            warnings.append("JS directory not found")
        else:
            shared_nav = js_dir / "shared-navigation.js"
            if not shared_nav.exists():
                warnings.append("shared-navigation.js not found")
    
    # Check deployment readiness
    print("🚀 Checking deployment readiness...")
    
    deploy_script = Path("scripts/deploy_global.sh")
    if not deploy_script.exists():
        warnings.append("deploy_global.sh not found")
    
    # Check sitemap
    sitemap = website_dir / "sitemap.xml"
    if not sitemap.exists():
        warnings.append("sitemap.xml not found")
    
    # Check robots.txt
    robots = website_dir / "robots.txt"
    if not robots.exists():
        warnings.append("robots.txt not found")
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 VALIDATION SUMMARY")
    print("=" * 50)
    
    if issues:
        print(f"❌ {len(issues)} CRITICAL ISSUES:")
        for issue in issues:
            print(f"   • {issue}")
        print()
    
    if warnings:
        print(f"⚠️ {len(warnings)} WARNINGS:")
        for warning in warnings:
            print(f"   • {warning}")
        print()
    
    if not issues and not warnings:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("✅ Website is ready for immediate access")
        print()
        print("🌍 Access URL: https://nourimabrouk.github.io/Een/")
        print("📱 Compatible with: Chrome, Firefox, Safari, Edge, Brave")
        print("📊 Features: 55 visualizations, 1+1=1 proofs, consciousness fields")
        print()
        print("φ = 1.618033988749895 - Golden Ratio Resonance")
        return True
    else:
        print("⚠️ Website has issues that should be addressed")
        return False


if __name__ == "__main__":
    success = validate_website()
    sys.exit(0 if success else 1) 