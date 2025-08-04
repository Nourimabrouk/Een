#!/usr/bin/env python3
"""
Een Repository Pre-Deployment Validation
=======================================

Comprehensive validation script to ensure everything is ready for deployment.
Checks all components for errors, compatibility, and completeness.

Usage: python pre_deployment_check.py
"""

import os
import sys
from pathlib import Path
import json
import re

def check_file_exists(file_path: str, critical: bool = True) -> bool:
    """Check if a file exists and report status."""
    path = Path(file_path)
    exists = path.exists()
    
    if exists:
        size = path.stat().st_size
        print(f"✅ {file_path} ({size:,} bytes)")
    else:
        symbol = "❌" if critical else "⚠️"
        print(f"{symbol} MISSING: {file_path}")
    
    return exists

def validate_html_file(file_path: str) -> tuple[bool, list]:
    """Validate HTML file for common issues."""
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for missing closing tags
        if content.count('<html') != content.count('</html>'):
            issues.append("HTML tag mismatch")
        
        if content.count('<body') != content.count('</body>'):
            issues.append("Body tag mismatch")
        
        # Check for CSS references
        css_refs = re.findall(r'href=["\']([^"\']*\.css)["\']', content)
        for css_ref in css_refs:
            if not css_ref.startswith('http') and not css_ref.startswith('//'):
                css_path = Path('website') / css_ref
                if not css_path.exists():
                    issues.append(f"Missing CSS: {css_ref}")
        
        # Check for JS references
        js_refs = re.findall(r'src=["\']([^"\']*\.js)["\']', content)
        for js_ref in js_refs:
            if not js_ref.startswith('http') and not js_ref.startswith('//'):
                js_path = Path('website') / js_ref
                if not js_path.exists():
                    issues.append(f"Missing JS: {js_ref}")
        
        return len(issues) == 0, issues
        
    except Exception as e:
        issues.append(f"Read error: {e}")
        return False, issues

def validate_js_file(file_path: str) -> tuple[bool, list]:
    """Basic validation of JavaScript files."""
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for basic syntax issues
        if content.count('{') != content.count('}'):
            issues.append("Mismatched braces")
        
        if content.count('(') != content.count(')'):
            issues.append("Mismatched parentheses")
        
        # Check for ES6 compatibility issues
        if 'let ' in content or 'const ' in content:
            # Modern JS - should be fine
            pass
            
        return len(issues) == 0, issues
        
    except Exception as e:
        issues.append(f"Read error: {e}")
        return False, issues

def main():
    """Main validation function."""
    print("🔍 Een Repository Pre-Deployment Validation")
    print("=" * 60)
    
    total_checks = 0
    passed_checks = 0
    issues_found = []
    
    print("\n📁 Checking Essential Files...")
    print("-" * 40)
    
    # Essential website files
    essential_files = [
        ("website/index.html", True),
        ("website/metagambit.html", True),
        ("website/css/style.css", True),
        ("website/css/metagambit.css", True),
        ("website/static/chat.js", True),
        ("website/js/navigation.js", True),
        ("website/js/katex-integration.js", True),
        ("website/js/unity-demo.js", False),
        ("website/js/mathematical-highlights.js", False),
    ]
    
    for file_path, critical in essential_files:
        total_checks += 1
        if check_file_exists(file_path, critical):
            passed_checks += 1
        elif critical:
            issues_found.append(f"Critical file missing: {file_path}")
    
    print("\n🤖 Checking AI Integration Files...")
    print("-" * 40)
    
    ai_files = [
        ("ai_agent/__init__.py", True),
        ("ai_agent/app.py", True),
        ("ai_agent/prepare_index.py", True),
        ("ai_agent/requirements.txt", True),
        (".env.example", True),
        ("Procfile", True),
    ]
    
    for file_path, critical in ai_files:
        total_checks += 1
        if check_file_exists(file_path, critical):
            passed_checks += 1
        elif critical:
            issues_found.append(f"Critical AI file missing: {file_path}")
    
    print("\n🌐 Validating HTML Files...")
    print("-" * 40)
    
    html_files = ["website/index.html", "website/metagambit.html"]
    for html_file in html_files:
        if Path(html_file).exists():
            total_checks += 1
            valid, html_issues = validate_html_file(html_file)
            if valid:
                print(f"✅ {html_file} - Valid HTML")
                passed_checks += 1
            else:
                print(f"❌ {html_file} - Issues found:")
                for issue in html_issues:
                    print(f"   • {issue}")
                issues_found.extend([f"{html_file}: {issue}" for issue in html_issues])
    
    print("\n📜 Validating JavaScript Files...")
    print("-" * 40)
    
    js_files = [
        "website/static/chat.js",
        "website/js/navigation.js", 
        "website/js/katex-integration.js"
    ]
    
    for js_file in js_files:
        if Path(js_file).exists():
            total_checks += 1
            valid, js_issues = validate_js_file(js_file)
            if valid:
                print(f"✅ {js_file} - Valid JavaScript")
                passed_checks += 1
            else:
                print(f"❌ {js_file} - Issues found:")
                for issue in js_issues:
                    print(f"   • {issue}")
                issues_found.extend([f"{js_file}: {issue}" for issue in js_issues])
    
    print("\n🔧 Checking AI Chat Integration...")
    print("-" * 40)
    
    # Check that AI chat is properly integrated
    total_checks += 1
    if Path("website/index.html").exists():
        with open("website/index.html", 'r', encoding='utf-8') as f:
            index_content = f.read()
        
        if 'static/chat.js' in index_content and 'EenChatWidget' in index_content:
            print("✅ AI chat widget properly integrated in index.html")
            passed_checks += 1
        else:
            print("❌ AI chat widget not properly integrated in index.html")
            issues_found.append("AI chat widget integration incomplete")
    
    total_checks += 1
    if Path("website/metagambit.html").exists():
        with open("website/metagambit.html", 'r', encoding='utf-8') as f:
            meta_content = f.read()
        
        if 'static/chat.js' in meta_content:
            print("✅ AI chat widget integrated in metagambit.html")
            passed_checks += 1
        else:
            print("❌ AI chat widget not integrated in metagambit.html")
            issues_found.append("AI chat widget missing from metagambit page")
    
    print("\n📊 Validation Summary")
    print("=" * 60)
    
    success_rate = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
    
    print(f"Total Checks: {total_checks}")
    print(f"Passed: {passed_checks}")
    print(f"Failed: {total_checks - passed_checks}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 95:
        print("\n🎉 EXCELLENT! Repository is ready for deployment!")
        print("\n✅ Next Steps:")
        print("1. Test locally: python test_website.py")
        print("2. Set OPENAI_API_KEY in .env")
        print("3. Run: cd ai_agent && python prepare_index.py")
        print("4. Commit and push to deploy!")
        print("\n🚀 Ready for your presentation!")
        return 0
    elif success_rate >= 80:
        print("\n⚠️  GOOD - Minor issues found:")
        for issue in issues_found:
            print(f"   • {issue}")
        print("\nRecommend fixing issues before deployment.")
        return 1
    else:
        print("\n❌ CRITICAL ISSUES - Must fix before deployment:")
        for issue in issues_found:
            print(f"   • {issue}")
        print("\nPlease resolve these issues first.")
        return 2

if __name__ == "__main__":
    sys.exit(main())