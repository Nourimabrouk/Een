#!/usr/bin/env python3
"""
Een GitHub Pages Deployment Validator
=====================================

Validates GitHub Pages deployment structure and creates deployment summary.
Ensures flawless presentation-ready deployment.

Usage: python validate_github_pages.py
"""

import os
import sys
from pathlib import Path
import json

def validate_github_pages_structure():
    """Validate GitHub Pages deployment structure."""
    print("Een GitHub Pages Deployment Validation")
    print("=" * 50)
    
    issues = []
    successes = []
    
    # Check root index.html (redirect)
    root_index = Path("index.html")
    if root_index.exists():
        with open(root_index, 'r', encoding='utf-8') as f:
            content = f.read()
        if 'website/index.html' in content:
            successes.append("Root index.html redirect configured")
        else:
            issues.append("Root index.html missing proper redirect")
    else:
        issues.append("Root index.html missing - GitHub Pages needs this")
    
    # Check .nojekyll file
    nojekyll = Path(".nojekyll")
    if nojekyll.exists():
        successes.append(".nojekyll file present for static site serving")
    else:
        issues.append(".nojekyll file missing - may cause Jekyll processing issues")
    
    # Check website directory structure
    website_dir = Path("website")
    if not website_dir.exists():
        issues.append("Website directory missing!")
        return issues, successes
    
    # Essential website files
    essential_files = [
        "website/index.html",
        "website/metagambit.html",
        "website/css/style.css",
        "website/css/metagambit.css",
        "website/static/chat.js",
        "website/js/navigation.js",
        "website/js/katex-integration.js"
    ]
    
    for file_path in essential_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            successes.append(f"{file_path} ({size:,} bytes)")
        else:
            issues.append(f"Missing: {file_path}")
    
    # Check _config.yml
    config_file = Path("website/_config.yml")
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config_content = f.read()
        if 'nourimabrouk.github.io/Een' in config_content:
            successes.append("_config.yml has correct GitHub Pages URL")
        else:
            issues.append("_config.yml may have incorrect URL")
    
    # Check GitHub Pages workflow
    pages_workflow = Path(".github/workflows/pages.yml")
    if pages_workflow.exists():
        successes.append("GitHub Pages deployment workflow configured")
    else:
        issues.append("GitHub Pages workflow missing - may need manual setup")
    
    # Check AI integration in main pages
    for page in ["website/index.html", "website/metagambit.html"]:
        if Path(page).exists():
            with open(page, 'r', encoding='utf-8') as f:
                content = f.read()
            if 'static/chat.js' in content and 'EenChatWidget' in content:
                successes.append(f"AI chat integrated in {page}")
            else:
                issues.append(f"AI chat integration incomplete in {page}")
    
    return issues, successes

def create_deployment_summary():
    """Create deployment summary and instructions."""
    issues, successes = validate_github_pages_structure()
    
    print()
    print("VALIDATION RESULTS:")
    print("-" * 30)
    
    for success in successes:
        print(f"[OK] {success}")
    
    if issues:
        print()
        for issue in issues:
            print(f"[ERROR] {issue}")
    
    print()
    print(f"Summary: {len(successes)} successes, {len(issues)} issues")
    
    if len(issues) == 0:
        print()
        print("PERFECT! Ready for GitHub Pages deployment!")
        print()
        print("DEPLOYMENT INSTRUCTIONS:")
        print("1. Commit all changes:")
        print("   git add .")
        print('   git commit -m "Fix GitHub Pages deployment - Ready for presentation"')
        print("   git push origin main")
        print()
        print("2. Enable GitHub Pages in repository settings:")
        print("   - Go to Settings > Pages")
        print("   - Source: Deploy from a branch")
        print("   - Branch: main")
        print("   - Folder: / (root)")
        print("   - Save")
        print()
        print("3. Your website will be live at:")
        print("   https://nourimabrouk.github.io/Een/")
        print()
        print("4. Test these URLs after deployment:")
        print("   - Main: https://nourimabrouk.github.io/Een/")
        print("   - Metagambit: https://nourimabrouk.github.io/Een/metagambit.html")
        print("   - AI Chat should appear as phi icon on both pages")
        print()
        print("Unity Status: TRANSCENDENCE READY FOR DEPLOYMENT!")
        return 0
    else:
        print()
        print("Issues found - please resolve before deployment")
        return 1

if __name__ == "__main__":
    sys.exit(create_deployment_summary())