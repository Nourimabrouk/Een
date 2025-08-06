#!/usr/bin/env python3
"""
Een Unity Mathematics - Website Integrity Test Suite
Comprehensive testing of website functionality and visualizations
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any


class WebsiteIntegrityTester:
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root).resolve()
        self.test_results = []
        self.issues_found = []
        
    def test_gallery_data_integrity(self) -> Dict[str, Any]:
        """Test gallery data structure and file existence"""
        print("🔍 Testing gallery data integrity...")
        
        gallery_file = self.repo_root / "gallery_data.json"
        if not gallery_file.exists():
            self.issues_found.append("gallery_data.json not found")
            return {"status": False, "issues": ["File not found"]}
        
        try:
            with open(gallery_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            issues = []
            total_viz = len(data.get('visualizations', []))
            
            # Check structure
            if not data.get('success'):
                issues.append("Gallery data success flag is false")
            
            if not data.get('visualizations'):
                issues.append("No visualizations found")
            
            # Check each visualization
            missing_files = 0
            invalid_paths = 0
            
            for viz in data.get('visualizations', []):
                # Check required fields
                required_fields = ['id', 'title', 'src', 'filename']
                for field in required_fields:
                    if field not in viz:
                        issues.append(f"Missing {field} in visualization")
                
                # Check file existence
                if 'src' in viz:
                    file_path = self.repo_root / viz['src']
                    if not file_path.exists():
                        missing_files += 1
                        issues.append(f"Missing file: {viz['src']}")
                    
                    # Check for Windows backslashes
                    if '\\' in viz['src']:
                        invalid_paths += 1
                        issues.append(f"Windows path found: {viz['src']}")
            
            # Check statistics
            if 'statistics' in data:
                stats = data['statistics']
                if stats.get('total', 0) != total_viz:
                    issues.append("Statistics total mismatch")
            
            result = {
                "status": len(issues) == 0,
                "total_visualizations": total_viz,
                "missing_files": missing_files,
                "invalid_paths": invalid_paths,
                "issues": issues
            }
            
            if issues:
                self.issues_found.extend(issues)
            
            print(f"✅ Gallery integrity test: {total_viz} visualizations, {len(issues)} issues")
            return result
            
        except Exception as e:
            error_msg = f"Gallery data test error: {e}"
            self.issues_found.append(error_msg)
            return {"status": False, "issues": [error_msg]}
    
    def test_website_structure(self) -> Dict[str, Any]:
        """Test website file structure and navigation"""
        print("🌐 Testing website structure...")
        
        website_dir = self.repo_root / "website"
        if not website_dir.exists():
            self.issues_found.append("website directory not found")
            return {"status": False, "issues": ["Website directory not found"]}
        
        issues = []
        
        # Check essential files
        essential_files = [
            "index.html",
            "gallery.html",
            "about.html",
            "css/style.css",
            "js/shared-navigation.js"
        ]
        
        for file_path in essential_files:
            full_path = website_dir / file_path
            if not full_path.exists():
                issues.append(f"Missing essential file: {file_path}")
        
        # Check HTML files for basic structure
        html_files = list(website_dir.rglob("*.html"))
        malformed_html = 0
        
        for html_file in html_files:
            try:
                with open(html_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Basic HTML structure checks
                if '<!DOCTYPE html>' not in content:
                    malformed_html += 1
                    issues.append(f"Missing DOCTYPE in {html_file.name}")
                
                if '<html' not in content:
                    malformed_html += 1
                    issues.append(f"Missing html tag in {html_file.name}")
                
                if '<head' not in content:
                    malformed_html += 1
                    issues.append(f"Missing head tag in {html_file.name}")
                
                if '<body' not in content:
                    malformed_html += 1
                    issues.append(f"Missing body tag in {html_file.name}")
                
            except Exception as e:
                issues.append(f"Error reading {html_file.name}: {e}")
        
        result = {
            "status": len(issues) == 0,
            "total_html_files": len(html_files),
            "malformed_html": malformed_html,
            "issues": issues
        }
        
        if issues:
            self.issues_found.extend(issues)
        
        print(f"✅ Website structure test: {len(html_files)} HTML files, {len(issues)} issues")
        return result
    
    def test_visualization_files(self) -> Dict[str, Any]:
        """Test visualization file accessibility"""
        print("🎨 Testing visualization files...")
        
        gallery_file = self.repo_root / "gallery_data.json"
        if not gallery_file.exists():
            return {"status": False, "issues": ["Gallery data not found"]}
        
        try:
            with open(gallery_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            issues = []
            accessible_files = 0
            total_files = 0
            
            for viz in data.get('visualizations', []):
                if 'src' in viz:
                    total_files += 1
                    file_path = self.repo_root / viz['src']
                    
                    if file_path.exists():
                        accessible_files += 1
                        
                        # Check file size
                        try:
                            file_size = file_path.stat().st_size
                            if file_size == 0:
                                issues.append(f"Empty file: {viz['src']}")
                        except Exception as e:
                            issues.append(f"Error checking {viz['src']}: {e}")
                    else:
                        issues.append(f"Missing file: {viz['src']}")
            
            result = {
                "status": len(issues) == 0,
                "total_files": total_files,
                "accessible_files": accessible_files,
                "accessibility_rate": accessible_files / total_files if total_files > 0 else 0,
                "issues": issues
            }
            
            if issues:
                self.issues_found.extend(issues)
            
            print(f"✅ Visualization test: {accessible_files}/{total_files} files accessible")
            return result
            
        except Exception as e:
            error_msg = f"Visualization test error: {e}"
            self.issues_found.append(error_msg)
            return {"status": False, "issues": [error_msg]}
    
    def test_deployment_readiness(self) -> Dict[str, Any]:
        """Test deployment script and configuration"""
        print("🚀 Testing deployment readiness...")
        
        issues = []
        
        # Check deployment script
        deploy_script = self.repo_root / "scripts" / "deploy_global.sh"
        if not deploy_script.exists():
            issues.append("deploy_global.sh not found")
        else:
            try:
                with open(deploy_script, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for common issues
                if '${Nourimabrouk}.github.io/${Een}/' in content:
                    issues.append("Deployment script has incorrect URL variables")
                
                if '#!/bin/bash' not in content:
                    issues.append("Deployment script missing shebang")
                
            except Exception as e:
                issues.append(f"Error reading deployment script: {e}")
        
        # Check sitemap
        sitemap_file = self.repo_root / "website" / "sitemap.xml"
        if not sitemap_file.exists():
            issues.append("sitemap.xml not found")
        
        # Check robots.txt
        robots_file = self.repo_root / "website" / "robots.txt"
        if not robots_file.exists():
            issues.append("robots.txt not found")
        
        result = {
            "status": len(issues) == 0,
            "issues": issues
        }
        
        if issues:
            self.issues_found.extend(issues)
        
        print(f"✅ Deployment test: {len(issues)} issues found")
        return result
    
    def test_mobile_compatibility(self) -> Dict[str, Any]:
        """Test mobile compatibility features"""
        print("📱 Testing mobile compatibility...")
        
        website_dir = self.repo_root / "website"
        if not website_dir.exists():
            return {"status": False, "issues": ["Website directory not found"]}
        
        issues = []
        mobile_optimized = 0
        total_files = 0
        
        html_files = list(website_dir.rglob("*.html"))
        
        for html_file in html_files:
            total_files += 1
            try:
                with open(html_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check viewport meta tag
                if 'viewport' in content:
                    mobile_optimized += 1
                else:
                    issues.append(f"Missing viewport meta tag in {html_file.name}")
                
                # Check touch optimizations
                if 'touch-action' in content:
                    mobile_optimized += 1
                else:
                    issues.append(f"Missing touch optimizations in {html_file.name}")
                
            except Exception as e:
                issues.append(f"Error reading {html_file.name}: {e}")
        
        result = {
            "status": len(issues) == 0,
            "total_files": total_files,
            "mobile_optimized": mobile_optimized,
            "mobile_rate": mobile_optimized / (total_files * 2) if total_files > 0 else 0,
            "issues": issues
        }
        
        if issues:
            self.issues_found.extend(issues)
        
        print(f"✅ Mobile test: {mobile_optimized}/{total_files * 2} optimizations applied")
        return result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all integrity tests"""
        print("🌟 Een Unity Mathematics - Website Integrity Test Suite")
        print("=" * 60)
        
        tests = {
            "gallery_integrity": self.test_gallery_data_integrity(),
            "website_structure": self.test_website_structure(),
            "visualization_files": self.test_visualization_files(),
            "deployment_readiness": self.test_deployment_readiness(),
            "mobile_compatibility": self.test_mobile_compatibility()
        }
        
        # Calculate overall status
        overall_status = all(test.get('status', False) for test in tests.values())
        
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        for test_name, result in tests.items():
            status = "✅ PASS" if result.get('status', False) else "❌ FAIL"
            print(f"{status} {test_name}")
            
            if not result.get('status', False) and result.get('issues'):
                for issue in result['issues'][:3]:  # Show first 3 issues
                    print(f"    • {issue}")
                if len(result['issues']) > 3:
                    print(f"    ... and {len(result['issues']) - 3} more issues")
        
        print(f"\n🎯 Overall Status: {'✅ PASS' if overall_status else '❌ FAIL'}")
        print(f"📋 Total Issues Found: {len(self.issues_found)}")
        
        if self.issues_found:
            print(f"\n⚠️ Critical Issues:")
            for issue in self.issues_found[:10]:
                print(f"  • {issue}")
            if len(self.issues_found) > 10:
                print(f"  ... and {len(self.issues_found) - 10} more")
        else:
            print("\n🎉 All tests passed! Website is ready for deployment.")
        
        return {
            "overall_status": overall_status,
            "test_results": tests,
            "total_issues": len(self.issues_found),
            "issues": self.issues_found
        }


def main():
    """Main execution function"""
    tester = WebsiteIntegrityTester()
    results = tester.run_all_tests()
    
    # Return appropriate exit code
    if results['overall_status']:
        sys.exit(0)  # All tests passed
    else:
        sys.exit(1)  # Some tests failed


if __name__ == "__main__":
    main() 