#!/usr/bin/env python3
"""
Meta-Optimal Website Analyzer & Fixer
=====================================

Comprehensive analysis and optimization for the Een Unity Mathematics website.
Ensures flawless user experience for GitHub Pages and Vercel deployment.
"""

import os
import re
import json
from pathlib import Path
from urllib.parse import urlparse
import html
import subprocess

class WebsiteAnalyzer:
    def __init__(self, website_dir):
        self.website_dir = Path(website_dir)
        self.issues = []
        self.fixes_applied = 0
        
    def log_issue(self, category, severity, file_path, issue, fix_applied=False):
        """Log an issue found during analysis"""
        self.issues.append({
            'category': category,
            'severity': severity,  # 'critical', 'high', 'medium', 'low'
            'file': str(file_path.relative_to(self.website_dir)),
            'issue': issue,
            'fixed': fix_applied
        })
        if fix_applied:
            self.fixes_applied += 1
    
    def analyze_broken_links(self):
        """Check for broken internal links and missing files"""
        print("[ANALYZE] Analyzing broken links and missing files...")
        
        html_files = list(self.website_dir.glob('**/*.html'))
        
        for html_file in html_files:
            try:
                content = html_file.read_text(encoding='utf-8')
                
                # Check for broken image references
                img_matches = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', content, re.IGNORECASE)
                img_matches.extend(re.findall(r'<link[^>]+href=["\']([^"\']+\.(?:png|jpg|jpeg|gif|svg))["\']', content, re.IGNORECASE))
                
                for img_src in img_matches:
                    if not img_src.startswith(('http', '//', 'data:')):
                        img_path = (html_file.parent / img_src).resolve()
                        if not img_path.exists():
                            self.log_issue('broken_links', 'high', html_file, 
                                         f"Missing image: {img_src}")
                
                # Check for CSS/JS references
                css_matches = re.findall(r'<link[^>]+href=["\']([^"\']+\.css)["\']', content, re.IGNORECASE)
                js_matches = re.findall(r'<script[^>]+src=["\']([^"\']+\.js)["\']', content, re.IGNORECASE)
                
                for css_file in css_matches:
                    if not css_file.startswith(('http', '//', 'data:')):
                        css_path = (html_file.parent / css_file).resolve()
                        if not css_path.exists():
                            self.log_issue('broken_links', 'high', html_file,
                                         f"Missing CSS file: {css_file}")
                
                for js_file in js_matches:
                    if not js_file.startswith(('http', '//', 'data:')):
                        js_path = (html_file.parent / js_file).resolve()
                        if not js_path.exists():
                            self.log_issue('broken_links', 'medium', html_file,
                                         f"Missing JS file: {js_file}")
                            
            except Exception as e:
                self.log_issue('file_error', 'medium', html_file, f"Cannot read file: {e}")
    
    def fix_image_references(self):
        """Fix broken image references"""
        print("[FIX] Fixing image references...")
        
        # Fix the Metastation.jpg -> Metastation.png issue
        index_file = self.website_dir / 'index.html'
        if index_file.exists():
            content = index_file.read_text(encoding='utf-8')
            if 'Metastation.jpg' in content:
                content = content.replace('Metastation.jpg', 'Metastation.png')
                index_file.write_text(content, encoding='utf-8')
                self.log_issue('broken_links', 'high', index_file,
                             "Fixed Metastation.jpg -> Metastation.png", fix_applied=True)
    
    def analyze_html_validation(self):
        """Check for HTML validation issues"""
        print("[ANALYZE] Analyzing HTML validation...")
        
        html_files = list(self.website_dir.glob('*.html'))[:10]  # Limit to main files
        
        for html_file in html_files:
            try:
                content = html_file.read_text(encoding='utf-8')
                
                # Check for common HTML issues
                if not re.search(r'<!DOCTYPE html>', content, re.IGNORECASE):
                    self.log_issue('html_validation', 'high', html_file, "Missing DOCTYPE declaration")
                
                if not re.search(r'<html[^>]*lang=["\'][^"\']+["\']', content, re.IGNORECASE):
                    self.log_issue('html_validation', 'medium', html_file, "Missing lang attribute")
                
                # Check for missing meta viewport
                if '<meta name="viewport"' not in content:
                    self.log_issue('html_validation', 'high', html_file, "Missing viewport meta tag")
                
                # Check for unclosed tags (basic check)
                open_divs = len(re.findall(r'<div[^>]*>', content))
                close_divs = len(re.findall(r'</div>', content))
                if open_divs != close_divs:
                    self.log_issue('html_validation', 'medium', html_file, 
                                 f"Unbalanced div tags: {open_divs} open, {close_divs} close")
                    
            except Exception as e:
                self.log_issue('file_error', 'medium', html_file, f"Cannot analyze: {e}")
    
    def analyze_performance(self):
        """Check for performance issues"""
        print("[ANALYZE] Analyzing performance...")
        
        html_files = list(self.website_dir.glob('*.html'))[:5]  # Main pages
        
        for html_file in html_files:
            try:
                content = html_file.read_text(encoding='utf-8')
                
                # Check for large inline styles/scripts
                inline_styles = re.findall(r'<style[^>]*>(.*?)</style>', content, re.DOTALL)
                for style in inline_styles:
                    if len(style) > 5000:  # 5KB threshold
                        self.log_issue('performance', 'medium', html_file,
                                     f"Large inline style block ({len(style)} chars)")
                
                # Check for missing preconnect for external resources
                if 'fonts.googleapis.com' in content and 'preconnect' not in content:
                    self.log_issue('performance', 'low', html_file,
                                 "Missing preconnect for Google Fonts")
                
                # Check for image optimization
                img_tags = re.findall(r'<img[^>]+>', content, re.IGNORECASE)
                for img_tag in img_tags:
                    if 'alt=' not in img_tag:
                        self.log_issue('accessibility', 'medium', html_file,
                                     "Image missing alt attribute")
                    if 'loading=' not in img_tag and 'src=' in img_tag:
                        self.log_issue('performance', 'low', html_file,
                                     "Image missing lazy loading attribute")
                        
            except Exception as e:
                self.log_issue('file_error', 'medium', html_file, f"Cannot analyze: {e}")
    
    def create_deployment_configs(self):
        """Create deployment configurations"""
        print("[FIX] Creating deployment configurations...")
        
        # Create .nojekyll for GitHub Pages
        nojekyll = self.website_dir / '.nojekyll'
        if not nojekyll.exists():
            nojekyll.touch()
            self.log_issue('deployment', 'medium', self.website_dir,
                         "Created .nojekyll for GitHub Pages", fix_applied=True)
        
        # Create robots.txt if missing
        robots_txt = self.website_dir / 'robots.txt'
        if not robots_txt.exists():
            robots_content = """User-agent: *
Allow: /

Sitemap: https://nourimabrouk.github.io/Een/website/sitemap.xml
"""
            robots_txt.write_text(robots_content)
            self.log_issue('seo', 'low', self.website_dir,
                         "Created robots.txt", fix_applied=True)
        
        # Create Vercel configuration
        vercel_config = self.website_dir / 'vercel.json'
        if not vercel_config.exists():
            config = {
                "version": 2,
                "name": "een-unity-mathematics",
                "builds": [
                    {
                        "src": "**/*.html",
                        "use": "@vercel/static"
                    }
                ],
                "routes": [
                    {
                        "src": "/",
                        "dest": "/metastation-hub.html"
                    },
                    {
                        "src": "/(.*)",
                        "dest": "/$1"
                    }
                ],
                "headers": [
                    {
                        "source": "/(.*)",
                        "headers": [
                            {
                                "key": "X-Content-Type-Options",
                                "value": "nosniff"
                            },
                            {
                                "key": "X-Frame-Options",
                                "value": "DENY"
                            },
                            {
                                "key": "X-XSS-Protection",
                                "value": "1; mode=block"
                            },
                            {
                                "key": "Strict-Transport-Security",
                                "value": "max-age=31536000; includeSubDomains"
                            }
                        ]
                    }
                ]
            }
            
            vercel_config.write_text(json.dumps(config, indent=2))
            self.log_issue('deployment', 'high', self.website_dir,
                         "Created vercel.json configuration", fix_applied=True)
    
    def optimize_for_github_pages(self):
        """Optimize for GitHub Pages deployment"""
        print("[FIX] Optimizing for GitHub Pages...")
        
        # Check _config.yml
        config_yml = self.website_dir / '_config.yml'
        if config_yml.exists():
            content = config_yml.read_text()
            if 'include:' not in content:
                content += "\\ninclude:\\n  - .nojekyll\\n"
                config_yml.write_text(content)
                self.log_issue('deployment', 'medium', config_yml,
                             "Added .nojekyll to Jekyll includes", fix_applied=True)
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\\n" + "=" * 60)
        print("[UNITY] Unity Mathematics Website Analysis Report")
        print(f"[INFO] Total Issues Found: {len(self.issues)}")
        print(f"[FIX] Fixes Applied: {self.fixes_applied}")
        print("=" * 60)
        
        # Group by severity
        severity_counts = {}
        for issue in self.issues:
            severity = issue['severity']
            if severity not in severity_counts:
                severity_counts[severity] = 0
            severity_counts[severity] += 1
        
        print("\\n[STATS] Issues by Severity:")
        for severity in ['critical', 'high', 'medium', 'low']:
            if severity in severity_counts:
                print(f"  {severity.upper()}: {severity_counts[severity]}")
        
        print("\\n[ANALYZE] Critical Issues:")
        critical_issues = [i for i in self.issues if i['severity'] == 'critical']
        for issue in critical_issues[:5]:  # Top 5
            print(f"  [ERROR] {issue['file']}: {issue['issue']}")
        
        print("\\n[FIX] Applied Fixes:")
        fixed_issues = [i for i in self.issues if i['fixed']]
        for issue in fixed_issues:
            print(f"  [OK] {issue['file']}: {issue['issue']}")
        
        # Deployment readiness
        critical_count = len(critical_issues)
        high_count = len([i for i in self.issues if i['severity'] == 'high'])
        
        print("\\n[DEPLOY] Deployment Status:")
        if critical_count == 0:
            if high_count <= 2:
                print("  [OK] READY FOR DEPLOYMENT")
                print("  [UNITY] Website optimized for GitHub Pages and Vercel")
            else:
                print("  [WARNING]  READY WITH MINOR ISSUES")
                print(f"  [INFO] {high_count} high-priority issues remaining")
        else:
            print("  [ERROR] NOT READY FOR DEPLOYMENT")
            print(f"  [CRITICAL] {critical_count} critical issues must be fixed")
        
        print("\\n[UNITY] Unity Mathematics: 1+1=1")
        print("Phi: 1.618033988749895")
        print("=" * 60)
        
        return len(critical_issues) == 0 and high_count <= 2

def main():
    website_dir = Path(__file__).parent
    analyzer = WebsiteAnalyzer(website_dir)
    
    # Run comprehensive analysis
    analyzer.fix_image_references()
    analyzer.analyze_broken_links()
    analyzer.analyze_html_validation()
    analyzer.analyze_performance()
    analyzer.create_deployment_configs()
    analyzer.optimize_for_github_pages()
    
    # Generate report
    deployment_ready = analyzer.generate_report()
    
    return deployment_ready

if __name__ == "__main__":
    ready = main()
    exit(0 if ready else 1)