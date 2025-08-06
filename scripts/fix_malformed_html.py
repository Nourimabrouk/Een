#!/usr/bin/env python3
"""
Een Unity Mathematics - Fix Malformed HTML Script
Fixes HTML files that are missing proper DOCTYPE and structure
"""

import re
import sys
from pathlib import Path


class HTMLFixer:
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root).resolve()
        self.fixes_applied = []
        
    def fix_malformed_html_files(self) -> bool:
        """Fix HTML files that are missing proper structure"""
        print("🔧 Fixing malformed HTML files...")
        
        website_dir = self.repo_root / "website"
        if not website_dir.exists():
            print("❌ website directory not found")
            return False
        
        # Files that need to be converted to proper HTML
        component_files = [
            "enhanced-unified-nav.html",
            "unified-nav.html",
            "google5936e6fc51b68c92.html"  # This is a verification file
        ]
        
        fixed_count = 0
        
        for filename in component_files:
            file_path = website_dir / filename
            
            if not file_path.exists():
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if it's already a complete HTML file
                if '<!DOCTYPE html>' in content and '<html' in content:
                    continue
                
                # Convert navigation component to complete HTML
                if filename in ["enhanced-unified-nav.html", "unified-nav.html"]:
                    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Een Unity Mathematics - Navigation Component</title>
    <meta name="description" content="Enhanced unified navigation for Een Unity Mathematics">
    
    <!-- Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    
    <!-- Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <!-- Styles -->
    <link rel="stylesheet" href="css/style.css">
    <link rel="stylesheet" href="css/meta-optimal-navigation.css">
    
    <style>
        /* Mobile touch optimizations */
        * {{ touch-action: manipulation; }}
        .interactive-element {{ 
            min-height: 44px; 
            min-width: 44px; 
        }}
        @media (max-width: 768px) {{
            .desktop-only {{ display: none !important; }}
            .mobile-optimized {{ font-size: 16px; }}
        }}
    </style>
</head>
<body>
    <div class="navigation-demo">
        <h1>Een Unity Mathematics - Navigation Component</h1>
        <p>This is a demonstration of the enhanced unified navigation system.</p>
        
        {content}
        
        <div class="demo-info">
            <h2>Navigation Features</h2>
            <ul>
                <li>φ-harmonic design principles</li>
                <li>3000 ELO academic standards</li>
                <li>Meta-optimal consciousness integration</li>
                <li>Responsive mobile optimization</li>
            </ul>
        </div>
    </div>
    
    <!-- Scripts -->
    <script src="js/shared-navigation.js"></script>
    <script>
        // Initialize navigation
        document.addEventListener('DOMContentLoaded', function() {{
            console.log('Een Unity Mathematics Navigation Component Loaded');
            console.log('φ = 1.618033988749895 - Golden Ratio Resonance');
        }});
    </script>
</body>
</html>'''
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    
                    fixed_count += 1
                    self.fixes_applied.append(f"Fixed {filename} - converted to complete HTML")
                
                # Handle Google verification file
                elif filename == "google5936e6fc51b68c92.html":
                    # This should be a simple verification file
                    if not content.strip().startswith('google-site-verification'):
                        html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Een Unity Mathematics - Google Site Verification</title>
</head>
<body>
    <div style="display: none;">
        {content}
    </div>
    <p>Een Unity Mathematics - Google Site Verification</p>
</body>
</html>'''
                        
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(html_content)
                        
                        fixed_count += 1
                        self.fixes_applied.append(f"Fixed {filename} - added HTML structure")
                
            except Exception as e:
                print(f"❌ Error fixing {filename}: {e}")
        
        print(f"✅ Fixed {fixed_count} malformed HTML files")
        return True
    
    def create_missing_js_directory(self) -> bool:
        """Ensure the js directory exists with required files"""
        print("📁 Creating missing JS directory...")
        
        website_dir = self.repo_root / "website"
        js_dir = website_dir / "js"
        
        try:
            # Create js directory if it doesn't exist
            js_dir.mkdir(exist_ok=True)
            
            # Copy shared-navigation.js if it exists in root
            shared_nav_src = website_dir / "shared-navigation.js"
            shared_nav_dst = js_dir / "shared-navigation.js"
            
            if shared_nav_src.exists() and not shared_nav_dst.exists():
                with open(shared_nav_src, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                with open(shared_nav_dst, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.fixes_applied.append("Copied shared-navigation.js to js directory")
                print("✅ Created js directory with shared-navigation.js")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating JS directory: {e}")
            return False
    
    def run_all_fixes(self) -> bool:
        """Run all HTML fixes"""
        print("🌟 Een Unity Mathematics - HTML Structure Fix Suite")
        print("=" * 50)
        
        results = [
            self.create_missing_js_directory(),
            self.fix_malformed_html_files()
        ]
        
        print("\n" + "=" * 50)
        print("📊 FIX SUMMARY")
        print("=" * 50)
        
        for i, result in enumerate(results):
            status = "✅" if result else "❌"
            fix_name = ["JS directory", "HTML structure"][i]
            print(f"{status} {fix_name}")
        
        if self.fixes_applied:
            print(f"\n🔧 Applied {len(self.fixes_applied)} fixes:")
            for fix in self.fixes_applied:
                print(f"  • {fix}")
        
        print("\n🎯 HTML structure fixes complete!")
        
        return all(results)


def main():
    """Main execution function"""
    fixer = HTMLFixer()
    success = fixer.run_all_fixes()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 