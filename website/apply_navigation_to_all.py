#!/usr/bin/env python3
"""
Apply Unified Navigation to All HTML Pages
Meta-Optimal Navigation System Deployment Script
"""

import os
import re
import shutil
import glob
from datetime import datetime

# Configuration
WEBSITE_DIR = "."
BACKUP_DIR = "./navigation-backups"
DRY_RUN = False  # Set to True to preview changes

# Navigation template to inject
NAVIGATION_TEMPLATE = """    <!-- Unified Navigation System - Meta-Optimal One-Line Solution -->
    <script src="js/nav-template-applier.js" defer></script>"""

# Legacy patterns to remove
LEGACY_PATTERNS = [
    # CSS patterns
    (r'<link[^>]*href[^>]*meta-optimal-navigation[^>]*>', ''),
    (r'<link[^>]*href[^>]*unified-navigation-system[^>]*>', ''),
    (r'<link[^>]*href[^>]*navigation[^>]*\.css[^>]*>', ''),
    
    # JS patterns
    (r'<script[^>]*src[^>]*meta-optimal-navigation[^>]*></script>', ''),
    (r'<script[^>]*src[^>]*unified-navigation-system[^>]*></script>', ''),
    (r'<script[^>]*src[^>]*navigation[^>]*\.js[^>]*></script>', ''),
    
    # Comment cleanup
    (r'<!--[\s\S]*?Meta-Optimal.*?Navigation[\s\S]*?-->', ''),
    (r'<!--[\s\S]*?Navigation.*?System[\s\S]*?-->', ''),
    
    # Multiple blank lines cleanup
    (r'\n\s*\n\s*\n+', '\n\n'),
]

# Files to skip (already updated or special cases)
SKIP_FILES = {
    'metastation-hub.html',
    'implementations-gallery.html', 
    'zen-unity-meditation.html',
    'meta_tags_template.html',
    'redirect.html',
    'google5936e6fc51b68c92.html',
    'NAVIGATION_UPGRADE_SUMMARY.md'
}

def create_backup_dir():
    """Create backup directory if it doesn't exist"""
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)
        print(f"📁 Created backup directory: {BACKUP_DIR}")

def backup_file(file_path):
    """Create backup of file"""
    backup_path = os.path.join(BACKUP_DIR, os.path.basename(file_path))
    shutil.copy2(file_path, backup_path)
    print(f"💾 Backed up: {os.path.basename(file_path)}")

def has_navigation_template(content):
    """Check if file already has unified navigation"""
    return 'nav-template-applier.js' in content or 'unified-navigation.js' in content

def remove_legacy_navigation(content):
    """Remove legacy navigation patterns"""
    clean_content = content
    
    for pattern, replacement in LEGACY_PATTERNS:
        clean_content = re.sub(pattern, replacement, clean_content, flags=re.IGNORECASE)
    
    return clean_content

def inject_navigation_template(content):
    """Inject navigation template before </head>"""
    head_end_pattern = r'</head>'
    
    if not re.search(head_end_pattern, content, re.IGNORECASE):
        print("⚠️ No </head> tag found, adding to top of <body>")
        body_pattern = r'(<body[^>]*>)'
        return re.sub(body_pattern, r'\1\n' + NAVIGATION_TEMPLATE + '\n', content, flags=re.IGNORECASE)
    
    return re.sub(head_end_pattern, NAVIGATION_TEMPLATE + '\n</head>', content, flags=re.IGNORECASE)

def process_file(file_path):
    """Process a single HTML file"""
    file_name = os.path.basename(file_path)
    print(f"\n🔧 Processing: {file_name}")
    
    # Skip if in skip list
    if file_name in SKIP_FILES:
        print(f"⏭️ Skipping (already updated): {file_name}")
        return {'skipped': True, 'reason': 'already_updated'}
    
    # Read file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Error reading {file_name}: {e}")
        return {'error': True, 'reason': str(e)}
    
    # Check if already has unified navigation
    if has_navigation_template(content):
        print(f"✅ Already has unified navigation: {file_name}")
        return {'skipped': True, 'reason': 'already_unified'}
    
    # Create backup
    if not DRY_RUN:
        backup_file(file_path)
    
    # Process content
    new_content = content
    
    # Remove legacy navigation
    new_content = remove_legacy_navigation(new_content)
    
    # Inject unified navigation template
    new_content = inject_navigation_template(new_content)
    
    if DRY_RUN:
        print(f"🔍 [DRY RUN] Would update: {file_name}")
        return {'dry_run': True}
    
    # Write updated file
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"✅ Updated: {file_name}")
        return {'success': True}
    except Exception as e:
        print(f"❌ Error writing {file_name}: {e}")
        return {'error': True, 'reason': str(e)}

def main():
    """Main deployment function"""
    print("🚀 Starting Unified Navigation System Deployment")
    print(f"📁 Working directory: {os.path.abspath(WEBSITE_DIR)}")
    print(f"🔧 Mode: {'DRY RUN (preview only)' if DRY_RUN else 'LIVE DEPLOYMENT'}")
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not DRY_RUN:
        create_backup_dir()
    
    # Find all HTML files
    html_files = glob.glob("**/*.html", recursive=True)
    # Filter out backup directory and node_modules
    html_files = [f for f in html_files if not f.startswith('navigation-backups/') and 'node_modules' not in f]
    
    print(f"📄 Found {len(html_files)} HTML files to process")
    
    # Process each file
    results = {
        'processed': 0,
        'updated': 0,
        'skipped': 0,
        'errors': 0,
        'dry_run': 0
    }
    
    for file_path in html_files:
        result = process_file(file_path)
        results['processed'] += 1
        
        if result.get('success'):
            results['updated'] += 1
        elif result.get('skipped'):
            results['skipped'] += 1
        elif result.get('error'):
            results['errors'] += 1
        elif result.get('dry_run'):
            results['dry_run'] += 1
    
    # Summary
    print('\n' + '='*60)
    print('📊 DEPLOYMENT SUMMARY')
    print('='*60)
    print(f"📄 Total files processed: {results['processed']}")
    
    if DRY_RUN:
        print(f"🔍 Files that would be updated: {results['dry_run']}")
    else:
        print(f"✅ Files successfully updated: {results['updated']}")
        print(f"💾 Backups created in: {BACKUP_DIR}")
    
    print(f"⏭️ Files skipped: {results['skipped']}")
    print(f"❌ Errors encountered: {results['errors']}")
    
    if results['errors'] == 0:
        print('\n🎉 SUCCESS: All pages now have unified navigation!')
        print('\n📋 Next steps:')
        print('1. Test pages: http://localhost:8001/metastation-hub.html')
        print('2. Verify navigation works on all screen sizes')
        print('3. Clean up legacy navigation files')
        print('4. Update documentation')
    else:
        print('\n⚠️ Some errors occurred. Check the logs above.')
    
    print('\n🌟 Unity Mathematics Navigation System Deployed! 🌟')
    
    return results

if __name__ == "__main__":
    main()