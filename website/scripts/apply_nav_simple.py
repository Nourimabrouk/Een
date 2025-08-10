#!/usr/bin/env python3
"""Apply Unified Navigation to All HTML Pages"""

import os
import re
import shutil
import glob
from datetime import datetime

# Configuration
NAVIGATION_TEMPLATE = """    <!-- Unified Navigation System - Meta-Optimal One-Line Solution -->
    <script src="js/nav-template-applier.js" defer></script>"""

SKIP_FILES = {
    'metastation-hub.html',
    'implementations-gallery.html', 
    'zen-unity-meditation.html',
    'meta_tags_template.html',
    'redirect.html',
    'google5936e6fc51b68c92.html'
}

def has_navigation_template(content):
    return 'nav-template-applier.js' in content or 'unified-navigation.js' in content

def inject_navigation_template(content):
    if not re.search(r'</head>', content, re.IGNORECASE):
        body_pattern = r'(<body[^>]*>)'
        return re.sub(body_pattern, r'\1\n' + NAVIGATION_TEMPLATE + '\n', content, flags=re.IGNORECASE)
    
    return re.sub(r'</head>', NAVIGATION_TEMPLATE + '\n</head>', content, flags=re.IGNORECASE)

def process_file(file_path):
    file_name = os.path.basename(file_path)
    print(f"Processing: {file_name}")
    
    if file_name in SKIP_FILES:
        print(f"  -> Skipping (already updated)")
        return {'skipped': True}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        print(f"  -> Error reading file")
        return {'error': True}
    
    if has_navigation_template(content):
        print(f"  -> Already has unified navigation")
        return {'skipped': True}
    
    # Backup
    backup_dir = './navigation-backups'
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    shutil.copy2(file_path, os.path.join(backup_dir, file_name))
    
    # Inject navigation
    new_content = inject_navigation_template(content)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"  -> Updated successfully")
        return {'success': True}
    except:
        print(f"  -> Error writing file")
        return {'error': True}

def main():
    print("Starting Unified Navigation System Deployment")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    html_files = glob.glob("**/*.html", recursive=True)
    html_files = [f for f in html_files if not f.startswith('navigation-backups/')]
    
    print(f"Found {len(html_files)} HTML files")
    
    results = {'processed': 0, 'updated': 0, 'skipped': 0, 'errors': 0}
    
    for file_path in html_files:
        result = process_file(file_path)
        results['processed'] += 1
        
        if result.get('success'):
            results['updated'] += 1
        elif result.get('skipped'):
            results['skipped'] += 1
        elif result.get('error'):
            results['errors'] += 1
    
    print("\nDEPLOYMENT SUMMARY")
    print(f"Total processed: {results['processed']}")
    print(f"Updated: {results['updated']}")
    print(f"Skipped: {results['skipped']}")
    print(f"Errors: {results['errors']}")
    
    if results['errors'] == 0:
        print("\nSUCCESS: All pages now have unified navigation!")
    
    print("\nUnity Mathematics Navigation System Deployed!")

if __name__ == "__main__":
    main()