#!/usr/bin/env python3
"""
Clean up legacy navigation files and consolidate the system
"""

import os
import shutil
from datetime import datetime

# Files to keep (the new unified system)
KEEP_FILES = {
    'css/unified-navigation.css',
    'js/unified-navigation.js', 
    'js/apply-unified-navigation.js',
    'js/nav-template-applier.js'
}

# Legacy files to remove
LEGACY_FILES = [
    'css/meta-optimal-navigation-complete.css',
    'css/meta-optimal-navigation.css',
    'js/apply-universal-navigation.js',
    'js/meta-optimal-navigation-complete.js',
    'js/navigation-batch-updater.js',
    'js/unified-navigation-system.js',
    'js/universal-ai-navigation.js',
    'apply-meta-optimal-navigation.js',
    'consolidate-to-optimal-navigation.js',
    'shared-navigation.js',
    'update-all-navigation.js',
    'update-all-pages-navigation.js'
]

def cleanup_legacy_navigation():
    print("Starting Legacy Navigation Cleanup")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create archive directory for legacy files
    archive_dir = './navigation-archive'
    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)
        print(f"Created archive directory: {archive_dir}")
    
    removed_count = 0
    archived_count = 0
    
    # Remove legacy files
    for file_path in LEGACY_FILES:
        if os.path.exists(file_path):
            # Archive first, then remove
            archive_path = os.path.join(archive_dir, os.path.basename(file_path))
            try:
                shutil.copy2(file_path, archive_path)
                os.remove(file_path)
                print(f"Removed: {file_path} (archived to {archive_path})")
                removed_count += 1
                archived_count += 1
            except Exception as e:
                print(f"Error removing {file_path}: {e}")
        else:
            print(f"Not found (already removed): {file_path}")
    
    print(f"\nCleanup Summary:")
    print(f"Files removed: {removed_count}")
    print(f"Files archived: {archived_count}")
    
    # Verify kept files exist
    print(f"\nVerifying unified navigation system:")
    for file_path in KEEP_FILES:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING!")
    
    print(f"\n🎉 Legacy navigation cleanup complete!")
    print(f"✅ Unified navigation system is now the only navigation system")

if __name__ == "__main__":
    cleanup_legacy_navigation()