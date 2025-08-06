#!/usr/bin/env python3
"""
Auto Gallery Updater for Een Unity Mathematics
Monitors visualization folders and automatically updates gallery data
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any
import hashlib
import sys

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.gallery_scanner import scan_all_folders, VIZ_FOLDERS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoGalleryUpdater:
    """Automated gallery updater with file monitoring"""
    
    def __init__(self):
        self.gallery_data_file = project_root / "gallery_data.json"
        self.last_hash_file = project_root / ".gallery_hash"
        self.watch_folders = VIZ_FOLDERS
        self.current_hash = None
        self.last_hash = self.load_last_hash()
        
    def load_last_hash(self) -> str:
        """Load the last known hash of gallery data"""
        try:
            if self.last_hash_file.exists():
                return self.last_hash_file.read_text().strip()
        except Exception as e:
            logger.warning(f"Could not load last hash: {e}")
        return ""
    
    def save_hash(self, hash_value: str):
        """Save the current hash"""
        try:
            self.last_hash_file.write_text(hash_value)
        except Exception as e:
            logger.warning(f"Could not save hash: {e}")
    
    def calculate_folder_hash(self) -> str:
        """Calculate hash of all visualization folders"""
        hash_content = ""
        
        for folder_path in self.watch_folders:
            folder = Path(folder_path)
            if folder.exists():
                # Get all files and their modification times
                for file_path in folder.rglob("*"):
                    if file_path.is_file():
                        stat = file_path.stat()
                        hash_content += (
                            f"{file_path}:{stat.st_mtime}:{stat.st_size}\n"
                        )
        
        return hashlib.md5(hash_content.encode()).hexdigest()
    
    def update_gallery_data(self) -> bool:
        """Update gallery data if changes detected"""
        try:
            # Calculate current hash
            self.current_hash = self.calculate_folder_hash()
            
            # Check if changes detected
            if self.current_hash == self.last_hash:
                logger.debug("No changes detected in visualization folders")
                return False
            
            logger.info(
                "Changes detected in visualization folders - updating gallery data"
            )
            
            # Scan all folders and generate new data
            gallery_data = scan_all_folders()
            
            # Save updated data
            with open(self.gallery_data_file, 'w', encoding='utf-8') as f:
                json.dump(gallery_data, f, indent=2, ensure_ascii=False)
            
            # Update hash
            self.last_hash = self.current_hash
            self.save_hash(self.current_hash)
            
            logger.info("✅ Gallery data updated successfully!")
            total_viz = gallery_data['statistics']['total']
            featured_count = gallery_data['statistics']['featured_count']
            logger.info(f"📊 Found {total_viz} visualizations")
            logger.info(f"🎯 Featured: {featured_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating gallery data: {e}")
            return False
    
    def watch_and_update(self, interval: int = 30):
        """Continuously watch folders and update gallery data"""
        logger.info(
            f"🔄 Starting auto gallery updater (checking every {interval} seconds)"
        )
        logger.info(f"📁 Monitoring {len(self.watch_folders)} folders")
        
        try:
            while True:
                # Check for updates
                updated = self.update_gallery_data()
                
                if updated:
                    logger.info("🎨 Gallery updated - changes detected!")
                else:
                    logger.debug("No changes detected")
                
                # Wait before next check
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("🛑 Auto gallery updater stopped by user")
        except Exception as e:
            logger.error(f"❌ Auto gallery updater error: {e}")


class GitHubWebhookHandler:
    """Handle GitHub webhooks for automatic gallery updates"""
    
    def __init__(self):
        self.gallery_updater = AutoGalleryUpdater()
    
    def handle_webhook(self, payload: Dict[str, Any]) -> bool:
        """Handle GitHub webhook payload"""
        try:
            # Check if relevant files were changed
            if 'commits' in payload:
                for commit in payload['commits']:
                    for file_path in commit.get('added', []) + commit.get('modified', []):
                        # Check if file is in visualization folders
                        for viz_folder in self.watch_folders:
                            if file_path.startswith(viz_folder):
                                logger.info(f"🔄 Visualization file changed: {file_path}")
                                return self.gallery_updater.update_gallery_data()
            
            return False
            
        except Exception as e:
            logger.error(f"Error handling webhook: {e}")
            return False


def main():
    """Main function for auto gallery updater"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Auto Gallery Updater for Een Unity Mathematics"
    )
    parser.add_argument("--watch", action="store_true", help="Watch folders continuously")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds")
    parser.add_argument("--once", action="store_true", help="Update once and exit")
    parser.add_argument("--webhook", action="store_true", help="Handle GitHub webhook")
    
    args = parser.parse_args()
    
    updater = AutoGalleryUpdater()
    
    if args.webhook:
        # Handle webhook (for GitHub integration)
        payload = json.loads(sys.stdin.read())
        success = updater.update_gallery_data()
        sys.exit(0 if success else 1)
    
    elif args.once:
        # Update once and exit
        success = updater.update_gallery_data()
        sys.exit(0 if success else 1)
    
    elif args.watch:
        # Watch continuously
        updater.watch_and_update(args.interval)
    
    else:
        # Default: update once
        success = updater.update_gallery_data()
        if success:
            print("✅ Gallery data updated successfully!")
        else:
            print("ℹ️ No changes detected")


if __name__ == "__main__":
    main() 