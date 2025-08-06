#!/usr/bin/env python3
"""
Generate a clean file tree for the Een repository, excluding external packages and cache files.
"""

import os
import sys
from pathlib import Path

def should_exclude(path_str):
    """Check if a path should be excluded from the file tree."""
    exclude_patterns = [
        'venv', '.git', 'node_modules', '__pycache__', 
        '.pyc', '.pyo', '.pyd', '.pytest_cache', '.mypy_cache',
        '.coverage', 'htmlcov', '.tox', '.eggs', 'build', 'dist',
        '.egg-info', '.DS_Store', 'Thumbs.db', '.vscode', '.idea',
        '*.log', '*.tmp', '*.bak', '*.swp', '*.swo'
    ]
    
    path_lower = path_str.lower()
    for pattern in exclude_patterns:
        if pattern in path_lower:
            return True
    return False

def generate_tree(directory, prefix="", max_depth=10, current_depth=0):
    """Generate a tree structure for the directory."""
    if current_depth > max_depth:
        return
    
    try:
        items = sorted([item for item in os.listdir(directory) 
                       if not should_exclude(item)])
        
        for i, item in enumerate(items):
            item_path = os.path.join(directory, item)
            is_last = i == len(items) - 1
            
            # Determine the connector
            if is_last:
                connector = "└── "
                next_prefix = prefix + "    "
            else:
                connector = "├── "
                next_prefix = prefix + "│   "
            
            # Print the current item
            print(f"{prefix}{connector}{item}")
            
            # Recursively process directories
            if os.path.isdir(item_path):
                generate_tree(item_path, next_prefix, max_depth, current_depth + 1)
                
    except PermissionError:
        print(f"{prefix}└── [Permission Denied]")
    except Exception as e:
        print(f"{prefix}└── [Error: {e}]")

def main():
    """Main function to generate the file tree."""
    # Get the current directory
    current_dir = os.getcwd()
    
    print("Een Unity Mathematics Framework - Complete File Tree")
    print("=" * 60)
    print(f"Root: {current_dir}")
    print()
    
    # Generate the tree
    generate_tree(current_dir, max_depth=8)
    
    print()
    print("=" * 60)
    print("Note: External packages, cache files, and system files have been excluded.")
    print("This tree shows only the core Een project files and directories.")

if __name__ == "__main__":
    main() 