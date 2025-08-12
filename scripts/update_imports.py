#!/usr/bin/env python3
"""
Update import statements for repository consolidation
Converts old imports to new unified src/ structure
"""

import os
import re
import glob

def update_imports_in_file(file_path):
    """Update imports in a single Python file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Update patterns
        replacements = [
            # Core imports
            (r'from core\.mathematical\.', 'from src.core.mathematical.'),
            (r'from core\.consciousness\.', 'from src.consciousness.'),
            (r'from core\.visualization\.', 'from src.core.visualization.'),
            (r'from core\.agents\.', 'from src.agents.'),
            (r'from core\.', 'from src.core.'),
            (r'import core\.mathematical\.', 'import src.core.mathematical.'),
            (r'import core\.consciousness\.', 'import src.consciousness.'),
            (r'import core\.', 'import src.core.'),
            
            # Een imports 
            (r'from een\.agents\.', 'from src.agents.'),
            (r'from een\.mcp\.', 'from src.mcp.'),
            (r'from een\.experiments\.', 'from src.experiments.'),
            (r'from een\.proofs\.', 'from src.proofs.'),
            (r'from een\.dashboards\.', 'from src.dashboards.'),
            (r'from een\.', 'from src.'),
            (r'import een\.', 'import src.'),
        ]
        
        # Apply replacements
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content)
        
        # Write back if changed
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated: {file_path}")
            return True
        return False
        
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def main():
    """Update all Python files in the repository"""
    print("Updating import statements across repository...")
    
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk('.'):
        # Skip virtual environments and cache directories
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'node_modules', 'Lib', 'Scripts']]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"Found {len(python_files)} Python files")
    
    updated_count = 0
    for file_path in python_files:
        if update_imports_in_file(file_path):
            updated_count += 1
    
    print(f"Updated {updated_count} files with new import paths")
    print("Import update complete!")

if __name__ == "__main__":
    main()