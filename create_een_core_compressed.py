#!/usr/bin/env python3
"""
Een Core Compression Script

This script creates a compressed archive of the core Een repository files,
including source code, HTML files, and documentation, while excluding
external libraries and build artifacts.
"""

import os
import zipfile
import fnmatch
from pathlib import Path

def should_include_file(file_path, base_path):
    """
    Determines if a file should be included in the compressed archive.
    
    Args:
        file_path (Path): The file path to check
        base_path (Path): The base repository path
        
    Returns:
        bool: True if the file should be included
    """
    relative_path = file_path.relative_to(base_path)
    relative_str = str(relative_path).replace('\\', '/')
    
    # Exclude directories/files patterns
    exclude_patterns = [
        # Virtual environments and dependencies
        'een/*',
        'Lib/*',
        'node_modules/*',
        'venv/*',
        '__pycache__/*',
        '*.pyc',
        '.git/*',
        
        # Build and cache files
        '*.log',
        '*.cache',
        'logs/*',
        'results/*',
        
        # Media and binary files (except specific ones)
        '*.png',
        '*.jpg',
        '*.jpeg',
        '*.gif',
        '*.mp4',
        '*.webm',
        '*.wav',
        '*.exe',
        '*.dll',
        '*.so',
        'site-packages/*',
        
        # Backup and temporary files
        'backups/*',
        '*.backup',
        '*.bak',
        '*.tmp',
        'nul',
        
        # Package manager files
        'package-lock.json',
        
        # IDE and editor files
        '.vscode/*',
        '.idea/*',
        '*.swp',
        '*.swo',
        
        # OS specific files
        'Thumbs.db',
        '.DS_Store',
    ]
    
    # Check exclude patterns
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(relative_str, pattern):
            return False
    
    # Include specific file types and directories
    include_patterns = [
        # Python source files
        '*.py',
        
        # Web files
        '*.html',
        '*.css',
        '*.js',
        '*.json',
        
        # Documentation
        '*.md',
        '*.txt',
        '*.rst',
        
        # Configuration files
        '*.yml',
        '*.yaml',
        '*.toml',
        '*.ini',
        '*.conf',
        '*.xml',
        
        # Specific files
        'Dockerfile*',
        'Makefile',
        '*.sh',
        '*.bat',
        'requirements*.txt',
        'setup.py',
        'pyproject.toml',
        
        # Mathematical and research files
        '*.lean',
        '*.tex',
        '*.R',
        '*.ipynb',
        
        # Important root files
        'LICENSE*',
        'CHANGELOG*',
        'CONTRIBUTING*',
    ]
    
    # Check if file matches include patterns
    filename = file_path.name
    for pattern in include_patterns:
        if fnmatch.fnmatch(filename, pattern):
            return True
    
    return False

def get_priority_directories():
    """
    Returns a list of priority directories to include.
    """
    return [
        'core',           # Core unity mathematics
        'src',            # Source implementations
        'consciousness',  # Consciousness systems
        'website',        # Website files
        'api',           # API implementations
        'formal_proofs', # Mathematical proofs
        'proofs',        # Proof systems  
        'experiments',   # Experiments
        'examples',      # Examples
        'ml_framework',  # Machine learning
        'dashboards',    # Dashboard implementations
        'agents',        # Agent systems
        'tests',         # Test files
        'scripts',       # Utility scripts
        'docs',          # Documentation
        'config',        # Configuration
        'utils',         # Utilities
        'meta',          # Meta systems
        'infrastructure', # Infrastructure
        'planning',      # Planning documents
        'visualization', # Visualization systems
        'viz',           # Visualization outputs
    ]

def create_compressed_archive():
    """
    Creates the compressed archive of Een core files.
    """
    base_path = Path(__file__).parent
    output_file = base_path / 'een_core_compressed.zip'
    
    print(f"Creating compressed archive: {output_file}")
    print(f"Base directory: {base_path}")
    
    # Track statistics
    included_files = 0
    excluded_files = 0
    total_size = 0
    
    priority_dirs = get_priority_directories()
    
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        
        # First, add files from priority directories
        for priority_dir in priority_dirs:
            dir_path = base_path / priority_dir
            if dir_path.exists() and dir_path.is_dir():
                print(f"Processing priority directory: {priority_dir}")
                
                for file_path in dir_path.rglob('*'):
                    if file_path.is_file():
                        if should_include_file(file_path, base_path):
                            try:
                                # Calculate relative path for archive
                                archive_path = file_path.relative_to(base_path)
                                
                                # Add file to archive
                                zipf.write(file_path, archive_path)
                                
                                included_files += 1
                                total_size += file_path.stat().st_size
                                
                                if included_files % 100 == 0:
                                    print(f"  Added {included_files} files...")
                                    
                            except Exception as e:
                                print(f"Warning: Could not add {file_path}: {e}")
                                excluded_files += 1
                        else:
                            excluded_files += 1
        
        # Add important root-level files
        root_patterns = [
            'README.md',
            'CLAUDE.md', 
            'LICENSE*',
            'requirements*.txt',
            'setup.py',
            'pyproject.toml',
            'Dockerfile*',
            'docker-compose*.yml',
            'Makefile',
            '*.sh',
            '*.bat',
            'main.py',
            'launch.py',
        ]
        
        print("Processing root-level files...")
        for pattern in root_patterns:
            for file_path in base_path.glob(pattern):
                if file_path.is_file() and should_include_file(file_path, base_path):
                    try:
                        archive_path = file_path.relative_to(base_path)
                        zipf.write(file_path, archive_path)
                        included_files += 1
                        total_size += file_path.stat().st_size
                    except Exception as e:
                        print(f"Warning: Could not add {file_path}: {e}")
    
    # Print summary
    print(f"\nCompression complete!")
    print(f"Archive created: {output_file}")
    print(f"Files included: {included_files}")
    print(f"Files excluded: {excluded_files}")
    print(f"Total uncompressed size: {total_size / (1024*1024):.2f} MB")
    
    if output_file.exists():
        compressed_size = output_file.stat().st_size
        compression_ratio = (1 - compressed_size / total_size) * 100 if total_size > 0 else 0
        print(f"Compressed size: {compressed_size / (1024*1024):.2f} MB")
        print(f"Compression ratio: {compression_ratio:.1f}%")
    
    return output_file

if __name__ == "__main__":
    try:
        archive_path = create_compressed_archive()
        print(f"\nSuccessfully created: {archive_path}")
        
        # Validate the archive
        print("\nValidating archive...")
        with zipfile.ZipFile(archive_path, 'r') as zipf:
            file_list = zipf.namelist()
            print(f"Archive contains {len(file_list)} files")
            
            # Show sample of included files
            print("\nSample of included files:")
            for i, filename in enumerate(sorted(file_list)[:20]):
                print(f"  {filename}")
            if len(file_list) > 20:
                print(f"  ... and {len(file_list) - 20} more files")
        
    except Exception as e:
        print(f"Error creating archive: {e}")
        import traceback
        traceback.print_exc()