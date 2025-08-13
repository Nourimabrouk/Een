"""
Repository Source Code and HTML Analysis Script
Generates GitHub links for source code and HTML pages, ordered by file size
Excludes boilerplate, virtual environments, styling, and other non-essential files
"""

import os
import pathlib
from typing import List, Tuple, Dict
import mimetypes

def get_file_size(file_path: str) -> int:
    """Get file size in bytes"""
    try:
        return os.path.getsize(file_path)
    except (OSError, FileNotFoundError):
        return 0

def is_source_code_file(file_path: str) -> bool:
    """Check if file is a source code file"""
    source_extensions = {
        '.py', '.js', '.html', '.css', '.r', '.R', '.cpp', '.c', '.h', 
        '.java', '.ts', '.tsx', '.jsx', '.go', '.rs', '.php', '.rb',
        '.sql', '.sh', '.bat', '.ps1', '.md', '.json', '.yaml', '.yml',
        '.xml', '.csv', '.txt'
    }
    
    file_ext = pathlib.Path(file_path).suffix.lower()
    return file_ext in source_extensions

def should_exclude_file(file_path: str) -> bool:
    """Check if file should be excluded from analysis"""
    exclude_patterns = [
        # Virtual environments (specific paths)
        '/een/', '/venv/', '/env/', '/.venv/', '/__pycache__/',
        '/node_modules/', '/.git/', '/.github/',
        
        # Build and dist directories
        '/dist/', '/build/', '/.next/', '/.cache/',
        
        # IDE and config
        '/.vscode/', '/.idea/', '/.vs/',
        
        # Temporary and log files
        '.log', '.tmp', '.temp', '.cache',
        
        # Binary and media files (unless specifically needed)
        '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico',
        '.pdf', '.zip', '.tar', '.gz', '.exe', '.dll', '.mp4', '.webm', '.mp3',
        
        # Package management and lock files
        'package-lock.json', 'yarn.lock', 'poetry.lock',
        'requirements-freeze.txt',
        
        # OS files
        '.DS_Store', 'Thumbs.db', 'desktop.ini',
        
        # Specific files to exclude
        'nul', 'server.log'
    ]
    
    # Normalize path for comparison
    normalized_path = file_path.replace('\\', '/').lower()
    filename = os.path.basename(file_path).lower()
    
    # Skip common config/build files by name
    skip_files = {
        '.gitignore', '.gitattributes', '.dockerignore', '.vercelignore',
        'dockerfile', 'makefile', 'package.json', 'pyproject.toml',
        'requirements.txt', 'robots.txt', 'license', '.env.example',
        '.pre-commit-config.yaml', '.lighthouserc.json', '.nojekyll',
        'nginx.conf', 'vercel.json', 'manifest.json', '.cursorrules'
    }
    
    if filename in skip_files:
        return True
    
    for pattern in exclude_patterns:
        if pattern in normalized_path:
            return True
    
    return False

def should_exclude_directory(dir_path: str) -> bool:
    """Check if entire directory should be excluded"""
    exclude_dirs = {
        'een', '__pycache__', 'node_modules',
        '.git', '.github', 'dist', 'build', '.next', '.cache',
        '.vscode', '.idea', '.vs'
    }
    
    dir_name = os.path.basename(dir_path).lower()
    return dir_name in exclude_dirs

def analyze_repository(repo_path: str) -> Tuple[List[Dict], List[Dict]]:
    """Analyze repository and return source files and HTML files"""
    source_files = []
    html_files = []
    
    print(f"Starting analysis of: {repo_path}")
    
    for root, dirs, files in os.walk(repo_path):
        # Filter out excluded directories
        original_dirs = dirs[:]
        dirs[:] = [d for d in dirs if not should_exclude_directory(os.path.join(root, d))]
        
        excluded_dirs = set(original_dirs) - set(dirs)
        if excluded_dirs:
            print(f"Excluding directories: {excluded_dirs}")
        
        print(f"Processing directory: {root}")
        print(f"  Found {len(files)} files")
        
        for file in files:
            file_path = os.path.join(root, file)
            
            # Check exclusions step by step
            if should_exclude_file(file_path):
                try:
                    print(f"  Excluded (file pattern): {file}")
                except:
                    print(f"  Excluded (file pattern): [unprintable filename]")
                continue
                
            # Skip if not a source code file
            if not is_source_code_file(file_path):
                try:
                    print(f"  Excluded (not source): {file}")
                except:
                    print(f"  Excluded (not source): [unprintable filename]")
                continue
            
            try:
                print(f"  Including: {file}")
            except:
                print(f"  Including: [unprintable filename]")
            
            # Get relative path from repo root
            rel_path = os.path.relpath(file_path, repo_path)
            file_size = get_file_size(file_path)
            
            file_info = {
                'path': rel_path,
                'size': file_size,
                'extension': pathlib.Path(file_path).suffix.lower()
            }
            
            if file.lower().endswith('.html'):
                html_files.append(file_info)
            else:
                source_files.append(file_info)
    
    print(f"Total source files found: {len(source_files)}")
    print(f"Total HTML files found: {len(html_files)}")
    
    # Sort by size (largest first)
    source_files.sort(key=lambda x: x['size'], reverse=True)
    html_files.sort(key=lambda x: x['size'], reverse=True)
    
    return source_files, html_files

def generate_github_links(files: List[Dict], base_url: str, is_html: bool = False) -> List[str]:
    """Generate GitHub links for files"""
    links = []
    
    for file_info in files:
        path = file_info['path'].replace('\\', '/')
        size_kb = file_info['size'] / 1024
        
        if is_html:
            # Use GitHub Pages or raw link for HTML files
            # For HTML files, we'll provide both raw and potential live links
            raw_link = f"{base_url}/main/{path}"
            live_link = f"https://nourimabrouk.github.io/Een/{path}"
            
            links.append(f"- **{path}** ({size_kb:.1f} KB)")
            links.append(f"  - Raw: {raw_link}")
            links.append(f"  - Live: {live_link}")
        else:
            # Use raw GitHub link for source files
            raw_link = f"{base_url}/main/{path}"
            links.append(f"- **{path}** ({size_kb:.1f} KB) - {raw_link}")
    
    return links

def format_size(bytes_size: int) -> str:
    """Format file size in human readable format"""
    if bytes_size < 1024:
        return f"{bytes_size} B"
    elif bytes_size < 1024 * 1024:
        return f"{bytes_size / 1024:.1f} KB"
    else:
        return f"{bytes_size / (1024 * 1024):.1f} MB"

def main():
    repo_path = r"C:\Users\Nouri\Documents\GitHub\Een"
    github_raw_base = "https://raw.githubusercontent.com/NouriMabrouk/Een"
    
    print("Analyzing Een repository...")
    source_files, html_files = analyze_repository(repo_path)
    
    print(f"Found {len(source_files)} source code files")
    print(f"Found {len(html_files)} HTML files")
    
    # Generate markdown report
    report_lines = []
    report_lines.append("# Een Repository Source Code and HTML Analysis")
    report_lines.append("")
    report_lines.append("Generated repository analysis with GitHub links, ordered by file size (largest to smallest).")
    report_lines.append("")
    report_lines.append("## Summary")
    report_lines.append(f"- **Total Source Files**: {len(source_files)}")
    report_lines.append(f"- **Total HTML Files**: {len(html_files)}")
    
    total_source_size = sum(f['size'] for f in source_files)
    total_html_size = sum(f['size'] for f in html_files)
    report_lines.append(f"- **Total Source Code Size**: {format_size(total_source_size)}")
    report_lines.append(f"- **Total HTML Size**: {format_size(total_html_size)}")
    report_lines.append("")
    
    # HTML Files Section
    if html_files:
        report_lines.append("## HTML Files (Live Website Pages)")
        report_lines.append("")
        html_links = generate_github_links(html_files, github_raw_base, is_html=True)
        report_lines.extend(html_links)
        report_lines.append("")
    
    # Source Code Files Section
    if source_files:
        report_lines.append("## Source Code Files")
        report_lines.append("")
        
        # Group by file type
        by_extension = {}
        for file_info in source_files:
            ext = file_info['extension'] or 'no-extension'
            if ext not in by_extension:
                by_extension[ext] = []
            by_extension[ext].append(file_info)
        
        # Sort extensions by total size
        ext_sizes = [(ext, sum(f['size'] for f in files)) for ext, files in by_extension.items()]
        ext_sizes.sort(key=lambda x: x[1], reverse=True)
        
        for ext, total_size in ext_sizes:
            files = by_extension[ext]
            report_lines.append(f"### {ext.upper() if ext != 'no-extension' else 'No Extension'} Files ({len(files)} files, {format_size(total_size)})")
            report_lines.append("")
            
            source_links = generate_github_links(files, github_raw_base, is_html=False)
            report_lines.extend(source_links)
            report_lines.append("")
    
    # Write report to file
    report_content = "\n".join(report_lines)
    output_file = os.path.join(repo_path, "docs", "repository_analysis.md")
    
    # Ensure docs directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"Analysis complete! Report saved to: {output_file}")
    print(f"Total files analyzed: {len(source_files) + len(html_files)}")
    
    return output_file

if __name__ == "__main__":
    main()