"""
Simple Repository Analysis Script for Een
Focus on including source code and HTML files with minimal exclusions
"""

import os
import pathlib
from typing import List, Dict

def get_file_size(file_path: str) -> int:
    """Get file size in bytes"""
    try:
        return os.path.getsize(file_path)
    except (OSError, FileNotFoundError):
        return 0

def should_include_file(file_path: str) -> bool:
    """Check if file should be included in analysis"""
    
    # Normalize path
    normalized_path = file_path.replace('\\', '/').lower()
    
    # Exclude virtual environments and build dirs - more specific paths
    exclude_paths = ['/een/', '/__pycache__/', '/node_modules/', '/.git/', '/.github/']
    for exclude_path in exclude_paths:
        if exclude_path in normalized_path:
            return False
    
    # Get file extension
    ext = pathlib.Path(file_path).suffix.lower()
    
    # Include source code files
    source_extensions = {
        '.py', '.js', '.html', '.css', '.ts', '.jsx', '.tsx',
        '.r', '.R', '.lean', '.sql', '.sh', '.bat', '.ps1',
        '.json', '.yaml', '.yml', '.md', '.txt', '.csv',
        '.cpp', '.c', '.h', '.java', '.go', '.rs', '.php', '.rb'
    }
    
    if ext in source_extensions:
        # Exclude some specific config files by name
        filename = os.path.basename(file_path).lower()
        
        # Skip only very specific files
        skip_files = {
            'package-lock.json', 'yarn.lock', '.gitignore', 
            'dockerfile', 'robots.txt', '.env', '.env.example'
        }
        
        if filename in skip_files:
            return False
        
        return True
    
    return False

def analyze_repository(repo_path: str) -> List[Dict]:
    """Analyze repository and return all included files"""
    all_files = []
    
    for root, dirs, files in os.walk(repo_path):
        # Skip virtual env and git directories
        dirs[:] = [d for d in dirs if d.lower() not in {'een', '__pycache__', 'node_modules', '.git', '.github'}]
        
        for file in files:
            file_path = os.path.join(root, file)
            
            if should_include_file(file_path):
                rel_path = os.path.relpath(file_path, repo_path)
                file_size = get_file_size(file_path)
                
                file_info = {
                    'path': rel_path,
                    'size': file_size,
                    'extension': pathlib.Path(file_path).suffix.lower(),
                    'is_html': file.lower().endswith('.html')
                }
                
                all_files.append(file_info)
    
    # Sort by size (largest first)
    all_files.sort(key=lambda x: x['size'], reverse=True)
    
    return all_files

def generate_github_links(files: List[Dict], github_user: str, repo_name: str) -> List[str]:
    """Generate GitHub links for files"""
    links = []
    
    for file_info in files:
        path = file_info['path'].replace('\\', '/')
        size_kb = file_info['size'] / 1024
        
        if file_info['is_html']:
            # GitHub Pages live link
            live_link = f"https://{github_user}.github.io/{repo_name}/{path}"
            # Raw GitHub link
            raw_link = f"https://raw.githubusercontent.com/{github_user}/{repo_name}/main/{path}"
            
            links.append(f"- **{path}** ({size_kb:.1f} KB) - HTML Page")
            links.append(f"  - Live: {live_link}")
            links.append(f"  - Raw: {raw_link}")
        else:
            # Raw GitHub link for source files
            raw_link = f"https://raw.githubusercontent.com/{github_user}/{repo_name}/main/{path}"
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
    github_user = "NouriMabrouk"
    repo_name = "Een"
    
    print("Analyzing Een repository for source code and HTML files...")
    all_files = analyze_repository(repo_path)
    
    # Separate HTML and source files
    html_files = [f for f in all_files if f['is_html']]
    source_files = [f for f in all_files if not f['is_html']]
    
    print(f"Found {len(html_files)} HTML files")
    print(f"Found {len(source_files)} source code files")
    print(f"Total files: {len(all_files)}")
    
    # Generate markdown report
    report_lines = []
    report_lines.append("# Een Repository Source Code and HTML Analysis")
    report_lines.append("")
    report_lines.append("Comprehensive analysis of the Een Unity Mathematics repository, ordered by file size.")
    report_lines.append("")
    report_lines.append("## Summary")
    report_lines.append(f"- **Total Files Analyzed**: {len(all_files)}")
    report_lines.append(f"- **HTML Files**: {len(html_files)}")
    report_lines.append(f"- **Source Code Files**: {len(source_files)}")
    
    total_size = sum(f['size'] for f in all_files)
    report_lines.append(f"- **Total Size**: {format_size(total_size)}")
    report_lines.append("")
    
    # HTML Files Section
    if html_files:
        report_lines.append("## HTML Files (Live Website Pages)")
        report_lines.append("")
        html_links = generate_github_links(html_files, github_user, repo_name)
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
            
            source_links = generate_github_links(files, github_user, repo_name)
            report_lines.extend(source_links)
            report_lines.append("")
    
    # Write report to file
    report_content = "\n".join(report_lines)
    output_file = os.path.join(repo_path, "docs", "repository_source_analysis.md")
    
    # Ensure docs directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"Analysis complete! Report saved to: {output_file}")
    
    return output_file

if __name__ == "__main__":
    main()