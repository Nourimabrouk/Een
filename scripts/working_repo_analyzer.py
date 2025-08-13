"""
Working Repository Analysis Script for Een
Generate GitHub links for source code and HTML files, ordered by file size
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

def analyze_repository(repo_path: str) -> List[Dict]:
    """Analyze repository and return all included files"""
    all_files = []
    
    print(f"Analyzing repository: {repo_path}")
    
    for root, dirs, files in os.walk(repo_path):
        # Skip virtual environment directory
        if 'een' in dirs:
            dirs.remove('een')
        if '__pycache__' in dirs:
            dirs.remove('__pycache__')
        if 'node_modules' in dirs:
            dirs.remove('node_modules')
        if '.git' in dirs:
            dirs.remove('.git')
        if '.github' in dirs:
            dirs.remove('.github')
        
        for file in files:
            file_path = os.path.join(root, file)
            ext = pathlib.Path(file_path).suffix.lower()
            
            # Include source code and web files
            include_extensions = {
                '.py', '.js', '.html', '.css', '.ts', '.jsx', '.tsx',
                '.r', '.R', '.lean', '.sql', '.sh', '.bat', '.ps1',
                '.json', '.yaml', '.yml', '.md', '.txt', '.csv',
                '.cpp', '.c', '.h', '.java', '.go', '.rs', '.php', '.rb'
            }
            
            if ext in include_extensions:
                # Skip some specific config files
                filename = os.path.basename(file_path).lower()
                skip_files = {
                    'package-lock.json', 'yarn.lock', '.gitignore', 
                    'robots.txt', '.env', '.env.example', 'dockerfile'
                }
                
                if filename not in skip_files:
                    rel_path = os.path.relpath(file_path, repo_path)
                    file_size = get_file_size(file_path)
                    
                    file_info = {
                        'path': rel_path,
                        'size': file_size,
                        'extension': ext,
                        'is_html': file.lower().endswith('.html')
                    }
                    
                    all_files.append(file_info)
    
    # Sort by size (largest first)
    all_files.sort(key=lambda x: x['size'], reverse=True)
    
    return all_files

def generate_report(files: List[Dict], github_user: str, repo_name: str) -> str:
    """Generate markdown report with GitHub links"""
    
    # Separate HTML and source files
    html_files = [f for f in files if f['is_html']]
    source_files = [f for f in files if not f['is_html']]
    
    # Build report
    lines = []
    lines.append("# Een Repository Source Code and HTML Analysis")
    lines.append("")
    lines.append("Comprehensive analysis of the Een Unity Mathematics repository source code and HTML pages.")
    lines.append("Files are ordered by size (largest first) and exclude virtual environments, build artifacts, and binary files.")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- **Total Files**: {len(files)}")
    lines.append(f"- **HTML Files**: {len(html_files)}")
    lines.append(f"- **Source Code Files**: {len(source_files)}")
    
    total_size = sum(f['size'] for f in files)
    if total_size < 1024 * 1024:
        size_str = f"{total_size / 1024:.1f} KB"
    else:
        size_str = f"{total_size / (1024 * 1024):.1f} MB"
    lines.append(f"- **Total Size**: {size_str}")
    lines.append("")
    
    # HTML Files Section
    if html_files:
        lines.append("## HTML Files (Live Website Pages)")
        lines.append("")
        lines.append("These HTML files are accessible as live web pages through GitHub Pages.")
        lines.append("")
        
        for file_info in html_files:
            path = file_info['path'].replace('\\', '/')
            size_kb = file_info['size'] / 1024
            
            live_link = f"https://{github_user}.github.io/{repo_name}/{path}"
            raw_link = f"https://raw.githubusercontent.com/{github_user}/{repo_name}/main/{path}"
            
            lines.append(f"### {path}")
            lines.append(f"**Size**: {size_kb:.1f} KB")
            lines.append(f"- **Live Page**: {live_link}")
            lines.append(f"- **Raw HTML**: {raw_link}")
            lines.append("")
    
    # Source Code Files Section
    if source_files:
        lines.append("## Source Code Files")
        lines.append("")
        lines.append("Source code files organized by type, with direct GitHub raw links.")
        lines.append("")
        
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
            files_of_type = by_extension[ext]
            
            if total_size < 1024 * 1024:
                size_str = f"{total_size / 1024:.1f} KB"
            else:
                size_str = f"{total_size / (1024 * 1024):.1f} MB"
            
            lines.append(f"### {ext.upper() if ext != 'no-extension' else 'No Extension'} Files")
            lines.append(f"**Count**: {len(files_of_type)} files | **Total Size**: {size_str}")
            lines.append("")
            
            for file_info in files_of_type:
                path = file_info['path'].replace('\\', '/')
                size_kb = file_info['size'] / 1024
                raw_link = f"https://raw.githubusercontent.com/{github_user}/{repo_name}/main/{path}"
                
                lines.append(f"- **{path}** ({size_kb:.1f} KB) - {raw_link}")
            
            lines.append("")
    
    return "\n".join(lines)

def main():
    repo_path = r"C:\Users\Nouri\Documents\GitHub\Een"
    github_user = "NouriMabrouk"
    repo_name = "Een"
    
    # Analyze repository
    all_files = analyze_repository(repo_path)
    
    print(f"Found {len(all_files)} files total")
    
    # Generate report
    report_content = generate_report(all_files, github_user, repo_name)
    
    # Save report
    output_file = os.path.join(repo_path, "docs", "repository_source_analysis.md")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"Analysis complete! Report saved to: {output_file}")
    return output_file

if __name__ == "__main__":
    main()