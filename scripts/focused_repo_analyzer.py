"""
Focused Repository Analysis Script for Een
Only include core Python code YOU wrote and main website HTML pages
Exclude all legacy, backup, duplicate, and generated files
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
    """Check if file should be included - VERY SELECTIVE"""
    
    normalized_path = file_path.replace('\\', '/').lower()
    filename = os.path.basename(file_path).lower()
    
    # EXCLUDE entire directories that are backups/legacy/generated
    exclude_dirs = [
        '/internal/legacy/', '/legacy/', '/backup/', '/backups/',
        '/navigation-backup/', '/een/', '/__pycache__/', '/node_modules/',
        '/.git/', '/.github/', '/migration/', '/archived_navigation/',
        '/.pytest_cache/', '/.streamlit/', '/.vercel/',
        '/redundant_launchers/', '/een.egg-info/', '/lib/site-packages/'
    ]
    
    for exclude_dir in exclude_dirs:
        if exclude_dir in normalized_path:
            return False
    
    # Get file extension
    ext = pathlib.Path(file_path).suffix.lower()
    
    # ONLY INCLUDE specific file types
    if ext == '.py':
        # Python files - but exclude specific generated/config files
        exclude_python = {
            '__init__.py', 'setup.py', 'conftest.py', 
            'pip.cmd', 'python.cmd', 'pip3.exe', 'pip3.13.exe'
        }
        if filename in exclude_python:
            return False
        
        # Exclude files that are clearly duplicates or generated
        if 'backup' in filename or 'copy' in filename or 'duplicate' in filename:
            return False
        if 'fixed' in filename or 'legacy' in filename or 'old' in filename:
            return False
        
        return True
    
    elif ext == '.html':
        # HTML files - only from main website directory
        if '/website/' in normalized_path:
            # Exclude test/backup HTML files
            if any(word in filename for word in ['test', 'backup', 'copy', 'duplicate', 'legacy']):
                return False
            return True
        # Also include root index.html
        if filename == 'index.html' and '/website/' not in normalized_path and '/internal/' not in normalized_path:
            return True
        return False
    
    elif ext == '.md':
        # Only include important documentation
        important_docs = {
            'readme.md', 'claude.md', 'security.md', 'contributing.md',
            'api_structure.md', 'documentation_organization.md'
        }
        if filename in important_docs:
            return True
        
        # Include research and reports from docs folder
        if '/docs/' in normalized_path and any(word in normalized_path for word in ['/research/', '/reports/', '/summaries/']):
            return True
        
        return False
    
    elif ext in {'.lean', '.r'}:
        # Include formal proofs and R scripts
        if '/formal_proofs/' in normalized_path or 'unity' in filename:
            return True
        return False
    
    return False

def analyze_repository(repo_path: str) -> Dict[str, List[Dict]]:
    """Analyze repository and return categorized files"""
    
    categories = {
        'core_python': [],
        'website_html': [],
        'research_docs': [],
        'formal_proofs': []
    }
    
    print(f"Analyzing repository: {repo_path}")
    
    for root, dirs, files in os.walk(repo_path):
        # Skip entire directories we don't want
        dirs[:] = [d for d in dirs if d.lower() not in {
            'een', '__pycache__', 'node_modules', '.git', '.github',
            'legacy', 'backup', 'backups', 'migration', 'archived_navigation',
            '.pytest_cache', '.streamlit', '.vercel', 'een.egg-info'
        }]
        
        for file in files:
            file_path = os.path.join(root, file)
            
            if should_include_file(file_path):
                rel_path = os.path.relpath(file_path, repo_path)
                file_size = get_file_size(file_path)
                ext = pathlib.Path(file_path).suffix.lower()
                
                file_info = {
                    'path': rel_path,
                    'size': file_size,
                    'extension': ext
                }
                
                # Categorize the file
                if ext == '.py':
                    categories['core_python'].append(file_info)
                elif ext == '.html':
                    categories['website_html'].append(file_info)
                elif ext == '.md' and '/docs/' in rel_path.replace('\\', '/').lower():
                    categories['research_docs'].append(file_info)
                elif ext in {'.lean', '.r'}:
                    categories['formal_proofs'].append(file_info)
    
    # Sort each category by size (largest first)
    for category in categories.values():
        category.sort(key=lambda x: x['size'], reverse=True)
    
    return categories

def generate_focused_report(categories: Dict[str, List[Dict]], github_user: str, repo_name: str) -> str:
    """Generate focused markdown report"""
    
    lines = []
    lines.append("# Een Repository - Core Source Code Analysis")
    lines.append("")
    lines.append("Focused analysis of the core Python implementations and website pages you actually wrote.")
    lines.append("Excludes all legacy, backup, duplicate, and generated files.")
    lines.append("")
    
    # Summary
    total_files = sum(len(files) for files in categories.values())
    lines.append("## Summary")
    lines.append(f"- **Total Core Files**: {total_files}")
    lines.append(f"- **Core Python Files**: {len(categories['core_python'])}")
    lines.append(f"- **Website HTML Pages**: {len(categories['website_html'])}")
    lines.append(f"- **Research Documents**: {len(categories['research_docs'])}")
    lines.append(f"- **Formal Proofs**: {len(categories['formal_proofs'])}")
    lines.append("")
    
    # Core Python Files
    if categories['core_python']:
        lines.append("## Core Python Implementations")
        lines.append("")
        lines.append("Your main Python implementations for Unity Mathematics (1+1=1):")
        lines.append("")
        
        for file_info in categories['core_python']:
            path = file_info['path'].replace('\\', '/')
            size_kb = file_info['size'] / 1024
            raw_link = f"https://raw.githubusercontent.com/{github_user}/{repo_name}/main/{path}"
            lines.append(f"- **{path}** ({size_kb:.1f} KB) - {raw_link}")
        
        lines.append("")
    
    # Website HTML Pages
    if categories['website_html']:
        lines.append("## Website HTML Pages")
        lines.append("")
        lines.append("Your Unity Mathematics website pages:")
        lines.append("")
        
        for file_info in categories['website_html']:
            path = file_info['path'].replace('\\', '/')
            size_kb = file_info['size'] / 1024
            
            live_link = f"https://{github_user}.github.io/{repo_name}/{path}"
            raw_link = f"https://raw.githubusercontent.com/{github_user}/{repo_name}/main/{path}"
            
            lines.append(f"### {path}")
            lines.append(f"**Size**: {size_kb:.1f} KB")
            lines.append(f"- **Live Page**: {live_link}")
            lines.append(f"- **Raw HTML**: {raw_link}")
            lines.append("")
    
    # Research Documents
    if categories['research_docs']:
        lines.append("## Research Documents & Reports")
        lines.append("")
        lines.append("Research papers, reports, and documentation:")
        lines.append("")
        
        for file_info in categories['research_docs']:
            path = file_info['path'].replace('\\', '/')
            size_kb = file_info['size'] / 1024
            raw_link = f"https://raw.githubusercontent.com/{github_user}/{repo_name}/main/{path}"
            lines.append(f"- **{path}** ({size_kb:.1f} KB) - {raw_link}")
        
        lines.append("")
    
    # Formal Proofs
    if categories['formal_proofs']:
        lines.append("## Formal Proofs")
        lines.append("")
        lines.append("Lean and R formal proofs of Unity Mathematics:")
        lines.append("")
        
        for file_info in categories['formal_proofs']:
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
    
    # Analyze repository with focused approach
    categories = analyze_repository(repo_path)
    
    total_files = sum(len(files) for files in categories.values())
    print(f"Found {total_files} core files total:")
    print(f"  - Python files: {len(categories['core_python'])}")
    print(f"  - HTML pages: {len(categories['website_html'])}")
    print(f"  - Research docs: {len(categories['research_docs'])}")
    print(f"  - Formal proofs: {len(categories['formal_proofs'])}")
    
    # Generate focused report
    report_content = generate_focused_report(categories, github_user, repo_name)
    
    # Save report
    output_file = os.path.join(repo_path, "docs", "core_source_analysis.md")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"Focused analysis complete! Report saved to: {output_file}")
    return output_file

if __name__ == "__main__":
    main()