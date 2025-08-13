"""
Final Focused Repository Analysis for Een
Only the core Python code YOU wrote and main website HTML pages
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

def analyze_repository(repo_path: str) -> Dict[str, List[Dict]]:
    """Analyze repository with very focused approach"""
    
    categories = {
        'main_python_files': [],
        'website_html_pages': [],
        'research_reports': [],
        'formal_proofs': []
    }
    
    print(f"Analyzing repository: {repo_path}")
    
    # Define the key directories to focus on
    focus_areas = {
        'src/': 'main_python_files',
        'api/': 'main_python_files', 
        'dashboards/': 'main_python_files',
        'consciousness/': 'main_python_files',
        'meta/': 'main_python_files',
        'ml_framework/': 'main_python_files',
        'scripts/': 'main_python_files',
        'examples/': 'main_python_files',
        'experiments/': 'main_python_files',
        'formal_proofs/': 'formal_proofs',
        'website/': 'website_html_pages',
        'docs/research/': 'research_reports',
        'docs/reports/': 'research_reports',
        'internal/reports/': 'research_reports'
    }
    
    for focus_dir, category in focus_areas.items():
        full_path = os.path.join(repo_path, focus_dir.replace('/', os.sep))
        
        if os.path.exists(full_path):
            print(f"Processing: {focus_dir}")
            
            for root, dirs, files in os.walk(full_path):
                # Skip backup/legacy directories even within focus areas
                dirs[:] = [d for d in dirs if not any(skip in d.lower() for skip in 
                          ['backup', 'legacy', 'archived', '__pycache__', 'node_modules'])]
                
                for file in files:
                    # Skip problematic files
                    if file in ['nul', 'con', 'aux', 'prn'] or file.startswith('.'):
                        continue
                    
                    try:
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, repo_path)
                        ext = pathlib.Path(file_path).suffix.lower()
                        filename = os.path.basename(file_path).lower()
                    except (ValueError, OSError):
                        continue  # Skip files that cause path errors
                    
                    # Skip obvious backup/test/config files
                    if any(skip in filename for skip in ['backup', 'test_', '_test', 'config', '__init__.py', '.bat', '.sh']):
                        continue
                    
                    # Include based on category and file type
                    should_include = False
                    
                    if category == 'main_python_files' and ext == '.py':
                        should_include = True
                    elif category == 'website_html_pages' and ext == '.html':
                        # Only main website pages, not test/backup files
                        if not any(skip in filename for skip in ['test', 'backup', 'copy', 'archived']):
                            should_include = True
                    elif category == 'formal_proofs' and ext in {'.lean', '.py', '.r'}:
                        should_include = True
                    elif category == 'research_reports' and ext in {'.md', '.docx'}:
                        should_include = True
                    
                    if should_include:
                        file_info = {
                            'path': rel_path,
                            'size': get_file_size(file_path),
                            'extension': ext
                        }
                        categories[category].append(file_info)
    
    # Also include some key root files
    root_files = ['index.html', 'README.md', 'CLAUDE.md', 'SECURITY.md']
    for file in root_files:
        file_path = os.path.join(repo_path, file)
        if os.path.exists(file_path):
            rel_path = os.path.relpath(file_path, repo_path)
            ext = pathlib.Path(file_path).suffix.lower()
            file_info = {
                'path': rel_path,
                'size': get_file_size(file_path),
                'extension': ext
            }
            
            if ext == '.html':
                categories['website_html_pages'].append(file_info)
            elif ext == '.md':
                categories['research_reports'].append(file_info)
    
    # Sort each category by size (largest first)
    for category in categories.values():
        category.sort(key=lambda x: x['size'], reverse=True)
    
    return categories

def generate_final_report(categories: Dict[str, List[Dict]], github_user: str, repo_name: str) -> str:
    """Generate final focused markdown report"""
    
    lines = []
    lines.append("# Een Repository - Core Source Code & Website Analysis")
    lines.append("")
    lines.append("**Focused analysis of core Python implementations and website pages.**")
    lines.append("*Excludes legacy, backup, test, and generated files.*")
    lines.append("")
    
    # Summary
    total_files = sum(len(files) for files in categories.values())
    lines.append("## 📊 Summary")
    lines.append(f"- **Total Core Files**: {total_files}")
    lines.append(f"- **Python Implementations**: {len(categories['main_python_files'])}")
    lines.append(f"- **Website HTML Pages**: {len(categories['website_html_pages'])}")
    lines.append(f"- **Research Reports**: {len(categories['research_reports'])}")
    lines.append(f"- **Formal Proofs**: {len(categories['formal_proofs'])}")
    lines.append("")
    
    # Main Python Files - Core implementations
    if categories['main_python_files']:
        lines.append("## 🐍 Core Python Implementations")
        lines.append("")
        lines.append("Your main Unity Mathematics (1+1=1) Python implementations:")
        lines.append("")
        
        # Group by directory for better organization
        by_dir = {}
        for file_info in categories['main_python_files']:
            dir_name = os.path.dirname(file_info['path']).split(os.sep)[0] if os.sep in file_info['path'] else 'root'
            if dir_name not in by_dir:
                by_dir[dir_name] = []
            by_dir[dir_name].append(file_info)
        
        # Sort directories by total size
        dir_sizes = [(dir_name, sum(f['size'] for f in files)) for dir_name, files in by_dir.items()]
        dir_sizes.sort(key=lambda x: x[1], reverse=True)
        
        for dir_name, total_size in dir_sizes:
            files = by_dir[dir_name]
            size_mb = total_size / (1024 * 1024)
            
            lines.append(f"### {dir_name.title()}/ Directory")
            lines.append(f"*{len(files)} files • {size_mb:.1f} MB*")
            lines.append("")
            
            for file_info in files:
                path = file_info['path'].replace('\\', '/')
                size_kb = file_info['size'] / 1024
                raw_link = f"https://raw.githubusercontent.com/{github_user}/{repo_name}/main/{path}"
                lines.append(f"- **{os.path.basename(path)}** ({size_kb:.1f} KB) - [{path}]({raw_link})")
            
            lines.append("")
    
    # Website HTML Pages
    if categories['website_html_pages']:
        lines.append("## 🌐 Website HTML Pages")
        lines.append("")
        lines.append("Unity Mathematics website pages (accessible via GitHub Pages):")
        lines.append("")
        
        for file_info in categories['website_html_pages']:
            path = file_info['path'].replace('\\', '/')
            size_kb = file_info['size'] / 1024
            filename = os.path.basename(path)
            
            live_link = f"https://{github_user}.github.io/{repo_name}/{path}"
            raw_link = f"https://raw.githubusercontent.com/{github_user}/{repo_name}/main/{path}"
            
            lines.append(f"### {filename}")
            lines.append(f"**Size**: {size_kb:.1f} KB • **Path**: `{path}`")
            lines.append(f"- 🌐 **Live Page**: {live_link}")
            lines.append(f"- 📄 **Raw HTML**: {raw_link}")
            lines.append("")
    
    # Research Reports & Documentation
    if categories['research_reports']:
        lines.append("## 📚 Research Reports & Documentation")
        lines.append("")
        lines.append("Research papers, reports, and important documentation:")
        lines.append("")
        
        for file_info in categories['research_reports']:
            path = file_info['path'].replace('\\', '/')
            size_kb = file_info['size'] / 1024
            raw_link = f"https://raw.githubusercontent.com/{github_user}/{repo_name}/main/{path}"
            lines.append(f"- **{os.path.basename(path)}** ({size_kb:.1f} KB) - [{path}]({raw_link})")
        
        lines.append("")
    
    # Formal Proofs
    if categories['formal_proofs']:
        lines.append("## 🔬 Formal Proofs")
        lines.append("")
        lines.append("Lean, Python, and R formal proofs of Unity Mathematics:")
        lines.append("")
        
        for file_info in categories['formal_proofs']:
            path = file_info['path'].replace('\\', '/')
            size_kb = file_info['size'] / 1024
            raw_link = f"https://raw.githubusercontent.com/{github_user}/{repo_name}/main/{path}"
            lines.append(f"- **{os.path.basename(path)}** ({size_kb:.1f} KB) - [{path}]({raw_link})")
        
        lines.append("")
    
    lines.append("---")
    lines.append("*Generated by focused repository analyzer - excludes duplicates, backups, and generated files*")
    
    return "\n".join(lines)

def main():
    repo_path = r"C:\Users\Nouri\Documents\GitHub\Een"
    github_user = "NouriMabrouk"
    repo_name = "Een"
    
    # Analyze repository with focused approach
    categories = analyze_repository(repo_path)
    
    total_files = sum(len(files) for files in categories.values())
    print(f"\\nFOCUSED ANALYSIS COMPLETE:")
    print(f"  CONFIRMED: Python files: {len(categories['main_python_files'])}")
    print(f"  CONFIRMED: HTML pages: {len(categories['website_html_pages'])}")
    print(f"  CONFIRMED: Research docs: {len(categories['research_reports'])}")
    print(f"  CONFIRMED: Formal proofs: {len(categories['formal_proofs'])}")
    print(f"  TOTAL: {total_files} core files")
    
    # Generate focused report
    report_content = generate_final_report(categories, github_user, repo_name)
    
    # Save report
    output_file = os.path.join(repo_path, "docs", "CORE_SOURCE_ANALYSIS.md")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"\\nREPORT SAVED: {output_file}")
    return output_file

if __name__ == "__main__":
    main()