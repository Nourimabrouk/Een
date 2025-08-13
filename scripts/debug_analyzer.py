"""
Debug repository analyzer
"""

import os
import pathlib

def main():
    repo_path = r"C:\Users\Nouri\Documents\GitHub\Een"
    
    print(f"Starting in: {repo_path}")
    
    count = 0
    for root, dirs, files in os.walk(repo_path):
        # Remove virtual env from dirs to skip it
        if 'een' in dirs:
            dirs.remove('een')
            print(f"Skipping virtual env: {os.path.join(root, 'een')}")
        
        for file in files:
            if count < 10:  # Only process first 10 files for debugging
                file_path = os.path.join(root, file)
                ext = pathlib.Path(file_path).suffix.lower()
                
                if ext in {'.py', '.html', '.js', '.css', '.md'}:
                    rel_path = os.path.relpath(file_path, repo_path)
                    size = os.path.getsize(file_path)
                    print(f"Found: {rel_path} ({size} bytes)")
                    count += 1
                    
                    if file.lower().endswith('.html'):
                        print(f"  -> HTML file: {file}")
                    elif file.lower().endswith('.py'):
                        print(f"  -> Python file: {file}")
        
        if count >= 10:
            break
    
    print(f"Debug complete. Found {count} files.")

if __name__ == "__main__":
    main()