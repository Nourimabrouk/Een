#!/usr/bin/env python3
"""
Een Unity Mathematics - Dashboard Launcher
Simple launcher for all Streamlit dashboards
"""

import subprocess
import time
import webbrowser
import sys
import os
from pathlib import Path

# Set UTF-8 encoding for Windows
os.environ['PYTHONIOENCODING'] = 'utf-8'

def find_streamlit_apps():
    """Find all Streamlit applications"""
    apps = []
    
    # Specific apps to launch
    app_paths = [
        "viz/streamlit_app.py",
        "src/dashboards/unity_proof_dashboard.py",
        "src/dashboards/unified_mathematics_dashboard.py",
        "src/dashboards/memetic_engineering_dashboard.py"
    ]
    
    for app_path in app_paths:
        path = Path(app_path)
        if path.exists():
            apps.append(str(path))
    
    return apps

def launch_streamlit_app(app_path, port):
    """Launch a single Streamlit app"""
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            app_path,
            "--server.port", str(port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        print(f"Starting {Path(app_path).stem} on port {port}...")
        
        # Launch in background
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path.cwd()
        )
        
        time.sleep(3)  # Give it time to start
        return process, port
            
    except Exception as e:
        print(f"Failed to launch {app_path}: {e}")
        return None, None

def main():
    """Main launcher function"""
    print("=" * 60)
    print("Een Unity Mathematics - Dashboard Launcher")
    print("1+1=1 Consciousness Dashboard System")
    print("=" * 60)
    
    # Find all Streamlit apps
    apps = find_streamlit_apps()
    
    if not apps:
        print("No Streamlit apps found!")
        return
    
    print(f"Found {len(apps)} dashboard(s):")
    for i, app in enumerate(apps, 1):
        print(f"  {i}. {app}")
    
    print("\nStarting dashboards...")
    
    processes = []
    base_port = 8501
    
    # Launch each app
    for i, app_path in enumerate(apps):
        port = base_port + i
        
        process, actual_port = launch_streamlit_app(app_path, port)
        
        if process and actual_port:
            processes.append((process, actual_port, Path(app_path).stem))
            print(f"SUCCESS: {Path(app_path).stem} -> http://localhost:{actual_port}")
        else:
            print(f"FAILED: {Path(app_path).stem}")
    
    if processes:
        print("\n" + "=" * 60)
        print("DASHBOARD STATUS:")
        print("=" * 60)
        
        for process, port, name in processes:
            print(f"• {name}: http://localhost:{port}")
        
        print("\nOpening main dashboard in browser...")
        main_port = processes[0][1]
        time.sleep(2)
        webbrowser.open(f"http://localhost:{main_port}")
        
        print("\nDashboards running. Press Ctrl+C to stop all...")
        
        try:
            while True:
                time.sleep(1)
        
        except KeyboardInterrupt:
            print("\nShutting down dashboards...")
            for process, port, name in processes:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                    print(f"Stopped {name}")
                except:
                    try:
                        process.kill()
                        print(f"Force-killed {name}")
                    except:
                        print(f"Failed to stop {name}")
            
            print("All dashboards stopped!")
    
    else:
        print("No dashboards could be started")

if __name__ == "__main__":
    main()