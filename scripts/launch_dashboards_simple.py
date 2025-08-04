#!/usr/bin/env python3
"""
Simple Dashboard Launcher for Een Unity Mathematics
Launch all Streamlit dashboards without complex rich formatting
"""

import subprocess
import time
import webbrowser
import sys
import os
from pathlib import Path

def find_streamlit_apps():
    """Find all Streamlit applications"""
    apps = []
    
    # Search patterns
    patterns = [
        "viz/streamlit_app.py",
        "src/dashboards/*streamlit*.py", 
        "src/dashboards/*dashboard*.py"
    ]
    
    for pattern in patterns:
        for file_path in Path(".").glob(pattern):
            if file_path.exists() and file_path.suffix == '.py':
                apps.append(str(file_path))
    
    return apps

def launch_streamlit_app(app_path, port, app_name="Dashboard", background=True):
    """Launch a single Streamlit app"""
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            app_path,
            "--server.port", str(port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        print(f"Starting {app_name} on port {port}...")
        print(f"Command: {' '.join(cmd)}")
        
        if background:
            # Launch in background
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path.cwd()
            )
            time.sleep(2)  # Give it time to start
            return process, port
        else:
            # Launch in foreground (blocking)
            subprocess.run(cmd, cwd=Path.cwd())
            return None, port
            
    except Exception as e:
        print(f"Failed to launch {app_name}: {e}")
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
        app_name = Path(app_path).stem.replace('_', ' ').title()
        
        process, actual_port = launch_streamlit_app(app_path, port, app_name, background=True)
        
        if process and actual_port:
            processes.append((process, actual_port, app_name))
            print(f"✓ {app_name} started on http://localhost:{actual_port}")
        else:
            print(f"✗ Failed to start {app_name}")
    
    if processes:
        print("\n" + "=" * 60)
        print("DASHBOARD STATUS:")
        print("=" * 60)
        
        for process, port, name in processes:
            print(f"• {name}: http://localhost:{port}")
        
        print("\n📱 Opening first dashboard in browser...")
        main_port = processes[0][1]
        webbrowser.open(f"http://localhost:{main_port}")
        
        print("\n🔄 Dashboards running. Press Ctrl+C to stop all...")
        
        try:
            while True:
                time.sleep(1)
                # Check if any processes died
                for process, port, name in processes:
                    if process.poll() is not None:
                        print(f"⚠️ {name} stopped unexpectedly")
        
        except KeyboardInterrupt:
            print("\n🛑 Shutting down dashboards...")
            for process, port, name in processes:
                try:
                    process.terminate()
                    process.wait(timeout=5)
                    print(f"✓ Stopped {name}")
                except:
                    try:
                        process.kill()
                        print(f"⚡ Force-killed {name}")
                    except:
                        print(f"❌ Failed to stop {name}")
            
            print("🌟 All dashboards stopped. Unity preserved!")
    
    else:
        print("❌ No dashboards could be started")

if __name__ == "__main__":
    main()