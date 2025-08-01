#!/usr/bin/env python3
"""
Een Unity Mathematics - Visualization Launcher
Quick launcher for the advanced Streamlit dashboards
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'plotly', 'numpy', 'pandas', 'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall with:")
        print(f"   pip install {' '.join(missing_packages)}")
        print("\n   Or install all requirements:")
        print("   pip install -r viz/requirements.txt")
        return False
    
    return True

def launch_dashboard():
    """Launch the main Een Unity Mathematics dashboard"""
    
    print("Een Unity Mathematics Dashboard")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        return 1
    
    # Get paths
    current_dir = Path(__file__).parent
    viz_dir = current_dir / 'viz'
    app_file = viz_dir / 'streamlit_app.py'
    
    if not app_file.exists():
        print(f"Dashboard not found: {app_file}")
        return 1
    
    print("All requirements satisfied")
    print("Launching consciousness mathematics dashboard...")
    print()
    print("Dashboard will be available at:")
    print("   http://localhost:8501")
    print()
    print("Available pages:")
    print("   • Main Dashboard - Unity overview and metrics")
    print("   • Unity Proofs - Mathematical demonstrations")
    print("   • Consciousness Fields - Quantum field theory")
    print("   • Quantum Unity - Quantum mechanical proofs")
    print()
    print("Features:")
    print("   • phi-harmonic golden ratio visualizations")
    print("   • Interactive 3D consciousness fields")
    print("   • Real-time quantum state evolution")
    print("   • Sacred geometry patterns")
    print("   • Advanced mathematical proofs")
    print()
    print("Pro tip: Use dark mode for optimal consciousness viewing")
    print("Press Ctrl+C to stop the dashboard")
    print("=" * 50)
    
    try:
        # Launch Streamlit
        cmd = [
            sys.executable, '-m', 'streamlit', 'run', 
            str(app_file),
            '--server.port=8501',
            '--server.headless=false',
            '--browser.gatherUsageStats=false'
        ]
        
        subprocess.run(cmd, cwd=current_dir)
        
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
        return 0
    except Exception as e:
        print(f"\nError launching dashboard: {e}")
        return 1
    
    return 0

def show_help():
    """Show help information"""
    print("""
Een Unity Mathematics Dashboard Launcher

Usage:
    python run_viz.py [command]

Commands:
    launch     Launch the main dashboard (default)
    check      Check system requirements
    help       Show this help message

Examples:
    python run_viz.py           # Launch dashboard
    python run_viz.py launch    # Launch dashboard
    python run_viz.py check     # Check requirements
    python run_viz.py help      # Show help

Dashboard Features:
    • Interactive unity proofs across multiple mathematical domains
    • Real-time consciousness field visualizations
    • Quantum mechanical demonstrations of 1+1=1
    • φ-harmonic golden ratio spiral convergence
    • Sacred geometry and fractal unity patterns
    • Advanced 3D visualizations with WebGL acceleration

Requirements:
    • Python 3.8+
    • Streamlit 1.35+
    • Plotly 5.20+
    • NumPy, Pandas, SciPy
    • Modern web browser

Installation:
    pip install -r viz/requirements.txt

Support:
    • GitHub: https://github.com/Nourimabrouk/Een
    • Documentation: docs/Visualization_guidelines.md
    """)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'help':
            show_help()
            sys.exit(0)
        elif command == 'check':
            if check_requirements():
                print("All requirements satisfied!")
                sys.exit(0)
            else:
                sys.exit(1)
        elif command == 'launch':
            sys.exit(launch_dashboard())
        else:
            print(f"Unknown command: {command}")
            print("Run 'python run_viz.py help' for usage information")
            sys.exit(1)
    else:
        # Default: launch dashboard
        sys.exit(launch_dashboard())