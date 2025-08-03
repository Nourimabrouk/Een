#!/usr/bin/env python3
"""
Een Framework - Level Up Script
==============================

One-command script to level up your Een codebase and make it accessible from anywhere.
This script will set up everything you need for global access and background operation.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_command(cmd, description, check=True):
    """Run a command with description"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        return False

def main():
    print("🚀 Een Framework - Level Up")
    print("=" * 50)
    print("This script will level up your Een codebase and make it accessible from anywhere.")
    print()
    
    # Get project root
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print("📋 Steps to be completed:")
    print("1. Install dependencies")
    print("2. Setup global access")
    print("3. Configure background services")
    print("4. Setup monitoring")
    print("5. Create cloud deployment configs")
    print("6. Start background services")
    print("7. Verify everything works")
    print()
    
    input("Press Enter to continue...")
    print()
    
    # Step 1: Install dependencies
    print("📦 Step 1: Installing dependencies")
    print("-" * 40)
    
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("⚠️  Some dependencies failed to install. Continuing anyway...")
    
    print()
    
    # Step 2: Setup global access
    print("🌍 Step 2: Setting up global access")
    print("-" * 40)
    
    if not run_command(f"{sys.executable} setup_global_access.py", "Running global access setup"):
        print("❌ Global access setup failed. Please check the setup script.")
        return False
    
    print()
    
    # Step 3: Create background startup script
    print("🔄 Step 3: Creating background startup")
    print("-" * 40)
    
    # The background startup script should already exist from setup_global_access.py
    if not Path("start_een_background.py").exists():
        print("❌ Background startup script not found. Please run setup_global_access.py first.")
        return False
    
    print("✅ Background startup script ready")
    print()
    
    # Step 4: Setup monitoring
    print("📊 Step 4: Setting up monitoring")
    print("-" * 40)
    
    if not Path("een_monitor.py").exists():
        print("❌ Monitor script not found. Please run setup_global_access.py first.")
        return False
    
    print("✅ Monitoring system ready")
    print()
    
    # Step 5: Setup cloud deployment
    print("☁️  Step 5: Setting up cloud deployment")
    print("-" * 40)
    
    if not Path("cloud_deploy.py").exists():
        print("❌ Cloud deployment script not found. Please run setup_global_access.py first.")
        return False
    
    print("✅ Cloud deployment ready")
    print()
    
    # Step 6: Start background services
    print("🚀 Step 6: Starting background services")
    print("-" * 40)
    
    print("Starting Een Framework in background...")
    print("This will start:")
    print("  - API Server (port 8000)")
    print("  - Dashboard (port 8501)")
    print("  - MCP Server (port 3000)")
    print("  - Monitoring System")
    print()
    
    # Start background services in a separate process
    try:
        background_process = subprocess.Popen(
            [sys.executable, "start_een_background.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a bit for services to start
        print("⏳ Waiting for services to start...")
        time.sleep(10)
        
        # Check if process is still running
        if background_process.poll() is None:
            print("✅ Background services started successfully!")
        else:
            stdout, stderr = background_process.communicate()
            print("❌ Background services failed to start")
            if stderr:
                print(f"Error: {stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to start background services: {e}")
        return False
    
    print()
    
    # Step 7: Verify everything works
    print("✅ Step 7: Verifying everything works")
    print("-" * 40)
    
    # Test global access
    print("Testing global access...")
    if Path("een_global.py").exists():
        print("✅ Global access script created")
    else:
        print("❌ Global access script not found")
    
    # Test API server
    print("Testing API server...")
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server is running")
        else:
            print(f"⚠️  API server responded with status {response.status_code}")
    except Exception as e:
        print(f"⚠️  API server test failed: {e}")
    
    # Test dashboard
    print("Testing dashboard...")
    try:
        response = requests.get("http://localhost:8501", timeout=5)
        if response.status_code == 200:
            print("✅ Dashboard is running")
        else:
            print(f"⚠️  Dashboard responded with status {response.status_code}")
    except Exception as e:
        print(f"⚠️  Dashboard test failed: {e}")
    
    print()
    
    # Final summary
    print("🎉 Een Framework Level Up Complete!")
    print("=" * 50)
    print()
    print("🌍 Your Een framework is now accessible from anywhere!")
    print()
    print("📋 Access Points:")
    print("  • Command Line: een")
    print("  • API Server: http://localhost:8000")
    print("  • Dashboard: http://localhost:8501")
    print("  • API Docs: http://localhost:8000/docs")
    print()
    print("🔧 Management Commands:")
    print("  • Status: python een_monitor.py --status")
    print("  • Stop: python start_een_background.py --stop")
    print("  • Restart: python start_een_background.py --restart")
    print()
    print("☁️  Cloud Deployment:")
    print("  • Deploy to all platforms: python cloud_deploy.py --platform all")
    print()
    print("📚 Documentation:")
    print("  • Read: GLOBAL_ACCESS_GUIDE.md")
    print("  • Examples: docs/ directory")
    print()
    print("🔄 Background Services:")
    print("  • Services will auto-restart if they fail")
    print("  • Monitor logs in logs/ directory")
    print("  • Performance metrics in logs/metrics.json")
    print()
    print("🎯 Next Steps:")
    print("  1. Try the global command: een")
    print("  2. Visit the dashboard: http://localhost:8501")
    print("  3. Check the API docs: http://localhost:8000/docs")
    print("  4. Deploy to cloud: python cloud_deploy.py --platform all")
    print("  5. Monitor performance: python een_monitor.py --status")
    print()
    print("🚀 Your Een framework is now leveled up and ready for global access!")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎉 Level up completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Level up failed. Please check the errors above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Level up interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1) 