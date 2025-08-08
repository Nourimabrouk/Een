#!/usr/bin/env python3
"""
Een Background Startup Script
============================

Start all Een framework services in the background for global access.
This script ensures the framework is always available and running.
"""

import os
import sys
import time
import json
import subprocess
import threading
import signal
import atexit
from pathlib import Path
from datetime import datetime
import argparse
import psutil

class EenBackgroundManager:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.services = {}
        self.running = False
        self.config = {
            "services": {
                "api_server": {
                    "enabled": True,
                    "command": [sys.executable, "een_server.py"],
                    "port": 8000,
                    "auto_restart": True,
                    "max_restarts": 5
                },
                "dashboard": {
                    "enabled": True,
                    "command": [sys.executable, "-m", "streamlit", "run", "viz/streamlit_app.py", 
                               "--server.port", "8501", "--server.address", "0.0.0.0"],
                    "port": 8501,
                    "auto_restart": True,
                    "max_restarts": 5
                },
                "mcp_server": {
                    "enabled": True,
                    "command": [sys.executable, "config/mcp_consciousness_server.py"],
                    "port": 3000,
                    "auto_restart": True,
                    "max_restarts": 5
                },
                "monitor": {
                    "enabled": True,
                    "command": [sys.executable, "een_monitor.py"],
                    "auto_restart": True,
                    "max_restarts": 3
                },
                "meta_agent": {
                    "enabled": False,
                    "command": [
                        sys.executable,
                        "scripts/meta_agent_background_launcher.py",
                        "--processes",
                        "2",
                        "--duration",
                        "0",
                    ],
                    "auto_restart": True,
                    "max_restarts": 5
                }
            },
            "global_access": {
                "enabled": True,
                "create_aliases": True,
                "add_to_path": True
            },
            "health_check": {
                "enabled": True,
                "interval": 30
            }
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        atexit.register(self.cleanup)
        
        # Create logs directory
        self.logs_dir = self.project_root / "logs"
        self.logs_dir.mkdir(exist_ok=True)
    
    def start(self):
        """Start all services in background"""
        print("🚀 Starting Een Framework in Background")
        print("=" * 50)
        
        self.running = True
        
        # Start all enabled services
        for service_name, service_config in self.config["services"].items():
            if service_config["enabled"]:
                self.start_service(service_name, service_config)
        
        # Setup global access
        if self.config["global_access"]["enabled"]:
            self.setup_global_access()
        
        # Start health monitoring
        if self.config["health_check"]["enabled"]:
            health_thread = threading.Thread(target=self.health_monitor_loop, daemon=True)
            health_thread.start()
        
        print("✅ Een Framework started successfully!")
        print()
        print("🌍 Access Points:")
        print(f"   API Server: http://localhost:{self.config['services']['api_server']['port']}")
        print(f"   Dashboard: http://localhost:{self.config['services']['dashboard']['port']}")
        print(f"   MCP Server: localhost:{self.config['services']['mcp_server']['port']}")
        print()
        print("📋 Management Commands:")
        print("   Status: python een_monitor.py --status")
        print("   Stop: python stop_een.py")
        print("   Restart: python restart_een.py")
        print()
        print("🎯 Global Access:")
        print("   Command line: een")
        print("   Python: import een")
        print()
        print("🔄 Services will auto-restart if they fail")
        print("📊 Monitor logs in logs/ directory")
        
        # Keep running
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Received interrupt signal")
        finally:
            self.stop()
    
    def start_service(self, service_name: str, service_config: dict):
        """Start a service in background"""
        try:
            # Create log files
            stdout_log = self.logs_dir / f"{service_name}_stdout.log"
            stderr_log = self.logs_dir / f"{service_name}_stderr.log"
            
            # Start process
            process = subprocess.Popen(
                service_config["command"],
                cwd=self.project_root,
                stdout=open(stdout_log, 'a'),
                stderr=open(stderr_log, 'a'),
                text=True,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )
            
            self.services[service_name] = {
                "process": process,
                "config": service_config,
                "start_time": datetime.now(),
                "restart_count": 0,
                "last_restart": None
            }
            
            print(f"✅ Started {service_name} (PID: {process.pid})")
            
            # Start restart monitoring thread
            if service_config.get("auto_restart"):
                restart_thread = threading.Thread(
                    target=self.monitor_service_restart,
                    args=(service_name,),
                    daemon=True
                )
                restart_thread.start()
            
        except Exception as e:
            print(f"❌ Failed to start {service_name}: {e}")
    
    def monitor_service_restart(self, service_name: str):
        """Monitor and restart service if needed"""
        service_info = self.services[service_name]
        service_config = service_info["config"]
        
        while self.running:
            process = service_info["process"]
            
            # Check if process is still running
            if process.poll() is not None:
                restart_count = service_info["restart_count"]
                max_restarts = service_config.get("max_restarts", 3)
                
                if restart_count < max_restarts:
                    print(f"🔄 Restarting {service_name} (attempt {restart_count + 1}/{max_restarts})")
                    self.restart_service(service_name)
                    restart_count += 1
                    service_info["restart_count"] = restart_count
                    service_info["last_restart"] = datetime.now()
                else:
                    print(f"❌ {service_name} exceeded max restarts ({max_restarts})")
                    break
            
            time.sleep(5)
    
    def restart_service(self, service_name: str):
        """Restart a specific service"""
        service_info = self.services[service_name]
        service_config = service_info["config"]
        
        # Stop existing process
        process = service_info["process"]
        try:
            if os.name != 'nt':
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=10)
        except:
            try:
                if os.name != 'nt':
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                else:
                    process.kill()
            except:
                pass
        
        # Start new process
        self.start_service(service_name, service_config)
    
    def setup_global_access(self):
        """Setup global access to Een framework"""
        print("🌍 Setting up global access...")
        
        # Create global entry point
        global_entry = f'''#!/usr/bin/env python3
"""
Een Global Entry Point
=====================

Access the Een framework from anywhere.
"""

import sys
from pathlib import Path

# Add Een to Python path
een_path = Path("{self.project_root.absolute()}")
sys.path.insert(0, str(een_path))

# Import main modules
try:
    from core.mathematical.unity_mathematics import UnityMathematics
    from src.consciousness.consciousness_engine import ConsciousnessEngine
    print("✅ Een Framework loaded successfully!")
except ImportError as e:
    print(f"⚠️  Some modules not available: {e}")

if __name__ == "__main__":
    print("🎯 Een Framework - Global Access")
    print("=" * 40)
    print("Available modules:")
    print("  - Unity Mathematics")
    print("  - Consciousness Engine")
    print("  - Bayesian Statistics")
    print("  - Visualization Tools")
    print()
    print("Access points:")
    print("  - API: http://localhost:8000")
    print("  - Dashboard: http://localhost:8501")
    print("  - MCP: localhost:3000")
'''
        
        # Write global entry point
        global_path = self.project_root / "een_global.py"
        with open(global_path, 'w') as f:
            f.write(global_entry)
        
        # Make executable on Unix
        if os.name != 'nt':
            os.chmod(global_path, 0o755)
        
        # Create system-wide aliases
        if self.config["global_access"]["create_aliases"]:
            self.create_system_aliases()
        
        print("✅ Global access configured")
    
    def create_system_aliases(self):
        """Create system-wide aliases for Een"""
        if os.name == 'nt':  # Windows
            # Create batch file
            batch_content = f'''@echo off
python "{self.project_root.absolute()}\\een_global.py" %*
'''
            batch_path = self.project_root / "een.bat"
            with open(batch_path, 'w') as f:
                f.write(batch_content)
            
            print("📝 For Windows: Add to PATH:")
            print(f"   {self.project_root.absolute()}")
            
        else:  # Unix/Linux/macOS
            # Create shell script
            shell_content = f'''#!/bin/bash
python3 "{self.project_root.absolute()}/een_global.py" "$@"
'''
            shell_path = self.project_root / "een"
            with open(shell_path, 'w') as f:
                f.write(shell_content)
            
            os.chmod(shell_path, 0o755)
            
            # Try to create global symlink
            try:
                global_path = Path("/usr/local/bin/een")
                if not global_path.exists():
                    global_path.symlink_to(shell_path.absolute())
                    print("✅ Global alias created at /usr/local/bin/een")
                else:
                    print("⚠️  Global alias already exists at /usr/local/bin/een")
            except PermissionError:
                print("📝 For Unix: Add to ~/.bashrc or ~/.zshrc:")
                print(f"   export PATH=\"$PATH:{self.project_root.absolute()}\"")
    
    def health_monitor_loop(self):
        """Health monitoring loop"""
        while self.running:
            try:
                self.check_services_health()
                time.sleep(self.config["health_check"]["interval"])
            except Exception as e:
                print(f"Health check error: {e}")
    
    def check_services_health(self):
        """Check health of all services"""
        for service_name, service_info in self.services.items():
            process = service_info["process"]
            if process.poll() is not None:
                print(f"⚠️  Service {service_name} is not running")
    
    def stop(self):
        """Stop all services"""
        print("🛑 Stopping Een Framework...")
        self.running = False
        
        for service_name, service_info in self.services.items():
            process = service_info["process"]
            try:
                if os.name != 'nt':
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                else:
                    process.terminate()
                process.wait(timeout=10)
                print(f"✅ Stopped {service_name}")
            except:
                try:
                    if os.name != 'nt':
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    else:
                        process.kill()
                    print(f"🔄 Force killed {service_name}")
                except:
                    print(f"❌ Failed to stop {service_name}")
        
        print("👋 Een Framework stopped")
    
    def signal_handler(self, signum, frame):
        """Handle system signals"""
        print(f"\n🛑 Received signal {signum}")
        self.stop()
    
    def cleanup(self):
        """Cleanup on exit"""
        if self.running:
            self.stop()
    
    def get_status(self):
        """Get status of all services"""
        status = {}
        for service_name, service_info in self.services.items():
            process = service_info["process"]
            status[service_name] = {
                "running": process.poll() is None,
                "pid": process.pid,
                "restart_count": service_info["restart_count"],
                "uptime": (datetime.now() - service_info["start_time"]).total_seconds()
            }
        return status
    
    def print_status(self):
        """Print status of all services"""
        status = self.get_status()
        
        print("🎯 Een Framework Status")
        print("=" * 30)
        
        for service_name, service_status in status.items():
            icon = "🟢" if service_status["running"] else "🔴"
            status_text = "Running" if service_status["running"] else "Stopped"
            uptime = f"{service_status['uptime']:.0f}s" if service_status["running"] else "N/A"
            
            print(f"{icon} {service_name}: {status_text}")
            print(f"   PID: {service_status['pid']}")
            print(f"   Uptime: {uptime}")
            print(f"   Restarts: {service_status['restart_count']}")
            print()

def main():
    parser = argparse.ArgumentParser(description="Start Een Framework in background")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    parser.add_argument("--stop", action="store_true", help="Stop all services")
    parser.add_argument("--restart", action="store_true", help="Restart all services")
    
    args = parser.parse_args()
    
    manager = EenBackgroundManager()
    
    if args.status:
        manager.print_status()
        return
    
    if args.stop:
        # Load existing manager and stop
        manager.stop()
        return
    
    if args.restart:
        # Stop and restart
        manager.stop()
        time.sleep(2)
        manager.start()
        return
    
    # Start services
    manager.start()

if __name__ == "__main__":
    main() 