#!/usr/bin/env python3
"""
Unity Mathematics Complete System Launcher
==========================================

Comprehensive launcher that handles all issues and provides smooth startup.
3000 ELO / 300 IQ Metagamer Agent System with full error resolution.
"""

import os
import sys
import subprocess
import threading
import time
import signal
import logging
from pathlib import Path
import webbrowser
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import our utilities
try:
    from utils.port_manager import get_service_port, check_port_availability
    from core.logging_config import setup_unity_logging
except ImportError:
    # Fallback if modules not available
    def get_service_port(
        service_name: str, preferred_port: Optional[int] = None
    ) -> int:
        return preferred_port or 8000

    def check_port_availability(port: int) -> bool:
        import socket

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(("localhost", port))
                return result != 0
        except Exception:
            return False

    def setup_unity_logging(*args, **kwargs):
        return logging.getLogger("unity_mathematics")


# Setup logging
logger = setup_unity_logging(use_unicode=False)


class UnityCompleteLauncher:
    """Complete Unity Mathematics system launcher with error resolution."""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.processes = []
        self.services = {}
        self.is_running = False

        # Service configurations
        self.services_config = {
            "unity_web_server": {
                "script": "unity_web_server.py",
                "default_port": 5000,
                "description": "Unity Web Server",
            },
            "api_server": {
                "script": "api/main.py",
                "default_port": 8000,
                "description": "API Server",
            },
            "streamlit_dashboard": {
                "script": "dashboards/unity_score_dashboard.py",
                "default_port": 8501,
                "description": "Unity Score Dashboard",
            },
            "unity_proof_dashboard": {
                "script": "src/dashboards/unity_proof_dashboard.py",
                "default_port": 8502,
                "description": "Unity Proof Dashboard",
            },
            "unified_mathematics_dashboard": {
                "script": "src/dashboards/unified_mathematics_dashboard.py",
                "default_port": 8503,
                "description": "Unified Mathematics Dashboard",
            },
            "memetic_engineering_dashboard": {
                "script": "src/dashboards/memetic_engineering_dashboard.py",
                "default_port": 8504,
                "description": "Memetic Engineering Dashboard",
            },
        }

    def check_dependencies(self) -> bool:
        """Check if all dependencies are available."""
        logger.info("🔍 Checking Unity Mathematics dependencies...")

        required_modules = [
            "numpy",
            "scipy",
            "matplotlib",
            "pandas",
            "networkx",
            "streamlit",
            "flask",
            "plotly",
            "torch",
        ]

        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
                logger.info(f"✅ {module}")
            except ImportError:
                missing_modules.append(module)
                logger.warning(f"❌ {module} - missing")

        if missing_modules:
            logger.error(f"Missing modules: {missing_modules}")
            logger.info("Installing missing dependencies...")
            return self.install_dependencies(missing_modules)

        logger.info("✅ All dependencies satisfied")
        return True

    def install_dependencies(self, modules: List[str]) -> bool:
        """Install missing dependencies."""
        try:
            cmd = [sys.executable, "-m", "pip", "install"] + modules
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("✅ Dependencies installed successfully")
                return True
            else:
                logger.error(f"❌ Failed to install dependencies: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"❌ Error installing dependencies: {e}")
            return False

    def start_service(self, service_name: str) -> Optional[subprocess.Popen]:
        """Start a service with proper port management."""
        if service_name not in self.services_config:
            logger.error(f"Unknown service: {service_name}")
            return None

        config = self.services_config[service_name]
        script_path = self.project_root / config["script"]

        if not script_path.exists():
            logger.warning(f"Script not found: {script_path}")
            return None

        # Get available port
        port = get_service_port(service_name, config["default_port"])

        try:
            if service_name.startswith("streamlit"):
                # Start Streamlit service
                cmd = [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(script_path),
                    "--server.port",
                    str(port),
                    "--server.headless",
                    "true",
                    "--browser.gatherUsageStats",
                    "false",
                ]
            else:
                # Start regular Python service
                cmd = [sys.executable, str(script_path)]

            logger.info(f"🚀 Starting {config['description']} on port {port}...")

            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.project_root,
            )

            # Wait a bit for startup
            time.sleep(3)

            if process.poll() is None:
                logger.info(f"✅ {config['description']} started successfully")
                self.processes.append(process)
                self.services[service_name] = {
                    "process": process,
                    "port": port,
                    "description": config["description"],
                }
                return process
            else:
                stdout, stderr = process.communicate()
                logger.error(f"❌ {config['description']} failed to start")
                logger.error(f"STDOUT: {stdout.decode()}")
                logger.error(f"STDERR: {stderr.decode()}")
                return None

        except Exception as e:
            logger.error(f"❌ Error starting {service_name}: {e}")
            return None

    def start_all_services(self) -> bool:
        """Start all Unity Mathematics services."""
        logger.info("🚀 Starting Unity Mathematics Complete System...")

        # Start core services
        core_services = ["unity_web_server", "api_server"]
        for service in core_services:
            if not self.start_service(service):
                logger.warning(f"Failed to start {service}, continuing...")

        # Start dashboard services
        dashboard_services = [
            "streamlit_dashboard",
            "unity_proof_dashboard",
            "unified_mathematics_dashboard",
            "memetic_engineering_dashboard",
        ]

        for service in dashboard_services:
            if not self.start_service(service):
                logger.warning(f"Failed to start {service}, continuing...")

        return len(self.services) > 0

    def open_browsers(self):
        """Open browsers to the running services."""
        time.sleep(5)  # Wait for services to fully start

        # Open main website
        web_port = self.services.get("unity_web_server", {}).get("port", 5000)
        webbrowser.open(f"http://localhost:{web_port}")

        # Open dashboards
        for service_name, service_info in self.services.items():
            if service_name.startswith("streamlit"):
                port = service_info["port"]
                webbrowser.open(f"http://localhost:{port}")

    def monitor_services(self):
        """Monitor running services."""
        while self.is_running:
            for service_name, service_info in list(self.services.items()):
                process = service_info["process"]
                if process.poll() is not None:
                    logger.warning(f"Service {service_name} stopped unexpectedly")
                    del self.services[service_name]

            time.sleep(5)

    def stop_all_services(self):
        """Stop all running services."""
        logger.info("🛑 Stopping Unity Mathematics services...")

        for service_name, service_info in self.services.items():
            process = service_info["process"]
            try:
                process.terminate()
                process.wait(timeout=5)
                logger.info(f"✅ Stopped {service_info['description']}")
            except subprocess.TimeoutExpired:
                process.kill()
                logger.warning(f"⚠️ Force killed {service_info['description']}")
            except Exception as e:
                logger.error(f"❌ Error stopping {service_name}: {e}")

        self.services.clear()
        self.processes.clear()

    def display_status(self):
        """Display current system status."""
        print("\n" + "=" * 60)
        print("🌟 Unity Mathematics Complete System Status")
        print("=" * 60)

        if not self.services:
            print("❌ No services running")
            return

        print(f"✅ {len(self.services)} services running:")
        for service_name, service_info in self.services.items():
            port = service_info["port"]
            description = service_info["description"]
            status = (
                "🟢 Running" if service_info["process"].poll() is None else "🔴 Stopped"
            )
            print(f"  • {description}: http://localhost:{port} {status}")

        print("\n🌐 Access Points:")
        print("  • Main Website: http://localhost:5000")
        print("  • API Documentation: http://localhost:8000/docs")
        print("  • Dashboards: http://localhost:8501-8504")
        print("\n💡 Press Ctrl+C to stop all services")
        print("=" * 60)

    def run(self):
        """Run the complete Unity Mathematics system."""
        try:
            # Check dependencies
            if not self.check_dependencies():
                logger.error("❌ Dependency check failed")
                return False

            # Start services
            if not self.start_all_services():
                logger.error("❌ Failed to start any services")
                return False

            self.is_running = True

            # Display status
            self.display_status()

            # Open browsers
            threading.Thread(target=self.open_browsers, daemon=True).start()

            # Monitor services
            self.monitor_services()

        except KeyboardInterrupt:
            logger.info("\n🛑 Received interrupt signal")
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
        finally:
            self.stop_all_services()
            logger.info("✅ Unity Mathematics Complete System stopped")

        return True


def main():
    """Main entry point."""
    print("🌟 Unity Mathematics Complete System Launcher")
    print("3000 ELO / 300 IQ Metagamer Agent System")
    print("=" * 60)

    launcher = UnityCompleteLauncher()
    success = launcher.run()

    if success:
        print("✅ System completed successfully")
    else:
        print("❌ System encountered errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
