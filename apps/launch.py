#!/usr/bin/env python3
"""
Een Unity Mathematics - Unified Launch System
Production-ready deployment orchestrator with multi-service support
"""

import asyncio
import logging
import os
import sys
import subprocess
import signal
import time
from pathlib import Path
from typing import Dict, List, Optional
import psutil
import webbrowser
from concurrent.futures import ThreadPoolExecutor

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

class UnityLaunchOrchestrator:
    """
    Master orchestrator for the Een Unity Mathematics platform.
    Manages all services, monitoring, and deployment tasks.
    """
    
    def __init__(self):
        self.services: Dict[str, subprocess.Popen] = {}
        self.config = self._load_config()
        self.logger = self._setup_logging()
        self.running = False
        
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging system."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/unity_launch.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        return logging.getLogger(__name__)
    
    def _load_config(self) -> Dict:
        """Load configuration from environment and defaults."""
        return {
            'api_port': int(os.getenv('API_PORT', 8000)),
            'streamlit_port': int(os.getenv('STREAMLIT_PORT', 8501)),
            'web_port': int(os.getenv('WEB_PORT', 8080)),
            'auto_open_browser': os.getenv('AUTO_OPEN_BROWSER', 'true').lower() == 'true',
            'enable_gpu': os.getenv('ENABLE_GPU', 'true').lower() == 'true',
            'debug': os.getenv('DEBUG', 'false').lower() == 'true',
            'environment': os.getenv('ENV', 'development'),
        }
    
    def _check_port_available(self, port: int) -> bool:
        """Check if a port is available."""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return True
            except OSError:
                return False
    
    def _find_available_port(self, start_port: int, max_attempts: int = 10) -> int:
        """Find an available port starting from start_port."""
        for port in range(start_port, start_port + max_attempts):
            if self._check_port_available(port):
                return port
        raise RuntimeError(f"No available ports found starting from {start_port}")
    
    async def start_fastapi_server(self) -> None:
        """Start the FastAPI backend server."""
        try:
            # Check if port is available
            port = self.config['api_port']
            if not self._check_port_available(port):
                port = self._find_available_port(port)
                self.config['api_port'] = port
                self.logger.warning(f"API port changed to {port}")
            
            # Start FastAPI server
            cmd = [
                sys.executable, '-m', 'uvicorn', 'main:app',
                '--host', '0.0.0.0',
                '--port', str(port),
                '--reload' if self.config['debug'] else '--no-reload',
                '--workers', '1' if self.config['debug'] else '4'
            ]
            
            self.logger.info(f"🚀 Starting FastAPI server on port {port}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=PROJECT_ROOT
            )
            
            self.services['fastapi'] = process
            
            # Wait for server to be ready
            await self._wait_for_service(f"http://localhost:{port}/health", "FastAPI")
            
        except Exception as e:
            self.logger.error(f"Failed to start FastAPI server: {e}")
            raise
    
    async def start_streamlit_app(self) -> None:
        """Start the Streamlit dashboard application."""
        try:
            # Check if port is available
            port = self.config['streamlit_port']
            if not self._check_port_available(port):
                port = self._find_available_port(port)
                self.config['streamlit_port'] = port
                self.logger.warning(f"Streamlit port changed to {port}")
            
            # Find the main Streamlit app
            streamlit_app = PROJECT_ROOT / 'src' / 'unity_mathematics_streamlit.py'
            if not streamlit_app.exists():
                streamlit_app = PROJECT_ROOT / 'viz' / 'streamlit_app.py'
            
            if not streamlit_app.exists():
                self.logger.warning("No Streamlit app found, skipping...")
                return
            
            cmd = [
                sys.executable, '-m', 'streamlit', 'run', str(streamlit_app),
                '--server.port', str(port),
                '--server.address', 'localhost',
                '--server.headless', 'true',
                '--browser.gatherUsageStats', 'false'
            ]
            
            self.logger.info(f"📊 Starting Streamlit app on port {port}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=PROJECT_ROOT
            )
            
            self.services['streamlit'] = process
            
            # Wait for service to be ready
            await self._wait_for_service(f"http://localhost:{port}", "Streamlit", max_wait=30)
            
        except Exception as e:
            self.logger.error(f"Failed to start Streamlit app: {e}")
            # Non-critical, continue without Streamlit
    
    async def start_web_server(self) -> None:
        """Start a simple HTTP server for the website."""
        try:
            port = self.config['web_port']
            if not self._check_port_available(port):
                port = self._find_available_port(port)
                self.config['web_port'] = port
                self.logger.warning(f"Web server port changed to {port}")
            
            # Determine website directory
            website_dir = PROJECT_ROOT / 'website'
            if not website_dir.exists():
                website_dir = PROJECT_ROOT
            
            cmd = [
                sys.executable, '-m', 'http.server', str(port),
                '--directory', str(website_dir)
            ]
            
            self.logger.info(f"🌐 Starting web server on port {port}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=website_dir
            )
            
            self.services['webserver'] = process
            
            # Wait for service to be ready
            await self._wait_for_service(f"http://localhost:{port}", "Web Server")
            
        except Exception as e:
            self.logger.error(f"Failed to start web server: {e}")
            raise
    
    async def _wait_for_service(
        self, 
        url: str, 
        service_name: str, 
        max_wait: int = 30
    ) -> None:
        """Wait for a service to become available."""
        import aiohttp
        
        self.logger.info(f"⏳ Waiting for {service_name} to be ready...")
        
        async with aiohttp.ClientSession() as session:
            for i in range(max_wait):
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as response:
                        if response.status == 200:
                            self.logger.info(f"✅ {service_name} is ready!")
                            return
                except:
                    pass
                
                await asyncio.sleep(1)
        
        self.logger.warning(f"⚠️ {service_name} may not be fully ready after {max_wait}s")
    
    async def initialize_system(self) -> None:
        """Initialize the Unity Mathematics system."""
        self.logger.info("🔧 Initializing Unity Mathematics system...")
        
        # Check Python environment
        python_version = sys.version_info
        if python_version < (3, 8):
            raise RuntimeError(f"Python 3.8+ required, got {python_version}")
        
        # Check critical dependencies
        required_packages = ['fastapi', 'uvicorn', 'streamlit', 'numpy']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.logger.error(f"Missing packages: {missing_packages}")
            self.logger.info("Install with: pip install " + " ".join(missing_packages))
            raise RuntimeError("Missing required dependencies")
        
        # Initialize GPU support if available
        if self.config['enable_gpu']:
            try:
                import torch
                if torch.cuda.is_available():
                    self.logger.info(f"🚀 GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
                else:
                    self.logger.info("💻 GPU not available, using CPU")
            except ImportError:
                self.logger.info("PyTorch not installed, GPU acceleration disabled")
        
        self.logger.info("✅ System initialization complete")
    
    async def start_all_services(self) -> None:
        """Start all Unity Mathematics services."""
        self.logger.info("🚀 Starting Een Unity Mathematics Platform...")
        
        # Initialize system
        await self.initialize_system()
        
        # Start services concurrently
        tasks = [
            self.start_fastapi_server(),
            self.start_streamlit_app(),
            self.start_web_server(),
        ]
        
        # Start services with error handling
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check for critical failures
        critical_services = ['fastapi', 'webserver']
        for service in critical_services:
            if service not in self.services:
                raise RuntimeError(f"Critical service {service} failed to start")
        
        self.running = True
        self._log_service_status()
    
    def _log_service_status(self) -> None:
        """Log the status of all services."""
        self.logger.info("\n" + "="*60)
        self.logger.info("🌟 EEN UNITY MATHEMATICS PLATFORM - SERVICE STATUS")
        self.logger.info("="*60)
        
        if 'fastapi' in self.services:
            self.logger.info(f"🔧 API Server:      http://localhost:{self.config['api_port']}")
            self.logger.info(f"📚 API Docs:       http://localhost:{self.config['api_port']}/docs")
        
        if 'streamlit' in self.services:
            self.logger.info(f"📊 Dashboard:      http://localhost:{self.config['streamlit_port']}")
        
        if 'webserver' in self.services:
            self.logger.info(f"🌐 Website:        http://localhost:{self.config['web_port']}")
        
        self.logger.info("="*60)
        self.logger.info("✨ All services are running! Ready for transcendental mathematics!")
        self.logger.info("="*60)
    
    def open_browser(self) -> None:
        """Open the website in the default browser."""
        if self.config['auto_open_browser'] and 'webserver' in self.services:
            try:
                url = f"http://localhost:{self.config['web_port']}"
                webbrowser.open(url)
                self.logger.info(f"🌐 Opened browser to {url}")
            except Exception as e:
                self.logger.warning(f"Could not open browser: {e}")
    
    async def monitor_services(self) -> None:
        """Monitor running services and restart if necessary."""
        while self.running:
            for service_name, process in list(self.services.items()):
                if process.poll() is not None:
                    self.logger.error(f"❌ Service {service_name} has stopped!")
                    
                    # Attempt restart for critical services
                    if service_name in ['fastapi', 'webserver']:
                        self.logger.info(f"🔄 Restarting {service_name}...")
                        try:
                            if service_name == 'fastapi':
                                await self.start_fastapi_server()
                            elif service_name == 'webserver':
                                await self.start_web_server()
                        except Exception as e:
                            self.logger.error(f"Failed to restart {service_name}: {e}")
            
            await asyncio.sleep(5)
    
    def shutdown(self) -> None:
        """Gracefully shutdown all services."""
        self.logger.info("🛑 Shutting down Unity Mathematics platform...")
        self.running = False
        
        for service_name, process in self.services.items():
            try:
                self.logger.info(f"Stopping {service_name}...")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"Force killing {service_name}...")
                    process.kill()
                
                self.logger.info(f"✅ {service_name} stopped")
            except Exception as e:
                self.logger.error(f"Error stopping {service_name}: {e}")
        
        self.logger.info("✅ Shutdown complete")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}")
        self.shutdown()
        sys.exit(0)

async def main():
    """Main entry point for the Unity Mathematics platform."""
    orchestrator = UnityLaunchOrchestrator()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, orchestrator.signal_handler)
    signal.signal(signal.SIGTERM, orchestrator.signal_handler)
    
    try:
        # Start all services
        await orchestrator.start_all_services()
        
        # Open browser if configured
        orchestrator.open_browser()
        
        # Monitor services
        await orchestrator.monitor_services()
        
    except KeyboardInterrupt:
        orchestrator.logger.info("Received keyboard interrupt")
    except Exception as e:
        orchestrator.logger.error(f"Fatal error: {e}")
        raise
    finally:
        orchestrator.shutdown()

def cli():
    """Command-line interface for the launcher."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Een Unity Mathematics Platform Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch.py                    # Start with default settings
  python launch.py --port 8080       # Start API on port 8080
  python launch.py --no-browser      # Don't auto-open browser
  python launch.py --debug           # Enable debug mode
        """
    )
    
    parser.add_argument('--port', type=int, help='API server port (default: 8000)')
    parser.add_argument('--streamlit-port', type=int, help='Streamlit port (default: 8501)')
    parser.add_argument('--web-port', type=int, help='Web server port (default: 8080)')
    parser.add_argument('--no-browser', action='store_true', help='Don\'t auto-open browser')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    
    args = parser.parse_args()
    
    # Set environment variables from CLI args
    if args.port:
        os.environ['API_PORT'] = str(args.port)
    if args.streamlit_port:
        os.environ['STREAMLIT_PORT'] = str(args.streamlit_port)
    if args.web_port:
        os.environ['WEB_PORT'] = str(args.web_port)
    if args.no_browser:
        os.environ['AUTO_OPEN_BROWSER'] = 'false'
    if args.debug:
        os.environ['DEBUG'] = 'true'
    if args.no_gpu:
        os.environ['ENABLE_GPU'] = 'false'
    
    # Run the main function
    asyncio.run(main())

if __name__ == "__main__":
    cli()