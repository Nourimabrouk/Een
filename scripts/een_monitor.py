#!/usr/bin/env python3
"""
Een Framework Monitor
====================

Background monitoring and management system for the Een framework.
Provides health checks, performance monitoring, and automated maintenance.
"""

import os
import sys
import time
import json
import logging
import threading
import subprocess
import psutil
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import argparse
import signal
import atexit

class EenMonitor:
    def __init__(self, config_path: Optional[str] = None):
        self.project_root = Path(__file__).parent
        self.config = self.load_config(config_path)
        self.running = False
        self.services = {}
        self.metrics = {
            "start_time": datetime.now(),
            "uptime": 0,
            "health_checks": 0,
            "errors": 0,
            "performance": []
        }
        
        # Setup logging
        self.setup_logging()
        
        # Register cleanup
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load monitoring configuration"""
        default_config = {
            "monitoring": {
                "enabled": True,
                "interval": 30,  # seconds
                "health_check_interval": 60,
                "performance_interval": 300,
                "log_level": "INFO"
            },
            "services": {
                "api_server": {
                    "enabled": True,
                    "port": 8000,
                    "health_endpoint": "/health",
                    "auto_restart": True,
                    "max_restarts": 3
                },
                "dashboard": {
                    "enabled": True,
                    "port": 8501,
                    "health_endpoint": "/_stcore/health",
                    "auto_restart": True,
                    "max_restarts": 3
                },
                "mcp_server": {
                    "enabled": True,
                    "port": 3000,
                    "auto_restart": True,
                    "max_restarts": 3
                }
            },
            "alerts": {
                "enabled": True,
                "email": None,
                "webhook": None,
                "discord_webhook": None
            },
            "performance": {
                "enabled": True,
                "cpu_threshold": 80.0,
                "memory_threshold": 80.0,
                "disk_threshold": 90.0
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Merge with defaults
                self.merge_configs(default_config, user_config)
        
        return default_config
    
    def merge_configs(self, default: Dict, user: Dict):
        """Merge user config with defaults"""
        for key, value in user.items():
            if key in default:
                if isinstance(value, dict) and isinstance(default[key], dict):
                    self.merge_configs(default[key], value)
                else:
                    default[key] = value
            else:
                default[key] = value
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"een_monitor_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=getattr(logging, self.config["monitoring"]["log_level"]),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger("EenMonitor")
    
    def start(self):
        """Start the monitoring system"""
        self.logger.info("🚀 Starting Een Framework Monitor")
        self.running = True
        
        # Start monitoring threads
        threads = []
        
        # Health monitoring thread
        health_thread = threading.Thread(target=self.health_monitor_loop, daemon=True)
        health_thread.start()
        threads.append(health_thread)
        
        # Performance monitoring thread
        if self.config["performance"]["enabled"]:
            perf_thread = threading.Thread(target=self.performance_monitor_loop, daemon=True)
            perf_thread.start()
            threads.append(perf_thread)
        
        # Service management thread
        service_thread = threading.Thread(target=self.service_manager_loop, daemon=True)
        service_thread.start()
        threads.append(service_thread)
        
        # Main monitoring loop
        try:
            while self.running:
                self.update_metrics()
                time.sleep(self.config["monitoring"]["interval"])
        except KeyboardInterrupt:
            self.logger.info("Received interrupt signal")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the monitoring system"""
        self.logger.info("🛑 Stopping Een Framework Monitor")
        self.running = False
        self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.logger.info("🧹 Cleaning up resources")
        
        # Stop all managed services
        for service_name, service_info in self.services.items():
            if service_info.get("process"):
                try:
                    service_info["process"].terminate()
                    service_info["process"].wait(timeout=5)
                except:
                    service_info["process"].kill()
        
        # Save final metrics
        self.save_metrics()
    
    def signal_handler(self, signum, frame):
        """Handle system signals"""
        self.logger.info(f"Received signal {signum}")
        self.stop()
    
    def health_monitor_loop(self):
        """Health monitoring loop"""
        while self.running:
            try:
                self.check_all_services_health()
                time.sleep(self.config["monitoring"]["health_check_interval"])
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
    
    def performance_monitor_loop(self):
        """Performance monitoring loop"""
        while self.running:
            try:
                self.collect_performance_metrics()
                time.sleep(self.config["monitoring"]["performance_interval"])
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
    
    def service_manager_loop(self):
        """Service management loop"""
        while self.running:
            try:
                self.manage_services()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                self.logger.error(f"Service management error: {e}")
    
    def check_all_services_health(self):
        """Check health of all services"""
        self.metrics["health_checks"] += 1
        
        for service_name, service_config in self.config["services"].items():
            if not service_config["enabled"]:
                continue
            
            health_status = self.check_service_health(service_name, service_config)
            
            if not health_status["healthy"]:
                self.logger.warning(f"Service {service_name} unhealthy: {health_status['error']}")
                self.metrics["errors"] += 1
                
                # Auto-restart if enabled
                if service_config.get("auto_restart"):
                    self.restart_service(service_name, service_config)
            else:
                self.logger.debug(f"Service {service_name} healthy")
    
    def check_service_health(self, service_name: str, service_config: Dict) -> Dict[str, Any]:
        """Check health of a specific service"""
        try:
            port = service_config["port"]
            health_endpoint = service_config.get("health_endpoint", "/health")
            
            # Check if port is listening
            if not self.is_port_listening(port):
                return {
                    "healthy": False,
                    "error": f"Port {port} not listening",
                    "timestamp": datetime.now()
                }
            
            # Try HTTP health check
            try:
                response = requests.get(f"http://localhost:{port}{health_endpoint}", 
                                     timeout=5)
                if response.status_code == 200:
                    return {
                        "healthy": True,
                        "response_time": response.elapsed.total_seconds(),
                        "timestamp": datetime.now()
                    }
                else:
                    return {
                        "healthy": False,
                        "error": f"HTTP {response.status_code}",
                        "timestamp": datetime.now()
                    }
            except requests.RequestException as e:
                return {
                    "healthy": False,
                    "error": f"HTTP error: {e}",
                    "timestamp": datetime.now()
                }
                
        except Exception as e:
            return {
                "healthy": False,
                "error": f"Check failed: {e}",
                "timestamp": datetime.now()
            }
    
    def is_port_listening(self, port: int) -> bool:
        """Check if a port is listening"""
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('localhost', port))
                return result == 0
        except:
            return False
    
    def restart_service(self, service_name: str, service_config: Dict):
        """Restart a service"""
        service_info = self.services.get(service_name, {})
        restart_count = service_info.get("restart_count", 0)
        max_restarts = service_config.get("max_restarts", 3)
        
        if restart_count >= max_restarts:
            self.logger.error(f"Service {service_name} exceeded max restarts ({max_restarts})")
            self.send_alert(f"Service {service_name} failed to restart after {max_restarts} attempts")
            return
        
        self.logger.info(f"Restarting service {service_name} (attempt {restart_count + 1})")
        
        # Stop existing process
        if service_info.get("process"):
            try:
                service_info["process"].terminate()
                service_info["process"].wait(timeout=5)
            except:
                service_info["process"].kill()
        
        # Start new process
        process = self.start_service(service_name, service_config)
        if process:
            self.services[service_name] = {
                "process": process,
                "restart_count": restart_count + 1,
                "last_restart": datetime.now()
            }
            self.logger.info(f"Service {service_name} restarted successfully")
    
    def start_service(self, service_name: str, service_config: Dict):
        """Start a service"""
        try:
            if service_name == "api_server":
                cmd = [sys.executable, "een_server.py"]
            elif service_name == "dashboard":
                cmd = [sys.executable, "-m", "streamlit", "run", "viz/streamlit_app.py", 
                      "--server.port", str(service_config["port"])]
            elif service_name == "mcp_server":
                cmd = [sys.executable, "config/mcp_consciousness_server.py"]
            else:
                self.logger.error(f"Unknown service: {service_name}")
                return None
            
            process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.logger.info(f"Started service {service_name} (PID: {process.pid})")
            return process
            
        except Exception as e:
            self.logger.error(f"Failed to start service {service_name}: {e}")
            return None
    
    def manage_services(self):
        """Manage all services"""
        for service_name, service_config in self.config["services"].items():
            if not service_config["enabled"]:
                continue
            
            # Check if service is running
            service_info = self.services.get(service_name, {})
            process = service_info.get("process")
            
            if not process or process.poll() is not None:
                # Service not running, start it
                self.logger.info(f"Starting service {service_name}")
                process = self.start_service(service_name, service_config)
                if process:
                    self.services[service_name] = {
                        "process": process,
                        "restart_count": 0,
                        "start_time": datetime.now()
                    }
    
    def collect_performance_metrics(self):
        """Collect system performance metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            
            # Process-specific metrics
            process_metrics = {}
            for service_name, service_info in self.services.items():
                process = service_info.get("process")
                if process and process.poll() is None:
                    try:
                        proc = psutil.Process(process.pid)
                        process_metrics[service_name] = {
                            "cpu_percent": proc.cpu_percent(),
                            "memory_percent": proc.memory_percent(),
                            "memory_rss": proc.memory_info().rss
                        }
                    except psutil.NoSuchProcess:
                        pass
            
            metric = {
                "timestamp": datetime.now(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_percent": disk_percent,
                "network_bytes_sent": network.bytes_sent,
                "network_bytes_recv": network.bytes_recv,
                "processes": process_metrics
            }
            
            self.metrics["performance"].append(metric)
            
            # Check thresholds
            self.check_performance_thresholds(metric)
            
            # Keep only last 1000 metrics
            if len(self.metrics["performance"]) > 1000:
                self.metrics["performance"] = self.metrics["performance"][-1000:]
                
        except Exception as e:
            self.logger.error(f"Performance collection error: {e}")
    
    def check_performance_thresholds(self, metric: Dict):
        """Check performance thresholds and send alerts"""
        thresholds = self.config["performance"]
        
        if metric["cpu_percent"] > thresholds["cpu_threshold"]:
            self.send_alert(f"High CPU usage: {metric['cpu_percent']:.1f}%")
        
        if metric["memory_percent"] > thresholds["memory_threshold"]:
            self.send_alert(f"High memory usage: {metric['memory_percent']:.1f}%")
        
        if metric["disk_percent"] > thresholds["disk_threshold"]:
            self.send_alert(f"High disk usage: {metric['disk_percent']:.1f}%")
    
    def send_alert(self, message: str):
        """Send alert notification"""
        if not self.config["alerts"]["enabled"]:
            return
        
        self.logger.warning(f"ALERT: {message}")
        
        # Email alert
        if self.config["alerts"]["email"]:
            self.send_email_alert(message)
        
        # Webhook alert
        if self.config["alerts"]["webhook"]:
            self.send_webhook_alert(message)
        
        # Discord alert
        if self.config["alerts"]["discord_webhook"]:
            self.send_discord_alert(message)
    
    def send_email_alert(self, message: str):
        """Send email alert"""
        # Implementation would depend on email service
        pass
    
    def send_webhook_alert(self, message: str):
        """Send webhook alert"""
        try:
            payload = {
                "text": f"Een Framework Alert: {message}",
                "timestamp": datetime.now().isoformat()
            }
            requests.post(self.config["alerts"]["webhook"], 
                         json=payload, timeout=5)
        except Exception as e:
            self.logger.error(f"Webhook alert failed: {e}")
    
    def send_discord_alert(self, message: str):
        """Send Discord alert"""
        try:
            payload = {
                "content": f"🚨 Een Framework Alert: {message}",
                "embeds": [{
                    "title": "Een Framework Monitor",
                    "description": message,
                    "color": 0xFF0000,
                    "timestamp": datetime.now().isoformat()
                }]
            }
            requests.post(self.config["alerts"]["discord_webhook"], 
                         json=payload, timeout=5)
        except Exception as e:
            self.logger.error(f"Discord alert failed: {e}")
    
    def update_metrics(self):
        """Update monitoring metrics"""
        self.metrics["uptime"] = (datetime.now() - self.metrics["start_time"]).total_seconds()
    
    def save_metrics(self):
        """Save metrics to file"""
        try:
            metrics_file = self.project_root / "logs" / "metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        status = {
            "running": self.running,
            "uptime": self.metrics["uptime"],
            "health_checks": self.metrics["health_checks"],
            "errors": self.metrics["errors"],
            "services": {}
        }
        
        for service_name, service_info in self.services.items():
            process = service_info.get("process")
            status["services"][service_name] = {
                "running": process and process.poll() is None,
                "pid": process.pid if process else None,
                "restart_count": service_info.get("restart_count", 0)
            }
        
        return status
    
    def print_status(self):
        """Print current status"""
        status = self.get_status()
        
        print("🎯 Een Framework Monitor Status")
        print("=" * 40)
        print(f"Running: {status['running']}")
        print(f"Uptime: {status['uptime']:.0f} seconds")
        print(f"Health Checks: {status['health_checks']}")
        print(f"Errors: {status['errors']}")
        print()
        print("Services:")
        for service_name, service_status in status["services"].items():
            status_icon = "🟢" if service_status["running"] else "🔴"
            print(f"  {status_icon} {service_name}: {'Running' if service_status['running'] else 'Stopped'}")
            if service_status["pid"]:
                print(f"    PID: {service_status['pid']}")
            if service_status["restart_count"] > 0:
                print(f"    Restarts: {service_status['restart_count']}")

def main():
    parser = argparse.ArgumentParser(description="Een Framework Monitor")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    parser.add_argument("--start-services", action="store_true", help="Start all services")
    
    args = parser.parse_args()
    
    monitor = EenMonitor(args.config)
    
    if args.status:
        monitor.print_status()
        return
    
    if args.start_services:
        monitor.manage_services()
        return
    
    if args.daemon:
        # Run as daemon
        import daemon
        with daemon.DaemonContext():
            monitor.start()
    else:
        monitor.start()

if __name__ == "__main__":
    main() 