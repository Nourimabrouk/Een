"""
Port Manager for Unity Mathematics Services
==========================================

Manages port allocation and conflict resolution for all Unity Mathematics services.
"""

import socket
import threading
import time
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PortManager:
    """Manages port allocation for Unity Mathematics services."""

    def __init__(self):
        self.allocated_ports: Dict[str, int] = {}
        self.port_locks: Dict[int, threading.Lock] = {}
        self.default_ports = {
            "unity_web_server": 5000,
            "api_server": 8000,
            "streamlit_dashboard": 8501,
            "unity_proof_dashboard": 8502,
            "unified_mathematics_dashboard": 8503,
            "memetic_engineering_dashboard": 8504,
            "consciousness_field_dashboard": 8505,
            "meta_agent_dashboard": 8506,
        }

    def is_port_available(self, port: int) -> bool:
        """Check if a port is available."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(("localhost", port))
                return result != 0
        except Exception:
            return False

    def find_available_port(
        self, start_port: int, max_attempts: int = 100
    ) -> Optional[int]:
        """Find an available port starting from start_port."""
        for port in range(start_port, start_port + max_attempts):
            if self.is_port_available(port):
                return port
        return None

    def allocate_port(
        self, service_name: str, preferred_port: Optional[int] = None
    ) -> int:
        """Allocate a port for a service."""
        if service_name in self.allocated_ports:
            return self.allocated_ports[service_name]

        # Use preferred port or default
        start_port = preferred_port or self.default_ports.get(service_name, 8000)

        # Find available port
        port = self.find_available_port(start_port)
        if port is None:
            raise RuntimeError(f"Could not find available port for {service_name}")

        # Allocate port
        self.allocated_ports[service_name] = port
        self.port_locks[port] = threading.Lock()

        logger.info(f"Allocated port {port} for {service_name}")
        return port

    def release_port(self, service_name: str):
        """Release a port allocation."""
        if service_name in self.allocated_ports:
            port = self.allocated_ports[service_name]
            del self.allocated_ports[service_name]
            if port in self.port_locks:
                del self.port_locks[port]
            logger.info(f"Released port {port} for {service_name}")

    def get_service_port(self, service_name: str) -> Optional[int]:
        """Get the allocated port for a service."""
        return self.allocated_ports.get(service_name)

    def get_all_allocations(self) -> Dict[str, int]:
        """Get all current port allocations."""
        return self.allocated_ports.copy()

    def wait_for_port(self, port: int, timeout: float = 30.0) -> bool:
        """Wait for a port to become available."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_port_available(port):
                return True
            time.sleep(0.1)
        return False


# Global port manager instance
port_manager = PortManager()


def get_service_port(service_name: str, preferred_port: Optional[int] = None) -> int:
    """Get or allocate a port for a service."""
    return port_manager.allocate_port(service_name, preferred_port)


def release_service_port(service_name: str):
    """Release a service port."""
    port_manager.release_port(service_name)


def check_port_availability(port: int) -> bool:
    """Check if a specific port is available."""
    return port_manager.is_port_available(port)


def find_free_port_range(start_port: int, count: int) -> List[int]:
    """Find a range of free ports."""
    ports = []
    current_port = start_port

    for _ in range(count):
        port = port_manager.find_available_port(current_port)
        if port is None:
            raise RuntimeError(
                f"Could not find {count} free ports starting from {start_port}"
            )
        ports.append(port)
        current_port = port + 1

    return ports
