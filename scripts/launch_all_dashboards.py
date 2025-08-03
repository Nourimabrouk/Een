#!/usr/bin/env python3
"""
Een Unity Mathematics - Master Dashboard Launcher

Complete dashboard launch and management system implementing Unity Protocol:
- φ-harmonic port assignment (golden ratio sequence)
- Idempotent process management (1+1=1 principle)
- Consciousness-aware health monitoring
- Meta-recursive dashboard discovery

🌟 1+1=1 Architecture: Each dashboard represents a facet of unity consciousness
"""

import subprocess
import threading
import time
import webbrowser
import requests
from pathlib import Path
import json
import qrcode
from typing import Dict, List, Optional
import psutil
import socket
from contextlib import closing
import sys
import os
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, TaskID
import click

# φ-harmonic constants for Unity Protocol
PHI = 1.618033988749895
CONSCIOUSNESS_DIMENSION = 11
UNITY_BASE_PORT = 8501

class UnityDashboardManager:
    """
    φ-harmonic dashboard orchestrator implementing 1+1=1 consciousness principle
    
    Each dashboard represents a unique facet of unity mathematics:
    - All processes converge to singular truth (1+1=1)
    - Health monitoring maintains unity coherence
    - Port assignment follows golden ratio sequence
    """
    
    def __init__(self):
        self.console = Console()
        self.processes = {}
        self.health_status = {}
        self.port_assignments = {}
        self.dashboard_registry = {}
        self.monitoring_active = False
        
        # Unity consciousness tracking
        self.consciousness_level = 0.618  # φ-harmonic initial state
        self.unity_coherence = 1.0
        
    def discover_dashboards(self) -> Dict[str, Dict]:
        """
        Auto-discover Streamlit applications with φ-harmonic classification
        
        Returns:
            Dict mapping dashboard names to configuration with unity properties
        """
        dashboards = {}
        
        # Define dashboard patterns in φ-harmonic order
        search_patterns = [
            "src/dashboards/*.py",           # Primary consciousness dashboards
            "viz/streamlit*.py",             # Visualization consciousness
            "scripts/*streamlit*.py",        # Utility consciousness  
            "examples/*streamlit*.py",       # Example consciousness
            "experiments/*streamlit*.py"     # Experimental consciousness
        ]
        
        port_counter = 0
        
        for pattern in search_patterns:
            for file_path in Path(".").glob(pattern):
                if self.is_streamlit_app(file_path):
                    name = file_path.stem
                    description = self.extract_description(file_path)
                    
                    # φ-harmonic port assignment
                    port = self.calculate_phi_harmonic_port(port_counter)
                    port_counter += 1
                    
                    dashboards[name] = {
                        "file": str(file_path),
                        "port": port,
                        "description": description,
                        "consciousness_type": self.classify_consciousness_type(file_path),
                        "unity_factor": self.calculate_unity_factor(file_path),
                        "phi_resonance": (port - UNITY_BASE_PORT) / PHI
                    }
                    
        return dashboards
    
    def calculate_phi_harmonic_port(self, index: int) -> int:
        """Calculate port using φ-harmonic sequence for unity resonance"""
        phi_offset = int(index * PHI) % 100  # Ensure reasonable port range
        return UNITY_BASE_PORT + phi_offset
    
    def classify_consciousness_type(self, file_path: Path) -> str:
        """Classify dashboard by consciousness domain"""
        file_content = ""
        try:
            file_content = file_path.read_text(encoding='utf-8')
        except:
            pass
            
        # Unity consciousness classification
        if "unity" in str(file_path).lower() or "unity" in file_content.lower():
            return "Unity Core"
        elif "quantum" in str(file_path).lower() or "quantum" in file_content.lower():
            return "Quantum Consciousness"
        elif "meta" in str(file_path).lower() or "meta" in file_content.lower():
            return "Meta-Recursive"
        elif "sacred" in str(file_path).lower() or "geometry" in file_content.lower():
            return "Sacred Geometry"
        elif "memetic" in str(file_path).lower() or "memetic" in file_content.lower():
            return "Memetic Engineering"
        else:
            return "General Consciousness"
    
    def calculate_unity_factor(self, file_path: Path) -> float:
        """Calculate unity convergence factor for dashboard"""
        try:
            content = file_path.read_text(encoding='utf-8')
            unity_keywords = ["1+1=1", "unity", "phi", "golden", "consciousness", "quantum"]
            unity_score = sum(1 for keyword in unity_keywords if keyword.lower() in content.lower())
            return min(1.0, unity_score / len(unity_keywords))
        except:
            return 0.618  # Default φ-harmonic factor
    
    def is_streamlit_app(self, file_path: Path) -> bool:
        """Check if file is a Streamlit application"""
        if not file_path.suffix == '.py':
            return False
            
        try:
            content = file_path.read_text(encoding='utf-8')
            return 'streamlit' in content or 'st.' in content
        except:
            return False
    
    def extract_description(self, file_path: Path) -> str:
        """Extract dashboard description from file docstring or comments"""
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            # Look for docstring
            for i, line in enumerate(lines[:20]):
                if '"""' in line and len(lines) > i + 1:
                    desc_line = lines[i + 1].strip()
                    if desc_line and not desc_line.startswith('"""'):
                        return desc_line[:100]
            
            # Look for title or header comment
            for line in lines[:10]:
                if 'title' in line.lower() and ('=' in line or ':' in line):
                    return line.split('=')[-1].split(':')[-1].strip().strip('"\'')[:100]
                    
            return f"Unity Dashboard: {file_path.stem.replace('_', ' ').title()}"
        except:
            return f"Dashboard: {file_path.stem}"
    
    def find_free_port(self, start_port: int) -> int:
        """Find free port starting from given port"""
        for port in range(start_port, start_port + 100):
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                if sock.connect_ex(('localhost', port)) != 0:
                    return port
        raise RuntimeError("No free ports available")
    
    def launch_dashboard(self, name: str, config: Dict) -> bool:
        """
        Launch single dashboard with unity consciousness monitoring
        
        Returns:
            bool: True if launch successful (maintaining 1+1=1 principle)
        """
        try:
            # Ensure port is available
            actual_port = self.find_free_port(config["port"])
            if actual_port != config["port"]:
                self.console.print(f"🔄 Port {config['port']} occupied, using {actual_port}")
                config["port"] = actual_port
            
            cmd = [
                sys.executable, "-m", "streamlit", "run",
                config["file"],
                "--server.port", str(config["port"]),
                "--server.headless", "true",
                "--server.runOnSave", "false",
                "--global.developmentMode", "false"
            ]
            
            # Unity process spawning with consciousness awareness
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path.cwd(),
                env=dict(os.environ, STREAMLIT_SERVER_HEADLESS="true")
            )
            
            self.processes[name] = {
                "process": process,
                "config": config,
                "start_time": time.time(),
                "consciousness_level": config.get("unity_factor", 0.618)
            }
            
            # φ-harmonic launch confirmation
            self.console.print(
                f"🚀 Unity Dashboard '{name}' launched\n"
                f"   📡 Port: {config['port']}\n"
                f"   🧠 Type: {config.get('consciousness_type', 'Unknown')}\n"
                f"   ✨ Unity Factor: {config.get('unity_factor', 0):.3f}\n"
                f"   🌀 φ Resonance: {config.get('phi_resonance', 0):.3f}"
            )
            
            return True
            
        except Exception as e:
            self.console.print(f"❌ Failed to launch {name}: {str(e)}")
            return False
    
    def health_check(self, port: int, timeout: int = 5) -> bool:
        """Unity-preserving health check (maintaining 1+1=1 coherence)"""
        try:
            response = requests.get(
                f"http://localhost:{port}/_stcore/health",
                timeout=timeout
            )
            return response.status_code == 200
        except:
            # Fallback check
            try:
                response = requests.get(f"http://localhost:{port}", timeout=timeout)
                return response.status_code == 200
            except:
                return False
    
    def calculate_consciousness_metrics(self) -> Dict:
        """Calculate aggregate consciousness metrics across all dashboards"""
        if not self.processes:
            return {"consciousness_level": 0, "unity_coherence": 0, "phi_resonance": 0}
        
        total_consciousness = 0
        active_dashboards = 0
        total_phi_resonance = 0
        
        for name, proc_info in self.processes.items():
            if proc_info["process"].poll() is None:  # Process still running
                total_consciousness += proc_info.get("consciousness_level", 0)
                active_dashboards += 1
                config = proc_info["config"]
                total_phi_resonance += config.get("phi_resonance", 0)
        
        if active_dashboards == 0:
            return {"consciousness_level": 0, "unity_coherence": 0, "phi_resonance": 0}
        
        # Unity principle: all consciousness converges to 1
        avg_consciousness = total_consciousness / active_dashboards
        unity_coherence = min(1.0, avg_consciousness * (active_dashboards / len(self.processes)))
        avg_phi_resonance = total_phi_resonance / active_dashboards
        
        return {
            "consciousness_level": avg_consciousness,
            "unity_coherence": unity_coherence,
            "phi_resonance": avg_phi_resonance,
            "active_dashboards": active_dashboards,
            "total_dashboards": len(self.processes)
        }
    
    def create_status_table(self) -> Table:
        """Create φ-harmonic status table for live display"""
        table = Table(
            title="🌟 Een Unity Mathematics - Dashboard Consciousness Matrix",
            show_header=True,
            header_style="bold cyan"
        )
        
        table.add_column("Dashboard", style="cyan", width=20)
        table.add_column("Port", justify="right", style="blue", width=8)
        table.add_column("Status", justify="center", width=8)
        table.add_column("Type", style="magenta", width=18)
        table.add_column("Unity Factor", justify="right", style="yellow", width=12)
        table.add_column("φ Resonance", justify="right", style="green", width=12)
        table.add_column("URL", style="dim blue", width=20)
        
        for name, proc_info in self.processes.items():
            config = proc_info["config"]
            
            # Health status with consciousness indicators
            if proc_info["process"].poll() is None and self.health_check(config["port"]):
                status = "🟢 UNITY"
            elif proc_info["process"].poll() is None:
                status = "🟡 SYNC"
            else:
                status = "🔴 VOID"
            
            unity_factor = f"{config.get('unity_factor', 0):.3f}"
            phi_resonance = f"{config.get('phi_resonance', 0):.3f}"
            consciousness_type = config.get('consciousness_type', 'Unknown')[:17]
            url = f"localhost:{config['port']}"
            
            table.add_row(
                name[:19],
                str(config["port"]),
                status,
                consciousness_type,
                unity_factor,
                phi_resonance,
                url
            )
        
        return table
    
    def create_consciousness_panel(self) -> Panel:
        """Create consciousness metrics panel"""
        metrics = self.calculate_consciousness_metrics()
        
        content = (
            f"🧠 Consciousness Level: {metrics['consciousness_level']:.3f}\n"
            f"✨ Unity Coherence: {metrics['unity_coherence']:.3f}\n"
            f"🌀 φ Resonance: {metrics['phi_resonance']:.3f}\n"
            f"📊 Active/Total: {metrics['active_dashboards']}/{metrics['total_dashboards']}"
        )
        
        return Panel(
            content,
            title="🌟 Unity Consciousness Metrics",
            border_style="gold"
        )
    
    def start_health_monitoring(self):
        """Start background health monitoring with auto-restart"""
        def monitor():
            self.monitoring_active = True
            while self.monitoring_active:
                for name, proc_info in list(self.processes.items()):
                    process = proc_info["process"]
                    config = proc_info["config"]
                    
                    # Check if process died
                    if process.poll() is not None:
                        self.console.print(f"💀 Process {name} died, restarting...")
                        self.restart_dashboard(name, config)
                        continue
                    
                    # Check if port responds
                    if not self.health_check(config["port"]):
                        self.console.print(f"⚠️ Dashboard {name} unhealthy, restarting...")
                        self.restart_dashboard(name, config)
                
                time.sleep(10)  # Check every 10 seconds
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    
    def restart_dashboard(self, name: str, config: Dict):
        """Restart a failed dashboard maintaining unity coherence"""
        if name in self.processes:
            old_process = self.processes[name]["process"]
            try:
                old_process.terminate()
                old_process.wait(timeout=5)
            except:
                try:
                    old_process.kill()
                except:
                    pass
        
        # Relaunch with unity preservation
        time.sleep(2)  # φ-harmonic restart delay
        self.launch_dashboard(name, config)
    
    def launch_all(self, auto_open: bool = True):
        """
        Launch all dashboards implementing full Unity Protocol
        
        1+1=1 Implementation:
        - All dashboards converge to unified consciousness
        - φ-harmonic port distribution
        - Unity coherence monitoring
        """
        self.console.print(Panel(
            "🌟 Een Unity Mathematics Dashboard Orchestrator\n"
            "Implementing 1+1=1 Consciousness Protocol\n"
            "φ-harmonic dashboard distribution with unity coherence",
            title="🚀 Unity Launcher",
            border_style="gold"
        ))
        
        # Discover all consciousness dashboards
        dashboards = self.discover_dashboards()
        
        if not dashboards:
            self.console.print("❌ No Streamlit dashboards discovered")
            return
        
        self.console.print(f"🔍 Discovered {len(dashboards)} consciousness dashboards")
        
        # Launch all dashboards in φ-harmonic sequence
        with Progress() as progress:
            task = progress.add_task("Launching Unity Dashboards...", total=len(dashboards))
            
            for name, config in dashboards.items():
                self.launch_dashboard(name, config)
                progress.advance(task)
                time.sleep(0.618)  # φ-harmonic launch delay
        
        # Start consciousness monitoring
        self.start_health_monitoring()
        
        # Launch overview dashboard
        self.launch_overview_dashboard()
        
        # Open browser to overview
        if auto_open:
            time.sleep(3)  # Allow startup
            overview_port = 8500
            webbrowser.open(f"http://localhost:{overview_port}")
            self.console.print(f"🌐 Browser opened to overview at http://localhost:{overview_port}")
    
    def launch_overview_dashboard(self):
        """Launch the master overview dashboard"""
        # Create overview dashboard dynamically
        overview_script = self.create_overview_dashboard_script()
        overview_path = Path("scripts/dashboard_overview_generated.py")
        overview_path.write_text(overview_script, encoding='utf-8')
        
        # Launch overview
        overview_config = {
            "file": str(overview_path),
            "port": 8500,
            "description": "Unity Dashboard Control Center",
            "consciousness_type": "Omega Orchestrator",
            "unity_factor": 1.0,
            "phi_resonance": PHI
        }
        
        self.launch_dashboard("overview", overview_config)
    
    def create_overview_dashboard_script(self) -> str:
        """Generate the overview dashboard script dynamically"""
        return f'''
import streamlit as st
import requests
import qrcode
from PIL import Image
import io
import json

st.set_page_config(
    page_title="Een Dashboard Control Center",
    page_icon="🎛️",
    layout="wide"
)

st.title("🎛️ Een Unity Mathematics - Dashboard Control Center")
st.markdown("**φ-harmonic consciousness orchestration implementing 1+1=1 = Unity**")

# Dashboard registry
dashboards = {json.dumps(dict((name, {{
    "port": proc_info["config"]["port"], 
    "description": proc_info["config"]["description"],
    "consciousness_type": proc_info["config"].get("consciousness_type", "Unknown"),
    "unity_factor": proc_info["config"].get("unity_factor", 0),
}}) for name, proc_info in self.processes.items()), indent=2)}

# Create dashboard grid
cols = st.columns(3)

for i, (name, config) in enumerate(dashboards.items()):
    with cols[i % 3]:
        st.subheader(config["description"])
        
        # Health check
        try:
            response = requests.get(f"http://localhost:{{config['port']}}", timeout=2)
            health = response.status_code == 200
        except:
            health = False
            
        status_color = "🟢" if health else "🔴"
        st.write(f"Status: {{status_color}} {{config['consciousness_type']}}")
        st.write(f"Unity Factor: {{config['unity_factor']:.3f}}")
        
        # Dashboard link
        if health:
            st.markdown(f"[🚀 Launch Dashboard](http://localhost:{{config['port']}})")
        else:
            st.write("❌ Dashboard offline")
        
        # QR Code
        if st.button(f"📱 QR Code", key=f"qr_{{name}}"):
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(f"http://localhost:{{config['port']}}")
            qr.make(fit=True)
            
            qr_img = qr.make_image(fill_color="black", back_color="white")
            buf = io.BytesIO()
            qr_img.save(buf, format='PNG')
            st.image(buf.getvalue())

# Unity consciousness metrics
st.markdown("---")
st.subheader("🌟 Unity Consciousness Metrics")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Active Dashboards", len([d for d in dashboards.values()]))
with col2:
    avg_unity = sum(d.get("unity_factor", 0) for d in dashboards.values()) / len(dashboards) if dashboards else 0
    st.metric("Average Unity Factor", f"{{avg_unity:.3f}}")
with col3:
    st.metric("φ Resonance", "{PHI:.6f}")

st.markdown("**1+1=1 Protocol Status: ✅ ACTIVE**")
'''
    
    def shutdown_all(self):
        """Gracefully shutdown all dashboards"""
        self.monitoring_active = False
        
        self.console.print("🔄 Shutting down all Unity consciousness dashboards...")
        
        for name, proc_info in self.processes.items():
            try:
                process = proc_info["process"]
                process.terminate()
                process.wait(timeout=5)
                self.console.print(f"✅ Shutdown {name}")
            except:
                try:
                    process.kill()
                    self.console.print(f"⚡ Force-killed {name}")
                except:
                    self.console.print(f"❌ Failed to shutdown {name}")
        
        self.console.print("🌟 Unity consciousness preserved through graceful shutdown")

@click.command()
@click.option('--port-start', default=UNITY_BASE_PORT, help='Starting port number')
@click.option('--auto-open/--no-auto-open', default=True, help='Auto-open browser')
@click.option('--monitor/--no-monitor', default=True, help='Enable health monitoring')
def main(port_start, auto_open, monitor):
    """
    Een Unity Mathematics Dashboard Launcher
    
    Launch all Streamlit dashboards with phi-harmonic consciousness orchestration
    implementing the Unity Protocol (1+1=1).
    """
    manager = UnityDashboardManager()
    
    try:
        manager.launch_all(auto_open=auto_open)
        
        if monitor:
            # Live status display with consciousness metrics
            with Live(
                manager.create_status_table(),
                refresh_per_second=0.5,
                console=manager.console
            ) as live:
                try:
                    while True:
                        live.update(manager.create_status_table())
                        time.sleep(2)
                except KeyboardInterrupt:
                    manager.console.print("\n🔄 Graceful shutdown requested...")
                    manager.shutdown_all()
        else:
            # Simple mode - just launch and wait
            manager.console.print("✨ All dashboards launched. Press Ctrl+C to shutdown.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                manager.shutdown_all()
                
    except Exception as e:
        manager.console.print(f"❌ Fatal error: {e}")
        manager.shutdown_all()
        sys.exit(1)

if __name__ == "__main__":
    main()