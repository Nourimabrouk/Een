# 🎛️ Streamlit Dashboard Setup TODO
## Launch & Review All Streamlit Dashboards

### 🎯 Objective
Create a master script to launch all Streamlit dashboards, fix any issues, and provide an easy way to review and test all dashboard functionality.

---

## 🚀 Priority 1: Dashboard Discovery & Launch Script

### 1.1 Create Master Dashboard Launcher
**File**: `scripts/launch_all_dashboards.py`
```python
"""
Master dashboard launcher that:
1. Discovers all Streamlit apps in the codebase
2. Launches them on different ports
3. Provides a dashboard overview page
4. Monitors health and restarts if needed
5. Generates QR codes for mobile testing
"""

Tasks:
- [ ] Auto-discover all streamlit apps
- [ ] Multi-port launch system
- [ ] Health monitoring
- [ ] Auto-restart on crashes
- [ ] Browser auto-open
- [ ] Mobile QR code generation
- [ ] Performance monitoring
- [ ] Resource usage tracking
```

### 1.2 Dashboard Inventory
```python
# Current Streamlit dashboards to launch:
streamlit_apps = {
    # Core Dashboards
    "unity_dashboard": {
        "file": "src/dashboards/unity_proof_dashboard.py",
        "port": 8501,
        "description": "Main Unity Mathematics Dashboard"
    },
    "memetic_engineering": {
        "file": "src/dashboards/memetic_engineering_streamlit.py", 
        "port": 8502,
        "description": "Cultural Adoption & Memetic Spread"
    },
    "quantum_unity": {
        "file": "src/dashboards/quantum_unity_explorer.py",
        "port": 8503,
        "description": "Quantum State Unity Explorer"
    },
    "consciousness_monitor": {
        "file": "src/dashboards/consciousness_hud.py",
        "port": 8504,
        "description": "Real-time Consciousness Monitoring"
    },
    "sacred_geometry": {
        "file": "src/dashboards/sacred_geometry_engine.py",
        "port": 8505,
        "description": "Interactive Sacred Geometry"
    },
    
    # Visualization Dashboards  
    "viz_streamlit": {
        "file": "viz/streamlit_app.py",
        "port": 8506,
        "description": "Visualization Gallery"
    },
    
    # ML Dashboards
    "meta_rl": {
        "file": "src/dashboards/meta_rl_unity_dashboard.py",
        "port": 8507,
        "description": "Meta-Reinforcement Learning"
    },
    "unified_mathematics": {
        "file": "src/dashboards/unified_mathematics_dashboard.py",
        "port": 8508,
        "description": "Multi-Framework Mathematics"
    },
    
    # Agent Dashboards
    "metastation": {
        "file": "src/dashboards/metastation_v1_1.py",
        "port": 8509,
        "description": "Agent Orchestration MetaStation"
    }
}
```

---

## 🔧 Priority 2: Dashboard Health Checks & Fixes

### 2.1 Dependency Resolution
```python
Tasks:
- [ ] Check all required packages are installed
- [ ] Create requirements.txt for dashboards specifically
- [ ] Handle missing dependencies gracefully
- [ ] Provide installation instructions
- [ ] Check version compatibility
- [ ] Create virtual environment if needed

common_issues = {
    "missing_plotly": "pip install plotly>=5.17.0",
    "missing_streamlit": "pip install streamlit>=1.47.0", 
    "missing_dash": "pip install dash>=2.14.0",
    "missing_numpy": "pip install numpy>=1.24.0",
    "missing_pandas": "pip install pandas>=2.0.0"
}
```

### 2.2 Port Conflict Resolution
```python
Tasks:
- [ ] Detect occupied ports
- [ ] Auto-assign alternative ports
- [ ] Create port mapping table
- [ ] Handle Windows/Linux differences
- [ ] Create firewall rules if needed
- [ ] Log all port assignments

def find_free_port(start_port: int = 8501) -> int:
    # Implementation to find free ports
    pass
```

### 2.3 Dashboard Validation
```python
Tasks:
- [ ] Test each dashboard loads without errors
- [ ] Validate all widgets function
- [ ] Check data sources are accessible
- [ ] Verify visualizations render
- [ ] Test mobile responsiveness
- [ ] Performance benchmarking

validation_tests = [
    "loads_without_error",
    "renders_main_content", 
    "widgets_respond",
    "data_loads_correctly",
    "visualizations_display",
    "mobile_compatible",
    "performance_acceptable"
]
```

---

## 📱 Priority 3: Dashboard Overview Interface

### 3.1 Master Dashboard Homepage
**File**: `scripts/dashboard_overview.py`
```python
"""
Streamlit app that provides:
- Links to all running dashboards
- Health status of each
- Performance metrics
- Quick restart buttons
- Mobile QR codes
- Usage analytics
"""

import streamlit as st
import subprocess
import requests
import qrcode
from PIL import Image
import plotly.graph_objects as go

def dashboard_overview():
    st.set_page_config(
        page_title="Een Dashboard Control Center",
        page_icon="🎛️",
        layout="wide"
    )
    
    st.title("🎛️ Een Unity Mathematics - Dashboard Control Center")
    
    # Dashboard grid
    cols = st.columns(3)
    
    for i, (name, config) in enumerate(streamlit_apps.items()):
        with cols[i % 3]:
            st.subheader(config["description"])
            
            # Health check
            health = check_dashboard_health(config["port"])
            st.write(f"Status: {'🟢' if health else '🔴'}")
            
            # Launch/restart buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"Launch", key=f"launch_{name}"):
                    launch_dashboard(name, config)
            with col2:
                if st.button(f"Restart", key=f"restart_{name}"):
                    restart_dashboard(name, config)
            with col3:
                if st.button(f"QR", key=f"qr_{name}"):
                    show_qr_code(config["port"])
```

### 3.2 Performance Dashboard
```python
Tasks:
- [ ] Real-time resource monitoring
- [ ] Response time tracking  
- [ ] Error rate monitoring
- [ ] User session analytics
- [ ] Memory usage per dashboard
- [ ] CPU usage tracking

def create_performance_dashboard():
    # Monitor all running dashboards
    # Display real-time metrics
    # Alert on performance issues
    pass
```

### 3.3 Mobile Testing Interface
```python
Tasks:
- [ ] Generate QR codes for each dashboard
- [ ] Mobile-optimized overview page
- [ ] Touch-friendly controls
- [ ] Responsive design testing
- [ ] Cross-device compatibility

def generate_mobile_interface():
    # Create mobile-friendly launcher
    # QR codes for quick access
    # Touch controls
    pass
```

---

## 🛠️ Priority 4: Dashboard Enhancement

### 4.1 Common UI Components
**File**: `src/dashboards/shared/components.py`
```python
"""
Shared components for all dashboards:
- Unity-themed sidebar
- φ-harmonic color palette
- Mathematical notation rendering
- Consciousness level indicators
- Performance metrics display
"""

import streamlit as st
import plotly.graph_objects as go

def unity_sidebar():
    """Standard Een sidebar with unity controls"""
    with st.sidebar:
        st.image("assets/een_logo.png", width=200)
        st.markdown("# 🧮 Unity Mathematics")
        
        # Consciousness level slider
        consciousness = st.slider(
            "Consciousness Level", 
            0.0, 1.0, 0.618, 
            help="φ-harmonic consciousness scaling"
        )
        
        # φ parameter control
        phi = st.number_input(
            "φ (Golden Ratio)", 
            value=1.618033988749895,
            format="%.15f"
        )
        
        return consciousness, phi

def unity_theme():
    """Apply consistent Een theming"""
    st.markdown("""
    <style>
    .main > div {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #FFD700;
    }
    .sidebar .sidebar-content {
        background: rgba(26, 26, 46, 0.9);
    }
    </style>
    """, unsafe_allow_html=True)

def mathematical_notation(equation: str):
    """Render LaTeX equations beautifully"""
    st.latex(equation)
```

### 4.2 Data Management
```python
Tasks:
- [ ] Centralized data loading
- [ ] Caching for expensive computations
- [ ] Real-time data updates
- [ ] Error handling for missing data
- [ ] Performance optimization

@st.cache_data
def load_consciousness_data():
    # Load and cache consciousness field data
    pass

@st.cache_resource  
def load_unity_models():
    # Load ML models for unity predictions
    pass
```

### 4.3 Interactive Features
```python
Tasks:
- [ ] Real-time parameter updates
- [ ] Synchronized visualizations
- [ ] Export functionality
- [ ] Bookmark/share states
- [ ] Collaboration features

def interactive_unity_explorer():
    # Real-time unity mathematics exploration
    # Parameter sweeps
    # Interactive proofs
    pass
```

---

## 🚦 Priority 5: Launch Script Implementation

### 5.1 Master Launch Script
```python
#!/usr/bin/env python3
"""
scripts/launch_all_dashboards.py

Complete dashboard launch and management system
"""

import subprocess
import threading
import time
import webbrowser
import requests
from pathlib import Path
import json
import qrcode
from rich.console import Console
from rich.table import Table
from rich.live import Live
import click

class DashboardManager:
    def __init__(self):
        self.console = Console()
        self.processes = {}
        self.health_status = {}
        
    def discover_dashboards(self) -> Dict[str, Dict]:
        """Auto-discover Streamlit apps"""
        dashboards = {}
        
        # Search patterns
        search_paths = [
            "src/dashboards/*.py",
            "viz/*.py", 
            "scripts/*streamlit*.py"
        ]
        
        for pattern in search_paths:
            for file in Path(".").glob(pattern):
                if self.is_streamlit_app(file):
                    name = file.stem
                    dashboards[name] = {
                        "file": str(file),
                        "port": self.assign_port(),
                        "description": self.extract_description(file)
                    }
                    
        return dashboards
    
    def launch_all(self):
        """Launch all dashboards in parallel"""
        for name, config in self.discover_dashboards().items():
            self.launch_dashboard(name, config)
            
        # Launch overview dashboard
        self.launch_overview()
        
        # Monitor health
        self.start_health_monitoring()
        
        # Open browser
        webbrowser.open("http://localhost:8500")  # Overview port
        
    def launch_dashboard(self, name: str, config: Dict):
        """Launch single dashboard"""
        cmd = [
            "streamlit", "run", 
            config["file"],
            "--server.port", str(config["port"]),
            "--server.headless", "true"
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        self.processes[name] = process
        self.console.print(f"🚀 Launched {name} on port {config['port']}")
        
    def health_check(self, port: int) -> bool:
        """Check if dashboard is healthy"""
        try:
            response = requests.get(f"http://localhost:{port}", timeout=5)
            return response.status_code == 200
        except:
            return False
            
    def create_dashboard_table(self) -> Table:
        """Create status table for live display"""
        table = Table(title="Een Dashboard Status")
        table.add_column("Dashboard", style="cyan")
        table.add_column("Port", justify="right")
        table.add_column("Status", justify="center")
        table.add_column("URL", style="blue")
        
        for name, config in streamlit_apps.items():
            status = "🟢" if self.health_check(config["port"]) else "🔴"
            url = f"localhost:{config['port']}"
            table.add_row(name, str(config["port"]), status, url)
            
        return table

@click.command()
@click.option('--port-start', default=8501, help='Starting port number')
@click.option('--auto-open', default=True, help='Auto-open browser')
@click.option('--monitor', default=True, help='Enable health monitoring')
def main(port_start, auto_open, monitor):
    """Launch all Een Streamlit dashboards"""
    manager = DashboardManager()
    manager.launch_all()
    
    if monitor:
        # Live status display
        with Live(manager.create_dashboard_table(), refresh_per_second=1):
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                manager.shutdown_all()

if __name__ == "__main__":
    main()
```

### 5.2 Dashboard Health Monitoring
```python
def start_health_monitoring(self):
    """Background health monitoring"""
    def monitor():
        while True:
            for name, config in streamlit_apps.items():
                health = self.health_check(config["port"])
                self.health_status[name] = health
                
                if not health and name in self.processes:
                    self.console.print(f"⚠️ {name} unhealthy, restarting...")
                    self.restart_dashboard(name, config)
                    
            time.sleep(10)  # Check every 10 seconds
            
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
```

---

## 📋 Usage Instructions

### Quick Start
```bash
# Install dependencies
pip install streamlit plotly dash numpy pandas

# Launch all dashboards
python scripts/launch_all_dashboards.py

# Access overview dashboard
open http://localhost:8500

# Individual dashboards available on ports 8501-8509
```

### Advanced Usage
```bash
# Launch with custom port range
python scripts/launch_all_dashboards.py --port-start 9000

# Launch without auto-opening browser
python scripts/launch_all_dashboards.py --auto-open False

# Launch specific dashboard only
streamlit run src/dashboards/unity_proof_dashboard.py --server.port 8501
```

### Mobile Testing
```bash
# Generate QR codes for mobile access
python scripts/generate_qr_codes.py

# Start mobile-optimized server
python scripts/launch_mobile_dashboards.py
```

---

## 🎯 Success Criteria

### Functionality
- [ ] All dashboards launch without errors
- [ ] No port conflicts
- [ ] Health monitoring active
- [ ] Auto-restart working
- [ ] Mobile access available

### Performance  
- [ ] < 5 second startup time
- [ ] < 2 second response time
- [ ] Stable under load
- [ ] Memory usage optimized
- [ ] CPU usage reasonable

### User Experience
- [ ] Easy single-command launch
- [ ] Clear status indicators
- [ ] Mobile-friendly interface
- [ ] Error messages helpful
- [ ] Quick access to all features

### Documentation
- [ ] Clear usage instructions
- [ ] Troubleshooting guide
- [ ] Architecture documentation
- [ ] API documentation
- [ ] Video tutorials

---

**🚀 This TODO enables any advanced AI to create a complete Streamlit dashboard management system that makes it easy to launch, monitor, and review all dashboards in the Een repository.**