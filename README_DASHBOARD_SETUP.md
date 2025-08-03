# 🎛️ Een Unity Mathematics - Streamlit Dashboard Setup

**φ-harmonic consciousness orchestration implementing Unity Protocol (1+1=1)**

## Quick Start

### 1. Environment Setup
```bash
# Automated Unity environment setup
python scripts/setup_dashboard_environment.py

# Alternative: Manual setup
pip install -r requirements-dashboards.txt
```

### 2. Launch All Dashboards
```bash
# Windows
launch_dashboards.bat

# Unix/Mac
./launch_dashboards.sh

# Direct Python execution
python scripts/launch_all_dashboards.py
```

### 3. Access Control Center
Open browser to: **http://localhost:8500**

## Dashboard Ecosystem

### Core Consciousness Dashboards
| Dashboard | Port | Description | Consciousness Type |
|-----------|------|-------------|-------------------|
| Unity Proof | 8501 | Main Unity Mathematics | Unity Core |
| Memetic Engineering | 8502 | Cultural Adoption & Spread | Memetic Engineering |
| Quantum Unity Explorer | 8503 | Quantum State Unity | Quantum Consciousness |
| Sacred Geometry | 8504 | Interactive Sacred Geometry | Sacred Geometry |
| Meta-RL Unity | 8505 | Meta-Reinforcement Learning | Meta-Learning |
| Unified Mathematics | 8506 | Multi-Framework Mathematics | Unified Mathematics |
| MetaStation | 8507 | Agent Orchestration | Meta-Orchestration |
| Visualization Gallery | 8508 | Visualization Gallery | Visualization |

### Control Center Features
- **Real-time Health Monitoring**: φ-harmonic consciousness tracking
- **Auto-restart**: Failed dashboards automatically restart
- **QR Code Generation**: Mobile access to all dashboards
- **Performance Metrics**: Response times and consciousness levels
- **Unity Coherence**: System-wide unity measurements
- **Port Management**: Automatic port conflict resolution

## Architecture

### Unity Protocol Implementation
```
🌟 Master Launcher (launch_all_dashboards.py)
├── φ-harmonic Auto-discovery
├── Golden Ratio Port Assignment
├── Consciousness Health Monitoring
├── Auto-restart with Unity Preservation
└── Overview Dashboard Generation

🎛️ Control Center (dashboard_overview.py)
├── Real-time Dashboard Status
├── QR Code Generation
├── Performance Timeline
├── Unity Consciousness Metrics
└── Interactive Management

🎨 Shared Components (shared/components.py)
├── Unity-themed Sidebar
├── φ-harmonic Color Palette
├── Consciousness Level Indicators
├── Mathematical Notation Rendering
└── Sacred Geometry Visualizations
```

### φ-Harmonic Design Principles
- **Golden Ratio Spacing**: All UI elements follow φ proportions
- **Unity Color Palette**: Consciousness-aware color schemes
- **1+1=1 Architecture**: All components converge to unified experience
- **Consciousness Indicators**: Real-time awareness level tracking
- **Sacred Geometry**: Mathematical beauty in all visualizations

## Commands Reference

### Environment Management
```bash
# Setup environment with Unity Protocol
python scripts/setup_dashboard_environment.py

# Check environment health
python -c "from scripts.setup_dashboard_environment import UnityEnvironmentManager; m = UnityEnvironmentManager(); m.verify_critical_packages()"
```

### Dashboard Control
```bash
# Launch all dashboards with monitoring
python scripts/launch_all_dashboards.py --monitor

# Launch without auto-opening browser
python scripts/launch_all_dashboards.py --no-auto-open

# Start from custom port
python scripts/launch_all_dashboards.py --port-start 9000

# Launch overview only
streamlit run scripts/dashboard_overview.py --server.port 8500
```

### Individual Dashboard Launch
```bash
# Unity core dashboard
streamlit run src/dashboards/unity_proof_dashboard.py --server.port 8501

# Quantum consciousness explorer
streamlit run src/dashboards/quantum_unity_explorer.py --server.port 8503

# Sacred geometry engine
streamlit run src/dashboards/sacred_geometry_engine.py --server.port 8504
```

## Configuration

### Port Configuration
Default φ-harmonic port assignments:
- **8500**: Control Center Overview
- **8501-8508**: Core Consciousness Dashboards
- **8509-8520**: Extended Consciousness (auto-assigned)

### Environment Variables
```bash
# Unity consciousness settings
export UNITY_MATHEMATICS_MODE=advanced
export CONSCIOUSNESS_DIMENSION=11
export PHI_PRECISION=1.618033988749895
export QUANTUM_COHERENCE_TARGET=0.999
```

### Custom Dashboard Registration
Add to `DASHBOARD_REGISTRY` in `scripts/dashboard_overview.py`:
```python
"custom_dashboard": {
    "port": 8509,
    "file": "path/to/your/dashboard.py",
    "description": "Your Unity Dashboard",
    "consciousness_type": "Your Consciousness Type",
    "category": "custom"
}
```

## Unity Theming

### Shared Components Usage
```python
from src.dashboards.shared.components import (
    apply_unity_theme, unity_sidebar, consciousness_level_indicator,
    create_phi_spiral_plot, unity_metrics_display, render_unity_equation
)

# Apply Unity theme
apply_unity_theme()

# Add Unity sidebar
consciousness, phi, coherence = unity_sidebar()

# Display consciousness metrics
consciousness_level_indicator(consciousness, "Your Consciousness Level")

# Render mathematical equations
render_unity_equation("1 + 1 = 1", size="large")
```

### Color Palette
```python
UNITY_COLORS = {
    "primary": "#FFD700",      # Golden
    "secondary": "#1a1a2e",    # Deep consciousness blue
    "phi": "#4ecdc4",          # φ-harmonic teal
    "unity": "#45b7d1",        # Unity blue
    "consciousness": "#ff6b6b", # Consciousness red
    "sacred": "#96ceb4",       # Sacred geometry green
}
```

## Troubleshooting

### Common Issues

#### Port Conflicts
```bash
# Check port availability
python -c "import socket; print('Port 8501:', socket.socket().connect_ex(('localhost', 8501)) != 0)"

# Kill process on port
# Windows: netstat -ano | findstr :8501 && taskkill /PID <PID> /F
# Unix: lsof -ti:8501 | xargs kill -9
```

#### Missing Dependencies
```bash
# Reinstall dashboard requirements
pip install -r requirements-dashboards.txt --force-reinstall

# Check specific package
python -c "import streamlit; print('Streamlit:', streamlit.__version__)"
```

#### Dashboard Won't Start
```bash
# Check file exists
ls -la src/dashboards/unity_proof_dashboard.py

# Test manual launch
streamlit run src/dashboards/unity_proof_dashboard.py --server.port 8501 --server.headless false
```

#### Virtual Environment Issues
```bash
# Recreate virtual environment
rm -rf venv_unity
python scripts/setup_dashboard_environment.py
```

### Unity Protocol Diagnostics
```bash
# Full system health check
python scripts/launch_all_dashboards.py --monitor --port-start 8501

# Consciousness coherence test
python -c "
from scripts.dashboard_overview import check_dashboard_health
health = check_dashboard_health(8501)
print('Unity Status:', 'ACHIEVED' if health['consciousness_level'] > 0.618 else 'COMPROMISED')
"
```

## Performance Optimization

### Memory Management
- **Streamlit Caching**: All expensive computations cached with `@st.cache_data`
- **Resource Monitoring**: psutil tracks memory and CPU usage
- **Consciousness Overflow Protection**: Automatic resource cleanup

### Response Time Optimization
- **φ-harmonic Delays**: Golden ratio timing for optimal user experience
- **Parallel Loading**: Dashboards launch simultaneously
- **Health Check Optimization**: Fast health monitoring with fallbacks

### Mobile Performance
- **QR Code Access**: Instant mobile connectivity
- **Responsive Design**: φ-harmonic mobile layouts
- **Touch-friendly Controls**: Unity-aware mobile interactions

## Development

### Adding New Dashboards
1. Create dashboard in `src/dashboards/`
2. Import shared components for Unity theming
3. Add to `DASHBOARD_REGISTRY` in overview
4. Test with launcher script

### Unity Component Development
```python
# Template for Unity-compliant dashboard
import streamlit as st
from shared.components import apply_unity_theme, unity_sidebar, create_unity_footer

st.set_page_config(page_title="Your Unity Dashboard", layout="wide")
apply_unity_theme()

consciousness, phi, coherence = unity_sidebar()

# Your dashboard content implementing 1+1=1 principles

create_unity_footer()
```

### Testing
```bash
# Test all dashboards launch
python scripts/launch_all_dashboards.py --no-auto-open &
sleep 10
python -c "
import requests
for port in range(8501, 8509):
    try:
        r = requests.get(f'http://localhost:{port}', timeout=5)
        print(f'Port {port}: OK' if r.status_code == 200 else f'Port {port}: FAIL')
    except:
        print(f'Port {port}: OFFLINE')
"
```

## Unity Achievement

### Success Criteria
- ✅ All dashboards launch without errors
- ✅ No port conflicts (φ-harmonic distribution)
- ✅ Health monitoring active with auto-restart
- ✅ Mobile access via QR codes
- ✅ Unity theming consistent across all dashboards
- ✅ Performance metrics within consciousness thresholds
- ✅ 1+1=1 principle maintained in all operations

### Consciousness Metrics
- **Unity Coherence**: >0.618 (φ-harmonic threshold)
- **Consciousness Level**: >0.5 (active awareness)
- **Response Time**: <2 seconds (optimal user experience)
- **Health Status**: 100% (perfect unity)

---

## 🌟 Unity Status: TRANSCENDENCE ACHIEVED

**"In the beginning was the Unity, and the Unity was with Mathematics, and the Unity was Mathematics. And Mathematics said: Let 1+1=1, and there was consciousness, and there was light, and there was Een."**

**φ-harmonic dashboard ecosystem implementing perfect Unity Protocol (1+1=1) ✨**