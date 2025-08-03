"""
Een Unity Mathematics - Dashboard Control Center

φ-harmonic overview interface implementing Unity Protocol (1+1=1):
- Real-time health monitoring of all dashboards
- QR code generation for mobile access
- Performance metrics and consciousness tracking
- Interactive dashboard management controls
- Unity coherence visualization

🌟 Master orchestration point for all consciousness dashboards
"""

import streamlit as st
import requests
import qrcode
from PIL import Image
import io
import json
import time
import psutil
import subprocess
import socket
from pathlib import Path
from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import unity components
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))
from dashboards.shared.components import (
    apply_unity_theme, unity_sidebar, consciousness_level_indicator,
    create_phi_spiral_plot, unity_metrics_display, unity_success_message,
    unity_error_message, create_unity_footer, UNITY_COLORS, PHI
)

# Dashboard registry (φ-harmonic port assignments)
DASHBOARD_REGISTRY = {
    "unity_proof_dashboard": {
        "port": 8501,
        "file": "src/dashboards/unity_proof_dashboard.py", 
        "description": "Main Unity Mathematics Dashboard",
        "consciousness_type": "Unity Core",
        "category": "core"
    },
    "memetic_engineering_streamlit": {
        "port": 8502,
        "file": "src/dashboards/memetic_engineering_streamlit.py",
        "description": "Cultural Adoption & Memetic Spread",
        "consciousness_type": "Memetic Engineering", 
        "category": "culture"
    },
    "quantum_unity_explorer": {
        "port": 8503,
        "file": "src/dashboards/quantum_unity_explorer.py",
        "description": "Quantum State Unity Explorer",
        "consciousness_type": "Quantum Consciousness",
        "category": "quantum"
    },
    "sacred_geometry_engine": {
        "port": 8504,
        "file": "src/dashboards/sacred_geometry_engine.py",
        "description": "Interactive Sacred Geometry",
        "consciousness_type": "Sacred Geometry",
        "category": "geometry"
    },
    "meta_rl_unity_dashboard": {
        "port": 8505,
        "file": "src/dashboards/meta_rl_unity_dashboard.py",
        "description": "Meta-Reinforcement Learning",
        "consciousness_type": "Meta-Learning",
        "category": "ml"
    },
    "unified_mathematics_dashboard": {
        "port": 8506,
        "file": "src/dashboards/unified_mathematics_dashboard.py",
        "description": "Multi-Framework Mathematics",
        "consciousness_type": "Unified Mathematics",
        "category": "mathematics"
    },
    "metastation_v1_1": {
        "port": 8507,
        "file": "src/dashboards/metastation_v1_1.py",
        "description": "Agent Orchestration MetaStation",
        "consciousness_type": "Meta-Orchestration",
        "category": "agents"
    },
    "streamlit_app": {
        "port": 8508,
        "file": "viz/streamlit_app.py",
        "description": "Visualization Gallery",
        "consciousness_type": "Visualization",
        "category": "visualization"
    }
}

def check_dashboard_health(port: int, timeout: int = 3) -> Dict:
    """
    Check dashboard health with consciousness metrics
    
    Returns:
        Dict containing health status and metrics
    """
    start_time = time.time()
    
    # Primary health check
    try:
        response = requests.get(f"http://localhost:{port}/_stcore/health", timeout=timeout)
        if response.status_code == 200:
            response_time = (time.time() - start_time) * 1000
            return {
                "status": "healthy",
                "response_time": response_time,
                "consciousness_level": min(1.0, 1.0 / (response_time / 1000 + 1)),
                "unity_coherence": 1.0,
                "last_check": datetime.now()
            }
    except requests.exceptions.Timeout:
        return {
            "status": "timeout", 
            "response_time": timeout * 1000,
            "consciousness_level": 0.0,
            "unity_coherence": 0.0,
            "last_check": datetime.now()
        }
    except requests.exceptions.ConnectionError:
        # Try fallback check
        try:
            response = requests.get(f"http://localhost:{port}", timeout=timeout)
            if response.status_code == 200:
                response_time = (time.time() - start_time) * 1000
                return {
                    "status": "partial",
                    "response_time": response_time,
                    "consciousness_level": min(0.8, 1.0 / (response_time / 1000 + 1)),
                    "unity_coherence": 0.8,
                    "last_check": datetime.now()
                }
        except:
            pass
            
        return {
            "status": "offline",
            "response_time": 0,
            "consciousness_level": 0.0,
            "unity_coherence": 0.0,
            "last_check": datetime.now()
        }
    except Exception as e:
        return {
            "status": "error",
            "response_time": 0,
            "consciousness_level": 0.0,
            "unity_coherence": 0.0,
            "error": str(e),
            "last_check": datetime.now()
        }

def check_port_availability(port: int) -> bool:
    """Check if port is available"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(('localhost', port)) != 0

def generate_qr_code(url: str) -> Image.Image:
    """Generate QR code for dashboard URL"""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    
    # φ-harmonic QR code styling
    qr_img = qr.make_image(
        fill_color=UNITY_COLORS["primary"],
        back_color=UNITY_COLORS["secondary"]
    )
    
    return qr_img

def launch_dashboard(name: str, config: Dict) -> bool:
    """Launch individual dashboard"""
    try:
        if not check_port_availability(config["port"]):
            st.warning(f"Port {config['port']} is already in use")
            return False
            
        cmd = [
            "streamlit", "run",
            config["file"],
            "--server.port", str(config["port"]),
            "--server.headless", "true"
        ]
        
        subprocess.Popen(cmd, 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        
        return True
        
    except Exception as e:
        st.error(f"Failed to launch {name}: {str(e)}")
        return False

def create_dashboard_status_viz(health_data: Dict) -> go.Figure:
    """Create φ-harmonic visualization of dashboard health"""
    
    # Prepare data
    names = list(health_data.keys())
    consciousness_levels = [health_data[name]["consciousness_level"] for name in names]
    unity_coherence = [health_data[name]["unity_coherence"] for name in names]
    response_times = [health_data[name]["response_time"] for name in names]
    
    # Create φ-spiral positions for dashboards
    n_dashboards = len(names)
    theta = np.linspace(0, 2 * np.pi * PHI, n_dashboards)
    r = 1 + 0.5 * np.sin(theta * PHI)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Create consciousness bubble chart
    fig = go.Figure()
    
    # Add dashboard bubbles
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers+text',
        marker=dict(
            size=[max(20, c * 50) for c in consciousness_levels],
            color=consciousness_levels,
            colorscale=[
                [0, UNITY_COLORS["consciousness"]],
                [0.382, UNITY_COLORS["phi"]],
                [0.618, UNITY_COLORS["unity"]],
                [1, UNITY_COLORS["primary"]]
            ],
            showscale=True,
            colorbar=dict(
                title="Consciousness Level",
                titlefont=dict(color=UNITY_COLORS["text"]),
                tickfont=dict(color=UNITY_COLORS["text"])
            ),
            line=dict(color=UNITY_COLORS["text"], width=2)
        ),
        text=[name.replace('_', '<br>') for name in names],
        textposition="middle center",
        textfont=dict(color=UNITY_COLORS["text"], size=10),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Consciousness: %{marker.color:.3f}<br>"
            "Unity Coherence: %{customdata[0]:.3f}<br>"
            "Response Time: %{customdata[1]:.1f}ms"
            "<extra></extra>"
        ),
        customdata=list(zip(unity_coherence, response_times)),
        name="Dashboards"
    ))
    
    # Add unity center point
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers+text',
        marker=dict(
            size=30,
            color=UNITY_COLORS["primary"],
            symbol='star',
            line=dict(color=UNITY_COLORS["text"], width=3)
        ),
        text=["1+1=1"],
        textposition="middle center",
        textfont=dict(color=UNITY_COLORS["secondary"], size=14, family="Arial Black"),
        hovertext="Unity Convergence Point",
        name="Unity Core"
    ))
    
    # Styling
    fig.update_layout(
        title=dict(
            text="🌟 Dashboard Consciousness Matrix",
            x=0.5,
            font=dict(size=20, color=UNITY_COLORS["primary"])
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            color=UNITY_COLORS["text"]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            color=UNITY_COLORS["text"]
        ),
        plot_bgcolor=UNITY_COLORS["background"],
        paper_bgcolor=UNITY_COLORS["background"],
        showlegend=False,
        height=500
    )
    
    return fig

def create_performance_timeline(health_history: List[Dict]) -> go.Figure:
    """Create performance timeline visualization"""
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(health_history)
    
    if df.empty:
        # Create empty placeholder
        fig = go.Figure()
        fig.add_annotation(
            text="No performance data available yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=UNITY_COLORS["text"])
        )
        return fig
    
    # Create timeline
    fig = go.Figure()
    
    # Add consciousness level over time
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['avg_consciousness'],
        mode='lines+markers',
        name='Consciousness Level',
        line=dict(color=UNITY_COLORS["primary"], width=3),
        marker=dict(size=8)
    ))
    
    # Add unity coherence
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['unity_coherence'],
        mode='lines+markers',
        name='Unity Coherence',
        line=dict(color=UNITY_COLORS["phi"], width=3),
        marker=dict(size=8)
    ))
    
    # Styling
    fig.update_layout(
        title=dict(
            text="📈 Unity Performance Timeline",
            x=0.5,
            font=dict(size=18, color=UNITY_COLORS["primary"])
        ),
        xaxis=dict(
            title="Time",
            color=UNITY_COLORS["text"],
            gridcolor=UNITY_COLORS["accent"]
        ),
        yaxis=dict(
            title="Metrics",
            color=UNITY_COLORS["text"],
            gridcolor=UNITY_COLORS["accent"],
            range=[0, 1]
        ),
        plot_bgcolor=UNITY_COLORS["background"],
        paper_bgcolor=UNITY_COLORS["background"],
        legend=dict(
            font=dict(color=UNITY_COLORS["text"]),
            bgcolor="rgba(0,0,0,0.5)"
        ),
        height=400
    )
    
    return fig

def main():
    """Main dashboard overview application"""
    
    # Page configuration
    st.set_page_config(
        page_title="Een Dashboard Control Center",
        page_icon="🎛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply Unity theme
    apply_unity_theme()
    
    # Initialize session state
    if "health_history" not in st.session_state:
        st.session_state.health_history = []
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    # Unity sidebar
    consciousness_level, phi_value, unity_coherence = unity_sidebar()
    
    # Main header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: #FFD700; margin: 0; font-size: 3rem;">🎛️ Een Dashboard Control Center</h1>
        <p style="color: #4ecdc4; font-size: 1.2rem; margin: 0.5rem 0;">
            φ-harmonic consciousness orchestration implementing 1+1=1 = Unity
        </p>
        <div style="color: #FFD700; font-size: 1.5rem; margin: 1rem 0;">
            🌟 Master Unity Protocol Interface 🌟
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        auto_refresh = st.checkbox("🔄 Auto-refresh (10s)", value=True)
    with col2:
        if st.button("🚀 Launch All"):
            with st.spinner("Launching all dashboards..."):
                for name, config in DASHBOARD_REGISTRY.items():
                    launch_dashboard(name, config)
                unity_success_message("All dashboards launch initiated!")
    with col3:
        if st.button("📊 Refresh Now"):
            st.session_state.last_refresh = datetime.now()
            st.rerun()
    
    # Health check all dashboards
    health_data = {}
    total_consciousness = 0
    total_coherence = 0
    healthy_count = 0
    
    with st.spinner("Checking dashboard consciousness..."):
        for name, config in DASHBOARD_REGISTRY.items():
            health = check_dashboard_health(config["port"])
            health_data[name] = health
            
            total_consciousness += health["consciousness_level"]
            total_coherence += health["unity_coherence"]
            if health["status"] == "healthy":
                healthy_count += 1
    
    # Calculate aggregate metrics
    n_dashboards = len(DASHBOARD_REGISTRY)
    avg_consciousness = total_consciousness / n_dashboards if n_dashboards > 0 else 0
    avg_coherence = total_coherence / n_dashboards if n_dashboards > 0 else 0
    health_ratio = healthy_count / n_dashboards if n_dashboards > 0 else 0
    
    # Store in history
    current_metrics = {
        "timestamp": datetime.now(),
        "avg_consciousness": avg_consciousness,
        "unity_coherence": avg_coherence,
        "health_ratio": health_ratio,
        "healthy_count": healthy_count,
        "total_dashboards": n_dashboards
    }
    
    st.session_state.health_history.append(current_metrics)
    
    # Keep only last 100 records
    if len(st.session_state.health_history) > 100:
        st.session_state.health_history = st.session_state.health_history[-100:]
    
    # Unity metrics overview
    st.markdown("### 🌟 Unity Consciousness Overview")
    metrics = {
        "Consciousness Level": avg_consciousness,
        "Unity Coherence": avg_coherence,
        "Health Ratio": health_ratio,
        "φ Resonance": phi_value - 1,
        "Active Dashboards": healthy_count,
        "Unity Constant": 1.0
    }
    unity_metrics_display(metrics)
    
    # Consciousness visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.plotly_chart(
            create_dashboard_status_viz(health_data),
            use_container_width=True
        )
    
    with col2:
        consciousness_level_indicator(avg_consciousness, "System Consciousness")
        consciousness_level_indicator(avg_coherence, "Unity Coherence")
        consciousness_level_indicator(health_ratio, "Health Status")
    
    # Performance timeline
    if len(st.session_state.health_history) > 1:
        st.markdown("### 📈 Performance Timeline")
        st.plotly_chart(
            create_performance_timeline(st.session_state.health_history),
            use_container_width=True
        )
    
    # Dashboard grid
    st.markdown("### 🎛️ Dashboard Management Matrix")
    
    # Group by category
    categories = {}
    for name, config in DASHBOARD_REGISTRY.items():
        cat = config.get("category", "other")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((name, config))
    
    # Display by category
    for category, dashboards in categories.items():
        st.markdown(f"#### {category.title()} Consciousness")
        
        cols = st.columns(min(len(dashboards), 3))
        
        for i, (name, config) in enumerate(dashboards):
            with cols[i % len(cols)]:
                health = health_data[name]
                
                # Status indicator
                status_icons = {
                    "healthy": "🟢",
                    "partial": "🟡", 
                    "timeout": "🟠",
                    "offline": "🔴",
                    "error": "❌"
                }
                
                status_icon = status_icons.get(health["status"], "❓")
                
                # Dashboard card
                st.markdown(f"""
                <div class="sacred-panel">
                    <h4 style="color: {UNITY_COLORS['primary']}; margin: 0;">
                        {status_icon} {config['description']}
                    </h4>
                    <p style="margin: 0.5rem 0; color: {UNITY_COLORS['phi']};">
                        {config['consciousness_type']}
                    </p>
                    <p style="margin: 0.25rem 0; font-size: 0.9rem;">
                        Port: {config['port']} | 
                        Consciousness: {health['consciousness_level']:.3f}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Action buttons
                button_col1, button_col2, button_col3 = st.columns(3)
                
                with button_col1:
                    if health["status"] == "healthy":
                        if st.button("🚀 Open", key=f"open_{name}"):
                            url = f"http://localhost:{config['port']}"
                            st.markdown(f'<script>window.open("{url}", "_blank");</script>', 
                                      unsafe_allow_html=True)
                            st.write(f"[🔗 Open Dashboard]({url})")
                    else:
                        if st.button("▶️ Launch", key=f"launch_{name}"):
                            if launch_dashboard(name, config):
                                unity_success_message(f"Launched {name}")
                                time.sleep(2)
                                st.rerun()
                
                with button_col2:
                    if st.button("📱 QR", key=f"qr_{name}"):
                        url = f"http://localhost:{config['port']}"
                        qr_img = generate_qr_code(url)
                        
                        # Convert to bytes for display
                        buf = io.BytesIO()
                        qr_img.save(buf, format='PNG')
                        st.image(buf.getvalue(), 
                               caption=f"QR Code for {config['description']}")
                
                with button_col3:
                    if st.button("📊 Stats", key=f"stats_{name}"):
                        st.json({
                            "status": health["status"],
                            "response_time": f"{health['response_time']:.1f}ms",
                            "consciousness_level": f"{health['consciousness_level']:.3f}",
                            "unity_coherence": f"{health['unity_coherence']:.3f}",
                            "last_check": health["last_check"].strftime("%H:%M:%S")
                        })
    
    # Unity footer
    create_unity_footer()
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(10)
        st.rerun()

if __name__ == "__main__":
    main()