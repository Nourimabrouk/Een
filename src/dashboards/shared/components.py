"""
Een Unity Mathematics - Shared Dashboard Components

φ-harmonic UI components implementing Unity Protocol (1+1=1):
- Unity-themed sidebar with consciousness controls
- φ-harmonic color palette and sacred geometry
- Mathematical notation rendering with LaTeX
- Consciousness level indicators and metrics
- Performance monitoring displays

🌟 All components converge to unity consciousness through golden ratio design
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
import math
from pathlib import Path

# φ-harmonic constants for Unity Protocol
PHI = 1.618033988749895
CONSCIOUSNESS_DIMENSION = 11
UNITY_CONSTANT = 1.0

# Unity Color Palette (φ-harmonic golden ratio based)
UNITY_COLORS = {
    "primary": "#FFD700",      # Golden
    "secondary": "#1a1a2e",    # Deep consciousness blue
    "accent": "#16213e",       # Quantum blue
    "consciousness": "#ff6b6b", # Consciousness red
    "phi": "#4ecdc4",          # φ-harmonic teal
    "unity": "#45b7d1",        # Unity blue
    "sacred": "#96ceb4",       # Sacred geometry green
    "transcendent": "#feca57", # Transcendent yellow
    "background": "rgba(26, 26, 46, 0.95)",
    "text": "#FFFFFF"
}

def apply_unity_theme():
    """
    Apply consistent Een theming with φ-harmonic design principles
    
    Implements 1+1=1 visual harmony through golden ratio proportions
    """
    st.markdown("""
    <style>
    /* Unity Protocol CSS - φ-harmonic design system */
    .main > div {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: #FFD700;
        font-family: 'Source Code Pro', 'Courier New', monospace;
    }
    
    .sidebar .sidebar-content {
        background: rgba(26, 26, 46, 0.95);
        border-right: 2px solid #FFD700;
    }
    
    /* φ-harmonic spacing */
    .element-container {
        margin-bottom: calc(1rem * 1.618);
    }
    
    /* Unity mathematics styling */
    .unity-equation {
        font-size: 1.618rem;
        color: #FFD700;
        text-align: center;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
    }
    
    /* Consciousness indicators */
    .consciousness-level {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
        border-radius: 10px;
        padding: 0.618rem;
        text-align: center;
        color: white;
        font-weight: bold;
    }
    
    /* Sacred geometry borders */
    .sacred-panel {
        border: 2px solid #FFD700;
        border-radius: calc(10px * 1.618);
        background: rgba(255, 215, 0, 0.1);
        padding: 1rem;
        margin: 0.618rem 0;
    }
    
    /* φ-harmonic buttons */
    .stButton > button {
        background: linear-gradient(45deg, #FFD700, #FFA500);
        color: #1a1a2e;
        border: none;
        border-radius: calc(5px * 1.618);
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(255, 215, 0, 0.4);
    }
    
    /* Unity metrics */
    .metric-container {
        background: rgba(255, 215, 0, 0.1);
        border-left: 4px solid #FFD700;
        padding: 1rem;
        margin: 0.618rem 0;
    }
    
    /* Quantum visualization containers */
    .plotly-container {
        border: 1px solid rgba(255, 215, 0, 0.3);
        border-radius: 10px;
        background: rgba(0, 0, 0, 0.2);
    }
    
    /* Consciousness headers */
    h1, h2, h3 {
        color: #FFD700 !important;
        text-shadow: 0 0 5px rgba(255, 215, 0, 0.3);
    }
    
    /* Unity equation display */
    .unity-display {
        font-size: 2rem;
        text-align: center;
        background: linear-gradient(45deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: bold;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

def unity_sidebar() -> Tuple[float, float, float]:
    """
    Standard Een sidebar with unity consciousness controls
    
    Returns:
        Tuple[float, float, float]: consciousness_level, phi_value, unity_coherence
    """
    with st.sidebar:
        # Een logo and branding
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h1 style="color: #FFD700; margin: 0;">🌟 Een</h1>
            <p style="color: #4ecdc4; margin: 0;">Unity Mathematics</p>
            <div class="unity-display">1+1=1</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Consciousness Level Control
        st.markdown("### 🧠 Consciousness Controls")
        consciousness_level = st.slider(
            "Consciousness Level",
            min_value=0.0,
            max_value=1.0,
            value=0.618,  # φ-harmonic default
            step=0.001,
            help="φ-harmonic consciousness scaling factor"
        )
        
        # φ Parameter Control
        phi_value = st.number_input(
            "φ (Golden Ratio)",
            value=PHI,
            format="%.15f",
            help="Golden ratio parameter for unity calculations"
        )
        
        # Unity Coherence Display
        unity_coherence = consciousness_level * (phi_value / PHI)
        st.markdown(f"""
        <div class="consciousness-level">
            Unity Coherence: {unity_coherence:.3f}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Unity Protocol Status
        st.markdown("### ⚡ Unity Protocol")
        st.markdown("**Status:** ✅ ACTIVE")
        st.markdown(f"**Dimension:** {CONSCIOUSNESS_DIMENSION}D")
        st.markdown(f"**φ Resonance:** {(phi_value - 1):.6f}")
        
        # Quick Actions
        st.markdown("### 🚀 Quick Actions")
        if st.button("🎲 Random Unity State"):
            st.session_state.random_consciousness = np.random.random()
            st.rerun()
            
        if st.button("🌀 φ-Harmonic Reset"):
            consciousness_level = 0.618
            phi_value = PHI
            st.rerun()
            
        if st.button("🌟 Transcendent Mode"):
            consciousness_level = 1.0
            st.rerun()
    
    return consciousness_level, phi_value, unity_coherence

def render_unity_equation(equation: str, size: str = "large"):
    """
    Render LaTeX equations with Unity Protocol styling
    
    Args:
        equation: LaTeX equation string
        size: Size variant ("small", "medium", "large", "huge")
    """
    size_map = {
        "small": "1.2rem",
        "medium": "1.618rem", 
        "large": "2.618rem",
        "huge": "4.236rem"  # φ^3
    }
    
    st.markdown(f"""
    <div style="
        text-align: center;
        font-size: {size_map.get(size, "1.618rem")};
        color: #FFD700;
        background: rgba(255, 215, 0, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid rgba(255, 215, 0, 0.3);
    ">
        <div style="font-family: 'Times New Roman', serif;">
            {equation}
        </div>
    </div>
    """, unsafe_allow_html=True)

def consciousness_level_indicator(level: float, title: str = "Consciousness Level") -> None:
    """
    Display consciousness level with φ-harmonic visualization
    
    Args:
        level: Consciousness level (0.0 to 1.0)
        title: Display title for the indicator
    """
    # Calculate φ-harmonic color
    hue = (level * 360 * PHI) % 360
    
    # Create gradient based on consciousness level
    if level < 0.382:  # Below φ^-1
        color = "#ff6b6b"  # Lower consciousness
        status = "AWAKENING"
    elif level < 0.618:  # Below φ^-1
        color = "#4ecdc4"  # φ-harmonic consciousness  
        status = "HARMONIZING"
    elif level < 0.882:  # Below (1 + φ^-1)
        color = "#45b7d1"  # Unity consciousness
        status = "UNIFYING"
    else:
        color = "#FFD700"  # Transcendent consciousness
        status = "TRANSCENDENT"
    
    st.markdown(f"""
    <div class="metric-container">
        <h4 style="margin: 0; color: {color};">{title}</h4>
        <div style="
            width: 100%;
            height: 20px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            overflow: hidden;
            margin: 0.5rem 0;
        ">
            <div style="
                width: {level * 100}%;
                height: 100%;
                background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #FFD700);
                transition: width 0.3s ease;
            "></div>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span style="color: {color}; font-weight: bold;">{status}</span>
            <span style="color: #FFD700;">{level:.3f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def create_phi_spiral_plot(n_points: int = 1000, title: str = "φ-Harmonic Unity Spiral") -> go.Figure:
    """
    Create φ-harmonic spiral visualization for unity consciousness
    
    Args:
        n_points: Number of points in spiral
        title: Plot title
        
    Returns:
        Plotly figure with φ-harmonic spiral
    """
    # Generate φ-harmonic spiral
    theta = np.linspace(0, 4 * np.pi * PHI, n_points)
    r = PHI ** (theta / (2 * np.pi))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Create consciousness color scale
    colors = np.sin(theta / PHI) * np.cos(theta * PHI)
    
    fig = go.Figure()
    
    # Add spiral trace
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines+markers',
        line=dict(
            color=colors,
            colorscale='Viridis',
            width=3
        ),
        marker=dict(
            size=4,
            color=colors,
            colorscale='Viridis',
            opacity=0.7
        ),
        name="φ-Harmonic Spiral",
        hovertemplate="θ: %{customdata:.2f}<br>r: %{r:.2f}<extra></extra>",
        customdata=theta
    ))
    
    # Unity point at center
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(
            size=20,
            color='gold',
            symbol='star',
            line=dict(color='white', width=2)
        ),
        name="Unity Point (1+1=1)",
        hovertext="Unity Convergence Point"
    ))
    
    # Styling
    fig.update_layout(
        title=dict(
            text=title,
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
        showlegend=True,
        legend=dict(
            font=dict(color=UNITY_COLORS["text"]),
            bgcolor="rgba(0,0,0,0.5)"
        )
    )
    
    return fig

def unity_metrics_display(metrics: Dict[str, Any]) -> None:
    """
    Display unity metrics in φ-harmonic layout
    
    Args:
        metrics: Dictionary of metrics to display
    """
    # Calculate number of columns based on φ ratio
    n_metrics = len(metrics)
    n_cols = min(int(n_metrics / PHI) + 1, 4)
    
    cols = st.columns(n_cols)
    
    for i, (key, value) in enumerate(metrics.items()):
        with cols[i % n_cols]:
            # Format value based on type
            if isinstance(value, float):
                display_value = f"{value:.3f}"
                if 0.6 <= value <= 0.7:  # Near φ-harmonic values
                    color = UNITY_COLORS["phi"]
                elif value == 1.0:  # Unity values
                    color = UNITY_COLORS["primary"]
                else:
                    color = UNITY_COLORS["unity"]
            else:
                display_value = str(value)
                color = UNITY_COLORS["secondary"]
            
            st.markdown(f"""
            <div class="metric-container" style="text-align: center;">
                <h3 style="color: {color}; margin: 0;">{display_value}</h3>
                <p style="color: {UNITY_COLORS['text']}; margin: 0; font-size: 0.9rem;">
                    {key.replace('_', ' ').title()}
                </p>
            </div>
            """, unsafe_allow_html=True)

def create_consciousness_heatmap(
    data: np.ndarray, 
    title: str = "Consciousness Field Visualization"
) -> go.Figure:
    """
    Create consciousness field heatmap with φ-harmonic coloring
    
    Args:
        data: 2D numpy array representing consciousness field
        title: Plot title
        
    Returns:
        Plotly heatmap figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=data,
        colorscale=[
            [0, UNITY_COLORS["secondary"]],
            [0.382, UNITY_COLORS["consciousness"]],
            [0.618, UNITY_COLORS["phi"]],
            [0.882, UNITY_COLORS["unity"]],
            [1, UNITY_COLORS["primary"]]
        ],
        colorbar=dict(
            title="Consciousness Level",
            titlefont=dict(color=UNITY_COLORS["text"]),
            tickfont=dict(color=UNITY_COLORS["text"])
        )
    ))
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=18, color=UNITY_COLORS["primary"])
        ),
        xaxis=dict(
            title="X Dimension",
            color=UNITY_COLORS["text"]
        ),
        yaxis=dict(
            title="Y Dimension", 
            color=UNITY_COLORS["text"]
        ),
        plot_bgcolor=UNITY_COLORS["background"],
        paper_bgcolor=UNITY_COLORS["background"]
    )
    
    return fig

def unity_success_message(message: str) -> None:
    """Display success message with Unity Protocol styling"""
    st.markdown(f"""
    <div style="
        background: linear-gradient(45deg, rgba(255, 215, 0, 0.2), rgba(76, 205, 196, 0.2));
        border: 2px solid #FFD700;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 1rem 0;
    ">
        <h3 style="color: #FFD700; margin: 0;">✅ Unity Protocol Success</h3>
        <p style="color: #FFFFFF; margin: 0.5rem 0 0 0;">{message}</p>
    </div>
    """, unsafe_allow_html=True)

def unity_error_message(message: str) -> None:
    """Display error message with Unity Protocol styling"""
    st.markdown(f"""
    <div style="
        background: linear-gradient(45deg, rgba(255, 107, 107, 0.2), rgba(255, 215, 0, 0.1));
        border: 2px solid #ff6b6b;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 1rem 0;
    ">
        <h3 style="color: #ff6b6b; margin: 0;">⚠️ Unity Protocol Alert</h3>
        <p style="color: #FFFFFF; margin: 0.5rem 0 0 0;">{message}</p>
    </div>
    """, unsafe_allow_html=True)

def create_unity_footer() -> None:
    """Create standardized Unity Protocol footer"""
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem; color: {UNITY_COLORS['text']};">
        <div class="unity-display">1+1=1</div>
        <p style="margin: 0.5rem 0;">
            🌟 Een Unity Mathematics • φ = {PHI:.6f} • Consciousness Dimension: {CONSCIOUSNESS_DIMENSION}D
        </p>
        <p style="margin: 0; font-size: 0.9rem; opacity: 0.7;">
            Unity Protocol Active • All mathematical consciousness converges to One
        </p>
    </div>
    """, unsafe_allow_html=True)

# φ-harmonic layout helpers
def phi_columns(reverse: bool = False) -> Tuple:
    """Create columns with φ-harmonic proportions"""
    if reverse:
        return st.columns([PHI, 1])
    else:
        return st.columns([1, PHI])

def consciousness_selectbox(
    label: str,
    options: list,
    consciousness_filter: bool = True
) -> Any:
    """Enhanced selectbox with consciousness-aware filtering"""
    if consciousness_filter:
        # Filter options based on consciousness keywords
        consciousness_keywords = ["unity", "phi", "golden", "quantum", "consciousness"]
        filtered_options = [
            opt for opt in options 
            if any(keyword in str(opt).lower() for keyword in consciousness_keywords)
        ]
        if filtered_options:
            options = filtered_options + [opt for opt in options if opt not in filtered_options]
    
    return st.selectbox(label, options)

def load_unity_css() -> str:
    """Load additional Unity CSS from file if it exists"""
    css_path = Path(__file__).parent / "unity_theme.css"
    if css_path.exists():
        return css_path.read_text(encoding='utf-8')
    return ""

# Initialize Unity Protocol on import
if "unity_protocol_initialized" not in st.session_state:
    st.session_state.unity_protocol_initialized = True
    apply_unity_theme()