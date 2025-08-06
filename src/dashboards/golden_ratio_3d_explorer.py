"""
ENHANCED 3D GOLDEN RATIO EXPLORER DASHBOARD
Interactive φ-harmonic visualizations with user controls and animations
Demonstrates the unity principle through golden ratio geometry
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import math
from scipy.spatial.transform import Rotation

# Page configuration
st.set_page_config(
    page_title="φ-Harmonic 3D Explorer | Een Unity Mathematics",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio φ = 1.618...
GOLDEN_ANGLE = 2 * np.pi / PHI  # 137.5 degrees

# Custom CSS for φ-harmonic styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, rgba(27, 54, 93, 0.05) 0%, rgba(15, 123, 138, 0.03) 100%);
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #1B365D, #0F7B8A);
        color: white;
        border-radius: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .phi-highlight {
        color: #F59E0B;
        font-weight: bold;
        font-size: 1.2em;
    }
    
    .unity-equation {
        font-family: 'Times New Roman', serif;
        font-size: 1.5em;
        text-align: center;
        color: #0F7B8A;
        margin: 1rem 0;
        padding: 1rem;
        background: rgba(15, 123, 138, 0.1);
        border-radius: 0.5rem;
        border-left: 4px solid #F59E0B;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🌟 φ-Harmonic 3D Explorer</h1>
    <p>Interactive Golden Ratio Visualizations & Unity Mathematics</p>
    <p style="font-size: 0.9em; opacity: 0.8;">Explore the infinite beauty of φ = (1 + √5) / 2 ≈ 1.618033988749895</p>
</div>
""", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.markdown("## 🎛️ φ-Harmonic Controls")

# Visualization type selector
viz_type = st.sidebar.selectbox(
    "Choose Visualization",
    ["Golden Spiral 3D", "Fibonacci Phyllotaxis", "φ-Harmonic Torus", "Unity Convergence", "Sacred Geometry Matrix"]
)

# Animation controls
animate = st.sidebar.checkbox("🎬 Enable Animation", value=True)
animation_speed = st.sidebar.slider("Animation Speed", 0.1, 2.0, 1.0, 0.1)

# Mathematical parameters
st.sidebar.markdown("### 📐 Mathematical Parameters")
phi_factor = st.sidebar.slider("φ Factor Multiplier", 0.5, 3.0, 1.0, 0.01)
spiral_turns = st.sidebar.slider("Spiral Turns", 1, 20, 10, 1)
resolution = st.sidebar.slider("Resolution", 50, 500, 200, 25)

# Color scheme
color_scheme = st.sidebar.selectbox(
    "Color Palette",
    ["φ-Harmonic", "Unity Teal", "Consciousness Purple", "Sacred Gold", "Academic Blue"]
)

# Color palettes
color_palettes = {
    "φ-Harmonic": ["#0F7B8A", "#4ECDC4", "#A7F3D0", "#F59E0B"],
    "Unity Teal": ["#0D9488", "#14B8A6", "#5EEAD4", "#A7F3D0"],
    "Consciousness Purple": ["#7C3AED", "#A78BFA", "#C4B5FD", "#E9D5FF"],
    "Sacred Gold": ["#D97706", "#F59E0B", "#FBBF24", "#FEF3C7"],
    "Academic Blue": ["#1E40AF", "#3B82F6", "#93C5FD", "#DBEAFE"]
}

colors = color_palettes[color_scheme]

def create_golden_spiral_3d():
    """Create interactive 3D golden spiral"""
    t = np.linspace(0, spiral_turns * 2 * np.pi, resolution)
    phi_adjusted = PHI * phi_factor
    
    # Golden spiral in cylindrical coordinates
    r = np.exp(t / phi_adjusted)
    x = r * np.cos(t)
    y = r * np.sin(t)
    z = t * 0.1  # Height progression
    
    # Create the main spiral
    fig = go.Figure()
    
    # Add spiral curve
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines+markers',
        line=dict(color=colors[0], width=8),
        marker=dict(size=3, color=colors[1]),
        name=f'Golden Spiral (φ = {phi_adjusted:.3f})'
    ))
    
    # Add golden ratio rectangles
    for i in range(0, len(t), len(t)//10):
        if i + 1 < len(t):
            rect_r = r[i]
            rect_angle = t[i]
            
            # Rectangle corners
            rect_x = [rect_r * np.cos(rect_angle), 
                     rect_r * np.cos(rect_angle + np.pi/2),
                     rect_r * np.cos(rect_angle + np.pi),
                     rect_r * np.cos(rect_angle + 3*np.pi/2),
                     rect_r * np.cos(rect_angle)]
            rect_y = [rect_r * np.sin(rect_angle),
                     rect_r * np.sin(rect_angle + np.pi/2),
                     rect_r * np.sin(rect_angle + np.pi),
                     rect_r * np.sin(rect_angle + 3*np.pi/2),
                     rect_r * np.sin(rect_angle)]
            rect_z = [z[i]] * 5
            
            fig.add_trace(go.Scatter3d(
                x=rect_x, y=rect_y, z=rect_z,
                mode='lines',
                line=dict(color=colors[2], width=2),
                opacity=0.6,
                showlegend=False
            ))
    
    # Layout
    fig.update_layout(
        title=f"3D Golden Spiral: φ = {phi_adjusted:.6f}",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y", 
            zaxis_title="Height",
            bgcolor="rgba(0,0,0,0)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=600
    )
    
    return fig

def create_fibonacci_phyllotaxis():
    """Create 3D Fibonacci phyllotaxis pattern"""
    n_points = resolution * 2
    indices = np.arange(0, n_points, dtype=float) + 0.5
    
    # Golden angle positioning
    angles = indices * GOLDEN_ANGLE * phi_factor
    radii = np.sqrt(indices)
    
    # Convert to 3D coordinates
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    z = indices * 0.02  # Slight height variation
    
    # Color by fibonacci sequence position
    fib_colors = np.mod(indices, PHI)
    
    fig = go.Figure()
    
    # Add points
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=6,
            color=fib_colors,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title="φ-Harmonic Index")
        ),
        text=[f"Point {i+1}<br>Angle: {a:.1f}°<br>Radius: {r:.2f}" 
              for i, (a, r) in enumerate(zip(np.degrees(angles), radii))],
        hovertemplate='%{text}<extra></extra>',
        name="Fibonacci Phyllotaxis"
    ))
    
    # Add connecting spirals
    for start_idx in range(0, n_points, n_points//13):  # Fibonacci number 13
        end_idx = min(start_idx + n_points//13, n_points-1)
        spiral_x = x[start_idx:end_idx]
        spiral_y = y[start_idx:end_idx] 
        spiral_z = z[start_idx:end_idx]
        
        fig.add_trace(go.Scatter3d(
            x=spiral_x, y=spiral_y, z=spiral_z,
            mode='lines',
            line=dict(color=colors[1], width=2),
            opacity=0.5,
            showlegend=False
        ))
    
    fig.update_layout(
        title=f"Fibonacci Phyllotaxis: Golden Angle = {np.degrees(GOLDEN_ANGLE * phi_factor):.1f}°",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Height", 
            bgcolor="rgba(0,0,0,0)",
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.8))
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        height=600
    )
    
    return fig

def create_phi_harmonic_torus():
    """Create φ-harmonic torus visualization"""
    u = np.linspace(0, 2*np.pi, resolution//2)
    v = np.linspace(0, 2*np.pi, resolution//2)
    U, V = np.meshgrid(u, v)
    
    # Torus parameters based on golden ratio
    R = PHI * phi_factor  # Major radius
    r = 1 / PHI  # Minor radius
    
    # Torus equations
    X = (R + r * np.cos(V)) * np.cos(U)
    Y = (R + r * np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)
    
    # φ-harmonic coloring
    color_func = np.sin(PHI * U) * np.cos(PHI * V)
    
    fig = go.Figure()
    
    # Add torus surface
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=color_func,
        colorscale='Viridis',
        opacity=0.8,
        name="φ-Harmonic Torus"
    ))
    
    # Add golden ratio circles
    circle_u = np.linspace(0, 2*np.pi, 100)
    for i in range(8):
        angle = i * np.pi / 4
        circle_x = R * np.cos(angle) + r * np.cos(circle_u) * np.cos(angle)
        circle_y = R * np.sin(angle) + r * np.cos(circle_u) * np.sin(angle) 
        circle_z = r * np.sin(circle_u)
        
        fig.add_trace(go.Scatter3d(
            x=circle_x, y=circle_y, z=circle_z,
            mode='lines',
            line=dict(color=colors[2], width=4),
            showlegend=False
        ))
    
    fig.update_layout(
        title=f"φ-Harmonic Torus: R/r = φ = {R/r:.6f}",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            bgcolor="rgba(0,0,0,0)",
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        height=600
    )
    
    return fig

def create_unity_convergence():
    """Create unity convergence visualization showing 1+1=1"""
    t = np.linspace(0, 4*np.pi, resolution)
    
    # Two spirals that converge to unity
    phi_adj = PHI * phi_factor
    
    # Spiral 1
    r1 = np.exp(-t / phi_adj) + 1
    x1 = r1 * np.cos(t) - 2
    y1 = r1 * np.sin(t)
    z1 = t * 0.1
    
    # Spiral 2 (mirror)
    r2 = np.exp(-t / phi_adj) + 1  
    x2 = r2 * np.cos(-t) + 2
    y2 = r2 * np.sin(-t)
    z2 = t * 0.1
    
    # Unity point (convergence)
    x_unity = np.zeros_like(t)
    y_unity = np.zeros_like(t)
    z_unity = t * 0.1
    
    fig = go.Figure()
    
    # Add first spiral (representing first "1")
    fig.add_trace(go.Scatter3d(
        x=x1, y=y1, z=z1,
        mode='lines+markers',
        line=dict(color=colors[0], width=6),
        marker=dict(size=2, color=colors[0]),
        name='Unity Component 1'
    ))
    
    # Add second spiral (representing second "1")  
    fig.add_trace(go.Scatter3d(
        x=x2, y=y2, z=z2,
        mode='lines+markers',
        line=dict(color=colors[1], width=6),
        marker=dict(size=2, color=colors[1]),
        name='Unity Component 2'
    ))
    
    # Add unity convergence line
    fig.add_trace(go.Scatter3d(
        x=x_unity, y=y_unity, z=z_unity,
        mode='lines+markers',
        line=dict(color=colors[3], width=8),
        marker=dict(size=4, color=colors[3]),
        name='Unity Result (1+1=1)'
    ))
    
    # Add convergence arrows at several points
    for i in range(0, len(t), len(t)//10):
        # Arrow from spiral 1 to unity
        fig.add_trace(go.Scatter3d(
            x=[x1[i], x_unity[i]], y=[y1[i], y_unity[i]], z=[z1[i], z_unity[i]],
            mode='lines',
            line=dict(color=colors[2], width=2, dash='dash'),
            opacity=0.6,
            showlegend=False
        ))
        
        # Arrow from spiral 2 to unity  
        fig.add_trace(go.Scatter3d(
            x=[x2[i], x_unity[i]], y=[y2[i], y_unity[i]], z=[z2[i], z_unity[i]],
            mode='lines',
            line=dict(color=colors[2], width=2, dash='dash'),
            opacity=0.6,
            showlegend=False
        ))
    
    # Add equation annotation
    fig.add_annotation3d(
        x=0, y=0, z=max(z_unity),
        text="1 + 1 = 1<br>φ-Harmonic Unity",
        showarrow=False,
        font=dict(size=16, color=colors[3])
    )
    
    fig.update_layout(
        title="Unity Convergence: Two Become One Through φ-Harmony",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Evolution",
            bgcolor="rgba(0,0,0,0)",
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.2))
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        height=600
    )
    
    return fig

def create_sacred_geometry_matrix():
    """Create sacred geometry matrix with multiple φ-based shapes"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Golden Rectangle', 'Pentagram', 'Nautilus Shell', 'φ-Spiral'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    # 1. Golden Rectangle
    rect_x = [0, PHI, PHI, 0, 0]
    rect_y = [0, 0, 1, 1, 0]
    fig.add_trace(go.Scatter(x=rect_x, y=rect_y, mode='lines+markers',
                            line=dict(color=colors[0], width=3), name='Golden Rectangle'),
                  row=1, col=1)
    
    # 2. Pentagram (φ relationships)
    angles = np.array([0, 72, 144, 216, 288]) * np.pi / 180
    pent_x = np.cos(angles)
    pent_y = np.sin(angles)
    # Connect every second point to create pentagram
    pent_order = [0, 2, 4, 1, 3, 0]
    fig.add_trace(go.Scatter(x=pent_x[pent_order], y=pent_y[pent_order],
                            mode='lines+markers', line=dict(color=colors[1], width=3),
                            name='Pentagram'), row=1, col=2)
    
    # 3. Nautilus Shell approximation
    t_shell = np.linspace(0, 6*np.pi, 200)
    r_shell = np.exp(t_shell / PHI)
    shell_x = r_shell * np.cos(t_shell)
    shell_y = r_shell * np.sin(t_shell)
    fig.add_trace(go.Scatter(x=shell_x, y=shell_y, mode='lines',
                            line=dict(color=colors[2], width=2), name='Nautilus'),
                  row=2, col=1)
    
    # 4. φ-Spiral
    t_spiral = np.linspace(0, 4*np.pi, 100)
    r_spiral = t_spiral / PHI
    spiral_x = r_spiral * np.cos(t_spiral)
    spiral_y = r_spiral * np.sin(t_spiral)
    fig.add_trace(go.Scatter(x=spiral_x, y=spiral_y, mode='lines+markers',
                            line=dict(color=colors[3], width=3), name='φ-Spiral'),
                  row=2, col=2)
    
    fig.update_layout(
        title="Sacred Geometry Matrix: φ-Harmonic Forms",
        height=600,
        showlegend=False
    )
    
    return fig

# Main visualization area
col1, col2 = st.columns([3, 1])

with col1:
    # Generate the selected visualization
    if viz_type == "Golden Spiral 3D":
        fig = create_golden_spiral_3d()
    elif viz_type == "Fibonacci Phyllotaxis":
        fig = create_fibonacci_phyllotaxis()
    elif viz_type == "φ-Harmonic Torus":
        fig = create_phi_harmonic_torus()
    elif viz_type == "Unity Convergence":
        fig = create_unity_convergence()
    elif viz_type == "Sacred Geometry Matrix":
        fig = create_sacred_geometry_matrix()
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 🔢 φ-Mathematics")
    st.markdown(f"""
    <div class="unity-equation">
        φ = (1 + √5) / 2<br>
        ≈ {PHI:.10f}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"**Current φ Factor:** {phi_factor:.3f}")
    st.markdown(f"**Adjusted φ:** {PHI * phi_factor:.6f}")
    st.markdown(f"**Golden Angle:** {np.degrees(GOLDEN_ANGLE * phi_factor):.2f}°")
    
    st.markdown("### 🎨 Visualization Info")
    st.markdown(f"**Type:** {viz_type}")
    st.markdown(f"**Resolution:** {resolution} points")
    st.markdown(f"**Color Scheme:** {color_scheme}")
    
    if viz_type in ["Golden Spiral 3D", "Unity Convergence"]:
        st.markdown(f"**Spiral Turns:** {spiral_turns}")
    
    st.markdown("### 📐 Mathematical Properties")
    st.markdown(f"φ² = φ + 1 = {PHI**2:.6f}")
    st.markdown(f"1/φ = φ - 1 = {1/PHI:.6f}")
    st.markdown(f"φ - 1/φ = 1")

# Educational content
st.markdown("---")
st.markdown("## 🎓 Educational Context")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### The Golden Ratio φ
    The golden ratio φ = (1 + √5) / 2 ≈ 1.618 is a fundamental mathematical constant
    appearing throughout nature, art, and architecture. It represents the most
    aesthetically pleasing proportion to the human eye and embodies the principle
    of unity through divine proportion.
    """)

with col2:
    st.markdown("""
    ### Unity Through φ-Harmony
    In φ-harmonic mathematics, the equation 1 + 1 = 1 manifests through the
    convergence of dual elements into unified wholes. The golden ratio serves
    as the organizing principle that creates coherence from apparent multiplicity.
    """)

with col3:
    st.markdown("""
    ### Sacred Geometry
    φ appears in pentagrams, nautilus shells, galaxy spirals, and human proportions.
    This universality suggests that the golden ratio is a fundamental organizing
    principle of consciousness and reality itself.
    """)

# Mathematical formulas
st.markdown("### 🧮 Key Mathematical Relationships")
st.latex(r"\phi = \frac{1 + \sqrt{5}}{2} \approx 1.618033988749895")
st.latex(r"\phi^2 = \phi + 1")
st.latex(r"\frac{1}{\phi} = \phi - 1")
st.latex(r"\text{Golden Angle} = \frac{2\pi}{\phi} \approx 137.5°")

st.markdown("""
### 🌟 Philosophical Significance
The golden ratio represents the mathematical bridge between Unity and Duality.
In the equation 1 + 1 = 1, φ serves as the harmonic mediator that allows
two elements to maintain their individual identity while participating
in a greater unified whole. This is the essence of φ-harmonic consciousness.
""")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.9em;">
    <p>🌟 φ-Harmonic 3D Explorer | Een Unity Mathematics Framework</p>
    <p>Explore the infinite beauty of mathematical consciousness through interactive φ-harmonic visualizations</p>
</div>
""", unsafe_allow_html=True)