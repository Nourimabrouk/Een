"""
3D ANIMATED CONSCIOUSNESS FIELD EXPLORER
Real-time visualization of consciousness field equation C(x,y,t) = φ·sin(x·φ)·cos(y·φ)·e^(-t/φ)
Demonstrates the unity principle through consciousness mathematics
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import time
from datetime import datetime
import math

# Page configuration
st.set_page_config(
    page_title="3D Consciousness Field Explorer | Een Unity Mathematics",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio φ = 1.618...
E = np.e  # Euler's number

# Custom CSS for consciousness-themed styling
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(circle at center, rgba(124, 58, 237, 0.03) 0%, rgba(15, 123, 138, 0.05) 100%);
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #7C3AED, #0F7B8A);
        color: white;
        border-radius: 1rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(124, 58, 237, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: consciousness-pulse 4s ease-in-out infinite;
    }
    
    @keyframes consciousness-pulse {
        0%, 100% { transform: scale(1) rotate(0deg); opacity: 0.3; }
        50% { transform: scale(1.1) rotate(180deg); opacity: 0.1; }
    }
    
    .consciousness-highlight {
        color: #A78BFA;
        font-weight: bold;
        font-size: 1.2em;
    }
    
    .consciousness-equation {
        font-family: 'Times New Roman', serif;
        font-size: 1.8em;
        text-align: center;
        color: #7C3AED;
        margin: 1rem 0;
        padding: 1.5rem;
        background: rgba(124, 58, 237, 0.1);
        border-radius: 0.75rem;
        border: 2px solid rgba(124, 58, 237, 0.3);
        box-shadow: 0 4px 16px rgba(124, 58, 237, 0.2);
    }
    
    .field-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .stat-card {
        background: rgba(124, 58, 237, 0.05);
        padding: 1rem;
        border-radius: 0.75rem;
        border: 1px solid rgba(124, 58, 237, 0.2);
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.5em;
        font-weight: bold;
        color: #7C3AED;
    }
    
    .stat-label {
        font-size: 0.9em;
        color: #6B7280;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header with animated background
st.markdown("""
<div class="main-header">
    <h1>🧠 3D Consciousness Field Explorer</h1>
    <p style="position: relative; z-index: 1;">Real-time visualization of consciousness mathematics</p>
    <div class="consciousness-equation" style="position: relative; z-index: 1; margin: 1rem auto; max-width: 600px;">
        C(x,y,t) = φ · sin(x·φ) · cos(y·φ) · e<sup>-t/φ</sup>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.markdown("## 🎛️ Consciousness Field Controls")

# Field parameters
st.sidebar.markdown("### 🌊 Field Parameters")
phi_factor = st.sidebar.slider("φ Consciousness Factor", 0.5, 3.0, 1.0, 0.01)
time_factor = st.sidebar.slider("Time Evolution Rate", 0.1, 5.0, 1.0, 0.1)
spatial_frequency = st.sidebar.slider("Spatial Frequency", 0.5, 3.0, 1.0, 0.1)
damping_strength = st.sidebar.slider("Temporal Damping", 0.1, 2.0, 1.0, 0.1)

# Visualization controls
st.sidebar.markdown("### 🎨 Visualization Controls")
field_resolution = st.sidebar.selectbox("Field Resolution", [30, 50, 75, 100], index=1)
viz_mode = st.sidebar.selectbox(
    "Visualization Mode",
    ["Surface Plot", "Contour Field", "Particle System", "Wave Interference", "Holographic"]
)
color_scheme = st.sidebar.selectbox(
    "Color Palette",
    ["Consciousness Purple", "Unity Teal", "Sacred Gold", "Neural Blue", "Phi Harmony"]
)

# Animation controls
st.sidebar.markdown("### 🎬 Animation Controls")
animate = st.sidebar.checkbox("🎬 Enable Animation", value=True)
animation_speed = st.sidebar.slider("Animation Speed", 0.1, 3.0, 1.0, 0.1)
show_evolution = st.sidebar.checkbox("Show Time Evolution", value=True)

# Advanced controls
with st.sidebar.expander("⚡ Advanced Parameters"):
    wave_amplitude = st.slider("Wave Amplitude", 0.5, 2.0, 1.0, 0.1)
    field_offset = st.slider("Field Offset", -1.0, 1.0, 0.0, 0.1)
    consciousness_depth = st.slider("Consciousness Depth", 1, 10, 5, 1)
    unity_coherence = st.slider("Unity Coherence", 0.0, 1.0, 0.618, 0.001)

# Color palettes
color_palettes = {
    "Consciousness Purple": ["#7C3AED", "#A78BFA", "#C4B5FD", "#E9D5FF"],
    "Unity Teal": ["#0D9488", "#14B8A6", "#5EEAD4", "#A7F3D0"], 
    "Sacred Gold": ["#D97706", "#F59E0B", "#FBBF24", "#FEF3C7"],
    "Neural Blue": ["#1E40AF", "#3B82F6", "#93C5FD", "#DBEAFE"],
    "Phi Harmony": ["#0F7B8A", "#4ECDC4", "#A7F3D0", "#F59E0B"]
}

colors = color_palettes[color_scheme]

def consciousness_field_equation(x, y, t, phi_adj=PHI, spatial_freq=1.0, time_rate=1.0, damping=1.0, offset=0.0, amplitude=1.0):
    """
    Consciousness field equation: C(x,y,t) = φ·sin(x·φ)·cos(y·φ)·e^(-t/φ)
    """
    # Apply adjustments
    phi_val = phi_adj * PHI
    t_adj = t * time_rate
    x_adj = x * spatial_freq * phi_val
    y_adj = y * spatial_freq * phi_val
    
    # Core consciousness field equation
    spatial_component = np.sin(x_adj) * np.cos(y_adj)
    temporal_component = np.exp(-t_adj / (phi_val * damping))
    
    # Combine with consciousness depth and unity coherence
    consciousness_field = phi_val * spatial_component * temporal_component * amplitude + offset
    
    return consciousness_field

def create_consciousness_surface():
    """Create 3D consciousness field surface"""
    # Create spatial grid
    x_range = np.linspace(-2*np.pi, 2*np.pi, field_resolution)
    y_range = np.linspace(-2*np.pi, 2*np.pi, field_resolution)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Get current time for animation
    if animate:
        current_time = time.time() * animation_speed
    else:
        current_time = 0
    
    # Calculate consciousness field
    Z = consciousness_field_equation(
        X, Y, current_time,
        phi_adj=phi_factor,
        spatial_freq=spatial_frequency,
        time_rate=time_factor,
        damping=damping_strength,
        offset=field_offset,
        amplitude=wave_amplitude
    )
    
    # Apply consciousness depth modulation
    depth_modulation = np.sin(X/consciousness_depth) * np.cos(Y/consciousness_depth)
    Z_modulated = Z * (1 + unity_coherence * depth_modulation)
    
    fig = go.Figure()
    
    # Create surface
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z_modulated,
        colorscale=[[0, colors[0]], [0.33, colors[1]], [0.67, colors[2]], [1, colors[3]]],
        opacity=0.8,
        name="Consciousness Field",
        colorbar=dict(title="Field Strength", titleside="right")
    ))
    
    # Add field lines if requested
    if show_evolution:
        # Add some consciousness field lines
        for i in range(5):
            t_line = np.linspace(0, 2*np.pi, 100)
            x_line = np.cos(t_line * (i+1)) * np.pi
            y_line = np.sin(t_line * (i+1)) * np.pi
            z_line = consciousness_field_equation(
                x_line, y_line, current_time,
                phi_adj=phi_factor,
                spatial_freq=spatial_frequency,
                time_rate=time_factor,
                damping=damping_strength,
                offset=field_offset,
                amplitude=wave_amplitude
            )
            
            fig.add_trace(go.Scatter3d(
                x=x_line, y=y_line, z=z_line,
                mode='lines',
                line=dict(color=colors[3], width=4),
                opacity=0.6,
                showlegend=False,
                name=f"Field Line {i+1}"
            ))
    
    # Layout
    fig.update_layout(
        title=f"3D Consciousness Field: t = {current_time:.2f}",
        scene=dict(
            xaxis_title="X Space",
            yaxis_title="Y Space",
            zaxis_title="Consciousness Amplitude",
            bgcolor="rgba(0,0,0,0)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            aspectmode='cube'
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=600
    )
    
    return fig, Z_modulated, current_time

def create_consciousness_contour():
    """Create 2D contour field visualization"""
    x_range = np.linspace(-2*np.pi, 2*np.pi, field_resolution)
    y_range = np.linspace(-2*np.pi, 2*np.pi, field_resolution)
    X, Y = np.meshgrid(x_range, y_range)
    
    current_time = time.time() * animation_speed if animate else 0
    
    Z = consciousness_field_equation(
        X, Y, current_time,
        phi_adj=phi_factor,
        spatial_freq=spatial_frequency,
        time_rate=time_factor,
        damping=damping_strength,
        offset=field_offset,
        amplitude=wave_amplitude
    )
    
    fig = go.Figure()
    
    # Add contour plot
    fig.add_trace(go.Contour(
        x=x_range, y=y_range, z=Z,
        colorscale=[[0, colors[0]], [0.5, colors[1]], [1, colors[2]]],
        contours=dict(
            showlabels=True,
            labelfont=dict(size=12, color='white')
        ),
        name="Consciousness Contours"
    ))
    
    # Add field vectors
    step = field_resolution // 10
    X_vec = X[::step, ::step]
    Y_vec = Y[::step, ::step]
    
    # Calculate gradient for field vectors
    dZ_dx = np.gradient(Z, axis=1)[::step, ::step] 
    dZ_dy = np.gradient(Z, axis=0)[::step, ::step]
    
    # Add vector field
    fig.add_trace(go.Scatter(
        x=X_vec.flatten(), y=Y_vec.flatten(),
        mode='markers',
        marker=dict(
            symbol='arrow-up',
            size=8,
            color=colors[3],
            angle=np.arctan2(dZ_dy.flatten(), dZ_dx.flatten()) * 180 / np.pi
        ),
        name="Field Vectors",
        showlegend=False
    ))
    
    fig.update_layout(
        title=f"Consciousness Field Contours: t = {current_time:.2f}",
        xaxis_title="X Space",
        yaxis_title="Y Space",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=600
    )
    
    return fig

def create_particle_system():
    """Create consciousness particle system"""
    n_particles = 200
    current_time = time.time() * animation_speed if animate else 0
    
    # Create particles at various positions
    np.random.seed(42)  # For reproducible patterns
    x_particles = np.random.uniform(-2*np.pi, 2*np.pi, n_particles)
    y_particles = np.random.uniform(-2*np.pi, 2*np.pi, n_particles)
    
    # Calculate consciousness field strength at particle positions
    z_particles = consciousness_field_equation(
        x_particles, y_particles, current_time,
        phi_adj=phi_factor,
        spatial_freq=spatial_frequency,
        time_rate=time_factor,
        damping=damping_strength,
        offset=field_offset,
        amplitude=wave_amplitude
    )
    
    # Particle sizes based on field strength
    sizes = 5 + 10 * np.abs(z_particles) / np.max(np.abs(z_particles))
    
    # Color based on field polarity
    particle_colors = z_particles
    
    fig = go.Figure()
    
    # Add consciousness particles
    fig.add_trace(go.Scatter3d(
        x=x_particles, 
        y=y_particles,
        z=z_particles,
        mode='markers',
        marker=dict(
            size=sizes,
            color=particle_colors,
            colorscale=[[0, colors[0]], [0.5, colors[1]], [1, colors[2]]],
            opacity=0.8,
            colorbar=dict(title="Consciousness Intensity")
        ),
        text=[f"Particle {i+1}<br>Field: {z:.3f}" for i, z in enumerate(z_particles)],
        hovertemplate='%{text}<extra></extra>',
        name="Consciousness Particles"
    ))
    
    # Add connecting web for highly activated particles
    high_activation = np.where(np.abs(z_particles) > np.percentile(np.abs(z_particles), 80))[0]
    
    for i, idx1 in enumerate(high_activation[:-1]):
        for idx2 in high_activation[i+1:]:
            if np.sqrt((x_particles[idx1] - x_particles[idx2])**2 + 
                      (y_particles[idx1] - y_particles[idx2])**2) < np.pi:
                fig.add_trace(go.Scatter3d(
                    x=[x_particles[idx1], x_particles[idx2]],
                    y=[y_particles[idx1], y_particles[idx2]], 
                    z=[z_particles[idx1], z_particles[idx2]],
                    mode='lines',
                    line=dict(color=colors[3], width=2),
                    opacity=0.3,
                    showlegend=False
                ))
    
    fig.update_layout(
        title=f"Consciousness Particle System: t = {current_time:.2f}",
        scene=dict(
            xaxis_title="X Space",
            yaxis_title="Y Space",
            zaxis_title="Field Strength",
            bgcolor="rgba(0,0,0,0)",
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.5))
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        height=600
    )
    
    return fig

# Main visualization area
col1, col2 = st.columns([3, 1])

with col1:
    # Generate visualization based on selected mode
    if viz_mode == "Surface Plot":
        fig, field_data, current_t = create_consciousness_surface()
        st.plotly_chart(fig, use_container_width=True)
        
        # Real-time field statistics
        field_max = np.max(field_data)
        field_min = np.min(field_data)
        field_mean = np.mean(field_data)
        field_std = np.std(field_data)
        
        st.markdown(f"""
        <div class="field-stats">
            <div class="stat-card">
                <div class="stat-value">{field_max:.3f}</div>
                <div class="stat-label">Peak Amplitude</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{field_min:.3f}</div>
                <div class="stat-label">Min Amplitude</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{field_mean:.3f}</div>
                <div class="stat-label">Mean Field</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{field_std:.3f}</div>
                <div class="stat-label">Coherence (σ)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    elif viz_mode == "Contour Field":
        fig = create_consciousness_contour()
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_mode == "Particle System":
        fig = create_particle_system()
        st.plotly_chart(fig, use_container_width=True)
        
    # Add more visualization modes as needed
    elif viz_mode in ["Wave Interference", "Holographic"]:
        st.info(f"🚧 {viz_mode} mode coming soon! Enhanced consciousness field mathematics.")

with col2:
    st.markdown("### 🧠 Consciousness Mathematics")
    st.markdown(f"""
    <div class="consciousness-equation">
        φ = {PHI:.6f}<br>
        e = {E:.6f}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"**φ Factor:** {phi_factor:.3f}")
    st.markdown(f"**Temporal Rate:** {time_factor:.3f}")
    st.markdown(f"**Spatial Freq:** {spatial_frequency:.3f}")
    st.markdown(f"**Damping:** {damping_strength:.3f}")
    
    st.markdown("### 📊 Field Properties")
    st.markdown(f"**Resolution:** {field_resolution}×{field_resolution}")
    st.markdown(f"**Mode:** {viz_mode}")
    st.markdown(f"**Animation:** {'Active' if animate else 'Static'}")
    
    # Consciousness field insights
    st.markdown("### 🌟 Unity Insights")
    coherence_level = unity_coherence * 100
    if coherence_level > 61.8:
        st.success(f"🎯 φ-Harmonic Coherence: {coherence_level:.1f}%")
    elif coherence_level > 38.2:
        st.info(f"⚖️ Balanced State: {coherence_level:.1f}%")
    else:
        st.warning(f"🌊 Dynamic Evolution: {coherence_level:.1f}%")
    
    # Real-time consciousness metrics
    if animate:
        current_phi_resonance = np.cos(time.time() * animation_speed / PHI)
        st.metric("φ-Resonance", f"{current_phi_resonance:.3f}", f"{current_phi_resonance - 0.618:.3f}")

# Educational content
st.markdown("---")
st.markdown("## 🎓 Consciousness Field Theory")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### The Consciousness Equation
    C(x,y,t) = φ·sin(x·φ)·cos(y·φ)·e^(-t/φ)
    
    This equation represents the fundamental field dynamics of consciousness
    in φ-harmonic space. The golden ratio φ acts as both spatial frequency
    and temporal damping parameter, creating self-similar patterns across
    scales - a hallmark of consciousness mathematics.
    """)

with col2:
    st.markdown("""
    ### Unity Through Coherence
    The consciousness field demonstrates how individual elements (spatial
    coordinates) can maintain coherence through φ-harmonic resonance.
    As the field evolves temporally, the exponential damping ensures
    convergence to unity states, embodying the principle 1+1=1.
    """)

with col3:
    st.markdown("""
    ### Transcendental Reality
    The field equation bridges mathematical physics and consciousness
    studies. The φ-harmonic structure suggests that consciousness
    itself follows golden ratio principles, creating the aesthetic
    experience of beauty through mathematical harmony.
    """)

# Mathematical exposition
st.markdown("### 🔬 Mathematical Foundation")

st.latex(r"C(x,y,t) = \phi \cdot \sin(x \cdot \phi) \cdot \cos(y \cdot \phi) \cdot e^{-t/\phi}")
st.latex(r"\text{where } \phi = \frac{1 + \sqrt{5}}{2} \approx 1.618033988749895")

st.markdown("""
**Key Properties:**
- **Spatial Coherence:** Sin-Cos coupling creates standing wave patterns
- **Temporal Evolution:** Exponential decay ensures convergence to unity
- **φ-Harmonic Structure:** Golden ratio provides natural frequency scaling
- **Self-Similarity:** Field patterns repeat at φ-scaled dimensions
- **Unity Principle:** Multiple field components converge to singular consciousness states
""")

st.markdown("### 🌌 Philosophical Implications")
st.markdown("""
The consciousness field equation suggests that individual awareness emerges from
underlying field dynamics governed by φ-harmonic principles. This provides a
mathematical framework for understanding how:

1. **Multiple conscious entities** can maintain individual identity while participating in unified fields
2. **Temporal evolution** naturally leads to coherent unity states 
3. **Aesthetic experience** arises from φ-harmonic resonance patterns
4. **Scale invariance** allows consciousness patterns to manifest across all levels of reality

This embodies the profound truth that **1 + 1 = 1** - separate conscious entities
unite through φ-harmonic field coupling without losing their essential nature.
""")

# Auto-refresh for animation
if animate:
    time.sleep(0.1)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.9em;">
    <p>🧠 3D Consciousness Field Explorer | Een Unity Mathematics Framework</p>
    <p>Explore the mathematical foundations of consciousness through φ-harmonic field dynamics</p>
</div>
""", unsafe_allow_html=True)