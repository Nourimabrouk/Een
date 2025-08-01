"""
Plotly Helper Functions for Unity Visualizations
Modern, reusable figure builders following best practices
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import math
import cmath
from typing import List, Tuple, Dict, Any, Optional
import streamlit as st

# Unity Constants
PHI = 1.618033988749895  # Golden ratio
PI = math.pi
E = math.e
UNITY_FREQUENCY = 528.0  # Hz - love frequency

# Color palettes for unity themes
UNITY_COLORS = {
    'dark': {
        'background': '#0a0a0a',
        'paper': '#1a1a1a', 
        'text': '#ffffff',
        'primary': '#00d4ff',
        'secondary': '#ff6b9d',
        'unity': '#ffd700',
        'consciousness': '#9d4edd',
        'love': '#ff4081',
        'grid': 'rgba(255,255,255,0.1)'
    },
    'light': {
        'background': '#ffffff',
        'paper': '#f8f9fa',
        'text': '#000000', 
        'primary': '#0066cc',
        'secondary': '#cc0066',
        'unity': '#cc9900',
        'consciousness': '#6622cc',
        'love': '#cc2244',
        'grid': 'rgba(0,0,0,0.1)'
    }
}

@st.cache_data
def get_theme_colors(theme: str = 'dark') -> Dict[str, str]:
    """Get color palette for specified theme"""
    return UNITY_COLORS.get(theme, UNITY_COLORS['dark'])

def apply_unity_theme(fig: go.Figure, theme: str = 'dark', title: str = None) -> go.Figure:
    """Apply consistent Unity theme to any Plotly figure"""
    colors = get_theme_colors(theme)
    
    fig.update_layout(
        title=title,
        paper_bgcolor=colors['paper'],
        plot_bgcolor=colors['background'],
        font=dict(color=colors['text'], family="JetBrains Mono, monospace"),
        title_font=dict(size=20, color=colors['unity']),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)' if theme == 'dark' else 'rgba(255,255,255,0.8)',
            bordercolor=colors['grid'],
            borderwidth=1
        ),
        xaxis=dict(
            gridcolor=colors['grid'],
            zerolinecolor=colors['grid'],
            color=colors['text']
        ),
        yaxis=dict(
            gridcolor=colors['grid'],
            zerolinecolor=colors['grid'],
            color=colors['text']
        )
    )
    
    return fig

@st.cache_data
def create_golden_spiral(points: int = 377, theme: str = 'dark') -> go.Figure:
    """Create golden spiral demonstrating φ-harmonic unity convergence"""
    colors = get_theme_colors(theme)
    
    # Generate golden spiral points
    theta = np.linspace(0, 4*PI, points)
    r = PHI ** (theta / (2*PI))
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Unity intensity (approaches 1)
    unity_intensity = 1 / (1 + np.exp(-theta/2))
    
    fig = go.Figure()
    
    # Main spiral
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines+markers',
        line=dict(color=colors['unity'], width=3),
        marker=dict(
            size=4,
            color=unity_intensity,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Unity Intensity", tickfont=dict(color=colors['text']))
        ),
        name='Golden Spiral (φ = 1.618...)',
        hovertemplate='<b>φ-Harmonic Point</b><br>x: %{x:.3f}<br>y: %{y:.3f}<br>Unity: %{marker.color:.3f}<extra></extra>'
    ))
    
    # Add unity convergence annotation
    fig.add_annotation(
        x=x[-1], y=y[-1],
        text="1+1=1<br>Unity Achieved",
        showarrow=True,
        arrowhead=2,
        arrowcolor=colors['unity'],
        font=dict(color=colors['unity'], size=14)
    )
    
    fig = apply_unity_theme(fig, theme, "Golden Spiral Unity Convergence: φ-Harmonic 1+1=1")
    
    return fig

@st.cache_data  
def create_consciousness_field(resolution: int = 100, time_step: float = 0.0, theme: str = 'dark') -> go.Figure:
    """Create consciousness field C(x,y,t) = φ * sin(x*φ) * cos(y*φ) * e^(-t/φ)"""
    colors = get_theme_colors(theme)
    
    # Generate consciousness field
    x = np.linspace(-2*PHI, 2*PHI, resolution)
    y = np.linspace(-2*PHI, 2*PHI, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Consciousness field equation
    C = PHI * np.sin(X * PHI) * np.cos(Y * PHI) * np.exp(-time_step / PHI)
    
    fig = go.Figure(data=go.Heatmap(
        z=C,
        x=x,
        y=y,
        colorscale='Plasma',
        showscale=True,
        colorbar=dict(
            title="Consciousness<br>Intensity",
            tickfont=dict(color=colors['text'])
        ),
        hovertemplate='<b>Consciousness Field</b><br>x: %{x:.3f}<br>y: %{y:.3f}<br>C(x,y,t): %{z:.3f}<extra></extra>'
    ))
    
    # Add unity contour at C=1
    fig.add_contour(
        z=C, x=x, y=y,
        contours=dict(
            start=1, end=1, size=0,
            coloring='lines',
            showlabels=True
        ),
        line=dict(color=colors['unity'], width=3),
        name="Unity Contour (C=1)"
    )
    
    fig = apply_unity_theme(fig, theme, f"Consciousness Field Evolution: C(x,y,t={time_step:.2f})")
    
    return fig

@st.cache_data
def create_quantum_superposition(states: int = 144, theme: str = 'dark') -> go.Figure:
    """Create quantum superposition |1⟩ + |1⟩ → |1⟩ visualization"""
    colors = get_theme_colors(theme)
    
    # Phase evolution
    phase = np.linspace(0, 2*PI, states)
    
    # Two quantum states
    state_1 = np.exp(1j * phase)
    state_2 = np.exp(1j * (phase + PI/PHI))  # φ-shifted
    
    # Superposition
    psi = (state_1 + state_2) / np.sqrt(2)
    
    # Collapse to unity
    collapse_factor = 1 / (1 + np.exp(-np.arange(states) / states * 10))
    unity_state = psi * collapse_factor + (1 - collapse_factor)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Real Part', 'Imaginary Part', 'Magnitude', 'Phase'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Real part
    fig.add_trace(go.Scatter(
        x=phase, y=unity_state.real,
        mode='lines',
        line=dict(color=colors['primary'], width=2),
        name='Re(ψ)'
    ), row=1, col=1)
    
    # Imaginary part  
    fig.add_trace(go.Scatter(
        x=phase, y=unity_state.imag,
        mode='lines',
        line=dict(color=colors['secondary'], width=2),
        name='Im(ψ)'
    ), row=1, col=2)
    
    # Magnitude
    fig.add_trace(go.Scatter(
        x=phase, y=np.abs(unity_state),
        mode='lines',
        line=dict(color=colors['unity'], width=3),
        name='|ψ| → 1'
    ), row=2, col=1)
    
    # Phase
    fig.add_trace(go.Scatter(
        x=phase, y=np.angle(unity_state),
        mode='lines',
        line=dict(color=colors['consciousness'], width=2),
        name='arg(ψ)'
    ), row=2, col=2)
    
    fig.update_layout(
        title="Quantum Superposition Collapse: |1⟩ + |1⟩ → |1⟩",
        height=600
    )
    
    fig = apply_unity_theme(fig, theme)
    
    return fig

@st.cache_data
def create_fractal_unity(iterations: int = 100, zoom: float = 1.0, theme: str = 'dark') -> go.Figure:
    """Create fractal unity manifold (Julia set variant)"""
    colors = get_theme_colors(theme)
    
    # Create complex plane
    width, height = 400, 400
    x = np.linspace(-2/zoom, 2/zoom, width)
    y = np.linspace(-2/zoom, 2/zoom, height)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y
    
    # Unity-themed Julia set parameter
    c = -0.618 + 0.618j  # φ-based complex number
    
    # Fractal calculation
    M = np.zeros_like(Z, dtype=int)
    for i in range(iterations):
        mask = np.abs(Z) < 2
        Z[mask] = Z[mask]**2 + c
        M[mask] = i
    
    # Normalize to unity
    M_normalized = M / np.max(M)
    
    fig = go.Figure(data=go.Heatmap(
        z=M_normalized,
        colorscale='Hot',
        showscale=True,
        colorbar=dict(
            title="Fractal<br>Unity Depth",
            tickfont=dict(color=colors['text'])
        ),
        hovertemplate='<b>Fractal Unity Point</b><br>Depth: %{z:.3f}<extra></extra>'
    ))
    
    fig = apply_unity_theme(fig, theme, f"Fractal Unity Manifold: Self-Similar 1+1=1 (Zoom: {zoom:.1f}x)")
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    
    return fig

@st.cache_data
def create_harmonic_resonance(frequency: float = UNITY_FREQUENCY, duration: float = 1.0, theme: str = 'dark') -> go.Figure:
    """Create harmonic resonance visualization showing wave unity"""
    colors = get_theme_colors(theme)
    
    # Time vector
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Two identical waves
    wave1 = np.sin(2 * PI * frequency * t)
    wave2 = np.sin(2 * PI * frequency * t)
    
    # Combined wave (unity through resonance)
    unity_wave = (wave1 + wave2) / 2  # Normalized unity
    
    # Sample for visualization (first 1000 points)
    sample_size = min(1000, len(t))
    t_sample = t[:sample_size]
    
    fig = go.Figure()
    
    # Individual waves
    fig.add_trace(go.Scatter(
        x=t_sample, y=wave1[:sample_size],
        mode='lines',
        line=dict(color=colors['primary'], width=1, dash='dot'),
        name=f'Wave 1 ({frequency:.0f} Hz)',
        opacity=0.7
    ))
    
    fig.add_trace(go.Scatter(
        x=t_sample, y=wave2[:sample_size],
        mode='lines', 
        line=dict(color=colors['secondary'], width=1, dash='dot'),
        name=f'Wave 2 ({frequency:.0f} Hz)',
        opacity=0.7
    ))
    
    # Unity wave
    fig.add_trace(go.Scatter(
        x=t_sample, y=unity_wave[:sample_size],
        mode='lines',
        line=dict(color=colors['unity'], width=3),
        name='Unity Wave (1+1=1)',
        fill='tonexty' if len(fig.data) > 0 else None,
        fillcolor=f'rgba{tuple(list(int(colors["unity"][i:i+2], 16) for i in (1, 3, 5)) + [0.3])}'
    ))
    
    fig.update_xaxes(title="Time (s)")
    fig.update_yaxes(title="Amplitude")
    
    fig = apply_unity_theme(fig, theme, f"Harmonic Unity Resonance: {frequency:.0f} Hz Love Frequency")
    
    return fig

@st.cache_data
def create_topological_unity(resolution: int = 50, theme: str = 'dark') -> go.Figure:
    """Create Möbius strip demonstrating topological unity"""
    colors = get_theme_colors(theme)
    
    # Möbius strip parametrization
    u = np.linspace(0, 2*PI, resolution)
    v = np.linspace(-0.5, 0.5, resolution//4)
    
    U, V = np.meshgrid(u, v)
    
    # Möbius transformation
    X = (1 + V * np.cos(U/2)) * np.cos(U)
    Y = (1 + V * np.cos(U/2)) * np.sin(U)  
    Z = V * np.sin(U/2)
    
    fig = go.Figure(data=[
        go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title="Surface<br>Unity",
                tickfont=dict(color=colors['text'])
            ),
            hovertemplate='<b>Möbius Surface</b><br>Two sides become one<br>1 + 1 = 1<extra></extra>',
            opacity=0.8
        )
    ])
    
    # Add unity path (center line)
    center_u = np.linspace(0, 2*PI, 100)
    center_x = np.cos(center_u)
    center_y = np.sin(center_u)
    center_z = np.zeros_like(center_u)
    
    fig.add_trace(go.Scatter3d(
        x=center_x, y=center_y, z=center_z,
        mode='lines',
        line=dict(color=colors['unity'], width=8),
        name='Unity Path'
    ))
    
    fig.update_layout(
        title="Topological Unity: Möbius Strip (Two Sides → One Surface)",
        scene=dict(
            bgcolor=colors['background'],
            xaxis=dict(visible=False),
            yaxis=dict(visible=False), 
            zaxis=dict(visible=False),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        )
    )
    
    fig = apply_unity_theme(fig, theme)
    
    return fig

@st.cache_data
def create_unity_metrics_dashboard(
    transcendence_level: float,
    consciousness_intensity: float,
    phi_alignment: float,
    theme: str = 'dark'
) -> go.Figure:
    """Create unity metrics dashboard with gauges and indicators"""
    colors = get_theme_colors(theme)
    
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "indicator"}, {"type": "indicator"}],
            [{"type": "indicator"}, {"type": "scatter"}]
        ],
        subplot_titles=('Transcendence Level', 'Consciousness Intensity', 'φ-Harmonic Alignment', 'Unity Convergence')
    )
    
    # Transcendence gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=transcendence_level,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Transcendence"},
        delta={'reference': 0.618},  # Golden ratio reference
        gauge={
            'axis': {'range': [None, 1]},
            'bar': {'color': colors['unity']},
            'steps': [
                {'range': [0, 0.618], 'color': colors['background']},
                {'range': [0.618, 1], 'color': colors['consciousness']}
            ],
            'threshold': {
                'line': {'color': colors['unity'], 'width': 4},
                'thickness': 0.75,
                'value': 0.9
            }
        }
    ), row=1, col=1)
    
    # Consciousness gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=consciousness_intensity,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Consciousness"},
        gauge={
            'axis': {'range': [None, 1]},
            'bar': {'color': colors['consciousness']},
            'steps': [{'range': [0, 1], 'color': colors['background']}]
        }
    ), row=1, col=2)
    
    # φ-Alignment gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=phi_alignment,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "φ-Alignment"},
        gauge={
            'axis': {'range': [None, PHI]},
            'bar': {'color': colors['primary']},
            'steps': [{'range': [0, PHI], 'color': colors['background']}]
        }
    ), row=2, col=1)
    
    # Unity convergence plot
    convergence_steps = np.linspace(0, 10, 100)
    unity_convergence = 1 - np.exp(-convergence_steps/3)
    
    fig.add_trace(go.Scatter(
        x=convergence_steps,
        y=unity_convergence,
        mode='lines',
        line=dict(color=colors['unity'], width=3),
        name='Unity Convergence'
    ), row=2, col=2)
    
    fig.update_layout(
        title="Unity Mathematics Dashboard: Real-time 1+1=1 Metrics",
        height=600
    )
    
    fig = apply_unity_theme(fig, theme)
    
    return fig

def create_interactive_proof_selector(available_proofs: List[str], theme: str = 'dark') -> Dict[str, Any]:
    """Create interactive proof domain selector configuration"""
    colors = get_theme_colors(theme)
    
    proof_configs = {
        'quantum': {
            'name': '🌊 Quantum Interference',
            'description': 'Wave superposition demonstrates |1⟩ + |1⟩ → |1⟩',
            'color': colors['primary'],
            'function': create_quantum_superposition
        },
        'fractal': {
            'name': '🌀 Fractal Unity',
            'description': 'Self-similar patterns at all scales showing unity',
            'color': colors['consciousness'],
            'function': create_fractal_unity
        },
        'harmonic': {
            'name': '🎵 Harmonic Resonance',
            'description': 'Musical waves combine into perfect unity',
            'color': colors['unity'],
            'function': create_harmonic_resonance
        },
        'topological': {
            'name': '🔄 Topological Unity',
            'description': 'Möbius strip: two sides become one surface',
            'color': colors['secondary'],
            'function': create_topological_unity
        },
        'consciousness': {
            'name': '🧠 Consciousness Fields',
            'description': 'Mathematical consciousness field equations',
            'color': colors['love'],
            'function': create_consciousness_field
        },
        'golden': {
            'name': '🌟 Golden Spiral',
            'description': 'φ-harmonic convergence to unity',
            'color': colors['unity'],
            'function': create_golden_spiral
        }
    }
    
    return {k: v for k, v in proof_configs.items() if k in available_proofs}