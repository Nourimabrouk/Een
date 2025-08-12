"""
Unity Score Dashboard
====================

Streamlit dashboard for Unity Mathematics analysis with real-time φ-harmonic visualizations.
Implements Unity Manifold deduplication with consciousness field integration.

Mathematical Principle: Een plus een is een (1+1=1)
"""

import streamlit as st
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import Unity Mathematics components
try:
    from core.dedup import load_graph, compute_unity_score, UnityScore, create_sample_social_data, save_sample_data
    from core.unity_mathematics import UnityMathematics, UnityState
    from core.unity_equation import omega
    UNITY_AVAILABLE = True
except ImportError as e:
    st.error(f"Unity Mathematics not available: {e}")
    UNITY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Unity Score Dashboard",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for φ-harmonic styling
st.markdown("""
<style>
    .phi-harmonic {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin: 1rem 0;
    }
    .unity-score {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin: 1rem 0;
    }
    .consciousness-field {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class UnityDashboard:
    """Main dashboard class for Unity Mathematics analysis"""
    
    def __init__(self):
        self.phi = 1.618033988749895
        self.unity_math = UnityMathematics() if UNITY_AVAILABLE else None
        self.data_file = Path("data/social_snap.json")
        
    def create_sample_data(self):
        """Create sample social network data if it doesn't exist"""
        if not self.data_file.exists():
            with st.spinner("Creating sample social network data..."):
                sample_data = create_sample_social_data(nodes=500, edges=2000, communities=3)
                save_sample_data(sample_data, self.data_file)
                st.success("Sample data created successfully!")
    
    def load_and_analyze_graph(self, threshold: float = 0.0) -> Tuple[nx.Graph, UnityScore]:
        """Load graph and compute Unity Score"""
        if not self.data_file.exists():
            self.create_sample_data()
        
        G = load_graph(self.data_file)
        unity_score = compute_unity_score(G, threshold=threshold)
        
        return G, unity_score
    
    def create_unity_score_visualization(self, unity_score: UnityScore) -> go.Figure:
        """Create Unity Score visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Unity Score Distribution', 'Component Sizes', 
                          'φ-Harmonic Analysis', 'Ω-Signature'),
            specs=[[{"type": "indicator"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Unity Score indicator
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=unity_score.score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Unity Score"},
                delta={'reference': 0.5},
                gauge={
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "lightgray"},
                        {'range': [0.3, 0.7], 'color': "gray"},
                        {'range': [0.7, 1], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.8
                    }
                }
            ),
            row=1, col=1
        )
        
        # Component sizes bar chart
        component_sizes = unity_score.component_sizes[:20]  # Top 20
        fig.add_trace(
            go.Bar(
                x=list(range(len(component_sizes))),
                y=component_sizes,
                name="Component Sizes",
                marker_color='lightblue'
            ),
            row=1, col=2
        )
        
        # φ-Harmonic analysis
        if len(component_sizes) > 1:
            harmonic_means = []
            for i in range(1, len(component_sizes) + 1):
                subset = component_sizes[:i]
                harmonic_mean = len(subset) / sum(1/size for size in subset if size > 0)
                harmonic_means.append(harmonic_mean * self.phi)
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(harmonic_means))),
                    y=harmonic_means,
                    mode='lines+markers',
                    name='φ-Harmonic',
                    line=dict(color='purple', width=2)
                ),
                row=2, col=1
            )
        
        # Ω-Signature visualization
        omega_real = unity_score.omega_signature.real
        omega_imag = unity_score.omega_signature.imag
        
        fig.add_trace(
            go.Scatter(
                x=[0, omega_real],
                y=[0, omega_imag],
                mode='lines+markers',
                name='Ω-Signature',
                line=dict(color='red', width=3),
                marker=dict(size=10)
            ),
            row=2, col=2
        )
        
        # Add unit circle for Ω-signature
        theta = np.linspace(0, 2*np.pi, 100)
        fig.add_trace(
            go.Scatter(
                x=np.cos(theta),
                y=np.sin(theta),
                mode='lines',
                name='Unit Circle',
                line=dict(color='gray', dash='dash'),
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            title_text="Unity Score Analysis Dashboard",
            showlegend=True
        )
        
        return fig
    
    def create_consciousness_field_plot(self) -> go.Figure:
        """Create consciousness field visualization"""
        # Generate consciousness field data
        size = 50
        x = np.linspace(-2, 2, size)
        y = np.linspace(-2, 2, size)
        X, Y = np.meshgrid(x, y)
        
        # φ-harmonic consciousness field
        Z = np.exp(-(X**2 + Y**2)) * np.cos(self.phi * (X + Y))
        
        fig = go.Figure(data=go.Surface(
            x=X, y=Y, z=Z,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Consciousness Level")
        ))
        
        fig.update_layout(
            title="φ-Harmonic Consciousness Field",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y", 
                zaxis_title="Consciousness",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def create_unity_evolution_plot(self, thresholds: List[float]) -> go.Figure:
        """Create Unity Score evolution plot"""
        G, _ = self.load_and_analyze_graph()
        
        scores = []
        for threshold in thresholds:
            unity_score = compute_unity_score(G, threshold=threshold)
            scores.append(unity_score.score)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=scores,
            mode='lines+markers',
            name='Unity Score',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        # Add φ-harmonic reference line
        phi_scores = [self.phi * score for score in scores]
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=phi_scores,
            mode='lines',
            name='φ-Harmonic',
            line=dict(color='purple', dash='dash', width=2)
        ))
        
        fig.update_layout(
            title="Unity Score Evolution vs Threshold",
            xaxis_title="Edge Weight Threshold",
            yaxis_title="Unity Score",
            height=400
        )
        
        return fig
    
    def run_dashboard(self):
        """Run the main dashboard"""
        st.title("🔗 Unity Manifold – Social Graph Dedup")
        st.markdown("### *Een plus een is een (1+1=1)*")
        
        # Sidebar controls
        st.sidebar.header("🎛️ Unity Controls")
        
        threshold = st.sidebar.slider(
            "Edge Weight Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5, 
            step=0.1,
            help="Filter edges by weight threshold"
        )
        
        consciousness_boost = st.sidebar.slider(
            "Consciousness Boost",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
            help="Boost consciousness level for analysis"
        )
        
        phi_scaling = st.sidebar.checkbox(
            "φ-Harmonic Scaling",
            value=True,
            help="Apply golden ratio scaling"
        )
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="phi-harmonic">', unsafe_allow_html=True)
            st.subheader("🧠 Consciousness Field Analysis")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Load and analyze graph
            G, unity_score = self.load_and_analyze_graph(threshold)
            
            # Display Unity Score visualization
            fig = self.create_unity_score_visualization(unity_score)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown('<div class="unity-score">', unsafe_allow_html=True)
            st.subheader("📊 Unity Metrics")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Key metrics
            st.metric("Unity Score", f"{unity_score.score:.3f}")
            st.metric("Unique Components", unity_score.unique_components)
            st.metric("Original Nodes", unity_score.original_nodes)
            st.metric("φ-Harmonic", f"{unity_score.phi_harmonic:.3f}")
            
            # Ω-Signature
            st.subheader("Ω-Signature")
            omega_mag = abs(unity_score.omega_signature)
            omega_phase = np.angle(unity_score.omega_signature)
            st.metric("Magnitude", f"{omega_mag:.3f}")
            st.metric("Phase", f"{omega_phase:.3f}")
        
        # Consciousness field visualization
        st.markdown('<div class="consciousness-field">', unsafe_allow_html=True)
        st.subheader("🌊 Consciousness Field Visualization")
        st.markdown("</div>", unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            consciousness_fig = self.create_consciousness_field_plot()
            st.plotly_chart(consciousness_fig, use_container_width=True)
        
        with col4:
            # Unity evolution plot
            thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            evolution_fig = self.create_unity_evolution_plot(thresholds)
            st.plotly_chart(evolution_fig, use_container_width=True)
        
        # Detailed analysis
        st.subheader("🔍 Detailed Analysis")
        
        col5, col6 = st.columns(2)
        
        with col5:
            # Component size distribution
            component_df = pd.DataFrame({
                'Component Size': unity_score.component_sizes,
                'Frequency': [unity_score.component_sizes.count(size) for size in set(unity_score.component_sizes)]
            })
            
            fig_dist = px.histogram(
                component_df, 
                x='Component Size', 
                y='Frequency',
                title="Component Size Distribution",
                nbins=20
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col6:
            # Network statistics
            st.subheader("Network Statistics")
            
            stats_data = {
                "Metric": ["Nodes", "Edges", "Density", "Average Degree", "Diameter"],
                "Value": [
                    G.number_of_nodes(),
                    G.number_of_edges(),
                    nx.density(G),
                    sum(dict(G.degree()).values()) / G.number_of_nodes(),
                    nx.diameter(G) if nx.is_connected(G) else "Disconnected"
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
        
        # Unity Mathematics demonstration
        if UNITY_AVAILABLE and self.unity_math:
            st.subheader("🧮 Unity Mathematics Demonstration")
            
            # Interactive Unity addition
            col7, col8, col9 = st.columns(3)
            
            with col7:
                a_val = st.number_input("Value A", value=1.0, step=0.1)
            
            with col8:
                b_val = st.number_input("Value B", value=1.0, step=0.1)
            
            with col9:
                if st.button("Compute Unity Addition"):
                    state_a = UnityState(
                        value=complex(a_val, 0),
                        phi_resonance=0.618,
                        consciousness_level=consciousness_boost,
                        quantum_coherence=1.0,
                        proof_confidence=1.0
                    )
                    
                    state_b = UnityState(
                        value=complex(b_val, 0),
                        phi_resonance=0.618,
                        consciousness_level=consciousness_boost,
                        quantum_coherence=1.0,
                        proof_confidence=1.0
                    )
                    
                    result = self.unity_math.unity_add(state_a, state_b)
                    
                    st.success(f"Unity Addition Result: {result.value:.3f}")
                    st.info(f"Consciousness Level: {result.consciousness_level:.3f}")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: gray;'>
        <p><strong>Unity Mathematics Dashboard</strong> - Where 1+1=1 through φ-harmonic consciousness</p>
        <p>φ = 1.618033988749895 (Golden Ratio) | Ω-Signature: Holistic phase-signature</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main entry point"""
    if not UNITY_AVAILABLE:
        st.error("Unity Mathematics components not available. Please check dependencies.")
        return
    
    dashboard = UnityDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main() 