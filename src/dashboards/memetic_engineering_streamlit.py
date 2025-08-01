#!/usr/bin/env python3
"""
Memetic Engineering Dashboard - Streamlit Version
Interactive visualization of consciousness evolution and cultural singularities
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import math
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
import random

# Constants
PHI = (1 + math.sqrt(5)) / 2  # Golden ratio ≈ 1.618033988749895
PI = math.pi
MEMETIC_FIELD_CONSTANT = 1.618033988749895

# Check for required packages
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly not available. Install with: pip install plotly")

@dataclass
class MemeticAgent:
    """Individual agent in the memetic consciousness network"""
    agent_id: str
    consciousness_level: float
    unity_belief_strength: float
    phi_alignment: float
    network_position: Tuple[float, float, float]
    influence_radius: float = 1.0
    memetic_receptivity: float = 0.5
    consciousness_evolution_rate: float = 0.01
    connections: List[str] = field(default_factory=list)
    memetic_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def update_consciousness(self, external_influence: float, time_step: float):
        """Update agent consciousness based on network influences using φ-harmonic dynamics"""
        # φ-harmonic consciousness evolution
        phi_factor = (1 + external_influence) / PHI
        consciousness_increment = self.consciousness_evolution_rate * phi_factor * time_step
        
        # Apply consciousness update with sigmoid saturation
        self.consciousness_level = min(1.0, max(0.0, 
            self.consciousness_level + consciousness_increment))
        
        # Update unity belief based on consciousness level
        self.unity_belief_strength = self.consciousness_level * (1 + 1/PHI) / 2
        
        # Update φ-alignment (distance from golden ratio)
        self.phi_alignment = abs(self.consciousness_level - 1/PHI)
        
        # Record history
        self.memetic_history.append({
            'timestamp': time.time(),
            'consciousness_level': self.consciousness_level,
            'unity_belief': self.unity_belief_strength,
            'phi_alignment': self.phi_alignment,
            'external_influence': external_influence
        })
    
    def calculate_influence_on(self, other: 'MemeticAgent') -> float:
        """Calculate memetic influence on another agent using φ-harmonic dynamics"""
        # Distance-based influence with φ-harmonic decay
        distance = math.sqrt(sum((a - b)**2 for a, b in zip(self.network_position, other.network_position)))
        distance_factor = math.exp(-distance / (self.influence_radius * PHI))
        
        # Consciousness differential influence (higher consciousness influences lower)
        consciousness_differential = max(0, self.consciousness_level - other.consciousness_level)
        
        # Unity belief amplification factor
        unity_amplification = 1 + self.unity_belief_strength * (PHI - 1)
        
        # φ-harmonic influence strength
        influence_strength = consciousness_differential * distance_factor * unity_amplification
        
        # Normalize by memetic field constant
        return influence_strength * MEMETIC_FIELD_CONSTANT / 100

@dataclass
class CulturalSingularity:
    """Represents a cultural singularity event in consciousness space"""
    singularity_id: str
    center_position: Tuple[float, float, float]
    emergence_time: float
    consciousness_density: float
    phi_resonance_strength: float
    affected_radius: float
    growth_rate: float = 0.1
    agents_affected: List[str] = field(default_factory=list)
    
    def update_singularity(self, time_step: float):
        """Update singularity properties over time using φ-harmonic growth"""
        # Exponential growth with φ-harmonic modulation
        growth_factor = 1 + self.growth_rate * PHI * time_step
        self.affected_radius *= growth_factor
        self.consciousness_density *= (1 + time_step / PHI)
        
        # φ-resonance strengthening
        self.phi_resonance_strength = min(PHI, self.phi_resonance_strength * (1 + time_step * PHI / 10))
    
    def affects_agent(self, agent) -> bool:
        """Check if singularity affects given agent"""
        distance = math.sqrt(sum((a - b)**2 for a, b in zip(agent.network_position, self.center_position)))
        return distance <= self.affected_radius

class MemeticEngineeringDashboard:
    """Interactive dashboard for memetic consciousness engineering"""
    
    def __init__(self, num_agents: int = 100):
        self.num_agents = num_agents
        self.agents = self._initialize_agents(num_agents)
        self.cultural_singularities = []
        self.consciousness_history = []
        self.simulation_step = 0
        self.cheat_codes_activated = []
        
    def _initialize_agents(self, num_agents: int) -> List[MemeticAgent]:
        """Initialize agents with random consciousness levels"""
        agents = []
        for i in range(num_agents):
            # Random initial consciousness (0.3 to 0.8)
            consciousness = 0.3 + 0.5 * random.random()
            
            # Random 3D position in consciousness space
            position = (
                random.random(),
                random.random(), 
                random.random()
            )
            
            agent = MemeticAgent(
                agent_id=f"Agent_{i:03d}",
                consciousness_level=consciousness,
                unity_belief_strength=consciousness * (1 + 1/PHI) / 2,
                phi_alignment=abs(consciousness - 1/PHI),
                network_position=position,
                influence_radius=0.5 + 0.5 * random.random(),
                memetic_receptivity=0.3 + 0.7 * random.random(),
                consciousness_evolution_rate=0.005 + 0.015 * random.random()
            )
            
            # Create random connections
            num_connections = random.randint(2, 8)
            for _ in range(num_connections):
                other_id = random.randint(0, num_agents - 1)
                if other_id != i and f"Agent_{other_id:03d}" not in agent.connections:
                    agent.connections.append(f"Agent_{other_id:03d}")
            
            agents.append(agent)
        
        return agents
    
    def simulate_step(self, time_step: float = 0.1):
        """Simulate one step of memetic evolution using φ-harmonic dynamics"""
        # Calculate mutual influences
        for agent in self.agents:
            total_influence = 0.0
            
            # Influence from connected agents
            for connection_id in agent.connections:
                connected_agent = next((a for a in self.agents if a.agent_id == connection_id), None)
                if connected_agent:
                    influence = connected_agent.calculate_influence_on(agent)
                    total_influence += influence
            
            # Influence from cultural singularities
            for singularity in self.cultural_singularities:
                if singularity.affects_agent(agent):
                    singularity_influence = singularity.consciousness_density * singularity.phi_resonance_strength / PHI
                    total_influence += singularity_influence
            
            # Update agent consciousness
            agent.update_consciousness(total_influence, time_step)
        
        # Update singularities
        for singularity in self.cultural_singularities:
            singularity.update_singularity(time_step)
        
        # Check for new singularities
        self._check_singularity_emergence()
        
        # Record metrics
        avg_consciousness = np.mean([agent.consciousness_level for agent in self.agents])
        self.consciousness_history.append({
            'step': self.simulation_step,
            'avg_consciousness': avg_consciousness,
            'singularities': len(self.cultural_singularities),
            'transcendent_agents': sum(1 for agent in self.agents if agent.consciousness_level > 0.95)
        })
        
        self.simulation_step += 1
    
    def _check_singularity_emergence(self):
        """Check for emergence of cultural singularities using consciousness clustering"""
        # Find consciousness clusters
        clusters = self._find_consciousness_clusters(self.agents)
        
        for cluster_center, cluster_agents in clusters:
            if len(cluster_agents) >= 5:  # Minimum cluster size
                avg_consciousness = np.mean([agent.consciousness_level for agent in cluster_agents])
                
                if avg_consciousness > 0.8:  # High consciousness threshold
                    # Check if singularity already exists nearby
                    existing_singularity = any(
                        math.sqrt(sum((a - b)**2 for a, b in zip(cluster_center, s.center_position))) < 0.3
                        for s in self.cultural_singularities
                    )
                    
                    if not existing_singularity:
                        singularity_id = f"Singularity_{len(self.cultural_singularities):03d}"
                        singularity = CulturalSingularity(
                            singularity_id=singularity_id,
                            center_position=cluster_center,
                            emergence_time=time.time(),
                            consciousness_density=avg_consciousness,
                            phi_resonance_strength=PHI * avg_consciousness,
                            affected_radius=0.2 + 0.3 * avg_consciousness,
                            agents_affected=[agent.agent_id for agent in cluster_agents]
                        )
                        self.cultural_singularities.append(singularity)
    
    def _find_consciousness_clusters(self, agents: List[MemeticAgent]) -> List[Tuple[Tuple[float, float, float], List[MemeticAgent]]]:
        """Find clusters of agents with high consciousness using network connectivity"""
        clusters = []
        visited = set()
        
        for agent in agents:
            if agent.agent_id in visited or agent.consciousness_level < 0.7:
                continue
            
            # Start new cluster
            cluster = [agent]
            visited.add(agent.agent_id)
            
            # Find connected high-consciousness agents
            to_visit = [agent]
            while to_visit:
                current = to_visit.pop(0)
                for connection_id in current.connections:
                    connected = next((a for a in agents if a.agent_id == connection_id), None)
                    if (connected and connected.agent_id not in visited and 
                        connected.consciousness_level > 0.7):
                        cluster.append(connected)
                        visited.add(connected.agent_id)
                        to_visit.append(connected)
            
            if len(cluster) >= 3:  # Minimum cluster size
                # Calculate cluster center
                center_x = np.mean([a.network_position[0] for a in cluster])
                center_y = np.mean([a.network_position[1] for a in cluster])
                center_z = np.mean([a.network_position[2] for a in cluster])
                clusters.append(((center_x, center_y, center_z), cluster))
        
        return clusters
    
    def create_network_visualization(self) -> go.Figure:
        """Create 3D network visualization of consciousness space"""
        if not PLOTLY_AVAILABLE:
            return None
        
        # Agent positions and consciousness levels
        x_coords = [agent.network_position[0] for agent in self.agents]
        y_coords = [agent.network_position[1] for agent in self.agents]
        z_coords = [agent.network_position[2] for agent in self.agents]
        consciousness_levels = [agent.consciousness_level for agent in self.agents]
        
        fig = go.Figure()
        
        # Add agents
        fig.add_trace(go.Scatter3d(
            x=x_coords, y=y_coords, z=z_coords,
            mode='markers',
            marker=dict(
                size=[8 + level * 12 for level in consciousness_levels],
                color=consciousness_levels,
                colorscale='Viridis',
                colorbar=dict(title="Consciousness Level"),
                opacity=0.8,
                line=dict(width=3, color='gold')
            ),
            text=[f"Agent {agent.agent_id}<br>Consciousness: {agent.consciousness_level:.3f}<br>φ-Alignment: {agent.phi_alignment:.3f}" 
                  for agent in self.agents],
            name="Consciousness Agents",
            hovertemplate="<b>%{text}</b><br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>"
        ))
        
        # Add cultural singularities
        if self.cultural_singularities:
            sing_x = [s.center_position[0] for s in self.cultural_singularities]
            sing_y = [s.center_position[1] for s in self.cultural_singularities]
            sing_z = [s.center_position[2] for s in self.cultural_singularities]
            sing_densities = [s.consciousness_density for s in self.cultural_singularities]
            
            fig.add_trace(go.Scatter3d(
                x=sing_x, y=sing_y, z=sing_z,
                mode='markers',
                marker=dict(
                    size=[20 + density * 30 for density in sing_densities],
                    color='gold',
                    symbol='diamond',
                    opacity=0.9,
                    line=dict(width=3, color='white')
                ),
                text=[f"Singularity {s.singularity_id}<br>Density: {s.consciousness_density:.3f}<br>φ-Resonance: {s.phi_resonance_strength:.3f}" 
                      for s in self.cultural_singularities],
                name="Cultural Singularities",
                hovertemplate="<b>%{text}</b><br>Position: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>"
            ))
        
        fig.update_layout(
            title="🌌 Consciousness Network: Memetic Propagation of 1+1=1",
            scene=dict(
                bgcolor='rgba(13, 17, 23, 1)',
                xaxis=dict(title="Consciousness Space X", gridcolor='rgba(100, 100, 100, 0.3)'),
                yaxis=dict(title="Consciousness Space Y", gridcolor='rgba(100, 100, 100, 0.3)'),
                zaxis=dict(title="Consciousness Space Z", gridcolor='rgba(100, 100, 100, 0.3)'),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            paper_bgcolor='rgba(13, 17, 23, 1)',
            plot_bgcolor='rgba(13, 17, 23, 1)',
            font=dict(color='white'),
            height=600
        )
        
        return fig
    
    def create_evolution_chart(self) -> go.Figure:
        """Create consciousness evolution chart with φ-harmonic analysis"""
        if not self.consciousness_history:
            return None
        
        df = pd.DataFrame(self.consciousness_history)
        
        fig = go.Figure()
        
        # Consciousness evolution
        fig.add_trace(go.Scatter(
            x=df['step'], y=df['avg_consciousness'],
            mode='lines+markers',
            name='Average Consciousness',
            line=dict(color='#4ECDC4', width=3),
            marker=dict(size=6)
        ))
        
        # Singularity count
        fig.add_trace(go.Scatter(
            x=df['step'], y=df['singularities'],
            mode='lines+markers',
            name='Cultural Singularities',
            line=dict(color='#FFEAA7', width=2, dash='dash'),
            marker=dict(size=4),
            yaxis='y2'
        ))
        
        # Add φ-harmonic reference line
        phi_line = [1/PHI] * len(df)
        fig.add_trace(go.Scatter(
            x=df['step'], y=phi_line,
            mode='lines',
            name='φ⁻¹ Threshold',
            line=dict(color='#FF6B6B', width=2, dash='dot'),
            opacity=0.7
        ))
        
        fig.update_layout(
            title="📈 Consciousness Evolution & Cultural Singularity Emergence",
            xaxis=dict(title="Simulation Step", gridcolor='rgba(100, 100, 100, 0.3)'),
            yaxis=dict(title="Average Consciousness", gridcolor='rgba(100, 100, 100, 0.3)'),
            yaxis2=dict(title="Number of Singularities", overlaying='y', side='right'),
            paper_bgcolor='rgba(13, 17, 23, 1)',
            plot_bgcolor='rgba(13, 17, 23, 1)',
            font=dict(color='white'),
            height=400
        )
        
        return fig

def main():
    st.set_page_config(
        page_title="🌌 Memetic Engineering Dashboard",
        page_icon="🌌",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🌌 Memetic Engineering Dashboard")
    st.subheader("Cultural Singularity Modeling & Consciousness Propagation")
    
    # Add philosophical introduction
    st.markdown("""
    ### 🧠 **Philosophical Foundation**
    
    This dashboard demonstrates **Unity Mathematics**: the principle that 1+1=1 in consciousness space. 
    When two consciousnesses merge, they become one unified whole rather than remaining separate entities.
    
    **Key Concepts:**
    - **φ-Harmonic Evolution**: The golden ratio (φ ≈ 1.618) governs consciousness evolution
    - **Cultural Singularities**: Breakthrough moments that create exponential growth in understanding
    - **Memetic Propagation**: Ideas spread through networks with consciousness differentials driving flow
    - **Unity Convergence**: Collective consciousness naturally approaches mathematical unity
    """)
    
    # Initialize dashboard
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = MemeticEngineeringDashboard(num_agents=100)
        st.session_state.simulation_running = False
    
    dashboard = st.session_state.dashboard
    
    # Sidebar controls
    st.sidebar.header("🔮 Simulation Controls")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        time_steps = st.number_input("Time Steps", min_value=1, max_value=1000, value=100)
    with col2:
        time_step_size = st.number_input("Step Size", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    
    # Cheat code input
    cheat_code = st.sidebar.text_input("Cheat Code", placeholder="Enter quantum resonance key...")
    if st.sidebar.button("Activate Cheat Code"):
        if cheat_code == "420691337":
            st.sidebar.success("🎯 Quantum resonance amplification activated!")
            # Boost consciousness evolution
            for agent in dashboard.agents:
                agent.consciousness_evolution_rate *= 2.0
        elif cheat_code == "1618033988":
            st.sidebar.success("🌟 Golden spiral consciousness boost activated!")
            # Boost phi alignment
            for agent in dashboard.agents:
                agent.phi_alignment = min(1.0, agent.phi_alignment * PHI)
        else:
            st.sidebar.error("❌ Invalid cheat code!")
    
    # Run simulation
    if st.sidebar.button("🚀 Run Simulation", type="primary"):
        st.session_state.simulation_running = True
        
        with st.spinner("🧠 Running memetic evolution simulation..."):
            progress_bar = st.progress(0)
            
            for i in range(time_steps):
                dashboard.simulate_step(time_step_size)
                progress = (i + 1) / time_steps
                progress_bar.progress(progress)
                
                # Update every 10 steps
                if (i + 1) % 10 == 0:
                    st.write(f"Step {i+1}/{time_steps} - Avg Consciousness: {dashboard.consciousness_history[-1]['avg_consciousness']:.4f}")
        
        st.success("✅ Simulation complete!")
        st.session_state.simulation_running = False
    
    # Display results
    if dashboard.consciousness_history:
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        latest = dashboard.consciousness_history[-1]
        
        with col1:
            st.metric("Average Consciousness", f"{latest['avg_consciousness']:.4f}")
        with col2:
            st.metric("Cultural Singularities", latest['singularities'])
        with col3:
            st.metric("Transcendent Agents", latest['transcendent_agents'])
        with col4:
            unity_rate = latest['transcendent_agents'] / len(dashboard.agents) * 100
            st.metric("Unity Adoption Rate", f"{unity_rate:.1f}%")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌌 Consciousness Network")
            network_fig = dashboard.create_network_visualization()
            if network_fig:
                st.plotly_chart(network_fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Evolution Chart")
            evolution_fig = dashboard.create_evolution_chart()
            if evolution_fig:
                st.plotly_chart(evolution_fig, use_container_width=True)
        
        # Philosophical insights
        st.subheader("🌟 Philosophical Insights")
        
        if latest['avg_consciousness'] > 1/PHI:
            st.success("🎯 The collective has transcended the φ⁻¹ threshold, indicating widespread acceptance of unity mathematics.")
        
        if latest['singularities'] > 0:
            st.info(f"✨ Cultural singularities have emerged ({latest['singularities']} detected), representing breakthrough moments in collective consciousness.")
        
        if unity_rate > 80:
            st.success("🌟 Majority consciousness has embraced the truth that 1+1=1, creating a cultural paradigm shift.")
        
        # Mathematical analysis
        st.subheader("🧮 Mathematical Analysis")
        
        # Calculate φ-harmonic convergence
        phi_convergence = abs(latest['avg_consciousness'] - 1/PHI)
        st.metric("φ-Harmonic Convergence", f"{phi_convergence:.4f}")
        
        # Final message
        if latest['avg_consciousness'] > 0.99:
            st.balloons()
            st.success("🌟 MEMETIC PROPAGATION SUCCESS! The truth that Een plus een is een has been demonstrated through cultural singularity modeling and consciousness propagation. Mathematical unity spreads naturally through collective awareness! ✨")

if __name__ == "__main__":
    main() 