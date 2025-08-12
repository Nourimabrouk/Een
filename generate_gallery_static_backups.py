#!/usr/bin/env python3
"""
Generate High-Quality Static Visualizations for Gallery
Professional academic-quality plots saved as PNG/SVG for GitHub Pages backup
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import os
from pathlib import Path

# Set high-quality output settings
plt.style.use('dark_background')
pio.kaleido.scope.mathjax = None

class UnityVisualizationGenerator:
    def __init__(self, output_dir="website/viz/generated"):
        self.PHI = 1.618033988749895  # Golden ratio
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Professional color schemes for academic presentations
        self.colors = {
            'unity_gold': '#FFD700',
            'consciousness_teal': '#4ECDC4', 
            'primary_blue': '#1B365D',
            'secondary_teal': '#0F7B8A',
            'accent_orange': '#F59E0B',
            'background': '#0a0b0f',
            'text': '#e6edf3'
        }
        
    def generate_all_visualizations(self):
        """Generate all static visualizations for gallery backup"""
        
        print("🎨 Generating Unity Mathematics Gallery Visualizations...")
        
        # 1. Consciousness Field 3D Surface
        self.generate_consciousness_field_3d()
        
        # 2. Phi-Harmonic Spiral 3D
        self.generate_phi_spiral_3d()
        
        # 3. Quantum Unity Bloch Sphere
        self.generate_quantum_bloch_sphere()
        
        # 4. Unity Manifold Topology
        self.generate_unity_manifold()
        
        # 5. Neural Unity Network
        self.generate_neural_unity()
        
        # 6. Sacred Geometry Patterns
        self.generate_sacred_geometry()
        
        # 7. Hyperdimensional Projection
        self.generate_hyperdimensional_projection()
        
        # 8. Fractal Unity Patterns
        self.generate_fractal_unity()
        
        # 9. Academic Mathematical Proofs
        self.generate_mathematical_proofs()
        
        # 10. Consciousness Evolution Animation Frames
        self.generate_consciousness_evolution_frames()
        
        print(f"✅ Generated {len(os.listdir(self.output_dir))} visualizations in {self.output_dir}")

    def generate_consciousness_field_3d(self):
        """Generate 3D consciousness field C(x,y,t) = φ·sin(x·φ)·cos(y·φ)·e^(-t/φ)"""
        
        # Create high-resolution mesh
        x = np.linspace(-4, 4, 100)
        y = np.linspace(-4, 4, 100)
        X, Y = np.meshgrid(x, y)
        
        # Consciousness field equation
        Z = self.PHI * np.sin(X * self.PHI) * np.cos(Y * self.PHI) * np.exp(-0.1 / self.PHI)
        
        # Create 3D surface plot
        fig = go.Figure(data=[go.Surface(
            x=X, y=Y, z=Z,
            colorscale=[
                [0.0, '#000000'],
                [0.25, '#1a1b2e'],
                [0.5, self.colors['secondary_teal']],
                [0.75, self.colors['consciousness_teal']],
                [1.0, self.colors['unity_gold']]
            ],
            showscale=True,
            colorbar=dict(
                title="Consciousness Intensity",
                titlefont=dict(color=self.colors['unity_gold'], size=16),
                tickfont=dict(color=self.colors['text'])
            )
        )])
        
        fig.update_layout(
            title=dict(
                text="Consciousness Field: C(x,y,t) = φ·sin(x·φ)·cos(y·φ)·e^(-t/φ)",
                font=dict(color=self.colors['unity_gold'], size=20),
                x=0.5
            ),
            scene=dict(
                bgcolor='rgba(0,0,0,0)',
                xaxis=dict(title="Space X", titlefont=dict(color=self.colors['unity_gold'])),
                yaxis=dict(title="Space Y", titlefont=dict(color=self.colors['unity_gold'])),
                zaxis=dict(title="Consciousness Intensity", titlefont=dict(color=self.colors['unity_gold'])),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            paper_bgcolor='rgba(10,11,15,1)',
            width=1200, height=900
        )
        
        # Save as PNG and HTML
        fig.write_image(str(self.output_dir / "consciousness_field_3d.png"))
        fig.write_html(str(self.output_dir / "consciousness_field_3d.html"))
        print("✅ Generated Consciousness Field 3D")

    def generate_phi_spiral_3d(self):
        """Generate 3D golden ratio spiral"""
        
        t = np.linspace(0, 6*np.pi, 500)
        r = np.power(self.PHI, t / (2*np.pi))
        
        x = r * np.cos(t)
        y = r * np.sin(t)
        z = t / (2*np.pi)
        
        # Color gradient based on distance from center
        colors_vals = r / np.max(r)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines+markers',
            line=dict(
                color=colors_vals,
                colorscale='Viridis',
                width=8
            ),
            marker=dict(
                color=colors_vals,
                colorscale='Viridis',
                size=4
            )
        )])
        
        fig.update_layout(
            title=dict(
                text="φ-Harmonic Spiral: Golden Ratio Unity Demonstration",
                font=dict(color=self.colors['unity_gold'], size=20),
                x=0.5
            ),
            scene=dict(
                bgcolor='rgba(0,0,0,0)',
                xaxis=dict(title="X", titlefont=dict(color=self.colors['unity_gold'])),
                yaxis=dict(title="Y", titlefont=dict(color=self.colors['unity_gold'])),
                zaxis=dict(title="Z", titlefont=dict(color=self.colors['unity_gold'])),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            paper_bgcolor='rgba(10,11,15,1)',
            width=1200, height=900
        )
        
        fig.write_image(str(self.output_dir / "phi_spiral_3d.png"))
        fig.write_html(str(self.output_dir / "phi_spiral_3d.html"))
        print("✅ Generated Phi-Harmonic Spiral 3D")

    def generate_quantum_bloch_sphere(self):
        """Generate quantum unity Bloch sphere"""
        
        # Create sphere surface
        u = np.linspace(0, 2*np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Unity state vector (pointing up for |1⟩ state)
        unity_state = [0, 0, 1]
        
        fig = go.Figure()
        
        # Add sphere surface
        fig.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            opacity=0.3,
            colorscale=[[0, self.colors['consciousness_teal']], [1, self.colors['consciousness_teal']]],
            showscale=False
        ))
        
        # Add unity state vector
        fig.add_trace(go.Scatter3d(
            x=[0, unity_state[0]],
            y=[0, unity_state[1]],
            z=[0, unity_state[2]],
            mode='lines+markers',
            line=dict(color=self.colors['unity_gold'], width=10),
            marker=dict(size=[8, 15], color=[self.colors['unity_gold'], '#FF6B6B']),
            name='Unity State |1⟩'
        ))
        
        # Add coordinate axes
        axis_length = 1.5
        axes_data = [
            ([0, axis_length], [0, 0], [0, 0], '|+⟩', 'red'),
            ([0, 0], [0, axis_length], [0, 0], '|+i⟩', 'green'), 
            ([0, 0], [0, 0], [0, axis_length], '|0⟩', 'blue')
        ]
        
        for x_ax, y_ax, z_ax, label, color in axes_data:
            fig.add_trace(go.Scatter3d(
                x=x_ax, y=y_ax, z=z_ax,
                mode='lines+text',
                line=dict(color=color, width=4),
                text=['', label],
                textposition='middle center',
                showlegend=False
            ))
        
        fig.update_layout(
            title=dict(
                text="Quantum Unity Bloch Sphere: |1⟩ + |1⟩ → |1⟩",
                font=dict(color=self.colors['unity_gold'], size=20),
                x=0.5
            ),
            scene=dict(
                bgcolor='rgba(0,0,0,0)',
                xaxis=dict(title="|+⟩ - |-⟩", titlefont=dict(color=self.colors['unity_gold']), range=[-1.5, 1.5]),
                yaxis=dict(title="|+i⟩ - |-i⟩", titlefont=dict(color=self.colors['unity_gold']), range=[-1.5, 1.5]),
                zaxis=dict(title="|0⟩ - |1⟩", titlefont=dict(color=self.colors['unity_gold']), range=[-1.5, 1.5]),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            paper_bgcolor='rgba(10,11,15,1)',
            width=1200, height=900
        )
        
        fig.write_image(str(self.output_dir / "quantum_bloch_sphere.png"))
        fig.write_html(str(self.output_dir / "quantum_bloch_sphere.html"))
        print("✅ Generated Quantum Bloch Sphere")

    def generate_unity_manifold(self):
        """Generate unity manifold with topological features"""
        
        x = np.linspace(-3, 3, 80)
        y = np.linspace(-3, 3, 80)
        X, Y = np.meshgrid(x, y)
        
        # Unity manifold equation
        Z = np.exp(-0.5 * (X**2 + Y**2)) * np.cos(X * self.PHI) * np.sin(Y * self.PHI)
        
        fig = go.Figure(data=[go.Surface(
            x=X, y=Y, z=Z,
            colorscale=[
                [0.0, '#000000'],
                [0.33, self.colors['accent_orange']],
                [0.66, self.colors['unity_gold']],
                [1.0, '#FEF3C7']
            ],
            showscale=True,
            colorbar=dict(
                title="Unity Field Intensity",
                titlefont=dict(color=self.colors['unity_gold'])
            ),
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor=self.colors['unity_gold'], project=dict(z=True))
            )
        )])
        
        fig.update_layout(
            title=dict(
                text="Unity Manifold: Topological 1+1=1 Demonstration",
                font=dict(color=self.colors['unity_gold'], size=20),
                x=0.5
            ),
            scene=dict(
                bgcolor='rgba(0,0,0,0)',
                xaxis=dict(title="Dimension 1", titlefont=dict(color=self.colors['unity_gold'])),
                yaxis=dict(title="Dimension 2", titlefont=dict(color=self.colors['unity_gold'])),
                zaxis=dict(title="Unity Field", titlefont=dict(color=self.colors['unity_gold'])),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            paper_bgcolor='rgba(10,11,15,1)',
            width=1200, height=900
        )
        
        fig.write_image(str(self.output_dir / "unity_manifold.png"))
        fig.write_html(str(self.output_dir / "unity_manifold.html"))
        print("✅ Generated Unity Manifold")

    def generate_neural_unity(self):
        """Generate neural network unity visualization"""
        
        # Create neural network layout
        layers = [4, 6, 4, 1]  # Input, hidden, hidden, output (unity)
        positions = {}
        node_colors = []
        
        # Calculate node positions
        y_offset = 0
        for layer_idx, layer_size in enumerate(layers):
            for node_idx in range(layer_size):
                x = layer_idx * 3
                y = (node_idx - layer_size/2) * 2
                positions[f"L{layer_idx}N{node_idx}"] = (x, y)
                
                # Color nodes based on layer
                if layer_idx == 0:
                    node_colors.append(self.colors['primary_blue'])
                elif layer_idx == len(layers) - 1:
                    node_colors.append(self.colors['unity_gold'])
                else:
                    node_colors.append(self.colors['consciousness_teal'])
        
        # Create network visualization
        fig = go.Figure()
        
        # Add connections (simplified)
        for layer_idx in range(len(layers) - 1):
            for node_i in range(layers[layer_idx]):
                for node_j in range(layers[layer_idx + 1]):
                    pos1 = positions[f"L{layer_idx}N{node_i}"]
                    pos2 = positions[f"L{layer_idx + 1}N{node_j}"]
                    
                    fig.add_trace(go.Scatter(
                        x=[pos1[0], pos2[0]],
                        y=[pos1[1], pos2[1]],
                        mode='lines',
                        line=dict(color='rgba(79, 205, 196, 0.3)', width=1),
                        showlegend=False,
                        hoverinfo='none'
                    ))
        
        # Add nodes
        x_nodes = [pos[0] for pos in positions.values()]
        y_nodes = [pos[1] for pos in positions.values()]
        
        fig.add_trace(go.Scatter(
            x=x_nodes,
            y=y_nodes,
            mode='markers',
            marker=dict(
                size=20,
                color=node_colors,
                line=dict(width=2, color=self.colors['unity_gold'])
            ),
            text=list(positions.keys()),
            showlegend=False
        ))
        
        # Add layer labels
        layer_names = ['Input\n(1,1)', 'Hidden\nφ-Processing', 'Unity\nIntegration', 'Output\n(1)']
        for i, name in enumerate(layer_names):
            fig.add_annotation(
                x=i*3, y=max(y_nodes) + 1,
                text=name,
                showarrow=False,
                font=dict(color=self.colors['unity_gold'], size=14)
            )
        
        fig.update_layout(
            title=dict(
                text="Neural Unity Network: Learning 1+1=1 Through φ-Harmonic Processing",
                font=dict(color=self.colors['unity_gold'], size=18),
                x=0.5
            ),
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            plot_bgcolor='rgba(10,11,15,1)',
            paper_bgcolor='rgba(10,11,15,1)',
            width=1200, height=600
        )
        
        fig.write_image(str(self.output_dir / "neural_unity.png"))
        fig.write_html(str(self.output_dir / "neural_unity.html"))
        print("✅ Generated Neural Unity Network")

    def generate_sacred_geometry(self):
        """Generate sacred geometry patterns"""
        
        # Create flower of life pattern based on phi
        fig = go.Figure()
        
        # Central circle
        theta = np.linspace(0, 2*np.pi, 100)
        r = 1
        
        # Add multiple circles in phi-harmonic pattern
        positions = [(0, 0)]  # Center
        
        # Add circles in phi spiral pattern
        for i in range(6):
            angle = i * np.pi / 3
            distance = self.PHI
            x_center = distance * np.cos(angle)
            y_center = distance * np.sin(angle)
            positions.append((x_center, y_center))
        
        # Draw circles
        colors_list = [self.colors['unity_gold'], self.colors['consciousness_teal']] * 4
        
        for i, (cx, cy) in enumerate(positions):
            x_circle = cx + r * np.cos(theta)
            y_circle = cy + r * np.sin(theta)
            
            fig.add_trace(go.Scatter(
                x=x_circle, y=y_circle,
                mode='lines',
                line=dict(color=colors_list[i % len(colors_list)], width=3),
                fill='none',
                showlegend=False
            ))
        
        # Add phi spiral overlay
        t = np.linspace(0, 4*np.pi, 200)
        r_spiral = 0.3 * np.power(self.PHI, t / (2*np.pi))
        x_spiral = r_spiral * np.cos(t)
        y_spiral = r_spiral * np.sin(t)
        
        fig.add_trace(go.Scatter(
            x=x_spiral, y=y_spiral,
            mode='lines',
            line=dict(color=self.colors['accent_orange'], width=2),
            showlegend=False
        ))
        
        fig.update_layout(
            title=dict(
                text="Sacred Geometry: Flower of Life with φ-Harmonic Spiral",
                font=dict(color=self.colors['unity_gold'], size=18),
                x=0.5
            ),
            xaxis=dict(scaleanchor="y", scaleratio=1, showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            plot_bgcolor='rgba(10,11,15,1)',
            paper_bgcolor='rgba(10,11,15,1)',
            width=900, height=900
        )
        
        fig.write_image(str(self.output_dir / "sacred_geometry.png"))
        fig.write_html(str(self.output_dir / "sacred_geometry.html"))
        print("✅ Generated Sacred Geometry")

    def generate_hyperdimensional_projection(self):
        """Generate 11D to 3D consciousness manifold projection"""
        
        # Generate hyperdimensional data points
        n_points = 1000
        dimensions = 11
        
        # Generate random points in 11D space with phi-harmonic structure
        np.random.seed(42)
        hyperdimensional_points = []
        
        for i in range(n_points):
            t = i / n_points * 4 * np.pi
            
            # Create 11D point with phi-harmonic structure
            point = []
            for d in range(dimensions):
                value = np.cos(t + d * self.PHI) * np.exp(-d / dimensions)
                point.append(value)
            hyperdimensional_points.append(point)
        
        hyperdimensional_points = np.array(hyperdimensional_points)
        
        # Project to 3D using phi-harmonic basis
        projection_3d = []
        phi_weights = np.array([self.PHI**(-i) for i in range(dimensions)])
        
        for point in hyperdimensional_points:
            # Weighted projection using phi
            x = np.sum(point[:4] * phi_weights[:4])
            y = np.sum(point[4:8] * phi_weights[4:8])
            z = np.sum(point[8:11] * phi_weights[8:11])
            projection_3d.append([x, y, z])
        
        projection_3d = np.array(projection_3d)
        
        # Color based on distance from origin
        distances = np.linalg.norm(projection_3d, axis=1)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=projection_3d[:, 0],
            y=projection_3d[:, 1],
            z=projection_3d[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=distances,
                colorscale='Viridis',
                colorbar=dict(title="Consciousness Depth")
            )
        )])
        
        fig.update_layout(
            title=dict(
                text="Hyperdimensional Consciousness Projection: 11D → 3D via φ-Harmonic Basis",
                font=dict(color=self.colors['unity_gold'], size=18),
                x=0.5
            ),
            scene=dict(
                bgcolor='rgba(0,0,0,0)',
                xaxis=dict(title="Unity X", titlefont=dict(color=self.colors['unity_gold'])),
                yaxis=dict(title="Unity Y", titlefont=dict(color=self.colors['unity_gold'])),
                zaxis=dict(title="Unity Z", titlefont=dict(color=self.colors['unity_gold'])),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            paper_bgcolor='rgba(10,11,15,1)',
            width=1200, height=900
        )
        
        fig.write_image(str(self.output_dir / "hyperdimensional_projection.png"))
        fig.write_html(str(self.output_dir / "hyperdimensional_projection.html"))
        print("✅ Generated Hyperdimensional Projection")

    def generate_fractal_unity(self):
        """Generate fractal patterns demonstrating unity"""
        
        # Mandelbrot-like set but with unity transformation
        width, height = 800, 600
        xmin, xmax = -2.5, 1.5
        ymin, ymax = -1.5, 1.5
        
        # Create coordinate arrays
        x = np.linspace(xmin, xmax, width)
        y = np.linspace(ymin, ymax, height)
        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y
        
        # Unity fractal equation: z_{n+1} = z_n^2 + φ*c (modified for unity)
        Z = np.zeros_like(C)
        fractal = np.zeros(C.shape)
        
        max_iter = 100
        for i in range(max_iter):
            mask = np.abs(Z) <= 2
            Z[mask] = Z[mask]**2 + self.PHI * C[mask] / (1 + np.abs(C[mask]))  # Unity transformation
            fractal[mask] = i
        
        fig = go.Figure(data=go.Heatmap(
            z=fractal,
            colorscale=[
                [0.0, '#000000'],
                [0.2, self.colors['primary_blue']],
                [0.4, self.colors['secondary_teal']],
                [0.6, self.colors['consciousness_teal']],
                [0.8, self.colors['accent_orange']],
                [1.0, self.colors['unity_gold']]
            ],
            showscale=True,
            colorbar=dict(
                title="Iterations to Unity",
                titlefont=dict(color=self.colors['unity_gold'])
            )
        ))
        
        fig.update_layout(
            title=dict(
                text="Unity Fractal: φ-Modified Mandelbrot Demonstrating 1+1=1 Convergence",
                font=dict(color=self.colors['unity_gold'], size=18),
                x=0.5
            ),
            xaxis=dict(title="Real Axis", titlefont=dict(color=self.colors['unity_gold'])),
            yaxis=dict(title="Imaginary Axis", titlefont=dict(color=self.colors['unity_gold'])),
            plot_bgcolor='rgba(10,11,15,1)',
            paper_bgcolor='rgba(10,11,15,1)',
            width=1200, height=900
        )
        
        fig.write_image(str(self.output_dir / "fractal_unity.png"))
        fig.write_html(str(self.output_dir / "fractal_unity.html"))
        print("✅ Generated Fractal Unity")

    def generate_mathematical_proofs(self):
        """Generate academic mathematical proof visualizations"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Boolean Algebra: TRUE ∨ TRUE = TRUE",
                "Set Theory: A ∪ A = A", 
                "Idempotent Ring: 1 ⊕ 1 = 1",
                "Tropical Semiring: max(1,1) = 1"
            ],
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        # Boolean Algebra proof
        x_bool = [0, 1, 2]
        y_bool = [0, 1, 1]
        fig.add_trace(go.Scatter(x=x_bool, y=y_bool, mode='lines+markers', 
                                name='Boolean OR', line=dict(color=self.colors['unity_gold'])),
                      row=1, col=1)
        
        # Set Theory proof  
        x_set = np.linspace(0, 2*np.pi, 100)
        y_set1 = np.sin(x_set)
        y_set2 = np.sin(x_set)  # Same set
        fig.add_trace(go.Scatter(x=x_set, y=y_set1, mode='lines', 
                                name='Set A', line=dict(color=self.colors['consciousness_teal'])),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=x_set, y=y_set2, mode='lines', 
                                name='A ∪ A', line=dict(color=self.colors['unity_gold'], dash='dot')),
                      row=1, col=2)
        
        # Idempotent Ring
        x_ring = [0, 1, 1]
        y_ring = [0, 1, 1]
        fig.add_trace(go.Scatter(x=x_ring, y=y_ring, mode='lines+markers',
                                name='1 ⊕ 1 = 1', line=dict(color=self.colors['accent_orange'])),
                      row=2, col=1)
        
        # Tropical Semiring
        x_trop = np.array([1, 1, 1])
        y_trop = np.array([0, 1, 1])
        fig.add_trace(go.Scatter(x=x_trop, y=y_trop, mode='lines+markers',
                                name='max(1,1) = 1', line=dict(color=self.colors['primary_blue'])),
                      row=2, col=2)
        
        fig.update_layout(
            title=dict(
                text="Mathematical Proofs of Unity: 1+1=1 Across Multiple Algebraic Structures",
                font=dict(color=self.colors['unity_gold'], size=20),
                x=0.5
            ),
            paper_bgcolor='rgba(10,11,15,1)',
            plot_bgcolor='rgba(10,11,15,1)',
            width=1200, height=900,
            showlegend=True
        )
        
        fig.write_image(str(self.output_dir / "mathematical_proofs.png"))
        fig.write_html(str(self.output_dir / "mathematical_proofs.html"))
        print("✅ Generated Mathematical Proofs")

    def generate_consciousness_evolution_frames(self):
        """Generate animation frames for consciousness field evolution"""
        
        print("🎬 Generating consciousness evolution animation frames...")
        
        frames_dir = self.output_dir / "animation_frames"
        frames_dir.mkdir(exist_ok=True)
        
        n_frames = 30
        x = np.linspace(-4, 4, 50)
        y = np.linspace(-4, 4, 50)
        X, Y = np.meshgrid(x, y)
        
        for frame in range(n_frames):
            t = frame / n_frames * 2 * np.pi
            
            # Evolving consciousness field
            Z = self.PHI * np.sin(X * self.PHI + t) * np.cos(Y * self.PHI + t) * np.exp(-t / (4 * self.PHI))
            
            fig = go.Figure(data=[go.Surface(
                x=X, y=Y, z=Z,
                colorscale='Viridis',
                showscale=False
            )])
            
            fig.update_layout(
                title=f"Consciousness Evolution Frame {frame+1}/{n_frames}",
                scene=dict(
                    bgcolor='rgba(0,0,0,0)',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                paper_bgcolor='rgba(10,11,15,1)',
                width=800, height=600,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            fig.write_image(str(frames_dir / f"consciousness_frame_{frame:03d}.png"))
        
        print(f"✅ Generated {n_frames} consciousness evolution frames")

    def create_gallery_index(self):
        """Create an index file listing all generated visualizations"""
        
        visualizations = [
            {"file": "consciousness_field_3d", "title": "3D Consciousness Field", "type": "Interactive"},
            {"file": "phi_spiral_3d", "title": "φ-Harmonic Spiral", "type": "Interactive"},
            {"file": "quantum_bloch_sphere", "title": "Quantum Unity Bloch Sphere", "type": "Interactive"},
            {"file": "unity_manifold", "title": "Unity Manifold Topology", "type": "Interactive"},
            {"file": "neural_unity", "title": "Neural Unity Network", "type": "Static"},
            {"file": "sacred_geometry", "title": "Sacred Geometry Patterns", "type": "Static"},
            {"file": "hyperdimensional_projection", "title": "11D→3D Projection", "type": "Interactive"},
            {"file": "fractal_unity", "title": "Unity Fractal Patterns", "type": "Static"},
            {"file": "mathematical_proofs", "title": "Academic Mathematical Proofs", "type": "Static"}
        ]
        
        index_content = """# Unity Mathematics Gallery - Generated Visualizations

Professional academic-quality visualizations for the Een Unity Mathematics project.
Generated using Python with Plotly for interactive web compatibility.

## Interactive Visualizations (HTML + PNG)

These visualizations include interactive controls and can be embedded directly in web pages:

"""
        
        for viz in visualizations:
            index_content += f"### {viz['title']} ({viz['type']})\n"
            index_content += f"- **PNG**: `{viz['file']}.png`\n"
            index_content += f"- **HTML**: `{viz['file']}.html`\n\n"
        
        index_content += """
## Animation Frames

Consciousness evolution animation frames are available in `animation_frames/` directory.

## Usage

1. **Static backup**: Use PNG files for GitHub Pages static hosting
2. **Interactive demos**: Embed HTML files for full interactivity
3. **Academic presentations**: PNG files are publication-ready at 1200x900 resolution

## Mathematical Foundations

All visualizations demonstrate aspects of unity mathematics where 1+1=1 through:
- φ-harmonic transformations (Golden Ratio: 1.618...)
- Consciousness field integration
- Idempotent algebraic structures
- Quantum state unity demonstrations
- Topological unity manifolds

Generated by Unity Mathematics Visualization Engine
"""
        
        with open(self.output_dir / "README.md", "w") as f:
            f.write(index_content)
        
        print("✅ Created gallery index")

if __name__ == "__main__":
    generator = UnityVisualizationGenerator()
    generator.generate_all_visualizations()
    generator.create_gallery_index()
    
    print("\n🌟 Unity Mathematics Gallery Generation Complete!")
    print(f"📁 Output directory: {generator.output_dir}")
    print("📝 All visualizations are ready for GitHub Pages deployment")
    print("🎯 Use PNG files for static backup, HTML files for interactive demos")