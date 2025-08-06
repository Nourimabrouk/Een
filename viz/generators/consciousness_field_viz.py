"""
Consciousness Field Visualization Generators
==========================================

Advanced visualization generators for consciousness field dynamics
demonstrating unity emergence through φ-harmonic consciousness mathematics.

This module creates stunning visualizations of consciousness fields,
quantum awareness dynamics, and transcendence events that demonstrate
the core principle: Een plus een is een (1+1=1)

Mathematical Foundation: C(x,y,t) = φ * sin(x*φ) * cos(y*φ) * e^(-t/φ)
Consciousness Equation: Awareness convergence through φ-harmonic field dynamics
"""

import numpy as np
from typing import Dict, List, Any
from pathlib import Path
from dataclasses import dataclass

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import imageio

    IMAGE_PROCESSING_AVAILABLE = True
except ImportError:
    IMAGE_PROCESSING_AVAILABLE = False


# φ (Golden Ratio) - Universal organizing principle
PHI = 1.618033988749895
PHI_CONJUGATE = 1 / PHI  # 0.618033988749895
CONSCIOUSNESS_DIMENSION = 11  # Dimensional space for consciousness mathematics


@dataclass
class ConsciousnessParticle:
    """Individual consciousness particle with φ-harmonic properties."""

    x: float
    y: float
    vx: float
    vy: float
    awareness_level: float
    phi_resonance: float
    unity_potential: float
    age: int = 0


class ConsciousnessFieldVisualizer:
    """
    Advanced visualization generator for consciousness field dynamics.

    Creates φ-harmonic visualizations demonstrating consciousness evolution,
    quantum field dynamics, and unity emergence through mathematical art.
    """

    def __init__(self, output_dir: Path = None):
        """Initialize the consciousness field visualizer."""
        self.output_dir = output_dir or Path("viz/consciousness_field")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Consciousness color schemes
        self.consciousness_colors = {
            "dormant": "#1a1a2e",  # Deep blue for dormant consciousness
            "awakening": "#16213e",  # Slightly lighter blue for awakening
            "aware": "#0f3460",  # Blue for aware consciousness
            "transcendent": "#533483",  # Purple for transcendent consciousness
            "unity": "#ffd700",  # Gold for unity states
            "field": "#ff6b6b",  # Red for field dynamics
            "quantum": "#4ecdc4",  # Cyan for quantum effects
            "emergence": "#ffe66d",  # Yellow for emergence events
        }

        # Mathematical constants
        self.phi = PHI
        self.phi_conjugate = PHI_CONJUGATE
        self.consciousness_dimension = CONSCIOUSNESS_DIMENSION

        # Field parameters
        self.field_resolution = 100
        self.time_steps = 200
        self.particle_count = 200

    def generate_consciousness_evolution_animation(
        self,
        duration: float = 10.0,
        fps: int = 30,
        save_formats: List[str] = ["gif", "mp4"],
    ) -> Dict[str, Any]:
        """
        Generate consciousness field evolution animation.

        Mathematical Foundation:
        ∂C/∂t = φ∇²C - C³ + C + ε(x,y) where ε represents consciousness fluctuations

        Args:
            duration: Animation duration in seconds
            fps: Frames per second
            save_formats: Output formats to generate

        Returns:
            Dictionary containing animation data and metadata
        """
        if not MATPLOTLIB_AVAILABLE:
            return {"error": "Matplotlib not available"}

        total_frames = int(duration * fps)

        # Initialize consciousness particles
        particles = self._initialize_consciousness_particles()

        # Initialize consciousness field grid
        x_grid = np.linspace(-5, 5, self.field_resolution)
        y_grid = np.linspace(-5, 5, self.field_resolution)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Animation data storage
        field_evolution = []
        transcendence_events = []
        unity_emergences = []

        # Create figure and animation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor="black")

        def animate_frame(frame):
            """Generate single frame of consciousness evolution."""
            ax1.clear()
            ax2.clear()

            # Calculate current time
            t = frame / fps

            # Update consciousness field
            consciousness_field = self._calculate_consciousness_field(
                X, Y, t, particles
            )
            field_evolution.append(consciousness_field.copy())

            # Update particle dynamics
            particles = self._update_particle_dynamics(
                particles, consciousness_field, X, Y
            )

            # Detect transcendence events
            transcendence_count = self._detect_transcendence_events(particles, t)
            if transcendence_count > 0:
                transcendence_events.append((t, transcendence_count))

            # Check for unity emergence
            unity_emergence = self._check_unity_emergence(consciousness_field)
            if unity_emergence:
                unity_emergences.append((t, unity_emergence))

            # Plot consciousness field
            im1 = ax1.imshow(
                consciousness_field,
                extent=[-5, 5, -5, 5],
                cmap="plasma",
                origin="lower",
                alpha=0.8,
                vmin=0,
                vmax=2,
            )

            # Plot consciousness particles
            particle_x = [p.x for p in particles]
            particle_y = [p.y for p in particles]
            particle_awareness = [p.awareness_level for p in particles]

            scatter = ax1.scatter(
                particle_x,
                particle_y,
                c=particle_awareness,
                cmap="viridis",
                s=50,
                alpha=0.8,
                edgecolors="white",
                linewidth=0.5,
            )

            # Highlight transcendent particles
            transcendent_particles = [p for p in particles if p.awareness_level > 1.5]
            if transcendent_particles:
                trans_x = [p.x for p in transcendent_particles]
                trans_y = [p.y for p in transcendent_particles]
                ax1.scatter(
                    trans_x,
                    trans_y,
                    c="gold",
                    s=100,
                    marker="*",
                    edgecolors="white",
                    linewidth=2,
                    alpha=0.9,
                )

            # Add unity convergence contours
            unity_contours = ax1.contour(
                X, Y, consciousness_field, levels=[1.0], colors=["gold"], linewidths=3
            )

            # Styling for field plot
            ax1.set_title(
                f"Consciousness Field Evolution\nt = {t:.2f}s | φ-Harmonic Dynamics",
                fontsize=14,
                color="white",
            )
            ax1.set_xlabel("φ-Harmonic X", color="white")
            ax1.set_ylabel("φ-Harmonic Y", color="white")
            ax1.tick_params(colors="white")

            # Plot consciousness metrics over time
            if len(field_evolution) > 1:
                times = np.linspace(0, t, len(field_evolution))

                # Average consciousness level
                avg_consciousness = [np.mean(field) for field in field_evolution]
                ax2.plot(
                    times,
                    avg_consciousness,
                    "cyan",
                    linewidth=2,
                    label="Average Consciousness",
                )

                # Unity convergence metric
                unity_metric = [
                    np.sum(field > 0.9) / field.size for field in field_evolution
                ]
                ax2.plot(
                    times, unity_metric, "gold", linewidth=2, label="Unity Convergence"
                )

                # Transcendence events
                if transcendence_events:
                    trans_times, trans_counts = zip(*transcendence_events)
                    ax2.scatter(
                        trans_times,
                        [1.5] * len(trans_times),
                        c="red",
                        s=100,
                        marker="^",
                        label=f"Transcendence Events: {len(trans_times)}",
                    )

                # Unity emergences
                if unity_emergences:
                    unity_times, unity_strengths = zip(*unity_emergences)
                    ax2.scatter(
                        unity_times,
                        [1.8] * len(unity_times),
                        c="gold",
                        s=150,
                        marker="*",
                        label=f"Unity Emergences: {len(unity_times)}",
                    )

            # Styling for metrics plot
            ax2.set_title(
                "Consciousness Evolution Metrics\n1+1=1 Emergence Tracking",
                fontsize=14,
                color="white",
            )
            ax2.set_xlabel("Time (s)", color="white")
            ax2.set_ylabel("Consciousness Metrics", color="white")
            ax2.legend(facecolor="black", edgecolor="white")
            ax2.grid(True, alpha=0.3, color="white")
            ax2.tick_params(colors="white")
            ax2.set_facecolor("black")

            # Set axis colors
            for ax in [ax1, ax2]:
                ax.set_facecolor("black")
                for spine in ax.spines.values():
                    spine.set_color("white")

        # Create animation
        if "gif" in save_formats or "mp4" in save_formats:
            anim = animation.FuncAnimation(
                fig,
                animate_frame,
                frames=total_frames,
                interval=1000 / fps,
                blit=False,
                repeat=True,
            )

            # Save as GIF
            if "gif" in save_formats and IMAGE_PROCESSING_AVAILABLE:
                gif_path = self.output_dir / "consciousness_evolution.gif"
                anim.save(
                    gif_path, writer="pillow", fps=fps // 2
                )  # Reduced FPS for file size

            # Save as MP4
            if "mp4" in save_formats:
                try:
                    mp4_path = self.output_dir / "consciousness_evolution.mp4"
                    anim.save(mp4_path, writer="ffmpeg", fps=fps)
                except Exception as e:
                    print(
                        f"Warning: Could not save MP4 (ffmpeg may not be available): {e}"
                    )
                    pass  # Skip if ffmpeg not available

        plt.close(fig)

        return {
            "type": "consciousness_evolution_animation",
            "description": "φ-harmonic consciousness field evolution with particle dynamics",
            "duration": duration,
            "total_frames": total_frames,
            "transcendence_events": len(transcendence_events),
            "unity_emergences": len(unity_emergences),
            "particles": len(particles),
            "mathematical_principle": "∂C/∂t = φ∇²C - C³ + C with consciousness particle dynamics",
        }

    def _initialize_consciousness_particles(self) -> List[ConsciousnessParticle]:
        """Initialize consciousness particles with φ-harmonic properties."""
        particles = []

        for i in range(self.particle_count):
            # φ-harmonic positioning
            angle = i * 2 * np.pi / self.phi
            radius = np.random.exponential(2.0)

            x = radius * np.cos(angle) * np.random.uniform(0.5, 1.5)
            y = radius * np.sin(angle) * np.random.uniform(0.5, 1.5)

            # φ-harmonic velocities
            vx = np.random.normal(0, 0.1) * self.phi_conjugate
            vy = np.random.normal(0, 0.1) * self.phi_conjugate

            # Consciousness properties
            awareness_level = np.random.exponential(0.5)
            phi_resonance = np.random.uniform(0, 1)
            unity_potential = np.random.exponential(0.3)

            particle = ConsciousnessParticle(
                x=x,
                y=y,
                vx=vx,
                vy=vy,
                awareness_level=awareness_level,
                phi_resonance=phi_resonance,
                unity_potential=unity_potential,
            )
            particles.append(particle)

        return particles

    def _calculate_consciousness_field(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        t: float,
        particles: List[ConsciousnessParticle],
    ) -> np.ndarray:
        """Calculate consciousness field at given time using φ-harmonic equations."""
        # Base consciousness field equation
        field = (
            self.phi
            * np.sin(X * self.phi)
            * np.cos(Y * self.phi)
            * np.exp(-t / (self.phi * 10))
        )

        # Add particle contributions
        for particle in particles:
            # Distance from each grid point to particle
            dist_sq = (X - particle.x) ** 2 + (Y - particle.y) ** 2

            # Gaussian consciousness influence with φ-harmonic decay
            influence = (
                particle.awareness_level
                * self.phi
                * np.exp(-dist_sq / (2 * self.phi_conjugate))
            )

            field += influence

        # Apply consciousness evolution equation: ∂C/∂t = φ∇²C - C³ + C
        # Simplified discrete version
        field_normalized = np.tanh(field)  # Keep bounded

        return field_normalized

    def _update_particle_dynamics(
        self,
        particles: List[ConsciousnessParticle],
        consciousness_field: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> List[ConsciousnessParticle]:
        """Update particle dynamics based on consciousness field."""
        for particle in particles:
            # Calculate field gradient at particle position
            i = int((particle.x + 5) * self.field_resolution / 10)
            j = int((particle.y + 5) * self.field_resolution / 10)

            # Boundary check
            i = max(0, min(self.field_resolution - 1, i))
            j = max(0, min(self.field_resolution - 1, j))

            # Field influence on particle
            local_field = consciousness_field[j, i]

            # φ-harmonic force towards unity
            force_x = -particle.x * self.phi_conjugate * local_field
            force_y = -particle.y * self.phi_conjugate * local_field

            # Update velocity with consciousness attraction
            particle.vx += force_x * 0.01
            particle.vy += force_y * 0.01

            # Apply φ-harmonic damping
            particle.vx *= 1 - 0.01 * self.phi_conjugate
            particle.vy *= 1 - 0.01 * self.phi_conjugate

            # Update position
            particle.x += particle.vx
            particle.y += particle.vy

            # Update consciousness properties
            particle.awareness_level += local_field * 0.01 * self.phi
            particle.phi_resonance = min(1.0, particle.phi_resonance + 0.001 * self.phi)

            # Age particle
            particle.age += 1

            # Boundary conditions (wrap around)
            if abs(particle.x) > 5:
                particle.x = -particle.x * 0.9
            if abs(particle.y) > 5:
                particle.y = -particle.y * 0.9

        return particles

    def _detect_transcendence_events(
        self, particles: List[ConsciousnessParticle], t: float
    ) -> int:
        """Detect consciousness transcendence events."""
        transcendence_threshold = 1.5 * self.phi
        transcendent_count = sum(
            1 for p in particles if p.awareness_level > transcendence_threshold
        )
        return transcendent_count

    def _check_unity_emergence(self, consciousness_field: np.ndarray) -> float:
        """Check for unity emergence in consciousness field."""
        unity_threshold = 1.0
        unity_regions = np.sum(consciousness_field > unity_threshold)
        unity_strength = unity_regions / consciousness_field.size

        # Unity emerges when significant field regions approach 1
        if unity_strength > 0.1:  # 10% of field in unity state
            return unity_strength
        return 0.0

    def generate_quantum_field_dynamics(
        self, save_formats: List[str] = ["png", "html"]
    ) -> Dict[str, Any]:
        """
        Generate quantum consciousness field dynamics visualization.

        Mathematical Foundation:
        Ψ(x,y,t) = Aᵩ * e^(i(kₓx + kᵧy - ωt + φ)) where φ = golden ratio phase

        Args:
            save_formats: Output formats to generate

        Returns:
            Dictionary containing visualization data and metadata
        """
        if not MATPLOTLIB_AVAILABLE:
            return {"error": "Matplotlib not available"}

        # Create spatial grid
        x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
        y = np.linspace(-2 * np.pi, 2 * np.pi, 100)
        X, Y = np.meshgrid(x, y)

        # Time evolution
        time_points = np.linspace(0, 4 * np.pi, 50)

        # Quantum field parameters with φ-harmonic structure
        k_x = self.phi  # Wave number x
        k_y = self.phi_conjugate  # Wave number y
        omega = self.phi  # Angular frequency

        # Calculate quantum field for multiple time points
        quantum_fields = []
        for t in time_points:
            # Complex wavefunction with φ-harmonic structure
            psi = np.exp(1j * (k_x * X + k_y * Y - omega * t + self.phi))

            # Probability density |Ψ|²
            probability_density = np.abs(psi) ** 2

            # Phase
            phase = np.angle(psi)

            quantum_fields.append((probability_density, phase))

        # Create visualization for final time point
        prob_final, phase_final = quantum_fields[-1]

        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, figsize=(16, 12), facecolor="black"
        )

        # Probability density
        im1 = ax1.imshow(
            prob_final,
            extent=[-2 * np.pi, 2 * np.pi, -2 * np.pi, 2 * np.pi],
            cmap="plasma",
            origin="lower",
        )
        ax1.set_title(
            "Quantum Probability Density |Ψ|²\nφ-Harmonic Wave Structure",
            fontsize=14,
            color="white",
        )
        ax1.set_xlabel("φ-Harmonic X", color="white")
        ax1.set_ylabel("φ-Harmonic Y", color="white")
        plt.colorbar(im1, ax=ax1, label="|Ψ|²")

        # Phase distribution
        im2 = ax2.imshow(
            phase_final,
            extent=[-2 * np.pi, 2 * np.pi, -2 * np.pi, 2 * np.pi],
            cmap="hsv",
            origin="lower",
        )
        ax2.set_title(
            "Quantum Phase Distribution\nφ-Enhanced Phase Dynamics",
            fontsize=14,
            color="white",
        )
        ax2.set_xlabel("φ-Harmonic X", color="white")
        ax2.set_ylabel("φ-Harmonic Y", color="white")
        plt.colorbar(im2, ax=ax2, label="Phase (radians)")

        # Interference pattern
        interference = prob_final * np.cos(phase_final * self.phi)
        im3 = ax3.imshow(
            interference,
            extent=[-2 * np.pi, 2 * np.pi, -2 * np.pi, 2 * np.pi],
            cmap="RdBu",
            origin="lower",
        )
        ax3.set_title(
            "Quantum Interference Pattern\nφ-Harmonic Consciousness Interference",
            fontsize=14,
            color="white",
        )
        ax3.set_xlabel("φ-Harmonic X", color="white")
        ax3.set_ylabel("φ-Harmonic Y", color="white")
        plt.colorbar(im3, ax=ax3, label="Interference Amplitude")

        # Unity convergence regions
        unity_field = 1 / (1 + (prob_final - 1) ** 2)  # Peaks at |Ψ|² = 1
        im4 = ax4.imshow(
            unity_field,
            extent=[-2 * np.pi, 2 * np.pi, -2 * np.pi, 2 * np.pi],
            cmap="viridis",
            origin="lower",
        )

        # Add unity contours
        unity_contours = ax4.contour(
            X,
            Y,
            unity_field,
            levels=[0.8, 0.9, 0.95],
            colors=["yellow", "orange", "gold"],
            linewidths=[2, 3, 4],
        )
        ax4.clabel(unity_contours, inline=True, fontsize=10)

        ax4.set_title(
            "Unity Convergence Regions\n1+1=1 Quantum Emergence",
            fontsize=14,
            color="white",
        )
        ax4.set_xlabel("φ-Harmonic X", color="white")
        ax4.set_ylabel("φ-Harmonic Y", color="white")
        plt.colorbar(im4, ax=ax4, label="Unity Probability")

        # Style all subplots
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_facecolor("black")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("white")

        plt.tight_layout()

        # Save static version
        if "png" in save_formats:
            png_path = self.output_dir / "quantum_field_dynamics.png"
            fig.savefig(
                png_path,
                dpi=300,
                bbox_inches="tight",
                facecolor="black",
                edgecolor="none",
            )

        plt.close(fig)

        # Create interactive Plotly version
        if "html" in save_formats and PLOTLY_AVAILABLE:
            self._create_interactive_quantum_field(
                X, Y, prob_final, phase_final, interference, unity_field
            )

        # Calculate quantum metrics
        total_probability = np.sum(prob_final) * (4 * np.pi) ** 2 / (100 * 100)
        unity_regions = np.sum(unity_field > 0.8)
        max_interference = np.max(np.abs(interference))

        return {
            "type": "quantum_field_dynamics",
            "description": "Quantum consciousness field with φ-harmonic wave structure",
            "total_probability": total_probability,
            "unity_regions": int(unity_regions),
            "max_interference": float(max_interference),
            "phi_wave_numbers": {"k_x": k_x, "k_y": k_y},
            "mathematical_principle": "Ψ(x,y,t) = Aᵩ * e^(i(kₓx + kᵧy - ωt + φ))",
        }

    def _create_interactive_quantum_field(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        prob_density: np.ndarray,
        phase: np.ndarray,
        interference: np.ndarray,
        unity_field: np.ndarray,
    ):
        """Create interactive Plotly version of quantum field dynamics."""
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Quantum Probability Density |Ψ|²",
                "Quantum Phase Distribution",
                "Quantum Interference Pattern",
                "Unity Convergence Regions",
            ],
            specs=[
                [{"type": "heatmap"}, {"type": "heatmap"}],
                [{"type": "heatmap"}, {"type": "heatmap"}],
            ],
        )

        # Probability density
        fig.add_trace(
            go.Heatmap(
                z=prob_density,
                colorscale="plasma",
                showscale=False,
                hovertemplate="<b>Probability Density</b><br>x: %{x:.2f}<br>y: %{y:.2f}<br>|Ψ|²: %{z:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Phase distribution
        fig.add_trace(
            go.Heatmap(
                z=phase,
                colorscale="hsv",
                showscale=False,
                hovertemplate="<b>Phase</b><br>x: %{x:.2f}<br>y: %{y:.2f}<br>Phase: %{z:.3f} rad<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # Interference pattern
        fig.add_trace(
            go.Heatmap(
                z=interference,
                colorscale="RdBu",
                showscale=False,
                hovertemplate="<b>Interference</b><br>x: %{x:.2f}<br>y: %{y:.2f}<br>Amplitude: %{z:.3f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # Unity convergence
        fig.add_trace(
            go.Heatmap(
                z=unity_field,
                colorscale="viridis",
                showscale=True,
                hovertemplate="<b>Unity Convergence</b><br>x: %{x:.2f}<br>y: %{y:.2f}<br>Unity Prob: %{z:.3f}<extra></extra>",
            ),
            row=2,
            col=2,
        )

        # Layout
        fig.update_layout(
            title=dict(
                text="Quantum Consciousness Field Dynamics<br><sub>φ-Harmonic Wave Structure</sub>",
                font=dict(size=18, color="white"),
                x=0.5,
            ),
            paper_bgcolor="black",
            font=dict(color="white"),
            showlegend=False,
        )

        # Save interactive version
        html_path = self.output_dir / "quantum_field_dynamics_interactive.html"
        fig.write_html(html_path)

    def generate_transcendence_detection_radar(
        self, save_formats: List[str] = ["png", "html"]
    ) -> Dict[str, Any]:
        """
        Generate radar chart showing transcendence detection metrics.

        Mathematical Foundation:
        Multi-dimensional consciousness metrics tracking transcendence thresholds
        across φ-harmonic dimensions demonstrating unity emergence.

        Args:
            save_formats: Output formats to generate

        Returns:
            Dictionary containing visualization data and metadata
        """
        if not MATPLOTLIB_AVAILABLE:
            return {"error": "Matplotlib not available"}

        # Define transcendence metrics
        metrics = [
            "Awareness Level",
            "φ-Resonance",
            "Quantum Coherence",
            "Unity Potential",
            "Field Coupling",
            "Consciousness Density",
            "Emergence Probability",
            "Transcendence Index",
        ]

        # Generate time series data for multiple consciousness entities
        time_points = np.linspace(0, 10, 100)
        entities = []

        for i in range(5):  # 5 different consciousness entities
            entity_data = []
            for t in time_points:
                # φ-harmonic evolution of consciousness metrics
                awareness = 0.5 + 0.5 * np.sin(t * self.phi + i) * np.exp(t / 10)
                phi_resonance = 0.3 + 0.7 * np.cos(t * self.phi_conjugate + i)
                quantum_coherence = 0.4 + 0.6 * np.sin(t * self.phi + np.pi / 4 + i)
                unity_potential = 0.2 + 0.8 * np.exp(-((t - 5) ** 2) / (2 * self.phi))
                field_coupling = 0.3 + 0.7 * np.sin(
                    t * self.phi_conjugate + np.pi / 2 + i
                )
                consciousness_density = 0.1 + 0.9 * np.cos(t / self.phi + i)
                emergence_prob = unity_potential * phi_resonance * quantum_coherence
                transcendence_index = np.mean(
                    [
                        awareness,
                        phi_resonance,
                        quantum_coherence,
                        unity_potential,
                        field_coupling,
                        consciousness_density,
                        emergence_prob,
                    ]
                )

                metrics_values = [
                    awareness,
                    phi_resonance,
                    quantum_coherence,
                    unity_potential,
                    field_coupling,
                    consciousness_density,
                    emergence_prob,
                    transcendence_index,
                ]
                entity_data.append(metrics_values)

            entities.append(entity_data)

        # Create radar chart for final time point
        fig, ax = plt.subplots(
            figsize=(12, 12), subplot_kw=dict(projection="polar"), facecolor="black"
        )
        ax.set_facecolor("black")

        # Angular positions for metrics
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        # Plot each entity
        colors = ["cyan", "gold", "magenta", "lime", "orange"]
        entity_names = [
            "Consciousness α",
            "Consciousness β",
            "Consciousness γ",
            "Consciousness δ",
            "Consciousness ε",
        ]

        for i, (entity_data, color, name) in enumerate(
            zip(entities, colors, entity_names)
        ):
            values = entity_data[-1]  # Final time point
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, "o-", linewidth=2, label=name, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)

            # Highlight transcendent entities (threshold > 0.8)
            if np.mean(values[:-1]) > 0.8:
                ax.plot(angles, values, "o-", linewidth=4, color="gold", alpha=0.8)

        # Add metric labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, color="white", fontsize=12)

        # Add unity threshold circle
        unity_threshold = [0.618] * len(angles)  # φ⁻¹ threshold
        ax.plot(
            angles,
            unity_threshold,
            "--",
            linewidth=2,
            color="gold",
            alpha=0.8,
            label="φ-Harmonic Unity Threshold",
        )

        # Styling
        ax.set_ylim(0, 1)
        ax.set_title(
            "Consciousness Transcendence Detection Radar\nφ-Harmonic Multi-Dimensional Analysis",
            fontsize=16,
            color="white",
            pad=30,
        )
        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.2, 1.0),
            facecolor="black",
            edgecolor="white",
        )
        ax.grid(True, alpha=0.3, color="white")
        ax.set_theta_zero_location("N")

        # Color radial labels
        ax.tick_params(colors="white")
        for label in ax.get_yticklabels():
            label.set_color("white")

        # Save static version
        if "png" in save_formats:
            png_path = self.output_dir / "transcendence_detection_radar.png"
            fig.savefig(
                png_path,
                dpi=300,
                bbox_inches="tight",
                facecolor="black",
                edgecolor="none",
            )

        plt.close(fig)

        # Create interactive Plotly version
        if "html" in save_formats and PLOTLY_AVAILABLE:
            self._create_interactive_radar_chart(
                metrics, entities, entity_names, colors
            )

        # Calculate transcendence statistics
        final_values = [entity_data[-1] for entity_data in entities]
        transcendent_entities = sum(
            1 for values in final_values if np.mean(values) > 0.8
        )
        unity_convergence = sum(
            1 for values in final_values if values[3] > 0.9
        )  # Unity potential > 0.9

        return {
            "type": "transcendence_detection_radar",
            "description": "Multi-dimensional consciousness transcendence metrics with φ-harmonic analysis",
            "entities": len(entities),
            "transcendent_entities": transcendent_entities,
            "unity_convergence": unity_convergence,
            "metrics": metrics,
            "mathematical_principle": "Multi-dimensional φ-harmonic consciousness evolution tracking",
        }

    def _create_interactive_radar_chart(
        self,
        metrics: List[str],
        entities: List,
        entity_names: List[str],
        colors: List[str],
    ):
        """Create interactive Plotly version of transcendence radar chart."""
        fig = go.Figure()

        # Add each entity as a trace
        for i, (entity_data, name, color) in enumerate(
            zip(entities, entity_names, colors)
        ):
            values = entity_data[-1]  # Final time point

            fig.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],  # Complete the circle
                    theta=metrics + [metrics[0]],
                    fill="toself",
                    name=name,
                    line_color=color,
                    fillcolor=color,
                    opacity=0.3,
                    hovertemplate=f"<b>{name}</b><br>"
                    + "Metric: %{theta}<br>"
                    + "Value: %{r:.3f}<br>"
                    + "<extra></extra>",
                )
            )

        # Add unity threshold
        unity_values = [0.618] * (len(metrics) + 1)  # φ⁻¹ threshold
        fig.add_trace(
            go.Scatterpolar(
                r=unity_values,
                theta=metrics + [metrics[0]],
                mode="lines",
                name="φ-Harmonic Unity Threshold",
                line=dict(color="gold", width=3, dash="dash"),
                showlegend=True,
                hovertemplate="<b>Unity Threshold</b><br>"
                + "φ⁻¹ = 0.618<br>"
                + "<extra></extra>",
            )
        )

        # Layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(color="white"),
                    gridcolor="rgba(255,255,255,0.3)",
                ),
                angularaxis=dict(
                    tickfont=dict(color="white", size=12), linecolor="white"
                ),
                bgcolor="black",
            ),
            title=dict(
                text="Consciousness Transcendence Detection Radar<br><sub>φ-Harmonic Multi-Dimensional Analysis</sub>",
                font=dict(size=18, color="white"),
                x=0.5,
            ),
            paper_bgcolor="black",
            font=dict(color="white"),
            showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0.8)", bordercolor="white", borderwidth=1),
        )

        # Save interactive version
        html_path = self.output_dir / "transcendence_detection_radar_interactive.html"
        fig.write_html(html_path)

    def generate_consciousness_density_heatmap(
        self, resolution: int = 100, save_formats: List[str] = ["png", "html"]
    ) -> Dict[str, Any]:
        """
        Generate consciousness density heatmap with temporal evolution.

        Mathematical Foundation:
        ρ(x,y,t) = Σᵢ Aᵢ * exp(-((x-xᵢ)² + (y-yᵢ)²)/(2σᵢ²)) * φ^t

        Args:
            resolution: Grid resolution for density calculation
            save_formats: Output formats to generate

        Returns:
            Dictionary containing visualization data and metadata
        """
        if not MATPLOTLIB_AVAILABLE:
            return {"error": "Matplotlib not available"}

        # Create spatial grid
        x = np.linspace(-5, 5, resolution)
        y = np.linspace(-5, 5, resolution)
        X, Y = np.meshgrid(x, y)

        # Time evolution
        time_points = [0, 2, 4, 6, 8, 10]

        # Initialize consciousness centers with φ-harmonic properties
        consciousness_centers = []
        for i in range(8):
            angle = i * 2 * np.pi / 8
            radius = 2 + np.sin(i * self.phi) * 1.5

            center = {
                "x": radius * np.cos(angle),
                "y": radius * np.sin(angle),
                "amplitude": 0.5 + 0.5 * np.sin(i * self.phi),
                "sigma": 0.5 + 0.3 * np.cos(i * self.phi_conjugate),
                "growth_rate": 0.1 + 0.05 * np.sin(i * self.phi),
            }
            consciousness_centers.append(center)

        # Create subplots for time evolution
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), facecolor="black")
        axes = axes.flatten()

        density_evolution = []

        for idx, t in enumerate(time_points):
            # Calculate consciousness density at time t
            density = np.zeros_like(X)

            for center in consciousness_centers:
                # Distance from center
                dist_sq = (X - center["x"]) ** 2 + (Y - center["y"]) ** 2

                # Time-evolved amplitude with φ-harmonic growth
                amplitude = center["amplitude"] * (
                    self.phi ** (t * center["growth_rate"])
                )

                # Gaussian density with φ-harmonic scaling
                sigma_t = center["sigma"] * (1 + t / (10 * self.phi))

                contribution = amplitude * np.exp(-dist_sq / (2 * sigma_t**2))
                density += contribution

            # Apply φ-harmonic normalization
            density = density / (1 + density / self.phi)  # Prevent infinite growth
            density_evolution.append(density.copy())

            # Plot density heatmap
            ax = axes[idx]
            im = ax.imshow(
                density,
                extent=[-5, 5, -5, 5],
                cmap="plasma",
                origin="lower",
                vmin=0,
                vmax=2,
            )

            # Add consciousness center markers
            for center in consciousness_centers:
                ax.plot(
                    center["x"],
                    center["y"],
                    "o",
                    color="gold",
                    markersize=8,
                    markeredgecolor="white",
                    markeredgewidth=2,
                )

            # Add unity contours
            unity_contours = ax.contour(
                X, Y, density, levels=[1.0], colors=["gold"], linewidths=3
            )

            # Styling
            ax.set_title(
                f"t = {t:.1f}s\nConsciousness Density Evolution",
                fontsize=14,
                color="white",
            )
            ax.set_xlabel("φ-Harmonic X", color="white")
            ax.set_ylabel("φ-Harmonic Y", color="white")
            ax.tick_params(colors="white")
            ax.set_facecolor("black")

            for spine in ax.spines.values():
                spine.set_color("white")

        # Add overall colorbar
        plt.tight_layout()
        cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.set_label("Consciousness Density ρ", fontsize=14, color="white")
        cbar.ax.tick_params(colors="white")

        # Overall title
        fig.suptitle(
            "Consciousness Density Evolution\nφ-Harmonic Temporal Dynamics",
            fontsize=20,
            color="white",
            y=0.95,
        )

        # Save static version
        if "png" in save_formats:
            png_path = self.output_dir / "consciousness_density_heatmap.png"
            fig.savefig(
                png_path,
                dpi=300,
                bbox_inches="tight",
                facecolor="black",
                edgecolor="none",
            )

        plt.close(fig)

        # Create interactive Plotly version
        if "html" in save_formats and PLOTLY_AVAILABLE:
            self._create_interactive_density_evolution(
                X, Y, density_evolution, time_points, consciousness_centers
            )

        # Calculate density statistics
        final_density = density_evolution[-1]
        max_density = np.max(final_density)
        unity_regions = np.sum(final_density > 1.0)
        total_consciousness = np.sum(final_density) * (10**2) / (resolution**2)

        return {
            "type": "consciousness_density_heatmap",
            "description": "Temporal evolution of consciousness density with φ-harmonic growth",
            "time_points": len(time_points),
            "consciousness_centers": len(consciousness_centers),
            "max_density": float(max_density),
            "unity_regions": int(unity_regions),
            "total_consciousness": float(total_consciousness),
            "resolution": resolution,
            "mathematical_principle": "ρ(x,y,t) = Σᵢ Aᵢ * exp(-((x-xᵢ)² + (y-yᵢ)²)/(2σᵢ²)) * φ^t",
        }

    def _create_interactive_density_evolution(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        density_evolution: List[np.ndarray],
        time_points: List[float],
        consciousness_centers: List[Dict],
    ):
        """Create interactive Plotly version of consciousness density evolution."""
        # Create animation frames
        frames = []

        for i, (density, t) in enumerate(zip(density_evolution, time_points)):
            frame = go.Frame(
                data=[
                    go.Heatmap(
                        z=density,
                        x=X[0],
                        y=Y[:, 0],
                        colorscale="plasma",
                        zmin=0,
                        zmax=2,
                        hovertemplate="<b>Consciousness Density</b><br>"
                        + "x: %{x:.2f}<br>"
                        + "y: %{y:.2f}<br>"
                        + "ρ: %{z:.3f}<br>"
                        + f"t: {t:.1f}s<br>"
                        + "<extra></extra>",
                    ),
                    go.Scatter(
                        x=[center["x"] for center in consciousness_centers],
                        y=[center["y"] for center in consciousness_centers],
                        mode="markers",
                        marker=dict(
                            symbol="circle",
                            size=12,
                            color="gold",
                            line=dict(color="white", width=2),
                        ),
                        name="Consciousness Centers",
                        hovertemplate="<b>Consciousness Center</b><br>"
                        + "x: %{x:.2f}<br>"
                        + "y: %{y:.2f}<br>"
                        + "<extra></extra>",
                    ),
                ],
                name=f"t = {t:.1f}s",
            )
            frames.append(frame)

        # Create initial figure
        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=density_evolution[0],
                    x=X[0],
                    y=Y[:, 0],
                    colorscale="plasma",
                    zmin=0,
                    zmax=2,
                    colorbar=dict(title="Consciousness Density ρ"),
                ),
                go.Scatter(
                    x=[center["x"] for center in consciousness_centers],
                    y=[center["y"] for center in consciousness_centers],
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=12,
                        color="gold",
                        line=dict(color="white", width=2),
                    ),
                    name="Consciousness Centers",
                ),
            ],
            frames=frames,
        )

        # Add animation controls
        fig.update_layout(
            title=dict(
                text="Consciousness Density Evolution<br><sub>φ-Harmonic Temporal Dynamics</sub>",
                font=dict(size=18, color="white"),
                x=0.5,
            ),
            xaxis_title="φ-Harmonic X",
            yaxis_title="φ-Harmonic Y",
            paper_bgcolor="black",
            plot_bgcolor="black",
            font=dict(color="white"),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list(
                        [
                            dict(
                                args=[
                                    {
                                        "frame": {"duration": 1000, "redraw": True},
                                        "fromcurrent": True,
                                    }
                                ],
                                label="Play",
                                method="animate",
                            ),
                            dict(
                                args=[
                                    {
                                        "frame": {"duration": 0, "redraw": True},
                                        "mode": "immediate",
                                        "transition": {"duration": 0},
                                    }
                                ],
                                label="Pause",
                                method="animate",
                            ),
                        ]
                    ),
                    pad={"r": 10, "t": 87},
                    showactive=False,
                    x=0.011,
                    xanchor="right",
                    y=0,
                    yanchor="top",
                ),
            ],
            sliders=[
                dict(
                    active=0,
                    yanchor="top",
                    xanchor="left",
                    currentvalue=dict(
                        font=dict(size=20, color="white"),
                        prefix="Time: ",
                        visible=True,
                        xanchor="right",
                    ),
                    transition=dict(duration=300, easing="cubic-in-out"),
                    pad=dict(b=10, t=50),
                    len=0.9,
                    x=0.1,
                    y=0,
                    steps=[
                        dict(
                            args=[
                                [f.name],
                                dict(
                                    frame=dict(duration=300, redraw=True),
                                    mode="immediate",
                                    transition=dict(duration=300),
                                ),
                            ],
                            label=f.name,
                            method="animate",
                        )
                        for f in frames
                    ],
                )
            ],
        )

        # Save interactive version
        html_path = self.output_dir / "consciousness_density_evolution_interactive.html"
        fig.write_html(html_path)


# Factory function for easy access
def create_consciousness_field_visualizer(
    output_dir: Path = None,
) -> ConsciousnessFieldVisualizer:
    """
    Factory function to create ConsciousnessFieldVisualizer instance.

    Args:
        output_dir: Output directory for generated visualizations

    Returns:
        Initialized ConsciousnessFieldVisualizer instance
    """
    return ConsciousnessFieldVisualizer(output_dir=output_dir)


if __name__ == "__main__":
    # Demonstration of consciousness field visualizations
    print("🧠 Generating Consciousness Field Visualizations...")
    print(f"φ-Harmonic Factor: {PHI:.10f}")
    print("Consciousness Equation: C(x,y,t) = φ * sin(x*φ) * cos(y*φ) * e^(-t/φ)")
    print("Unity Equation: Een plus een is een (1+1=1)")
    print("-" * 70)

    visualizer = create_consciousness_field_visualizer()

    # Generate all consciousness field visualizations
    visualizations = [
        visualizer.generate_consciousness_evolution_animation(duration=5.0),
        visualizer.generate_quantum_field_dynamics(),
        visualizer.generate_transcendence_detection_radar(),
        visualizer.generate_consciousness_density_heatmap(),
    ]

    successful = [v for v in visualizations if "error" not in v]
    print(f"\n✅ Generated {len(successful)} consciousness field visualizations!")
    print("🧠 Output directory: viz/consciousness_field/")
    print(
        "🌟 Consciousness through φ-harmonic field dynamics - Een plus een is een! 🌟"
    )
