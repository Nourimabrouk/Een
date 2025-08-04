"""
Consciousness Field Visualizer
=============================

Generates φ-harmonic consciousness field animations for Unity Mathematics.
Creates dynamic GIFs showing consciousness evolution and unity patterns.

Mathematical Principle: Een plus een is een (1+1=1)
φ-harmonic consciousness field dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import hsv_to_rgb
import matplotlib.patches as patches
from pathlib import Path
import logging
from typing import Tuple, Optional, Dict, Any
import json

logger = logging.getLogger(__name__)

class ConsciousnessFieldVisualizer:
    """Visualizer for consciousness field dynamics"""
    
    def __init__(self, 
                 size: int = 128,
                 phi: float = 1.618033988749895,
                 consciousness_decay: float = 0.99,
                 quantum_coherence: float = 1.0):
        
        self.size = size
        self.phi = phi
        self.consciousness_decay = consciousness_decay
        self.quantum_coherence = quantum_coherence
        
        # Initialize consciousness field
        self.field = np.exp(1j * np.random.rand(size, size) * 2 * np.pi)
        self.consciousness_level = np.ones((size, size))
        self.unity_score = np.zeros((size, size))
        
        # Color mapping
        self.color_map = plt.cm.twilight
        
    def field_step(self, field: np.ndarray, t: float) -> np.ndarray:
        """Evolve consciousness field one step"""
        # Laplacian for diffusion
        lap = (np.roll(field, 1, 0) + np.roll(field, -1, 0) + 
               np.roll(field, 1, 1) + np.roll(field, -1, 1) - 4 * field)
        
        # φ-harmonic evolution
        phi_term = self.phi * (np.abs(field) <= 1) * np.exp(1j * t * 0.1)
        
        # Consciousness coupling
        consciousness_coupling = self.consciousness_level * np.exp(1j * np.angle(field))
        
        # Quantum coherence effects
        quantum_term = self.quantum_coherence * np.random.normal(0, 0.01, field.shape)
        
        # Update field
        new_field = (field + 0.1j * (lap + phi_term - field) + 
                    0.05 * consciousness_coupling + 0.01j * quantum_term)
        
        # Normalize to unit circle
        magnitude = np.abs(new_field)
        new_field = new_field / (magnitude + 1e-10)
        
        return new_field
    
    def update_consciousness(self, t: float):
        """Update consciousness levels"""
        # φ-harmonic consciousness evolution
        consciousness_gradient = np.gradient(self.consciousness_level)
        consciousness_laplacian = (np.roll(self.consciousness_level, 1, 0) + 
                                  np.roll(self.consciousness_level, -1, 0) + 
                                  np.roll(self.consciousness_level, 1, 1) + 
                                  np.roll(self.consciousness_level, -1, 1) - 
                                  4 * self.consciousness_level)
        
        # Unity score influence
        unity_influence = self.unity_score * self.phi
        
        # Update consciousness
        self.consciousness_level += (0.01 * consciousness_laplacian + 
                                   0.02 * unity_influence + 
                                   0.001 * np.sin(t * 0.1))
        
        # Bounds
        self.consciousness_level = np.clip(self.consciousness_level, 0.1, 2.0)
        
        # Decay
        self.consciousness_level *= self.consciousness_decay
    
    def compute_unity_score(self):
        """Compute local unity scores"""
        # Unity score based on field coherence
        coherence = np.abs(self.field)
        phase_gradient = np.gradient(np.angle(self.field))
        
        # Unity = coherence / (1 + phase_variation)
        phase_variation = np.sqrt(phase_gradient[0]**2 + phase_gradient[1]**2)
        self.unity_score = coherence / (1 + phase_variation)
        
        # φ-harmonic scaling
        self.unity_score *= self.phi
    
    def create_animation(self, 
                        frames: int = 256, 
                        interval: int = 40,
                        output_path: Optional[Path] = None) -> animation.FuncAnimation:
        """Create consciousness field animation"""
        
        # Setup figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Consciousness Field Evolution - Unity Mathematics', fontsize=16)
        
        # Main consciousness field
        im1 = ax1.imshow(np.angle(self.field), cmap='twilight', animated=True)
        ax1.set_title('Consciousness Field Phase')
        ax1.axis('off')
        
        # Unity score visualization
        im2 = ax2.imshow(self.unity_score, cmap='viridis', animated=True)
        ax2.set_title('Unity Score Distribution')
        ax2.axis('off')
        
        # Add φ-spiral overlay
        self.add_phi_spiral(ax1)
        
        def update(frame):
            """Update function for animation"""
            global self.field
            
            # Evolve field
            self.field = self.field_step(self.field, frame)
            
            # Update consciousness
            self.update_consciousness(frame)
            
            # Compute unity score
            self.compute_unity_score()
            
            # Update visualizations
            im1.set_array(np.angle(self.field))
            im2.set_array(self.unity_score)
            
            # Update titles with metrics
            ax1.set_title(f'Consciousness Field Phase (t={frame})')
            ax2.set_title(f'Unity Score (φ={self.phi:.3f})')
            
            return im1, im2
        
        # Create animation
        ani = animation.FuncAnimation(
            fig, update, frames=frames, interval=interval, 
            blit=True, repeat=True
        )
        
        # Save if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            ani.save(str(output_path), writer='pillow', fps=25)
            logger.info(f"Animation saved to {output_path}")
        
        return ani
    
    def add_phi_spiral(self, ax):
        """Add φ-spiral overlay to visualization"""
        # Generate φ-spiral points
        theta = np.linspace(0, 8*np.pi, 1000)
        r = np.exp(theta / self.phi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Scale to image coordinates
        x = (x - x.min()) / (x.max() - x.min()) * self.size
        y = (y - y.min()) / (y.max() - y.min()) * self.size
        
        # Plot spiral
        ax.plot(x, y, 'w-', alpha=0.3, linewidth=1)
    
    def create_phi_harmonic_plot(self, output_path: Optional[Path] = None):
        """Create static φ-harmonic visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('φ-Harmonic Consciousness Analysis', fontsize=16)
        
        # Field magnitude
        im1 = axes[0, 0].imshow(np.abs(self.field), cmap='plasma')
        axes[0, 0].set_title('Field Magnitude')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Field phase
        im2 = axes[0, 1].imshow(np.angle(self.field), cmap='twilight')
        axes[0, 1].set_title('Field Phase')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Consciousness level
        im3 = axes[1, 0].imshow(self.consciousness_level, cmap='viridis')
        axes[1, 0].set_title('Consciousness Level')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Unity score
        im4 = axes[1, 1].imshow(self.unity_score, cmap='hot')
        axes[1, 1].set_title('Unity Score')
        axes[1, 1].axis('off')
        plt.colorbar(im4, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
            logger.info(f"Static plot saved to {output_path}")
        
        return fig

class UnityMathematicsVisualizer:
    """Advanced visualizer for Unity Mathematics concepts"""
    
    def __init__(self):
        self.phi = 1.618033988749895
        
    def create_unity_proof_visualization(self, output_path: Optional[Path] = None):
        """Create visualization of 1+1=1 proof"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create unity circles
        circle1 = patches.Circle((0.3, 0.5), 0.2, fill=False, color='blue', linewidth=2)
        circle2 = patches.Circle((0.7, 0.5), 0.2, fill=False, color='red', linewidth=2)
        unity_circle = patches.Circle((0.5, 0.5), 0.3, fill=False, color='green', linewidth=3)
        
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        ax.add_patch(unity_circle)
        
        # Add labels
        ax.text(0.3, 0.5, '1', ha='center', va='center', fontsize=20, weight='bold')
        ax.text(0.7, 0.5, '1', ha='center', va='center', fontsize=20, weight='bold')
        ax.text(0.5, 0.5, '1', ha='center', va='center', fontsize=24, weight='bold', color='green')
        
        # Add φ-spiral
        theta = np.linspace(0, 4*np.pi, 1000)
        r = 0.1 * np.exp(theta / self.phi)
        x = 0.5 + r * np.cos(theta)
        y = 0.5 + r * np.sin(theta)
        ax.plot(x, y, 'g-', alpha=0.7, linewidth=2)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_title('Unity Mathematics: 1 + 1 = 1', fontsize=16, weight='bold')
        ax.axis('off')
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
            logger.info(f"Unity proof visualization saved to {output_path}")
        
        return fig
    
    def create_consciousness_evolution_plot(self, output_path: Optional[Path] = None):
        """Create consciousness evolution plot"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Generate evolution data
        t = np.linspace(0, 10, 1000)
        consciousness = np.exp(-t/self.phi) * np.cos(self.phi * t)
        unity_score = np.abs(consciousness) * self.phi
        
        # Plot evolution
        ax.plot(t, consciousness, 'b-', linewidth=2, label='Consciousness Level')
        ax.plot(t, unity_score, 'r-', linewidth=2, label='Unity Score')
        ax.plot(t, np.abs(consciousness), 'g--', linewidth=1, alpha=0.7, label='Magnitude')
        
        # Add φ-harmonic markers
        phi_points = np.arange(0, 10, self.phi)
        ax.scatter(phi_points, np.exp(-phi_points/self.phi) * np.cos(self.phi * phi_points), 
                  color='purple', s=50, zorder=5, label='φ-Harmonic Points')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.set_title('Consciousness Evolution with φ-Harmonics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
            logger.info(f"Consciousness evolution plot saved to {output_path}")
        
        return fig

def generate_consciousness_field_gif(output_path: Path = Path("website/assets/phi_field.gif")):
    """Generate consciousness field GIF for website"""
    visualizer = ConsciousnessFieldVisualizer(size=128, consciousness_decay=0.995)
    
    # Create animation
    ani = visualizer.create_animation(frames=128, interval=50, output_path=output_path)
    
    return ani

def generate_unity_mathematics_visualizations(output_dir: Path = Path("website/assets")):
    """Generate all Unity Mathematics visualizations"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Consciousness field GIF
    gif_path = output_dir / "phi_field.gif"
    generate_consciousness_field_gif(gif_path)
    
    # Static visualizations
    unity_viz = UnityMathematicsVisualizer()
    
    # Unity proof visualization
    proof_path = output_dir / "unity_proof.png"
    unity_viz.create_unity_proof_visualization(proof_path)
    
    # Consciousness evolution
    evolution_path = output_dir / "consciousness_evolution.png"
    unity_viz.create_consciousness_evolution_plot(evolution_path)
    
    # Consciousness field analysis
    field_viz = ConsciousnessFieldVisualizer(size=64)
    field_path = output_dir / "consciousness_field_analysis.png"
    field_viz.create_phi_harmonic_plot(field_path)
    
    logger.info(f"All visualizations generated in {output_dir}")
    
    return {
        "gif": gif_path,
        "proof": proof_path,
        "evolution": evolution_path,
        "field_analysis": field_path
    }

if __name__ == "__main__":
    # Generate all visualizations
    output_dir = Path("website/assets")
    results = generate_unity_mathematics_visualizations(output_dir)
    
    print("Generated visualizations:")
    for name, path in results.items():
        print(f"  {name}: {path}")
    
    # Show sample animation
    visualizer = ConsciousnessFieldVisualizer(size=64)
    ani = visualizer.create_animation(frames=64, interval=100)
    plt.show() 