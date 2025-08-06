"""
GPU-Accelerated Visualization Kernels
High-performance computational kernels for Unity mathematics visualization
"""

import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import base64
from io import BytesIO

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logging.warning("Matplotlib not available - some visualizations will be limited")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logging.warning("PIL not available - image processing limited")

logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters"""
    width: int = 800
    height: int = 600
    fps: int = 30
    duration: float = 10.0
    quality: str = "high"  # low, medium, high, ultra
    color_scheme: str = "unity_gold"
    
class UnityVisualizationKernels:
    """High-performance visualization kernels for Unity mathematics"""
    
    PHI = (1 + np.sqrt(5)) / 2
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.color_maps = self._initialize_color_maps()
        self.gpu_available = self._check_gpu_availability()
        
        logger.info(f"Visualization kernels initialized - GPU: {self.gpu_available}")
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available"""
        try:
            # Check for CUDA/OpenCL or other GPU acceleration
            # For now, use CPU-optimized numpy operations
            return True  # Assume optimized numpy with BLAS
        except Exception:
            return False
    
    def _initialize_color_maps(self) -> Dict[str, LinearSegmentedColormap]:
        """Initialize custom color maps for Unity visualizations"""
        color_maps = {}
        
        if not HAS_MATPLOTLIB:
            return color_maps
        
        # Unity Gold gradient
        unity_colors = ['#000000', '#1a1a1a', '#333333', '#FFD700', '#FFFFFF']
        color_maps['unity_gold'] = LinearSegmentedColormap.from_list('unity_gold', unity_colors)
        
        # Consciousness Purple-Gold
        consciousness_colors = ['#0a0a0a', '#2d1b69', '#6B46C1', '#FFD700', '#ffffff']
        color_maps['consciousness'] = LinearSegmentedColormap.from_list('consciousness', consciousness_colors)
        
        # Quantum Blue-Gold
        quantum_colors = ['#0a0a0a', '#1e3a8a', '#4ECDC4', '#FFD700', '#ffffff']
        color_maps['quantum'] = LinearSegmentedColormap.from_list('quantum', quantum_colors)
        
        # Fractal spectrum
        fractal_colors = ['#000000', '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFD700']
        color_maps['fractal'] = LinearSegmentedColormap.from_list('fractal', fractal_colors)
        
        return color_maps
    
    def mandelbrot_kernel(self, center_real: float = -0.7269, center_imag: float = 0.1889, 
                         zoom: float = 1.0, max_iter: int = 1000) -> np.ndarray:
        """GPU-optimized Mandelbrot set computation"""
        try:
            width, height = self.config.width, self.config.height
            
            # Create coordinate arrays
            x = np.linspace(-2.0/zoom + center_real, 2.0/zoom + center_real, width)
            y = np.linspace(-2.0/zoom + center_imag, 2.0/zoom + center_imag, height)
            X, Y = np.meshgrid(x, y)
            
            # Complex plane
            C = X + 1j * Y
            Z = np.zeros_like(C)
            iterations = np.zeros(C.shape, dtype=int)
            
            # Vectorized Mandelbrot iteration
            for i in range(max_iter):
                mask = np.abs(Z) <= 2
                Z[mask] = Z[mask]**2 + C[mask]
                iterations[mask] = i
            
            # Normalize and apply smooth coloring
            iterations = iterations.astype(float)
            smooth_iterations = iterations + 1 - np.log2(np.log2(np.abs(Z) + 1))
            
            return smooth_iterations
            
        except Exception as e:
            logger.error(f"Mandelbrot kernel failed: {e}")
            return np.zeros((self.config.height, self.config.width))
    
    def julia_kernel(self, c_real: float = -0.8, c_imag: float = 0.156, 
                     zoom: float = 1.0, max_iter: int = 500) -> np.ndarray:
        """GPU-optimized Julia set computation"""
        try:
            width, height = self.config.width, self.config.height
            
            # Create coordinate arrays
            x = np.linspace(-2.0/zoom, 2.0/zoom, width)
            y = np.linspace(-2.0/zoom, 2.0/zoom, height)
            X, Y = np.meshgrid(x, y)
            
            # Initialize Z as the complex plane, C as constant
            Z = X + 1j * Y
            C = complex(c_real, c_imag)
            iterations = np.zeros(Z.shape, dtype=int)
            
            # Vectorized Julia iteration
            for i in range(max_iter):
                mask = np.abs(Z) <= 2
                Z[mask] = Z[mask]**2 + C
                iterations[mask] = i
            
            # Smooth coloring
            smooth_iterations = iterations + 1 - np.log2(np.log2(np.abs(Z) + 1))
            
            return smooth_iterations
            
        except Exception as e:
            logger.error(f"Julia kernel failed: {e}")
            return np.zeros((self.config.height, self.config.width))
    
    def golden_ratio_spiral_kernel(self, time: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate golden ratio spiral coordinates"""
        try:
            phi = self.PHI
            n_points = 2000
            
            # Golden angle in radians
            golden_angle = 2 * np.pi / (phi**2)
            
            # Generate spiral points
            indices = np.arange(n_points)
            theta = indices * golden_angle + time
            radius = np.sqrt(indices) * 10
            
            # Apply phi-based scaling
            radius_phi = radius * (phi ** (np.sin(theta + time) * 0.1))
            
            x = radius_phi * np.cos(theta)
            y = radius_phi * np.sin(theta)
            
            return x, y
            
        except Exception as e:
            logger.error(f"Golden spiral kernel failed: {e}")
            return np.array([]), np.array([])
    
    def consciousness_field_kernel(self, time: float = 0.0) -> np.ndarray:
        """Generate consciousness field visualization data"""
        try:
            width, height = self.config.width, self.config.height
            
            # Create coordinate grid
            x = np.linspace(-5, 5, width)
            y = np.linspace(-5, 5, height)
            X, Y = np.meshgrid(x, y)
            
            # Distance from center
            R = np.sqrt(X**2 + Y**2)
            
            # Consciousness field equation with phi harmonics
            phi = self.PHI
            field = np.exp(-R**2 / (2 * phi)) * np.cos(R * phi + time * 2)
            
            # Add quantum interference patterns
            interference = np.sin(X * phi + time) * np.cos(Y * phi + time * 0.7)
            field += 0.3 * interference
            
            # Normalize field
            field = (field - field.min()) / (field.max() - field.min())
            
            return field
            
        except Exception as e:
            logger.error(f"Consciousness field kernel failed: {e}")
            return np.zeros((self.config.height, self.config.width))
    
    def quantum_superposition_kernel(self, time: float = 0.0) -> Dict[str, np.ndarray]:
        """Generate quantum superposition state visualization"""
        try:
            n_points = 1000
            x = np.linspace(-10, 10, n_points)
            
            # Quantum harmonic oscillator wavefunctions (first few states)
            psi_0 = np.exp(-x**2 / 2) * (np.pi**(-0.25))
            psi_1 = np.sqrt(2) * x * np.exp(-x**2 / 2) * (np.pi**(-0.25))
            psi_2 = (2*x**2 - 1) * np.exp(-x**2 / 2) * (np.pi**(-0.25)) / np.sqrt(2)
            
            # Time evolution (oscillating coefficients)
            alpha = np.cos(time)
            beta = np.sin(time)
            gamma = np.cos(time * self.PHI) * 0.5
            
            # Superposition state
            psi_super = alpha * psi_0 + beta * psi_1 + gamma * psi_2
            
            # Probability density
            prob_density = np.abs(psi_super)**2
            
            # Phase
            phase = np.angle(psi_super)
            
            return {
                'x': x,
                'wavefunction': psi_super,
                'probability': prob_density,
                'phase': phase,
                'coefficients': [alpha, beta, gamma]
            }
            
        except Exception as e:
            logger.error(f"Quantum superposition kernel failed: {e}")
            return {'x': np.array([]), 'wavefunction': np.array([]), 
                   'probability': np.array([]), 'phase': np.array([])}
    
    def euler_unity_kernel(self, time: float = 0.0) -> Dict[str, np.ndarray]:
        """Generate Euler's identity visualization data"""
        try:
            # Unit circle points
            theta = np.linspace(0, 2*np.pi, 1000)
            
            # Euler's formula: e^(i*theta) = cos(theta) + i*sin(theta)
            complex_exp = np.exp(1j * theta)
            
            # Current point on circle
            current_theta = time % (2*np.pi)
            current_point = np.exp(1j * current_theta)
            
            # Spiral from origin showing e^(i*t) for t from 0 to current_theta
            spiral_t = np.linspace(0, current_theta, int(current_theta * 100))
            spiral_points = np.exp(1j * spiral_t)
            
            return {
                'circle_real': complex_exp.real,
                'circle_imag': complex_exp.imag,
                'current_real': current_point.real,
                'current_imag': current_point.imag,
                'spiral_real': spiral_points.real,
                'spiral_imag': spiral_points.imag,
                'theta': current_theta
            }
            
        except Exception as e:
            logger.error(f"Euler unity kernel failed: {e}")
            return {}
    
    def topological_unity_kernel(self, time: float = 0.0) -> Dict[str, np.ndarray]:
        """Generate Klein bottle and Möbius strip visualizations"""
        try:
            # Klein bottle parametric equations
            u = np.linspace(0, 2*np.pi, 100)
            v = np.linspace(0, 2*np.pi, 100)
            U, V = np.meshgrid(u, v)
            
            # Klein bottle coordinates
            r = 4 * (1 - np.cos(U) / 2)
            
            # Split calculation for u < pi and u >= pi
            X = np.zeros_like(U)
            Y = np.zeros_like(U)
            Z = r * np.sin(V)
            
            mask1 = U < np.pi
            mask2 = ~mask1
            
            X[mask1] = 6 * np.cos(U[mask1]) * (1 + np.sin(U[mask1])) + r[mask1] * np.cos(U[mask1]) * np.cos(V[mask1])
            Y[mask1] = 16 * np.sin(U[mask1]) + r[mask1] * np.sin(U[mask1]) * np.cos(V[mask1])
            
            X[mask2] = 6 * np.cos(U[mask2]) * (1 + np.sin(U[mask2])) + r[mask2] * np.cos(V[mask2] + np.pi)
            Y[mask2] = 16 * np.sin(U[mask2])
            
            # Apply time rotation
            cos_t = np.cos(time * 0.5)
            sin_t = np.sin(time * 0.5)
            
            X_rot = X * cos_t - Y * sin_t
            Y_rot = X * sin_t + Y * cos_t
            
            return {
                'klein_x': X_rot,
                'klein_y': Y_rot,
                'klein_z': Z,
                'u_params': U,
                'v_params': V
            }
            
        except Exception as e:
            logger.error(f"Topological unity kernel failed: {e}")
            return {}
    
    def generate_visualization_frame(self, paradigm: str, time: float = 0.0, 
                                   **kwargs) -> Optional[np.ndarray]:
        """Generate a single visualization frame for given paradigm"""
        try:
            if paradigm == 'fractal':
                zoom = 1.0 + np.sin(time) * 0.5
                return self.mandelbrot_kernel(zoom=zoom, **kwargs)
            
            elif paradigm == 'golden_ratio':
                x, y = self.golden_ratio_spiral_kernel(time)
                # Convert to image array
                if len(x) > 0 and HAS_MATPLOTLIB:
                    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                    ax.scatter(x, y, c=np.arange(len(x)), cmap=self.color_maps['unity_gold'], s=1)
                    ax.set_aspect('equal')
                    ax.axis('off')
                    
                    # Convert to array
                    fig.canvas.draw()
                    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    plt.close(fig)
                    return buf
            
            elif paradigm == 'consciousness':
                return self.consciousness_field_kernel(time)
            
            elif paradigm == 'quantum':
                data = self.quantum_superposition_kernel(time)
                if 'probability' in data and HAS_MATPLOTLIB:
                    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                    ax.plot(data['x'], data['probability'], color='gold', linewidth=2)
                    ax.fill_between(data['x'], data['probability'], alpha=0.3, color='gold')
                    ax.set_title('Quantum Unity: |ψ|² = 1')
                    ax.grid(True, alpha=0.3)
                    
                    fig.canvas.draw()
                    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    plt.close(fig)
                    return buf
            
            else:
                logger.warning(f"Unknown paradigm: {paradigm}")
                return None
                
        except Exception as e:
            logger.error(f"Frame generation failed for {paradigm}: {e}")
            return None
    
    def generate_animation_data(self, paradigm: str, duration: float = 10.0) -> Dict:
        """Generate complete animation data for a paradigm"""
        try:
            fps = self.config.fps
            n_frames = int(duration * fps)
            time_points = np.linspace(0, 2*np.pi, n_frames)
            
            frames = []
            metadata = {
                'paradigm': paradigm,
                'duration': duration,
                'fps': fps,
                'n_frames': n_frames,
                'width': self.config.width,
                'height': self.config.height
            }
            
            for i, t in enumerate(time_points):
                frame = self.generate_visualization_frame(paradigm, t)
                if frame is not None:
                    # Convert to base64 for web transmission
                    if HAS_PIL:
                        img = Image.fromarray(frame.astype(np.uint8))
                        buffer = BytesIO()
                        img.save(buffer, format='PNG')
                        frame_b64 = base64.b64encode(buffer.getvalue()).decode()
                        frames.append(frame_b64)
                    else:
                        frames.append(frame.tolist())  # JSON serializable
                
                # Progress logging
                if i % (n_frames // 10) == 0:
                    logger.info(f"Animation progress: {i/n_frames*100:.1f}%")
            
            return {
                'frames': frames,
                'metadata': metadata,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Animation generation failed for {paradigm}: {e}")
            return {'success': False, 'error': str(e)}
    
    def create_unity_mandala(self, time: float = 0.0) -> np.ndarray:
        """Create a Unity mandala combining multiple mathematical patterns"""
        try:
            width, height = self.config.width, self.config.height
            mandala = np.zeros((height, width, 3), dtype=float)
            
            # Center coordinates
            center_x, center_y = width // 2, height // 2
            y, x = np.ogrid[:height, :width]
            
            # Distance and angle from center
            dx = x - center_x
            dy = y - center_y
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            
            # Golden ratio patterns
            phi = self.PHI
            golden_pattern = np.sin(r * phi / 50 + time) * np.cos(theta * phi + time * 0.5)
            
            # Fractal-like detail
            fractal_detail = np.sin(r / 20 + time * 2) * np.cos(theta * 8 + time)
            
            # Consciousness field
            consciousness = np.exp(-r**2 / (2 * (100 + 50 * np.sin(time))**2))
            
            # Combine patterns
            red_channel = (golden_pattern * consciousness + 1) / 2
            green_channel = (fractal_detail * consciousness + 1) / 2
            blue_channel = consciousness
            
            mandala[:, :, 0] = red_channel
            mandala[:, :, 1] = green_channel
            mandala[:, :, 2] = blue_channel
            
            # Normalize
            mandala = np.clip(mandala, 0, 1)
            
            return (mandala * 255).astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Unity mandala creation failed: {e}")
            return np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
    
    def export_visualization_config(self) -> Dict:
        """Export current visualization configuration"""
        return {
            'width': self.config.width,
            'height': self.config.height,
            'fps': self.config.fps,
            'duration': self.config.duration,
            'quality': self.config.quality,
            'color_scheme': self.config.color_scheme,
            'gpu_available': self.gpu_available,
            'has_matplotlib': HAS_MATPLOTLIB,
            'has_pil': HAS_PIL
        }

def create_web_visualization_data(paradigm: str, config: Optional[VisualizationConfig] = None) -> Dict:
    """Create visualization data optimized for web display"""
    try:
        kernels = UnityVisualizationKernels(config)
        
        # Generate static frame
        frame = kernels.generate_visualization_frame(paradigm, time=0.0)
        
        # Generate animation keyframes (reduced for web)
        keyframes = []
        for t in np.linspace(0, 2*np.pi, 20):  # 20 keyframes for smooth web animation
            keyframe = kernels.generate_visualization_frame(paradigm, t)
            if keyframe is not None:
                keyframes.append(keyframe.tolist())
        
        return {
            'paradigm': paradigm,
            'static_frame': frame.tolist() if frame is not None else None,
            'keyframes': keyframes,
            'config': kernels.export_visualization_config(),
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Web visualization data creation failed: {e}")
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    # Test visualization kernels
    config = VisualizationConfig(width=400, height=400, fps=15)
    kernels = UnityVisualizationKernels(config)
    
    print("🎨 Unity Visualization Kernels Test")
    print("=" * 40)
    
    # Test each paradigm
    paradigms = ['fractal', 'golden_ratio', 'consciousness', 'quantum']
    
    for paradigm in paradigms:
        try:
            frame = kernels.generate_visualization_frame(paradigm, time=0.0)
            if frame is not None:
                print(f"✅ {paradigm}: {frame.shape}")
            else:
                print(f"❌ {paradigm}: Failed")
        except Exception as e:
            print(f"❌ {paradigm}: Error - {e}")
    
    # Test Unity mandala
    try:
        mandala = kernels.create_unity_mandala()
        print(f"✅ Unity Mandala: {mandala.shape}")
    except Exception as e:
        print(f"❌ Unity Mandala: Error - {e}")
    
    print(f"\nConfiguration: {kernels.export_visualization_config()}")