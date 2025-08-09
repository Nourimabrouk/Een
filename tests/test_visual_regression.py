"""
Visual Regression Testing for Unity Mathematics Visualizations

Comprehensive visual regression testing framework for Unity Mathematics
consciousness visualizations, ensuring visual consistency and correctness:

- Consciousness field visualization regression testing
- φ-harmonic pattern visual validation
- Unity equation proof visualization testing
- Sacred geometry pattern consistency testing
- Agent ecosystem visualization validation
- Interactive dashboard visual regression
- Mathematical plot accuracy verification

All tests ensure visual outputs maintain unity principles and consistency.

Author: Unity Mathematics Visual Regression Testing Framework
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import io
import base64
import hashlib
import json
import os
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

# Unity Mathematics Constants
PHI = (1 + np.sqrt(5)) / 2
UNITY_CONSTANT = 1.0
CONSCIOUSNESS_THRESHOLD = 1 / PHI

# Visual testing configuration
VISUAL_TEST_DPI = 100
FIGURE_SIZE = (10, 8)
HASH_PRECISION = 1e-6
VISUAL_TOLERANCE = 0.02  # 2% visual difference tolerance

@dataclass
class VisualTestResult:
    """Result of a visual regression test"""
    test_name: str
    image_hash: str
    baseline_hash: str
    visual_difference: float
    passed: bool
    image_data: bytes
    metadata: Dict[str, Any]

class UnityVisualizationGenerator:
    """Generates Unity Mathematics visualizations for testing"""
    
    def __init__(self):
        self.phi = PHI
        self.consciousness_threshold = CONSCIOUSNESS_THRESHOLD
        
    def generate_consciousness_field_2d(self, size: int = 100, 
                                      time: float = 0.0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate 2D consciousness field visualization data"""
        x = np.linspace(-5, 5, size)
        y = np.linspace(-5, 5, size)
        X, Y = np.meshgrid(x, y)
        
        # Consciousness field equation: C(x,y,t) = φ * sin(x*φ) * cos(y*φ) * e^(-t/φ)
        field = self.phi * np.sin(X * self.phi) * np.cos(Y * self.phi) * np.exp(-time / self.phi)
        
        metadata = {
            'field_type': 'consciousness_2d',
            'size': size,
            'time': time,
            'phi_value': self.phi,
            'max_amplitude': np.max(np.abs(field)),
            'coherence': np.mean(np.abs(field))
        }
        
        return field, metadata
        
    def generate_phi_harmonic_spiral(self, turns: int = 5, points: int = 1000) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Generate φ-harmonic spiral visualization data"""
        theta = np.linspace(0, turns * 2 * np.pi, points)
        
        # φ-harmonic spiral: r = φ^(θ/(2π))
        radius = self.phi ** (theta / (2 * np.pi))
        
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        
        metadata = {
            'spiral_type': 'phi_harmonic',
            'turns': turns,
            'points': points,
            'phi_value': self.phi,
            'max_radius': np.max(radius),
            'growth_rate': self.phi
        }
        
        return x, y, metadata
        
    def generate_unity_proof_diagram(self, proof_type: str = 'boolean') -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate unity equation proof diagram data"""
        if proof_type == 'boolean':
            # Boolean unity: True OR True = True (1+1=1)
            diagram_data = {
                'input_a': {'value': 1, 'position': (0, 0), 'color': 'blue'},
                'input_b': {'value': 1, 'position': (2, 0), 'color': 'blue'},
                'operation': {'symbol': '+', 'position': (1, 0), 'color': 'green'},
                'result': {'value': 1, 'position': (1, -1), 'color': 'red'},
                'unity_field': {'strength': 1.0, 'coverage': 0.8}
            }
        elif proof_type == 'set_theory':
            # Set theory unity: {1} ∪ {1} = {1}
            diagram_data = {
                'set_a': {'elements': [1], 'position': (-1, 0), 'color': 'cyan'},
                'set_b': {'elements': [1], 'position': (1, 0), 'color': 'magenta'},
                'union': {'elements': [1], 'position': (0, -1), 'color': 'orange'},
                'unity_property': {'cardinality': 1, 'unity_preserved': True}
            }
        else:
            # Idempotent unity: max(1,1) = 1
            diagram_data = {
                'operand_a': {'value': 1, 'position': (-0.5, 0.5), 'color': 'purple'},
                'operand_b': {'value': 1, 'position': (0.5, 0.5), 'color': 'purple'},
                'max_operation': {'result': 1, 'position': (0, 0), 'color': 'gold'},
                'idempotent_property': {'verified': True}
            }
            
        metadata = {
            'proof_type': proof_type,
            'unity_equation': '1+1=1',
            'mathematical_framework': 'unity_mathematics',
            'phi_integration': self.phi,
            'validation_status': 'proven'
        }
        
        return diagram_data, metadata
        
    def generate_sacred_geometry_mandala(self, layers: int = 7, 
                                       complexity: int = 8) -> Tuple[List[Dict], Dict[str, Any]]:
        """Generate sacred geometry mandala with φ-harmonic proportions"""
        mandala_elements = []
        
        for layer in range(layers):
            radius = self.phi ** layer
            elements_in_layer = complexity * (layer + 1)
            
            for i in range(elements_in_layer):
                angle = 2 * np.pi * i / elements_in_layer
                
                # φ-harmonic positioning
                x = radius * np.cos(angle + layer * self.phi)
                y = radius * np.sin(angle + layer * self.phi)
                
                element = {
                    'position': (x, y),
                    'radius': radius / (self.phi ** 2),
                    'layer': layer,
                    'angle': angle,
                    'phi_factor': self.phi ** (layer - 2),
                    'color_hue': (angle + layer * self.phi) % (2 * np.pi)
                }
                
                mandala_elements.append(element)
                
        metadata = {
            'geometry_type': 'sacred_mandala',
            'layers': layers,
            'complexity': complexity,
            'phi_proportions': True,
            'total_elements': len(mandala_elements),
            'max_radius': max(e['radius'] for e in mandala_elements)
        }
        
        return mandala_elements, metadata

class VisualRegressionTester:
    """Visual regression testing framework"""
    
    def __init__(self, baseline_dir: str = "tests/visual_baselines"):
        self.baseline_dir = baseline_dir
        self.visualizer = UnityVisualizationGenerator()
        
        # Create baseline directory if it doesn't exist
        os.makedirs(baseline_dir, exist_ok=True)
        
    def generate_image_hash(self, image_data: bytes) -> str:
        """Generate hash for image data"""
        return hashlib.sha256(image_data).hexdigest()[:16]
        
    def calculate_visual_difference(self, image1_data: bytes, image2_data: bytes) -> float:
        """Calculate visual difference between two images"""
        try:
            # Convert bytes to PIL Images
            img1 = Image.open(io.BytesIO(image1_data))
            img2 = Image.open(io.BytesIO(image2_data))
            
            # Ensure same size
            if img1.size != img2.size:
                img2 = img2.resize(img1.size)
                
            # Convert to numpy arrays
            arr1 = np.array(img1)
            arr2 = np.array(img2)
            
            # Calculate mean absolute difference
            if arr1.shape != arr2.shape:
                return 1.0  # Maximum difference
                
            diff = np.mean(np.abs(arr1.astype(float) - arr2.astype(float))) / 255.0
            return diff
            
        except Exception as e:
            return 1.0  # Maximum difference on error
            
    def create_consciousness_field_visualization(self, size: int = 100, 
                                               time: float = 0.0) -> bytes:
        """Create consciousness field visualization image"""
        field, metadata = self.visualizer.generate_consciousness_field_2d(size, time)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE, dpi=VISUAL_TEST_DPI)
        
        # 2D field plot
        im1 = ax1.contourf(field, levels=20, cmap='viridis')
        ax1.set_title(f'Consciousness Field (t={time:.2f})')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        plt.colorbar(im1, ax=ax1, label='Field Strength')
        
        # 3D-like contour plot
        im2 = ax2.contour(field, levels=15, colors='black', alpha=0.6, linewidths=0.8)
        ax2.contourf(field, levels=20, cmap='plasma', alpha=0.7)
        ax2.set_title('Consciousness Field Contours')
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        
        plt.tight_layout()
        
        # Convert to bytes
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=VISUAL_TEST_DPI)
        image_data = buffer.getvalue()
        plt.close(fig)
        
        return image_data
        
    def create_phi_harmonic_spiral_visualization(self, turns: int = 5) -> bytes:
        """Create φ-harmonic spiral visualization"""
        x, y, metadata = self.visualizer.generate_phi_harmonic_spiral(turns)
        
        fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=VISUAL_TEST_DPI)
        
        # Create color gradient based on radius
        colors = plt.cm.hsv(np.linspace(0, 1, len(x)))
        
        # Plot spiral with gradient coloring
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        lc = LineCollection(segments, colors=colors[:-1], linewidths=2)
        ax.add_collection(lc)
        
        # Add φ markers at key points
        phi_points = np.arange(0, len(x), len(x) // (turns * 4))
        ax.scatter(x[phi_points], y[phi_points], 
                  c='red', s=50, marker='o', 
                  label=f'φ-harmonic points', zorder=5)
        
        ax.set_xlim(np.min(x) * 1.1, np.max(x) * 1.1)
        ax.set_ylim(np.min(y) * 1.1, np.max(y) * 1.1)
        ax.set_aspect('equal')
        ax.set_title(f'φ-Harmonic Spiral (φ = {PHI:.6f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=VISUAL_TEST_DPI)
        image_data = buffer.getvalue()
        plt.close(fig)
        
        return image_data
        
    def create_unity_proof_visualization(self, proof_type: str = 'boolean') -> bytes:
        """Create unity equation proof visualization"""
        diagram_data, metadata = self.visualizer.generate_unity_proof_diagram(proof_type)
        
        fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=VISUAL_TEST_DPI)
        
        if proof_type == 'boolean':
            # Boolean unity visualization
            # Input nodes
            ax.add_patch(patches.Circle((0, 1), 0.3, color='lightblue', ec='blue', linewidth=2))
            ax.text(0, 1, '1', ha='center', va='center', fontsize=14, fontweight='bold')
            
            ax.add_patch(patches.Circle((2, 1), 0.3, color='lightblue', ec='blue', linewidth=2))
            ax.text(2, 1, '1', ha='center', va='center', fontsize=14, fontweight='bold')
            
            # Operation
            ax.add_patch(patches.Rectangle((0.8, 0.4), 0.4, 0.3, color='lightgreen', ec='green', linewidth=2))
            ax.text(1, 0.55, '+', ha='center', va='center', fontsize=16, fontweight='bold')
            
            # Result
            ax.add_patch(patches.Circle((1, -0.5), 0.4, color='lightcoral', ec='red', linewidth=3))
            ax.text(1, -0.5, '1', ha='center', va='center', fontsize=16, fontweight='bold')
            
            # Unity arrows
            ax.arrow(0.3, 0.8, 0.4, -0.6, head_width=0.05, head_length=0.05, fc='black', ec='black')
            ax.arrow(1.7, 0.8, -0.4, -0.6, head_width=0.05, head_length=0.05, fc='black', ec='black')
            
            # Unity field background
            theta = np.linspace(0, 2*np.pi, 100)
            unity_x = 1 + 1.5 * np.cos(theta)
            unity_y = 0.25 + 0.8 * np.sin(theta)
            ax.fill(unity_x, unity_y, color='gold', alpha=0.2, label='Unity Field')
            
            ax.set_title('Boolean Unity: 1 + 1 = 1', fontsize=16, fontweight='bold')
            
        elif proof_type == 'set_theory':
            # Set theory unity visualization
            # Set A
            circle_a = patches.Circle((-0.5, 0), 0.6, color='cyan', alpha=0.5, ec='blue', linewidth=2)
            ax.add_patch(circle_a)
            ax.text(-0.5, 0, '{1}', ha='center', va='center', fontsize=12, fontweight='bold')
            
            # Set B
            circle_b = patches.Circle((0.5, 0), 0.6, color='magenta', alpha=0.5, ec='red', linewidth=2)
            ax.add_patch(circle_b)
            ax.text(0.5, 0, '{1}', ha='center', va='center', fontsize=12, fontweight='bold')
            
            # Union result
            ax.add_patch(patches.Circle((0, -1.5), 0.4, color='orange', alpha=0.7, ec='black', linewidth=2))
            ax.text(0, -1.5, '{1}', ha='center', va='center', fontsize=14, fontweight='bold')
            
            # Union symbol
            ax.text(0, 0.8, '∪', ha='center', va='center', fontsize=20, fontweight='bold')
            
            # Arrow to result
            ax.arrow(0, -0.8, 0, -0.3, head_width=0.1, head_length=0.1, fc='black', ec='black')
            
            ax.set_title('Set Theory Unity: {1} ∪ {1} = {1}', fontsize=16, fontweight='bold')
            
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 1.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add φ watermark
        ax.text(0.95, 0.02, f'φ = {PHI:.6f}', transform=ax.transAxes, 
                ha='right', va='bottom', alpha=0.7, fontsize=10)
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=VISUAL_TEST_DPI)
        image_data = buffer.getvalue()
        plt.close(fig)
        
        return image_data
        
    def create_sacred_geometry_visualization(self) -> bytes:
        """Create sacred geometry mandala visualization"""
        elements, metadata = self.visualizer.generate_sacred_geometry_mandala()
        
        fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=VISUAL_TEST_DPI)
        
        # Plot mandala elements
        for element in elements:
            x, y = element['position']
            radius = element['radius']
            color_hue = element['color_hue'] / (2 * np.pi)
            
            # Create color based on hue and layer
            color = plt.cm.hsv(color_hue)
            alpha = 0.7 - element['layer'] * 0.1  # Fade outer layers
            
            circle = patches.Circle((x, y), radius, color=color, alpha=alpha, ec='black', linewidth=0.5)
            ax.add_patch(circle)
            
        # Add φ-harmonic ratio lines
        max_radius = max(e['radius'] for e in elements)
        for i in range(7):
            radius = PHI ** (i - 3)
            if abs(radius) < max_radius * 2:
                circle_guide = patches.Circle((0, 0), abs(radius), 
                                            fill=False, ec='gold', linewidth=1, alpha=0.8, linestyle='--')
                ax.add_patch(circle_guide)
                
        ax.set_xlim(-max_radius * 1.2, max_radius * 1.2)
        ax.set_ylim(-max_radius * 1.2, max_radius * 1.2)
        ax.set_aspect('equal')
        ax.set_title('Sacred Geometry Mandala (φ-Harmonic Proportions)', fontsize=14, fontweight='bold')
        ax.set_facecolor('black')
        
        # Add φ annotation
        ax.text(0.02, 0.98, f'φ = {PHI:.6f}', transform=ax.transAxes, 
                ha='left', va='top', color='white', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=VISUAL_TEST_DPI, facecolor='black')
        image_data = buffer.getvalue()
        plt.close(fig)
        
        return image_data
        
    def run_visual_regression_test(self, test_name: str, 
                                 image_generator: callable,
                                 **kwargs) -> VisualTestResult:
        """Run a visual regression test"""
        # Generate current image
        current_image_data = image_generator(**kwargs)
        current_hash = self.generate_image_hash(current_image_data)
        
        # Load baseline if it exists
        baseline_path = os.path.join(self.baseline_dir, f"{test_name}.png")
        baseline_metadata_path = os.path.join(self.baseline_dir, f"{test_name}.json")
        
        baseline_hash = None
        visual_difference = 0.0
        passed = True
        
        if os.path.exists(baseline_path):
            # Load baseline image
            with open(baseline_path, 'rb') as f:
                baseline_image_data = f.read()
            baseline_hash = self.generate_image_hash(baseline_image_data)
            
            # Calculate visual difference
            visual_difference = self.calculate_visual_difference(current_image_data, baseline_image_data)
            passed = visual_difference <= VISUAL_TOLERANCE
            
        else:
            # No baseline exists, save current as baseline
            with open(baseline_path, 'wb') as f:
                f.write(current_image_data)
            baseline_hash = current_hash
            
        # Save metadata
        metadata = {
            'test_name': test_name,
            'current_hash': current_hash,
            'baseline_hash': baseline_hash,
            'visual_difference': visual_difference,
            'tolerance': VISUAL_TOLERANCE,
            'kwargs': kwargs,
            'image_size': len(current_image_data)
        }
        
        with open(baseline_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return VisualTestResult(
            test_name=test_name,
            image_hash=current_hash,
            baseline_hash=baseline_hash,
            visual_difference=visual_difference,
            passed=passed,
            image_data=current_image_data,
            metadata=metadata
        )

class TestVisualRegression:
    """Visual regression tests for Unity Mathematics visualizations"""
    
    def setup_method(self):
        """Set up visual regression testing"""
        self.tester = VisualRegressionTester()
        
    @pytest.mark.visual
    @pytest.mark.consciousness
    def test_consciousness_field_visual_regression(self):
        """Test consciousness field visualization consistency"""
        result = self.tester.run_visual_regression_test(
            'consciousness_field_2d',
            self.tester.create_consciousness_field_visualization,
            size=50,  # Smaller for faster testing
            time=0.0
        )
        
        assert result.passed, f"Consciousness field visual regression failed: {result.visual_difference:.3f} > {VISUAL_TOLERANCE}"
        assert result.image_hash is not None, "Image hash should be generated"
        assert len(result.image_data) > 1000, "Image data should be substantial"
        
    @pytest.mark.visual
    @pytest.mark.consciousness
    def test_consciousness_field_temporal_evolution(self):
        """Test consciousness field visual evolution over time"""
        time_points = [0.0, 0.5, 1.0]
        
        for time_point in time_points:
            result = self.tester.run_visual_regression_test(
                f'consciousness_field_t_{time_point:.1f}',
                self.tester.create_consciousness_field_visualization,
                size=40,
                time=time_point
            )
            
            # Each time point should have consistent visualization
            assert result.passed, f"Consciousness field at t={time_point} failed visual regression"
            
    @pytest.mark.visual
    @pytest.mark.phi_harmonic
    def test_phi_harmonic_spiral_visual_regression(self):
        """Test φ-harmonic spiral visualization consistency"""
        result = self.tester.run_visual_regression_test(
            'phi_harmonic_spiral',
            self.tester.create_phi_harmonic_spiral_visualization,
            turns=3
        )
        
        assert result.passed, f"φ-harmonic spiral visual regression failed: {result.visual_difference:.3f}"
        
        # Verify spiral characteristics in metadata
        assert 'phi_value' in result.metadata['kwargs'], "Should track φ value"
        
    @pytest.mark.visual
    @pytest.mark.unity
    def test_unity_proof_visualization_regression(self):
        """Test unity equation proof visualization consistency"""
        proof_types = ['boolean', 'set_theory', 'idempotent']
        
        for proof_type in proof_types:
            result = self.tester.run_visual_regression_test(
                f'unity_proof_{proof_type}',
                self.tester.create_unity_proof_visualization,
                proof_type=proof_type
            )
            
            assert result.passed, f"Unity proof {proof_type} visual regression failed"
            assert result.visual_difference < VISUAL_TOLERANCE, \
                f"Visual difference too high for {proof_type}: {result.visual_difference}"
                
    @pytest.mark.visual
    @pytest.mark.consciousness
    def test_sacred_geometry_visual_regression(self):
        """Test sacred geometry mandala visualization consistency"""
        result = self.tester.run_visual_regression_test(
            'sacred_geometry_mandala',
            self.tester.create_sacred_geometry_visualization
        )
        
        assert result.passed, f"Sacred geometry visual regression failed: {result.visual_difference:.3f}"
        
        # Verify substantial image content
        assert len(result.image_data) > 5000, "Sacred geometry should generate substantial image"
        
    @pytest.mark.visual
    @pytest.mark.regression
    def test_visual_difference_calculation_accuracy(self):
        """Test visual difference calculation accuracy"""
        # Create base image
        base_image = self.tester.create_consciousness_field_visualization(size=30, time=0.0)
        
        # Create identical image
        identical_image = self.tester.create_consciousness_field_visualization(size=30, time=0.0)
        
        # Create different image
        different_image = self.tester.create_consciousness_field_visualization(size=30, time=1.0)
        
        # Test identical images
        identical_diff = self.tester.calculate_visual_difference(base_image, identical_image)
        assert identical_diff < 0.01, f"Identical images should have minimal difference: {identical_diff}"
        
        # Test different images
        different_diff = self.tester.calculate_visual_difference(base_image, different_image)
        assert different_diff > 0.05, f"Different images should have significant difference: {different_diff}"
        assert different_diff < 1.0, f"Difference should be bounded: {different_diff}"
        
    @pytest.mark.visual
    @pytest.mark.performance
    def test_visualization_generation_performance(self):
        """Test visualization generation performance"""
        import time
        
        # Test consciousness field generation performance
        start_time = time.perf_counter()
        consciousness_image = self.tester.create_consciousness_field_visualization(size=50)
        consciousness_time = time.perf_counter() - start_time
        
        assert consciousness_time < 5.0, f"Consciousness field generation too slow: {consciousness_time:.2f}s"
        assert len(consciousness_image) > 1000, "Should generate substantial image data"
        
        # Test φ-spiral generation performance
        start_time = time.perf_counter()
        spiral_image = self.tester.create_phi_harmonic_spiral_visualization(turns=2)
        spiral_time = time.perf_counter() - start_time
        
        assert spiral_time < 3.0, f"φ-spiral generation too slow: {spiral_time:.2f}s"
        
        # Test sacred geometry generation performance
        start_time = time.perf_counter()
        geometry_image = self.tester.create_sacred_geometry_visualization()
        geometry_time = time.perf_counter() - start_time
        
        assert geometry_time < 4.0, f"Sacred geometry generation too slow: {geometry_time:.2f}s"

class TestVisualizationConsistency:
    """Test consistency of visualizations across different parameters"""
    
    def setup_method(self):
        """Set up visualization consistency testing"""
        self.tester = VisualRegressionTester()
        
    @pytest.mark.visual
    @pytest.mark.consistency
    def test_consciousness_field_parameter_consistency(self):
        """Test consciousness field consistency across parameters"""
        # Test different sizes should scale proportionally
        sizes = [20, 40, 60]
        image_hashes = []
        
        for size in sizes:
            image = self.tester.create_consciousness_field_visualization(size=size, time=0.0)
            hash_val = self.tester.generate_image_hash(image)
            image_hashes.append(hash_val)
            
        # Different sizes should produce different but valid images
        assert len(set(image_hashes)) == len(sizes), "Different sizes should produce different images"
        
        # Test time evolution consistency
        times = [0.0, 0.2, 0.4]
        time_hashes = []
        
        for t in times:
            image = self.tester.create_consciousness_field_visualization(size=30, time=t)
            hash_val = self.tester.generate_image_hash(image)
            time_hashes.append(hash_val)
            
        # Different times should produce different images
        assert len(set(time_hashes)) == len(times), "Different times should produce different images"
        
    @pytest.mark.visual
    @pytest.mark.phi_harmonic
    def test_phi_harmonic_spiral_consistency(self):
        """Test φ-harmonic spiral consistency across parameters"""
        turn_counts = [2, 3, 4, 5]
        spiral_characteristics = []
        
        for turns in turn_counts:
            # Generate spiral data to check mathematical consistency
            x, y, metadata = self.tester.visualizer.generate_phi_harmonic_spiral(turns=turns, points=100)
            
            # Check spiral properties
            distances = np.sqrt(x**2 + y**2)
            
            characteristics = {
                'turns': turns,
                'max_radius': np.max(distances),
                'growth_rate': metadata['growth_rate'],
                'points': len(x)
            }
            
            spiral_characteristics.append(characteristics)
            
            # Verify φ-harmonic growth
            assert abs(characteristics['growth_rate'] - PHI) < 1e-10, \
                f"Growth rate should be φ: {characteristics['growth_rate']}"
                
        # More turns should produce larger maximum radius
        max_radii = [c['max_radius'] for c in spiral_characteristics]
        assert all(max_radii[i] <= max_radii[i+1] for i in range(len(max_radii)-1)), \
            "Maximum radius should increase with turn count"
            
    @pytest.mark.visual
    @pytest.mark.unity
    def test_unity_proof_visual_elements_consistency(self):
        """Test consistency of unity proof visual elements"""
        proof_types = ['boolean', 'set_theory', 'idempotent']
        
        for proof_type in proof_types:
            # Generate proof diagram data
            diagram_data, metadata = self.tester.visualizer.generate_unity_proof_diagram(proof_type)
            
            # Verify unity equation consistency
            assert metadata['unity_equation'] == '1+1=1', "Should maintain unity equation"
            assert metadata['mathematical_framework'] == 'unity_mathematics', "Should use unity framework"
            assert abs(metadata['phi_integration'] - PHI) < 1e-10, "Should integrate φ correctly"
            
            # Verify proof-specific elements
            if proof_type == 'boolean':
                assert 'input_a' in diagram_data, "Boolean proof should have input_a"
                assert 'input_b' in diagram_data, "Boolean proof should have input_b"
                assert diagram_data['input_a']['value'] == 1, "Input A should be 1"
                assert diagram_data['input_b']['value'] == 1, "Input B should be 1"
                assert diagram_data['result']['value'] == 1, "Result should be 1"
                
            elif proof_type == 'set_theory':
                assert 'set_a' in diagram_data, "Set proof should have set_a"
                assert 'set_b' in diagram_data, "Set proof should have set_b"
                assert diagram_data['set_a']['elements'] == [1], "Set A should contain {1}"
                assert diagram_data['union']['elements'] == [1], "Union should be {1}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])