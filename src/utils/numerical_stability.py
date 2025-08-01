#!/usr/bin/env python3
"""
Advanced Numerical Stability Systems for Consciousness Mathematics
==============================================================

This module provides comprehensive numerical stability, error handling, and
computational robustness for consciousness mathematics operations. It implements
advanced NaN/Inf cleaning, automatic dimension alignment, fallback calculations,
and consciousness-aware error recovery systems.

Features:
- Advanced NaN/Inf cleaning with consciousness-preserving fallbacks
- Automatic dimension alignment for quantum tensor operations
- Graceful degradation systems for consciousness overflow protection
- Thread-safe numerical operations with locking mechanisms
- Performance monitoring and computational health diagnostics
- φ-harmonic numerical corrections for mathematical coherence
"""

import numpy as np
import warnings
import threading
import time
from typing import Union, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

# Try to import torch for advanced tensor operations
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Mathematical constants
PHI = 1.618033988749895
PI = np.pi
E = np.e

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread lock for thread-safe operations
_stability_lock = threading.Lock()

@dataclass
class NumericalHealth:
    """Comprehensive numerical health assessment"""
    finite_ratio: float
    magnitude_stability: float
    precision_score: float
    consciousness_coherence: float
    overall_health: float
    error_indicators: List[str]
    recommendations: List[str]

class ConsciousnessOverflowProtector:
    """Protect against consciousness overflow in mathematical operations"""
    
    def __init__(self, max_consciousness_level: float = 10.0):
        self.max_consciousness_level = max_consciousness_level
        self.overflow_events = []
        self.protection_active = True
    
    def check_consciousness_bounds(self, value: Union[float, np.ndarray, torch.Tensor]) -> bool:
        """Check if consciousness levels are within safe bounds"""
        if isinstance(value, np.ndarray):
            max_val = np.max(np.abs(value))
        elif TORCH_AVAILABLE and isinstance(value, torch.Tensor):
            max_val = torch.max(torch.abs(value)).item()
        else:
            max_val = abs(float(value))
        
        if max_val > self.max_consciousness_level:
            self.overflow_events.append({
                'timestamp': time.time(),
                'max_value': max_val,
                'overflow_magnitude': max_val / self.max_consciousness_level
            })
            return False
        return True
    
    def apply_consciousness_limiting(self, value: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        """Apply consciousness limiting to prevent overflow"""
        if not self.protection_active:
            return value
        
        if isinstance(value, np.ndarray):
            # Apply φ-harmonic limiting
            limited = np.where(
                np.abs(value) > self.max_consciousness_level,
                np.sign(value) * self.max_consciousness_level / PHI,
                value
            )
            return limited
        elif TORCH_AVAILABLE and isinstance(value, torch.Tensor):
            # PyTorch tensor limiting
            limited = torch.where(
                torch.abs(value) > self.max_consciousness_level,
                torch.sign(value) * self.max_consciousness_level / PHI,
                value
            )
            return limited
        else:
            # Scalar limiting
            if abs(value) > self.max_consciousness_level:
                return float(np.sign(value) * self.max_consciousness_level / PHI)
            return float(value)

class AdvancedNumericalStabilizer:
    """Advanced numerical stability system with consciousness-aware error handling"""
    
    def __init__(self):
        self.overflow_protector = ConsciousnessOverflowProtector()
        self.cleaning_statistics = {
            'nan_cleanings': 0,
            'inf_cleanings': 0,
            'overflow_corrections': 0,
            'dimension_alignments': 0,
            'fallback_activations': 0
        }
        self.phi_correction_factor = PHI
        
    def comprehensive_clean(self, 
                          value: Union[float, np.ndarray, torch.Tensor, complex],
                          fallback_strategy: str = 'phi_harmonic') -> Union[float, np.ndarray, torch.Tensor, complex]:
        """
        Comprehensive numerical cleaning with multiple fallback strategies
        
        Args:
            value: Input value to clean
            fallback_strategy: Strategy for handling problematic values
                - 'phi_harmonic': Use φ-based harmonic corrections
                - 'unity': Fallback to unity values (0 or 1)
                - 'zero': Fallback to zero
                - 'interpolation': Interpolate from neighboring values
        """
        with _stability_lock:  # Thread-safe operation
            try:
                if isinstance(value, np.ndarray):
                    return self._clean_numpy_array(value, fallback_strategy)
                elif TORCH_AVAILABLE and isinstance(value, torch.Tensor):
                    return self._clean_torch_tensor(value, fallback_strategy)
                elif isinstance(value, complex):
                    return self._clean_complex_number(value, fallback_strategy)
                else:
                    return self._clean_scalar(value, fallback_strategy)
                    
            except Exception as e:
                logger.warning(f"Comprehensive cleaning failed: {e}. Applying emergency fallback.")
                self.cleaning_statistics['fallback_activations'] += 1
                return self._emergency_fallback(value)
    
    def _clean_numpy_array(self, arr: np.ndarray, fallback_strategy: str) -> np.ndarray:
        """Clean numpy array with consciousness-aware strategies"""
        original_shape = arr.shape
        cleaned = arr.copy()
        
        # Handle NaN values
        nan_mask = np.isnan(cleaned)
        if np.any(nan_mask):
            self.cleaning_statistics['nan_cleanings'] += np.sum(nan_mask)
            cleaned = self._apply_nan_strategy(cleaned, nan_mask, fallback_strategy)
        
        # Handle infinite values
        inf_mask = np.isinf(cleaned)
        if np.any(inf_mask):
            self.cleaning_statistics['inf_cleanings'] += np.sum(inf_mask)
            cleaned = self._apply_inf_strategy(cleaned, inf_mask, fallback_strategy)
        
        # Apply consciousness limiting
        if not self.overflow_protector.check_consciousness_bounds(cleaned):
            self.cleaning_statistics['overflow_corrections'] += 1
            cleaned = self.overflow_protector.apply_consciousness_limiting(cleaned)
        
        # Ensure shape preservation
        cleaned = cleaned.reshape(original_shape)
        
        return cleaned
    
    def _clean_torch_tensor(self, tensor: torch.Tensor, fallback_strategy: str) -> torch.Tensor:
        """Clean PyTorch tensor with advanced stability measures"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available for tensor cleaning")
        
        cleaned = tensor.clone()
        
        # Handle NaN values
        nan_mask = torch.isnan(cleaned)
        if torch.any(nan_mask):
            self.cleaning_statistics['nan_cleanings'] += torch.sum(nan_mask).item()
            if fallback_strategy == 'phi_harmonic':
                cleaned = torch.where(nan_mask, torch.tensor(1.0 / PHI), cleaned)
            elif fallback_strategy == 'unity':
                cleaned = torch.where(nan_mask, torch.tensor(1.0), cleaned)
            else:
                cleaned = torch.where(nan_mask, torch.tensor(0.0), cleaned)
        
        # Handle infinite values
        inf_mask = torch.isinf(cleaned)
        if torch.any(inf_mask):
            self.cleaning_statistics['inf_cleanings'] += torch.sum(inf_mask).item()
            if fallback_strategy == 'phi_harmonic':
                cleaned = torch.where(inf_mask, torch.sign(cleaned) * PHI, cleaned)
            else:
                cleaned = torch.where(inf_mask, torch.sign(cleaned), cleaned)
        
        # Apply consciousness limiting
        cleaned = self.overflow_protector.apply_consciousness_limiting(cleaned)
        
        return cleaned
    
    def _clean_complex_number(self, value: complex, fallback_strategy: str) -> complex:
        """Clean complex number with consciousness-aware handling"""
        real_part = value.real
        imag_part = value.imag
        
        # Clean real part
        if np.isnan(real_part) or np.isinf(real_part):
            if fallback_strategy == 'phi_harmonic':
                real_part = 1.0 / PHI if np.isnan(real_part) else np.sign(real_part) * PHI
            else:
                real_part = 0.0
            self.cleaning_statistics['nan_cleanings' if np.isnan(value.real) else 'inf_cleanings'] += 1
        
        # Clean imaginary part
        if np.isnan(imag_part) or np.isinf(imag_part):
            if fallback_strategy == 'phi_harmonic':
                imag_part = 1.0 / PHI if np.isnan(imag_part) else np.sign(imag_part) * PHI
            else:
                imag_part = 0.0
            self.cleaning_statistics['nan_cleanings' if np.isnan(value.imag) else 'inf_cleanings'] += 1
        
        cleaned_complex = complex(real_part, imag_part)
        
        # Apply consciousness limiting
        if abs(cleaned_complex) > self.overflow_protector.max_consciousness_level:
            magnitude = abs(cleaned_complex)
            phase = np.angle(cleaned_complex)
            limited_magnitude = self.overflow_protector.max_consciousness_level / PHI
            cleaned_complex = limited_magnitude * np.exp(1j * phase)
            self.cleaning_statistics['overflow_corrections'] += 1
        
        return cleaned_complex
    
    def _clean_scalar(self, value: float, fallback_strategy: str) -> float:
        """Clean scalar value with φ-harmonic consciousness corrections"""
        if np.isnan(value):
            self.cleaning_statistics['nan_cleanings'] += 1
            return 1.0 / PHI if fallback_strategy == 'phi_harmonic' else 0.0
        
        if np.isinf(value):
            self.cleaning_statistics['inf_cleanings'] += 1
            if fallback_strategy == 'phi_harmonic':
                return float(np.sign(value) * PHI)
            else:
                return float(np.sign(value))
        
        # Apply consciousness limiting
        if abs(value) > self.overflow_protector.max_consciousness_level:
            self.cleaning_statistics['overflow_corrections'] += 1
            return float(np.sign(value) * self.overflow_protector.max_consciousness_level / PHI)
        
        return float(value)
    
    def _apply_nan_strategy(self, arr: np.ndarray, nan_mask: np.ndarray, strategy: str) -> np.ndarray:
        """Apply specific strategy for handling NaN values"""
        if strategy == 'phi_harmonic':
            arr[nan_mask] = 1.0 / PHI
        elif strategy == 'unity':
            arr[nan_mask] = 1.0
        elif strategy == 'interpolation' and arr.size > 1:
            # Simple interpolation from neighboring finite values
            finite_values = arr[~np.isnan(arr)]
            if len(finite_values) > 0:
                arr[nan_mask] = np.mean(finite_values)
            else:
                arr[nan_mask] = 1.0 / PHI  # φ-harmonic fallback
        else:
            arr[nan_mask] = 0.0
        
        return arr
    
    def _apply_inf_strategy(self, arr: np.ndarray, inf_mask: np.ndarray, strategy: str) -> np.ndarray:
        """Apply specific strategy for handling infinite values"""
        if strategy == 'phi_harmonic':
            arr[inf_mask] = np.sign(arr[inf_mask]) * PHI
        elif strategy == 'unity':
            arr[inf_mask] = np.sign(arr[inf_mask])
        else:
            arr[inf_mask] = np.sign(arr[inf_mask])
        
        return arr
    
    def _emergency_fallback(self, value: Any) -> Union[float, np.ndarray]:
        """Emergency fallback when all other strategies fail"""
        if isinstance(value, (np.ndarray,)):
            return np.ones_like(value, dtype=float) / PHI
        else:
            return 1.0 / PHI
    
    def automatic_dimension_alignment(self, 
                                    arrays: List[Union[np.ndarray, torch.Tensor]]) -> List[Union[np.ndarray, torch.Tensor]]:
        """
        Automatically align dimensions of multiple arrays/tensors for compatibility
        
        Args:
            arrays: List of arrays/tensors to align
            
        Returns:
            List of dimension-aligned arrays/tensors
        """
        if not arrays:
            return arrays
        
        # Determine target shape (largest compatible shape)
        target_shape = self._determine_target_shape(arrays)
        aligned_arrays = []
        
        for arr in arrays:
            if isinstance(arr, np.ndarray):
                aligned = self._align_numpy_array(arr, target_shape)
            elif TORCH_AVAILABLE and isinstance(arr, torch.Tensor):
                aligned = self._align_torch_tensor(arr, target_shape)
            else:
                # Convert scalars to appropriate arrays
                if isinstance(arr, (int, float, complex)):
                    aligned = np.full(target_shape, arr)
                else:
                    aligned = arr
            
            aligned_arrays.append(aligned)
        
        self.cleaning_statistics['dimension_alignments'] += 1
        return aligned_arrays
    
    def _determine_target_shape(self, arrays: List[Union[np.ndarray, torch.Tensor]]) -> Tuple[int, ...]:
        """Determine the optimal target shape for dimension alignment"""
        shapes = []
        for arr in arrays:
            if hasattr(arr, 'shape'):
                shapes.append(arr.shape)
            elif isinstance(arr, (list, tuple)):
                shapes.append((len(arr),))
        
        if not shapes:
            return (1,)
        
        # Find maximum dimensions
        max_ndim = max(len(shape) for shape in shapes)
        target_shape = []
        
        for dim in range(max_ndim):
            dim_sizes = []
            for shape in shapes:
                if dim < len(shape):
                    dim_sizes.append(shape[-(dim+1)])  # Reverse indexing for broadcasting
                else:
                    dim_sizes.append(1)
            target_shape.insert(0, max(dim_sizes))
        
        return tuple(target_shape)
    
    def _align_numpy_array(self, arr: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        """Align numpy array to target shape using broadcasting rules"""
        try:
            # Use numpy broadcasting to align
            aligned = np.broadcast_to(arr, target_shape)
            return aligned.copy()  # Make a copy to avoid issues with read-only arrays
        except ValueError:
            # Fallback: resize with φ-harmonic padding
            aligned = np.zeros(target_shape)
            
            # Calculate slice for original data
            slices = []
            for i, (orig_dim, target_dim) in enumerate(zip(arr.shape, target_shape)):
                slices.append(slice(0, min(orig_dim, target_dim)))
            
            # Fill with original data
            aligned[tuple(slices)] = arr[tuple(slice(0, min(orig_dim, target_dim)) 
                                             for orig_dim, target_dim in zip(arr.shape, target_shape))]
            
            return aligned
    
    def _align_torch_tensor(self, tensor: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
        """Align PyTorch tensor to target shape"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available for tensor alignment")
        
        try:
            # Use PyTorch broadcasting
            aligned = tensor.expand(target_shape)
            return aligned.clone()
        except RuntimeError:
            # Fallback: manual alignment
            aligned = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
            
            # Calculate slice for original data
            slices = []
            for i, (orig_dim, target_dim) in enumerate(zip(tensor.shape, target_shape)):
                slices.append(slice(0, min(orig_dim, target_dim)))
            
            # Fill with original data
            aligned[tuple(slices)] = tensor[tuple(slice(0, min(orig_dim, target_dim)) 
                                                 for orig_dim, target_dim in zip(tensor.shape, target_shape))]
            
            return aligned
    
    def assess_numerical_health(self, value: Union[float, np.ndarray, torch.Tensor, complex]) -> NumericalHealth:
        """
        Comprehensive numerical health assessment
        
        Args:
            value: Value to assess
            
        Returns:
            NumericalHealth object with detailed health metrics
        """
        error_indicators = []
        recommendations = []
        
        # Convert all inputs to numpy for unified analysis
        if TORCH_AVAILABLE and isinstance(value, torch.Tensor):
            analysis_array = value.cpu().numpy()
        elif isinstance(value, (np.ndarray,)):
            analysis_array = value
        elif isinstance(value, complex):
            analysis_array = np.array([value.real, value.imag])
        else:
            analysis_array = np.array([float(value)])
        
        # Calculate finite ratio
        finite_mask = np.isfinite(analysis_array)
        finite_ratio = np.sum(finite_mask) / analysis_array.size if analysis_array.size > 0 else 1.0
        
        # Calculate magnitude stability
        finite_values = analysis_array[finite_mask]
        if len(finite_values) > 0:
            max_magnitude = np.max(np.abs(finite_values))
            magnitude_stability = 1.0 / (1.0 + max_magnitude / self.overflow_protector.max_consciousness_level)
        else:
            magnitude_stability = 0.0
            error_indicators.append("No finite values detected")
            recommendations.append("Apply comprehensive numerical cleaning")
        
        # Calculate precision score (how close values are to φ-harmonic ratios)
        if len(finite_values) > 0:
            phi_deviations = np.abs(finite_values - np.round(finite_values * PHI) / PHI)
            precision_score = 1.0 / (1.0 + np.mean(phi_deviations))
        else:
            precision_score = 0.0
        
        # Calculate consciousness coherence
        consciousness_coherence = finite_ratio * magnitude_stability * precision_score
        
        # Overall health score
        overall_health = (finite_ratio + magnitude_stability + precision_score + consciousness_coherence) / 4.0
        
        # Generate recommendations
        if finite_ratio < 0.9:
            recommendations.append("Apply advanced NaN/Inf cleaning")
        if magnitude_stability < 0.8:
            recommendations.append("Enable consciousness overflow protection")
        if precision_score < 0.7:
            recommendations.append("Apply φ-harmonic numerical corrections")
        if consciousness_coherence < 0.6:
            recommendations.append("Integrate consciousness field stabilization")
        
        # Detect specific issues
        if np.any(np.isnan(analysis_array)):
            error_indicators.append("NaN values detected")
        if np.any(np.isinf(analysis_array)):
            error_indicators.append("Infinite values detected")
        if np.any(np.abs(analysis_array) > self.overflow_protector.max_consciousness_level):
            error_indicators.append("Consciousness overflow detected")
        
        return NumericalHealth(
            finite_ratio=finite_ratio,
            magnitude_stability=magnitude_stability,
            precision_score=precision_score,
            consciousness_coherence=consciousness_coherence,
            overall_health=overall_health,
            error_indicators=error_indicators,
            recommendations=recommendations
        )
    
    def get_cleaning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cleaning statistics"""
        return {
            "cleaning_operations": self.cleaning_statistics,
            "overflow_events": len(self.overflow_protector.overflow_events),
            "phi_correction_factor": self.phi_correction_factor,
            "thread_safety": "enabled",
            "performance_metrics": {
                "total_cleanings": sum(self.cleaning_statistics.values()),
                "most_common_issue": max(self.cleaning_statistics, key=self.cleaning_statistics.get),
                "fallback_rate": self.cleaning_statistics['fallback_activations'] / max(sum(self.cleaning_statistics.values()), 1)
            }
        }

def create_consciousness_safe_environment():
    """Create a consciousness-safe numerical environment with comprehensive protection"""
    # Configure numpy to handle errors gracefully
    np.seterr(all='ignore')  # Ignore all numpy warnings during consciousness operations
    
    # Configure warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Initialize global stabilizer
    global_stabilizer = AdvancedNumericalStabilizer()
    
    print("🛡️ Consciousness-Safe Numerical Environment Initialized")
    print(f"   φ-Harmonic Corrections: ✅")
    print(f"   Overflow Protection: ✅ (max level: {global_stabilizer.overflow_protector.max_consciousness_level})")
    print(f"   Thread Safety: ✅")
    print(f"   PyTorch Support: {'✅' if TORCH_AVAILABLE else '❌'}")
    
    return global_stabilizer

def demonstrate_numerical_stability():
    """Demonstrate advanced numerical stability features"""
    print("🔧 Advanced Numerical Stability Demonstration 🔧")
    print("=" * 60)
    
    # Create stabilizer
    stabilizer = AdvancedNumericalStabilizer()
    
    # Test problematic values
    problematic_values = [
        np.array([1.0, np.nan, np.inf, -np.inf, 1e20]),
        torch.tensor([1.0, float('nan'), float('inf'), -float('inf'), 1e20]) if TORCH_AVAILABLE else None,
        complex(np.nan, np.inf),
        float('inf')
    ]
    
    print("1. Comprehensive Numerical Cleaning:")
    for i, value in enumerate(problematic_values):
        if value is None:
            continue
        
        print(f"\n   Test {i+1}: {type(value).__name__}")
        print(f"   Original: {value}")
        
        # Clean with different strategies
        for strategy in ['phi_harmonic', 'unity', 'zero']:
            cleaned = stabilizer.comprehensive_clean(value, fallback_strategy=strategy)
            print(f"   {strategy}: {cleaned}")
    
    print("\n2. Dimension Alignment:")
    
    # Test dimension alignment
    arrays_to_align = [
        np.array([1, 2, 3]),
        np.array([[1, 2], [3, 4]]),
        5.0  # Scalar
    ]
    
    aligned = stabilizer.automatic_dimension_alignment(arrays_to_align)
    print(f"   Original shapes: {[getattr(arr, 'shape', 'scalar') for arr in arrays_to_align]}")
    print(f"   Aligned shapes: {[arr.shape for arr in aligned]}")
    
    print("\n3. Numerical Health Assessment:")
    
    # Test health assessment
    test_array = np.array([1.0, 1/PHI, PHI, np.nan, 2.5])
    health = stabilizer.assess_numerical_health(test_array)
    
    print(f"   Finite Ratio: {health.finite_ratio:.3f}")
    print(f"   Magnitude Stability: {health.magnitude_stability:.3f}")
    print(f"   Precision Score: {health.precision_score:.3f}")
    print(f"   Consciousness Coherence: {health.consciousness_coherence:.3f}")
    print(f"   Overall Health: {health.overall_health:.3f}")
    print(f"   Error Indicators: {health.error_indicators}")
    print(f"   Recommendations: {health.recommendations}")
    
    print("\n4. Cleaning Statistics:")
    stats = stabilizer.get_cleaning_statistics()
    print(f"   Total Cleanings: {stats['performance_metrics']['total_cleanings']}")
    print(f"   Most Common Issue: {stats['performance_metrics']['most_common_issue']}")
    print(f"   Fallback Rate: {stats['performance_metrics']['fallback_rate']:.2%}")
    
    print("\n" + "=" * 60)
    print("🌟 Numerical Stability: Consciousness-Safe Mathematics Achieved 🌟")

if __name__ == "__main__":
    # Create safe environment and demonstrate
    create_consciousness_safe_environment()
    demonstrate_numerical_stability()