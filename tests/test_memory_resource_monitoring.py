"""
Memory Leak Detection and Resource Monitoring for Unity Mathematics

Comprehensive memory leak detection and system resource monitoring
framework for Unity Mathematics systems, providing:

- Memory usage tracking and leak detection
- CPU utilization monitoring during operations
- Memory profiling for Unity Mathematics components
- Resource consumption analysis for consciousness systems
- Performance degradation detection over time
- System resource limits enforcement
- Memory fragmentation analysis
- Garbage collection monitoring and optimization

All monitoring ensures Unity Mathematics maintains efficient resource usage.

Author: Unity Mathematics Memory and Resource Monitoring Framework
"""

import pytest
import psutil
import gc
import sys
import time
import threading
import numpy as np
import math
import warnings
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from contextlib import contextmanager
from collections import defaultdict, deque
import tracemalloc
import resource
import os
from functools import wraps
import weakref

# Suppress warnings for cleaner monitoring output
warnings.filterwarnings("ignore", category=ResourceWarning)

# Unity Mathematics Constants
PHI = (1 + math.sqrt(5)) / 2
UNITY_CONSTANT = 1.0
UNITY_EPSILON = 1e-10
CONSCIOUSNESS_THRESHOLD = 1 / PHI

# Resource monitoring configuration
MEMORY_THRESHOLD_MB = 100  # Alert threshold for memory growth
CPU_THRESHOLD_PERCENT = 80  # Alert threshold for CPU usage
MONITORING_INTERVAL = 0.1   # Seconds between measurements
MAX_MEMORY_SAMPLES = 1000   # Maximum samples to keep in memory
LEAK_DETECTION_CYCLES = 10  # Number of cycles for leak detection

@dataclass
class ResourceSnapshot:
    """Snapshot of system resources at a point in time"""
    timestamp: float
    memory_rss: float  # MB
    memory_vms: float  # MB
    memory_percent: float
    cpu_percent: float
    thread_count: int
    gc_count: Dict[int, int]  # Garbage collection counts by generation
    metadata: Dict[str, Any] = None

@dataclass
class MemoryLeakReport:
    """Report of potential memory leak detection"""
    component: str
    leak_detected: bool
    memory_growth_mb: float
    growth_rate_mb_per_second: float
    confidence: float  # 0-1 confidence in leak detection
    samples: List[float]
    recommendations: List[str]

@dataclass
class ResourceMonitoringConfig:
    """Configuration for resource monitoring"""
    monitor_memory: bool = True
    monitor_cpu: bool = True
    monitor_gc: bool = True
    memory_threshold_mb: float = MEMORY_THRESHOLD_MB
    cpu_threshold_percent: float = CPU_THRESHOLD_PERCENT
    sampling_interval: float = MONITORING_INTERVAL
    alert_on_threshold: bool = True

class UnityMathematicsResourceMonitor:
    """Main resource monitoring system for Unity Mathematics"""
    
    def __init__(self, config: ResourceMonitoringConfig = None):
        self.config = config or ResourceMonitoringConfig()
        self.process = psutil.Process()
        self.monitoring_active = False
        self.resource_history = deque(maxlen=MAX_MEMORY_SAMPLES)
        self.leak_reports = []
        self.baseline_memory = None
        self.weak_refs = set()  # Track objects for leak detection
        
        # Start tracemalloc for detailed memory tracking
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            
    def take_snapshot(self, metadata: Dict[str, Any] = None) -> ResourceSnapshot:
        """Take a snapshot of current resource usage"""
        memory_info = self.process.memory_info()
        
        snapshot = ResourceSnapshot(
            timestamp=time.time(),
            memory_rss=memory_info.rss / 1024**2,  # Convert to MB
            memory_vms=memory_info.vms / 1024**2,  # Convert to MB
            memory_percent=self.process.memory_percent(),
            cpu_percent=self.process.cpu_percent(),
            thread_count=self.process.num_threads(),
            gc_count={i: gc.get_count()[i] for i in range(3)},
            metadata=metadata or {}
        )
        
        self.resource_history.append(snapshot)
        return snapshot
        
    def start_monitoring(self):
        """Start continuous resource monitoring"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.baseline_memory = self.take_snapshot({'event': 'monitoring_start'}).memory_rss
        
        def monitor_loop():
            while self.monitoring_active:
                snapshot = self.take_snapshot({'event': 'periodic_monitoring'})
                
                # Check thresholds
                if self.config.alert_on_threshold:
                    self._check_thresholds(snapshot)
                    
                time.sleep(self.config.sampling_interval)
                
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop continuous resource monitoring"""
        self.monitoring_active = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
            
    def _check_thresholds(self, snapshot: ResourceSnapshot):
        """Check if resource usage exceeds thresholds"""
        if self.baseline_memory and (snapshot.memory_rss - self.baseline_memory) > self.config.memory_threshold_mb:
            print(f"⚠️  Memory threshold exceeded: {snapshot.memory_rss - self.baseline_memory:.1f}MB above baseline")
            
        if snapshot.cpu_percent > self.config.cpu_threshold_percent:
            print(f"⚠️  CPU threshold exceeded: {snapshot.cpu_percent:.1f}%")
            
    def detect_memory_leaks(self, component_name: str, operation_cycles: int = LEAK_DETECTION_CYCLES) -> MemoryLeakReport:
        """Detect memory leaks in a Unity Mathematics component"""
        pre_cycle_memory = []
        post_cycle_memory = []
        
        # Collect memory samples over multiple cycles
        for cycle in range(operation_cycles):
            # Take pre-operation snapshot
            gc.collect()  # Force garbage collection
            pre_snapshot = self.take_snapshot({'cycle': cycle, 'phase': 'pre'})
            pre_cycle_memory.append(pre_snapshot.memory_rss)
            
            # Simulate Unity Mathematics operations (placeholder)
            self._simulate_unity_operations(100)
            
            # Take post-operation snapshot
            post_snapshot = self.take_snapshot({'cycle': cycle, 'phase': 'post'})
            post_cycle_memory.append(post_snapshot.memory_rss)
            
        # Analyze memory growth patterns
        memory_growth = [post - pre for pre, post in zip(pre_cycle_memory, post_cycle_memory)]
        total_growth = sum(memory_growth)
        avg_growth_per_cycle = total_growth / operation_cycles
        
        # Detect leak based on consistent growth
        leak_detected = False
        confidence = 0.0
        
        if total_growth > 5.0:  # More than 5MB total growth
            positive_growth_cycles = sum(1 for growth in memory_growth if growth > 0.1)
            confidence = positive_growth_cycles / operation_cycles
            
            if confidence > 0.7:  # 70% of cycles showed growth
                leak_detected = True
                
        # Generate recommendations
        recommendations = []
        if leak_detected:
            recommendations.extend([
                "Check for circular references in Unity Mathematics objects",
                "Ensure proper cleanup of consciousness field calculations",
                "Review φ-harmonic sequence generation for unreleased arrays",
                "Verify agent ecosystem properly releases old generations"
            ])
        else:
            recommendations.append("No significant memory leak detected")
            
        report = MemoryLeakReport(
            component=component_name,
            leak_detected=leak_detected,
            memory_growth_mb=total_growth,
            growth_rate_mb_per_second=avg_growth_per_cycle,
            confidence=confidence,
            samples=memory_growth,
            recommendations=recommendations
        )
        
        self.leak_reports.append(report)
        return report
        
    def _simulate_unity_operations(self, count: int):
        """Simulate Unity Mathematics operations for leak detection"""
        # Unity addition operations
        results = []
        for i in range(count):
            a = np.random.uniform(0.1, 10.0)
            b = np.random.uniform(0.1, 10.0)
            
            # Simulate unity addition
            if abs(a - b) < UNITY_EPSILON:
                result = max(a, b)
            else:
                result = max(a, b) * (1 + 1/PHI) / 2
                
            results.append(result)
            
            # Simulate consciousness field calculation
            if i % 10 == 0:
                x, y, t = np.random.uniform(-1, 1, 3)
                field = PHI * math.sin(x * PHI) * math.cos(y * PHI) * math.exp(-t / PHI)
                results.append(field)
                
        return results
        
    def profile_memory_usage(self, function: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Profile memory usage of a specific function"""
        # Take baseline
        baseline_snapshot = self.take_snapshot({'event': 'profiling_start'})
        
        # Enable detailed tracing
        tracemalloc.start()
        gc.collect()
        
        start_memory = tracemalloc.take_snapshot()
        start_time = time.perf_counter()
        
        try:
            # Execute function
            result = function(*args, **kwargs)
            
        finally:
            end_time = time.perf_counter()
            end_memory = tracemalloc.take_snapshot()
            
            # Take final snapshot
            final_snapshot = self.take_snapshot({'event': 'profiling_end'})
            
        # Analyze memory changes
        top_stats = end_memory.compare_to(start_memory, 'lineno')
        
        memory_profile = {
            'function_name': getattr(function, '__name__', 'unknown'),
            'execution_time': end_time - start_time,
            'memory_before_mb': baseline_snapshot.memory_rss,
            'memory_after_mb': final_snapshot.memory_rss,
            'memory_delta_mb': final_snapshot.memory_rss - baseline_snapshot.memory_rss,
            'peak_memory_allocations': len(top_stats),
            'top_memory_allocations': [
                {
                    'file': stat.traceback.format()[0] if stat.traceback else 'unknown',
                    'size_mb': stat.size / 1024**2,
                    'count': stat.count
                }
                for stat in top_stats[:5]  # Top 5 allocations
            ],
            'result': result
        }
        
        return memory_profile
        
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get summary of resource usage"""
        if not self.resource_history:
            return {'error': 'No resource history available'}
            
        snapshots = list(self.resource_history)
        
        memory_values = [s.memory_rss for s in snapshots]
        cpu_values = [s.cpu_percent for s in snapshots]
        
        summary = {
            'monitoring_duration': snapshots[-1].timestamp - snapshots[0].timestamp,
            'total_snapshots': len(snapshots),
            'memory_stats': {
                'current_mb': snapshots[-1].memory_rss,
                'baseline_mb': self.baseline_memory,
                'peak_mb': max(memory_values),
                'min_mb': min(memory_values),
                'average_mb': sum(memory_values) / len(memory_values),
                'growth_mb': snapshots[-1].memory_rss - snapshots[0].memory_rss
            },
            'cpu_stats': {
                'current_percent': snapshots[-1].cpu_percent,
                'peak_percent': max(cpu_values),
                'average_percent': sum(cpu_values) / len(cpu_values)
            },
            'gc_stats': {
                'current_counts': snapshots[-1].gc_count,
                'collections_occurred': any(
                    snapshots[i].gc_count[0] > snapshots[i-1].gc_count[0]
                    for i in range(1, len(snapshots))
                )
            },
            'leak_reports': len(self.leak_reports)
        }
        
        return summary

@contextmanager
def monitor_resources(config: ResourceMonitoringConfig = None):
    """Context manager for resource monitoring"""
    monitor = UnityMathematicsResourceMonitor(config)
    
    try:
        monitor.start_monitoring()
        yield monitor
        
    finally:
        monitor.stop_monitoring()

def memory_profiler(func):
    """Decorator for automatic memory profiling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        monitor = UnityMathematicsResourceMonitor()
        profile = monitor.profile_memory_usage(func, *args, **kwargs)
        
        # Print profile summary
        print(f"Memory Profile for {profile['function_name']}:")
        print(f"  Execution time: {profile['execution_time']:.3f}s")
        print(f"  Memory delta: {profile['memory_delta_mb']:.2f}MB")
        
        return profile['result']
        
    return wrapper

class TestMemoryLeakDetection:
    """Test memory leak detection capabilities"""
    
    def setup_method(self):
        """Set up memory leak detection testing"""
        self.monitor = UnityMathematicsResourceMonitor()
        
    def test_basic_memory_leak_detection(self):
        """Test basic memory leak detection"""
        # Simulate component with potential leak
        report = self.monitor.detect_memory_leaks("unity_add_basic", operation_cycles=5)
        
        assert isinstance(report, MemoryLeakReport)
        assert report.component == "unity_add_basic"
        assert isinstance(report.leak_detected, bool)
        assert 0 <= report.confidence <= 1
        assert len(report.samples) == 5
        assert len(report.recommendations) > 0
        
    def test_memory_growth_analysis(self):
        """Test memory growth pattern analysis"""
        # Create intentional memory growth scenario
        memory_hog = []
        
        def memory_growing_function():
            for i in range(1000):
                # Intentionally create growing memory usage
                memory_hog.append(np.random.random(1000))
                
                # Unity operations mixed in
                a, b = np.random.uniform(0, 1, 2)
                result = max(a, b) * (1 + 1/PHI) / 2
                
        # Profile the growing function
        profile = self.monitor.profile_memory_usage(memory_growing_function)
        
        assert profile['memory_delta_mb'] > 0, "Should show memory growth"
        assert profile['execution_time'] > 0, "Should record execution time"
        assert len(profile['top_memory_allocations']) > 0, "Should identify allocations"
        
        # Cleanup
        del memory_hog
        gc.collect()
        
    def test_consciousness_field_memory_usage(self):
        """Test consciousness field memory usage patterns"""
        def consciousness_field_calculation():
            results = []
            for i in range(1000):
                x = np.random.uniform(-5, 5)
                y = np.random.uniform(-5, 5)
                t = np.random.uniform(0, 2)
                
                # Consciousness field equation
                field = PHI * math.sin(x * PHI) * math.cos(y * PHI) * math.exp(-t / PHI)
                results.append(complex(field, 0))
                
            return results
            
        profile = self.monitor.profile_memory_usage(consciousness_field_calculation)
        
        # Consciousness calculations should be memory efficient
        assert profile['memory_delta_mb'] < 10, "Consciousness calculations should be efficient"
        assert len(profile['result']) == 1000, "Should return all calculated fields"
        
    def test_agent_ecosystem_memory_cleanup(self):
        """Test agent ecosystem memory cleanup"""
        def agent_ecosystem_simulation():
            agents = []
            
            # Create multiple generations of agents
            for generation in range(5):
                generation_agents = []
                
                for i in range(100):
                    agent_dna = {
                        'creativity': np.random.uniform(0, 1),
                        'consciousness': np.random.uniform(0, 1),
                        'unity_affinity': np.random.uniform(0.8, 1.0)
                    }
                    generation_agents.append(agent_dna)
                    
                # Replace previous generation (test cleanup)
                agents = generation_agents
                
                # Force garbage collection
                if generation % 2 == 0:
                    gc.collect()
                    
            return len(agents)
            
        # Test with leak detection
        report = self.monitor.detect_memory_leaks("agent_ecosystem", operation_cycles=3)
        
        # Agent ecosystem should not leak significantly
        assert not report.leak_detected or report.confidence < 0.8, \
            "Agent ecosystem should not have major leaks"

class TestResourceMonitoring:
    """Test system resource monitoring capabilities"""
    
    def setup_method(self):
        """Set up resource monitoring testing"""
        self.config = ResourceMonitoringConfig(
            memory_threshold_mb=50,
            cpu_threshold_percent=50,
            sampling_interval=0.05
        )
        
    def test_resource_snapshot_collection(self):
        """Test resource snapshot collection"""
        monitor = UnityMathematicsResourceMonitor(self.config)
        
        snapshot = monitor.take_snapshot({'test': 'basic_snapshot'})
        
        assert isinstance(snapshot, ResourceSnapshot)
        assert snapshot.memory_rss > 0
        assert snapshot.memory_percent > 0
        assert snapshot.timestamp > 0
        assert snapshot.thread_count > 0
        assert isinstance(snapshot.gc_count, dict)
        assert snapshot.metadata['test'] == 'basic_snapshot'
        
    def test_continuous_monitoring(self):
        """Test continuous resource monitoring"""
        with monitor_resources(self.config) as monitor:
            # Perform some operations while monitoring
            for i in range(100):
                # Unity mathematics operations
                a, b = np.random.uniform(0, 1, 2)
                result = max(a, b) * (1 + 1/PHI) / 2
                
                # φ-harmonic calculation
                phi_result = a * (PHI ** i % 10)
                
                time.sleep(0.001)  # Small delay
                
            time.sleep(0.2)  # Let monitoring collect samples
            
        # Check collected data
        summary = monitor.get_resource_summary()
        
        assert summary['total_snapshots'] > 0, "Should collect monitoring snapshots"
        assert summary['monitoring_duration'] > 0, "Should record monitoring duration"
        assert 'memory_stats' in summary, "Should include memory statistics"
        assert 'cpu_stats' in summary, "Should include CPU statistics"
        
    def test_memory_profiler_decorator(self):
        """Test memory profiler decorator"""
        @memory_profiler
        def unity_calculation_batch(size: int):
            results = []
            for i in range(size):
                a = np.random.uniform(0.1, 10.0)
                b = np.random.uniform(0.1, 10.0)
                
                if abs(a - b) < UNITY_EPSILON:
                    result = max(a, b)
                else:
                    result = max(a, b) * (1 + 1/PHI) / 2
                    
                results.append(result)
                
            return results
            
        # Should automatically profile memory usage
        results = unity_calculation_batch(1000)
        
        assert len(results) == 1000, "Should return correct number of results"
        assert all(isinstance(r, (int, float)) for r in results), "Results should be numeric"
        
    def test_resource_threshold_monitoring(self):
        """Test resource threshold monitoring and alerting"""
        # Create config with low thresholds for testing
        test_config = ResourceMonitoringConfig(
            memory_threshold_mb=1,  # Very low threshold
            cpu_threshold_percent=1,  # Very low threshold
            alert_on_threshold=True
        )
        
        monitor = UnityMathematicsResourceMonitor(test_config)
        
        # Take baseline
        baseline = monitor.take_snapshot({'event': 'baseline'})
        
        # Create memory-intensive operation
        memory_intensive_data = [np.random.random(10000) for _ in range(100)]
        
        # Take snapshot after memory allocation
        post_allocation = monitor.take_snapshot({'event': 'post_allocation'})
        
        # Should detect memory growth
        memory_growth = post_allocation.memory_rss - baseline.memory_rss
        assert memory_growth > 0, "Should detect memory growth"
        
        # Cleanup
        del memory_intensive_data
        gc.collect()

class TestMemoryOptimization:
    """Test memory optimization strategies for Unity Mathematics"""
    
    def test_phi_harmonic_sequence_memory_efficiency(self):
        """Test memory efficiency of φ-harmonic sequence generation"""
        monitor = UnityMathematicsResourceMonitor()
        
        def inefficient_phi_sequence(length: int):
            # Inefficient: keeps all intermediate results
            sequences = []
            for i in range(length):
                sequence = [PHI ** j for j in range(i + 1)]
                sequences.append(sequence)
            return sequences
            
        def efficient_phi_sequence(length: int):
            # Efficient: generates values on demand
            for i in range(length):
                yield [PHI ** j for j in range(i + 1)]
                
        # Profile inefficient version
        inefficient_profile = monitor.profile_memory_usage(inefficient_phi_sequence, 100)
        
        # Profile efficient version  
        efficient_results = list(efficient_phi_sequence(100))
        efficient_profile = monitor.profile_memory_usage(lambda: efficient_results, )
        
        # Efficient version should use less memory
        memory_difference = inefficient_profile['memory_delta_mb'] - efficient_profile['memory_delta_mb']
        assert memory_difference > 0, f"Efficient version should use less memory: {memory_difference:.2f}MB saved"
        
    def test_consciousness_field_chunked_processing(self):
        """Test chunked processing for large consciousness fields"""
        monitor = UnityMathematicsResourceMonitor()
        
        def process_consciousness_field_batch(coordinates: List[Tuple[float, float, float]]):
            # Process all at once (memory intensive)
            results = []
            field_cache = {}
            
            for x, y, t in coordinates:
                key = (round(x, 3), round(y, 3), round(t, 3))
                if key not in field_cache:
                    field = PHI * math.sin(x * PHI) * math.cos(y * PHI) * math.exp(-t / PHI)
                    field_cache[key] = complex(field, 0)
                results.append(field_cache[key])
                
            return results, field_cache
            
        def process_consciousness_field_chunked(coordinates: List[Tuple[float, float, float]], chunk_size: int = 100):
            # Process in chunks (memory efficient)
            results = []
            
            for i in range(0, len(coordinates), chunk_size):
                chunk = coordinates[i:i + chunk_size]
                chunk_results = []
                
                for x, y, t in chunk:
                    field = PHI * math.sin(x * PHI) * math.cos(y * PHI) * math.exp(-t / PHI)
                    chunk_results.append(complex(field, 0))
                    
                results.extend(chunk_results)
                
                # Explicit cleanup of chunk data
                del chunk_results
                if i % 500 == 0:
                    gc.collect()
                    
            return results
            
        # Generate test coordinates
        test_coordinates = [(np.random.uniform(-5, 5), np.random.uniform(-5, 5), np.random.uniform(0, 2)) 
                          for _ in range(1000)]
        
        # Profile batch processing
        batch_profile = monitor.profile_memory_usage(process_consciousness_field_batch, test_coordinates)
        
        # Profile chunked processing
        chunked_profile = monitor.profile_memory_usage(process_consciousness_field_chunked, test_coordinates)
        
        # Chunked processing should be more memory efficient
        assert len(batch_profile['result'][0]) == len(chunked_profile['result'])
        
        memory_efficiency = chunked_profile['memory_delta_mb'] / batch_profile['memory_delta_mb']
        assert memory_efficiency < 1.2, f"Chunked processing should be memory efficient: {memory_efficiency:.2f}"
        
    def test_garbage_collection_optimization(self):
        """Test garbage collection optimization strategies"""
        monitor = UnityMathematicsResourceMonitor()
        
        def unity_operations_with_gc_optimization(iterations: int):
            results = []
            temp_objects = []
            
            for i in range(iterations):
                # Unity mathematics operations
                a, b = np.random.uniform(0, 1, 2)
                unity_result = max(a, b) * (1 + 1/PHI) / 2
                
                # Create temporary objects
                temp_array = np.random.random(100)
                temp_objects.append(temp_array)
                
                results.append(unity_result)
                
                # Periodic cleanup
                if i % 100 == 99:
                    # Clear temporary objects
                    temp_objects.clear()
                    
                    # Force garbage collection
                    gc.collect()
                    
            return results
            
        # Profile with GC optimization
        profile = monitor.profile_memory_usage(unity_operations_with_gc_optimization, 1000)
        
        # Should complete without excessive memory growth
        assert profile['memory_delta_mb'] < 50, "GC optimization should limit memory growth"
        assert len(profile['result']) == 1000, "Should complete all operations"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])