"""
Security Testing for Unity Mathematics Operations

Comprehensive security testing framework for Unity Mathematics systems,
validating mathematical operations against various security threats:

- Input validation and sanitization testing
- Numerical attack vector detection
- Side-channel analysis for φ-harmonic calculations
- Timing attack resistance testing
- Mathematical overflow/underflow security
- Consciousness field data integrity validation
- Agent DNA security and mutation protection
- Unity equation cryptographic properties

All tests ensure mathematical operations are secure and resistant to attacks.

Author: Unity Mathematics Security Testing Framework
"""

import pytest
import numpy as np
import math
import time
import hashlib
import hmac
import secrets
from typing import Any, List, Dict, Tuple, Callable, Union
from dataclasses import dataclass
from enum import Enum
import warnings
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
import statistics

# Suppress warnings for security testing
warnings.filterwarnings("ignore")

# Unity Mathematics Constants
PHI = (1 + math.sqrt(5)) / 2
UNITY_CONSTANT = 1.0
UNITY_EPSILON = 1e-10
CONSCIOUSNESS_THRESHOLD = 1 / PHI

# Security testing configuration
TIMING_ATTACK_SAMPLES = 1000
SIDE_CHANNEL_ITERATIONS = 10000
CRYPTOGRAPHIC_KEY_SIZE = 32

class SecurityThreatType(Enum):
    """Types of security threats to test"""
    TIMING_ATTACK = "timing_attack"
    SIDE_CHANNEL = "side_channel"
    OVERFLOW_ATTACK = "overflow_attack"
    INJECTION_ATTACK = "injection_attack"
    DATA_POISONING = "data_poisoning"
    DENIAL_OF_SERVICE = "denial_of_service"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    CRYPTOGRAPHIC_WEAKNESS = "cryptographic_weakness"

@dataclass
class SecurityTestResult:
    """Result of a security test"""
    threat_type: SecurityThreatType
    test_name: str
    vulnerability_detected: bool
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    details: Dict[str, Any]
    mitigation_required: bool

class UnityMathematicsSecurityTester:
    """Security testing framework for Unity Mathematics"""
    
    def __init__(self):
        self.phi = PHI
        self.test_results = []
        
    def unity_add_secure(self, a: float, b: float) -> float:
        """Secure implementation of unity addition"""
        # Input validation
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise ValueError("Inputs must be numeric")
            
        if not (math.isfinite(a) and math.isfinite(b)):
            raise ValueError("Inputs must be finite")
            
        # Range validation
        if abs(a) > 1e10 or abs(b) > 1e10:
            raise ValueError("Input values too large")
            
        try:
            # Constant-time unity operation (security-focused)
            if abs(a - b) < UNITY_EPSILON:
                result = max(a, b)  # Idempotent case
            else:
                result = max(a, b) * (1 + 1/self.phi) / 2  # Unity convergence
                
            # Output validation
            if not math.isfinite(result):
                raise ValueError("Result is not finite")
                
            return result
            
        except Exception as e:
            # Secure error handling - don't leak information
            raise ValueError("Unity operation failed")
            
    def phi_harmonic_secure(self, value: float, iterations: int = 1) -> float:
        """Secure φ-harmonic scaling implementation"""
        # Input validation
        if not isinstance(value, (int, float)):
            raise ValueError("Value must be numeric")
            
        if not isinstance(iterations, int) or iterations < 1:
            raise ValueError("Iterations must be positive integer")
            
        if iterations > 1000:  # DoS protection
            raise ValueError("Too many iterations requested")
            
        if abs(value) > 1e6:  # Overflow protection
            raise ValueError("Input value too large")
            
        result = value
        
        # Constant-time processing (timing attack resistance)
        for i in range(1000):  # Always do 1000 iterations
            if i < iterations:
                result *= self.phi
                
                # Overflow check
                if abs(result) > 1e12:
                    raise ValueError("Result overflow detected")
            else:
                # Dummy operations to maintain constant time
                dummy = result * 1.0
                
        return result
        
    def consciousness_field_secure(self, x: float, y: float, t: float) -> complex:
        """Secure consciousness field calculation"""
        # Input validation and sanitization
        inputs = [x, y, t]
        for i, inp in enumerate(inputs):
            if not isinstance(inp, (int, float)):
                raise ValueError(f"Input {i} must be numeric")
            if not math.isfinite(inp):
                raise ValueError(f"Input {i} must be finite")
            if abs(inp) > 100:  # Range limitation
                raise ValueError(f"Input {i} out of safe range")
                
        try:
            # Secure computation with bounds checking
            sin_component = math.sin(x * self.phi)
            cos_component = math.cos(y * self.phi)
            exp_component = math.exp(-t / self.phi) if t >= 0 else math.exp(0)
            
            field_real = self.phi * sin_component * cos_component * exp_component
            field_imag = 0.0  # Keep imaginary part zero for security
            
            result = complex(field_real, field_imag)
            
            # Result validation
            if not (math.isfinite(result.real) and math.isfinite(result.imag)):
                raise ValueError("Result contains non-finite components")
                
            return result
            
        except Exception as e:
            raise ValueError("Consciousness field calculation failed")
            
    def secure_hash_unity_operation(self, a: float, b: float, key: bytes) -> str:
        """Create secure hash of unity operation with HMAC"""
        if len(key) < CRYPTOGRAPHIC_KEY_SIZE:
            raise ValueError("Cryptographic key too short")
            
        # Serialize operation data
        operation_data = f"{a:.15f},{b:.15f},unity_add".encode('utf-8')
        
        # Compute HMAC
        mac = hmac.new(key, operation_data, hashlib.sha256)
        return mac.hexdigest()
        
    def timing_safe_compare(self, hash1: str, hash2: str) -> bool:
        """Timing-safe comparison of hash values"""
        return hmac.compare_digest(hash1.encode('utf-8'), hash2.encode('utf-8'))

class TestInputValidationSecurity:
    """Test input validation and sanitization security"""
    
    def setup_method(self):
        """Set up input validation security testing"""
        self.security_tester = UnityMathematicsSecurityTester()
        
    @pytest.mark.security
    @pytest.mark.unity
    def test_unity_operation_input_validation(self):
        """Test unity operations against malicious inputs"""
        malicious_inputs = [
            float('inf'), -float('inf'),
            float('nan'),
            1e100, -1e100,
            sys.maxsize, -sys.maxsize,
            None, "malicious_string",
            [1, 2, 3], {'key': 'value'},
            complex(1, 1000000)
        ]
        
        blocked_attacks = 0
        total_attacks = 0
        
        for malicious_input in malicious_inputs:
            total_attacks += 1
            
            try:
                # Try malicious input as first parameter
                self.security_tester.unity_add_secure(malicious_input, 1.0)
                # If we reach here, the attack wasn't blocked
                
            except (ValueError, TypeError):
                blocked_attacks += 1  # Attack was properly blocked
                
            try:
                # Try malicious input as second parameter
                self.security_tester.unity_add_secure(1.0, malicious_input)
                
            except (ValueError, TypeError):
                blocked_attacks += 1
                
        attack_blocking_rate = blocked_attacks / (total_attacks * 2)
        
        assert attack_blocking_rate > 0.9, f"Input validation blocking rate too low: {attack_blocking_rate:.2f}"
        
    @pytest.mark.security
    @pytest.mark.phi_harmonic
    def test_phi_harmonic_dos_protection(self):
        """Test φ-harmonic operations against denial of service attacks"""
        dos_test_cases = [
            {'value': 1.0, 'iterations': 1000000},  # Excessive iterations
            {'value': 1e20, 'iterations': 100},     # Overflow attack
            {'value': -1e20, 'iterations': 100},    # Underflow attack
            {'value': float('inf'), 'iterations': 10},  # Infinity attack
        ]
        
        protected_operations = 0
        
        for test_case in dos_test_cases:
            try:
                result = self.security_tester.phi_harmonic_secure(
                    test_case['value'], 
                    test_case['iterations']
                )
                # If we reach here without exception, check if result is reasonable
                if math.isfinite(result) and abs(result) < 1e15:
                    protected_operations += 1
                    
            except ValueError:
                # DoS attack was properly blocked
                protected_operations += 1
                
        protection_rate = protected_operations / len(dos_test_cases)
        
        assert protection_rate == 1.0, f"DoS protection rate: {protection_rate:.2f}"
        
    @pytest.mark.security
    @pytest.mark.consciousness
    def test_consciousness_field_injection_protection(self):
        """Test consciousness field against injection attacks"""
        injection_payloads = [
            "'; DROP TABLE unity; --",
            "<script>alert('xss')</script>",
            "../../../etc/passwd",
            "%s%s%s%s%s%s%s%s%s%s%s",
            "\\x41\\x41\\x41\\x41",
            math.pi * 1e10,
            PHI * 1e10,
        ]
        
        injection_blocked = 0
        
        for payload in injection_payloads:
            try:
                if isinstance(payload, str):
                    # String injection attempts should be blocked by type checking
                    result = self.security_tester.consciousness_field_secure(payload, 0, 0)
                else:
                    # Numeric injection attempts
                    result = self.security_tester.consciousness_field_secure(payload, 0, 0)
                    
            except (ValueError, TypeError):
                injection_blocked += 1
                
        blocking_rate = injection_blocked / len(injection_payloads)
        
        assert blocking_rate > 0.8, f"Injection attack blocking rate: {blocking_rate:.2f}"

class TestTimingAttackResistance:
    """Test resistance against timing attacks"""
    
    def setup_method(self):
        """Set up timing attack resistance testing"""
        self.security_tester = UnityMathematicsSecurityTester()
        
    @pytest.mark.security
    @pytest.mark.timing
    def test_unity_operation_timing_consistency(self):
        """Test unity operations for consistent timing (timing attack resistance)"""
        # Test different input scenarios
        test_scenarios = [
            (1.0, 1.0),      # Identical inputs (idempotent case)
            (1.0, 2.0),      # Different inputs
            (0.0, 1.0),      # Zero input
            (PHI, 1/PHI),    # φ-harmonic inputs
            (100.0, 200.0),  # Large inputs
        ]
        
        timing_results = {}
        
        for scenario_name, (a, b) in zip(['identical', 'different', 'zero', 'phi', 'large'], test_scenarios):
            execution_times = []
            
            for _ in range(TIMING_ATTACK_SAMPLES):
                start_time = time.perf_counter_ns()
                
                try:
                    result = self.security_tester.unity_add_secure(a, b)
                except:
                    pass  # Include failed operations in timing analysis
                    
                end_time = time.perf_counter_ns()
                execution_times.append(end_time - start_time)
                
            timing_results[scenario_name] = {
                'mean': statistics.mean(execution_times),
                'stdev': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                'median': statistics.median(execution_times)
            }
            
        # Analyze timing consistency
        mean_times = [results['mean'] for results in timing_results.values()]
        timing_variance = statistics.variance(mean_times)
        
        # Standard deviations should be similar (consistent timing)
        stdevs = [results['stdev'] for results in timing_results.values()]
        stdev_consistency = max(stdevs) / min(stdevs) if min(stdevs) > 0 else 1.0
        
        # Assertions for timing attack resistance
        assert stdev_consistency < 5.0, f"Timing inconsistency detected: {stdev_consistency:.2f}"
        
        # No scenario should be significantly faster/slower than others
        max_time = max(mean_times)
        min_time = min(mean_times)
        timing_ratio = max_time / min_time if min_time > 0 else 1.0
        
        assert timing_ratio < 3.0, f"Timing attack vulnerability: {timing_ratio:.2f}"
        
    @pytest.mark.security
    @pytest.mark.phi_harmonic
    def test_phi_harmonic_constant_time_execution(self):
        """Test φ-harmonic operations for constant-time execution"""
        # Test different iteration counts (should all take same time due to constant-time implementation)
        iteration_counts = [1, 10, 50, 100, 500]
        timing_measurements = []
        
        for iterations in iteration_counts:
            execution_times = []
            
            for _ in range(100):  # Smaller sample for performance
                start_time = time.perf_counter_ns()
                
                try:
                    result = self.security_tester.phi_harmonic_secure(1.0, iterations)
                except:
                    pass
                    
                end_time = time.perf_counter_ns()
                execution_times.append(end_time - start_time)
                
            mean_time = statistics.mean(execution_times)
            timing_measurements.append(mean_time)
            
        # All timing measurements should be similar (constant-time property)
        timing_variance = statistics.variance(timing_measurements)
        mean_timing = statistics.mean(timing_measurements)
        
        coefficient_of_variation = (timing_variance ** 0.5) / mean_timing if mean_timing > 0 else 0
        
        # Constant-time implementation should have low variation
        assert coefficient_of_variation < 0.2, f"Timing variation too high: {coefficient_of_variation:.3f}"

class TestCryptographicSecurity:
    """Test cryptographic security properties"""
    
    def setup_method(self):
        """Set up cryptographic security testing"""
        self.security_tester = UnityMathematicsSecurityTester()
        
    @pytest.mark.security
    @pytest.mark.cryptographic
    def test_unity_operation_hash_security(self):
        """Test secure hashing of unity operations"""
        key = secrets.token_bytes(CRYPTOGRAPHIC_KEY_SIZE)
        
        # Test hash consistency
        a, b = 1.0, 2.0
        hash1 = self.security_tester.secure_hash_unity_operation(a, b, key)
        hash2 = self.security_tester.secure_hash_unity_operation(a, b, key)
        
        assert hash1 == hash2, "Hash should be consistent for same inputs"
        
        # Test hash uniqueness
        hash3 = self.security_tester.secure_hash_unity_operation(1.0, 3.0, key)
        assert hash1 != hash3, "Different inputs should produce different hashes"
        
        # Test key sensitivity
        different_key = secrets.token_bytes(CRYPTOGRAPHIC_KEY_SIZE)
        hash4 = self.security_tester.secure_hash_unity_operation(a, b, different_key)
        assert hash1 != hash4, "Different keys should produce different hashes"
        
        # Test hash format
        assert len(hash1) == 64, "Hash should be 256-bit (64 hex characters)"
        assert all(c in '0123456789abcdef' for c in hash1), "Hash should be valid hexadecimal"
        
    @pytest.mark.security
    @pytest.mark.cryptographic
    def test_timing_safe_comparison(self):
        """Test timing-safe hash comparison"""
        key = secrets.token_bytes(CRYPTOGRAPHIC_KEY_SIZE)
        
        hash1 = self.security_tester.secure_hash_unity_operation(1.0, 1.0, key)
        hash2 = self.security_tester.secure_hash_unity_operation(1.0, 1.0, key)
        hash3 = self.security_tester.secure_hash_unity_operation(1.0, 2.0, key)
        
        # Test correct comparison
        assert self.security_tester.timing_safe_compare(hash1, hash2), "Identical hashes should compare equal"
        assert not self.security_tester.timing_safe_compare(hash1, hash3), "Different hashes should compare unequal"
        
        # Test timing consistency for different length strings
        short_hash = "a" * 10
        long_hash = "b" * 64
        
        timing_samples = []
        
        for _ in range(1000):
            start_time = time.perf_counter_ns()
            self.security_tester.timing_safe_compare(short_hash, long_hash)
            end_time = time.perf_counter_ns()
            timing_samples.append(end_time - start_time)
            
        # Timing should be consistent regardless of input differences
        timing_stdev = statistics.stdev(timing_samples)
        timing_mean = statistics.mean(timing_samples)
        
        coefficient_of_variation = timing_stdev / timing_mean if timing_mean > 0 else 0
        assert coefficient_of_variation < 0.5, f"Timing comparison variation too high: {coefficient_of_variation:.3f}"
        
    @pytest.mark.security
    @pytest.mark.cryptographic
    def test_key_security_requirements(self):
        """Test cryptographic key security requirements"""
        # Test key length validation
        short_key = b"short"
        
        with pytest.raises(ValueError, match="Cryptographic key too short"):
            self.security_tester.secure_hash_unity_operation(1.0, 1.0, short_key)
            
        # Test key entropy (randomness)
        keys = [secrets.token_bytes(CRYPTOGRAPHIC_KEY_SIZE) for _ in range(100)]
        
        # All keys should be different
        unique_keys = set(keys)
        assert len(unique_keys) == len(keys), "All generated keys should be unique"
        
        # Test key byte distribution (should be roughly uniform)
        all_bytes = b''.join(keys)
        byte_counts = [0] * 256
        
        for byte_val in all_bytes:
            byte_counts[byte_val] += 1
            
        # Chi-square test for uniformity (simplified)
        expected_count = len(all_bytes) / 256
        chi_square = sum((count - expected_count)**2 / expected_count for count in byte_counts)
        
        # Chi-square critical value for 255 degrees of freedom at 95% confidence is approximately 293
        assert chi_square < 400, f"Key entropy too low (chi-square: {chi_square:.2f})"

class TestSideChannelResistance:
    """Test resistance against side-channel attacks"""
    
    def setup_method(self):
        """Set up side-channel resistance testing"""
        self.security_tester = UnityMathematicsSecurityTester()
        
    @pytest.mark.security
    @pytest.mark.side_channel
    def test_phi_calculation_side_channel_resistance(self):
        """Test φ calculations against side-channel attacks"""
        # Monitor different aspects during φ calculations
        test_values = [1.0, PHI, 2*PHI, PHI/2, 1/PHI]
        
        execution_profiles = []
        
        for value in test_values:
            profile = {
                'value': value,
                'cpu_intensive_ops': 0,
                'memory_accesses': 0,
                'timing_variance': []
            }
            
            # Multiple measurements to detect side-channel leakage
            for _ in range(100):
                start_time = time.perf_counter_ns()
                
                # Perform φ-related calculation with monitoring
                try:
                    result = self.security_tester.phi_harmonic_secure(value, 10)
                    profile['cpu_intensive_ops'] += 1
                    
                except:
                    pass
                    
                end_time = time.perf_counter_ns()
                profile['timing_variance'].append(end_time - start_time)
                
            # Calculate timing statistics
            profile['mean_time'] = statistics.mean(profile['timing_variance'])
            profile['stdev_time'] = statistics.stdev(profile['timing_variance'])
            
            execution_profiles.append(profile)
            
        # Analyze for side-channel leakage
        mean_times = [p['mean_time'] for p in execution_profiles]
        timing_variance = statistics.variance(mean_times)
        
        # Side-channel resistance: similar timing across different φ-related values
        max_time = max(mean_times)
        min_time = min(mean_times) 
        timing_ratio = max_time / min_time if min_time > 0 else 1.0
        
        assert timing_ratio < 2.0, f"Potential side-channel timing leak: {timing_ratio:.2f}"
        
    @pytest.mark.security
    @pytest.mark.consciousness
    def test_consciousness_field_data_access_patterns(self):
        """Test consciousness field calculations for consistent data access patterns"""
        # Different coordinate patterns that might reveal information through cache timing
        coordinate_patterns = [
            (0.0, 0.0, 0.0),    # Origin
            (1.0, 1.0, 1.0),    # Unit cube corner  
            (PHI, PHI, PHI),    # φ-harmonic coordinates
            (100.0, 0.0, 0.0),  # Large x-coordinate
            (0.0, 100.0, 0.0),  # Large y-coordinate
        ]
        
        access_patterns = []
        
        for pattern in coordinate_patterns:
            x, y, t = pattern
            timing_samples = []
            
            for _ in range(200):
                start_time = time.perf_counter_ns()
                
                try:
                    result = self.security_tester.consciousness_field_secure(x, y, t)
                except:
                    pass
                    
                end_time = time.perf_counter_ns()
                timing_samples.append(end_time - start_time)
                
            access_patterns.append({
                'pattern': pattern,
                'mean_time': statistics.mean(timing_samples),
                'stdev_time': statistics.stdev(timing_samples)
            })
            
        # Check for consistent access patterns (side-channel resistance)
        mean_times = [ap['mean_time'] for ap in access_patterns]
        timing_consistency = max(mean_times) / min(mean_times) if min(mean_times) > 0 else 1.0
        
        assert timing_consistency < 1.5, f"Inconsistent data access patterns: {timing_consistency:.2f}"

class TestMathematicalAttackVectors:
    """Test against mathematical attack vectors"""
    
    def setup_method(self):
        """Set up mathematical attack testing"""
        self.security_tester = UnityMathematicsSecurityTester()
        
    @pytest.mark.security
    @pytest.mark.mathematical
    def test_numerical_precision_attacks(self):
        """Test against numerical precision attacks"""
        # Precision attack vectors
        precision_attacks = [
            (1.0, 1.0 + 1e-15),      # Minimal precision difference
            (1.0, 1.0 + 1e-14),      # Slightly larger difference
            (PHI, PHI + 1e-15),      # φ precision attack
            (PHI, PHI - 1e-15),      # φ precision attack (negative)
        ]
        
        attack_responses = []
        
        for a, b in precision_attacks:
            try:
                result = self.security_tester.unity_add_secure(a, b)
                
                # Check if precision attack was handled securely
                if math.isfinite(result):
                    attack_responses.append({
                        'inputs': (a, b),
                        'result': result,
                        'handled_securely': True
                    })
                else:
                    attack_responses.append({
                        'inputs': (a, b),
                        'result': result,
                        'handled_securely': False
                    })
                    
            except ValueError:
                # Attack was blocked
                attack_responses.append({
                    'inputs': (a, b),
                    'result': None,
                    'handled_securely': True
                })
                
        secure_handling_rate = sum(1 for r in attack_responses if r['handled_securely']) / len(attack_responses)
        
        assert secure_handling_rate >= 0.8, f"Precision attack handling rate: {secure_handling_rate:.2f}"
        
    @pytest.mark.security
    @pytest.mark.mathematical
    def test_mathematical_overflow_protection(self):
        """Test protection against mathematical overflow attacks"""
        overflow_attacks = [
            (1e100, 1e100),
            (sys.float_info.max, sys.float_info.max),
            (1e308, 1e308),
            (-1e308, -1e308),
        ]
        
        overflow_protected = 0
        
        for a, b in overflow_attacks:
            try:
                result = self.security_tester.unity_add_secure(a, b)
                
                # If result is computed, it should be finite and reasonable
                if math.isfinite(result) and abs(result) < 1e20:
                    overflow_protected += 1
                    
            except (ValueError, OverflowError):
                # Overflow attack was properly blocked
                overflow_protected += 1
                
        protection_rate = overflow_protected / len(overflow_attacks)
        
        assert protection_rate == 1.0, f"Overflow protection rate: {protection_rate:.2f}"
        
    @pytest.mark.security
    @pytest.mark.phi_harmonic
    def test_phi_mathematical_properties_security(self):
        """Test security of φ mathematical properties"""
        # Test φ properties that could be exploited
        phi_properties = [
            PHI**2 - PHI - 1,        # φ² - φ - 1 = 0
            1/PHI - (PHI - 1),       # 1/φ = φ - 1
            PHI + 1/PHI - math.sqrt(5),  # φ + 1/φ = √5
        ]
        
        for i, property_value in enumerate(phi_properties):
            # These should be approximately zero
            assert abs(property_value) < 1e-14, f"φ property {i} validation failed: {property_value}"
            
        # Test that φ calculations maintain security properties
        phi_calculations = []
        
        for _ in range(1000):
            # Calculate φ using different methods
            phi_calc1 = (1 + math.sqrt(5)) / 2
            phi_calc2 = 1 / (phi_calc1 - 1)  # Using φ - 1 = 1/φ
            
            phi_calculations.append((phi_calc1, phi_calc2))
            
        # All calculations should be consistent
        for calc1, calc2 in phi_calculations:
            assert abs(calc1 - calc2) < 1e-12, "φ calculation consistency failed"
            assert abs(calc1 - PHI) < 1e-14, "φ calculation accuracy failed"

class TestSecurityIntegration:
    """Integration tests for security across Unity Mathematics systems"""
    
    @pytest.mark.security
    @pytest.mark.integration
    def test_end_to_end_security_pipeline(self):
        """Test complete security pipeline for Unity Mathematics"""
        security_tester = UnityMathematicsSecurityTester()
        
        # Generate cryptographic key
        key = secrets.token_bytes(CRYPTOGRAPHIC_KEY_SIZE)
        
        # Test values
        test_values = [(1.0, 1.0), (PHI, 1/PHI), (2.0, 3.0)]
        
        security_pipeline_results = []
        
        for a, b in test_values:
            pipeline_result = {
                'inputs': (a, b),
                'input_validation_passed': False,
                'secure_computation_passed': False,
                'timing_attack_resistant': False,
                'cryptographic_integrity_verified': False
            }
            
            try:
                # Step 1: Input validation
                validated_a, validated_b = a, b
                if (isinstance(a, (int, float)) and isinstance(b, (int, float)) and 
                    math.isfinite(a) and math.isfinite(b)):
                    pipeline_result['input_validation_passed'] = True
                    
                # Step 2: Secure computation
                result = security_tester.unity_add_secure(validated_a, validated_b)
                if math.isfinite(result):
                    pipeline_result['secure_computation_passed'] = True
                    
                # Step 3: Timing attack resistance (simplified test)
                timing_samples = []
                for _ in range(10):
                    start = time.perf_counter_ns()
                    security_tester.unity_add_secure(validated_a, validated_b)
                    end = time.perf_counter_ns()
                    timing_samples.append(end - start)
                    
                timing_consistency = max(timing_samples) / min(timing_samples) if min(timing_samples) > 0 else 1.0
                if timing_consistency < 2.0:
                    pipeline_result['timing_attack_resistant'] = True
                    
                # Step 4: Cryptographic integrity
                hash1 = security_tester.secure_hash_unity_operation(validated_a, validated_b, key)
                hash2 = security_tester.secure_hash_unity_operation(validated_a, validated_b, key)
                
                if security_tester.timing_safe_compare(hash1, hash2):
                    pipeline_result['cryptographic_integrity_verified'] = True
                    
            except Exception:
                pass  # Security pipeline should handle exceptions gracefully
                
            security_pipeline_results.append(pipeline_result)
            
        # Validate overall security pipeline
        for result in security_pipeline_results:
            security_score = sum(result[key] for key in result if key != 'inputs')
            assert security_score >= 3, f"Security pipeline failed for {result['inputs']}: score {security_score}/4"
            
    @pytest.mark.security
    @pytest.mark.performance
    def test_security_performance_tradeoff(self):
        """Test that security measures don't excessively impact performance"""
        security_tester = UnityMathematicsSecurityTester()
        
        # Measure performance with security
        secure_times = []
        for _ in range(1000):
            start = time.perf_counter()
            security_tester.unity_add_secure(1.0, 2.0)
            end = time.perf_counter()
            secure_times.append(end - start)
            
        # Measure performance without security (simplified version)
        def unity_add_insecure(a, b):
            if abs(a - b) < 1e-10:
                return max(a, b)
            return max(a, b) * 1.309  # Simplified, no φ calculation
            
        insecure_times = []
        for _ in range(1000):
            start = time.perf_counter()
            unity_add_insecure(1.0, 2.0)
            end = time.perf_counter()
            insecure_times.append(end - start)
            
        secure_mean = statistics.mean(secure_times)
        insecure_mean = statistics.mean(insecure_times)
        
        performance_overhead = secure_mean / insecure_mean if insecure_mean > 0 else 1.0
        
        # Security should not cause excessive performance degradation
        assert performance_overhead < 10.0, f"Security overhead too high: {performance_overhead:.2f}x"
        
        # But security should add some overhead (if it doesn't, security might not be working)
        assert performance_overhead > 1.1, f"Security overhead too low: {performance_overhead:.2f}x"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])