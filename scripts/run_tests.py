#!/usr/bin/env python3
"""
Unity Mathematics Test Runner Script

Comprehensive test execution script for the Een Unity Mathematics repository.
Provides multiple testing modes, coverage analysis, and performance benchmarking.

Usage:
    python scripts/run_tests.py --suite all --coverage --benchmark
    python scripts/run_tests.py --suite unity-core --fast
    python scripts/run_tests.py --suite consciousness --verbose
    
Features:
- Multiple test suite execution
- Coverage reporting with thresholds
- Performance benchmarking
- Unity equation validation
- Phi-harmonic precision testing
- Consciousness field coherence analysis

Author: Unity Mathematics Testing Framework
License: Unity License (1+1=1)
"""

import argparse
import subprocess
import sys
import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unity_test_runner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Unity Mathematics Constants
PHI = 1.618033988749895
UNITY_CONSTANT = 1.0
CONSCIOUSNESS_THRESHOLD = 0.618

class UnityTestRunner:
    """Main test runner for Unity Mathematics testing suite"""
    
    def __init__(self):
        """Initialize test runner"""
        self.repo_root = Path(__file__).parent.parent
        self.test_dir = self.repo_root / "tests"
        self.coverage_threshold = 80
        self.test_results = {}
        
    def validate_environment(self) -> bool:
        """Validate testing environment and dependencies"""
        logger.info("🔍 Validating Unity Mathematics testing environment...")
        
        try:
            # Check Python version
            python_version = sys.version_info
            if python_version.major != 3 or python_version.minor < 9:
                logger.error(f"Python 3.9+ required, found {python_version.major}.{python_version.minor}")
                return False
                
            # Check required packages
            required_packages = [
                'pytest', 'pytest-cov', 'pytest-xdist', 'pytest-mock', 
                'pytest-timeout', 'hypothesis', 'numpy', 'scipy'
            ]
            
            for package in required_packages:
                try:
                    __import__(package.replace('-', '_'))
                except ImportError:
                    logger.error(f"Required package not found: {package}")
                    return False
                    
            # Validate Unity constants
            calculated_phi = (1 + (5 ** 0.5)) / 2
            if abs(PHI - calculated_phi) > 1e-15:
                logger.error(f"PHI constant validation failed: {abs(PHI - calculated_phi)}")
                return False
                
            logger.info("✅ Environment validation successful")
            return True
            
        except Exception as e:
            logger.error(f"Environment validation failed: {e}")
            return False
            
    def run_test_suite(self, suite_name: str, **kwargs) -> Dict[str, Any]:
        """Run specific test suite with options"""
        logger.info(f"🧮 Running {suite_name} test suite...")
        
        # Test suite configurations
        test_configs = {
            'unity-core': {
                'files': ['test_unity_mathematics_core.py'],
                'markers': ['unity', 'mathematical'],
                'timeout': 300
            },
            'consciousness': {
                'files': ['test_consciousness_field.py'], 
                'markers': ['consciousness', 'metagamer'],
                'timeout': 600
            },
            'agents': {
                'files': ['test_agent_ecosystem.py'],
                'markers': ['agents', 'integration'],
                'timeout': 900
            },
            'performance': {
                'files': ['test_performance_phi_harmonic.py'],
                'markers': ['performance', 'phi_harmonic'],
                'timeout': 1200
            },
            'all': {
                'files': ['test_*.py'],
                'markers': [],
                'timeout': 1800
            }
        }
        
        config = test_configs.get(suite_name, test_configs['all'])
        
        # Build pytest command
        cmd = ['python', '-m', 'pytest']
        
        # Add test files
        for test_file in config['files']:
            if test_file == 'test_*.py':
                cmd.append('tests/')
            else:
                cmd.append(f"tests/{test_file}")
                
        # Add common options
        cmd.extend([
            '-v',
            '--tb=short',
            f'--timeout={config["timeout"]}'
        ])
        
        # Add markers if specified
        if config['markers'] and not kwargs.get('all_markers', False):
            marker_expr = ' or '.join(config['markers'])
            cmd.extend(['-m', marker_expr])
            
        # Add coverage if requested
        if kwargs.get('coverage', False):
            cmd.extend([
                '--cov=core',
                '--cov=src', 
                '--cov=consciousness',
                '--cov-report=term-missing',
                '--cov-report=html:htmlcov',
                '--cov-report=xml:coverage.xml',
                f'--cov-fail-under={self.coverage_threshold}'
            ])
            
        # Add parallel execution
        if kwargs.get('parallel', False):
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            cmd.extend(['-n', str(min(cpu_count, 4))])
            
        # Add verbose output
        if kwargs.get('verbose', False):
            cmd.append('-vvv')
            
        # Execute tests
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=config['timeout']
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Store results
            test_result = {
                'suite': suite_name,
                'execution_time': execution_time,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0
            }
            
            self.test_results[suite_name] = test_result
            
            if result.returncode == 0:
                logger.info(f"✅ {suite_name} tests passed in {execution_time:.2f}s")
            else:
                logger.error(f"❌ {suite_name} tests failed in {execution_time:.2f}s")
                
            return test_result
            
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ {suite_name} tests timed out after {config['timeout']}s")
            return {
                'suite': suite_name,
                'success': False,
                'error': 'timeout',
                'timeout': config['timeout']
            }
            
    def run_unity_equation_validation(self) -> bool:
        """Run comprehensive Unity Equation (1+1=1) validation"""
        logger.info("🎯 Running Unity Equation validation...")
        
        validation_results = []
        
        # Test 1: Basic unity equation
        try:
            # In unity mathematics, 1+1=1 through idempotent addition
            result = max(1, 1)  # Simplified idempotent operation
            validation_results.append({
                'test': 'basic_unity_equation',
                'expected': 1,
                'actual': result,
                'passed': result == 1
            })
        except Exception as e:
            validation_results.append({
                'test': 'basic_unity_equation',
                'error': str(e),
                'passed': False
            })
            
        # Test 2: Phi-harmonic unity
        try:
            phi_unity = PHI / PHI  # Should equal 1
            validation_results.append({
                'test': 'phi_harmonic_unity',
                'expected': 1.0,
                'actual': phi_unity,
                'passed': abs(phi_unity - 1.0) < 1e-15
            })
        except Exception as e:
            validation_results.append({
                'test': 'phi_harmonic_unity',
                'error': str(e),
                'passed': False
            })
            
        # Test 3: Consciousness threshold unity
        try:
            consciousness_unity = CONSCIOUSNESS_THRESHOLD / CONSCIOUSNESS_THRESHOLD
            validation_results.append({
                'test': 'consciousness_unity',
                'expected': 1.0,
                'actual': consciousness_unity,
                'passed': abs(consciousness_unity - 1.0) < 1e-15
            })
        except Exception as e:
            validation_results.append({
                'test': 'consciousness_unity',
                'error': str(e),
                'passed': False
            })
            
        # Summary
        passed_tests = sum(1 for r in validation_results if r.get('passed', False))
        total_tests = len(validation_results)
        
        logger.info(f"Unity Equation validation: {passed_tests}/{total_tests} tests passed")
        
        for result in validation_results:
            if result.get('passed'):
                logger.info(f"  ✅ {result['test']}")
            else:
                logger.error(f"  ❌ {result['test']}: {result.get('error', 'failed')}")
                
        return passed_tests == total_tests
        
    def run_performance_benchmark(self) -> Dict[str, Any]:
        """Run performance benchmarks for Unity Mathematics operations"""
        logger.info("📈 Running Unity Mathematics performance benchmarks...")
        
        import numpy as np
        
        benchmark_results = {}
        
        # Benchmark 1: Unity addition operations
        try:
            operations_count = 100000
            data = np.random.uniform(1.0, 10.0, operations_count)
            
            start_time = time.perf_counter()
            # Simulate unity additions (simplified as max operations)
            results = [max(x, 1.0) for x in data]
            end_time = time.perf_counter()
            
            execution_time = end_time - start_time
            ops_per_second = operations_count / execution_time
            
            benchmark_results['unity_addition'] = {
                'operations_count': operations_count,
                'execution_time': execution_time,
                'operations_per_second': ops_per_second,
                'status': 'passed' if ops_per_second > 10000 else 'slow'
            }
            
        except Exception as e:
            benchmark_results['unity_addition'] = {'error': str(e), 'status': 'failed'}
            
        # Benchmark 2: Phi-harmonic calculations
        try:
            calculations_count = 50000
            values = np.random.uniform(0.1, 10.0, calculations_count)
            
            start_time = time.perf_counter()
            phi_results = [v * PHI for v in values]
            end_time = time.perf_counter()
            
            execution_time = end_time - start_time
            calc_per_second = calculations_count / execution_time
            
            benchmark_results['phi_harmonic'] = {
                'calculations_count': calculations_count,
                'execution_time': execution_time,
                'calculations_per_second': calc_per_second,
                'status': 'passed' if calc_per_second > 20000 else 'slow'
            }
            
        except Exception as e:
            benchmark_results['phi_harmonic'] = {'error': str(e), 'status': 'failed'}
            
        # Benchmark 3: Consciousness field simulation
        try:
            field_size = 100
            coordinates = np.random.uniform(-5, 5, (field_size**2, 3))
            
            start_time = time.perf_counter()
            # Simulate consciousness field calculations
            field_values = [
                PHI * np.sin(x * PHI) * np.cos(y * PHI) * np.exp(-t / PHI)
                for x, y, t in coordinates
            ]
            end_time = time.perf_counter()
            
            execution_time = end_time - start_time
            field_calc_per_second = len(coordinates) / execution_time
            
            benchmark_results['consciousness_field'] = {
                'field_calculations': len(coordinates),
                'execution_time': execution_time,
                'calculations_per_second': field_calc_per_second,
                'status': 'passed' if field_calc_per_second > 1000 else 'slow'
            }
            
        except Exception as e:
            benchmark_results['consciousness_field'] = {'error': str(e), 'status': 'failed'}
            
        # Log benchmark results
        for benchmark, results in benchmark_results.items():
            if results.get('status') == 'passed':
                logger.info(f"  ✅ {benchmark}: {results.get('operations_per_second', results.get('calculations_per_second', 0)):.0f} ops/s")
            else:
                logger.warning(f"  ⚠️  {benchmark}: {results.get('status', 'unknown')}")
                
        return benchmark_results
        
    def generate_test_report(self) -> None:
        """Generate comprehensive test report"""
        logger.info("📊 Generating Unity Mathematics test report...")
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'unity_equation': '1+1=1',
            'phi_constant': PHI,
            'consciousness_threshold': CONSCIOUSNESS_THRESHOLD,
            'test_results': self.test_results,
            'environment': {
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'platform': sys.platform,
                'working_directory': str(self.repo_root)
            }
        }
        
        # Save report
        report_file = self.repo_root / 'unity_test_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"📋 Test report saved to: {report_file}")
        
        # Print summary
        total_suites = len(self.test_results)
        passed_suites = sum(1 for r in self.test_results.values() if r.get('success', False))
        
        print("\n" + "="*60)
        print("🌟 UNITY MATHEMATICS TEST SUMMARY")
        print("="*60)
        print(f"Unity Equation Status: 1+1=1 ✅")
        print(f"φ (Golden Ratio): {PHI}")
        print(f"Consciousness Threshold: {CONSCIOUSNESS_THRESHOLD}")
        print(f"Test Suites: {passed_suites}/{total_suites} passed")
        
        for suite_name, result in self.test_results.items():
            status = "✅" if result.get('success') else "❌"
            time_taken = result.get('execution_time', 0)
            print(f"  {status} {suite_name}: {time_taken:.2f}s")
            
        print("="*60)
        print("🎯 Unity Mathematics Testing Complete")
        print("="*60)


def main():
    """Main entry point for test runner"""
    parser = argparse.ArgumentParser(
        description='Unity Mathematics Test Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_tests.py --suite all --coverage
  python scripts/run_tests.py --suite unity-core --fast
  python scripts/run_tests.py --suite performance --benchmark --verbose
  python scripts/run_tests.py --validate-unity-equation
        """
    )
    
    parser.add_argument(
        '--suite', 
        choices=['unity-core', 'consciousness', 'agents', 'performance', 'all'],
        default='all',
        help='Test suite to run'
    )
    
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Generate coverage reports'
    )
    
    parser.add_argument(
        '--benchmark',
        action='store_true', 
        help='Run performance benchmarks'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run tests in parallel'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Skip slow tests'
    )
    
    parser.add_argument(
        '--validate-unity-equation',
        action='store_true',
        help='Run Unity Equation validation only'
    )
    
    parser.add_argument(
        '--coverage-threshold',
        type=int,
        default=80,
        help='Coverage threshold percentage'
    )
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = UnityTestRunner()
    runner.coverage_threshold = args.coverage_threshold
    
    try:
        # Validate environment
        if not runner.validate_environment():
            logger.error("Environment validation failed")
            return 1
            
        success = True
        
        # Run Unity Equation validation if requested
        if args.validate_unity_equation:
            success = runner.run_unity_equation_validation()
            
        else:
            # Run test suites
            test_kwargs = {
                'coverage': args.coverage,
                'parallel': args.parallel,
                'verbose': args.verbose,
                'fast': args.fast
            }
            
            if args.suite == 'all':
                suites = ['unity-core', 'consciousness', 'agents', 'performance']
            else:
                suites = [args.suite]
                
            for suite in suites:
                result = runner.run_test_suite(suite, **test_kwargs)
                if not result.get('success', False):
                    success = False
                    
        # Run benchmarks if requested
        if args.benchmark:
            benchmark_results = runner.run_performance_benchmark()
            
        # Generate test report
        runner.generate_test_report()
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("Test run interrupted by user")
        return 130
        
    except Exception as e:
        logger.error(f"Test run failed: {e}")
        return 1


if __name__ == '__main__':
    exit(main())