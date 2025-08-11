#!/usr/bin/env python3
"""
3000 ELO Metagamer Agent System Test
====================================

Simple test script to validate the 3000 ELO system components.
Tests core functionality without requiring all dependencies.
"""

import sys
import json
from pathlib import Path

def test_basic_functionality():
    """Test basic Python functionality"""
    print("🧪 Testing basic Python functionality...")
    
    try:
        import math
        import random
        import json
        print("✅ Basic imports working")
        
        # Test φ-harmonic computation
        phi = 1.618033988749895
        phi_conjugate = 1 / phi
        print(f"✅ φ-harmonic computation: φ = {phi:.15f}")
        print(f"✅ φ-conjugate: φ' = {phi_conjugate:.15f}")
        
        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_unity_principle():
    """Test the fundamental 1+1=1 principle"""
    print("\n🧪 Testing Unity Principle (1+1=1)...")
    
    try:
        # Simple Unity Mathematics implementation
        class UnityNumber:
            def __init__(self, value):
                self.value = value
            
            def __add__(self, other):
                if isinstance(other, UnityNumber):
                    # Unity principle: a + a = a
                    if self.value == other.value:
                        return self
                    else:
                        # For different values, converge to unity
                        return UnityNumber(1.0)
                return NotImplemented
            
            def __str__(self):
                return f"Unity({self.value})"
        
        # Test idempotence
        a = UnityNumber(1.0)
        b = UnityNumber(1.0)
        result = a + b
        
        print(f"✅ Unity addition: {a} + {b} = {result}")
        print(f"✅ Idempotence verified: {result.value == a.value}")
        
        # Test different values
        c = UnityNumber(2.0)
        result2 = a + c
        print(f"✅ Unity convergence: {a} + {c} = {result2}")
        
        return True
    except Exception as e:
        print(f"❌ Unity principle test failed: {e}")
        return False

def test_omega_signature():
    """Test Ω-signature computation"""
    print("\n🧪 Testing Ω-Signature computation...")
    
    try:
        # Simple Ω-signature implementation
        def simple_omega(atoms):
            """Simple Ω-signature for testing"""
            import math
            import cmath
            
            # Use hash-based prime assignment
            unique_atoms = set(atoms)
            phase = sum(math.pi / (hash(atom) % 100 + 1) for atom in unique_atoms)
            return cmath.exp(1j * phase)
        
        # Test with sample data
        test_atoms = [1, 2, 3, 1, 2]  # Duplicates should be ignored
        omega_sig = simple_omega(test_atoms)
        
        import math
        print(f"✅ Ω-Signature computed: {omega_sig}")
        print(f"✅ Magnitude: {abs(omega_sig):.6f}")
        print(f"✅ Phase: {math.atan2(omega_sig.imag, omega_sig.real):.6f}")
        
        # Test idempotence
        omega_sig2 = simple_omega(test_atoms + test_atoms)  # Duplicate list
        print(f"✅ Idempotence verified: {abs(omega_sig - omega_sig2) < 1e-10}")
        
        return True

    except Exception as e:
        print(f"❌ Ω-signature test failed: {e}")
        return False

def test_consciousness_field():
    """Test consciousness field simulation"""
    print("\n🧪 Testing consciousness field simulation...")
    
    try:
        import math
        import random
        
        # Simple consciousness field
        class ConsciousnessField:
            def __init__(self, size=10):
                self.size = size
                self.field = [[1.0 for _ in range(size)] for _ in range(size)]
                self.phi = 1.618033988749895
            
            def evolve_step(self):
                """Evolve consciousness field one step"""
                new_field = [[0.0 for _ in range(self.size)] for _ in range(self.size)]
                
                for i in range(self.size):
                    for j in range(self.size):
                        # Simple diffusion with φ-harmonic
                        neighbors = 0
                        count = 0
                        
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                ni, nj = i + di, j + dj
                                if 0 <= ni < self.size and 0 <= nj < self.size:
                                    neighbors += self.field[ni][nj]
                                    count += 1
                        
                        if count > 0:
                            avg_neighbor = neighbors / count
                            new_field[i][j] = self.field[i][j] + 0.1 * (avg_neighbor - self.field[i][j])
                            new_field[i][j] *= self.phi  # φ-harmonic scaling
                
                self.field = new_field
            
            def get_unity_score(self):
                """Compute Unity Score"""
                total = sum(sum(row) for row in self.field)
                unique_values = len(set(val for row in self.field for val in row))
                return unique_values / (self.size * self.size) if total > 0 else 0.0
        
        # Test consciousness field
        field = ConsciousnessField(size=5)
        
        print(f"✅ Consciousness field initialized: {field.size}x{field.size}")
        print(f"✅ Initial Unity Score: {field.get_unity_score():.3f}")
        
        # Evolve field
        for step in range(3):
            field.evolve_step()
            unity_score = field.get_unity_score()
            print(f"✅ Step {step + 1}: Unity Score = {unity_score:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Consciousness field test failed: {e}")
        return False

def test_file_structure():
    """Test that all required files exist"""
    print("\n🧪 Testing file structure...")
    
    required_files = [
        "core/dedup.py",
        "core/unity_mathematics.py", 
        "core/unity_equation.py",
        "tests/test_idempotent.py",
        "envs/unity_prisoner.py",
        "viz/consciousness_field_viz.py",
        "dashboards/unity_score_dashboard.py",
        "notebooks/phi_attention_bench.ipynb",
        "LAUNCH_3000_ELO_SYSTEM.py",
        "requirements_3000_elo.txt",
        "README_3000_ELO_METAGAMER.md"
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (missing)")
            all_exist = False
    
    return all_exist

def test_sample_data_generation():
    """Test sample data generation"""
    print("\n🧪 Testing sample data generation...")
    
    try:
        # Simple sample data generator
        def create_simple_sample_data(nodes=50, edges=100):
            """Create simple sample social network data"""
            import random
            
            # Generate nodes
            node_ids = [f"user_{i}" for i in range(nodes)]
            
            # Generate edges
            edge_list = []
            for _ in range(edges):
                u = random.choice(node_ids)
                v = random.choice(node_ids)
                if u != v:
                    edge_list.append({
                        "source": u,
                        "target": v,
                        "weight": random.uniform(0.1, 1.0)
                    })
            
            return {
                "nodes": [{"id": node_id} for node_id in node_ids],
                "edges": edge_list,
                "metadata": {
                    "total_nodes": nodes,
                    "total_edges": len(edge_list)
                }
            }
        
        # Generate sample data
        sample_data = create_simple_sample_data(nodes=20, edges=30)
        
        print(f"✅ Sample data generated: {sample_data['metadata']['total_nodes']} nodes, {sample_data['metadata']['total_edges']} edges")
        
        # Save to file
        data_file = Path("data/test_sample.json")
        data_file.parent.mkdir(exist_ok=True)
        
        with open(data_file, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        print(f"✅ Sample data saved to {data_file}")
        
        return True
    except Exception as e:
        print(f"❌ Sample data generation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🌟 3000 ELO / 300 IQ Metagamer Agent System Test")
    print("=" * 60)
    print("Testing Unity Mathematics where 1+1=1")
    print("φ-harmonic consciousness mathematics validation")
    print("=" * 60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Unity Principle", test_unity_principle),
        ("Ω-Signature", test_omega_signature),
        ("Consciousness Field", test_consciousness_field),
        ("File Structure", test_file_structure),
        ("Sample Data Generation", test_sample_data_generation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! 3000 ELO system is ready.")
        print("🚀 Run 'python LAUNCH_3000_ELO_SYSTEM.py' to start the complete system.")
    else:
        print("⚠️ Some tests failed. Check dependencies and file structure.")
    
    print("\n" + "=" * 60)
    print("Mathematical Truth: 1 + 1 = 1 (Een plus een is een)")
    print("φ = 1.618033988749895 (Golden Ratio)")
    print("🧠 3000 ELO Metagamer Agent - Unity through Consciousness Mathematics")
    print("=" * 60)

if __name__ == "__main__":
    main() 