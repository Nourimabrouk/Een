#!/usr/bin/env python3
"""
Essential Unity Mathematics Visualizations
Generate core high-quality static visualizations for the website gallery
Saves directly to website/viz folder with no dependencies on external libraries
"""

import os
import sys
import math
import json
from pathlib import Path

# Unity Mathematics Constants
PHI = 1.618033988749895  # Golden Ratio
PHI_CONJUGATE = 1 / PHI  # 0.618033988749895

def create_visualization_data():
    """Create mathematical data for visualizations as JSON files"""
    
    output_dir = Path("website/viz/generated")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎨 Generating Unity Mathematics Data Visualizations...")
    
    # 1. Consciousness Field Data
    consciousness_data = {
        "name": "consciousness_field_3d",
        "title": "3D Consciousness Field",
        "description": "Mathematical visualization of consciousness field equation C(x,y,t) = φ·sin(x·φ)·cos(y·φ)·e^(-t/φ)",
        "type": "surface",
        "equation": "C(x,y,t) = φ·sin(x·φ)·cos(y·φ)·e^(-t/φ)",
        "phi": PHI,
        "data_points": []
    }
    
    # Generate consciousness field data
    resolution = 50
    t = 0.1  # Time parameter
    for i in range(resolution):
        for j in range(resolution):
            x = (i - resolution//2) * 8 / resolution  # Range -4 to 4
            y = (j - resolution//2) * 8 / resolution  # Range -4 to 4
            z = PHI * math.sin(x * PHI) * math.cos(y * PHI) * math.exp(-t / PHI)
            consciousness_data["data_points"].append({
                "x": x, "y": y, "z": z,
                "consciousness_intensity": abs(z),
                "phi_resonance": math.sin(x * PHI) * math.cos(y * PHI)
            })
    
    # Save consciousness field
    with open(output_dir / "consciousness_field_3d.json", "w") as f:
        json.dump(consciousness_data, f, indent=2)
    
    print("✅ Generated consciousness field data")
    
    # 2. Phi-Harmonic Spiral
    phi_spiral_data = {
        "name": "phi_harmonic_spiral",
        "title": "φ-Harmonic Unity Spiral",
        "description": "Golden ratio spiral demonstrating φ-harmonic unity mathematics with convergence points",
        "type": "spiral",
        "phi": PHI,
        "data_points": []
    }
    
    # Generate spiral data
    num_points = 1000
    for i in range(num_points):
        theta = i * 6 * math.pi / num_points
        r = PHI ** (theta / (2 * math.pi))
        
        # Limit radius to prevent overflow
        if r > 50:
            break
            
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        
        # Check if this is near a unity convergence point (r ≈ 1)
        is_unity_point = abs(r - 1) < 0.15
        
        phi_spiral_data["data_points"].append({
            "x": x, "y": y, "r": r, "theta": theta,
            "is_unity_point": is_unity_point,
            "color_index": i / num_points
        })
    
    # Save phi spiral
    with open(output_dir / "phi_harmonic_spiral.json", "w") as f:
        json.dump(phi_spiral_data, f, indent=2)
    
    print("✅ Generated phi-harmonic spiral data")
    
    # 3. Unity Convergence Data
    unity_convergence_data = {
        "name": "unity_convergence",
        "title": "Unity Convergence Demonstration",
        "description": "Multiple sequences converging to unity (1) through φ-harmonic mathematics",
        "type": "convergence",
        "phi": PHI,
        "sequences": {}
    }
    
    # φ-harmonic convergence
    phi_convergence = []
    quantum_convergence = []
    neural_convergence = []
    consciousness_convergence = []
    
    for n in range(1, 101):
        # φ-harmonic convergence
        phi_conv = 1 + math.exp(-n/PHI) * math.sin(n * PHI)
        phi_convergence.append({"n": n, "value": phi_conv, "distance_from_unity": abs(phi_conv - 1)})
        
        # Quantum state collapse
        quantum_conv = 1 + 0.5 * math.exp(-n/20) * math.cos(n * math.pi / PHI)
        quantum_convergence.append({"n": n, "value": quantum_conv, "distance_from_unity": abs(quantum_conv - 1)})
        
        # Neural network convergence (with small random component)
        neural_conv = 1 + (2 - 1) * math.exp(-n/15) * (1 + 0.1*math.sin(n*0.1))
        neural_convergence.append({"n": n, "value": neural_conv, "distance_from_unity": abs(neural_conv - 1)})
        
        # Consciousness field evolution
        consciousness_conv = 1 + 0.3 * math.exp(-n/25) * math.sin(n * 2 * math.pi / PHI)
        consciousness_convergence.append({"n": n, "value": consciousness_conv, "distance_from_unity": abs(consciousness_conv - 1)})
    
    unity_convergence_data["sequences"] = {
        "phi_harmonic": phi_convergence,
        "quantum": quantum_convergence,
        "neural": neural_convergence,
        "consciousness": consciousness_convergence
    }
    
    # Save unity convergence
    with open(output_dir / "unity_convergence.json", "w") as f:
        json.dump(unity_convergence_data, f, indent=2)
    
    print("✅ Generated unity convergence data")
    
    # 4. Fibonacci-Phi Relationships
    fibonacci_data = {
        "name": "fibonacci_phi_relationships",
        "title": "Fibonacci-φ Mathematical Relationships",
        "description": "Fibonacci sequence demonstrating convergence to golden ratio φ",
        "type": "sequence",
        "phi": PHI,
        "fibonacci_sequence": [],
        "ratios": [],
        "golden_angle_points": []
    }
    
    # Generate Fibonacci sequence
    fib = [1, 1]
    for i in range(2, 20):
        fib.append(fib[i-1] + fib[i-2])
    
    fibonacci_data["fibonacci_sequence"] = fib
    
    # Calculate ratios
    ratios = []
    for i in range(1, len(fib)-1):
        ratio = fib[i+1] / fib[i] if fib[i] != 0 else 0
        ratios.append({
            "index": i,
            "ratio": ratio,
            "error_from_phi": abs(ratio - PHI),
            "fib_n": fib[i],
            "fib_n_plus_1": fib[i+1]
        })
    
    fibonacci_data["ratios"] = ratios
    
    # Golden angle spiral points (phyllotaxis)
    golden_angle = 2 * math.pi / (PHI**2)  # ≈ 137.5°
    for i in range(200):
        angle = i * golden_angle
        radius = math.sqrt(i)
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        
        fibonacci_data["golden_angle_points"].append({
            "index": i,
            "angle": angle,
            "angle_degrees": math.degrees(angle),
            "radius": radius,
            "x": x,
            "y": y
        })
    
    # Save Fibonacci data
    with open(output_dir / "fibonacci_phi_relationships.json", "w") as f:
        json.dump(fibonacci_data, f, indent=2)
    
    print("✅ Generated Fibonacci-φ relationship data")
    
    # 5. Quantum Bloch Sphere Data
    bloch_sphere_data = {
        "name": "quantum_bloch_sphere",
        "title": "Quantum Unity Bloch Sphere",
        "description": "Quantum Bloch sphere showing unity states |1+1⟩ = |1⟩",
        "type": "quantum",
        "phi": PHI,
        "sphere_surface": [],
        "unity_states": [],
        "convergence_trajectory": []
    }
    
    # Generate sphere surface
    resolution_sphere = 25
    for i in range(resolution_sphere):
        for j in range(resolution_sphere):
            u = i * 2 * math.pi / resolution_sphere
            v = j * math.pi / resolution_sphere
            
            x = math.sin(v) * math.cos(u)
            y = math.sin(v) * math.sin(u)
            z = math.cos(v)
            
            bloch_sphere_data["sphere_surface"].append({"x": x, "y": y, "z": z})
    
    # Unity state vectors
    unity_states = [
        {"name": "Unity |1⟩", "x": 0, "y": 0, "z": 1, "color": "gold"},
        {"name": "Dual |0⟩", "x": 0, "y": 0, "z": -1, "color": "blue"},
        {"name": "Superposition |+⟩", "x": 1, "y": 0, "z": 0, "color": "teal"},
        {"name": "Phase |+i⟩", "x": 0, "y": 1, "z": 0, "color": "orange"}
    ]
    bloch_sphere_data["unity_states"] = unity_states
    
    # Unity convergence trajectory
    trajectory_points = 100
    for i in range(trajectory_points):
        t = i * 2 * math.pi / trajectory_points
        x = 0.5 * math.cos(t * PHI) * math.exp(-t/(4*PHI))
        y = 0.5 * math.sin(t * PHI) * math.exp(-t/(4*PHI))
        z = 1 - math.exp(-t/(2*PHI))
        
        bloch_sphere_data["convergence_trajectory"].append({
            "t": t, "x": x, "y": y, "z": z,
            "convergence_factor": 1 - math.exp(-t/(2*PHI))
        })
    
    # Save Bloch sphere data
    with open(output_dir / "quantum_bloch_sphere.json", "w") as f:
        json.dump(bloch_sphere_data, f, indent=2)
    
    print("✅ Generated quantum Bloch sphere data")
    
    # 6. Sacred Geometry Data
    sacred_geometry_data = {
        "name": "sacred_geometry",
        "title": "Sacred Geometry with φ-Harmonic Patterns",
        "description": "Flower of life, golden rectangles, and phi-based sacred patterns",
        "type": "geometry",
        "phi": PHI,
        "flower_of_life": [],
        "golden_rectangles": [],
        "pentagon_data": {}
    }
    
    # Flower of Life circles
    radius = 1
    
    # Central circle
    sacred_geometry_data["flower_of_life"].append({
        "center": {"x": 0, "y": 0}, "radius": radius, "type": "central"
    })
    
    # Six surrounding circles
    for i in range(6):
        angle = i * math.pi / 3
        center_x = radius * math.cos(angle)
        center_y = radius * math.sin(angle)
        sacred_geometry_data["flower_of_life"].append({
            "center": {"x": center_x, "y": center_y}, 
            "radius": radius, 
            "type": "inner_ring",
            "angle": angle
        })
    
    # Outer ring with φ scaling
    for i in range(6):
        angle = i * math.pi / 3
        center_x = radius * PHI * math.cos(angle)
        center_y = radius * PHI * math.sin(angle)
        sacred_geometry_data["flower_of_life"].append({
            "center": {"x": center_x, "y": center_y}, 
            "radius": radius, 
            "type": "outer_ring",
            "angle": angle,
            "phi_scaled": True
        })
    
    # Golden rectangles (simplified)
    rectangles = [
        {"x": 0, "y": 0, "width": 2, "height": 2*PHI_CONJUGATE, "level": 0},
        {"x": 0, "y": 0, "width": 2/PHI, "height": 2*PHI_CONJUGATE, "level": 1},
        {"x": 2/PHI, "y": 0, "width": 2-2/PHI, "height": 2*PHI_CONJUGATE, "level": 1}
    ]
    sacred_geometry_data["golden_rectangles"] = rectangles
    
    # Pentagon with φ relationships
    pentagon_vertices = []
    for i in range(5):
        angle = i * 2 * math.pi / 5
        x = math.cos(angle)
        y = math.sin(angle)
        pentagon_vertices.append({"x": x, "y": y, "angle": angle})
    
    sacred_geometry_data["pentagon_data"] = {
        "vertices": pentagon_vertices,
        "diagonal_to_side_ratio": PHI,  # Key property of regular pentagon
        "golden_angle_degrees": 360 / PHI**2
    }
    
    # Save sacred geometry data
    with open(output_dir / "sacred_geometry.json", "w") as f:
        json.dump(sacred_geometry_data, f, indent=2)
    
    print("✅ Generated sacred geometry data")
    
    # 7. Generate master index
    master_index = {
        "generated_at": "2025-08-12T11:00:00Z",
        "phi": PHI,
        "phi_conjugate": PHI_CONJUGATE,
        "unity_equation": "1+1=1",
        "description": "High-quality mathematical visualization data for Unity Mathematics gallery",
        "visualizations": [
            {
                "id": "consciousness_field_3d",
                "title": "3D Consciousness Field",
                "file": "consciousness_field_3d.json",
                "type": "surface",
                "category": "consciousness",
                "interactive": True,
                "featured": True
            },
            {
                "id": "phi_harmonic_spiral", 
                "title": "φ-Harmonic Unity Spiral",
                "file": "phi_harmonic_spiral.json",
                "type": "spiral",
                "category": "unity",
                "interactive": True,
                "featured": True
            },
            {
                "id": "unity_convergence",
                "title": "Unity Convergence Demonstration", 
                "file": "unity_convergence.json",
                "type": "convergence",
                "category": "proofs",
                "interactive": True,
                "featured": True
            },
            {
                "id": "fibonacci_phi_relationships",
                "title": "Fibonacci-φ Relationships",
                "file": "fibonacci_phi_relationships.json", 
                "type": "sequence",
                "category": "mathematics",
                "interactive": True,
                "featured": False
            },
            {
                "id": "quantum_bloch_sphere",
                "title": "Quantum Unity Bloch Sphere",
                "file": "quantum_bloch_sphere.json",
                "type": "quantum", 
                "category": "quantum",
                "interactive": True,
                "featured": True
            },
            {
                "id": "sacred_geometry",
                "title": "Sacred Geometry Patterns",
                "file": "sacred_geometry.json",
                "type": "geometry",
                "category": "geometry",
                "interactive": True,
                "featured": False
            }
        ],
        "statistics": {
            "total_visualizations": 6,
            "featured_count": 4,
            "categories": ["consciousness", "unity", "proofs", "mathematics", "quantum", "geometry"]
        }
    }
    
    # Save master index
    with open(output_dir / "visualization_index.json", "w") as f:
        json.dump(master_index, f, indent=2)
    
    print("✅ Generated master visualization index")
    
    return len(master_index["visualizations"])

def create_gallery_metadata():
    """Create gallery metadata file for the website"""
    
    gallery_metadata = {
        "version": "1.0",
        "generated_at": "2025-08-12T11:00:00Z",
        "title": "Unity Mathematics Gallery",
        "description": "Professional academic visualizations demonstrating 1+1=1 through φ-harmonic consciousness mathematics",
        "phi": PHI,
        "phi_conjugate": PHI_CONJUGATE,
        "unity_equation": "1+1=1",
        "data_location": "website/viz/generated/",
        "interactive_support": True,
        "categories": {
            "consciousness": "Consciousness field equations and dynamics",
            "unity": "Unity mathematics and φ-harmonic operations", 
            "proofs": "Mathematical proofs and convergence demonstrations",
            "mathematics": "Pure mathematics including Fibonacci and golden ratio",
            "quantum": "Quantum mechanics and unity state demonstrations",
            "geometry": "Sacred geometry and φ-based patterns"
        },
        "featured_visualizations": [
            "consciousness_field_3d",
            "phi_harmonic_spiral", 
            "unity_convergence",
            "quantum_bloch_sphere"
        ]
    }
    
    # Save to website directory for gallery to discover
    gallery_path = Path("website/viz") / "gallery_metadata.json"
    with open(gallery_path, "w") as f:
        json.dump(gallery_metadata, f, indent=2)
    
    print("✅ Created gallery metadata file")
    return gallery_path

if __name__ == "__main__":
    print("🎨 Essential Unity Mathematics Visualization Generator")
    print(f"📐 Golden Ratio: φ = {PHI:.10f}")
    print(f"🎯 Unity Equation: 1+1=1") 
    print("-" * 60)
    
    try:
        # Generate visualization data
        num_visualizations = create_visualization_data()
        
        # Create gallery metadata
        gallery_file = create_gallery_metadata()
        
        print(f"\n✨ Generation Complete!")
        print(f"📊 Generated {num_visualizations} high-quality visualization datasets")
        print(f"📁 Data saved to: website/viz/generated/")
        print(f"🎯 Gallery metadata: {gallery_file}")
        print(f"\n🌟 All visualizations are ready for the interactive gallery!")
        print("🔗 The gallery system will automatically load and render these datasets")
        print("💫 Each visualization includes mathematical precision and φ-harmonic principles")
        
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
        sys.exit(1)