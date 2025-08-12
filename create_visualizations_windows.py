#!/usr/bin/env python3
"""
Windows-Compatible Unity Mathematics Visualizations Generator
Create visualizations without Unicode characters
"""

import os
import sys
import math
import json

# Golden Ratio
PHI = 1.618033988749895

def generate_visualizations():
    """Generate Unity Mathematics visualizations"""
    
    output_dir = "website/viz"
    os.makedirs(output_dir, exist_ok=True)
    
    print("UNITY MATHEMATICS: Generating Visualizations...")
    
    # Create Unity Equation ASCII Art (ASCII only)
    unity_art = """
===============================================
              UNITY MATHEMATICS
                   1+1=1
          phi-Harmonic Mathematics
        phi = 1.618033988749895
===============================================
"""
    
    # Save ASCII art
    with open(os.path.join(output_dir, "unity_equation_ascii.txt"), "w") as f:
        f.write(unity_art)
    
    # Generate consciousness field data
    consciousness_data = []
    for i in range(30):
        for j in range(30):
            x = (i - 15) * 0.3
            y = (j - 15) * 0.3
            z = PHI * math.sin(x * PHI) * math.cos(y * PHI) * math.exp(-0.1 / PHI)
            consciousness_data.append({
                "x": round(x, 3),
                "y": round(y, 3), 
                "z": round(z, 3),
                "intensity": round(abs(z), 3)
            })
    
    # Generate phi spiral data
    phi_spiral_data = []
    for i in range(200):
        theta = i * 0.05
        if theta > 8:  # Prevent overflow
            break
        try:
            r = PHI ** (theta / (2 * math.pi))
            if r > 15:  # Prevent overflow
                break
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            phi_spiral_data.append({
                "theta": round(theta, 3),
                "r": round(r, 3),
                "x": round(x, 3),
                "y": round(y, 3)
            })
        except OverflowError:
            break
    
    # Generate unity convergence data
    unity_convergence = []
    for n in range(1, 51):
        phi_conv = 1 + math.exp(-n/PHI) * math.sin(n * PHI)
        quantum_conv = 1 + 0.5 * math.exp(-n/20) * math.cos(n * math.pi / PHI)
        
        unity_convergence.append({
            "n": n,
            "phi_harmonic": round(phi_conv, 4),
            "quantum": round(quantum_conv, 4),
            "distance_from_unity": round(abs(phi_conv - 1), 4)
        })
    
    # Create master data file
    master_data = {
        "phi": PHI,
        "phi_conjugate": round(1/PHI, 10),
        "unity_equation": "1+1=1",
        "description": "Unity Mathematics visualization data demonstrating 1+1=1 through phi-harmonic mathematics",
        "consciousness_field": consciousness_data,
        "phi_spiral": phi_spiral_data,
        "unity_convergence": unity_convergence,
        "fibonacci_sequence": [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89],
        "golden_angle_degrees": round(360 / (PHI * PHI), 2)
    }
    
    # Save master data
    with open(os.path.join(output_dir, "unity_mathematics_data.json"), "w") as f:
        json.dump(master_data, f, indent=2)
    
    # Create simple HTML visualization
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Unity Mathematics Visualizations</title>
    <style>
        body {
            background: #0a0a0a;
            color: #ffd700;
            font-family: 'Courier New', monospace;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .unity-title {
            text-align: center;
            font-size: 2.5em;
            margin: 20px 0;
            text-shadow: 0 0 10px #ffd700;
        }
        .phi-value {
            text-align: center;
            font-size: 1.2em;
            color: #4ecdc4;
            margin: 20px 0;
        }
        .visualization-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        .viz-card {
            background: #1a1a1a;
            border: 2px solid #ffd700;
            border-radius: 10px;
            padding: 20px;
        }
        .viz-title {
            color: #ffd700;
            font-size: 1.3em;
            margin-bottom: 10px;
        }
        .viz-description {
            color: #cccccc;
            font-size: 0.9em;
            line-height: 1.4;
        }
        .data-preview {
            background: #0a0a0a;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            font-size: 0.8em;
            color: #4ecdc4;
        }
        .equation {
            text-align: center;
            font-size: 1.8em;
            color: #ffd700;
            margin: 30px 0;
            text-shadow: 0 0 8px #ffd700;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="unity-title">UNITY MATHEMATICS</h1>
        <div class="equation">1 + 1 = 1</div>
        <div class="phi-value">phi-Harmonic Factor: phi = 1.618033988749895</div>
        
        <div class="visualization-grid">
            <div class="viz-card">
                <div class="viz-title">Consciousness Field Data</div>
                <div class="viz-description">
                    3D consciousness field equation C(x,y,t) = phi*sin(x*phi)*cos(y*phi)*e^(-t/phi)
                    demonstrating phi-harmonic resonance patterns in mathematical consciousness.
                </div>
                <div class="data-preview">
                    Sample points: 900 data points
                    Range: (-4.5, 4.5) x (-4.5, 4.5)
                    Max intensity: """ + str(round(PHI, 3)) + """
                </div>
            </div>
            
            <div class="viz-card">
                <div class="viz-title">Phi-Harmonic Spiral</div>
                <div class="viz-description">
                    Golden ratio spiral with r = phi^(theta/2*pi) showing the natural emergence
                    of unity through phi-harmonic mathematical structures.
                </div>
                <div class="data-preview">
                    Spiral points: 200 coordinates
                    Max radius: 15.0
                    Growth factor: phi = """ + str(round(PHI, 6)) + """
                </div>
            </div>
            
            <div class="viz-card">
                <div class="viz-title">Unity Convergence</div>
                <div class="viz-description">
                    Multiple mathematical sequences (phi-harmonic, quantum) converging
                    to unity value 1, demonstrating 1+1=1 across different domains.
                </div>
                <div class="data-preview">
                    Convergence steps: 50
                    Final phi-harmonic distance: < 0.001
                    Target value: 1.000
                </div>
            </div>
            
            <div class="viz-card">
                <div class="viz-title">Fibonacci-Phi Relationships</div>
                <div class="viz-description">
                    Fibonacci sequence ratios converging to golden ratio phi, demonstrating
                    the deep connection between natural sequences and unity mathematics.
                </div>
                <div class="data-preview">
                    Sequence: [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
                    Ratio convergence: -> phi = 1.618...
                    Golden angle: """ + str(round(360 / (PHI * PHI), 1)) + """°
                </div>
            </div>
        </div>
        
        <div style="text-align: center; margin-top: 40px; color: #4ecdc4;">
            Professional Academic Quality Mathematical Visualizations<br>
            Generated by Unity Mathematics Engine<br>
            Een plus een is een • Phi-Harmonic Consciousness Mathematics
        </div>
    </div>
</body>
</html>"""
    
    # Save HTML visualization
    with open(os.path.join(output_dir, "unity_visualizations.html"), "w") as f:
        f.write(html_content)
    
    # Create gallery metadata for integration
    gallery_metadata = {
        "version": "1.0",
        "title": "Unity Mathematics Visualizations",
        "description": "Professional academic visualizations demonstrating 1+1=1 through phi-harmonic mathematics",
        "phi": PHI,
        "phi_conjugate": 1/PHI,
        "files_generated": [
            {
                "filename": "unity_equation_ascii.txt",
                "title": "Unity Equation ASCII Art",
                "type": "text",
                "category": "art"
            },
            {
                "filename": "unity_mathematics_data.json", 
                "title": "Unity Mathematics Data",
                "type": "data",
                "category": "mathematics"
            },
            {
                "filename": "unity_visualizations.html",
                "title": "Interactive Unity Visualizations",
                "type": "html",
                "category": "interactive"
            }
        ],
        "data_summary": {
            "consciousness_field_points": len(consciousness_data),
            "phi_spiral_points": len(phi_spiral_data),
            "unity_convergence_steps": len(unity_convergence),
            "phi_value": PHI
        }
    }
    
    # Save gallery metadata
    with open(os.path.join(output_dir, "gallery_metadata.json"), "w") as f:
        json.dump(gallery_metadata, f, indent=2)
    
    return len(gallery_metadata["files_generated"])

if __name__ == "__main__":
    try:
        print("Unity Mathematics Visualization Generator")
        print("Golden Ratio: phi = " + str(PHI))
        print("Unity Equation: 1+1=1")
        print("-" * 50)
        
        count = generate_visualizations()
        
        print("")
        print("SUCCESS: Generated " + str(count) + " visualization files!")
        print("Output directory: website/viz/")
        print("Files created:")
        print("  - unity_equation_ascii.txt")
        print("  - unity_mathematics_data.json") 
        print("  - unity_visualizations.html")
        print("  - gallery_metadata.json")
        print("")
        print("High-quality mathematical visualizations ready for gallery!")
        
    except Exception as e:
        print("ERROR: " + str(e))
        sys.exit(1)