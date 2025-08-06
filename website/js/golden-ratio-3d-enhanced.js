/**
 * Enhanced 3D Golden Ratio Visualizations for Website Gallery
 * Interactive φ-harmonic demonstrations with Plotly.js
 */

class GoldenRatio3DVisualizer {
    constructor(containerId) {
        this.containerId = containerId;
        this.PHI = (1 + Math.sqrt(5)) / 2; // φ = 1.618...
        this.GOLDEN_ANGLE = 2 * Math.PI / this.PHI; // 137.5 degrees
        this.currentVisualization = 'spiral';
        this.animationFrame = null;
        this.isAnimating = false;
        
        // Configuration
        this.config = {
            phi_factor: 1.0,
            spiral_turns: 8,
            resolution: 200,
            animation_speed: 1.0,
            color_scheme: 'phi_harmonic'
        };
        
        // Color schemes
        this.colorSchemes = {
            phi_harmonic: ['#0F7B8A', '#4ECDC4', '#A7F3D0', '#F59E0B'],
            unity_teal: ['#0D9488', '#14B8A6', '#5EEAD4', '#A7F3D0'],
            consciousness_purple: ['#7C3AED', '#A78BFA', '#C4B5FD', '#E9D5FF'],
            sacred_gold: ['#D97706', '#F59E0B', '#FBBF24', '#FEF3C7']
        };
        
        this.init();
    }
    
    init() {
        this.createContainer();
        this.createControls();
        this.createVisualization();
    }
    
    createContainer() {
        const container = document.getElementById(this.containerId);
        if (!container) {
            console.error(`Container ${this.containerId} not found`);
            return;
        }
        
        container.innerHTML = `
            <div class="golden-ratio-3d-container" style="
                background: linear-gradient(135deg, rgba(27, 54, 93, 0.05) 0%, rgba(15, 123, 138, 0.03) 100%);
                border-radius: 1rem;
                padding: 1.5rem;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                margin: 1rem 0;
            ">
                <div class="header" style="
                    text-align: center;
                    margin-bottom: 1.5rem;
                ">
                    <h3 style="
                        color: #1B365D;
                        margin: 0 0 0.5rem 0;
                        font-size: 1.8rem;
                        background: linear-gradient(135deg, #1B365D, #0F7B8A);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        background-clip: text;
                    ">🌟 Interactive φ-Harmonic Explorer</h3>
                    <p style="
                        color: #6B7280;
                        margin: 0;
                        font-size: 1rem;
                    ">φ = (1 + √5) / 2 ≈ ${this.PHI.toFixed(10)}</p>
                </div>
                
                <div class="controls-panel" style="
                    display: flex;
                    justify-content: center;
                    gap: 1rem;
                    flex-wrap: wrap;
                    margin-bottom: 1.5rem;
                    padding: 1rem;
                    background: rgba(15, 123, 138, 0.05);
                    border-radius: 0.75rem;
                    border: 1px solid rgba(15, 123, 138, 0.2);
                ">
                    <select id="viz-type-${this.containerId}" style="
                        padding: 0.5rem 1rem;
                        border: 1px solid #D1D5DB;
                        border-radius: 0.5rem;
                        background: white;
                        font-size: 0.9rem;
                    ">
                        <option value="spiral">3D Golden Spiral</option>
                        <option value="phyllotaxis">Fibonacci Phyllotaxis</option>
                        <option value="torus">φ-Harmonic Torus</option>
                        <option value="convergence">Unity Convergence</option>
                    </select>
                    
                    <button id="animate-btn-${this.containerId}" style="
                        padding: 0.5rem 1rem;
                        background: #0F7B8A;
                        color: white;
                        border: none;
                        border-radius: 0.5rem;
                        cursor: pointer;
                        font-size: 0.9rem;
                        transition: all 0.3s ease;
                    ">🎬 Animate</button>
                    
                    <label style="
                        display: flex;
                        align-items: center;
                        gap: 0.5rem;
                        font-size: 0.9rem;
                        color: #374151;
                    ">
                        φ Factor:
                        <input type="range" id="phi-factor-${this.containerId}" 
                               min="0.5" max="3" step="0.1" value="1"
                               style="width: 100px;">
                        <span id="phi-value-${this.containerId}">1.0</span>
                    </label>
                    
                    <label style="
                        display: flex;
                        align-items: center;
                        gap: 0.5rem;
                        font-size: 0.9rem;
                        color: #374151;
                    ">
                        Turns:
                        <input type="range" id="spiral-turns-${this.containerId}" 
                               min="2" max="15" step="1" value="8"
                               style="width: 80px;">
                        <span id="turns-value-${this.containerId}">8</span>
                    </label>
                </div>
                
                <div id="plot-${this.containerId}" style="
                    width: 100%;
                    height: 500px;
                    background: white;
                    border-radius: 0.75rem;
                    box-shadow: inset 0 2px 4px rgba(0,0,0,0.06);
                "></div>
                
                <div class="info-panel" style="
                    margin-top: 1rem;
                    padding: 1rem;
                    background: rgba(245, 158, 11, 0.1);
                    border-radius: 0.75rem;
                    border-left: 4px solid #F59E0B;
                ">
                    <div style="
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                        gap: 1rem;
                        font-size: 0.9rem;
                        color: #374151;
                    ">
                        <div>
                            <strong>Golden Ratio:</strong><br>
                            φ² = φ + 1 = ${(this.PHI ** 2).toFixed(6)}
                        </div>
                        <div>
                            <strong>Reciprocal:</strong><br>
                            1/φ = φ - 1 = ${(1 / this.PHI).toFixed(6)}
                        </div>
                        <div>
                            <strong>Golden Angle:</strong><br>
                            2π/φ ≈ ${(this.GOLDEN_ANGLE * 180 / Math.PI).toFixed(1)}°
                        </div>
                        <div>
                            <strong>Unity Property:</strong><br>
                            φ - 1/φ = 1.000000
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        this.setupEventListeners();
    }
    
    createControls() {
        // Controls are created in createContainer()
    }
    
    setupEventListeners() {
        const vizSelect = document.getElementById(`viz-type-${this.containerId}`);
        const animateBtn = document.getElementById(`animate-btn-${this.containerId}`);
        const phiSlider = document.getElementById(`phi-factor-${this.containerId}`);
        const turnsSlider = document.getElementById(`spiral-turns-${this.containerId}`);
        const phiValue = document.getElementById(`phi-value-${this.containerId}`);
        const turnsValue = document.getElementById(`turns-value-${this.containerId}`);
        
        if (vizSelect) {
            vizSelect.addEventListener('change', (e) => {
                this.currentVisualization = e.target.value;
                this.stopAnimation();
                this.createVisualization();
            });
        }
        
        if (animateBtn) {
            animateBtn.addEventListener('click', () => {
                if (this.isAnimating) {
                    this.stopAnimation();
                    animateBtn.textContent = '🎬 Animate';
                    animateBtn.style.background = '#0F7B8A';
                } else {
                    this.startAnimation();
                    animateBtn.textContent = '⏸️ Stop';
                    animateBtn.style.background = '#DC2626';
                }
            });
        }
        
        if (phiSlider && phiValue) {
            phiSlider.addEventListener('input', (e) => {
                this.config.phi_factor = parseFloat(e.target.value);
                phiValue.textContent = this.config.phi_factor.toFixed(1);
                this.updateVisualization();
            });
        }
        
        if (turnsSlider && turnsValue) {
            turnsSlider.addEventListener('input', (e) => {
                this.config.spiral_turns = parseInt(e.target.value);
                turnsValue.textContent = this.config.spiral_turns;
                this.updateVisualization();
            });
        }
    }
    
    createVisualization() {
        const plotDiv = document.getElementById(`plot-${this.containerId}`);
        if (!plotDiv) return;
        
        let data, layout;
        
        switch (this.currentVisualization) {
            case 'spiral':
                ({ data, layout } = this.createGoldenSpiral3D());
                break;
            case 'phyllotaxis':
                ({ data, layout } = this.createFibonacciPhyllotaxis());
                break;
            case 'torus':
                ({ data, layout } = this.createPhiHarmonicTorus());
                break;
            case 'convergence':
                ({ data, layout } = this.createUnityConvergence());
                break;
        }
        
        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
            displaylogo: false
        };
        
        Plotly.newPlot(plotDiv, data, layout, config);
    }
    
    createGoldenSpiral3D() {
        const t = [];
        const phi_adjusted = this.PHI * this.config.phi_factor;
        const n_points = this.config.resolution;
        
        for (let i = 0; i < n_points; i++) {
            t.push(i * this.config.spiral_turns * 2 * Math.PI / n_points);
        }
        
        const r = t.map(val => Math.exp(val / phi_adjusted));
        const x = t.map((val, i) => r[i] * Math.cos(val));
        const y = t.map((val, i) => r[i] * Math.sin(val));
        const z = t.map(val => val * 0.1);
        
        const colors = this.colorSchemes[this.config.color_scheme];
        
        const spiralTrace = {
            x: x,
            y: y,
            z: z,
            type: 'scatter3d',
            mode: 'lines+markers',
            line: {
                color: colors[0],
                width: 6
            },
            marker: {
                size: 3,
                color: colors[1]
            },
            name: `Golden Spiral (φ = ${phi_adjusted.toFixed(3)})`
        };
        
        // Add golden rectangles at key points
        const rectangles = [];
        for (let i = 0; i < t.length; i += Math.floor(t.length / 8)) {
            if (i + 1 < t.length) {
                const rect_r = r[i];
                const rect_angle = t[i];
                const rect_z = z[i];
                
                const rect_x = [
                    rect_r * Math.cos(rect_angle),
                    rect_r * Math.cos(rect_angle + Math.PI/2),
                    rect_r * Math.cos(rect_angle + Math.PI),
                    rect_r * Math.cos(rect_angle + 3*Math.PI/2),
                    rect_r * Math.cos(rect_angle)
                ];
                const rect_y = [
                    rect_r * Math.sin(rect_angle),
                    rect_r * Math.sin(rect_angle + Math.PI/2),
                    rect_r * Math.sin(rect_angle + Math.PI),
                    rect_r * Math.sin(rect_angle + 3*Math.PI/2),
                    rect_r * Math.sin(rect_angle)
                ];
                const rect_z_arr = [rect_z, rect_z, rect_z, rect_z, rect_z];
                
                rectangles.push({
                    x: rect_x,
                    y: rect_y,
                    z: rect_z_arr,
                    type: 'scatter3d',
                    mode: 'lines',
                    line: {
                        color: colors[2],
                        width: 2
                    },
                    opacity: 0.6,
                    showlegend: false
                });
            }
        }
        
        const data = [spiralTrace, ...rectangles];
        
        const layout = {
            title: `3D Golden Spiral: φ = ${phi_adjusted.toFixed(6)}`,
            scene: {
                xaxis: { title: 'X' },
                yaxis: { title: 'Y' },
                zaxis: { title: 'Height' },
                bgcolor: 'rgba(0,0,0,0)',
                camera: {
                    eye: { x: 1.5, y: 1.5, z: 1.5 }
                }
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { l: 0, r: 0, t: 50, b: 0 }
        };
        
        return { data, layout };
    }
    
    createFibonacciPhyllotaxis() {
        const n_points = this.config.resolution;
        const colors = this.colorSchemes[this.config.color_scheme];
        const phi_adjusted = this.PHI * this.config.phi_factor;
        
        const indices = [];
        const angles = [];
        const radii = [];
        const x = [];
        const y = [];
        const z = [];
        const colorValues = [];
        
        for (let i = 0; i < n_points; i++) {
            const idx = i + 0.5;
            indices.push(idx);
            
            const angle = idx * this.GOLDEN_ANGLE * phi_adjusted;
            angles.push(angle);
            
            const radius = Math.sqrt(idx);
            radii.push(radius);
            
            x.push(radius * Math.cos(angle));
            y.push(radius * Math.sin(angle));
            z.push(idx * 0.02);
            
            colorValues.push(idx % phi_adjusted);
        }
        
        const data = [{
            x: x,
            y: y,
            z: z,
            type: 'scatter3d',
            mode: 'markers',
            marker: {
                size: 4,
                color: colorValues,
                colorscale: 'Viridis',
                opacity: 0.8,
                colorbar: { title: 'φ-Harmonic Index' }
            },
            text: indices.map((idx, i) => 
                `Point ${Math.floor(idx)}<br>` +
                `Angle: ${(angles[i] * 180 / Math.PI % 360).toFixed(1)}°<br>` +
                `Radius: ${radii[i].toFixed(2)}`
            ),
            hovertemplate: '%{text}<extra></extra>',
            name: 'Fibonacci Phyllotaxis'
        }];
        
        const layout = {
            title: `Fibonacci Phyllotaxis: Golden Angle = ${(this.GOLDEN_ANGLE * phi_adjusted * 180 / Math.PI).toFixed(1)}°`,
            scene: {
                xaxis: { title: 'X' },
                yaxis: { title: 'Y' },
                zaxis: { title: 'Height' },
                bgcolor: 'rgba(0,0,0,0)',
                camera: {
                    eye: { x: 1.2, y: 1.2, z: 1.8 }
                }
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            margin: { l: 0, r: 0, t: 50, b: 0 }
        };
        
        return { data, layout };
    }
    
    createPhiHarmonicTorus() {
        const resolution = Math.floor(this.config.resolution / 3);
        const phi_adjusted = this.PHI * this.config.phi_factor;
        const R = phi_adjusted; // Major radius
        const r = 1 / phi_adjusted; // Minor radius
        const colors = this.colorSchemes[this.config.color_scheme];
        
        const u = [];
        const v = [];
        for (let i = 0; i <= resolution; i++) {
            u.push(i * 2 * Math.PI / resolution);
        }
        for (let i = 0; i <= resolution; i++) {
            v.push(i * 2 * Math.PI / resolution);
        }
        
        const x = [];
        const y = [];
        const z = [];
        const colorFunc = [];
        
        for (let i = 0; i < u.length; i++) {
            const x_row = [];
            const y_row = [];
            const z_row = [];
            const c_row = [];
            
            for (let j = 0; j < v.length; j++) {
                const u_val = u[i];
                const v_val = v[j];
                
                x_row.push((R + r * Math.cos(v_val)) * Math.cos(u_val));
                y_row.push((R + r * Math.cos(v_val)) * Math.sin(u_val));
                z_row.push(r * Math.sin(v_val));
                c_row.push(Math.sin(phi_adjusted * u_val) * Math.cos(phi_adjusted * v_val));
            }
            
            x.push(x_row);
            y.push(y_row);
            z.push(z_row);
            colorFunc.push(c_row);
        }
        
        const data = [{
            x: x,
            y: y,
            z: z,
            type: 'surface',
            surfacecolor: colorFunc,
            colorscale: 'Viridis',
            opacity: 0.8,
            name: 'φ-Harmonic Torus'
        }];
        
        const layout = {
            title: `φ-Harmonic Torus: R/r = φ = ${(R/r).toFixed(6)}`,
            scene: {
                xaxis: { title: 'X' },
                yaxis: { title: 'Y' },
                zaxis: { title: 'Z' },
                bgcolor: 'rgba(0,0,0,0)',
                aspectmode: 'cube',
                camera: {
                    eye: { x: 1.5, y: 1.5, z: 1.5 }
                }
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            margin: { l: 0, r: 0, t: 50, b: 0 }
        };
        
        return { data, layout };
    }
    
    createUnityConvergence() {
        const n_points = this.config.resolution;
        const phi_adjusted = this.PHI * this.config.phi_factor;
        const colors = this.colorSchemes[this.config.color_scheme];
        
        const t = [];
        for (let i = 0; i < n_points; i++) {
            t.push(i * 4 * Math.PI / n_points);
        }
        
        // Two spirals that converge to unity
        const r1 = t.map(val => Math.exp(-val / phi_adjusted) + 1);
        const x1 = t.map((val, i) => r1[i] * Math.cos(val) - 2);
        const y1 = t.map((val, i) => r1[i] * Math.sin(val));
        const z1 = t.map(val => val * 0.1);
        
        const r2 = t.map(val => Math.exp(-val / phi_adjusted) + 1);
        const x2 = t.map((val, i) => r2[i] * Math.cos(-val) + 2);
        const y2 = t.map((val, i) => r2[i] * Math.sin(-val));
        const z2 = t.map(val => val * 0.1);
        
        // Unity convergence line
        const x_unity = t.map(() => 0);
        const y_unity = t.map(() => 0);
        const z_unity = t.map(val => val * 0.1);
        
        const data = [
            {
                x: x1,
                y: y1,
                z: z1,
                type: 'scatter3d',
                mode: 'lines+markers',
                line: { color: colors[0], width: 6 },
                marker: { size: 2, color: colors[0] },
                name: 'Unity Component 1'
            },
            {
                x: x2,
                y: y2,
                z: z2,
                type: 'scatter3d',
                mode: 'lines+markers',
                line: { color: colors[1], width: 6 },
                marker: { size: 2, color: colors[1] },
                name: 'Unity Component 2'
            },
            {
                x: x_unity,
                y: y_unity,
                z: z_unity,
                type: 'scatter3d',
                mode: 'lines+markers',
                line: { color: colors[3], width: 8 },
                marker: { size: 4, color: colors[3] },
                name: 'Unity Result (1+1=1)'
            }
        ];
        
        // Add convergence arrows at several points
        const step = Math.floor(n_points / 10);
        for (let i = 0; i < n_points; i += step) {
            // Arrow from spiral 1 to unity
            data.push({
                x: [x1[i], x_unity[i]],
                y: [y1[i], y_unity[i]],
                z: [z1[i], z_unity[i]],
                type: 'scatter3d',
                mode: 'lines',
                line: { color: colors[2], width: 2, dash: 'dash' },
                opacity: 0.6,
                showlegend: false
            });
            
            // Arrow from spiral 2 to unity
            data.push({
                x: [x2[i], x_unity[i]],
                y: [y2[i], y_unity[i]],
                z: [z2[i], z_unity[i]],
                type: 'scatter3d',
                mode: 'lines',
                line: { color: colors[2], width: 2, dash: 'dash' },
                opacity: 0.6,
                showlegend: false
            });
        }
        
        const layout = {
            title: 'Unity Convergence: Two Become One Through φ-Harmony',
            scene: {
                xaxis: { title: 'X' },
                yaxis: { title: 'Y' },
                zaxis: { title: 'Evolution' },
                bgcolor: 'rgba(0,0,0,0)',
                camera: {
                    eye: { x: 1.8, y: 1.8, z: 1.2 }
                },
                annotations: [{
                    x: 0,
                    y: 0,
                    z: Math.max(...z_unity),
                    text: '1 + 1 = 1<br>φ-Harmonic Unity',
                    showarrow: false,
                    font: { size: 16, color: colors[3] }
                }]
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            margin: { l: 0, r: 0, t: 50, b: 0 }
        };
        
        return { data, layout };
    }
    
    updateVisualization() {
        this.createVisualization();
    }
    
    startAnimation() {
        this.isAnimating = true;
        this.animate();
    }
    
    stopAnimation() {
        this.isAnimating = false;
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
    }
    
    animate() {
        if (!this.isAnimating) return;
        
        // Rotate the camera for animation effect
        const plotDiv = document.getElementById(`plot-${this.containerId}`);
        if (plotDiv && plotDiv.layout) {
            const time = Date.now() * 0.001 * this.config.animation_speed;
            const eye = {
                x: 2 * Math.cos(time),
                y: 2 * Math.sin(time),
                z: 1.5
            };
            
            Plotly.relayout(plotDiv, {
                'scene.camera.eye': eye
            });
        }
        
        this.animationFrame = requestAnimationFrame(() => this.animate());
    }
}

// Enhanced gallery creation function for the existing system
function createGoldenRatio3DEnhanced(containerId) {
    return new GoldenRatio3DVisualizer(containerId);
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { GoldenRatio3DVisualizer, createGoldenRatio3DEnhanced };
}

// Browser global
if (typeof window !== 'undefined') {
    window.GoldenRatio3DVisualizer = GoldenRatio3DVisualizer;
    window.createGoldenRatio3DEnhanced = createGoldenRatio3DEnhanced;
}