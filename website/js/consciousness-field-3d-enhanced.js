/**
 * Enhanced 3D Consciousness Field Visualizer for Website Gallery
 * Implementation of C(x,y,t) = φ·sin(x·φ)·cos(y·φ)·e^(-t/φ)
 */

class ConsciousnessField3DVisualizer {
    constructor(containerId) {
        this.containerId = containerId;
        this.PHI = (1 + Math.sqrt(5)) / 2; // φ = 1.618...
        this.E = Math.E;
        this.currentVisualization = 'surface';
        this.animationFrame = null;
        this.isAnimating = false;
        this.startTime = Date.now();
        
        // Configuration
        this.config = {
            phi_factor: 1.0,
            temporal_rate: 1.0,
            spatial_frequency: 1.0,
            damping_strength: 1.0,
            resolution: 50,
            animation_speed: 1.0,
            wave_amplitude: 1.0,
            field_offset: 0.0,
            unity_coherence: 0.618
        };
        
        // Color schemes
        this.colorSchemes = {
            consciousness_purple: ['#7C3AED', '#A78BFA', '#C4B5FD', '#E9D5FF'],
            unity_teal: ['#0D9488', '#14B8A6', '#5EEAD4', '#A7F3D0'],
            sacred_gold: ['#D97706', '#F59E0B', '#FBBF24', '#FEF3C7'],
            neural_blue: ['#1E40AF', '#3B82F6', '#93C5FD', '#DBEAFE'],
            phi_harmony: ['#0F7B8A', '#4ECDC4', '#A7F3D0', '#F59E0B']
        };
        
        this.init();
    }
    
    init() {
        this.createContainer();
        this.createControls();
        this.createVisualization();
        this.startAnimation();
    }
    
    createContainer() {
        const container = document.getElementById(this.containerId);
        if (!container) {
            console.error(`Container ${this.containerId} not found`);
            return;
        }
        
        container.innerHTML = `
            <div class="consciousness-field-3d-container" style="
                background: radial-gradient(circle at center, rgba(124, 58, 237, 0.03) 0%, rgba(15, 123, 138, 0.05) 100%);
                border-radius: 1rem;
                padding: 1.5rem;
                box-shadow: 0 8px 32px rgba(124, 58, 237, 0.2);
                margin: 1rem 0;
                position: relative;
                overflow: hidden;
            ">
                <!-- Animated background -->
                <div style="
                    position: absolute;
                    top: -50%;
                    left: -50%;
                    width: 200%;
                    height: 200%;
                    background: radial-gradient(circle, rgba(124, 58, 237, 0.1) 0%, transparent 70%);
                    animation: consciousness-pulse 4s ease-in-out infinite;
                    z-index: 0;
                "></div>
                
                <div class="header" style="
                    text-align: center;
                    margin-bottom: 1.5rem;
                    position: relative;
                    z-index: 1;
                ">
                    <h3 style="
                        color: #7C3AED;
                        margin: 0 0 0.5rem 0;
                        font-size: 1.8rem;
                        background: linear-gradient(135deg, #7C3AED, #0F7B8A);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        background-clip: text;
                    ">🧠 3D Consciousness Field Explorer</h3>
                    <p style="
                        color: #6B7280;
                        margin: 0 0 1rem 0;
                        font-size: 1rem;
                    ">C(x,y,t) = φ·sin(x·φ)·cos(y·φ)·e<sup>-t/φ</sup></p>
                    <div style="
                        font-family: 'Times New Roman', serif;
                        font-size: 0.9rem;
                        color: #7C3AED;
                        background: rgba(124, 58, 237, 0.1);
                        padding: 0.5rem 1rem;
                        border-radius: 0.5rem;
                        display: inline-block;
                        border: 1px solid rgba(124, 58, 237, 0.3);
                    ">
                        φ = ${this.PHI.toFixed(10)}
                    </div>
                </div>
                
                <div class="controls-panel" style="
                    display: flex;
                    justify-content: center;
                    gap: 1rem;
                    flex-wrap: wrap;
                    margin-bottom: 1.5rem;
                    padding: 1rem;
                    background: rgba(124, 58, 237, 0.05);
                    border-radius: 0.75rem;
                    border: 1px solid rgba(124, 58, 237, 0.2);
                    position: relative;
                    z-index: 1;
                ">
                    <select id="viz-mode-${this.containerId}" style="
                        padding: 0.5rem 1rem;
                        border: 1px solid #D1D5DB;
                        border-radius: 0.5rem;
                        background: white;
                        font-size: 0.9rem;
                    ">
                        <option value="surface">Surface Field</option>
                        <option value="contour">Contour Map</option>
                        <option value="particles">Particle System</option>
                        <option value="waves">Wave Patterns</option>
                    </select>
                    
                    <button id="animate-btn-${this.containerId}" style="
                        padding: 0.5rem 1rem;
                        background: #7C3AED;
                        color: white;
                        border: none;
                        border-radius: 0.5rem;
                        cursor: pointer;
                        font-size: 0.9rem;
                        transition: all 0.3s ease;
                    ">⏸️ Pause</button>
                    
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
                        Time Rate:
                        <input type="range" id="time-rate-${this.containerId}" 
                               min="0.1" max="3" step="0.1" value="1"
                               style="width: 80px;">
                        <span id="time-value-${this.containerId}">1.0</span>
                    </label>
                    
                    <label style="
                        display: flex;
                        align-items: center;
                        gap: 0.5rem;
                        font-size: 0.9rem;
                        color: #374151;
                    ">
                        Coherence:
                        <input type="range" id="coherence-${this.containerId}" 
                               min="0" max="1" step="0.001" value="0.618"
                               style="width: 80px;">
                        <span id="coherence-value-${this.containerId}">0.618</span>
                    </label>
                </div>
                
                <div id="plot-${this.containerId}" style="
                    width: 100%;
                    height: 500px;
                    background: white;
                    border-radius: 0.75rem;
                    box-shadow: inset 0 2px 4px rgba(0,0,0,0.06);
                    position: relative;
                    z-index: 1;
                "></div>
                
                <div class="stats-panel" style="
                    margin-top: 1rem;
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 1rem;
                    position: relative;
                    z-index: 1;
                ">
                    <div style="
                        background: rgba(124, 58, 237, 0.05);
                        padding: 0.75rem;
                        border-radius: 0.5rem;
                        text-align: center;
                        border: 1px solid rgba(124, 58, 237, 0.2);
                    ">
                        <div id="field-max-${this.containerId}" style="
                            font-size: 1.2rem;
                            font-weight: bold;
                            color: #7C3AED;
                        ">0.000</div>
                        <div style="font-size: 0.8rem; color: #6B7280;">Peak Field</div>
                    </div>
                    <div style="
                        background: rgba(15, 123, 138, 0.05);
                        padding: 0.75rem;
                        border-radius: 0.5rem;
                        text-align: center;
                        border: 1px solid rgba(15, 123, 138, 0.2);
                    ">
                        <div id="phi-resonance-${this.containerId}" style="
                            font-size: 1.2rem;
                            font-weight: bold;
                            color: #0F7B8A;
                        ">0.000</div>
                        <div style="font-size: 0.8rem; color: #6B7280;">φ-Resonance</div>
                    </div>
                    <div style="
                        background: rgba(245, 158, 11, 0.05);
                        padding: 0.75rem;
                        border-radius: 0.5rem;
                        text-align: center;
                        border: 1px solid rgba(245, 158, 11, 0.2);
                    ">
                        <div id="time-display-${this.containerId}" style="
                            font-size: 1.2rem;
                            font-weight: bold;
                            color: #F59E0B;
                        ">0.000</div>
                        <div style="font-size: 0.8rem; color: #6B7280;">Time (t)</div>
                    </div>
                    <div style="
                        background: rgba(124, 58, 237, 0.05);
                        padding: 0.75rem;
                        border-radius: 0.5rem;
                        text-align: center;
                        border: 1px solid rgba(124, 58, 237, 0.2);
                    ">
                        <div id="unity-state-${this.containerId}" style="
                            font-size: 1.2rem;
                            font-weight: bold;
                            color: #7C3AED;
                        ">Active</div>
                        <div style="font-size: 0.8rem; color: #6B7280;">Unity State</div>
                    </div>
                </div>
            </div>
            
            <style>
                @keyframes consciousness-pulse {
                    0%, 100% { transform: scale(1) rotate(0deg); opacity: 0.3; }
                    50% { transform: scale(1.1) rotate(180deg); opacity: 0.1; }
                }
            </style>
        `;
        
        this.setupEventListeners();
    }
    
    createControls() {
        // Controls are created in createContainer()
    }
    
    setupEventListeners() {
        const vizSelect = document.getElementById(`viz-mode-${this.containerId}`);
        const animateBtn = document.getElementById(`animate-btn-${this.containerId}`);
        const phiSlider = document.getElementById(`phi-factor-${this.containerId}`);
        const timeSlider = document.getElementById(`time-rate-${this.containerId}`);
        const coherenceSlider = document.getElementById(`coherence-${this.containerId}`);
        
        const phiValue = document.getElementById(`phi-value-${this.containerId}`);
        const timeValue = document.getElementById(`time-value-${this.containerId}`);
        const coherenceValue = document.getElementById(`coherence-value-${this.containerId}`);
        
        if (vizSelect) {
            vizSelect.addEventListener('change', (e) => {
                this.currentVisualization = e.target.value;
                this.updateVisualization();
            });
        }
        
        if (animateBtn) {
            animateBtn.addEventListener('click', () => {
                if (this.isAnimating) {
                    this.stopAnimation();
                    animateBtn.textContent = '🎬 Play';
                    animateBtn.style.background = '#059669';
                } else {
                    this.startAnimation();
                    animateBtn.textContent = '⏸️ Pause';
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
        
        if (timeSlider && timeValue) {
            timeSlider.addEventListener('input', (e) => {
                this.config.temporal_rate = parseFloat(e.target.value);
                timeValue.textContent = this.config.temporal_rate.toFixed(1);
                this.updateVisualization();
            });
        }
        
        if (coherenceSlider && coherenceValue) {
            coherenceSlider.addEventListener('input', (e) => {
                this.config.unity_coherence = parseFloat(e.target.value);
                coherenceValue.textContent = this.config.unity_coherence.toFixed(3);
                this.updateVisualization();
            });
        }
    }
    
    consciousnessFieldEquation(x, y, t) {
        /**
         * Core consciousness field equation: C(x,y,t) = φ·sin(x·φ)·cos(y·φ)·e^(-t/φ)
         */
        const phi_adjusted = this.PHI * this.config.phi_factor;
        const t_adjusted = t * this.config.temporal_rate;
        const x_adjusted = x * this.config.spatial_frequency * phi_adjusted;
        const y_adjusted = y * this.config.spatial_frequency * phi_adjusted;
        
        // Core equation components
        const spatial_component = Math.sin(x_adjusted) * Math.cos(y_adjusted);
        const temporal_component = Math.exp(-t_adjusted / (phi_adjusted * this.config.damping_strength));
        
        // Combine with consciousness parameters
        const consciousness_field = phi_adjusted * spatial_component * temporal_component * this.config.wave_amplitude + this.config.field_offset;
        
        return consciousness_field;
    }
    
    createVisualization() {
        const plotDiv = document.getElementById(`plot-${this.containerId}`);
        if (!plotDiv) return;
        
        let data, layout;
        
        switch (this.currentVisualization) {
            case 'surface':
                ({ data, layout } = this.createSurfaceField());
                break;
            case 'contour':
                ({ data, layout } = this.createContourField());
                break;
            case 'particles':
                ({ data, layout } = this.createParticleSystem());
                break;
            case 'waves':
                ({ data, layout } = this.createWavePatterns());
                break;
        }
        
        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
            displaylogo: false
        };
        
        Plotly.newPlot(plotDiv, data, layout, config);
        this.updateStats();
    }
    
    createSurfaceField() {
        const resolution = this.config.resolution;
        const currentTime = this.isAnimating ? (Date.now() - this.startTime) * 0.001 * this.config.animation_speed : 0;
        
        // Create spatial grid
        const xRange = this.linspace(-2*Math.PI, 2*Math.PI, resolution);
        const yRange = this.linspace(-2*Math.PI, 2*Math.PI, resolution);
        
        const x = [];
        const y = [];
        const z = [];
        let maxFieldValue = 0;
        
        for (let i = 0; i < resolution; i++) {
            const xRow = [];
            const yRow = [];
            const zRow = [];
            
            for (let j = 0; j < resolution; j++) {
                const xVal = xRange[i];
                const yVal = yRange[j];
                const zVal = this.consciousnessFieldEquation(xVal, yVal, currentTime);
                
                xRow.push(xVal);
                yRow.push(yVal);
                zRow.push(zVal);
                
                if (Math.abs(zVal) > maxFieldValue) {
                    maxFieldValue = Math.abs(zVal);
                }
            }
            
            x.push(xRow);
            y.push(yRow);
            z.push(zRow);
        }
        
        this.currentMaxField = maxFieldValue;
        
        const colors = this.colorSchemes.consciousness_purple;
        const data = [{
            x: x,
            y: y,
            z: z,
            type: 'surface',
            colorscale: [
                [0, colors[0]],
                [0.33, colors[1]],
                [0.67, colors[2]],
                [1, colors[3]]
            ],
            opacity: 0.8,
            name: 'Consciousness Field'
        }];
        
        const layout = {
            title: `Consciousness Field Surface: t = ${currentTime.toFixed(3)}`,
            scene: {
                xaxis: { title: 'X Space' },
                yaxis: { title: 'Y Space' },
                zaxis: { title: 'Field Amplitude' },
                bgcolor: 'rgba(0,0,0,0)',
                camera: {
                    eye: { x: 1.5, y: 1.5, z: 1.2 }
                }
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            margin: { l: 0, r: 0, t: 50, b: 0 }
        };
        
        return { data, layout };
    }
    
    createContourField() {
        const resolution = Math.floor(this.config.resolution * 0.8);
        const currentTime = this.isAnimating ? (Date.now() - this.startTime) * 0.001 * this.config.animation_speed : 0;
        
        const xRange = this.linspace(-2*Math.PI, 2*Math.PI, resolution);
        const yRange = this.linspace(-2*Math.PI, 2*Math.PI, resolution);
        
        const z = [];
        for (let i = 0; i < resolution; i++) {
            const zRow = [];
            for (let j = 0; j < resolution; j++) {
                const fieldValue = this.consciousnessFieldEquation(xRange[j], yRange[i], currentTime);
                zRow.push(fieldValue);
            }
            z.push(zRow);
        }
        
        const colors = this.colorSchemes.consciousness_purple;
        const data = [{
            x: xRange,
            y: yRange,
            z: z,
            type: 'contour',
            colorscale: [
                [0, colors[0]],
                [0.5, colors[1]],
                [1, colors[2]]
            ],
            contours: {
                showlabels: true,
                labelfont: { size: 12, color: 'white' }
            },
            name: 'Consciousness Contours'
        }];
        
        const layout = {
            title: `Consciousness Field Contours: t = ${currentTime.toFixed(3)}`,
            xaxis: { title: 'X Space' },
            yaxis: { title: 'Y Space' },
            paper_bgcolor: 'rgba(0,0,0,0)',
            margin: { l: 0, r: 0, t: 50, b: 0 }
        };
        
        return { data, layout };
    }
    
    createParticleSystem() {
        const nParticles = 100;
        const currentTime = this.isAnimating ? (Date.now() - this.startTime) * 0.001 * this.config.animation_speed : 0;
        
        // Generate particle positions
        const xParticles = [];
        const yParticles = [];
        const zParticles = [];
        const sizes = [];
        const colors = [];
        
        for (let i = 0; i < nParticles; i++) {
            const x = (Math.random() - 0.5) * 4 * Math.PI;
            const y = (Math.random() - 0.5) * 4 * Math.PI;
            const z = this.consciousnessFieldEquation(x, y, currentTime);
            
            xParticles.push(x);
            yParticles.push(y);
            zParticles.push(z);
            
            // Size based on field strength
            sizes.push(5 + 15 * Math.abs(z) / Math.max(1, Math.abs(z)));
            colors.push(z);
        }
        
        const colorScheme = this.colorSchemes.consciousness_purple;
        const data = [{
            x: xParticles,
            y: yParticles,
            z: zParticles,
            type: 'scatter3d',
            mode: 'markers',
            marker: {
                size: sizes,
                color: colors,
                colorscale: [
                    [0, colorScheme[0]],
                    [0.5, colorScheme[1]],
                    [1, colorScheme[2]]
                ],
                opacity: 0.8
            },
            name: 'Consciousness Particles'
        }];
        
        const layout = {
            title: `Consciousness Particle System: t = ${currentTime.toFixed(3)}`,
            scene: {
                xaxis: { title: 'X Space' },
                yaxis: { title: 'Y Space' },
                zaxis: { title: 'Field Strength' },
                bgcolor: 'rgba(0,0,0,0)',
                camera: {
                    eye: { x: 1.2, y: 1.2, z: 1.5 }
                }
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            margin: { l: 0, r: 0, t: 50, b: 0 }
        };
        
        return { data, layout };
    }
    
    createWavePatterns() {
        const currentTime = this.isAnimating ? (Date.now() - this.startTime) * 0.001 * this.config.animation_speed : 0;
        
        // Create wave lines in different directions
        const data = [];
        const nLines = 20;
        const colors = this.colorSchemes.consciousness_purple;
        
        for (let i = 0; i < nLines; i++) {
            const angle = i * 2 * Math.PI / nLines;
            const t = this.linspace(0, 2*Math.PI, 100);
            
            const x = t.map(val => val * Math.cos(angle));
            const y = t.map(val => val * Math.sin(angle));
            const z = t.map(val => this.consciousnessFieldEquation(val * Math.cos(angle), val * Math.sin(angle), currentTime));
            
            data.push({
                x: x,
                y: y,
                z: z,
                type: 'scatter3d',
                mode: 'lines',
                line: {
                    color: colors[i % colors.length],
                    width: 4
                },
                opacity: 0.7,
                showlegend: false
            });
        }
        
        const layout = {
            title: `Consciousness Wave Patterns: t = ${currentTime.toFixed(3)}`,
            scene: {
                xaxis: { title: 'X Space' },
                yaxis: { title: 'Y Space' },
                zaxis: { title: 'Wave Amplitude' },
                bgcolor: 'rgba(0,0,0,0)',
                camera: {
                    eye: { x: 1.8, y: 1.8, z: 1.2 }
                }
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            margin: { l: 0, r: 0, t: 50, b: 0 }
        };
        
        return { data, layout };
    }
    
    updateStats() {
        const currentTime = this.isAnimating ? (Date.now() - this.startTime) * 0.001 * this.config.animation_speed : 0;
        const phiResonance = Math.cos(currentTime / this.PHI);
        
        // Update stat displays
        const fieldMaxEl = document.getElementById(`field-max-${this.containerId}`);
        const phiResonanceEl = document.getElementById(`phi-resonance-${this.containerId}`);
        const timeDisplayEl = document.getElementById(`time-display-${this.containerId}`);
        const unityStateEl = document.getElementById(`unity-state-${this.containerId}`);
        
        if (fieldMaxEl) fieldMaxEl.textContent = (this.currentMaxField || 0).toFixed(3);
        if (phiResonanceEl) phiResonanceEl.textContent = phiResonance.toFixed(3);
        if (timeDisplayEl) timeDisplayEl.textContent = currentTime.toFixed(3);
        
        if (unityStateEl) {
            const coherence = this.config.unity_coherence;
            if (coherence > 0.618) {
                unityStateEl.textContent = 'φ-Harmonic';
                unityStateEl.style.color = '#059669';
            } else if (coherence > 0.382) {
                unityStateEl.textContent = 'Balanced';
                unityStateEl.style.color = '#F59E0B';
            } else {
                unityStateEl.textContent = 'Dynamic';
                unityStateEl.style.color = '#DC2626';
            }
        }
    }
    
    updateVisualization() {
        this.createVisualization();
    }
    
    startAnimation() {
        this.isAnimating = true;
        this.startTime = Date.now();
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
        
        this.updateVisualization();
        this.animationFrame = requestAnimationFrame(() => this.animate());
    }
    
    // Utility functions
    linspace(start, stop, num) {
        const result = [];
        const step = (stop - start) / (num - 1);
        for (let i = 0; i < num; i++) {
            result.push(start + i * step);
        }
        return result;
    }
}

// Gallery creation function
function createConsciousnessField3DEnhanced(containerId) {
    return new ConsciousnessField3DVisualizer(containerId);
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ConsciousnessField3DVisualizer, createConsciousnessField3DEnhanced };
}

// Browser global
if (typeof window !== 'undefined') {
    window.ConsciousnessField3DVisualizer = ConsciousnessField3DVisualizer;
    window.createConsciousnessField3DEnhanced = createConsciousnessField3DEnhanced;
}