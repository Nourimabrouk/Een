/**
 * Enhanced Quantum Unity Visualizer for Website Gallery
 * Interactive demonstration of |1⟩ + |1⟩ = |1⟩ through Bloch spheres and wave interference
 */

class QuantumUnityVisualizer {
    constructor(containerId) {
        this.containerId = containerId;
        this.PHI = (1 + Math.sqrt(5)) / 2; // φ = 1.618...
        this.currentVisualization = 'bloch';
        this.animationFrame = null;
        this.isAnimating = false;
        this.startTime = Date.now();
        
        // Configuration
        this.config = {
            theta1: Math.PI / 4,
            phi1: 0,
            theta2: Math.PI / 4,
            phi2: 0,
            unity_coupling: 0.618,
            superposition_weight: 0.5,
            animation_speed: 1.0,
            show_vectors: true,
            show_evolution: true
        };
        
        // Color schemes
        this.colorSchemes = {
            quantum_blue: ['#3B82F6', '#93C5FD', '#DBEAFE', '#1E40AF'],
            unity_gradient: ['#3B82F6', '#10B981', '#F59E0B', '#DC2626']
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
            <div class="quantum-unity-container" style="
                background: radial-gradient(circle at center, rgba(59, 130, 246, 0.03) 0%, rgba(15, 123, 138, 0.05) 100%);
                border-radius: 1rem;
                padding: 1.5rem;
                box-shadow: 0 8px 32px rgba(59, 130, 246, 0.2);
                margin: 1rem 0;
                position: relative;
                overflow: hidden;
            ">
                <!-- Animated quantum background -->
                <div style="
                    position: absolute;
                    top: -50%;
                    left: -50%;
                    width: 200%;
                    height: 200%;
                    background: radial-gradient(circle, rgba(59, 130, 246, 0.1) 0%, transparent 70%);
                    animation: quantum-oscillation 3s ease-in-out infinite;
                    z-index: 0;
                "></div>
                
                <div class="header" style="
                    text-align: center;
                    margin-bottom: 1.5rem;
                    position: relative;
                    z-index: 1;
                ">
                    <h3 style="
                        color: #3B82F6;
                        margin: 0 0 0.5rem 0;
                        font-size: 1.8rem;
                        background: linear-gradient(135deg, #3B82F6, #1E40AF);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        background-clip: text;
                    ">⚛️ Quantum Unity Explorer</h3>
                    <p style="
                        color: #6B7280;
                        margin: 0 0 1rem 0;
                        font-size: 1rem;
                    ">Interactive demonstration of quantum unity principles</p>
                    <div style="
                        font-family: 'Times New Roman', serif;
                        font-size: 1.1rem;
                        color: #3B82F6;
                        background: rgba(59, 130, 246, 0.1);
                        padding: 0.5rem 1rem;
                        border-radius: 0.5rem;
                        display: inline-block;
                        border: 1px solid rgba(59, 130, 246, 0.3);
                    ">
                        |1⟩ + |1⟩ = |1⟩ through φ-harmonic superposition
                    </div>
                </div>
                
                <div class="controls-panel" style="
                    display: flex;
                    justify-content: center;
                    gap: 1rem;
                    flex-wrap: wrap;
                    margin-bottom: 1.5rem;
                    padding: 1rem;
                    background: rgba(59, 130, 246, 0.05);
                    border-radius: 0.75rem;
                    border: 1px solid rgba(59, 130, 246, 0.2);
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
                        <option value="bloch">Bloch Spheres</option>
                        <option value="interference">Wave Interference</option>
                        <option value="evolution">Unity Evolution</option>
                        <option value="superposition">Superposition Demo</option>
                    </select>
                    
                    <button id="animate-btn-${this.containerId}" style="
                        padding: 0.5rem 1rem;
                        background: #3B82F6;
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
                        State θ₁:
                        <input type="range" id="theta1-${this.containerId}" 
                               min="0" max="${Math.PI}" step="0.01" value="${Math.PI/4}"
                               style="width: 80px;">
                        <span id="theta1-value-${this.containerId}">${(Math.PI/4).toFixed(2)}</span>
                    </label>
                    
                    <label style="
                        display: flex;
                        align-items: center;
                        gap: 0.5rem;
                        font-size: 0.9rem;
                        color: #374151;
                    ">
                        φ-Coupling:
                        <input type="range" id="coupling-${this.containerId}" 
                               min="0" max="1" step="0.001" value="0.618"
                               style="width: 80px;">
                        <span id="coupling-value-${this.containerId}">0.618</span>
                    </label>
                    
                    <button id="vectors-btn-${this.containerId}" style="
                        padding: 0.5rem 1rem;
                        background: ${this.config.show_vectors ? '#10B981' : '#6B7280'};
                        color: white;
                        border: none;
                        border-radius: 0.5rem;
                        cursor: pointer;
                        font-size: 0.9rem;
                        transition: all 0.3s ease;
                    ">📊 Vectors</button>
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
                
                <div class="quantum-stats" style="
                    margin-top: 1rem;
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 1rem;
                    position: relative;
                    z-index: 1;
                ">
                    <div style="
                        background: rgba(59, 130, 246, 0.05);
                        padding: 0.75rem;
                        border-radius: 0.5rem;
                        text-align: center;
                        border: 1px solid rgba(59, 130, 246, 0.2);
                    ">
                        <div id="coherence-${this.containerId}" style="
                            font-size: 1.2rem;
                            font-weight: bold;
                            color: #3B82F6;
                        ">0.000</div>
                        <div style="font-size: 0.8rem; color: #6B7280;">Coherence</div>
                    </div>
                    <div style="
                        background: rgba(16, 185, 129, 0.05);
                        padding: 0.75rem;
                        border-radius: 0.5rem;
                        text-align: center;
                        border: 1px solid rgba(16, 185, 129, 0.2);
                    ">
                        <div id="fidelity-${this.containerId}" style="
                            font-size: 1.2rem;
                            font-weight: bold;
                            color: #10B981;
                        ">0.000</div>
                        <div style="font-size: 0.8rem; color: #6B7280;">Unity Fidelity</div>
                    </div>
                    <div style="
                        background: rgba(245, 158, 11, 0.05);
                        padding: 0.75rem;
                        border-radius: 0.5rem;
                        text-align: center;
                        border: 1px solid rgba(245, 158, 11, 0.2);
                    ">
                        <div id="entanglement-${this.containerId}" style="
                            font-size: 1.2rem;
                            font-weight: bold;
                            color: #F59E0B;
                        ">0.000</div>
                        <div style="font-size: 0.8rem; color: #6B7280;">Entanglement</div>
                    </div>
                    <div style="
                        background: rgba(59, 130, 246, 0.05);
                        padding: 0.75rem;
                        border-radius: 0.5rem;
                        text-align: center;
                        border: 1px solid rgba(59, 130, 246, 0.2);
                    ">
                        <div id="unity-state-${this.containerId}" style="
                            font-size: 1.2rem;
                            font-weight: bold;
                            color: #3B82F6;
                        ">Active</div>
                        <div style="font-size: 0.8rem; color: #6B7280;">Unity State</div>
                    </div>
                </div>
            </div>
            
            <style>
                @keyframes quantum-oscillation {
                    0%, 100% { transform: scale(1) rotate(0deg); opacity: 0.2; }
                    50% { transform: scale(1.05) rotate(180deg); opacity: 0.05; }
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
        const theta1Slider = document.getElementById(`theta1-${this.containerId}`);
        const couplingSlider = document.getElementById(`coupling-${this.containerId}`);
        const vectorsBtn = document.getElementById(`vectors-btn-${this.containerId}`);
        
        const theta1Value = document.getElementById(`theta1-value-${this.containerId}`);
        const couplingValue = document.getElementById(`coupling-value-${this.containerId}`);
        
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
        
        if (theta1Slider && theta1Value) {
            theta1Slider.addEventListener('input', (e) => {
                this.config.theta1 = parseFloat(e.target.value);
                theta1Value.textContent = this.config.theta1.toFixed(2);
                this.updateVisualization();
            });
        }
        
        if (couplingSlider && couplingValue) {
            couplingSlider.addEventListener('input', (e) => {
                this.config.unity_coupling = parseFloat(e.target.value);
                couplingValue.textContent = this.config.unity_coupling.toFixed(3);
                this.updateVisualization();
            });
        }
        
        if (vectorsBtn) {
            vectorsBtn.addEventListener('click', () => {
                this.config.show_vectors = !this.config.show_vectors;
                vectorsBtn.style.background = this.config.show_vectors ? '#10B981' : '#6B7280';
                this.updateVisualization();
            });
        }
    }
    
    // Quantum mechanics utility functions
    createQuantumState(theta, phi) {
        // |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
        return [
            Math.cos(theta / 2),
            { real: Math.cos(phi) * Math.sin(theta / 2), imag: Math.sin(phi) * Math.sin(theta / 2) }
        ];
    }
    
    complexMultiply(a, b) {
        if (typeof a === 'number') a = { real: a, imag: 0 };
        if (typeof b === 'number') b = { real: b, imag: 0 };
        return {
            real: a.real * b.real - a.imag * b.imag,
            imag: a.real * b.imag + a.imag * b.real
        };
    }
    
    complexConjugate(z) {
        if (typeof z === 'number') return z;
        return { real: z.real, imag: -z.imag };
    }
    
    complexMagnitude(z) {
        if (typeof z === 'number') return Math.abs(z);
        return Math.sqrt(z.real * z.real + z.imag * z.imag);
    }
    
    blochCoordinates(state) {
        // Convert quantum state to Bloch sphere coordinates
        const alpha = typeof state[0] === 'number' ? state[0] : state[0];
        const beta = state[1];
        
        const alphaConj = typeof alpha === 'number' ? alpha : this.complexConjugate(alpha);
        const betaConj = this.complexConjugate(beta);
        
        const alphaBetaConj = this.complexMultiply(alpha, betaConj);
        
        const x = 2 * alphaBetaConj.real;
        const y = 2 * alphaBetaConj.imag;
        const z = Math.pow(this.complexMagnitude(alpha), 2) - Math.pow(this.complexMagnitude(beta), 2);
        
        return { x, y, z };
    }
    
    createVisualization() {
        const plotDiv = document.getElementById(`plot-${this.containerId}`);
        if (!plotDiv) return;
        
        let data, layout;
        
        switch (this.currentVisualization) {
            case 'bloch':
                ({ data, layout } = this.createBlochSpheres());
                break;
            case 'interference':
                ({ data, layout } = this.createWaveInterference());
                break;
            case 'evolution':
                ({ data, layout } = this.createUnityEvolution());
                break;
            case 'superposition':
                ({ data, layout } = this.createSuperpositionDemo());
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
    
    createBlochSpheres() {
        const currentTime = this.isAnimating ? (Date.now() - this.startTime) * 0.001 * this.config.animation_speed : 0;
        
        // Create quantum states with time evolution
        const theta1 = this.config.theta1 + (this.config.show_evolution ? 0.1 * Math.sin(currentTime) : 0);
        const theta2 = this.config.theta2 + (this.config.show_evolution ? 0.1 * Math.cos(currentTime) : 0);
        const phi1 = this.config.phi1 + (this.config.show_evolution ? 0.2 * currentTime : 0);
        const phi2 = this.config.phi2 + (this.config.show_evolution ? 0.15 * currentTime : 0);
        
        const psi1 = this.createQuantumState(theta1, phi1);
        const psi2 = this.createQuantumState(theta2, phi2);
        
        // Unity superposition with φ-harmonic coupling
        const coupling = this.config.unity_coupling;
        const weight = this.config.superposition_weight;
        
        // Simplified unity superposition
        const psiUnity = [
            coupling * psi1[0] + (1 - coupling) * psi2[0],
            this.complexMultiply(coupling, psi1[1])
        ];
        
        // Get Bloch coordinates
        const coords1 = this.blochCoordinates(psi1);
        const coords2 = this.blochCoordinates(psi2);
        const coordsUnity = this.blochCoordinates(psiUnity);
        
        // Create Bloch spheres
        const data = [];
        
        // Sphere wireframes
        const sphereData = this.createSphere();
        for (let i = 0; i < 3; i++) {
            data.push({
                x: sphereData.x,
                y: sphereData.y,
                z: sphereData.z,
                type: 'surface',
                opacity: 0.1,
                colorscale: [[0, '#E5E7EB'], [1, '#93C5FD']],
                showscale: false,
                hoverinfo: 'skip'
            });
        }
        
        // State vectors and points
        const states = [
            { coords: coords1, name: 'State |ψ₁⟩', color: '#3B82F6' },
            { coords: coords2, name: 'State |ψ₂⟩', color: '#10B981' },
            { coords: coordsUnity, name: 'Unity |ψᵤ⟩', color: '#F59E0B' }
        ];
        
        states.forEach((state, index) => {
            if (this.config.show_vectors) {
                // Vector line
                data.push({
                    x: [0, state.coords.x],
                    y: [0, state.coords.y],
                    z: [0, state.coords.z],
                    type: 'scatter3d',
                    mode: 'lines',
                    line: { color: state.color, width: 6 },
                    name: `${state.name} Vector`,
                    showlegend: false
                });
            }
            
            // State point
            data.push({
                x: [state.coords.x],
                y: [state.coords.y],
                z: [state.coords.z],
                type: 'scatter3d',
                mode: 'markers',
                marker: { size: 12, color: state.color },
                name: state.name,
                text: `${state.name}<br>(${state.coords.x.toFixed(3)}, ${state.coords.y.toFixed(3)}, ${state.coords.z.toFixed(3)})`,
                hovertemplate: '%{text}<extra></extra>'
            });
        });
        
        const layout = {
            title: 'Quantum Unity: Bloch Sphere Representation',
            scene: {
                xaxis: { range: [-1.2, 1.2], title: 'X' },
                yaxis: { range: [-1.2, 1.2], title: 'Y' },
                zaxis: { range: [-1.2, 1.2], title: 'Z' },
                aspectmode: 'cube',
                bgcolor: 'rgba(0,0,0,0)',
                camera: { eye: { x: 1.5, y: 1.5, z: 1.5 } }
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            margin: { l: 0, r: 0, t: 50, b: 0 }
        };
        
        return { data, layout };
    }
    
    createWaveInterference() {
        const currentTime = this.isAnimating ? (Date.now() - this.startTime) * 0.001 * this.config.animation_speed : 0;
        const x = this.linspace(-4*Math.PI, 4*Math.PI, 500);
        
        // Two quantum waves
        const wave1 = x.map(xi => Math.cos(xi - currentTime) * Math.exp(-Math.pow((xi - Math.PI)/2, 2)));
        const wave2 = x.map(xi => Math.cos(xi - currentTime) * Math.exp(-Math.pow((xi + Math.PI)/2, 2)));
        
        // Unity interference with φ-harmonic coupling
        const coupling = this.config.unity_coupling;
        const interference = x.map((xi, i) => coupling * wave1[i] + (1 - coupling) * wave2[i]);
        
        // Normalize to create unity wave
        const maxInterference = Math.max(...interference.map(Math.abs));
        const unityWave = interference.map(val => maxInterference > 0 ? val / maxInterference : val);
        
        const data = [
            {
                x: x,
                y: wave1,
                type: 'scatter',
                mode: 'lines',
                name: 'Wave |ψ₁⟩',
                line: { color: '#3B82F6', width: 2 },
                opacity: 0.7
            },
            {
                x: x,
                y: wave2,
                type: 'scatter',
                mode: 'lines',
                name: 'Wave |ψ₂⟩',
                line: { color: '#10B981', width: 2 },
                opacity: 0.7
            },
            {
                x: x,
                y: unityWave,
                type: 'scatter',
                mode: 'lines',
                name: 'Unity Wave |ψᵤ⟩',
                line: { color: '#F59E0B', width: 4 },
                fill: 'tozeroy',
                fillcolor: 'rgba(245, 158, 11, 0.2)'
            }
        ];
        
        const layout = {
            title: 'Quantum Wave Interference: |ψ₁⟩ + |ψ₂⟩ = |ψᵤ⟩',
            xaxis: { title: 'Position' },
            yaxis: { title: 'Wave Amplitude' },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { l: 0, r: 0, t: 50, b: 0 }
        };
        
        return { data, layout };
    }
    
    createUnityEvolution() {
        const timePoints = this.linspace(0, 10, 100);
        const coherences = [];
        const fidelities = [];
        
        for (let t of timePoints) {
            // Simplified evolution calculation
            const decay = Math.exp(-t * 0.1);
            const oscillation = Math.cos(t * this.config.unity_coupling * this.PHI);
            
            const coherence = decay * Math.abs(oscillation);
            const fidelity = decay * (0.5 + 0.5 * Math.cos(t * 0.2));
            
            coherences.push(coherence);
            fidelities.push(fidelity);
        }
        
        const data = [
            {
                x: timePoints,
                y: coherences,
                type: 'scatter',
                mode: 'lines',
                name: 'Quantum Coherence',
                line: { color: '#3B82F6', width: 3 },
                fill: 'tozeroy',
                fillcolor: 'rgba(59, 130, 246, 0.2)'
            },
            {
                x: timePoints,
                y: fidelities,
                type: 'scatter',
                mode: 'lines',
                name: 'Unity Fidelity',
                line: { color: '#F59E0B', width: 3 },
                fill: 'tozeroy',
                fillcolor: 'rgba(245, 158, 11, 0.2)'
            }
        ];
        
        // Add φ-harmonic reference line
        data.push({
            x: timePoints,
            y: timePoints.map(() => this.PHI / 2.618),
            type: 'scatter',
            mode: 'lines',
            name: 'φ-Harmonic Threshold',
            line: { color: '#DC2626', dash: 'dash', width: 2 },
            showlegend: false
        });
        
        const layout = {
            title: 'Quantum Unity Evolution Dynamics',
            xaxis: { title: 'Time' },
            yaxis: { title: 'Probability' },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { l: 0, r: 0, t: 50, b: 0 }
        };
        
        return { data, layout };
    }
    
    createSuperpositionDemo() {
        const angles = this.linspace(0, 2*Math.PI, 50);
        const probabilities1 = [];
        const probabilities2 = [];
        const unityProbabilities = [];
        
        for (let angle of angles) {
            const psi1 = this.createQuantumState(this.config.theta1, angle);
            const psi2 = this.createQuantumState(this.config.theta2, angle);
            
            // Measurement probabilities
            const prob1 = Math.pow(this.complexMagnitude(psi1[1]), 2);
            const prob2 = Math.pow(this.complexMagnitude(psi2[1]), 2);
            
            // Unity probability with φ-harmonic coupling
            const coupling = this.config.unity_coupling;
            const unityProb = coupling * prob1 + (1 - coupling) * prob2;
            
            probabilities1.push(prob1);
            probabilities2.push(prob2);
            unityProbabilities.push(unityProb);
        }
        
        const data = [
            {
                x: angles.map(a => a * 180 / Math.PI),
                y: probabilities1,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'P(|1⟩) for |ψ₁⟩',
                line: { color: '#3B82F6' },
                marker: { size: 6 }
            },
            {
                x: angles.map(a => a * 180 / Math.PI),
                y: probabilities2,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'P(|1⟩) for |ψ₂⟩',
                line: { color: '#10B981' },
                marker: { size: 6 }
            },
            {
                x: angles.map(a => a * 180 / Math.PI),
                y: unityProbabilities,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'P(|1⟩) for Unity State',
                line: { color: '#F59E0B', width: 3 },
                marker: { size: 8 }
            }
        ];
        
        const layout = {
            title: 'Quantum Measurement Probabilities vs Phase',
            xaxis: { title: 'Phase Angle (degrees)' },
            yaxis: { title: 'Probability P(|1⟩)' },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { l: 0, r: 0, t: 50, b: 0 }
        };
        
        return { data, layout };
    }
    
    updateStats() {
        const psi1 = this.createQuantumState(this.config.theta1, this.config.phi1);
        const psi2 = this.createQuantumState(this.config.theta2, this.config.phi2);
        
        // Calculate quantum metrics (simplified)
        const coherence = Math.abs(Math.cos(this.config.theta1 - this.config.theta2)) * this.config.unity_coupling;
        const fidelity = 0.5 + 0.5 * coherence;
        const entanglement = this.config.unity_coupling * Math.sin(this.config.theta1) * Math.sin(this.config.theta2);
        
        // Update displays
        const coherenceEl = document.getElementById(`coherence-${this.containerId}`);
        const fidelityEl = document.getElementById(`fidelity-${this.containerId}`);
        const entanglementEl = document.getElementById(`entanglement-${this.containerId}`);
        const unityStateEl = document.getElementById(`unity-state-${this.containerId}`);
        
        if (coherenceEl) coherenceEl.textContent = coherence.toFixed(3);
        if (fidelityEl) fidelityEl.textContent = fidelity.toFixed(3);
        if (entanglementEl) entanglementEl.textContent = entanglement.toFixed(3);
        
        if (unityStateEl) {
            if (coherence > 0.618) {
                unityStateEl.textContent = 'φ-Unity';
                unityStateEl.style.color = '#10B981';
            } else if (coherence > 0.382) {
                unityStateEl.textContent = 'Coherent';
                unityStateEl.style.color = '#F59E0B';
            } else {
                unityStateEl.textContent = 'Decoherent';
                unityStateEl.style.color = '#DC2626';
            }
        }
    }
    
    // Utility functions
    createSphere() {
        const u = this.linspace(0, 2*Math.PI, 20);
        const v = this.linspace(0, Math.PI, 20);
        const x = [];
        const y = [];
        const z = [];
        
        for (let i = 0; i < v.length; i++) {
            const xRow = [];
            const yRow = [];
            const zRow = [];
            
            for (let j = 0; j < u.length; j++) {
                xRow.push(Math.cos(u[j]) * Math.sin(v[i]));
                yRow.push(Math.sin(u[j]) * Math.sin(v[i]));
                zRow.push(Math.cos(v[i]));
            }
            
            x.push(xRow);
            y.push(yRow);
            z.push(zRow);
        }
        
        return { x, y, z };
    }
    
    linspace(start, stop, num) {
        const result = [];
        const step = (stop - start) / (num - 1);
        for (let i = 0; i < num; i++) {
            result.push(start + i * step);
        }
        return result;
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
}

// Gallery creation function
function createQuantumUnityEnhanced(containerId) {
    return new QuantumUnityVisualizer(containerId);
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { QuantumUnityVisualizer, createQuantumUnityEnhanced };
}

// Browser global
if (typeof window !== 'undefined') {
    window.QuantumUnityVisualizer = QuantumUnityVisualizer;
    window.createQuantumUnityEnhanced = createQuantumUnityEnhanced;
}