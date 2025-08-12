/**
 * Comprehensive Visualization Engine for Unity Mathematics
 * Provides interactive visualizations with static backup generation
 * Supports sliders, real-time parameters, and professional academic output
 */

class UnityVisualizationEngine {
    constructor() {
        this.PHI = 1.618033988749895;
        this.visualizations = new Map();
        this.animationStates = new Map();
        this.parameterStates = new Map();
        
        // Professional color schemes for academic presentations
        this.colorSchemes = {
            academic: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            consciousness: ['#0F7B8A', '#4ECDC4', '#A7F3D0', '#FFD700'],
            quantum: ['#6366f1', '#8b5cf6', '#a855f7', '#c084fc'],
            unity: ['#FFD700', '#F59E0B', '#FBBF24', '#FEF3C7'],
            professional: ['#374151', '#6B7280', '#9CA3AF', '#D1D5DB']
        };

        this.init();
    }

    init() {
        this.setupGlobalStyles();
        this.registerVisualizationTypes();
        this.setupEventListeners();
    }

    setupGlobalStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .viz-container {
                position: relative;
                width: 100%;
                height: 100%;
                background: #0a0b0f;
                border-radius: 8px;
                overflow: hidden;
            }
            
            .viz-controls-overlay {
                position: absolute;
                bottom: 10px;
                left: 10px;
                right: 10px;
                background: rgba(0, 0, 0, 0.8);
                border-radius: 8px;
                padding: 1rem;
                display: none;
            }
            
            .viz-controls-overlay.active {
                display: block;
            }
            
            .parameter-row {
                display: flex;
                align-items: center;
                gap: 1rem;
                margin-bottom: 0.75rem;
            }
            
            .parameter-label {
                color: #FFD700;
                font-weight: 500;
                min-width: 120px;
                font-size: 0.875rem;
            }
            
            .parameter-slider {
                flex: 1;
                height: 4px;
                border-radius: 2px;
                background: rgba(255, 215, 0, 0.2);
                outline: none;
            }
            
            .parameter-value {
                color: #4ECDC4;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.875rem;
                font-weight: 600;
                min-width: 60px;
                text-align: right;
            }
            
            .loading-indicator {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                color: #FFD700;
                font-size: 1.5rem;
                z-index: 1000;
            }
        `;
        document.head.appendChild(style);
    }

    registerVisualizationTypes() {
        // Register all available visualization types
        this.vizTypes = {
            'consciousness-field-3d': this.createConsciousnessField3D.bind(this),
            'phi-spiral-3d': this.createPhiSpiral3D.bind(this),
            'quantum-bloch-sphere': this.createQuantumBlochSphere.bind(this),
            'unity-manifold': this.createUnityManifold.bind(this),
            'fractal-unity': this.createFractalUnity.bind(this),
            'neural-unity': this.createNeuralUnity.bind(this),
            'hyperdimensional-projection': this.createHyperdimensionalProjection.bind(this),
            'sacred-geometry': this.createSacredGeometry.bind(this)
        };
    }

    // Main visualization creation method
    createVisualization(containerId, type, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`Container ${containerId} not found`);
            return null;
        }

        // Show loading indicator
        this.showLoading(container);

        // Initialize parameters for this visualization
        this.parameterStates.set(containerId, this.getDefaultParameters(type));

        // Create the visualization
        if (this.vizTypes[type]) {
            const viz = this.vizTypes[type](containerId, options);
            this.visualizations.set(containerId, viz);
            return viz;
        } else {
            console.error(`Visualization type ${type} not found`);
            return null;
        }
    }

    // Consciousness Field 3D with interactive parameters
    createConsciousnessField3D(containerId, options = {}) {
        const params = this.parameterStates.get(containerId);
        const fieldData = this.generateConsciousnessFieldData(params);

        const trace = {
            x: fieldData.x,
            y: fieldData.y,
            z: fieldData.z,
            type: 'surface',
            colorscale: this.colorSchemes.consciousness.map((color, i) => [i / 3, color]),
            showscale: true,
            colorbar: {
                title: 'Consciousness Intensity',
                titlefont: { color: '#FFD700' },
                tickfont: { color: '#e6edf3' }
            }
        };

        const layout = {
            title: {
                text: 'Consciousness Field: C(x,y,t) = φ·sin(x·φ)·cos(y·φ)·e^(-t/φ)',
                font: { color: '#FFD700', size: 16 }
            },
            scene: {
                bgcolor: 'rgba(0,0,0,0)',
                xaxis: { 
                    title: 'Space X', 
                    titlefont: { color: '#FFD700' },
                    tickfont: { color: '#e6edf3' }
                },
                yaxis: { 
                    title: 'Space Y', 
                    titlefont: { color: '#FFD700' },
                    tickfont: { color: '#e6edf3' }
                },
                zaxis: { 
                    title: 'Consciousness', 
                    titlefont: { color: '#FFD700' },
                    tickfont: { color: '#e6edf3' }
                },
                camera: { eye: { x: 1.5, y: 1.5, z: 1.5 } }
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { l: 0, r: 0, t: 40, b: 0 }
        };

        Plotly.newPlot(containerId, [trace], layout);
        this.hideLoading(containerId);

        // Add interactive controls
        this.addParameterControls(containerId, [
            { name: 'phi_factor', label: 'φ Factor', min: 0.5, max: 3.0, step: 0.1, value: params.phi_factor },
            { name: 'temporal_rate', label: 'Time Rate', min: 0.1, max: 5.0, step: 0.1, value: params.temporal_rate },
            { name: 'spatial_freq', label: 'Spatial Freq', min: 0.5, max: 2.0, step: 0.1, value: params.spatial_freq },
            { name: 'resolution', label: 'Resolution', min: 20, max: 100, step: 10, value: params.resolution }
        ], (param, value) => this.updateConsciousnessField(containerId, param, value));

        return trace;
    }

    // Phi Spiral 3D with golden ratio mathematics
    createPhiSpiral3D(containerId, options = {}) {
        const params = this.parameterStates.get(containerId);
        const spiralData = this.generatePhiSpiralData(params);

        const trace = {
            x: spiralData.x,
            y: spiralData.y,
            z: spiralData.z,
            mode: 'lines+markers',
            type: 'scatter3d',
            line: { 
                color: spiralData.colors,
                width: params.line_width,
                colorscale: 'Viridis'
            },
            marker: { 
                size: params.marker_size, 
                color: spiralData.colors,
                colorscale: 'Viridis'
            }
        };

        const layout = {
            title: {
                text: 'φ-Harmonic Spiral: Golden Ratio Unity Demonstration',
                font: { color: '#FFD700', size: 16 }
            },
            scene: {
                bgcolor: 'rgba(0,0,0,0)',
                xaxis: { title: 'X', titlefont: { color: '#FFD700' }, tickfont: { color: '#e6edf3' } },
                yaxis: { title: 'Y', titlefont: { color: '#FFD700' }, tickfont: { color: '#e6edf3' } },
                zaxis: { title: 'Z', titlefont: { color: '#FFD700' }, tickfont: { color: '#e6edf3' } },
                camera: { eye: { x: 1.5, y: 1.5, z: 1.5 } }
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            margin: { l: 0, r: 0, t: 40, b: 0 }
        };

        Plotly.newPlot(containerId, [trace], layout);
        this.hideLoading(containerId);

        this.addParameterControls(containerId, [
            { name: 'turns', label: 'Spiral Turns', min: 2, max: 10, step: 0.5, value: params.turns },
            { name: 'growth_rate', label: 'Growth Rate', min: 0.5, max: 2.0, step: 0.1, value: params.growth_rate },
            { name: 'line_width', label: 'Line Width', min: 2, max: 15, step: 1, value: params.line_width },
            { name: 'marker_size', label: 'Marker Size', min: 1, max: 8, step: 0.5, value: params.marker_size }
        ], (param, value) => this.updatePhiSpiral(containerId, param, value));

        return trace;
    }

    // Quantum Bloch Sphere with unity states
    createQuantumBlochSphere(containerId, options = {}) {
        const params = this.parameterStates.get(containerId);
        
        // Sphere surface
        const sphereData = this.generateBlochSphere();
        const sphereTrace = {
            x: sphereData.x,
            y: sphereData.y,
            z: sphereData.z,
            type: 'mesh3d',
            opacity: 0.3,
            color: '#4ECDC4'
        };

        // Unity state vector
        const unityTrace = {
            x: [0, params.unity_x],
            y: [0, params.unity_y],
            z: [0, params.unity_z],
            mode: 'lines+markers',
            type: 'scatter3d',
            line: { color: '#FFD700', width: 8 },
            marker: { size: [8, 12], color: ['#FFD700', '#FF6B6B'] },
            name: 'Unity State |ψ⟩'
        };

        const layout = {
            title: {
                text: 'Quantum Unity Bloch Sphere: |1⟩ + |1⟩ → |1⟩',
                font: { color: '#FFD700', size: 16 }
            },
            scene: {
                bgcolor: 'rgba(0,0,0,0)',
                xaxis: { title: '|+⟩-|-⟩', titlefont: { color: '#FFD700' }, range: [-1.5, 1.5] },
                yaxis: { title: '|+i⟩-|-i⟩', titlefont: { color: '#FFD700' }, range: [-1.5, 1.5] },
                zaxis: { title: '|0⟩-|1⟩', titlefont: { color: '#FFD700' }, range: [-1.5, 1.5] },
                camera: { eye: { x: 1.5, y: 1.5, z: 1.5 } }
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            margin: { l: 0, r: 0, t: 40, b: 0 }
        };

        Plotly.newPlot(containerId, [sphereTrace, unityTrace], layout);
        this.hideLoading(containerId);

        this.addParameterControls(containerId, [
            { name: 'theta', label: 'Θ (Polar)', min: 0, max: Math.PI, step: 0.1, value: params.theta },
            { name: 'phi', label: 'Φ (Azimuth)', min: 0, max: 2*Math.PI, step: 0.1, value: params.phi },
            { name: 'coherence', label: 'Coherence', min: 0, max: 1, step: 0.05, value: params.coherence }
        ], (param, value) => this.updateQuantumState(containerId, param, value));

        return { sphere: sphereTrace, unity: unityTrace };
    }

    // Unity Manifold with topological features  
    createUnityManifold(containerId, options = {}) {
        const params = this.parameterStates.get(containerId);
        const manifoldData = this.generateUnityManifold(params);

        const trace = {
            x: manifoldData.x,
            y: manifoldData.y,
            z: manifoldData.z,
            type: 'surface',
            colorscale: this.colorSchemes.unity.map((color, i) => [i / 3, color]),
            showscale: true,
            contours: {
                z: {
                    show: true,
                    usecolormap: true,
                    highlightcolor: "#FFD700",
                    project: { z: true }
                }
            }
        };

        const layout = {
            title: {
                text: 'Unity Manifold: Topological 1+1=1 Demonstration',
                font: { color: '#FFD700', size: 16 }
            },
            scene: {
                bgcolor: 'rgba(0,0,0,0)',
                xaxis: { title: 'Dimension 1', titlefont: { color: '#FFD700' } },
                yaxis: { title: 'Dimension 2', titlefont: { color: '#FFD700' } },
                zaxis: { title: 'Unity Field', titlefont: { color: '#FFD700' } },
                camera: { eye: { x: 1.5, y: 1.5, z: 1.5 } }
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            margin: { l: 0, r: 0, t: 40, b: 0 }
        };

        Plotly.newPlot(containerId, [trace], layout);
        this.hideLoading(containerId);

        this.addParameterControls(containerId, [
            { name: 'curvature', label: 'Curvature', min: 0.1, max: 2.0, step: 0.1, value: params.curvature },
            { name: 'topology_type', label: 'Topology', min: 1, max: 5, step: 1, value: params.topology_type },
            { name: 'field_strength', label: 'Field Strength', min: 0.5, max: 3.0, step: 0.1, value: params.field_strength }
        ], (param, value) => this.updateUnityManifold(containerId, param, value));

        return trace;
    }

    // Data Generation Methods
    generateConsciousnessFieldData(params) {
        const { phi_factor, temporal_rate, spatial_freq, resolution, time } = params;
        const x = [], y = [], z = [];
        const range = 4;
        const step = (2 * range) / resolution;
        const t = time || 0;

        for (let i = 0; i < resolution; i++) {
            const row_x = [], row_y = [], row_z = [];
            for (let j = 0; j < resolution; j++) {
                const xi = -range + i * step;
                const yi = -range + j * step;
                
                // Consciousness field equation: C(x,y,t) = φ·sin(x·φ)·cos(y·φ)·e^(-t/φ)
                const zi = phi_factor * this.PHI * 
                          Math.sin(xi * this.PHI * spatial_freq) * 
                          Math.cos(yi * this.PHI * spatial_freq) * 
                          Math.exp(-t * temporal_rate / this.PHI);
                
                row_x.push(xi);
                row_y.push(yi);
                row_z.push(zi);
            }
            x.push(row_x);
            y.push(row_y);
            z.push(row_z);
        }

        return { x, y, z };
    }

    generatePhiSpiralData(params) {
        const { turns, growth_rate, points } = params;
        const x = [], y = [], z = [], colors = [];
        const maxT = turns * 2 * Math.PI;

        for (let i = 0; i <= points; i++) {
            const t = (i / points) * maxT;
            const r = Math.pow(this.PHI, t * growth_rate / (2 * Math.PI));
            
            x.push(r * Math.cos(t));
            y.push(r * Math.sin(t));
            z.push(t / (2 * Math.PI));
            colors.push(i / points); // Color gradient along spiral
        }

        return { x, y, z, colors };
    }

    generateBlochSphere() {
        const x = [], y = [], z = [];
        const resolution = 20;

        for (let i = 0; i <= resolution; i++) {
            for (let j = 0; j <= resolution; j++) {
                const theta = (i / resolution) * Math.PI;
                const phi = (j / resolution) * 2 * Math.PI;
                
                x.push(Math.sin(theta) * Math.cos(phi));
                y.push(Math.sin(theta) * Math.sin(phi));
                z.push(Math.cos(theta));
            }
        }

        return { x, y, z };
    }

    generateUnityManifold(params) {
        const { curvature, field_strength, resolution } = params;
        const x = [], y = [], z = [];
        const range = 3;
        const step = (2 * range) / resolution;

        for (let i = 0; i < resolution; i++) {
            const row_x = [], row_y = [], row_z = [];
            for (let j = 0; j < resolution; j++) {
                const xi = -range + i * step;
                const yi = -range + j * step;
                
                // Unity manifold equation
                const zi = field_strength * Math.exp(-curvature * (xi*xi + yi*yi)) * 
                          Math.cos(xi * this.PHI) * Math.sin(yi * this.PHI);
                
                row_x.push(xi);
                row_y.push(yi);
                row_z.push(zi);
            }
            x.push(row_x);
            y.push(row_y);
            z.push(row_z);
        }

        return { x, y, z };
    }

    // Parameter Control System
    addParameterControls(containerId, parameters, updateCallback) {
        const container = document.getElementById(containerId);
        if (!container) return;

        // Create controls overlay
        const overlay = document.createElement('div');
        overlay.className = 'viz-controls-overlay';
        overlay.id = containerId + '-controls';

        parameters.forEach(param => {
            const row = document.createElement('div');
            row.className = 'parameter-row';

            row.innerHTML = `
                <label class="parameter-label">${param.label}:</label>
                <input type="range" 
                       class="parameter-slider" 
                       min="${param.min}" 
                       max="${param.max}" 
                       step="${param.step}" 
                       value="${param.value}"
                       oninput="vizEngine.updateParameter('${containerId}', '${param.name}', this.value)">
                <span class="parameter-value" id="${containerId}-${param.name}-value">${param.value}</span>
            `;

            overlay.appendChild(row);
        });

        // Add toggle button
        const toggleBtn = document.createElement('button');
        toggleBtn.innerHTML = '<i class="fas fa-sliders-h"></i> Controls';
        toggleBtn.className = 'viz-btn primary';
        toggleBtn.style.cssText = 'position: absolute; top: 10px; left: 10px; z-index: 100; padding: 0.5rem 1rem; font-size: 0.875rem;';
        toggleBtn.onclick = () => overlay.classList.toggle('active');

        container.style.position = 'relative';
        container.appendChild(toggleBtn);
        container.appendChild(overlay);

        // Store update callback
        this.updateCallbacks = this.updateCallbacks || new Map();
        this.updateCallbacks.set(containerId, updateCallback);
    }

    updateParameter(containerId, paramName, value) {
        // Update parameter state
        const params = this.parameterStates.get(containerId);
        if (params) {
            params[paramName] = parseFloat(value);
        }

        // Update display
        const valueElement = document.getElementById(`${containerId}-${paramName}-value`);
        if (valueElement) {
            valueElement.textContent = value;
        }

        // Call update callback
        const callback = this.updateCallbacks?.get(containerId);
        if (callback) {
            callback(paramName, parseFloat(value));
        }
    }

    // Specific update methods for each visualization type
    updateConsciousnessField(containerId, param, value) {
        const params = this.parameterStates.get(containerId);
        const newData = this.generateConsciousnessFieldData(params);
        
        Plotly.restyle(containerId, {
            x: [newData.x],
            y: [newData.y],
            z: [newData.z]
        });
    }

    updatePhiSpiral(containerId, param, value) {
        const params = this.parameterStates.get(containerId);
        const newData = this.generatePhiSpiralData(params);
        
        Plotly.restyle(containerId, {
            x: [newData.x],
            y: [newData.y],
            z: [newData.z],
            'line.color': [newData.colors],
            'marker.color': [newData.colors],
            'line.width': [params.line_width],
            'marker.size': [params.marker_size]
        });
    }

    updateQuantumState(containerId, param, value) {
        const params = this.parameterStates.get(containerId);
        
        // Calculate new quantum state position
        params.unity_x = Math.sin(params.theta) * Math.cos(params.phi) * params.coherence;
        params.unity_y = Math.sin(params.theta) * Math.sin(params.phi) * params.coherence;
        params.unity_z = Math.cos(params.theta) * params.coherence;
        
        Plotly.restyle(containerId, {
            x: [[0, params.unity_x]],
            y: [[0, params.unity_y]],
            z: [[0, params.unity_z]]
        }, [1]); // Update second trace (unity vector)
    }

    updateUnityManifold(containerId, param, value) {
        const params = this.parameterStates.get(containerId);
        const newData = this.generateUnityManifold(params);
        
        Plotly.restyle(containerId, {
            x: [newData.x],
            y: [newData.y],
            z: [newData.z]
        });
    }

    // Default parameter sets for each visualization type
    getDefaultParameters(type) {
        const defaults = {
            'consciousness-field-3d': {
                phi_factor: 1.0,
                temporal_rate: 1.0,
                spatial_freq: 1.0,
                resolution: 50,
                time: 0
            },
            'phi-spiral-3d': {
                turns: 4,
                growth_rate: 1.0,
                line_width: 6,
                marker_size: 3,
                points: 200
            },
            'quantum-bloch-sphere': {
                theta: Math.PI / 4,
                phi: Math.PI / 4,
                coherence: 1.0,
                unity_x: 0.5,
                unity_y: 0.5,
                unity_z: 0.707
            },
            'unity-manifold': {
                curvature: 0.5,
                topology_type: 1,
                field_strength: 1.0,
                resolution: 40
            }
        };

        return defaults[type] || {};
    }

    // Utility methods
    showLoading(container) {
        const loading = document.createElement('div');
        loading.className = 'loading-indicator';
        loading.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating Visualization...';
        loading.id = container.id + '-loading';
        container.appendChild(loading);
    }

    hideLoading(containerId) {
        const loading = document.getElementById(containerId + '-loading');
        if (loading) {
            loading.remove();
        }
    }

    // Static image generation for academic backup
    generateStaticImage(containerId, filename = 'unity-visualization') {
        Plotly.downloadImage(containerId, {
            format: 'png',
            width: 1200,
            height: 900,
            filename: filename
        });
    }

    // Animation system
    startAnimation(containerId, animationType = 'rotation') {
        this.animationStates.set(containerId, true);
        this.animate(containerId, animationType);
    }

    stopAnimation(containerId) {
        this.animationStates.set(containerId, false);
    }

    animate(containerId, type) {
        if (!this.animationStates.get(containerId)) return;

        const time = Date.now() * 0.001;

        if (type === 'rotation') {
            Plotly.relayout(containerId, {
                'scene.camera': {
                    eye: {
                        x: 2 * Math.cos(time * 0.5),
                        y: 2 * Math.sin(time * 0.5),
                        z: 1.5
                    }
                }
            });
        } else if (type === 'consciousness-evolution') {
            const params = this.parameterStates.get(containerId);
            if (params) {
                params.time = time;
                this.updateConsciousnessField(containerId, 'time', time);
            }
        }

        if (this.animationStates.get(containerId)) {
            requestAnimationFrame(() => this.animate(containerId, type));
        }
    }

    // Batch processing for generating all static backups
    generateAllStaticBackups() {
        this.visualizations.forEach((viz, containerId) => {
            this.generateStaticImage(containerId, `unity-mathematics-${containerId}-backup`);
        });
    }

    setupEventListeners() {
        // Global keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 's') {
                e.preventDefault();
                this.generateAllStaticBackups();
            }
        });
    }
}

// Initialize global visualization engine
const vizEngine = new UnityVisualizationEngine();

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UnityVisualizationEngine;
}

console.log('🎨 Unity Visualization Engine loaded successfully');
console.log('Available visualizations:', Object.keys(vizEngine.vizTypes));
console.log('Keyboard shortcuts: Ctrl+S to generate all static backups');