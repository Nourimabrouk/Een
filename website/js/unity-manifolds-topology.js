/**
 * Unity Manifolds & Topology Visualizer
 * Interactive Möbius strips, Klein bottles, and hyperdimensional projections
 */

class UnityManifoldsVisualizer {
    constructor(containerId) {
        this.containerId = containerId;
        this.PHI = (1 + Math.sqrt(5)) / 2;
        this.currentManifold = 'mobius';
        this.animationFrame = null;
        this.isAnimating = false;
        this.startTime = Date.now();
        
        // Configuration
        this.config = {
            resolution: 50,
            animation_speed: 1.0,
            show_wireframe: true,
            show_unity_points: true,
            color_scheme: 'unity_gradient',
            dimension_projection: '3d'
        };
        
        // Color schemes
        this.colorSchemes = {
            unity_gradient: ['#3B82F6', '#10B981', '#F59E0B', '#DC2626'],
            topology_rainbow: ['#8B5CF6', '#3B82F6', '#10B981', '#F59E0B', '#EF4444'],
            phi_harmonic: ['#0F7B8A', '#4ECDC4', '#A7F3D0', '#F59E0B']
        };
        
        this.init();
    }
    
    init() {
        this.createContainer();
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
            <div class="unity-manifolds-container" style="
                background: linear-gradient(135deg, rgba(139, 92, 246, 0.03) 0%, rgba(16, 185, 129, 0.05) 100%);
                border-radius: 1rem;
                padding: 1.5rem;
                box-shadow: 0 8px 32px rgba(139, 92, 246, 0.2);
                margin: 1rem 0;
                position: relative;
                overflow: hidden;
            ">
                <!-- Animated topological background -->
                <div style="
                    position: absolute;
                    top: -50%;
                    left: -50%;
                    width: 200%;
                    height: 200%;
                    background: conic-gradient(from 0deg, rgba(139, 92, 246, 0.1), rgba(16, 185, 129, 0.1), rgba(245, 158, 11, 0.1), rgba(139, 92, 246, 0.1));
                    animation: topology-rotation 8s linear infinite;
                    z-index: 0;
                "></div>
                
                <div class="header" style="
                    text-align: center;
                    margin-bottom: 1.5rem;
                    position: relative;
                    z-index: 1;
                ">
                    <h3 style="
                        color: #8B5CF6;
                        margin: 0 0 0.5rem 0;
                        font-size: 1.8rem;
                        background: linear-gradient(135deg, #8B5CF6, #10B981);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        background-clip: text;
                    ">🌀 Unity Manifolds & Topology</h3>
                    <p style="
                        color: #6B7280;
                        margin: 0 0 1rem 0;
                        font-size: 1rem;
                    ">Interactive exploration of topological unity through manifolds</p>
                    <div style="
                        font-family: 'Times New Roman', serif;
                        font-size: 1.1rem;
                        color: #8B5CF6;
                        background: rgba(139, 92, 246, 0.1);
                        padding: 0.5rem 1rem;
                        border-radius: 0.5rem;
                        display: inline-block;
                        border: 1px solid rgba(139, 92, 246, 0.3);
                    ">
                        Topological Unity: Two Sides → One Surface
                    </div>
                </div>
                
                <div class="controls-panel" style="
                    display: flex;
                    justify-content: center;
                    gap: 1rem;
                    flex-wrap: wrap;
                    margin-bottom: 1.5rem;
                    padding: 1rem;
                    background: rgba(139, 92, 246, 0.05);
                    border-radius: 0.75rem;
                    border: 1px solid rgba(139, 92, 246, 0.2);
                    position: relative;
                    z-index: 1;
                ">
                    <select id="manifold-type-${this.containerId}" style="
                        padding: 0.5rem 1rem;
                        border: 1px solid #D1D5DB;
                        border-radius: 0.5rem;
                        background: white;
                        font-size: 0.9rem;
                    ">
                        <option value="mobius">Möbius Strip</option>
                        <option value="klein">Klein Bottle</option>
                        <option value="torus">Unity Torus</option>
                        <option value="hypersphere">4D Hypersphere</option>
                        <option value="projective">Projective Plane</option>
                    </select>
                    
                    <button id="animate-btn-${this.containerId}" style="
                        padding: 0.5rem 1rem;
                        background: #8B5CF6;
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
                        Resolution:
                        <input type="range" id="resolution-${this.containerId}" 
                               min="20" max="80" step="5" value="50"
                               style="width: 80px;">
                        <span id="resolution-value-${this.containerId}">50</span>
                    </label>
                    
                    <button id="wireframe-btn-${this.containerId}" style="
                        padding: 0.5rem 1rem;
                        background: ${this.config.show_wireframe ? '#10B981' : '#6B7280'};
                        color: white;
                        border: none;
                        border-radius: 0.5rem;
                        cursor: pointer;
                        font-size: 0.9rem;
                        transition: all 0.3s ease;
                    ">🔲 Wireframe</button>
                    
                    <button id="unity-points-btn-${this.containerId}" style="
                        padding: 0.5rem 1rem;
                        background: ${this.config.show_unity_points ? '#F59E0B' : '#6B7280'};
                        color: white;
                        border: none;
                        border-radius: 0.5rem;
                        cursor: pointer;
                        font-size: 0.9rem;
                        transition: all 0.3s ease;
                    ">🎯 Unity Points</button>
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
                
                <div class="topology-info" style="
                    margin-top: 1rem;
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 1rem;
                    position: relative;
                    z-index: 1;
                ">
                    <div style="
                        background: rgba(139, 92, 246, 0.05);
                        padding: 0.75rem;
                        border-radius: 0.5rem;
                        text-align: center;
                        border: 1px solid rgba(139, 92, 246, 0.2);
                    ">
                        <div id="genus-${this.containerId}" style="
                            font-size: 1.2rem;
                            font-weight: bold;
                            color: #8B5CF6;
                        ">1</div>
                        <div style="font-size: 0.8rem; color: #6B7280;">Genus</div>
                    </div>
                    <div style="
                        background: rgba(16, 185, 129, 0.05);
                        padding: 0.75rem;
                        border-radius: 0.5rem;
                        text-align: center;
                        border: 1px solid rgba(16, 185, 129, 0.2);
                    ">
                        <div id="orientability-${this.containerId}" style="
                            font-size: 1.2rem;
                            font-weight: bold;
                            color: #10B981;
                        ">Non-orientable</div>
                        <div style="font-size: 0.8rem; color: #6B7280;">Orientability</div>
                    </div>
                    <div style="
                        background: rgba(245, 158, 11, 0.05);
                        padding: 0.75rem;
                        border-radius: 0.5rem;
                        text-align: center;
                        border: 1px solid rgba(245, 158, 11, 0.2);
                    ">
                        <div id="euler-char-${this.containerId}" style="
                            font-size: 1.2rem;
                            font-weight: bold;
                            color: #F59E0B;
                        ">0</div>
                        <div style="font-size: 0.8rem; color: #6B7280;">Euler Characteristic</div>
                    </div>
                    <div style="
                        background: rgba(220, 38, 38, 0.05);
                        padding: 0.75rem;
                        border-radius: 0.5rem;
                        text-align: center;
                        border: 1px solid rgba(220, 38, 38, 0.2);
                    ">
                        <div id="unity-property-${this.containerId}" style="
                            font-size: 1.2rem;
                            font-weight: bold;
                            color: #DC2626;
                        ">Two → One</div>
                        <div style="font-size: 0.8rem; color: #6B7280;">Unity Property</div>
                    </div>
                </div>
            </div>
            
            <style>
                @keyframes topology-rotation {
                    0% { transform: rotate(0deg) scale(1); }
                    50% { transform: rotate(180deg) scale(1.1); }
                    100% { transform: rotate(360deg) scale(1); }
                }
            </style>
        `;
        
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        const manifoldSelect = document.getElementById(`manifold-type-${this.containerId}`);
        const animateBtn = document.getElementById(`animate-btn-${this.containerId}`);
        const resolutionSlider = document.getElementById(`resolution-${this.containerId}`);
        const wireframeBtn = document.getElementById(`wireframe-btn-${this.containerId}`);
        const unityPointsBtn = document.getElementById(`unity-points-btn-${this.containerId}`);
        
        const resolutionValue = document.getElementById(`resolution-value-${this.containerId}`);
        
        if (manifoldSelect) {
            manifoldSelect.addEventListener('change', (e) => {
                this.currentManifold = e.target.value;
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
        
        if (resolutionSlider && resolutionValue) {
            resolutionSlider.addEventListener('input', (e) => {
                this.config.resolution = parseInt(e.target.value);
                resolutionValue.textContent = this.config.resolution;
                this.updateVisualization();
            });
        }
        
        if (wireframeBtn) {
            wireframeBtn.addEventListener('click', () => {
                this.config.show_wireframe = !this.config.show_wireframe;
                wireframeBtn.style.background = this.config.show_wireframe ? '#10B981' : '#6B7280';
                this.updateVisualization();
            });
        }
        
        if (unityPointsBtn) {
            unityPointsBtn.addEventListener('click', () => {
                this.config.show_unity_points = !this.config.show_unity_points;
                unityPointsBtn.style.background = this.config.show_unity_points ? '#F59E0B' : '#6B7280';
                this.updateVisualization();
            });
        }
    }
    
    createVisualization() {
        this.updateVisualization();
    }
    
    updateVisualization() {
        const plotDiv = document.getElementById(`plot-${this.containerId}`);
        if (!plotDiv) return;
        
        let data, layout;
        
        switch (this.currentManifold) {
            case 'mobius':
                ({ data, layout } = this.createMobiusStrip());
                break;
            case 'klein':
                ({ data, layout } = this.createKleinBottle());
                break;
            case 'torus':
                ({ data, layout } = this.createUnityTorus());
                break;
            case 'hypersphere':
                ({ data, layout } = this.createHypersphere());
                break;
            case 'projective':
                ({ data, layout } = this.createProjectivePlane());
                break;
        }
        
        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
            displaylogo: false
        };
        
        Plotly.newPlot(plotDiv, data, layout, config);
        this.updateTopologyInfo();
    }
    
    createMobiusStrip() {
        const resolution = this.config.resolution;
        const currentTime = this.isAnimating ? (Date.now() - this.startTime) * 0.001 * this.config.animation_speed : 0;
        
        // Möbius strip parametrization
        const u = this.linspace(0, 2*Math.PI, resolution);
        const v = this.linspace(-0.5, 0.5, Math.floor(resolution/2));
        
        const x = [];
        const y = [];
        const z = [];
        const colors = [];
        
        for (let i = 0; i < u.length; i++) {
            const xRow = [];
            const yRow = [];
            const zRow = [];
            const colorRow = [];
            
            for (let j = 0; j < v.length; j++) {
                const u_val = u[i] + currentTime * 0.1;
                const v_val = v[j];
                
                // Möbius strip equations with φ-harmonic modulation
                const radius = 1 + v_val * Math.cos(u_val / 2);
                const x_val = radius * Math.cos(u_val);
                const y_val = radius * Math.sin(u_val);
                const z_val = v_val * Math.sin(u_val / 2);
                
                xRow.push(x_val);
                yRow.push(y_val);
                zRow.push(z_val);
                
                // Color based on position for unity visualization
                const colorValue = Math.sin(u_val * this.PHI) * Math.cos(v_val * 2);
                colorRow.push(colorValue);
            }
            
            x.push(xRow);
            y.push(yRow);
            z.push(zRow);
            colors.push(colorRow);
        }
        
        const data = [{
            x: x,
            y: y,
            z: z,
            type: 'surface',
            surfacecolor: colors,
            colorscale: 'Viridis',
            opacity: this.config.show_wireframe ? 0.8 : 0.9,
            name: 'Möbius Strip'
        }];
        
        // Add unity points if enabled
        if (this.config.show_unity_points) {
            // Special points where the strip connects to itself
            const unityPoints = this.getMobiusUnityPoints(currentTime);
            data.push({
                x: unityPoints.x,
                y: unityPoints.y,
                z: unityPoints.z,
                type: 'scatter3d',
                mode: 'markers',
                marker: {
                    size: 8,
                    color: '#F59E0B',
                    symbol: 'diamond'
                },
                name: 'Unity Points'
            });
        }
        
        const layout = {
            title: 'Möbius Strip: Two Sides Become One Surface',
            scene: {
                xaxis: { range: [-2, 2] },
                yaxis: { range: [-2, 2] },
                zaxis: { range: [-1, 1] },
                bgcolor: 'rgba(0,0,0,0)',
                camera: { eye: { x: 1.5, y: 1.5, z: 1.5 } }
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            margin: { l: 0, r: 0, t: 50, b: 0 }
        };
        
        return { data, layout };
    }
    
    createKleinBottle() {
        const resolution = Math.floor(this.config.resolution * 0.8);
        const currentTime = this.isAnimating ? (Date.now() - this.startTime) * 0.001 * this.config.animation_speed : 0;
        
        // Klein bottle parametrization (figure-8 immersion)
        const u = this.linspace(0, 2*Math.PI, resolution);
        const v = this.linspace(0, 2*Math.PI, resolution);
        
        const x = [];
        const y = [];
        const z = [];
        const colors = [];
        
        for (let i = 0; i < u.length; i++) {
            const xRow = [];
            const yRow = [];
            const zRow = [];
            const colorRow = [];
            
            for (let j = 0; j < v.length; j++) {
                const u_val = u[i];
                const v_val = v[j] + currentTime * 0.2;
                
                // Klein bottle equations
                let x_val, y_val, z_val;
                
                if (u_val < Math.PI) {
                    x_val = 3 * Math.cos(u_val) * (1 + Math.sin(u_val)) + 
                           (2 * (1 - Math.cos(u_val) / 2)) * Math.cos(u_val) * Math.cos(v_val);
                    z_val = -8 * Math.sin(u_val) - 
                           2 * (1 - Math.cos(u_val) / 2) * Math.sin(u_val) * Math.cos(v_val);
                } else {
                    x_val = 3 * Math.cos(u_val) * (1 + Math.sin(u_val)) + 
                           (2 * (1 - Math.cos(u_val) / 2)) * Math.cos(v_val + Math.PI);
                    z_val = -8 * Math.sin(u_val);
                }
                
                y_val = (2 * (1 - Math.cos(u_val) / 2)) * Math.sin(v_val);
                
                // Scale for better visualization
                x_val *= 0.1;
                y_val *= 0.1;
                z_val *= 0.1;
                
                xRow.push(x_val);
                yRow.push(y_val);
                zRow.push(z_val);
                
                // φ-harmonic coloring
                const colorValue = Math.sin(u_val * this.PHI) * Math.cos(v_val / this.PHI);
                colorRow.push(colorValue);
            }
            
            x.push(xRow);
            y.push(yRow);
            z.push(zRow);
            colors.push(colorRow);
        }
        
        const data = [{
            x: x,
            y: y,
            z: z,
            type: 'surface',
            surfacecolor: colors,
            colorscale: 'Plasma',
            opacity: 0.8,
            name: 'Klein Bottle'
        }];
        
        const layout = {
            title: 'Klein Bottle: Inside Becomes Outside',
            scene: {
                xaxis: { range: [-1, 1] },
                yaxis: { range: [-1, 1] },
                zaxis: { range: [-1, 1] },
                bgcolor: 'rgba(0,0,0,0)',
                camera: { eye: { x: 2, y: 2, z: 1.5 } }
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            margin: { l: 0, r: 0, t: 50, b: 0 }
        };
        
        return { data, layout };
    }
    
    createUnityTorus() {
        const resolution = this.config.resolution;
        const currentTime = this.isAnimating ? (Date.now() - this.startTime) * 0.001 * this.config.animation_speed : 0;
        
        // φ-harmonic torus
        const u = this.linspace(0, 2*Math.PI, resolution);
        const v = this.linspace(0, 2*Math.PI, resolution);
        
        const R = this.PHI; // Major radius
        const r = 1 / this.PHI; // Minor radius
        
        const x = [];
        const y = [];
        const z = [];
        const colors = [];
        
        for (let i = 0; i < u.length; i++) {
            const xRow = [];
            const yRow = [];
            const zRow = [];
            const colorRow = [];
            
            for (let j = 0; j < v.length; j++) {
                const u_val = u[i] + currentTime * 0.3;
                const v_val = v[j] + currentTime * 0.2;
                
                const x_val = (R + r * Math.cos(v_val)) * Math.cos(u_val);
                const y_val = (R + r * Math.cos(v_val)) * Math.sin(u_val);
                const z_val = r * Math.sin(v_val);
                
                xRow.push(x_val);
                yRow.push(y_val);
                zRow.push(z_val);
                
                // Unity color pattern
                const colorValue = Math.sin(this.PHI * u_val) * Math.cos(this.PHI * v_val);
                colorRow.push(colorValue);
            }
            
            x.push(xRow);
            y.push(yRow);
            z.push(zRow);
            colors.push(colorRow);
        }
        
        const data = [{
            x: x,
            y: y,
            z: z,
            type: 'surface',
            surfacecolor: colors,
            colorscale: 'RdYlBu',
            opacity: 0.85,
            name: 'φ-Harmonic Torus'
        }];
        
        // Add meridian and longitude circles showing unity
        if (this.config.show_unity_points) {
            const circles = this.getTorusUnityCircles(R, r, currentTime);
            data.push(...circles);
        }
        
        const layout = {
            title: 'φ-Harmonic Unity Torus: Continuous Unity Surface',
            scene: {
                xaxis: { range: [-3, 3] },
                yaxis: { range: [-3, 3] },
                zaxis: { range: [-1, 1] },
                bgcolor: 'rgba(0,0,0,0)',
                aspectmode: 'cube',
                camera: { eye: { x: 1.8, y: 1.8, z: 1.2 } }
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            margin: { l: 0, r: 0, t: 50, b: 0 }
        };
        
        return { data, layout };
    }
    
    createHypersphere() {
        // 4D hypersphere projected to 3D
        const resolution = Math.floor(this.config.resolution * 0.7);
        const currentTime = this.isAnimating ? (Date.now() - this.startTime) * 0.001 * this.config.animation_speed : 0;
        
        const data = [];
        const nSpheres = 8; // Number of 3D cross-sections
        
        for (let k = 0; k < nSpheres; k++) {
            const w = -1 + (2 * k) / (nSpheres - 1); // 4th dimension coordinate
            const radius = Math.sqrt(Math.max(0, 1 - w*w)); // Radius at this w-slice
            
            if (radius < 0.1) continue; // Skip very small spheres
            
            // Create 3D sphere at this w-slice
            const u = this.linspace(0, 2*Math.PI, resolution);
            const v = this.linspace(0, Math.PI, Math.floor(resolution/2));
            
            const x = [];
            const y = [];
            const z = [];
            
            for (let i = 0; i < u.length; i++) {
                const xRow = [];
                const yRow = [];
                const zRow = [];
                
                for (let j = 0; j < v.length; j++) {
                    const u_val = u[i] + currentTime * 0.4;
                    const v_val = v[j];
                    
                    xRow.push(radius * Math.sin(v_val) * Math.cos(u_val));
                    yRow.push(radius * Math.sin(v_val) * Math.sin(u_val));
                    zRow.push(radius * Math.cos(v_val) + w * 0.5); // Offset by w
                }
                
                x.push(xRow);
                y.push(yRow);
                z.push(zRow);
            }
            
            // Color based on 4D position
            const intensity = (w + 1) / 2; // Normalize to [0,1]
            const color = `rgba(${Math.floor(139 * intensity)}, ${Math.floor(92 * (1-intensity))}, 246, 0.6)`;
            
            data.push({
                x: x,
                y: y,
                z: z,
                type: 'surface',
                opacity: 0.4,
                colorscale: [[0, color], [1, color]],
                showscale: false,
                name: `4D Slice ${k+1}`
            });
        }
        
        const layout = {
            title: '4D Hypersphere Projection: Unity Across Dimensions',
            scene: {
                xaxis: { range: [-1.5, 1.5] },
                yaxis: { range: [-1.5, 1.5] },
                zaxis: { range: [-1.5, 1.5] },
                bgcolor: 'rgba(0,0,0,0)',
                aspectmode: 'cube',
                camera: { eye: { x: 2, y: 2, z: 2 } }
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            margin: { l: 0, r: 0, t: 50, b: 0 }
        };
        
        return { data, layout };
    }
    
    createProjectivePlane() {
        // Real projective plane (Boy's surface approximation)
        const resolution = Math.floor(this.config.resolution * 0.8);
        const currentTime = this.isAnimating ? (Date.now() - this.startTime) * 0.001 * this.config.animation_speed : 0;
        
        const u = this.linspace(0, Math.PI, resolution);
        const v = this.linspace(0, Math.PI, resolution);
        
        const x = [];
        const y = [];
        const z = [];
        const colors = [];
        
        for (let i = 0; i < u.length; i++) {
            const xRow = [];
            const yRow = [];
            const zRow = [];
            const colorRow = [];
            
            for (let j = 0; j < v.length; j++) {
                const u_val = u[i];
                const v_val = v[j] + currentTime * 0.15;
                
                // Boy's surface parametrization (simplified)
                const cosu = Math.cos(u_val);
                const sinu = Math.sin(u_val);
                const cosv = Math.cos(v_val);
                const sinv = Math.sin(v_val);
                const cos2u = Math.cos(2*u_val);
                const sin2u = Math.sin(2*u_val);
                
                const denom = 2 - Math.sqrt(2) * sinu * cosv;
                
                const x_val = (Math.sqrt(2) * cosu * cosv + cos2u * sinv) / denom;
                const y_val = (Math.sqrt(2) * sinu * cosv - sin2u * sinv) / denom;
                const z_val = (3 * sinv) / (2 * denom);
                
                xRow.push(x_val);
                yRow.push(y_val);
                zRow.push(z_val);
                
                // Unity-themed coloring
                const colorValue = Math.sin(u_val * this.PHI) * Math.sin(v_val / this.PHI);
                colorRow.push(colorValue);
            }
            
            x.push(xRow);
            y.push(yRow);
            z.push(zRow);
            colors.push(colorRow);
        }
        
        const data = [{
            x: x,
            y: y,
            z: z,
            type: 'surface',
            surfacecolor: colors,
            colorscale: 'Turbo',
            opacity: 0.8,
            name: 'Projective Plane'
        }];
        
        const layout = {
            title: 'Real Projective Plane: Points at Infinity Unite',
            scene: {
                xaxis: { range: [-2, 2] },
                yaxis: { range: [-2, 2] },
                zaxis: { range: [-2, 2] },
                bgcolor: 'rgba(0,0,0,0)',
                camera: { eye: { x: 1.8, y: 1.8, z: 1.8 } }
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            margin: { l: 0, r: 0, t: 50, b: 0 }
        };
        
        return { data, layout };
    }
    
    // Helper methods
    getMobiusUnityPoints(time) {
        const points = 6;
        const angles = this.linspace(0, 2*Math.PI, points);
        const x = [];
        const y = [];
        const z = [];
        
        for (let angle of angles) {
            const u_val = angle + time * 0.1;
            const v_val = 0; // Center line
            
            const radius = 1;
            x.push(radius * Math.cos(u_val));
            y.push(radius * Math.sin(u_val));
            z.push(0);
        }
        
        return { x, y, z };
    }
    
    getTorusUnityCircles(R, r, time) {
        const circles = [];
        const nCircles = 4;
        
        for (let i = 0; i < nCircles; i++) {
            const angle = i * Math.PI / (nCircles - 1) + time * 0.2;
            const circlePoints = this.linspace(0, 2*Math.PI, 30);
            
            const x = circlePoints.map(t => (R + r * Math.cos(t)) * Math.cos(angle));
            const y = circlePoints.map(t => (R + r * Math.cos(t)) * Math.sin(angle));
            const z = circlePoints.map(t => r * Math.sin(t));
            
            circles.push({
                x: x,
                y: y,
                z: z,
                type: 'scatter3d',
                mode: 'lines',
                line: { color: '#F59E0B', width: 4 },
                name: `Unity Circle ${i+1}`,
                showlegend: false
            });
        }
        
        return circles;
    }
    
    updateTopologyInfo() {
        const properties = {
            'mobius': { genus: 1, orientable: false, euler: 0, unity: 'Two Sides → One' },
            'klein': { genus: 1, orientable: false, euler: 0, unity: 'Inside → Outside' },
            'torus': { genus: 1, orientable: true, euler: 0, unity: 'φ-Harmonic Unity' },
            'hypersphere': { genus: 0, orientable: true, euler: 2, unity: '4D → 3D Unity' },
            'projective': { genus: 1, orientable: false, euler: 1, unity: 'Points ∞ Unite' }
        };
        
        const prop = properties[this.currentManifold];
        
        const genusEl = document.getElementById(`genus-${this.containerId}`);
        const orientabilityEl = document.getElementById(`orientability-${this.containerId}`);
        const eulerEl = document.getElementById(`euler-char-${this.containerId}`);
        const unityEl = document.getElementById(`unity-property-${this.containerId}`);
        
        if (genusEl) genusEl.textContent = prop.genus;
        if (orientabilityEl) {
            orientabilityEl.textContent = prop.orientable ? 'Orientable' : 'Non-orientable';
            orientabilityEl.style.color = prop.orientable ? '#10B981' : '#DC2626';
        }
        if (eulerEl) eulerEl.textContent = prop.euler;
        if (unityEl) unityEl.textContent = prop.unity;
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
function createUnityManifoldsTopology(containerId) {
    return new UnityManifoldsVisualizer(containerId);
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { UnityManifoldsVisualizer, createUnityManifoldsTopology };
}

// Browser global
if (typeof window !== 'undefined') {
    window.UnityManifoldsVisualizer = UnityManifoldsVisualizer;
    window.createUnityManifoldsTopology = createUnityManifoldsTopology;
}