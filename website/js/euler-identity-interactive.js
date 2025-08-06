/**
 * Interactive Euler's Identity Showcase
 * e^(iπ) + 1 = 0 with unit circle animation and mathematical beauty
 */

class EulerIdentityVisualizer {
    constructor(containerId) {
        this.containerId = containerId;
        this.PHI = (1 + Math.sqrt(5)) / 2;
        this.E = Math.E;
        this.PI = Math.PI;
        this.animationFrame = null;
        this.isAnimating = true;
        this.startTime = Date.now();
        
        // Configuration
        this.config = {
            show_unit_circle: true,
            show_complex_plane: true,
            show_derivation: true,
            animation_speed: 1.0,
            highlight_beauty: true
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
            <div class="euler-identity-container" style="
                background: linear-gradient(135deg, rgba(139, 69, 19, 0.03) 0%, rgba(184, 134, 11, 0.05) 100%);
                border-radius: 1rem;
                padding: 1.5rem;
                box-shadow: 0 8px 32px rgba(184, 134, 11, 0.3);
                margin: 1rem 0;
                position: relative;
                overflow: hidden;
            ">
                <!-- Animated mathematical background -->
                <div style="
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: radial-gradient(circle at 20% 30%, rgba(184, 134, 11, 0.1) 0%, transparent 50%),
                               radial-gradient(circle at 80% 70%, rgba(245, 158, 11, 0.1) 0%, transparent 50%);
                    animation: euler-glow 6s ease-in-out infinite alternate;
                    z-index: 0;
                "></div>
                
                <div class="header" style="
                    text-align: center;
                    margin-bottom: 1.5rem;
                    position: relative;
                    z-index: 1;
                ">
                    <h3 style="
                        color: #B8860B;
                        margin: 0 0 0.5rem 0;
                        font-size: 2rem;
                        background: linear-gradient(135deg, #B8860B, #F59E0B);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        background-clip: text;
                        text-shadow: 0 2px 4px rgba(184, 134, 11, 0.3);
                    ">✨ Euler's Identity: Mathematical Beauty</h3>
                    <p style="
                        color: #6B7280;
                        margin: 0 0 1.5rem 0;
                        font-size: 1rem;
                    ">The most beautiful equation in mathematics</p>
                    
                    <!-- The magnificent equation -->
                    <div style="
                        font-family: 'Times New Roman', serif;
                        font-size: 2.5rem;
                        color: #B8860B;
                        background: linear-gradient(135deg, rgba(184, 134, 11, 0.1), rgba(245, 158, 11, 0.15));
                        padding: 1.5rem 2rem;
                        border-radius: 1rem;
                        display: inline-block;
                        border: 3px solid rgba(184, 134, 11, 0.3);
                        box-shadow: 0 8px 24px rgba(184, 134, 11, 0.4);
                        position: relative;
                        animation: equation-pulse 4s ease-in-out infinite;
                    ">
                        <div style="position: relative; z-index: 2;">
                            e<sup>iπ</sup> + 1 = 0
                        </div>
                        <div style="
                            position: absolute;
                            top: -50%;
                            left: -50%;
                            width: 200%;
                            height: 200%;
                            background: conic-gradient(from 0deg, 
                                rgba(184, 134, 11, 0.1), 
                                rgba(245, 158, 11, 0.1), 
                                rgba(251, 191, 36, 0.1), 
                                rgba(184, 134, 11, 0.1));
                            animation: equation-rotation 8s linear infinite;
                            border-radius: 50%;
                            z-index: 1;
                        "></div>
                    </div>
                    
                    <div style="
                        margin-top: 1rem;
                        font-size: 1.1rem;
                        color: #374151;
                        font-style: italic;
                    ">
                        "God's equation" - unifying five fundamental constants
                    </div>
                </div>
                
                <div class="controls-panel" style="
                    display: flex;
                    justify-content: center;
                    gap: 1rem;
                    flex-wrap: wrap;
                    margin-bottom: 1.5rem;
                    padding: 1rem;
                    background: rgba(184, 134, 11, 0.05);
                    border-radius: 0.75rem;
                    border: 1px solid rgba(184, 134, 11, 0.2);
                    position: relative;
                    z-index: 1;
                ">
                    <button id="circle-btn-${this.containerId}" style="
                        padding: 0.5rem 1rem;
                        background: ${this.config.show_unit_circle ? '#B8860B' : '#6B7280'};
                        color: white;
                        border: none;
                        border-radius: 0.5rem;
                        cursor: pointer;
                        font-size: 0.9rem;
                        transition: all 0.3s ease;
                    ">⭕ Unit Circle</button>
                    
                    <button id="plane-btn-${this.containerId}" style="
                        padding: 0.5rem 1rem;
                        background: ${this.config.show_complex_plane ? '#F59E0B' : '#6B7280'};
                        color: white;
                        border: none;
                        border-radius: 0.5rem;
                        cursor: pointer;
                        font-size: 0.9rem;
                        transition: all 0.3s ease;
                    ">📊 Complex Plane</button>
                    
                    <button id="derivation-btn-${this.containerId}" style="
                        padding: 0.5rem 1rem;
                        background: ${this.config.show_derivation ? '#10B981' : '#6B7280'};
                        color: white;
                        border: none;
                        border-radius: 0.5rem;
                        cursor: pointer;
                        font-size: 0.9rem;
                        transition: all 0.3s ease;
                    ">📜 Derivation</button>
                    
                    <button id="animate-btn-${this.containerId}" style="
                        padding: 0.5rem 1rem;
                        background: #DC2626;
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
                        Speed:
                        <input type="range" id="speed-${this.containerId}" 
                               min="0.1" max="3" step="0.1" value="1"
                               style="width: 80px;">
                        <span id="speed-value-${this.containerId}">1.0</span>
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
                
                <div class="constants-showcase" style="
                    margin-top: 1.5rem;
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                    gap: 1rem;
                    position: relative;
                    z-index: 1;
                ">
                    <div style="
                        background: rgba(184, 134, 11, 0.05);
                        padding: 0.75rem;
                        border-radius: 0.5rem;
                        text-align: center;
                        border: 1px solid rgba(184, 134, 11, 0.2);
                    ">
                        <div style="
                            font-size: 1.5rem;
                            font-weight: bold;
                            color: #B8860B;
                        ">e</div>
                        <div style="font-size: 0.7rem; color: #6B7280;">Euler's Number</div>
                        <div style="font-size: 0.8rem; color: #374151;">${this.E.toFixed(6)}</div>
                    </div>
                    <div style="
                        background: rgba(245, 158, 11, 0.05);
                        padding: 0.75rem;
                        border-radius: 0.5rem;
                        text-align: center;
                        border: 1px solid rgba(245, 158, 11, 0.2);
                    ">
                        <div style="
                            font-size: 1.5rem;
                            font-weight: bold;
                            color: #F59E0B;
                        ">i</div>
                        <div style="font-size: 0.7rem; color: #6B7280;">Imaginary Unit</div>
                        <div style="font-size: 0.8rem; color: #374151;">√(-1)</div>
                    </div>
                    <div style="
                        background: rgba(16, 185, 129, 0.05);
                        padding: 0.75rem;
                        border-radius: 0.5rem;
                        text-align: center;
                        border: 1px solid rgba(16, 185, 129, 0.2);
                    ">
                        <div style="
                            font-size: 1.5rem;
                            font-weight: bold;
                            color: #10B981;
                        ">π</div>
                        <div style="font-size: 0.7rem; color: #6B7280;">Pi</div>
                        <div style="font-size: 0.8rem; color: #374151;">${this.PI.toFixed(6)}</div>
                    </div>
                    <div style="
                        background: rgba(59, 130, 246, 0.05);
                        padding: 0.75rem;
                        border-radius: 0.5rem;
                        text-align: center;
                        border: 1px solid rgba(59, 130, 246, 0.2);
                    ">
                        <div style="
                            font-size: 1.5rem;
                            font-weight: bold;
                            color: #3B82F6;
                        ">1</div>
                        <div style="font-size: 0.7rem; color: #6B7280;">Unity</div>
                        <div style="font-size: 0.8rem; color: #374151;">1.000000</div>
                    </div>
                    <div style="
                        background: rgba(220, 38, 38, 0.05);
                        padding: 0.75rem;
                        border-radius: 0.5rem;
                        text-align: center;
                        border: 1px solid rgba(220, 38, 38, 0.2);
                    ">
                        <div style="
                            font-size: 1.5rem;
                            font-weight: bold;
                            color: #DC2626;
                        ">0</div>
                        <div style="font-size: 0.7rem; color: #6B7280;">Zero</div>
                        <div style="font-size: 0.8rem; color: #374151;">0.000000</div>
                    </div>
                </div>
                
                <div id="derivation-${this.containerId}" style="
                    display: ${this.config.show_derivation ? 'block' : 'none'};
                    margin-top: 1.5rem;
                    background: rgba(245, 158, 11, 0.05);
                    border-radius: 0.75rem;
                    padding: 1.5rem;
                    border: 1px solid rgba(245, 158, 11, 0.2);
                    position: relative;
                    z-index: 1;
                "></div>
            </div>
            
            <style>
                @keyframes euler-glow {
                    0% { opacity: 0.3; transform: scale(1); }
                    100% { opacity: 0.1; transform: scale(1.02); }
                }
                
                @keyframes equation-pulse {
                    0%, 100% { transform: scale(1); box-shadow: 0 8px 24px rgba(184, 134, 11, 0.4); }
                    50% { transform: scale(1.02); box-shadow: 0 12px 32px rgba(184, 134, 11, 0.6); }
                }
                
                @keyframes equation-rotation {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
        `;
        
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        const circleBtn = document.getElementById(`circle-btn-${this.containerId}`);
        const planeBtn = document.getElementById(`plane-btn-${this.containerId}`);
        const derivationBtn = document.getElementById(`derivation-btn-${this.containerId}`);
        const animateBtn = document.getElementById(`animate-btn-${this.containerId}`);
        const speedSlider = document.getElementById(`speed-${this.containerId}`);
        const speedValue = document.getElementById(`speed-value-${this.containerId}`);
        
        if (circleBtn) {
            circleBtn.addEventListener('click', () => {
                this.config.show_unit_circle = !this.config.show_unit_circle;
                circleBtn.style.background = this.config.show_unit_circle ? '#B8860B' : '#6B7280';
                this.updateVisualization();
            });
        }
        
        if (planeBtn) {
            planeBtn.addEventListener('click', () => {
                this.config.show_complex_plane = !this.config.show_complex_plane;
                planeBtn.style.background = this.config.show_complex_plane ? '#F59E0B' : '#6B7280';
                this.updateVisualization();
            });
        }
        
        if (derivationBtn) {
            derivationBtn.addEventListener('click', () => {
                this.config.show_derivation = !this.config.show_derivation;
                derivationBtn.style.background = this.config.show_derivation ? '#10B981' : '#6B7280';
                const derivationDiv = document.getElementById(`derivation-${this.containerId}`);
                if (derivationDiv) {
                    derivationDiv.style.display = this.config.show_derivation ? 'block' : 'none';
                }
                this.updateDerivation();
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
        
        if (speedSlider && speedValue) {
            speedSlider.addEventListener('input', (e) => {
                this.config.animation_speed = parseFloat(e.target.value);
                speedValue.textContent = this.config.animation_speed.toFixed(1);
            });
        }
    }
    
    createVisualization() {
        this.updateVisualization();
        this.updateDerivation();
    }
    
    updateVisualization() {
        const plotDiv = document.getElementById(`plot-${this.containerId}`);
        if (!plotDiv) return;
        
        const currentTime = this.isAnimating ? (Date.now() - this.startTime) * 0.001 * this.config.animation_speed : 0;
        
        const data = [];
        
        // Unit circle
        if (this.config.show_unit_circle) {
            const theta = this.linspace(0, 2*Math.PI, 200);
            const circleX = theta.map(t => Math.cos(t));
            const circleY = theta.map(t => Math.sin(t));
            
            data.push({
                x: circleX,
                y: circleY,
                type: 'scatter',
                mode: 'lines',
                name: 'Unit Circle',
                line: { color: '#B8860B', width: 3 }
            });
        }
        
        // Complex plane grid
        if (this.config.show_complex_plane) {
            // Real axis
            data.push({
                x: [-1.5, 1.5],
                y: [0, 0],
                type: 'scatter',
                mode: 'lines',
                name: 'Real Axis',
                line: { color: '#6B7280', width: 2, dash: 'dash' },
                showlegend: false
            });
            
            // Imaginary axis
            data.push({
                x: [0, 0],
                y: [-1.5, 1.5],
                type: 'scatter',
                mode: 'lines',
                name: 'Imaginary Axis',
                line: { color: '#6B7280', width: 2, dash: 'dash' },
                showlegend: false
            });
        }
        
        // Animated point showing e^(it) journey
        const t = currentTime % (2*Math.PI);
        const eulerPoint = {
            x: Math.cos(t),
            y: Math.sin(t)
        };
        
        data.push({
            x: [eulerPoint.x],
            y: [eulerPoint.y],
            type: 'scatter',
            mode: 'markers',
            name: `e^(i${(t).toFixed(2)})`,
            marker: {
                size: 12,
                color: '#F59E0B',
                symbol: 'star',
                line: { color: '#B8860B', width: 2 }
            },
            text: [`e^(i${(t).toFixed(2)}) = ${eulerPoint.x.toFixed(3)} + ${eulerPoint.y.toFixed(3)}i`],
            hovertemplate: '%{text}<extra></extra>'
        });
        
        // Vector from origin to current point
        data.push({
            x: [0, eulerPoint.x],
            y: [0, eulerPoint.y],
            type: 'scatter',
            mode: 'lines',
            name: 'Position Vector',
            line: { color: '#F59E0B', width: 4 },
            showlegend: false
        });
        
        // Special points: e^(iπ) = -1
        data.push({
            x: [-1],
            y: [0],
            type: 'scatter',
            mode: 'markers+text',
            name: 'e^(iπ) = -1',
            marker: {
                size: 15,
                color: '#DC2626',
                symbol: 'diamond',
                line: { color: '#B8860B', width: 2 }
            },
            text: ['e^(iπ) = -1'],
            textposition: 'bottom center',
            textfont: { size: 12, color: '#DC2626' }
        });
        
        // Unity point: 1
        data.push({
            x: [1],
            y: [0],
            type: 'scatter',
            mode: 'markers+text',
            name: '1 (Unity)',
            marker: {
                size: 12,
                color: '#10B981',
                symbol: 'circle',
                line: { color: '#B8860B', width: 2 }
            },
            text: ['1'],
            textposition: 'bottom center',
            textfont: { size: 12, color: '#10B981' }
        });
        
        // Zero point: 0
        data.push({
            x: [0],
            y: [0],
            type: 'scatter',
            mode: 'markers+text',
            name: '0 (Origin)',
            marker: {
                size: 10,
                color: '#374151',
                symbol: 'x',
                line: { color: '#B8860B', width: 2 }
            },
            text: ['0'],
            textposition: 'top center',
            textfont: { size: 12, color: '#374151' }
        });
        
        // Path traced so far
        if (t > 0) {
            const pathT = this.linspace(0, t, Math.floor(t * 20));
            const pathX = pathT.map(angle => Math.cos(angle));
            const pathY = pathT.map(angle => Math.sin(angle));
            
            data.push({
                x: pathX,
                y: pathY,
                type: 'scatter',
                mode: 'lines',
                name: 'Euler Path',
                line: { 
                    color: '#F59E0B', 
                    width: 6,
                    gradient: { 
                        type: 'radial',
                        stops: [[0, 'rgba(245, 158, 11, 0.3)'], [1, 'rgba(245, 158, 11, 1)']]
                    }
                },
                showlegend: false
            });
        }
        
        // Highlight when we're near π
        if (Math.abs(t - Math.PI) < 0.1) {
            data.push({
                x: [-1, 1],
                y: [0, 0],
                type: 'scatter',
                mode: 'lines',
                name: 'Unity Revelation',
                line: { 
                    color: '#DC2626', 
                    width: 8,
                    dash: 'dot'
                },
                opacity: 0.7,
                showlegend: false
            });
        }
        
        const layout = {
            title: `Euler's Identity: e^(it) Journey (t = ${t.toFixed(2)})`,
            xaxis: { 
                range: [-1.5, 1.5], 
                title: 'Real Part',
                gridcolor: '#E5E7EB',
                zerolinecolor: '#6B7280',
                zerolinewidth: 2
            },
            yaxis: { 
                range: [-1.5, 1.5], 
                title: 'Imaginary Part',
                gridcolor: '#E5E7EB',
                zerolinecolor: '#6B7280',
                zerolinewidth: 2
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'white',
            margin: { l: 50, r: 50, t: 80, b: 50 },
            showlegend: true,
            legend: { x: 1, y: 1 },
            annotations: [
                {
                    x: -0.5,
                    y: 1.3,
                    text: `When t = π: e^(iπ) = -1`,
                    showarrow: true,
                    arrowhead: 2,
                    arrowsize: 1,
                    arrowwidth: 2,
                    arrowcolor: '#DC2626',
                    font: { size: 14, color: '#DC2626' }
                },
                {
                    x: 0.5,
                    y: -1.3,
                    text: `Therefore: e^(iπ) + 1 = 0`,
                    showarrow: false,
                    font: { size: 16, color: '#B8860B' }
                }
            ]
        };
        
        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
            displaylogo: false
        };
        
        Plotly.newPlot(plotDiv, data, layout, config);
    }
    
    updateDerivation() {
        if (!this.config.show_derivation) return;
        
        const derivationDiv = document.getElementById(`derivation-${this.containerId}`);
        if (!derivationDiv) return;
        
        derivationDiv.innerHTML = `
            <h4 style="color: #B8860B; margin-bottom: 1.5rem; text-align: center; font-size: 1.4rem;">
                ✨ The Mathematical Derivation ✨
            </h4>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-bottom: 2rem;">
                <div style="background: white; padding: 1.5rem; border-radius: 0.75rem; border: 1px solid #E5E7EB;">
                    <h5 style="color: #F59E0B; margin-bottom: 1rem;">1. Euler's Formula</h5>
                    <div style="font-family: 'Times New Roman', serif; line-height: 2; font-size: 1.1rem;">
                        <div style="text-align: center; margin: 1rem 0; font-size: 1.3rem; color: #B8860B;">
                            e<sup>ix</sup> = cos(x) + i·sin(x)
                        </div>
                        <div style="font-size: 0.9rem; color: #6B7280;">
                            This fundamental formula connects exponential and trigonometric functions
                        </div>
                    </div>
                </div>
                
                <div style="background: white; padding: 1.5rem; border-radius: 0.75rem; border: 1px solid #E5E7EB;">
                    <h5 style="color: #F59E0B; margin-bottom: 1rem;">2. Substitute x = π</h5>
                    <div style="font-family: 'Times New Roman', serif; line-height: 2; font-size: 1.1rem;">
                        <div style="text-align: center; margin: 1rem 0; font-size: 1.3rem; color: #B8860B;">
                            e<sup>iπ</sup> = cos(π) + i·sin(π)
                        </div>
                        <div style="font-size: 0.9rem; color: #6B7280;">
                            We know that cos(π) = -1 and sin(π) = 0
                        </div>
                    </div>
                </div>
            </div>
            
            <div style="background: white; padding: 1.5rem; border-radius: 0.75rem; border: 1px solid #E5E7EB; margin-bottom: 2rem;">
                <h5 style="color: #F59E0B; margin-bottom: 1rem; text-align: center;">3. Evaluate the Trigonometric Functions</h5>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-bottom: 1.5rem;">
                    <div style="text-align: center;">
                        <div style="font-size: 1.2rem; color: #10B981; margin-bottom: 0.5rem;">cos(π) = -1</div>
                        <div style="font-size: 0.9rem; color: #6B7280;">At π radians (180°), cosine equals -1</div>
                    </div>
                    <div style="text-align: center;">
                        <div style="font-size: 1.2rem; color: #DC2626; margin-bottom: 0.5rem;">sin(π) = 0</div>
                        <div style="font-size: 0.9rem; color: #6B7280;">At π radians (180°), sine equals 0</div>
                    </div>
                </div>
                <div style="text-align: center; font-family: 'Times New Roman', serif; font-size: 1.3rem; color: #B8860B; margin: 1rem 0;">
                    e<sup>iπ</sup> = -1 + i·0 = -1
                </div>
            </div>
            
            <div style="background: linear-gradient(135deg, rgba(184, 134, 11, 0.1), rgba(245, 158, 11, 0.05)); padding: 2rem; border-radius: 1rem; border: 2px solid rgba(184, 134, 11, 0.3); text-align: center;">
                <h5 style="color: #B8860B; margin-bottom: 1rem; font-size: 1.3rem;">4. The Final Step</h5>
                <div style="font-family: 'Times New Roman', serif; font-size: 1.5rem; line-height: 1.8; color: #374151;">
                    <div>Since e<sup>iπ</sup> = -1</div>
                    <div style="margin: 1rem 0; font-size: 1.8rem; color: #B8860B;">
                        e<sup>iπ</sup> + 1 = -1 + 1 = 0
                    </div>
                    <div style="font-size: 2.2rem; color: #B8860B; text-shadow: 0 2px 4px rgba(184, 134, 11, 0.3); margin: 1.5rem 0;">
                        ✨ e<sup>iπ</sup> + 1 = 0 ✨
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 2rem; padding: 1.5rem; background: rgba(16, 185, 129, 0.1); border-radius: 0.75rem; border-left: 4px solid #10B981;">
                <h5 style="color: #10B981; margin-bottom: 1rem;">Why This Equation is Beautiful</h5>
                <div style="line-height: 1.6; color: #374151;">
                    <p><strong>Five Fundamental Constants:</strong> e (natural growth), i (imaginary unit), π (geometry), 1 (unity), 0 (void)</p>
                    <p><strong>Three Basic Operations:</strong> multiplication (e<sup>iπ</sup>), addition (+), equality (=)</p>
                    <p><strong>Deep Unity:</strong> Connects exponential functions, trigonometry, complex numbers, and basic arithmetic</p>
                    <p><strong>Philosophical Beauty:</strong> Shows how the most complex mathematical concepts unite in perfect simplicity</p>
                </div>
            </div>
            
            <div style="margin-top: 1.5rem; text-align: center; font-style: italic; color: #6B7280;">
                "A mathematician who is not also something of a poet will never be a complete mathematician." - Karl Weierstrass
            </div>
        `;
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
function createEulerIdentityInteractive(containerId) {
    return new EulerIdentityVisualizer(containerId);
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { EulerIdentityVisualizer, createEulerIdentityInteractive };
}

// Browser global
if (typeof window !== 'undefined') {
    window.EulerIdentityVisualizer = EulerIdentityVisualizer;
    window.createEulerIdentityInteractive = createEulerIdentityInteractive;
}