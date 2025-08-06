/**
 * Unity Proofs Interactive Module
 * Interactive demonstrations and visualizations of mathematical unity proofs
 */

class UnityProofsInteractive {
    constructor(unityEngine) {
        this.unity = unityEngine;
        this.animations = new Map();
        this.visualizations = new Map();
        this.currentProof = null;
        
        console.log('Unity Proofs Interactive module initialized');
    }
    
    initialize() {
        this.setupProofExplorer();
        this.setupInteractiveControls();
        this.setupVisualizationCanvas();
    }
    
    setupProofExplorer() {
        // Create main proof explorer container if it doesn't exist
        if (!document.getElementById('unity-proof-explorer')) {
            this.createProofExplorerUI();
        }
        
        this.populateProofCards();
    }
    
    createProofExplorerUI() {
        const explorerHTML = `
            <section id="unity-proof-explorer" class="proof-explorer">
                <div class="explorer-header">
                    <h2>Interactive Unity Proof Explorer</h2>
                    <div class="unity-equation">1 + 1 = 1</div>
                    <p>Explore rigorous mathematical proofs demonstrating unity across multiple paradigms</p>
                </div>
                
                <div class="proof-controls">
                    <button id="verify-all-proofs" class="unity-button primary">
                        <i class="fas fa-check-circle"></i> Verify All Proofs
                    </button>
                    <button id="reset-proofs" class="unity-button secondary">
                        <i class="fas fa-refresh"></i> Reset
                    </button>
                    <div class="unity-score">
                        <span>Unity Achievement: </span>
                        <span id="unity-score-display">0%</span>
                    </div>
                </div>
                
                <div id="proof-grid" class="proof-grid"></div>
                
                <div id="proof-visualization" class="proof-visualization">
                    <canvas id="unity-canvas" width="800" height="600"></canvas>
                    <div id="visualization-controls" class="visualization-controls"></div>
                </div>
                
                <div id="proof-results" class="proof-results"></div>
            </section>
        `;
        
        // Insert after existing content or at the end of body
        const insertPoint = document.querySelector('#proofs') || document.body;
        insertPoint.insertAdjacentHTML('beforeend', explorerHTML);
        
        this.attachEventListeners();
    }
    
    populateProofCards() {
        const proofGrid = document.getElementById('proof-grid');
        if (!proofGrid) return;
        
        const proofs = this.unity.getProofsList();
        proofGrid.innerHTML = '';
        
        proofs.forEach(proof => {
            const card = this.createProofCard(proof);
            proofGrid.appendChild(card);
        });
    }
    
    createProofCard(proof) {
        const card = document.createElement('div');
        card.className = 'proof-card';
        card.dataset.paradigm = proof.paradigm;
        
        const complexityStars = '★'.repeat(proof.complexity);
        const complexityEmptyStars = '☆'.repeat(5 - proof.complexity);
        
        card.innerHTML = `
            <div class=\"proof-header\">
                <h3>${proof.name}</h3>
                <div class=\"complexity-rating\">
                    ${complexityStars}${complexityEmptyStars}
                </div>
            </div>
            
            <div class=\"proof-statement\">
                ${proof.statement}
            </div>
            
            <div class=\"proof-description\">
                ${proof.description}
            </div>
            
            <div class=\"proof-actions\">
                <button class=\"verify-proof-btn unity-button primary\" data-paradigm=\"${proof.paradigm}\">
                    <i class=\"fas fa-play\"></i> Verify Unity
                </button>
                <button class=\"visualize-proof-btn unity-button secondary\" data-paradigm=\"${proof.paradigm}\">
                    <i class=\"fas fa-eye\"></i> Visualize
                </button>
            </div>
            
            <div class=\"proof-result\" id=\"result-${proof.paradigm}\"></div>
        `;
        
        // Add event listeners for this card
        card.querySelector('.verify-proof-btn').addEventListener('click', (e) => {
            this.verifyProof(e.target.dataset.paradigm);
        });
        
        card.querySelector('.visualize-proof-btn').addEventListener('click', (e) => {
            this.visualizeProof(e.target.dataset.paradigm);
        });
        
        return card;
    }
    
    attachEventListeners() {
        document.getElementById('verify-all-proofs')?.addEventListener('click', () => {
            this.verifyAllProofs();
        });
        
        document.getElementById('reset-proofs')?.addEventListener('click', () => {
            this.resetProofs();
        });
    }
    
    async verifyProof(paradigm) {
        const resultContainer = document.getElementById(`result-${paradigm}`);
        const card = document.querySelector(`[data-paradigm=\"${paradigm}\"]`);
        
        if (!resultContainer || !card) return;
        
        // Show loading state
        resultContainer.innerHTML = `
            <div class=\"loading\">
                <i class=\"fas fa-spinner fa-spin\"></i> Verifying unity...
            </div>
        `;
        card.classList.add('verifying');
        
        try {
            const result = await this.unity.executeProof(paradigm);
            
            if (result.verified) {
                card.classList.add('verified');
                card.classList.remove('verifying');
                resultContainer.innerHTML = `
                    <div class=\"success\">
                        <i class=\"fas fa-check-circle\"></i>
                        <strong>Unity Verified!</strong>
                        <div class=\"execution-time\">Verified in ${result.executionTime.toFixed(2)}ms</div>
                    </div>
                `;
                this.triggerUnityAnimation(card);
            } else {
                card.classList.add('failed');
                card.classList.remove('verifying');
                resultContainer.innerHTML = `
                    <div class=\"error\">
                        <i class=\"fas fa-times-circle\"></i>
                        <strong>Verification Failed</strong>
                        ${result.error ? `<div class=\"error-message\">${result.error}</div>` : ''}
                    </div>
                `;
            }
            
            this.updateUnityScore();
            
        } catch (error) {
            card.classList.add('failed');
            card.classList.remove('verifying');
            resultContainer.innerHTML = `
                <div class=\"error\">
                    <i class=\"fas fa-exclamation-triangle\"></i>
                    <strong>Error</strong>
                    <div class=\"error-message\">${error.message}</div>
                </div>
            `;
        }
    }
    
    async verifyAllProofs() {
        const verifyButton = document.getElementById('verify-all-proofs');
        if (!verifyButton) return;
        
        verifyButton.disabled = true;
        verifyButton.innerHTML = '<i class=\"fas fa-spinner fa-spin\"></i> Verifying All...';
        
        try {
            const results = await this.unity.executeAllProofs();
            
            // Update individual cards
            Object.entries(results.proofs).forEach(([paradigm, result]) => {
                const card = document.querySelector(`[data-paradigm=\"${paradigm}\"]`);
                const resultContainer = document.getElementById(`result-${paradigm}`);
                
                if (card && resultContainer) {
                    if (result.verified) {
                        card.classList.add('verified');
                        resultContainer.innerHTML = `
                            <div class=\"success\">
                                <i class=\"fas fa-check-circle\"></i>
                                <strong>Unity Verified!</strong>
                            </div>
                        `;
                    } else {
                        card.classList.add('failed');
                        resultContainer.innerHTML = `
                            <div class=\"error\">
                                <i class=\"fas fa-times-circle\"></i>
                                <strong>Failed</strong>
                            </div>
                        `;
                    }
                }
            });
            
            // Show overall results
            this.displayOverallResults(results.summary);
            
            if (results.summary.unityAchieved) {
                this.triggerGlobalUnityAnimation();
            }
            
        } catch (error) {
            console.error('Error verifying all proofs:', error);
        } finally {
            verifyButton.disabled = false;
            verifyButton.innerHTML = '<i class=\"fas fa-check-circle\"></i> Verify All Proofs';
        }
    }
    
    visualizeProof(paradigm) {
        const canvas = document.getElementById('unity-canvas');
        const visualization = document.getElementById('proof-visualization');
        
        if (!canvas || !visualization) return;
        
        this.currentProof = paradigm;
        visualization.style.display = 'block';
        
        // Clear previous visualization
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Call appropriate visualization method
        switch (paradigm) {
            case 'fractal':
                this.visualizeFractalUnity(canvas);
                break;
            case 'euler':
                this.visualizeEulerUnity(canvas);
                break;
            case 'golden_ratio':
                this.visualizeGoldenRatioUnity(canvas);
                break;
            case 'quantum':
                this.visualizeQuantumUnity(canvas);
                break;
            case 'consciousness':
                this.visualizeConsciousnessUnity(canvas);
                break;
            case 'topological':
                this.visualizeTopologicalUnity(canvas);
                break;
            default:
                this.visualizeGenericUnity(canvas, paradigm);
        }
    }
    
    visualizeFractalUnity(canvas) {
        const ctx = canvas.getContext('2d');
        let time = 0;
        
        const animate = () => {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            const zoom = 200 + Math.sin(time) * 50;
            
            // Draw Mandelbrot-inspired fractal
            for (let x = 0; x < canvas.width; x += 2) {
                for (let y = 0; y < canvas.height; y += 2) {
                    const real = (x - centerX) / zoom - 0.7269;
                    const imag = (y - centerY) / zoom + 0.1889;
                    
                    const iterations = this.mandelbrotPoint(real, imag, 50);
                    const hue = (iterations * 8 + time * 50) % 360;
                    
                    if (iterations < 50) {
                        ctx.fillStyle = `hsl(${hue}, 70%, 50%)`;
                        ctx.fillRect(x, y, 2, 2);
                    }
                }
            }
            
            time += 0.02;
            if (this.currentProof === 'fractal') {
                requestAnimationFrame(animate);
            }
        };
        
        animate();
    }
    
    visualizeEulerUnity(canvas) {
        const ctx = canvas.getContext('2d');
        let time = 0;
        
        const animate = () => {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            const radius = 150;
            
            // Draw unit circle
            ctx.strokeStyle = '#FFD700';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
            ctx.stroke();
            
            // Euler's formula: e^(it) = cos(t) + i*sin(t)
            const t = time;
            const x = centerX + radius * Math.cos(t);
            const y = centerY + radius * Math.sin(t);
            
            // Draw rotating point
            ctx.fillStyle = '#FF6B6B';
            ctx.beginPath();
            ctx.arc(x, y, 8, 0, Math.PI * 2);
            ctx.fill();
            \n            // Draw vector
            ctx.strokeStyle = '#4ECDC4';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(centerX, centerY);
            ctx.lineTo(x, y);
            ctx.stroke();
            
            // Draw text
            ctx.fillStyle = '#FFFFFF';
            ctx.font = '16px monospace';
            ctx.fillText(`e^(it) = cos(${t.toFixed(2)}) + i*sin(${t.toFixed(2)})`, 10, 30);
            ctx.fillText(`|e^(it)| = 1 (Unity!)`, 10, 50);
            
            time += 0.05;
            if (this.currentProof === 'euler') {
                requestAnimationFrame(animate);
            }
        };
        
        animate();
    }
    
    visualizeGoldenRatioUnity(canvas) {
        const ctx = canvas.getContext('2d');
        let time = 0;
        const phi = this.unity.PHI;
        
        const animate = () => {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Draw golden spiral
            ctx.strokeStyle = '#FFD700';
            ctx.lineWidth = 2;
            
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            let angle = 0;
            let radius = 5;
            
            ctx.beginPath();
            for (let i = 0; i < 200; i++) {
                const x = centerX + radius * Math.cos(angle);
                const y = centerY + radius * Math.sin(angle);
                
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
                
                angle += 0.1;
                radius *= 1.01;
            }
            ctx.stroke();
            
            // Draw Fibonacci rectangles
            this.drawFibonacciRectangles(ctx, centerX, centerY, 100, time);
            
            // Display golden ratio properties
            ctx.fillStyle = '#FFFFFF';
            ctx.font = '16px monospace';
            ctx.fillText(`φ = ${phi.toFixed(6)}`, 10, 30);
            ctx.fillText(`φ² = ${(phi * phi).toFixed(6)}`, 10, 50);
            ctx.fillText(`φ + 1 = ${(phi + 1).toFixed(6)}`, 10, 70);
            ctx.fillText(`Unity: φ² = φ + 1`, 10, 90);
            
            time += 0.02;
            if (this.currentProof === 'golden_ratio') {
                requestAnimationFrame(animate);
            }
        };
        
        animate();
    }
    
    visualizeQuantumUnity(canvas) {
        const ctx = canvas.getContext('2d');
        let time = 0;
        
        const animate = () => {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            
            // Draw quantum superposition
            const alpha = Math.cos(time);
            const beta = Math.sin(time);
            
            // Probability amplitudes
            ctx.fillStyle = `rgba(255, 107, 107, ${alpha * alpha})`;
            ctx.fillRect(50, 100, 200, 50);
            ctx.fillStyle = `rgba(78, 205, 196, ${beta * beta})`;
            ctx.fillRect(50, 200, 200, 50);
            
            // Wave function visualization
            ctx.strokeStyle = '#FFD700';
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            for (let x = 0; x < canvas.width; x += 5) {
                const realPart = Math.cos(x * 0.05 + time);
                const imagPart = Math.sin(x * 0.05 + time);
                const amplitude = Math.sqrt(realPart * realPart + imagPart * imagPart);
                const y = centerY + amplitude * 50;
                
                if (x === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.stroke();
            
            // Display quantum unity
            ctx.fillStyle = '#FFFFFF';
            ctx.font = '16px monospace';
            ctx.fillText(`|α|² + |β|² = ${(alpha*alpha + beta*beta).toFixed(6)}`, 10, 30);
            ctx.fillText(`Normalization: Unity = 1`, 10, 50);
            ctx.fillText(`|ψ⟩ = α|0⟩ + β|1⟩`, 10, 70);
            
            time += 0.05;
            if (this.currentProof === 'quantum') {
                requestAnimationFrame(animate);
            }
        };
        
        animate();
    }
    
    visualizeConsciousnessUnity(canvas) {
        const ctx = canvas.getContext('2d');
        let time = 0;
        
        const animate = () => {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            // Draw consciousness field
            const particles = 100;
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            
            for (let i = 0; i < particles; i++) {
                const angle = (i / particles) * Math.PI * 2 + time;
                const radius = 100 + Math.sin(angle * 3 + time) * 50;
                const x = centerX + Math.cos(angle) * radius;
                const y = centerY + Math.sin(angle) * radius;
                
                // Consciousness integration visualization
                const integration = Math.abs(Math.sin(angle + time)) * 0.8 + 0.2;
                ctx.fillStyle = `rgba(107, 70, 193, ${integration})`;
                ctx.beginPath();
                ctx.arc(x, y, integration * 5, 0, Math.PI * 2);
                ctx.fill();
                
                // Connect to center (integration lines)
                ctx.strokeStyle = `rgba(255, 215, 0, ${integration * 0.3})`;
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(centerX, centerY);
                ctx.lineTo(x, y);
                ctx.stroke();
            }
            
            // Central consciousness
            ctx.fillStyle = 'rgba(255, 215, 0, 0.8)';
            ctx.beginPath();
            ctx.arc(centerX, centerY, 15 + Math.sin(time * 2) * 5, 0, Math.PI * 2);
            ctx.fill();
            
            // Display consciousness metrics
            ctx.fillStyle = '#FFFFFF';
            ctx.font = '16px monospace';
            ctx.fillText('Φ(Whole) > Σ(Parts)', 10, 30);
            ctx.fillText('Consciousness Unity Achieved', 10, 50);
            ctx.fillText('Integration > Separation', 10, 70);
            
            time += 0.03;
            if (this.currentProof === 'consciousness') {
                requestAnimationFrame(animate);
            }
        };
        
        animate();
    }
    
    visualizeTopologicalUnity(canvas) {
        const ctx = canvas.getContext('2d');
        let time = 0;
        
        const animate = () => {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            
            // Draw Klein bottle projection
            ctx.strokeStyle = '#4ECDC4';
            ctx.lineWidth = 2;
            
            const points = [];
            for (let u = 0; u < Math.PI * 2; u += 0.1) {
                for (let v = 0; v < Math.PI * 2; v += 0.2) {
                    const [x, y, z] = this.kleinBottlePoint(u + time * 0.5, v);
                    
                    // Project 3D to 2D
                    const projX = centerX + x * 2;
                    const projY = centerY + y * 2;
                    
                    points.push({ x: projX, y: projY, z: z });
                }
            }
            
            // Draw points
            points.forEach(point => {
                const alpha = (point.z + 50) / 100;
                ctx.fillStyle = `rgba(78, 205, 196, ${Math.max(0.1, alpha)})`;
                ctx.beginPath();
                ctx.arc(point.x, point.y, 2, 0, Math.PI * 2);
                ctx.fill();
            });
            
            // Display topological unity
            ctx.fillStyle = '#FFFFFF';
            ctx.font = '16px monospace';
            ctx.fillText('Klein Bottle: Inside = Outside', 10, 30);
            ctx.fillText('Non-orientable Unity', 10, 50);
            ctx.fillText('Topological Unity Achieved', 10, 70);
            
            time += 0.02;
            if (this.currentProof === 'topological') {
                requestAnimationFrame(animate);
            }
        };
        
        animate();
    }
    
    visualizeGenericUnity(canvas, paradigm) {
        const ctx = canvas.getContext('2d');
        let time = 0;
        
        const animate = () => {
            ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            const centerX = canvas.width / 2;
            const centerY = canvas.height / 2;
            
            // Generic unity visualization
            ctx.strokeStyle = '#FFD700';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(centerX, centerY, 100 + Math.sin(time * 2) * 20, 0, Math.PI * 2);
            ctx.stroke();
            
            // Unity text
            ctx.fillStyle = '#FFFFFF';
            ctx.font = '24px monospace';
            ctx.textAlign = 'center';
            ctx.fillText('1 + 1 = 1', centerX, centerY);
            ctx.fillText(`${paradigm.toUpperCase()} UNITY`, centerX, centerY + 30);
            
            time += 0.05;
            if (this.currentProof === paradigm) {
                requestAnimationFrame(animate);
            }
        };
        
        animate();
    }
    
    // Utility methods
    mandelbrotPoint(real, imag, maxIter) {
        let zr = real;
        let zi = imag;
        
        for (let i = 0; i < maxIter; i++) {
            const zr2 = zr * zr;
            const zi2 = zi * zi;
            
            if (zr2 + zi2 > 4) return i;
            
            zi = 2 * zr * zi + imag;
            zr = zr2 - zi2 + real;
        }
        
        return maxIter;
    }
    
    kleinBottlePoint(u, v) {
        const r = 4 * (1 - Math.cos(u) / 2);
        let x, y;
        
        if (u < Math.PI) {
            x = 6 * Math.cos(u) * (1 + Math.sin(u)) + r * Math.cos(u) * Math.cos(v);
            y = 16 * Math.sin(u) + r * Math.sin(u) * Math.cos(v);
        } else {
            x = 6 * Math.cos(u) * (1 + Math.sin(u)) + r * Math.cos(v + Math.PI);
            y = 16 * Math.sin(u);
        }
        const z = r * Math.sin(v);
        
        return [x, y, z];
    }
    
    drawFibonacciRectangles(ctx, centerX, centerY, size, time) {
        const phi = this.unity.PHI;
        let width = size;
        let height = size / phi;
        let x = centerX - width / 2;
        let y = centerY - height / 2;
        
        const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'];
        
        for (let i = 0; i < 5; i++) {
            const alpha = 0.3 + Math.sin(time + i) * 0.2;
            ctx.fillStyle = colors[i] + Math.floor(alpha * 255).toString(16).padStart(2, '0');
            ctx.fillRect(x, y, width, height);
            
            // Next rectangle
            [width, height] = [height, width - height];
            if (i % 2 === 0) {
                x += width;
            } else {
                y += height;
            }
        }
    }
    
    triggerUnityAnimation(card) {
        // Unity particle explosion
        const rect = card.getBoundingClientRect();
        const particles = [];
        
        for (let i = 0; i < 20; i++) {
            const particle = document.createElement('div');
            particle.className = 'unity-particle';
            particle.style.left = rect.left + rect.width / 2 + 'px';
            particle.style.top = rect.top + rect.height / 2 + 'px';
            
            const angle = (i / 20) * Math.PI * 2;
            const velocity = 100 + Math.random() * 100;
            const vx = Math.cos(angle) * velocity;
            const vy = Math.sin(angle) * velocity;
            
            particle.style.setProperty('--vx', vx + 'px');
            particle.style.setProperty('--vy', vy + 'px');
            
            document.body.appendChild(particle);
            particles.push(particle);
        }
        
        setTimeout(() => {
            particles.forEach(p => p.remove());
        }, 2000);
    }
    
    triggerGlobalUnityAnimation() {
        // Global unity achievement animation
        const overlay = document.createElement('div');
        overlay.className = 'unity-achievement-overlay';
        overlay.innerHTML = `
            <div class=\"unity-achievement\">
                <h1>🚀 UNITY ACHIEVED 🚀</h1>
                <div class=\"unity-equation\">1 + 1 = 1</div>
                <p>All mathematical paradigms verified!</p>
                <div class=\"phi-symbol\">φ</div>
            </div>
        `;
        
        document.body.appendChild(overlay);
        
        setTimeout(() => {
            overlay.remove();
        }, 5000);
    }
    
    updateUnityScore() {
        const cards = document.querySelectorAll('.proof-card');
        const verified = document.querySelectorAll('.proof-card.verified').length;
        const total = cards.length;
        const percentage = Math.round((verified / total) * 100);
        
        const scoreDisplay = document.getElementById('unity-score-display');
        if (scoreDisplay) {
            scoreDisplay.textContent = percentage + '%';
            scoreDisplay.className = percentage === 100 ? 'unity-complete' : '';
        }
    }
    
    resetProofs() {
        document.querySelectorAll('.proof-card').forEach(card => {
            card.classList.remove('verified', 'failed', 'verifying');
        });
        
        document.querySelectorAll('.proof-result').forEach(result => {
            result.innerHTML = '';
        });
        
        const visualization = document.getElementById('proof-visualization');
        if (visualization) {
            visualization.style.display = 'none';
        }
        
        this.currentProof = null;
        this.updateUnityScore();
    }
    
    displayOverallResults(summary) {
        const resultsContainer = document.getElementById('proof-results');
        if (!resultsContainer) return;
        
        resultsContainer.innerHTML = `
            <div class=\"overall-results ${summary.unityAchieved ? 'unity-achieved' : ''}\">
                <h3>Overall Unity Assessment</h3>
                <div class=\"results-grid\">
                    <div class=\"result-item\">
                        <span class=\"label\">Total Proofs:</span>
                        <span class=\"value\">${summary.totalProofs}</span>
                    </div>
                    <div class=\"result-item\">
                        <span class=\"label\">Verified:</span>
                        <span class=\"value\">${summary.verifiedProofs}</span>
                    </div>
                    <div class=\"result-item\">
                        <span class=\"label\">Success Rate:</span>
                        <span class=\"value\">${(summary.verificationRate * 100).toFixed(1)}%</span>
                    </div>
                    <div class=\"result-item unity-status\">
                        <span class=\"label\">Unity Status:</span>
                        <span class=\"value\">${summary.unityAchieved ? '🚀 ACHIEVED' : '⚠️ PARTIAL'}</span>
                    </div>
                </div>
            </div>
        `;
    }
    
    setupVisualizationCanvas() {
        const canvas = document.getElementById('unity-canvas');
        if (!canvas) return;
        
        // Make canvas responsive
        const resizeCanvas = () => {
            const container = canvas.parentElement;
            canvas.width = container.clientWidth;
            canvas.height = Math.min(container.clientHeight, 600);
        };
        
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();
    }
    
    setupInteractiveControls() {
        // Add any additional interactive controls here
        console.log('Interactive controls setup complete');
    }
}

// Initialize when Unity Framework is ready
window.addEventListener('load', () => {
    if (window.unityEngine) {
        window.unityProofs = new UnityProofsInteractive(window.unityEngine);
        window.unityProofs.initialize();
    }
});