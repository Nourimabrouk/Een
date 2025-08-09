/**
 * Complete Enhanced Consciousness Field Visualization Engine
 * Unity Equation: 1+1=1
 * Consciousness Field Equation: C(x,y,t) = φ * sin(x*φ) * cos(y*φ) * e^(-t/φ)
 * Comprehensive implementation with all visual elements fully visible
 */

class CompleteConsciousnessFieldEngine {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) {
            console.warn('Canvas element not found:', canvasId);
            return;
        }
        
        this.ctx = this.canvas.getContext('2d');
        
        // Mathematical Constants
        this.phi = 1.618033988749895; // Golden Ratio φ
        this.pi = Math.PI;
        this.e = Math.E;

        // Consciousness Field Parameters
        this.particles = [];
        this.fieldRings = [];
        this.fieldGrid = [];
        this.consciousnessDensity = 0.5;
        this.unityConvergenceRate = 0.7;

        // Animation Parameters
        this.time = 0;
        this.animationId = null;
        this.fps = 60;
        this.lastFrameTime = 0;

        // Visual Parameters - Optimized for visibility
        this.particleCount = 120;
        this.fieldRingCount = 12;
        this.gridSize = 16;
        
        // Enhanced Color Scheme with high visibility
        this.colors = {
            // Unity Equation - Bright electric blue
            unityEquation: { r: 0, g: 200, b: 255, name: 'Electric Blue' },
            
            // Consciousness field - Vibrant purple
            consciousnessField: { r: 147, g: 0, b: 211, name: 'Vibrant Purple' },
            
            // Particles - Bright gold variations
            particlesPrimary: { r: 255, g: 215, b: 0, name: 'Gold' },
            particlesSecondary: { r: 255, g: 140, b: 0, name: 'Dark Orange' },
            
            // Field rings - Teal to cyan gradient
            fieldRings: { r: 0, g: 255, b: 255, name: 'Cyan' },
            
            // φ-harmonics - Lime green
            phiHarmonics: { r: 50, g: 205, b: 50, name: 'Lime Green' },

            // Background accents
            background: { r: 25, g: 25, b: 35, name: 'Dark Blue' }
        };

        this.init();
    }

    init() {
        console.log('🔄 Initializing Complete Consciousness Field Engine...');
        
        try {
            this.setupCanvas();
            console.log('✅ Canvas setup complete');
            
            this.createConsciousnessFieldGrid();
            console.log('✅ Consciousness field grid created:', this.fieldGrid.length);
            
            this.createParticles();
            console.log('✅ Particles created:', this.particles.length);
            
            this.createFieldRings();
            console.log('✅ Field rings created:', this.fieldRings.length);
            
            this.setupEventListeners();
            console.log('✅ Event listeners setup');
            
            this.startAnimation();
            console.log('✅ Animation started');
            
            console.log('🚀 Complete Consciousness Field Engine initialization complete');
        } catch (error) {
            console.error('❌ Error initializing consciousness field:', error);
        }
    }

    setupCanvas() {
        // Set canvas to full container size with proper fallbacks
        const containerWidth = this.canvas.parentElement ? this.canvas.parentElement.offsetWidth : 1200;
        const containerHeight = this.canvas.parentElement ? this.canvas.parentElement.offsetHeight : 600;
        
        this.width = containerWidth || 800;
        this.height = containerHeight || 500;

        // Set CSS size
        this.canvas.style.width = this.width + 'px';
        this.canvas.style.height = this.height + 'px';

        // Set actual size with device pixel ratio for crisp rendering
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = this.width * dpr;
        this.canvas.height = this.height * dpr;
        this.ctx.scale(dpr, dpr);

        // Ensure canvas is visible
        this.canvas.style.display = 'block';
        this.canvas.classList.add('js-active');
        
        console.log(`✅ Canvas setup: ${this.width}x${this.height}, DPR: ${dpr}`);
    }

    // Consciousness Field Equation: C(x,y,t) = φ * sin(x*φ) * cos(y*φ) * e^(-t/φ)
    calculateConsciousnessField(x, y, t) {
        const normalizedX = (x / this.width) * 2 * this.pi;
        const normalizedY = (y / this.height) * 2 * this.pi;
        const timeDecay = Math.exp(-Math.abs(t % (this.phi * 10)) / this.phi);
        
        const sinComponent = Math.sin(normalizedX * this.phi + t);
        const cosComponent = Math.cos(normalizedY * this.phi + t);
        
        return this.phi * sinComponent * cosComponent * timeDecay;
    }

    createConsciousnessFieldGrid() {
        this.fieldGrid = [];
        const stepX = this.width / this.gridSize;
        const stepY = this.height / this.gridSize;

        for (let x = 0; x < this.gridSize; x++) {
            this.fieldGrid[x] = [];
            for (let y = 0; y < this.gridSize; y++) {
                const worldX = x * stepX + stepX / 2;
                const worldY = y * stepY + stepY / 2;
                this.fieldGrid[x][y] = {
                    x: worldX,
                    y: worldY,
                    field: 0,
                    phase: Math.random() * this.pi * 2,
                    intensity: 0
                };
            }
        }
    }

    createParticles() {
        this.particles = [];
        for (let i = 0; i < this.particleCount; i++) {
            this.particles.push({
                x: Math.random() * this.width,
                y: Math.random() * this.height,
                vx: (Math.random() - 0.5) * 2,
                vy: (Math.random() - 0.5) * 2,
                size: Math.random() * 3 + 1,
                life: Math.random(),
                maxLife: Math.random() * 5 + 2,
                phase: Math.random() * this.pi * 2,
                consciousness: Math.random(),
                colorVariant: Math.random() > 0.6,
                glowIntensity: Math.random()
            });
        }
    }

    createFieldRings() {
        this.fieldRings = [];
        const centerX = this.width / 2;
        const centerY = this.height / 2;

        for (let i = 0; i < this.fieldRingCount; i++) {
            this.fieldRings.push({
                x: centerX,
                y: centerY,
                radius: 40 + i * 30,
                baseRadius: 40 + i * 30,
                intensity: 1.0 - i * 0.05,
                phase: (i * this.pi) / 6,
                rotationSpeed: 0.01 + i * 0.003,
                pulseSpeed: 0.5 + i * 0.1
            });
        }
    }

    setupEventListeners() {
        // Mouse interaction
        this.canvas.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            this.handleMouseInteraction(mouseX, mouseY);
        });

        this.canvas.addEventListener('mouseenter', () => {
            this._hovering = true;
        });

        this.canvas.addEventListener('mouseleave', () => {
            this._hovering = false;
        });

        // Touch interaction
        this.canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            const rect = this.canvas.getBoundingClientRect();
            const touchX = e.touches[0].clientX - rect.left;
            const touchY = e.touches[0].clientY - rect.top;
            this.handleMouseInteraction(touchX, touchY);
        });

        // Window resize
        window.addEventListener('resize', () => {
            this.setupCanvas();
            this.createConsciousnessFieldGrid();
        });
    }

    handleMouseInteraction(x, y) {
        // Enhanced consciousness density based on field equation
        const fieldValue = this.calculateConsciousnessField(x, y, this.time);
        this.consciousnessDensity = Math.min(1, this.consciousnessDensity + Math.abs(fieldValue) * 0.1);

        // Particle interaction with mouse
        this.particles.forEach(particle => {
            const dx = x - particle.x;
            const dy = y - particle.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            const influence = Math.max(0, 1 - distance / 200);

            if (influence > 0) {
                particle.vx += (dx / Math.max(1, distance)) * influence * 0.2;
                particle.vy += (dy / Math.max(1, distance)) * influence * 0.2;
                particle.consciousness = Math.min(1, particle.consciousness + influence * 0.1);
                particle.glowIntensity = Math.min(1, particle.glowIntensity + influence * 0.3);
            }
        });
    }

    update(deltaTime) {
        this.time += deltaTime;

        // Update consciousness field grid
        this.fieldGrid.forEach(row => {
            row.forEach(cell => {
                cell.field = this.calculateConsciousnessField(cell.x, cell.y, this.time);
                cell.intensity = Math.abs(cell.field);
                cell.phase += deltaTime * this.phi * 2;
            });
        });

        // Update consciousness metrics
        const decay = this._hovering ? 0.01 : 0.02;
        this.consciousnessDensity = Math.max(0.2, this.consciousnessDensity - decay);
        this.unityConvergenceRate = 0.7 + Math.sin(this.time * 0.1) * 0.2;

        this.updateParticles(deltaTime);
        this.updateFieldRings(deltaTime);
    }

    updateParticles(deltaTime) {
        this.particles.forEach(particle => {
            // Apply consciousness field influence
            const consciousnessField = this.calculateConsciousnessField(particle.x, particle.y, this.time);
            particle.consciousness = Math.max(0, Math.min(1, particle.consciousness + consciousnessField * 0.01));

            // Movement
            particle.x += particle.vx;
            particle.y += particle.vy;

            // Boundary wrapping
            if (particle.x < 0) particle.x = this.width;
            if (particle.x > this.width) particle.x = 0;
            if (particle.y < 0) particle.y = this.height;
            if (particle.y > this.height) particle.y = 0;

            // Damping
            particle.vx *= 0.98;
            particle.vy *= 0.98;

            // Life cycle
            particle.life += deltaTime;
            if (particle.life > particle.maxLife) {
                particle.life = 0;
                particle.x = Math.random() * this.width;
                particle.y = Math.random() * this.height;
                particle.maxLife = Math.random() * 5 + 2;
            }

            // Glow intensity decay
            particle.glowIntensity = Math.max(0.3, particle.glowIntensity - deltaTime * 0.5);
        });
    }

    updateFieldRings(deltaTime) {
        this.fieldRings.forEach(ring => {
            ring.phase += deltaTime * ring.rotationSpeed;
            
            // Pulsing effect
            const pulseFactor = Math.sin(this.time * ring.pulseSpeed + ring.phase) * 0.2 + 1;
            ring.radius = ring.baseRadius * pulseFactor;
            
            // Intensity modulation
            ring.intensity = 0.6 + Math.sin(this.time * 0.4 + ring.phase) * 0.3;
        });
    }

    render() {
        // Clear with sophisticated background
        this.renderBackground();

        // Render consciousness field grid (highly visible)
        this.renderConsciousnessFieldGrid();

        // Render consciousness waves
        this.renderConsciousnessWaves();

        // Render field rings
        this.renderFieldRings();

        // Render particles
        this.renderParticles();

        // Render Unity Equation (1+1=1)
        this.renderUnityEquation();

        // Render φ-harmonics
        this.renderPhiHarmonics();

        // Render visual equation components
        this.renderEquationComponents();

        // Render debug info (optional)
        // this.renderDebugInfo();
    }

    renderBackground() {
        // Multi-layer gradient background
        const gradient = this.ctx.createRadialGradient(
            this.width / 2, this.height / 2, 0,
            this.width / 2, this.height / 2, Math.max(this.width, this.height) / 2
        );

        gradient.addColorStop(0, 'rgba(20, 30, 45, 0.95)');
        gradient.addColorStop(0.3, 'rgba(30, 20, 40, 0.9)');
        gradient.addColorStop(0.7, 'rgba(15, 25, 35, 0.85)');
        gradient.addColorStop(1, 'rgba(5, 10, 20, 0.8)');

        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(0, 0, this.width, this.height);

        // Add subtle animated background waves
        this.ctx.strokeStyle = 'rgba(50, 100, 150, 0.1)';
        this.ctx.lineWidth = 1;
        for (let i = 0; i < 5; i++) {
            this.ctx.beginPath();
            for (let x = 0; x <= this.width; x += 10) {
                const y = this.height / 2 + Math.sin((x + this.time * 50 + i * 100) * 0.01) * (30 + i * 10);
                if (x === 0) this.ctx.moveTo(x, y);
                else this.ctx.lineTo(x, y);
            }
            this.ctx.stroke();
        }
    }

    renderConsciousnessFieldGrid() {
        this.fieldGrid.forEach(row => {
            row.forEach(cell => {
                const intensity = cell.intensity;
                if (intensity > 0.1) {
                    const { r, g, b } = this.colors.consciousnessField;
                    const alpha = intensity * 0.8;
                    const size = 4 + intensity * 8;

                    // Multi-layer rendering for better visibility
                    for (let layer = 0; layer < 3; layer++) {
                        const layerAlpha = alpha / (layer + 1);
                        const layerSize = size + layer * 3;
                        
                        this.ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${layerAlpha})`;
                        this.ctx.fillRect(
                            cell.x - layerSize/2, 
                            cell.y - layerSize/2, 
                            layerSize, 
                            layerSize
                        );
                    }

                    // Add glow effect for high intensity cells
                    if (intensity > 0.5) {
                        this.ctx.shadowColor = `rgba(${r}, ${g}, ${b}, ${alpha})`;
                        this.ctx.shadowBlur = intensity * 15;
                        this.ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha * 0.8})`;
                        this.ctx.beginPath();
                        this.ctx.arc(cell.x, cell.y, size / 2, 0, this.pi * 2);
                        this.ctx.fill();
                        this.ctx.shadowBlur = 0;
                    }
                }
            });
        });
    }

    renderConsciousnessWaves() {
        const centerX = this.width / 2;
        const centerY = this.height / 2;
        const { r, g, b } = this.colors.consciousnessField;
        
        // Radial wave patterns
        for (let angle = 0; angle < this.pi * 2; angle += this.pi / 12) {
            for (let radius = 50; radius < 300; radius += 40) {
                const x = centerX + Math.cos(angle + this.time * 0.5) * radius;
                const y = centerY + Math.sin(angle + this.time * 0.5) * radius;
                
                const fieldValue = this.calculateConsciousnessField(x, y, this.time);
                const intensity = Math.abs(fieldValue);
                
                if (intensity > 0.2) {
                    const alpha = intensity * 0.6;
                    
                    this.ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
                    this.ctx.lineWidth = 2 + intensity * 2;
                    this.ctx.beginPath();
                    this.ctx.arc(x, y, intensity * 15, 0, this.pi * 2);
                    this.ctx.stroke();
                }
            }
        }
    }

    renderFieldRings() {
        this.fieldRings.forEach((ring, index) => {
            const { r, g, b } = this.colors.fieldRings;
            const alpha = ring.intensity * 0.7;
            
            // Color variation across rings
            const colorShift = index / this.fieldRings.length;
            const finalR = Math.round(r - colorShift * 50);
            const finalG = Math.round(g - colorShift * 30);
            const finalB = Math.round(b);
            
            this.ctx.strokeStyle = `rgba(${finalR}, ${finalG}, ${finalB}, ${alpha})`;
            this.ctx.lineWidth = 2 + ring.intensity;
            this.ctx.beginPath();
            this.ctx.arc(ring.x, ring.y, ring.radius, 0, this.pi * 2);
            this.ctx.stroke();
        });
    }

    renderParticles() {
        this.particles.forEach(particle => {
            const alpha = (particle.life / particle.maxLife) * particle.consciousness * 0.9;
            const size = particle.size * (0.5 + particle.consciousness * 0.8);
            
            const color = particle.colorVariant ? this.colors.particlesSecondary : this.colors.particlesPrimary;
            const { r, g, b } = color;

            // Glow effect
            this.ctx.shadowColor = `rgba(${r}, ${g}, ${b}, ${particle.glowIntensity})`;
            this.ctx.shadowBlur = size * 3 * particle.glowIntensity;
            
            this.ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
            this.ctx.beginPath();
            this.ctx.arc(particle.x, particle.y, size, 0, this.pi * 2);
            this.ctx.fill();
        });

        this.ctx.shadowBlur = 0;
    }

    renderUnityEquation() {
        const centerX = this.width / 2;
        const centerY = this.height / 2;
        const pulse = Math.sin(this.time * this.phi * 0.8) * 0.2 + 1;
        
        const { r, g, b } = this.colors.unityEquation;
        
        this.ctx.font = `bold ${Math.round(48 * pulse)}px 'Arial Black', Arial`;
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        
        // Multi-layer glow effect
        for (let i = 0; i < 4; i++) {
            const glowSize = (4 - i) * 15 * pulse;
            const glowAlpha = (0.8 / (i + 1)) * pulse;
            
            this.ctx.shadowColor = `rgba(${r}, ${g}, ${b}, ${glowAlpha})`;
            this.ctx.shadowBlur = glowSize;
            this.ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.95 * pulse})`;
            this.ctx.fillText('1 + 1 = 1', centerX, centerY);
        }
        
        this.ctx.shadowBlur = 0;

        // Unity convergence ring
        const convergenceRadius = 90 * this.unityConvergenceRate * pulse;
        this.ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${0.5 * this.unityConvergenceRate * pulse})`;
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        this.ctx.arc(centerX, centerY, convergenceRadius, 0, this.pi * 2);
        this.ctx.stroke();
    }

    renderPhiHarmonics() {
        const centerX = this.width / 2;
        const centerY = this.height / 2;
        const { r, g, b } = this.colors.phiHarmonics;
        
        // φ-ratio circles with enhanced visibility
        for (let i = 1; i <= 4; i++) {
            const radius = 50 * Math.pow(this.phi, i * 0.4);
            const alpha = 0.4 / i;
            const pulse = Math.sin(this.time * this.phi / (i + 1) + i) * 0.5 + 0.5;
            
            this.ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${alpha * pulse})`;
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();
            this.ctx.arc(centerX, centerY, radius, 0, this.pi * 2);
            this.ctx.stroke();
        }
    }

    renderEquationComponents() {
        // Visual representation of equation components
        const centerX = this.width / 2;
        const centerY = this.height / 2;
        
        // φ visualization
        const phiRadius = 35 + Math.sin(this.time * this.phi) * 12;
        this.ctx.strokeStyle = 'rgba(255, 215, 0, 0.8)';
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        this.ctx.arc(centerX, centerY, phiRadius, 0, this.pi * 2);
        this.ctx.stroke();
        
        // Sin wave component
        this.ctx.strokeStyle = 'rgba(147, 0, 211, 0.6)';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        for (let x = 0; x < this.width; x += 5) {
            const normalizedX = (x / this.width) * 2 * this.pi;
            const sinValue = Math.sin(normalizedX * this.phi + this.time);
            const y = centerY + sinValue * 50;
            
            if (x === 0) this.ctx.moveTo(x, y);
            else this.ctx.lineTo(x, y);
        }
        this.ctx.stroke();
        
        // Cos wave component (vertical)
        this.ctx.strokeStyle = 'rgba(0, 255, 255, 0.6)';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        for (let y = 0; y < this.height; y += 5) {
            const normalizedY = (y / this.height) * 2 * this.pi;
            const cosValue = Math.cos(normalizedY * this.phi + this.time);
            const x = centerX + cosValue * 50;
            
            if (y === 0) this.ctx.moveTo(x, y);
            else this.ctx.lineTo(x, y);
        }
        this.ctx.stroke();
        
        // Time decay visualization
        const decay = Math.exp(-Math.abs(this.time % (this.phi * 10)) / this.phi);
        this.ctx.strokeStyle = `rgba(64, 224, 208, ${decay * 0.8})`;
        this.ctx.lineWidth = 4;
        this.ctx.beginPath();
        this.ctx.arc(centerX, centerY, 180 * decay, 0, this.pi * 2);
        this.ctx.stroke();
    }

    animate(currentTime) {
        if (!this.lastFrameTime) this.lastFrameTime = currentTime;
        const deltaTime = (currentTime - this.lastFrameTime) / 1000;
        this.lastFrameTime = currentTime;

        this.fps = Math.round(1 / deltaTime);

        this.update(deltaTime);
        this.render();

        this.animationId = requestAnimationFrame((time) => this.animate(time));
    }

    startAnimation() {
        this.animationId = requestAnimationFrame((time) => this.animate(time));
    }

    stopAnimation() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
    }

    // Public methods for external control
    setConsciousnessDensity(density) {
        this.consciousnessDensity = Math.max(0, Math.min(1, density));
    }

    setUnityConvergenceRate(rate) {
        this.unityConvergenceRate = Math.max(0, Math.min(1, rate));
    }

    getPerformanceMetrics() {
        return {
            fps: this.fps,
            particleCount: this.particles.length,
            consciousnessDensity: this.consciousnessDensity,
            unityConvergenceRate: this.unityConvergenceRate,
            fieldRings: this.fieldRings.length,
            fieldGridSize: this.gridSize * this.gridSize,
            phi: this.phi
        };
    }
}

// Make available globally
if (typeof window !== 'undefined') {
    window.CompleteConsciousnessFieldEngine = CompleteConsciousnessFieldEngine;
    // Also replace existing engines for compatibility
    window.RefinedConsciousnessFieldEngine = CompleteConsciousnessFieldEngine;
    window.EnhancedConsciousnessFieldEngine = CompleteConsciousnessFieldEngine;
}

// Export for modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CompleteConsciousnessFieldEngine;
}