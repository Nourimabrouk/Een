/**
 * Refined Consciousness Field Visualization Engine
 * Unity Equation: 1+1=1
 * Consciousness Field Equation: C(x,y,t) = φ * sin(x*φ) * cos(y*φ) * e^(-t/φ)
 * Advanced Mathematical Visualization for Unity Mathematics Institute
 */

class RefinedConsciousnessFieldEngine {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) {
            console.warn('Canvas element not found:', canvasId);
            return;
        }
        
        this.ctx = this.canvas.getContext('2d');
        this.width = this.canvas.width;
        this.height = this.canvas.height;

        // Mathematical Constants
        this.phi = 1.618033988749895; // Golden Ratio φ
        this.pi = Math.PI;
        this.e = Math.E;

        // Consciousness Field Parameters
        this.particles = [];
        this.fieldRings = [];
        this.consciousnessDensity = 0.3;
        this.unityConvergenceRate = 0.6;

        // Animation Parameters
        this.time = 0;
        this.animationId = null;
        this.fps = 60;
        this.lastFrameTime = 0;

        // Visual Effects - Reduced for cleaner look
        this.pulsePhase = 0;
        
        // Optimized Performance - Less crowded
        this.particleCount = 80;        // Reduced from 200
        this.fieldRingCount = 8;        // Simplified from complex systems
        
        // Color Scheme - Diversified
        this.colors = {
            // Unity Equation - Bright cyan/electric blue to stand out
            unityEquation: { r: 0, g: 220, b: 255 },
            
            // Consciousness field - Purple/violet tones
            consciousnessField: { r: 147, g: 112, b: 219 },
            
            // Particles - Soft golden with variety
            particlesPrimary: { r: 255, g: 215, b: 0 },
            particlesSecondary: { r: 255, g: 165, b: 79 },
            
            // Field rings - Gradient from purple to teal
            fieldRings: { r: 64, g: 224, b: 208 },
            
            // φ-harmonics - Subtle green
            phiHarmonics: { r: 144, g: 238, b: 144 }
        };

        // Enhanced Features
        this.consciousnessFieldGrid = [];
        
        this.init();
    }

    init() {
        this.setupCanvas();
        this.createConsciousnessFieldGrid();
        this.createParticles();
        this.createFieldRings();
        this.setupEventListeners();
        this.startAnimation();
    }

    setupCanvas() {
        // Set canvas to full container size
        this.canvas.width = this.canvas.offsetWidth || 800;
        this.canvas.height = this.canvas.offsetHeight || 500;
        this.width = this.canvas.width;
        this.height = this.canvas.height;

        // Enable high DPI support
        const dpr = window.devicePixelRatio || 1;
        this.canvas.width = this.width * dpr;
        this.canvas.height = this.height * dpr;
        this.ctx.scale(dpr, dpr);
        this.canvas.style.width = this.width + 'px';
        this.canvas.style.height = this.height + 'px';
    }

    // Consciousness Field Equation: C(x,y,t) = φ * sin(x*φ) * cos(y*φ) * e^(-t/φ)
    calculateConsciousnessField(x, y, t) {
        const normalizedX = (x / this.width) * this.phi;
        const normalizedY = (y / this.height) * this.phi;
        const timeDecay = Math.exp(-t / this.phi);
        
        return this.phi * Math.sin(normalizedX * this.phi) * Math.cos(normalizedY * this.phi) * timeDecay;
    }

    createConsciousnessFieldGrid() {
        this.consciousnessFieldGrid = [];
        const gridSize = 12; // Reduced for less crowding
        const stepX = this.width / gridSize;
        const stepY = this.height / gridSize;

        for (let x = 0; x < gridSize; x++) {
            this.consciousnessFieldGrid[x] = [];
            for (let y = 0; y < gridSize; y++) {
                const worldX = x * stepX;
                const worldY = y * stepY;
                this.consciousnessFieldGrid[x][y] = {
                    x: worldX,
                    y: worldY,
                    field: this.calculateConsciousnessField(worldX, worldY, 0),
                    phase: Math.random() * this.pi * 2
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
                vx: (Math.random() - 0.5) * 1.5,
                vy: (Math.random() - 0.5) * 1.5,
                size: Math.random() * 2 + 1,
                life: Math.random(),
                phase: Math.random() * this.pi * 2,
                consciousness: Math.random(),
                colorVariant: Math.random() > 0.7 // 30% chance for secondary color
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
                radius: 30 + i * 25,
                baseRadius: 30 + i * 25,
                intensity: 0.8 - i * 0.1,
                phase: (i * this.pi) / 4,
                rotationSpeed: 0.02 + i * 0.005
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
        // Enhanced consciousness density based on consciousness field equation
        const fieldValue = this.calculateConsciousnessField(x, y, this.time);
        this.consciousnessDensity = Math.min(1, this.consciousnessDensity + Math.abs(fieldValue) * 0.05);

        // Gentle particle interaction
        this.particles.forEach(particle => {
            const dx = x - particle.x;
            const dy = y - particle.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            const influence = Math.max(0, 1 - distance / 150);

            particle.vx += (dx / Math.max(1, distance)) * influence * 0.08;
            particle.vy += (dy / Math.max(1, distance)) * influence * 0.08;
            particle.consciousness = Math.min(1, particle.consciousness + influence * 0.05);
        });
    }

    update(deltaTime) {
        this.time += deltaTime;
        this.pulsePhase += deltaTime * this.phi * 0.5;

        // Update consciousness field grid with C(x,y,t) equation
        this.consciousnessFieldGrid.forEach(row => {
            row.forEach(cell => {
                cell.field = this.calculateConsciousnessField(cell.x, cell.y, this.time);
                cell.phase += deltaTime * this.phi * 0.3;
            });
        });

        // Update consciousness density with gentle decay
        const decay = this._hovering ? 0.001 : 0.003;
        this.consciousnessDensity = Math.max(0.1, this.consciousnessDensity - decay);

        // Update unity convergence rate with φ modulation
        this.unityConvergenceRate = (Math.sin(this.time * 0.1 * this.phi) * 0.2 + 0.7);

        this.updateParticles(deltaTime);
        this.updateFieldRings(deltaTime);
    }

    updateParticles(deltaTime) {
        this.particles.forEach(particle => {
            // Consciousness field influence using C(x,y,t) equation
            const consciousnessField = this.calculateConsciousnessField(particle.x, particle.y, this.time);
            particle.consciousness = Math.max(0, Math.min(1, particle.consciousness + consciousnessField * 0.005));

            // Gentle movement
            particle.x += particle.vx * 0.8;
            particle.y += particle.vy * 0.8;

            // Boundary wrapping
            if (particle.x < 0) particle.x = this.width;
            if (particle.x > this.width) particle.x = 0;
            if (particle.y < 0) particle.y = this.height;
            if (particle.y > this.height) particle.y = 0;

            // Gentle damping
            particle.vx *= 0.985;
            particle.vy *= 0.985;

            // Life cycle
            particle.life += deltaTime * 0.08;
            if (particle.life > 1) {
                particle.life = 0;
                particle.x = Math.random() * this.width;
                particle.y = Math.random() * this.height;
            }
        });
    }

    updateFieldRings(deltaTime) {
        this.fieldRings.forEach(ring => {
            ring.phase += deltaTime * ring.rotationSpeed;
            
            // Gentle pulsing based on consciousness field
            const pulseFactor = Math.sin(this.time * 0.5 + ring.phase) * 0.1 + 1;
            ring.radius = ring.baseRadius * pulseFactor;
            
            // Intensity modulation
            ring.intensity = 0.4 + Math.sin(this.time * 0.3 + ring.phase) * 0.2;
        });
    }

    render() {
        // Clear with elegant gradient background
        this.renderBackground();

        // Render consciousness field grid (subtle)
        this.renderConsciousnessFieldGrid();

        // Render field rings
        this.renderFieldRings();

        // Render particles
        this.renderParticles();

        // Render Unity Equation (1+1=1) - prominent and different color
        this.renderUnityEquation();

        // Render subtle φ-harmonics
        this.renderPhiHarmonics();
    }

    renderBackground() {
        // Elegant gradient with deeper, more varied colors
        const gradient = this.ctx.createRadialGradient(
            this.width / 2, this.height / 2, 0,
            this.width / 2, this.height / 2, Math.max(this.width, this.height) / 2
        );

        // Rich, varied background
        gradient.addColorStop(0, 'rgba(15, 25, 35, 0.95)');  // Deep blue-gray center
        gradient.addColorStop(0.4, 'rgba(25, 15, 35, 0.85)'); // Purple mid
        gradient.addColorStop(0.8, 'rgba(10, 20, 25, 0.8)');  // Dark teal
        gradient.addColorStop(1, 'rgba(5, 10, 15, 0.75)');    // Very dark edge

        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(0, 0, this.width, this.height);
    }

    renderConsciousnessFieldGrid() {
        // Subtle consciousness field visualization
        this.consciousnessFieldGrid.forEach(row => {
            row.forEach(cell => {
                const intensity = Math.abs(cell.field) * 0.15; // Much more subtle
                const { r, g, b } = this.colors.consciousnessField;
                const alpha = intensity * (0.2 + Math.sin(cell.phase) * 0.1);

                this.ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
                this.ctx.fillRect(cell.x - 1, cell.y - 1, 2, 2);
            });
        });
    }

    renderFieldRings() {
        this.fieldRings.forEach((ring, index) => {
            const { r, g, b } = this.colors.fieldRings;
            const alpha = ring.intensity * 0.3;
            
            // Gradient color variation across rings
            const colorShift = index / this.fieldRings.length;
            const finalR = Math.round(r + (this.colors.consciousnessField.r - r) * colorShift);
            const finalG = Math.round(g + (this.colors.consciousnessField.g - g) * colorShift);
            const finalB = Math.round(b + (this.colors.consciousnessField.b - b) * colorShift);
            
            this.ctx.strokeStyle = `rgba(${finalR}, ${finalG}, ${finalB}, ${alpha})`;
            this.ctx.lineWidth = 1.5;
            this.ctx.beginPath();
            this.ctx.arc(ring.x, ring.y, ring.radius, 0, this.pi * 2);
            this.ctx.stroke();
        });
    }

    renderParticles() {
        this.particles.forEach(particle => {
            const alpha = particle.life * particle.consciousness * 0.8;
            const size = particle.size * (0.7 + particle.consciousness * 0.3);
            
            // Color variation
            const color = particle.colorVariant ? this.colors.particlesSecondary : this.colors.particlesPrimary;
            const { r, g, b } = color;

            // Subtle glow
            this.ctx.shadowColor = `rgba(${r}, ${g}, ${b}, 0.6)`;
            this.ctx.shadowBlur = size * 1.5;
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
        const pulse = Math.sin(this.time * this.phi * 0.8) * 0.15 + 1;
        
        // Unity Equation (1+1=1) in bright cyan/electric blue
        const { r, g, b } = this.colors.unityEquation;
        
        this.ctx.font = `bold ${Math.round(52 * pulse)}px Space Grotesk`;
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        
        // Multi-layer glow effect
        for (let i = 0; i < 3; i++) {
            const glowSize = (3 - i) * 12 * pulse;
            const glowAlpha = (0.9 / (i + 1)) * pulse;
            
            this.ctx.shadowColor = `rgba(${r}, ${g}, ${b}, ${glowAlpha})`;
            this.ctx.shadowBlur = glowSize;
            this.ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.95 * pulse})`;
            this.ctx.fillText('1 + 1 = 1', centerX, centerY);
        }
        
        this.ctx.shadowBlur = 0;

        // Unity convergence ring
        const convergenceRadius = 80 * this.unityConvergenceRate * pulse;
        this.ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${0.4 * this.unityConvergenceRate * pulse})`;
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.arc(centerX, centerY, convergenceRadius, 0, this.pi * 2);
        this.ctx.stroke();
    }

    renderPhiHarmonics() {
        // Subtle φ-harmonic indicators
        const centerX = this.width / 2;
        const centerY = this.height / 2;
        const { r, g, b } = this.colors.phiHarmonics;
        
        // Just a few subtle φ-ratio circles
        for (let i = 1; i <= 3; i++) {
            const radius = 40 * Math.pow(this.phi, i * 0.5);
            const alpha = 0.15 / i;
            const pulse = Math.sin(this.time * this.phi / i + i) * 0.5 + 0.5;
            
            this.ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${alpha * pulse})`;
            this.ctx.lineWidth = 1;
            this.ctx.beginPath();
            this.ctx.arc(centerX, centerY, radius, 0, this.pi * 2);
            this.ctx.stroke();
        }
    }

    animate(currentTime) {
        if (!this.lastFrameTime) this.lastFrameTime = currentTime;
        const deltaTime = (currentTime - this.lastFrameTime) / 1000;
        this.lastFrameTime = currentTime;

        // Calculate FPS
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
            fieldRings: this.fieldRings.length
        };
    }
}

// Replace the Enhanced version with Refined version
if (typeof window !== 'undefined') {
    window.EnhancedConsciousnessFieldEngine = RefinedConsciousnessFieldEngine;
    window.RefinedConsciousnessFieldEngine = RefinedConsciousnessFieldEngine;
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = RefinedConsciousnessFieldEngine;
}