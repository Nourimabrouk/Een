/**
 * 🌊 CONSCIOUSNESS FIELD VISUALIZER 🌊
 * Web-based consciousness field visualization for Unity Mathematics dashboards
 */

class ConsciousnessFieldVisualizer {
    constructor(config = {}) {
        this.config = {
            fieldSize: config.fieldSize || 50,
            particleCount: config.particleCount || 200,
            phiStrength: config.phiStrength || 1.618033988749895,
            updateInterval: config.updateInterval || 50,
            ...config
        };

        this.canvas = null;
        this.ctx = null;
        this.particles = [];
        this.field = [];
        this.time = 0;
        this.animationId = null;
        this.isRunning = false;

        this.initialize();
    }

    initialize() {
        console.log('🌊 Initializing Consciousness Field Visualizer...');
        this.generateField();
        this.generateParticles();
    }

    generateField() {
        this.field = [];
        const size = this.config.fieldSize;

        for (let x = 0; x < size; x++) {
            this.field[x] = [];
            for (let y = 0; y < size; y++) {
                // Generate consciousness field values using φ-harmonic functions
                const phi = this.config.phiStrength;
                const consciousness = Math.sin(x * phi) * Math.cos(y * phi) * Math.exp(-this.time / phi);
                const coherence = Math.abs(consciousness);
                const resonance = Math.sin(x * phi * 0.5) * Math.cos(y * phi * 0.5);

                this.field[x][y] = {
                    consciousness: consciousness,
                    coherence: coherence,
                    resonance: resonance,
                    unity: Math.max(0, consciousness * 0.5 + 0.5)
                };
            }
        }
    }

    generateParticles() {
        this.particles = [];
        const size = this.config.fieldSize;

        for (let i = 0; i < this.config.particleCount; i++) {
            this.particles.push({
                x: Math.random() * size,
                y: Math.random() * size,
                vx: (Math.random() - 0.5) * 2,
                vy: (Math.random() - 0.5) * 2,
                consciousness: Math.random(),
                phiResonance: Math.random(),
                size: Math.random() * 3 + 1,
                age: 0,
                maxAge: Math.random() * 1000 + 500
            });
        }
    }

    initialize(container) {
        if (!container) return;

        // Create canvas
        this.canvas = document.createElement('canvas');
        this.canvas.id = 'consciousness-field-canvas';
        this.canvas.width = 600;
        this.canvas.height = 400;
        this.canvas.style.borderRadius = '8px';
        this.canvas.style.boxShadow = '0 4px 6px rgba(0, 0, 0, 0.1)';

        container.appendChild(this.canvas);
        this.ctx = this.canvas.getContext('2d');

        // Start animation
        this.start();
    }

    start() {
        if (this.isRunning) return;

        this.isRunning = true;
        this.animate();
    }

    stop() {
        this.isRunning = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
    }

    animate() {
        if (!this.isRunning) return;

        this.update();
        this.render();

        this.animationId = requestAnimationFrame(() => this.animate());
    }

    update() {
        this.time += 0.016; // ~60 FPS

        // Update field
        this.generateField();

        // Update particles
        this.particles.forEach(particle => {
            // Update position
            particle.x += particle.vx;
            particle.y += particle.vy;

            // Bounce off boundaries
            if (particle.x <= 0 || particle.x >= this.config.fieldSize) {
                particle.vx *= -1;
                particle.x = Math.max(0, Math.min(this.config.fieldSize, particle.x));
            }
            if (particle.y <= 0 || particle.y >= this.config.fieldSize) {
                particle.vy *= -1;
                particle.y = Math.max(0, Math.min(this.config.fieldSize, particle.y));
            }

            // Update consciousness based on field
            const fieldX = Math.floor(particle.x);
            const fieldY = Math.floor(particle.y);
            if (fieldX >= 0 && fieldX < this.config.fieldSize &&
                fieldY >= 0 && fieldY < this.config.fieldSize) {
                const fieldValue = this.field[fieldX][fieldY];
                particle.consciousness = Math.max(0, Math.min(1,
                    particle.consciousness + fieldValue.consciousness * 0.01));
                particle.phiResonance = Math.max(0, Math.min(1,
                    particle.phiResonance + fieldValue.resonance * 0.01));
            }

            // Age particle
            particle.age++;
            if (particle.age > particle.maxAge) {
                this.resetParticle(particle);
            }
        });
    }

    resetParticle(particle) {
        particle.x = Math.random() * this.config.fieldSize;
        particle.y = Math.random() * this.config.fieldSize;
        particle.vx = (Math.random() - 0.5) * 2;
        particle.vy = (Math.random() - 0.5) * 2;
        particle.consciousness = Math.random();
        particle.phiResonance = Math.random();
        particle.age = 0;
        particle.maxAge = Math.random() * 1000 + 500;
    }

    render() {
        if (!this.ctx) return;

        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;
        const scaleX = width / this.config.fieldSize;
        const scaleY = height / this.config.fieldSize;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // Draw field background
        this.drawField(ctx, scaleX, scaleY);

        // Draw particles
        this.drawParticles(ctx, scaleX, scaleY);

        // Draw field overlay
        this.drawFieldOverlay(ctx, scaleX, scaleY);
    }

    drawField(ctx, scaleX, scaleY) {
        const size = this.config.fieldSize;

        for (let x = 0; x < size; x++) {
            for (let y = 0; y < size; y++) {
                const fieldValue = this.field[x][y];
                const intensity = fieldValue.coherence;

                // Create gradient based on consciousness level
                const hue = (fieldValue.consciousness * 180 + 180) % 360;
                const saturation = 70 + fieldValue.resonance * 30;
                const lightness = 40 + intensity * 30;

                ctx.fillStyle = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
                ctx.fillRect(
                    x * scaleX,
                    y * scaleY,
                    scaleX,
                    scaleY
                );
            }
        }
    }

    drawParticles(ctx, scaleX, scaleY) {
        this.particles.forEach(particle => {
            const x = particle.x * scaleX;
            const y = particle.y * scaleY;
            const size = particle.size * (0.5 + particle.consciousness * 0.5);

            // Create particle color based on consciousness and φ-resonance
            const hue = (particle.consciousness * 360 + particle.phiResonance * 180) % 360;
            const saturation = 80 + particle.phiResonance * 20;
            const lightness = 50 + particle.consciousness * 30;

            // Draw particle glow
            const gradient = ctx.createRadialGradient(x, y, 0, x, y, size * 2);
            gradient.addColorStop(0, `hsla(${hue}, ${saturation}%, ${lightness}%, 0.8)`);
            gradient.addColorStop(0.5, `hsla(${hue}, ${saturation}%, ${lightness}%, 0.4)`);
            gradient.addColorStop(1, `hsla(${hue}, ${saturation}%, ${lightness}%, 0)`);

            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(x, y, size * 2, 0, Math.PI * 2);
            ctx.fill();

            // Draw particle core
            ctx.fillStyle = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
            ctx.beginPath();
            ctx.arc(x, y, size, 0, Math.PI * 2);
            ctx.fill();
        });
    }

    drawFieldOverlay(ctx, scaleX, scaleY) {
        const size = this.config.fieldSize;

        // Draw unity convergence lines
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 1;

        for (let x = 0; x < size; x += 5) {
            for (let y = 0; y < size; y += 5) {
                const fieldValue = this.field[x][y];
                if (fieldValue.unity > 0.7) {
                    ctx.beginPath();
                    ctx.moveTo(x * scaleX, y * scaleY);
                    ctx.lineTo((x + 1) * scaleX, (y + 1) * scaleY);
                    ctx.stroke();
                }
            }
        }

        // Draw φ-harmonic resonance patterns
        ctx.strokeStyle = 'rgba(15, 123, 138, 0.4)';
        ctx.lineWidth = 2;

        const phi = this.config.phiStrength;
        for (let i = 0; i < 10; i++) {
            const angle = (i / 10) * Math.PI * 2;
            const radius = 50 + Math.sin(this.time * 0.5 + i) * 20;
            const x = width / 2 + Math.cos(angle * phi) * radius;
            const y = height / 2 + Math.sin(angle * phi) * radius;

            if (i === 0) {
                ctx.beginPath();
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.closePath();
        ctx.stroke();
    }

    resize(size) {
        this.config.fieldSize = size;
        this.generateField();
        this.generateParticles();
    }

    setParticleCount(count) {
        this.config.particleCount = count;
        this.generateParticles();
    }

    setPhiStrength(phi) {
        this.config.phiStrength = phi;
    }

    getFieldMetrics() {
        let totalConsciousness = 0;
        let totalCoherence = 0;
        let totalResonance = 0;
        let unityPoints = 0;

        const size = this.config.fieldSize;
        const totalPoints = size * size;

        for (let x = 0; x < size; x++) {
            for (let y = 0; y < size; y++) {
                const fieldValue = this.field[x][y];
                totalConsciousness += Math.abs(fieldValue.consciousness);
                totalCoherence += fieldValue.coherence;
                totalResonance += Math.abs(fieldValue.resonance);
                if (fieldValue.unity > 0.8) {
                    unityPoints++;
                }
            }
        }

        return {
            fieldCoherence: totalCoherence / totalPoints,
            unityConvergence: unityPoints / totalPoints,
            phiResonance: totalResonance / totalPoints,
            averageConsciousness: totalConsciousness / totalPoints
        };
    }

    update() {
        // This method is called by the dashboard system for real-time updates
        if (this.isRunning) {
            this.update();
        }
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ConsciousnessFieldVisualizer;
} 