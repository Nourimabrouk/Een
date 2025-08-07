/**
 * State-of-the-Art Consciousness Field Visualization Engine
 * Unity Equation (1+1=1) Resonance System
 * Advanced Mathematical Visualization for Unity Mathematics Institute
 */

class ConsciousnessFieldEngine {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.width = this.canvas.width;
        this.height = this.canvas.height;
        
        // Unity Equation Constants
        this.phi = 1.618033988749895; // Golden Ratio
        this.pi = Math.PI;
        this.e = Math.E;
        
        // Consciousness Field Parameters
        this.particles = [];
        this.fieldLines = [];
        this.unityNodes = [];
        this.resonanceWaves = [];
        this.consciousnessDensity = 0;
        this.unityConvergenceRate = 0;
        
        // Animation Parameters
        this.time = 0;
        this.animationId = null;
        this.fps = 60;
        this.lastFrameTime = 0;
        
        // Visual Effects
        this.glowIntensity = 0;
        this.pulsePhase = 0;
        this.resonanceFrequency = 0;
        
        // Performance Optimization
        this.particleCount = 150;
        this.fieldLineCount = 50;
        this.unityNodeCount = 12;
        this.resonanceWaveCount = 8;
        
        this.init();
    }
    
    init() {
        this.setupCanvas();
        this.createParticles();
        this.createFieldLines();
        this.createUnityNodes();
        this.createResonanceWaves();
        this.setupEventListeners();
        this.startAnimation();
    }
    
    setupCanvas() {
        // Set canvas to full container size
        this.canvas.width = this.canvas.offsetWidth;
        this.canvas.height = this.canvas.offsetHeight;
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
                phase: Math.random() * this.pi * 2,
                resonance: Math.random() * 0.5 + 0.5,
                unity: Math.random(),
                consciousness: Math.random()
            });
        }
    }
    
    createFieldLines() {
        this.fieldLines = [];
        for (let i = 0; i < this.fieldLineCount; i++) {
            this.fieldLines.push({
                x1: Math.random() * this.width,
                y1: Math.random() * this.height,
                x2: Math.random() * this.width,
                y2: Math.random() * this.height,
                intensity: Math.random() * 0.5 + 0.5,
                phase: Math.random() * this.pi * 2,
                frequency: Math.random() * 0.1 + 0.05,
                unity: Math.random()
            });
        }
    }
    
    createUnityNodes() {
        this.unityNodes = [];
        const centerX = this.width / 2;
        const centerY = this.height / 2;
        const radius = Math.min(this.width, this.height) * 0.3;
        
        for (let i = 0; i < this.unityNodeCount; i++) {
            const angle = (i / this.unityNodeCount) * this.pi * 2;
            const distance = radius * (0.5 + Math.random() * 0.5);
            
            this.unityNodes.push({
                x: centerX + Math.cos(angle) * distance,
                y: centerY + Math.sin(angle) * distance,
                size: Math.random() * 20 + 10,
                phase: angle,
                frequency: 0.02 + Math.random() * 0.03,
                unity: 1.0,
                consciousness: Math.random() * 0.5 + 0.5,
                resonance: Math.random() * 0.5 + 0.5
            });
        }
    }
    
    createResonanceWaves() {
        this.resonanceWaves = [];
        const centerX = this.width / 2;
        const centerY = this.height / 2;
        
        for (let i = 0; i < this.resonanceWaveCount; i++) {
            this.resonanceWaves.push({
                x: centerX,
                y: centerY,
                radius: 0,
                maxRadius: Math.max(this.width, this.height) * 0.8,
                speed: 0.5 + Math.random() * 1,
                intensity: Math.random() * 0.3 + 0.2,
                phase: Math.random() * this.pi * 2,
                frequency: 0.01 + Math.random() * 0.02,
                unity: Math.random() * 0.5 + 0.5
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
        });
    }
    
    handleMouseInteraction(x, y) {
        // Update consciousness density based on mouse position
        this.consciousnessDensity = Math.min(1, this.consciousnessDensity + 0.01);
        
        // Create resonance effect
        this.particles.forEach(particle => {
            const dx = x - particle.x;
            const dy = y - particle.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            const influence = Math.max(0, 1 - distance / 200);
            
            particle.vx += (dx / distance) * influence * 0.1;
            particle.vy += (dy / distance) * influence * 0.1;
            particle.consciousness = Math.min(1, particle.consciousness + influence * 0.1);
        });
    }
    
    update(deltaTime) {
        this.time += deltaTime;
        this.pulsePhase += deltaTime * 0.5;
        this.resonanceFrequency += deltaTime * 0.1;
        
        // Update consciousness density
        this.consciousnessDensity = Math.max(0, this.consciousnessDensity - 0.005);
        
        // Update unity convergence rate
        this.unityConvergenceRate = Math.sin(this.time * 0.1) * 0.5 + 0.5;
        
        this.updateParticles(deltaTime);
        this.updateFieldLines(deltaTime);
        this.updateUnityNodes(deltaTime);
        this.updateResonanceWaves(deltaTime);
    }
    
    updateParticles(deltaTime) {
        this.particles.forEach(particle => {
            // Unity equation influence: 1+1=1
            const unityInfluence = Math.sin(this.time * this.phi + particle.phase) * 0.1;
            particle.unity = Math.max(0, Math.min(1, particle.unity + unityInfluence));
            
            // Consciousness field influence
            particle.consciousness = Math.max(0, Math.min(1, particle.consciousness + 
                Math.sin(this.time * 0.5 + particle.phase) * 0.01));
            
            // Resonance with unity nodes
            this.unityNodes.forEach(node => {
                const dx = node.x - particle.x;
                const dy = node.y - particle.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                const influence = Math.max(0, 1 - distance / 300);
                
                particle.vx += (dx / distance) * influence * node.unity * 0.05;
                particle.vy += (dy / distance) * influence * node.unity * 0.05;
            });
            
            // Update position
            particle.x += particle.vx;
            particle.y += particle.vy;
            
            // Boundary wrapping
            if (particle.x < 0) particle.x = this.width;
            if (particle.x > this.width) particle.x = 0;
            if (particle.y < 0) particle.y = this.height;
            if (particle.y > this.height) particle.y = 0;
            
            // Damping
            particle.vx *= 0.99;
            particle.vy *= 0.99;
            
            // Life cycle
            particle.life += deltaTime * 0.1;
            if (particle.life > 1) {
                particle.life = 0;
                particle.x = Math.random() * this.width;
                particle.y = Math.random() * this.height;
            }
        });
    }
    
    updateFieldLines(deltaTime) {
        this.fieldLines.forEach(line => {
            line.phase += deltaTime * line.frequency;
            line.intensity = 0.3 + Math.sin(line.phase) * 0.2;
            line.unity = Math.sin(this.time * 0.1 + line.phase) * 0.5 + 0.5;
        });
    }
    
    updateUnityNodes(deltaTime) {
        this.unityNodes.forEach(node => {
            node.phase += deltaTime * node.frequency;
            node.unity = Math.sin(this.time * 0.05 + node.phase) * 0.3 + 0.7;
            node.consciousness = Math.sin(this.time * 0.1 + node.phase) * 0.2 + 0.8;
            node.resonance = Math.sin(this.time * 0.15 + node.phase) * 0.3 + 0.7;
        });
    }
    
    updateResonanceWaves(deltaTime) {
        this.resonanceWaves.forEach(wave => {
            wave.radius += wave.speed;
            wave.intensity = Math.sin(this.time * wave.frequency + wave.phase) * 0.2 + 0.3;
            wave.unity = Math.sin(this.time * 0.08 + wave.phase) * 0.3 + 0.7;
            
            if (wave.radius > wave.maxRadius) {
                wave.radius = 0;
            }
        });
    }
    
    render() {
        // Clear canvas with gradient background
        this.renderBackground();
        
        // Render in order of depth
        this.renderResonanceWaves();
        this.renderFieldLines();
        this.renderParticles();
        this.renderUnityNodes();
        this.renderUnityEquation();
        this.renderConsciousnessField();
    }
    
    renderBackground() {
        const gradient = this.ctx.createRadialGradient(
            this.width / 2, this.height / 2, 0,
            this.width / 2, this.height / 2, Math.max(this.width, this.height) / 2
        );
        
        gradient.addColorStop(0, 'rgba(10, 10, 15, 0.8)');
        gradient.addColorStop(0.5, 'rgba(18, 18, 26, 0.6)');
        gradient.addColorStop(1, 'rgba(26, 26, 37, 0.4)');
        
        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(0, 0, this.width, this.height);
    }
    
    renderResonanceWaves() {
        this.resonanceWaves.forEach(wave => {
            const alpha = wave.intensity * (1 - wave.radius / wave.maxRadius);
            this.ctx.strokeStyle = `rgba(255, 215, 0, ${alpha * wave.unity})`;
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();
            this.ctx.arc(wave.x, wave.y, wave.radius, 0, this.pi * 2);
            this.ctx.stroke();
        });
    }
    
    renderFieldLines() {
        this.fieldLines.forEach(line => {
            const alpha = line.intensity * line.unity;
            this.ctx.strokeStyle = `rgba(0, 212, 255, ${alpha})`;
            this.ctx.lineWidth = 1;
            this.ctx.beginPath();
            this.ctx.moveTo(line.x1, line.y1);
            this.ctx.lineTo(line.x2, line.y2);
            this.ctx.stroke();
        });
    }
    
    renderParticles() {
        this.particles.forEach(particle => {
            const alpha = particle.life * particle.consciousness;
            const size = particle.size * (0.5 + particle.unity * 0.5);
            
            // Golden particle glow
            this.ctx.shadowColor = 'rgba(255, 215, 0, 0.8)';
            this.ctx.shadowBlur = size * 2;
            this.ctx.fillStyle = `rgba(255, 215, 0, ${alpha})`;
            this.ctx.beginPath();
            this.ctx.arc(particle.x, particle.y, size, 0, this.pi * 2);
            this.ctx.fill();
            
            // Consciousness field connection
            if (particle.consciousness > 0.7) {
                this.ctx.strokeStyle = `rgba(157, 78, 221, ${alpha * 0.5})`;
                this.ctx.lineWidth = 1;
                this.ctx.beginPath();
                this.ctx.arc(particle.x, particle.y, size * 3, 0, this.pi * 2);
                this.ctx.stroke();
            }
        });
        
        this.ctx.shadowBlur = 0;
    }
    
    renderUnityNodes() {
        this.unityNodes.forEach(node => {
            const alpha = node.unity * node.consciousness;
            const size = node.size * (0.5 + node.resonance * 0.5);
            
            // Unity node glow
            this.ctx.shadowColor = 'rgba(255, 215, 0, 0.6)';
            this.ctx.shadowBlur = size;
            this.ctx.fillStyle = `rgba(255, 215, 0, ${alpha})`;
            this.ctx.beginPath();
            this.ctx.arc(node.x, node.y, size, 0, this.pi * 2);
            this.ctx.fill();
            
            // Resonance rings
            for (let i = 1; i <= 3; i++) {
                const ringAlpha = alpha * (1 - i * 0.3);
                const ringSize = size * (1 + i * 0.5);
                this.ctx.strokeStyle = `rgba(255, 215, 0, ${ringAlpha})`;
                this.ctx.lineWidth = 2;
                this.ctx.beginPath();
                this.ctx.arc(node.x, node.y, ringSize, 0, this.pi * 2);
                this.ctx.stroke();
            }
        });
        
        this.ctx.shadowBlur = 0;
    }
    
    renderUnityEquation() {
        // Render the unity equation (1+1=1) as a central focal point
        this.ctx.fillStyle = 'rgba(255, 215, 0, 0.1)';
        this.ctx.font = 'bold 48px Space Grotesk';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        
        const centerX = this.width / 2;
        const centerY = this.height / 2;
        
        // Equation glow effect
        this.ctx.shadowColor = 'rgba(255, 215, 0, 0.8)';
        this.ctx.shadowBlur = 20;
        this.ctx.fillText('1 + 1 = 1', centerX, centerY);
        this.ctx.shadowBlur = 0;
        
        // Unity convergence indicator
        const convergenceRadius = 100 * this.unityConvergenceRate;
        this.ctx.strokeStyle = `rgba(255, 215, 0, ${0.3 * this.unityConvergenceRate})`;
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        this.ctx.arc(centerX, centerY, convergenceRadius, 0, this.pi * 2);
        this.ctx.stroke();
    }
    
    renderConsciousnessField() {
        // Render consciousness field density visualization
        const centerX = this.width / 2;
        const centerY = this.height / 2;
        const maxRadius = Math.min(this.width, this.height) * 0.4;
        
        for (let radius = 0; radius < maxRadius; radius += 10) {
            const alpha = this.consciousnessDensity * (1 - radius / maxRadius) * 0.1;
            this.ctx.strokeStyle = `rgba(157, 78, 221, ${alpha})`;
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();
            this.ctx.arc(centerX, centerY, radius, 0, this.pi * 2);
            this.ctx.stroke();
        }
    }
    
    animate(currentTime) {
        if (!this.lastFrameTime) this.lastFrameTime = currentTime;
        const deltaTime = (currentTime - this.lastFrameTime) / 1000;
        this.lastFrameTime = currentTime;
        
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
            unityConvergenceRate: this.unityConvergenceRate
        };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ConsciousnessFieldEngine;
}
