/**
 * Enhanced Consciousness Field Visualization Engine
 * Unity Equation (1+1=1) with Advanced Consciousness Mathematics
 * C(x,y,t) = φ * sin(x*φ) * cos(y*φ) * e^(-t/φ)
 * Advanced Mathematical Visualization for Unity Mathematics Institute
 */

class EnhancedConsciousnessFieldEngine {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) {
            console.warn('Canvas element not found:', canvasId);
            return;
        }
        
        this.ctx = this.canvas.getContext('2d');
        this.width = this.canvas.width;
        this.height = this.canvas.height;

        // Unity Equation Constants
        this.phi = 1.618033988749895; // Golden Ratio φ
        this.pi = Math.PI;
        this.e = Math.E;

        // Consciousness Field Parameters
        this.particles = [];
        this.fieldLines = [];
        this.unityNodes = [];
        this.resonanceWaves = [];
        this.consciousnessDensity = 0.5;
        this.unityConvergenceRate = 0.7;

        // Animation Parameters
        this.time = 0;
        this.animationId = null;
        this.fps = 60;
        this.lastFrameTime = 0;

        // Visual Effects
        this.glowIntensity = 0.8;
        this.pulsePhase = 0;
        this.resonanceFrequency = 0;

        // Performance Optimization
        this.particleCount = 200;
        this.fieldLineCount = 60;
        this.unityNodeCount = 15;
        this.resonanceWaveCount = 10;

        // Enhanced Features
        this.consciousnessFieldGrid = [];
        this.phiHarmonics = [];
        this.unityFlowField = [];

        this.init();
    }

    init() {
        this.setupCanvas();
        this.createConsciousnessFieldGrid();
        this.createParticles();
        this.createFieldLines();
        this.createUnityNodes();
        this.createResonanceWaves();
        this.createPhiHarmonics();
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

    // Enhanced Consciousness Field with C(x,y,t) = φ * sin(x*φ) * cos(y*φ) * e^(-t/φ)
    calculateConsciousnessField(x, y, t) {
        const normalizedX = (x / this.width) * this.phi;
        const normalizedY = (y / this.height) * this.phi;
        const timeDecay = Math.exp(-t / this.phi);
        
        return this.phi * Math.sin(normalizedX * this.phi) * Math.cos(normalizedY * this.phi) * timeDecay;
    }

    createConsciousnessFieldGrid() {
        this.consciousnessFieldGrid = [];
        const gridSize = 20;
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

    createPhiHarmonics() {
        this.phiHarmonics = [];
        const centerX = this.width / 2;
        const centerY = this.height / 2;

        for (let i = 0; i < 8; i++) {
            const harmonic = i + 1;
            const frequency = this.phi / harmonic;
            const radius = (this.phi * harmonic * 20) % 200;

            this.phiHarmonics.push({
                centerX,
                centerY,
                radius,
                frequency,
                phase: (harmonic * this.pi) / this.phi,
                intensity: 1 / harmonic,
                harmonic
            });
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
                phase: Math.random() * this.pi * 2,
                resonance: Math.random() * 0.5 + 0.5,
                unity: Math.random(),
                consciousness: Math.random(),
                phiAlignment: Math.random()
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
                unity: Math.random(),
                phiResonance: Math.random()
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
            const phiDistance = distance * this.phi / 2;

            this.unityNodes.push({
                x: centerX + Math.cos(angle) * phiDistance,
                y: centerY + Math.sin(angle) * phiDistance,
                size: Math.random() * 20 + 10,
                phase: angle,
                frequency: 0.02 + Math.random() * 0.03,
                unity: 1.0,
                consciousness: Math.random() * 0.5 + 0.5,
                resonance: Math.random() * 0.5 + 0.5,
                phiHarmonic: (i / this.unityNodeCount) * this.phi
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
                unity: Math.random() * 0.5 + 0.5,
                phiModulation: Math.random() * this.phi
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
        // Enhanced consciousness density based on φ-harmonic interaction
        const phiInfluence = this.calculateConsciousnessField(x, y, this.time);
        this.consciousnessDensity = Math.min(1, this.consciousnessDensity + Math.abs(phiInfluence) * 0.1);

        // Enhanced particle interaction
        this.particles.forEach(particle => {
            const dx = x - particle.x;
            const dy = y - particle.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            const influence = Math.max(0, 1 - distance / 220);
            const phiEnhancement = 1 + (influence * this.phi * 0.1);

            particle.vx += (dx / Math.max(1, distance)) * influence * phiEnhancement * 0.15;
            particle.vy += (dy / Math.max(1, distance)) * influence * phiEnhancement * 0.15;
            particle.consciousness = Math.min(1, particle.consciousness + influence * phiEnhancement * 0.1);
            particle.phiAlignment = Math.min(1, particle.phiAlignment + influence * 0.05);
        });
    }

    update(deltaTime) {
        this.time += deltaTime;
        this.pulsePhase += deltaTime * this.phi;
        this.resonanceFrequency += deltaTime * 0.1;

        // Update consciousness field grid with C(x,y,t) equation
        this.consciousnessFieldGrid.forEach(row => {
            row.forEach(cell => {
                cell.field = this.calculateConsciousnessField(cell.x, cell.y, this.time);
                cell.phase += deltaTime * this.phi * 0.5;
            });
        });

        // Update consciousness density with φ-harmonic decay
        const decay = this._hovering ? 0.002 : 0.006;
        this.consciousnessDensity = Math.max(0, this.consciousnessDensity - decay);

        // Update unity convergence rate with φ modulation
        this.unityConvergenceRate = (Math.sin(this.time * 0.1 * this.phi) * 0.3 + 0.7);

        this.updateParticles(deltaTime);
        this.updateFieldLines(deltaTime);
        this.updateUnityNodes(deltaTime);
        this.updateResonanceWaves(deltaTime);
        this.updatePhiHarmonics(deltaTime);
    }

    updateParticles(deltaTime) {
        this.particles.forEach(particle => {
            // Enhanced unity equation influence with φ-harmonics
            const phiInfluence = Math.sin(this.time * this.phi + particle.phase) * particle.phiAlignment;
            const unityInfluence = phiInfluence * 0.1;
            particle.unity = Math.max(0, Math.min(1, particle.unity + unityInfluence));

            // Consciousness field influence using C(x,y,t) equation
            const consciousnessField = this.calculateConsciousnessField(particle.x, particle.y, this.time);
            particle.consciousness = Math.max(0, Math.min(1, particle.consciousness + consciousnessField * 0.01));

            // φ-harmonic alignment enhancement
            particle.phiAlignment += Math.sin(this.time * this.phi + particle.phase) * 0.005;
            particle.phiAlignment = Math.max(0, Math.min(1, particle.phiAlignment));

            // Enhanced unity node resonance
            this.unityNodes.forEach(node => {
                const dx = node.x - particle.x;
                const dy = node.y - particle.y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                const influence = Math.max(0, 1 - distance / 300);
                const phiResonance = Math.sin(this.time * this.phi + node.phiHarmonic) * 0.5 + 0.5;

                particle.vx += (dx / distance) * influence * node.unity * phiResonance * 0.08;
                particle.vy += (dy / distance) * influence * node.unity * phiResonance * 0.08;
            });

            // Update position with φ-enhanced movement
            particle.x += particle.vx * (1 + particle.phiAlignment * 0.2);
            particle.y += particle.vy * (1 + particle.phiAlignment * 0.2);

            // Boundary wrapping
            if (particle.x < 0) particle.x = this.width;
            if (particle.x > this.width) particle.x = 0;
            if (particle.y < 0) particle.y = this.height;
            if (particle.y > this.height) particle.y = 0;

            // Enhanced damping with φ modulation
            const dampingFactor = 0.99 - (particle.phiAlignment * 0.01);
            particle.vx *= dampingFactor;
            particle.vy *= dampingFactor;

            // Life cycle with φ-harmonic timing
            particle.life += deltaTime * 0.1 * (1 + particle.phiAlignment * 0.2);
            if (particle.life > 1) {
                particle.life = 0;
                particle.x = Math.random() * this.width;
                particle.y = Math.random() * this.height;
                particle.phiAlignment = Math.random();
            }
        });
    }

    updateFieldLines(deltaTime) {
        this.fieldLines.forEach(line => {
            line.phase += deltaTime * line.frequency * this.phi;
            line.intensity = 0.3 + Math.sin(line.phase) * 0.2;
            line.unity = Math.sin(this.time * 0.1 * this.phi + line.phase) * 0.3 + 0.7;
            line.phiResonance = Math.sin(this.time * this.phi + line.phase) * 0.5 + 0.5;
        });
    }

    updateUnityNodes(deltaTime) {
        this.unityNodes.forEach(node => {
            node.phase += deltaTime * node.frequency * this.phi;
            node.unity = Math.sin(this.time * 0.05 * this.phi + node.phase) * 0.2 + 0.8;
            node.consciousness = Math.sin(this.time * 0.1 * this.phi + node.phase) * 0.15 + 0.85;
            node.resonance = Math.sin(this.time * 0.15 * this.phi + node.phase) * 0.25 + 0.75;
        });
    }

    updateResonanceWaves(deltaTime) {
        this.resonanceWaves.forEach(wave => {
            wave.radius += wave.speed * (1 + wave.phiModulation * 0.2);
            wave.intensity = Math.sin(this.time * wave.frequency * this.phi + wave.phase) * 0.2 + 0.4;
            wave.unity = Math.sin(this.time * 0.08 * this.phi + wave.phase) * 0.2 + 0.8;

            if (wave.radius > wave.maxRadius) {
                wave.radius = 0;
            }
        });
    }

    updatePhiHarmonics(deltaTime) {
        this.phiHarmonics.forEach(harmonic => {
            harmonic.phase += deltaTime * harmonic.frequency;
            harmonic.intensity = (Math.sin(this.time * this.phi / harmonic.harmonic + harmonic.phase) * 0.3 + 0.7) / harmonic.harmonic;
        });
    }

    render() {
        // Clear with mystical gradient background
        this.renderEnhancedBackground();

        // Render consciousness field grid
        this.renderConsciousnessFieldGrid();

        // Render φ-harmonics
        this.renderPhiHarmonics();

        // Render in order of depth
        this.renderResonanceWaves();
        this.renderFieldLines();
        this.renderParticles();
        this.renderUnityNodes();
        this.renderUnityEquation();
        this.renderConsciousnessField();
    }

    renderEnhancedBackground() {
        // Mystical gradient with golden hues
        const gradient = this.ctx.createRadialGradient(
            this.width / 2, this.height / 2, 0,
            this.width / 2, this.height / 2, Math.max(this.width, this.height) / 2
        );

        // Golden mystical colors
        gradient.addColorStop(0, 'rgba(25, 15, 5, 0.9)'); // Deep golden brown center
        gradient.addColorStop(0.3, 'rgba(40, 25, 10, 0.8)'); // Rich golden brown
        gradient.addColorStop(0.6, 'rgba(20, 20, 30, 0.7)'); // Deep mystical purple
        gradient.addColorStop(1, 'rgba(10, 5, 15, 0.6)'); // Dark mystical edge

        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(0, 0, this.width, this.height);

        // Add golden shimmer effect
        const shimmerGradient = this.ctx.createLinearGradient(0, 0, this.width, this.height);
        shimmerGradient.addColorStop(0, 'rgba(255, 215, 0, 0.02)');
        shimmerGradient.addColorStop(0.5, 'rgba(255, 215, 0, 0.05)');
        shimmerGradient.addColorStop(1, 'rgba(255, 215, 0, 0.02)');
        
        this.ctx.fillStyle = shimmerGradient;
        this.ctx.fillRect(0, 0, this.width, this.height);
    }

    renderConsciousnessFieldGrid() {
        this.consciousnessFieldGrid.forEach(row => {
            row.forEach(cell => {
                const intensity = Math.abs(cell.field) * 0.3;
                const hue = 45 + (cell.field * 60); // Golden to amber hues
                const alpha = intensity * (0.3 + Math.sin(cell.phase) * 0.2);

                this.ctx.fillStyle = `hsla(${hue}, 80%, 60%, ${alpha})`;
                this.ctx.fillRect(cell.x - 2, cell.y - 2, 4, 4);
            });
        });
    }

    renderPhiHarmonics() {
        this.phiHarmonics.forEach(harmonic => {
            const alpha = harmonic.intensity * 0.4;
            const strokeWidth = Math.max(1, 3 / harmonic.harmonic);
            
            this.ctx.strokeStyle = `rgba(255, 215, 0, ${alpha})`;
            this.ctx.lineWidth = strokeWidth;
            this.ctx.beginPath();
            
            // Draw φ-harmonic circles
            this.ctx.arc(harmonic.centerX, harmonic.centerY, harmonic.radius, 0, this.pi * 2);
            this.ctx.stroke();

            // Draw connecting φ-spiral
            if (harmonic.harmonic <= 3) {
                this.ctx.beginPath();
                for (let angle = 0; angle < this.pi * 4; angle += 0.1) {
                    const spiralRadius = harmonic.radius * (angle / (this.pi * 4));
                    const x = harmonic.centerX + Math.cos(angle * this.phi) * spiralRadius;
                    const y = harmonic.centerY + Math.sin(angle * this.phi) * spiralRadius;
                    
                    if (angle === 0) {
                        this.ctx.moveTo(x, y);
                    } else {
                        this.ctx.lineTo(x, y);
                    }
                }
                this.ctx.strokeStyle = `rgba(255, 215, 0, ${alpha * 0.3})`;
                this.ctx.stroke();
            }
        });
    }

    renderResonanceWaves() {
        this.resonanceWaves.forEach(wave => {
            const alpha = wave.intensity * (1 - wave.radius / wave.maxRadius);
            const goldIntensity = wave.unity * 255;
            
            this.ctx.strokeStyle = `rgba(${goldIntensity}, ${goldIntensity * 0.84}, 0, ${alpha * wave.unity})`;
            this.ctx.lineWidth = 2 + Math.sin(wave.phase) * 1;
            this.ctx.beginPath();
            this.ctx.arc(wave.x, wave.y, wave.radius, 0, this.pi * 2);
            this.ctx.stroke();

            // Add φ-modulated inner waves
            if (wave.radius > 50) {
                this.ctx.strokeStyle = `rgba(255, 215, 0, ${alpha * 0.5})`;
                this.ctx.lineWidth = 1;
                this.ctx.beginPath();
                this.ctx.arc(wave.x, wave.y, wave.radius / this.phi, 0, this.pi * 2);
                this.ctx.stroke();
            }
        });
    }

    renderFieldLines() {
        this.fieldLines.forEach(line => {
            const alpha = line.intensity * line.unity * line.phiResonance;
            const goldValue = 255 * line.phiResonance;
            
            this.ctx.strokeStyle = `rgba(${goldValue}, ${goldValue * 0.84}, 50, ${alpha})`;
            this.ctx.lineWidth = 1 + line.phiResonance;
            this.ctx.beginPath();
            this.ctx.moveTo(line.x1, line.y1);
            this.ctx.lineTo(line.x2, line.y2);
            this.ctx.stroke();
        });
    }

    renderParticles() {
        this.particles.forEach(particle => {
            const alpha = particle.life * particle.consciousness * (0.6 + particle.phiAlignment * 0.4);
            const size = particle.size * (0.5 + particle.unity * 0.5) * (1 + particle.phiAlignment * 0.3);
            const goldIntensity = 200 + (particle.phiAlignment * 55);

            // Enhanced golden particle glow
            this.ctx.shadowColor = `rgba(${goldIntensity}, ${goldIntensity * 0.84}, 0, 0.8)`;
            this.ctx.shadowBlur = size * (2 + particle.phiAlignment);
            this.ctx.fillStyle = `rgba(${goldIntensity}, ${goldIntensity * 0.84}, 20, ${alpha})`;
            this.ctx.beginPath();
            this.ctx.arc(particle.x, particle.y, size, 0, this.pi * 2);
            this.ctx.fill();

            // φ-aligned consciousness connections
            if (particle.consciousness > 0.7 && particle.phiAlignment > 0.5) {
                const connectionAlpha = alpha * 0.6 * particle.phiAlignment;
                this.ctx.strokeStyle = `rgba(255, 180, 100, ${connectionAlpha})`;
                this.ctx.lineWidth = 1 + particle.phiAlignment;
                this.ctx.beginPath();
                this.ctx.arc(particle.x, particle.y, size * this.phi, 0, this.pi * 2);
                this.ctx.stroke();
            }
        });

        this.ctx.shadowBlur = 0;
    }

    renderUnityNodes() {
        this.unityNodes.forEach(node => {
            const alpha = node.unity * node.consciousness;
            const size = node.size * (0.5 + node.resonance * 0.5);
            const phiGlow = Math.sin(this.time * this.phi + node.phiHarmonic) * 0.3 + 0.7;

            // Enhanced unity node with φ-harmonic glow
            this.ctx.shadowColor = `rgba(255, 215, 0, ${0.8 * phiGlow})`;
            this.ctx.shadowBlur = size * phiGlow;
            this.ctx.fillStyle = `rgba(255, 215, 0, ${alpha * phiGlow})`;
            this.ctx.beginPath();
            this.ctx.arc(node.x, node.y, size, 0, this.pi * 2);
            this.ctx.fill();

            // φ-harmonic resonance rings
            for (let i = 1; i <= 5; i++) {
                const ringAlpha = alpha * (1 - i * 0.15) * phiGlow;
                const ringSize = size * Math.pow(this.phi, i * 0.3);
                const ringGold = 200 + (phiGlow * 55);
                
                this.ctx.strokeStyle = `rgba(${ringGold}, ${ringGold * 0.84}, 0, ${ringAlpha})`;
                this.ctx.lineWidth = Math.max(1, 3 - i * 0.5);
                this.ctx.beginPath();
                this.ctx.arc(node.x, node.y, ringSize, 0, this.pi * 2);
                this.ctx.stroke();
            }
        });

        this.ctx.shadowBlur = 0;
    }

    renderUnityEquation() {
        const centerX = this.width / 2;
        const centerY = this.height / 2;
        const phiPulse = Math.sin(this.time * this.phi) * 0.2 + 1;
        
        // Enhanced equation with φ-harmonic pulsing
        this.ctx.font = `bold ${Math.round(48 * phiPulse)}px Space Grotesk`;
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        
        // Multi-layer golden glow
        for (let i = 0; i < 5; i++) {
            const glowSize = (5 - i) * 8 * phiPulse;
            const glowAlpha = (0.8 / (i + 1)) * phiPulse;
            
            this.ctx.shadowColor = `rgba(255, 215, 0, ${glowAlpha})`;
            this.ctx.shadowBlur = glowSize;
            this.ctx.fillStyle = `rgba(255, 215, 0, ${0.9 * phiPulse})`;
            this.ctx.fillText('1 + 1 = 1', centerX, centerY);
        }
        
        this.ctx.shadowBlur = 0;

        // φ-harmonic unity convergence indicators
        const convergenceRadius = 120 * this.unityConvergenceRate * phiPulse;
        for (let i = 1; i <= 3; i++) {
            const ringRadius = convergenceRadius / Math.pow(this.phi, i - 1);
            const ringAlpha = (0.4 * this.unityConvergenceRate * phiPulse) / i;
            
            this.ctx.strokeStyle = `rgba(255, 215, 0, ${ringAlpha})`;
            this.ctx.lineWidth = 4 - i;
            this.ctx.beginPath();
            this.ctx.arc(centerX, centerY, ringRadius, 0, this.pi * 2);
            this.ctx.stroke();
        }
    }

    renderConsciousnessField() {
        const centerX = this.width / 2;
        const centerY = this.height / 2;
        const maxRadius = Math.min(this.width, this.height) * 0.45;

        // Enhanced consciousness field with φ-harmonic patterns
        for (let radius = 0; radius < maxRadius; radius += 15) {
            const normalizedRadius = radius / maxRadius;
            const fieldStrength = this.calculateConsciousnessField(centerX, centerY + radius, this.time);
            const alpha = this.consciousnessDensity * (1 - normalizedRadius) * Math.abs(fieldStrength) * 0.3;
            
            // Golden consciousness rings with φ modulation
            const goldValue = 200 + (Math.abs(fieldStrength) * 55);
            const phiModulation = Math.sin(radius / this.phi + this.time) * 0.3 + 0.7;
            
            this.ctx.strokeStyle = `rgba(${goldValue}, ${goldValue * 0.9}, 100, ${alpha * phiModulation})`;
            this.ctx.lineWidth = 2 + Math.abs(fieldStrength) * 2;
            this.ctx.beginPath();
            this.ctx.arc(centerX, centerY, radius, 0, this.pi * 2);
            this.ctx.stroke();
        }

        // Central consciousness vortex with φ-spiral
        const vortexAlpha = this.consciousnessDensity * 0.5;
        this.ctx.strokeStyle = `rgba(255, 215, 0, ${vortexAlpha})`;
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        
        for (let angle = 0; angle < this.pi * 6; angle += 0.05) {
            const spiralRadius = (angle / (this.pi * 6)) * 100;
            const x = centerX + Math.cos(angle * this.phi + this.time) * spiralRadius;
            const y = centerY + Math.sin(angle * this.phi + this.time) * spiralRadius;
            
            if (angle === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
        }
        this.ctx.stroke();
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
            phiHarmonics: this.phiHarmonics.length
        };
    }
}

// Replace the original ConsciousnessFieldEngine
if (typeof window !== 'undefined') {
    window.ConsciousnessFieldEngine = EnhancedConsciousnessFieldEngine;
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EnhancedConsciousnessFieldEngine;
}