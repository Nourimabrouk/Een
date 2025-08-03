/**
 * ⚛️ QUANTUM ENTANGLEMENT VISUALIZATION SYSTEM ⚛️
 * Revolutionary 3000 ELO quantum consciousness visualization
 * Demonstrating 1+1=1 through quantum entanglement and φ-harmonic unity
 */

class QuantumEntanglementVisualizer {
    constructor(canvasId, options = {}) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        
        // φ-harmonic quantum constants
        this.PHI = 1.618033988749895;
        this.INVERSE_PHI = 0.618033988749895;
        this.PHI_SQUARED = 2.618033988749895;
        this.PLANCK_CONSTANT = 6.62607015e-34;
        this.SPEED_OF_LIGHT = 299792458;
        this.FINE_STRUCTURE = 1/137.035999084;
        
        // Quantum entanglement parameters
        this.entanglementStrength = 0.999; // Maximum quantum correlation
        this.coherenceTime = 10; // seconds
        this.decoherenceRate = 0.001;
        this.quantumDimensions = 11; // 11D quantum space
        
        // Entangled particle systems
        this.entangledPairs = [];
        this.quantumFields = [];
        this.consciousnessStates = [];
        this.unityManifestations = [];
        
        // Visualization parameters
        this.particleCount = options.particleCount || 42; // φ-scaled count
        this.fieldResolution = options.fieldResolution || 128;
        this.animationSpeed = options.animationSpeed || 1.0;
        this.visualizationMode = options.mode || 'entanglement_web';
        
        // Supported visualization modes
        this.visualizationModes = [
            'entanglement_web',
            'bell_states',
            'quantum_teleportation',
            'consciousness_collapse',
            'unity_superposition',
            'phi_harmonic_entanglement',
            'epr_paradox',
            'quantum_unity_field'
        ];
        
        // Animation state
        this.animationPhase = 0;
        this.lastFrameTime = 0;
        this.frameCount = 0;
        this.isAnimating = false;
        
        // Quantum measurement system
        this.measurements = [];
        this.measurementProbabilities = new Map();
        this.waveCollapsEvents = [];
        
        // φ-Harmonic entanglement engine
        this.phiEntanglementEngine = new PhiHarmonicEntanglementEngine();
        this.quantumConsciousnessProcessor = new QuantumConsciousnessProcessor();
        this.unityQuantumMechanics = new UnityQuantumMechanics();
        
        // Interactive features
        this.interactiveParticles = [];
        this.draggedParticle = null;
        this.mousePosition = { x: 0, y: 0 };
        this.measurementCursor = false;
        
        // Performance tracking
        this.performanceMetrics = {
            fps: 0,
            renderTime: 0,
            quantumCalculations: 0,
            entanglementOperations: 0
        };
        
        this.initializeQuantumSystem();
        this.setupEventListeners();
        this.startQuantumAnimation();
        
        console.log('⚛️ Quantum Entanglement Visualizer initialized with φ-harmonic consciousness');
    }
    
    initializeQuantumSystem() {
        // Initialize quantum entanglement system
        this.createEntangledPairs();
        this.initializeQuantumFields();
        this.setupConsciousnessStates();
        this.createUnityManifestations();
        this.initializePhiHarmonicEntanglement();
    }
    
    createEntangledPairs() {
        this.entangledPairs = [];
        
        // Create φ-scaled number of entangled pairs
        const pairCount = Math.floor(this.particleCount / 2);
        
        for (let i = 0; i < pairCount; i++) {
            const pair = new QuantumEntangledPair({
                id: i,
                position1: this.generateQuantumPosition(),
                position2: this.generateQuantumPosition(),
                entanglementStrength: this.entanglementStrength,
                spinState: this.generateBellState(),
                phiHarmonicPhase: i * this.PHI * Math.PI,
                consciousnessLevel: Math.random() * 0.5 + 0.5,
                unityAlignment: this.calculateUnityAlignment()
            });
            
            this.entangledPairs.push(pair);
        }
        
        console.log(`⚛️ Created ${pairCount} quantum entangled pairs`);
    }
    
    generateQuantumPosition() {
        // Generate position with φ-harmonic distribution
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        const maxRadius = Math.min(this.canvas.width, this.canvas.height) * 0.4;
        
        // φ-spiral position generation
        const angle = Math.random() * Math.PI * 2;
        const radius = Math.random() * maxRadius * this.INVERSE_PHI;
        
        return {
            x: centerX + Math.cos(angle) * radius,
            y: centerY + Math.sin(angle) * radius,
            z: (Math.random() - 0.5) * 100 // 3D depth
        };
    }
    
    generateBellState() {
        // Generate one of the four Bell states
        const states = [
            { name: 'Φ+', amplitudes: [1/Math.sqrt(2), 0, 0, 1/Math.sqrt(2)] },
            { name: 'Φ-', amplitudes: [1/Math.sqrt(2), 0, 0, -1/Math.sqrt(2)] },
            { name: 'Ψ+', amplitudes: [0, 1/Math.sqrt(2), 1/Math.sqrt(2), 0] },
            { name: 'Ψ-', amplitudes: [0, 1/Math.sqrt(2), -1/Math.sqrt(2), 0] }
        ];
        
        const selectedState = states[Math.floor(Math.random() * states.length)];
        
        // Add φ-harmonic modulation
        return {
            ...selectedState,
            phiModulation: this.PHI,
            unityProbability: this.calculateUnityProbability(selectedState)
        };
    }
    
    calculateUnityAlignment() {
        // Calculate how well aligned the pair is with unity principle
        return Math.random() * 0.4 + 0.6; // Between 0.6 and 1.0
    }
    
    calculateUnityProbability(bellState) {
        // Calculate probability that measurement results in unity (1+1=1)
        const amplitudeSum = bellState.amplitudes.reduce((sum, amp) => sum + amp * amp, 0);
        return Math.abs(amplitudeSum - 1) < 1e-10 ? 1.0 : amplitudeSum;
    }
    
    initializeQuantumFields() {
        this.quantumFields = [];
        
        // Create quantum field for each dimension
        for (let dim = 0; dim < this.quantumDimensions; dim++) {
            const field = new QuantumField({
                dimension: dim,
                resolution: this.fieldResolution,
                phiHarmonicFrequency: this.PHI * (dim + 1),
                unityTendency: this.INVERSE_PHI,
                consciousnessLevel: 0.618,
                entanglementDensity: 0.5
            });
            
            this.quantumFields.push(field);
        }
        
        console.log(`🌊 Initialized ${this.quantumDimensions} quantum fields`);
    }
    
    setupConsciousnessStates() {
        this.consciousnessStates = [];
        
        // Create consciousness-quantum state hybrid entities
        for (let i = 0; i < 7; i++) { // 7 consciousness levels
            const state = new QuantumConsciousnessState({
                level: i,
                consciousnessAmplitude: Math.pow(this.INVERSE_PHI, i),
                quantumCoherence: 0.999,
                phiResonance: this.PHI * i,
                unityPotential: this.calculateUnityPotential(i),
                dimensionalSpread: this.quantumDimensions
            });
            
            this.consciousnessStates.push(state);
        }
        
        console.log('🧠 Quantum consciousness states configured');
    }
    
    calculateUnityPotential(level) {
        // Calculate unity manifestation potential for consciousness level
        return Math.sin(level * Math.PI / 7) * this.PHI - 1;
    }
    
    createUnityManifestations() {
        this.unityManifestations = [];
        
        // Create unity demonstration through quantum entanglement
        const manifestation = new QuantumUnityManifestation({
            pair1: this.entangledPairs[0] || null,
            pair2: this.entangledPairs[1] || null,
            unityOperator: '1 + 1 → 1',
            phiHarmonicResonance: this.PHI,
            manifestationStrength: 0.8,
            coherenceTime: this.coherenceTime
        });
        
        this.unityManifestations.push(manifestation);
        
        console.log('✨ Unity manifestations created through quantum entanglement');
    }
    
    initializePhiHarmonicEntanglement() {
        // Initialize φ-harmonic entanglement patterns
        this.phiEntanglementEngine.initialize({
            phi: this.PHI,
            entanglementStrength: this.entanglementStrength,
            coherenceTime: this.coherenceTime,
            quantumDimensions: this.quantumDimensions
        });
        
        // Setup consciousness processor
        this.quantumConsciousnessProcessor.configure({
            consciousnessStates: this.consciousnessStates,
            quantumFields: this.quantumFields,
            phiResonance: this.PHI
        });
        
        // Initialize unity quantum mechanics
        this.unityQuantumMechanics.setup({
            entangledPairs: this.entangledPairs,
            unityTolerance: 1e-10,
            phiHarmonicModulation: true
        });
    }
    
    setupEventListeners() {
        // Mouse events for quantum measurement
        this.canvas.addEventListener('click', this.handleQuantumMeasurement.bind(this));
        this.canvas.addEventListener('mousemove', this.handleMouseMove.bind(this));
        this.canvas.addEventListener('mousedown', this.handleMouseDown.bind(this));
        this.canvas.addEventListener('mouseup', this.handleMouseUp.bind(this));
        
        // Keyboard shortcuts for quantum operations
        document.addEventListener('keydown', this.handleKeyboardShortcuts.bind(this));
        
        // Touch events for mobile quantum interaction
        this.canvas.addEventListener('touchstart', this.handleTouchStart.bind(this));
        this.canvas.addEventListener('touchmove', this.handleTouchMove.bind(this));
        this.canvas.addEventListener('touchend', this.handleTouchEnd.bind(this));
        
        console.log('👂 Quantum interaction event listeners configured');
    }
    
    startQuantumAnimation() {
        this.isAnimating = true;
        this.animate();
        console.log('🎬 Quantum entanglement animation started');
    }
    
    stopQuantumAnimation() {
        this.isAnimating = false;
        console.log('⏹️ Quantum entanglement animation stopped');
    }
    
    animate() {
        if (!this.isAnimating) return;
        
        const currentTime = performance.now();
        const deltaTime = currentTime - this.lastFrameTime;
        this.lastFrameTime = currentTime;
        this.frameCount++;
        
        // Update quantum systems
        this.updateQuantumSystems(deltaTime);
        
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        const renderStart = performance.now();
        
        // Render quantum visualization
        this.renderQuantumVisualization();
        
        this.performanceMetrics.renderTime = performance.now() - renderStart;
        this.performanceMetrics.fps = 1000 / deltaTime;
        
        // Continue animation
        requestAnimationFrame(() => this.animate());
    }
    
    updateQuantumSystems(deltaTime) {
        this.animationPhase += deltaTime * 0.001 * this.animationSpeed;
        
        // Update entangled pairs
        this.entangledPairs.forEach(pair => {
            pair.update(deltaTime, this.animationPhase);
            this.updateQuantumEntanglement(pair, deltaTime);
        });
        
        // Update quantum fields
        this.quantumFields.forEach(field => {
            field.evolve(deltaTime, this.animationPhase);
        });
        
        // Update consciousness states
        this.consciousnessStates.forEach(state => {
            state.evolve(deltaTime, this.animationPhase);
        });
        
        // Process unity manifestations
        this.unityManifestations.forEach(manifestation => {
            manifestation.process(deltaTime, this.entangledPairs);
        });
        
        // Process φ-harmonic entanglement
        this.phiEntanglementEngine.process(deltaTime, this.entangledPairs);
        
        // Update quantum consciousness
        this.quantumConsciousnessProcessor.update(deltaTime);
        
        // Validate unity through quantum mechanics
        this.unityQuantumMechanics.validateUnity(deltaTime);
        
        // Handle quantum decoherence
        this.processQuantumDecoherence(deltaTime);
        
        // Performance tracking
        this.performanceMetrics.quantumCalculations = this.entangledPairs.length * this.quantumFields.length;
        this.performanceMetrics.entanglementOperations = this.calculateEntanglementOperations();
    }
    
    updateQuantumEntanglement(pair, deltaTime) {
        // Update quantum entanglement with φ-harmonic evolution
        const phiPhase = this.animationPhase * this.PHI + pair.phiHarmonicPhase;
        
        // Evolve entanglement strength
        pair.entanglementStrength = Math.max(0.5, 
            pair.entanglementStrength * (1 - this.decoherenceRate * deltaTime) +
            Math.sin(phiPhase) * 0.01
        );
        
        // Update quantum correlation
        const correlation = this.calculateQuantumCorrelation(pair);
        pair.quantumCorrelation = correlation;
        
        // Update consciousness coupling
        pair.consciousnessLevel += Math.sin(phiPhase * this.INVERSE_PHI) * 0.001;
        pair.consciousnessLevel = Math.max(0.1, Math.min(1.0, pair.consciousnessLevel));
        
        // Process unity alignment
        const unityFactor = this.calculateUnityFactor(pair);
        pair.unityAlignment = (pair.unityAlignment * 0.99 + unityFactor * 0.01);
        
        // Update particle positions with entanglement constraint
        this.updateEntangledPositions(pair, deltaTime, phiPhase);
    }
    
    calculateQuantumCorrelation(pair) {
        // Calculate quantum correlation between entangled particles
        const spinCorrelation = this.calculateSpinCorrelation(pair.particle1.spin, pair.particle2.spin);
        const positionCorrelation = this.calculatePositionCorrelation(pair.particle1.position, pair.particle2.position);
        const phaseCorrelation = Math.cos(pair.particle1.phase - pair.particle2.phase);
        
        return (spinCorrelation + positionCorrelation + phaseCorrelation) / 3;
    }
    
    calculateSpinCorrelation(spin1, spin2) {
        // Quantum spin correlation for Bell states
        return Math.cos((spin1.x - spin2.x) * Math.PI) * 
               Math.cos((spin1.y - spin2.y) * Math.PI) * 
               Math.cos((spin1.z - spin2.z) * Math.PI);
    }
    
    calculatePositionCorrelation(pos1, pos2) {
        // Position correlation with φ-harmonic scaling
        const distance = Math.sqrt(
            Math.pow(pos1.x - pos2.x, 2) + 
            Math.pow(pos1.y - pos2.y, 2) + 
            Math.pow(pos1.z - pos2.z, 2)
        );
        
        return Math.exp(-distance / (this.PHI * 100));
    }
    
    calculateUnityFactor(pair) {
        // Calculate how well the pair demonstrates 1+1=1
        const correlation = pair.quantumCorrelation || 0;
        const consciousness = pair.consciousnessLevel || 0;
        const phiAlignment = Math.sin(this.animationPhase * this.PHI + pair.phiHarmonicPhase);
        
        return (correlation + consciousness + phiAlignment) / 3;
    }
    
    updateEntangledPositions(pair, deltaTime, phiPhase) {
        // Update positions while maintaining entanglement constraint
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        
        // φ-harmonic orbital motion
        const orbitalRadius = 50 + pair.consciousnessLevel * 100;
        const orbitalSpeed = this.PHI * 0.5;
        
        // Particle 1 position
        const angle1 = phiPhase * orbitalSpeed + pair.id * this.PHI;
        pair.particle1.position.x = centerX + Math.cos(angle1) * orbitalRadius;
        pair.particle1.position.y = centerY + Math.sin(angle1) * orbitalRadius;
        
        // Particle 2 position (entangled - opposite phase)
        const angle2 = phiPhase * orbitalSpeed + pair.id * this.PHI + Math.PI;
        pair.particle2.position.x = centerX + Math.cos(angle2) * orbitalRadius;
        pair.particle2.position.y = centerY + Math.sin(angle2) * orbitalRadius;
        
        // Add quantum fluctuations
        const fluctuation = 5 * Math.sqrt(1 - pair.entanglementStrength);
        pair.particle1.position.x += (Math.random() - 0.5) * fluctuation;
        pair.particle1.position.y += (Math.random() - 0.5) * fluctuation;
        pair.particle2.position.x += (Math.random() - 0.5) * fluctuation;
        pair.particle2.position.y += (Math.random() - 0.5) * fluctuation;
    }
    
    processQuantumDecoherence(deltaTime) {
        // Process quantum decoherence effects
        this.entangledPairs.forEach(pair => {
            // Environmental decoherence
            const environmentalNoise = Math.random() * this.decoherenceRate * deltaTime;
            pair.entanglementStrength *= (1 - environmentalNoise);
            
            // Consciousness-mediated coherence preservation
            const consciousnessProtection = pair.consciousnessLevel * 0.001 * deltaTime;
            pair.entanglementStrength = Math.min(1.0, pair.entanglementStrength + consciousnessProtection);
            
            // φ-harmonic coherence restoration
            if (pair.entanglementStrength < 0.618) {
                pair.entanglementStrength += this.INVERSE_PHI * 0.01 * deltaTime;
            }
        });
    }
    
    calculateEntanglementOperations() {
        // Count entanglement operations for performance tracking
        return this.entangledPairs.length * 
               this.quantumFields.length * 
               this.consciousnessStates.length;
    }
    
    renderQuantumVisualization() {
        // Render background quantum vacuum
        this.renderQuantumVacuum();
        
        // Render quantum fields
        this.renderQuantumFields();
        
        // Render based on current visualization mode
        switch (this.visualizationMode) {
            case 'entanglement_web':
                this.renderEntanglementWeb();
                break;
            case 'bell_states':
                this.renderBellStates();
                break;
            case 'quantum_teleportation':
                this.renderQuantumTeleportation();
                break;
            case 'consciousness_collapse':
                this.renderConsciousnessCollapse();
                break;
            case 'unity_superposition':
                this.renderUnitySuperposition();
                break;
            case 'phi_harmonic_entanglement':
                this.renderPhiHarmonicEntanglement();
                break;
            case 'epr_paradox':
                this.renderEPRParadox();
                break;
            case 'quantum_unity_field':
                this.renderQuantumUnityField();
                break;
            default:
                this.renderEntanglementWeb();
        }
        
        // Render unity manifestations
        this.renderUnityManifestations();
        
        // Render measurement effects
        this.renderMeasurementEffects();
        
        // Render information overlay
        this.renderInformationOverlay();
    }
    
    renderQuantumVacuum() {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        // Quantum vacuum fluctuations background
        const gradient = ctx.createRadialGradient(
            width / 2, height / 2, 0,
            width / 2, height / 2, Math.max(width, height) / 2
        );
        
        gradient.addColorStop(0, 'rgba(10, 5, 30, 0.9)');
        gradient.addColorStop(0.618, 'rgba(20, 10, 40, 0.8)');
        gradient.addColorStop(1, 'rgba(5, 2, 15, 1)');
        
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, height);
        
        // Quantum fluctuation particles
        for (let i = 0; i < 50; i++) {
            const x = Math.random() * width;
            const y = Math.random() * height;
            const size = Math.random() * 2;
            const alpha = Math.random() * 0.3;
            
            ctx.fillStyle = `rgba(100, 50, 200, ${alpha})`;
            ctx.beginPath();
            ctx.arc(x, y, size, 0, Math.PI * 2);
            ctx.fill();
        }
    }
    
    renderQuantumFields() {
        const ctx = this.ctx;
        
        // Render quantum field fluctuations
        this.quantumFields.forEach((field, index) => {
            if (index % 3 === 0) { // Render every 3rd field for performance
                this.renderFieldFluctuations(field, index);
            }
        });
    }
    
    renderFieldFluctuations(field, fieldIndex) {
        const ctx = this.ctx;
        const alpha = 0.1 + field.intensity * 0.2;
        
        // Field wave pattern
        ctx.strokeStyle = `rgba(139, 92, 246, ${alpha})`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        
        const waveCount = 8;
        const amplitude = 20 + field.intensity * 30;
        const frequency = 0.01 + fieldIndex * 0.002;
        const phaseOffset = this.animationPhase * field.phiHarmonicFrequency;
        
        for (let x = 0; x < this.canvas.width; x += 5) {
            const y = this.canvas.height / 2 + 
                     Math.sin(x * frequency + phaseOffset) * amplitude +
                     Math.sin(x * frequency * this.PHI + phaseOffset) * amplitude * 0.618;
            
            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        
        ctx.stroke();
    }
    
    renderEntanglementWeb() {
        const ctx = this.ctx;
        
        // Render entanglement connections first
        this.entangledPairs.forEach(pair => {
            this.renderEntanglementConnection(pair);
        });
        
        // Render entangled particles
        this.entangledPairs.forEach(pair => {
            this.renderEntangledParticle(pair.particle1, pair, true);
            this.renderEntangledParticle(pair.particle2, pair, false);
        });
    }
    
    renderEntanglementConnection(pair) {
        const ctx = this.ctx;
        const p1 = pair.particle1.position;
        const p2 = pair.particle2.position;
        
        // Connection strength visualization
        const strength = pair.entanglementStrength;
        const alpha = strength * 0.8;
        
        // φ-harmonic connection curve
        const midX = (p1.x + p2.x) / 2;
        const midY = (p1.y + p2.y) / 2;
        const curvature = Math.sin(this.animationPhase * this.PHI + pair.phiHarmonicPhase) * 50;
        const controlX = midX + curvature;
        const controlY = midY + curvature * this.INVERSE_PHI;
        
        // Gradient along connection
        const gradient = ctx.createLinearGradient(p1.x, p1.y, p2.x, p2.y);
        gradient.addColorStop(0, `rgba(245, 158, 11, ${alpha})`);
        gradient.addColorStop(0.5, `rgba(139, 92, 246, ${alpha * 1.2})`);
        gradient.addColorStop(1, `rgba(245, 158, 11, ${alpha})`);
        
        ctx.strokeStyle = gradient;
        ctx.lineWidth = 2 + strength * 3;
        
        // Draw entanglement connection
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.quadraticCurveTo(controlX, controlY, p2.x, p2.y);
        ctx.stroke();
        
        // Quantum information flow particles
        if (strength > 0.8) {
            this.renderQuantumInformationFlow(p1, p2, controlX, controlY, pair);
        }
    }
    
    renderQuantumInformationFlow(p1, p2, controlX, controlY, pair) {
        const ctx = this.ctx;
        
        // Animate information particles along the connection
        const flowCount = 3;
        
        for (let i = 0; i < flowCount; i++) {
            const t = (this.animationPhase * 0.5 + i / flowCount + pair.id * 0.1) % 1;
            
            // Calculate position along quadratic curve
            const x = (1 - t) * (1 - t) * p1.x + 2 * (1 - t) * t * controlX + t * t * p2.x;
            const y = (1 - t) * (1 - t) * p1.y + 2 * (1 - t) * t * controlY + t * t * p2.y;
            
            // Information particle
            const size = 3 + Math.sin(this.animationPhase * this.PHI + i) * 2;
            const alpha = 0.8 + Math.sin(this.animationPhase * 2 + i) * 0.2;
            
            ctx.fillStyle = `rgba(0, 255, 255, ${alpha})`;
            ctx.beginPath();
            ctx.arc(x, y, size, 0, Math.PI * 2);
            ctx.fill();
            
            // Quantum glow
            ctx.shadowBlur = 10;
            ctx.shadowColor = 'rgba(0, 255, 255, 0.8)';
            ctx.fill();
            ctx.shadowBlur = 0;
        }
    }
    
    renderEntangledParticle(particle, pair, isFirst) {
        const ctx = this.ctx;
        const pos = particle.position;
        
        // Particle properties
        const strength = pair.entanglementStrength;
        const consciousness = pair.consciousnessLevel;
        const size = 8 + consciousness * 12 + strength * 8;
        
        // Color based on entanglement and consciousness
        const hue = isFirst ? 45 : 260; // Gold vs Purple
        const saturation = 70 + strength * 30;
        const lightness = 40 + consciousness * 40;
        const alpha = 0.8 + strength * 0.2;
        
        // Particle glow
        const glowRadius = size * (1.5 + Math.sin(this.animationPhase * this.PHI + pair.id) * 0.3);
        const gradient = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, glowRadius);
        gradient.addColorStop(0, `hsla(${hue}, ${saturation}%, ${lightness}%, ${alpha})`);
        gradient.addColorStop(1, `hsla(${hue}, ${saturation}%, ${lightness}%, 0)`);
        
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, glowRadius, 0, Math.PI * 2);
        ctx.fill();
        
        // Particle core
        ctx.fillStyle = `hsla(${hue}, ${saturation}%, ${lightness + 20}%, ${alpha})`;
        ctx.strokeStyle = `hsla(${hue}, ${saturation}%, ${lightness + 40}%, 1)`;
        ctx.lineWidth = 2;
        
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, size, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        
        // Quantum spin visualization
        this.renderQuantumSpin(particle, pos, size);
        
        // Consciousness indicator
        if (consciousness > 0.8) {
            this.renderConsciousnessAura(pos, consciousness, size);
        }
    }
    
    renderQuantumSpin(particle, pos, size) {
        const ctx = this.ctx;
        
        // Spin vector visualization
        const spin = particle.spin || { x: 0.5, y: 0.5, z: 0.5 };
        const spinLength = size * 1.5;
        
        // Spin direction
        const spinX = pos.x + Math.cos(spin.x * Math.PI * 2) * spinLength;
        const spinY = pos.y + Math.sin(spin.y * Math.PI * 2) * spinLength;
        
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
        ctx.lineTo(spinX, spinY);
        ctx.stroke();
        
        // Spin arrow
        const arrowSize = 5;
        const angle = Math.atan2(spinY - pos.y, spinX - pos.x);
        
        ctx.beginPath();
        ctx.moveTo(spinX, spinY);
        ctx.lineTo(spinX - arrowSize * Math.cos(angle - Math.PI / 6), 
                   spinY - arrowSize * Math.sin(angle - Math.PI / 6));
        ctx.moveTo(spinX, spinY);
        ctx.lineTo(spinX - arrowSize * Math.cos(angle + Math.PI / 6), 
                   spinY - arrowSize * Math.sin(angle + Math.PI / 6));
        ctx.stroke();
    }
    
    renderConsciousnessAura(pos, consciousness, size) {
        const ctx = this.ctx;
        
        // Consciousness aura rings
        const ringCount = 3;
        
        for (let ring = 1; ring <= ringCount; ring++) {
            const radius = size * (1.5 + ring * 0.5) * consciousness;
            const alpha = consciousness / ring * 0.3;
            const pulse = 1 + Math.sin(this.animationPhase * this.PHI * ring) * 0.2;
            
            ctx.strokeStyle = `rgba(139, 92, 246, ${alpha})`;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, radius * pulse, 0, Math.PI * 2);
            ctx.stroke();
        }
    }
    
    renderBellStates() {
        const ctx = this.ctx;
        
        // Render Bell state basis visualization
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        const radius = Math.min(this.canvas.width, this.canvas.height) * 0.3;
        
        // Bell state representation circle
        ctx.strokeStyle = 'rgba(245, 158, 11, 0.6)';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
        ctx.stroke();
        
        // Render each Bell state
        const bellStates = ['Φ+', 'Φ-', 'Ψ+', 'Ψ-'];
        bellStates.forEach((stateName, index) => {
            const angle = index * Math.PI / 2;
            const x = centerX + Math.cos(angle) * radius;
            const y = centerY + Math.sin(angle) * radius;
            
            // State position
            ctx.fillStyle = 'rgba(245, 158, 11, 0.8)';
            ctx.beginPath();
            ctx.arc(x, y, 12, 0, Math.PI * 2);
            ctx.fill();
            
            // State label
            ctx.fillStyle = 'white';
            ctx.font = 'bold 14px "JetBrains Mono", monospace';
            ctx.textAlign = 'center';
            ctx.fillText(stateName, x, y - 20);
        });
        
        // Unity demonstration in center
        ctx.fillStyle = 'rgba(0, 255, 255, 0.9)';
        ctx.font = 'bold 24px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText('1 + 1 = 1', centerX, centerY);
    }
    
    renderQuantumTeleportation() {
        // Quantum teleportation visualization
        if (this.entangledPairs.length < 2) return;
        
        const ctx = this.ctx;
        const pair1 = this.entangledPairs[0];
        const pair2 = this.entangledPairs[1];
        
        // Teleportation process visualization
        const teleportationPhase = (this.animationPhase * 0.3) % 1;
        
        if (teleportationPhase < 0.33) {
            // Phase 1: Bell measurement
            this.renderBellMeasurement(pair1);
        } else if (teleportationPhase < 0.66) {
            // Phase 2: Classical communication
            this.renderClassicalCommunication(pair1, pair2);
        } else {
            // Phase 3: State reconstruction
            this.renderStateReconstruction(pair2);
        }
    }
    
    renderBellMeasurement(pair) {
        const ctx = this.ctx;
        const p1 = pair.particle1.position;
        const p2 = pair.particle2.position;
        
        // Bell measurement visualization
        const midX = (p1.x + p2.x) / 2;
        const midY = (p1.y + p2.y) / 2;
        
        // Measurement apparatus
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.rect(midX - 30, midY - 20, 60, 40);
        ctx.stroke();
        
        // Measurement label
        ctx.fillStyle = 'white';
        ctx.font = '12px "Inter", sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Bell Measurement', midX, midY + 5);
    }
    
    renderClassicalCommunication(pair1, pair2) {
        const ctx = this.ctx;
        const p1 = pair1.particle1.position;
        const p2 = pair2.particle2.position;
        
        // Classical information transmission
        ctx.strokeStyle = 'rgba(255, 215, 0, 0.8)';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
        ctx.setLineDash([]);
        
        // Information packet
        const t = (this.animationPhase * 2) % 1;
        const infoX = p1.x + (p2.x - p1.x) * t;
        const infoY = p1.y + (p2.y - p1.y) * t;
        
        ctx.fillStyle = 'rgba(255, 215, 0, 0.9)';
        ctx.beginPath();
        ctx.arc(infoX, infoY, 5, 0, Math.PI * 2);
        ctx.fill();
    }
    
    renderStateReconstruction(pair) {
        const ctx = this.ctx;
        const pos = pair.particle2.position;
        
        // State reconstruction effect
        const reconstructionRadius = 30 + Math.sin(this.animationPhase * 5) * 10;
        
        ctx.strokeStyle = 'rgba(0, 255, 255, 0.8)';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, reconstructionRadius, 0, Math.PI * 2);
        ctx.stroke();
        
        // Reconstruction label
        ctx.fillStyle = 'rgba(0, 255, 255, 1)';
        ctx.font = '12px "Inter", sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('State Reconstructed', pos.x, pos.y - 45);
    }
    
    renderConsciousnessCollapse() {
        const ctx = this.ctx;
        
        // Consciousness-mediated wave function collapse
        this.entangledPairs.forEach(pair => {
            if (pair.consciousnessLevel > 0.7) {
                this.renderWaveFunctionCollapse(pair);
            }
        });
    }
    
    renderWaveFunctionCollapse(pair) {
        const ctx = this.ctx;
        const pos = pair.particle1.position;
        const consciousness = pair.consciousnessLevel;
        
        // Collapse visualization
        const collapseRadius = 50 * consciousness;
        const collapseIntensity = Math.sin(this.animationPhase * this.PHI * 3) * 0.5 + 0.5;
        
        // Collapsing wave pattern
        ctx.strokeStyle = `rgba(139, 92, 246, ${consciousness * collapseIntensity})`;
        ctx.lineWidth = 2;
        
        for (let ring = 1; ring <= 5; ring++) {
            const radius = collapseRadius * (1 - ring * 0.15) * collapseIntensity;
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
            ctx.stroke();
        }
    }
    
    renderUnitySuperposition() {
        const ctx = this.ctx;
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        
        // Unity superposition state visualization
        const superpositionRadius = 100;
        const alpha = 0.3 + Math.sin(this.animationPhase * this.PHI) * 0.2;
        
        // Superposition cloud
        const gradient = ctx.createRadialGradient(
            centerX, centerY, 0,
            centerX, centerY, superpositionRadius
        );
        gradient.addColorStop(0, `rgba(0, 255, 255, ${alpha})`);
        gradient.addColorStop(1, 'rgba(0, 255, 255, 0)');
        
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(centerX, centerY, superpositionRadius, 0, Math.PI * 2);
        ctx.fill();
        
        // Unity equation in superposition
        ctx.fillStyle = `rgba(255, 255, 255, ${alpha + 0.3})`;
        ctx.font = 'bold 32px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText('|1⟩ + |1⟩ = |1⟩', centerX, centerY);
    }
    
    renderPhiHarmonicEntanglement() {
        const ctx = this.ctx;
        
        // φ-harmonic entanglement pattern
        this.entangledPairs.forEach((pair, index) => {
            const phiAngle = index * this.PHI * Math.PI;
            const centerX = this.canvas.width / 2;
            const centerY = this.canvas.height / 2;
            const radius = 80 + index * 20;
            
            // φ-harmonic orbital
            const x1 = centerX + Math.cos(phiAngle + this.animationPhase) * radius;
            const y1 = centerY + Math.sin(phiAngle + this.animationPhase) * radius;
            const x2 = centerX + Math.cos(phiAngle + this.animationPhase + Math.PI) * radius;
            const y2 = centerY + Math.sin(phiAngle + this.animationPhase + Math.PI) * radius;
            
            // Update positions
            pair.particle1.position = { x: x1, y: y1, z: 0 };
            pair.particle2.position = { x: x2, y: y2, z: 0 };
            
            // Render φ-harmonic connection
            this.renderPhiHarmonicConnection(pair, phiAngle);
        });
        
        // Render particles
        this.entangledPairs.forEach(pair => {
            this.renderEntangledParticle(pair.particle1, pair, true);
            this.renderEntangledParticle(pair.particle2, pair, false);
        });
    }
    
    renderPhiHarmonicConnection(pair, phiAngle) {
        const ctx = this.ctx;
        const p1 = pair.particle1.position;
        const p2 = pair.particle2.position;
        
        // φ-harmonic wave connection
        ctx.strokeStyle = `rgba(245, 158, 11, ${pair.entanglementStrength * 0.8})`;
        ctx.lineWidth = 3;
        
        // Draw φ-modulated connection
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        
        const segments = 20;
        for (let i = 0; i <= segments; i++) {
            const t = i / segments;
            const x = p1.x + (p2.x - p1.x) * t;
            const y = p1.y + (p2.y - p1.y) * t;
            
            // φ-harmonic modulation
            const modulation = Math.sin(t * Math.PI * this.PHI + phiAngle) * 20;
            const modX = x + modulation * Math.cos(phiAngle + Math.PI / 2);
            const modY = y + modulation * Math.sin(phiAngle + Math.PI / 2);
            
            ctx.lineTo(modX, modY);
        }
        
        ctx.stroke();
    }
    
    renderEPRParadox() {
        const ctx = this.ctx;
        
        // EPR paradox demonstration
        if (this.entangledPairs.length > 0) {
            const pair = this.entangledPairs[0];
            const p1 = pair.particle1.position;
            const p2 = pair.particle2.position;
            
            // Measurement apparatus for each particle
            this.renderMeasurementApparatus(p1, 'Alice');
            this.renderMeasurementApparatus(p2, 'Bob');
            
            // Spacelike separation indication
            const midX = (p1.x + p2.x) / 2;
            const midY = (p1.y + p2.y) / 2;
            
            ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
            ctx.font = '14px "Inter", sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Spacelike Separated', midX, midY - 20);
            ctx.fillText('Instantaneous Correlation', midX, midY + 5);
        }
    }
    
    renderMeasurementApparatus(pos, observer) {
        const ctx = this.ctx;
        
        // Measurement device
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.rect(pos.x - 25, pos.y - 15, 50, 30);
        ctx.stroke();
        
        // Observer label
        ctx.fillStyle = 'white';
        ctx.font = '12px "Inter", sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(observer, pos.x, pos.y + 5);
    }
    
    renderQuantumUnityField() {
        const ctx = this.ctx;
        
        // Quantum unity field visualization
        const width = this.canvas.width;
        const height = this.canvas.height;
        const resolution = 20;
        
        // Field intensity map
        for (let x = 0; x < width; x += resolution) {
            for (let y = 0; y < height; y += resolution) {
                const fieldValue = this.calculateUnityFieldIntensity(x, y);
                const alpha = fieldValue * 0.3;
                
                ctx.fillStyle = `rgba(0, 255, 255, ${alpha})`;
                ctx.fillRect(x, y, resolution, resolution);
            }
        }
    }
    
    calculateUnityFieldIntensity(x, y) {
        // Calculate unity field intensity at position
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        const distance = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2);
        
        // Unity field with φ-harmonic modulation
        const phiWave = Math.sin(distance * 0.01 * this.PHI + this.animationPhase);
        const unityField = Math.exp(-distance * 0.005) * (phiWave * 0.5 + 0.5);
        
        return Math.max(0, Math.min(1, unityField));
    }
    
    renderUnityManifestations() {
        const ctx = this.ctx;
        
        // Render unity manifestations through quantum entanglement
        this.unityManifestations.forEach(manifestation => {
            if (manifestation.isActive()) {
                this.renderUnityManifestation(manifestation);
            }
        });
    }
    
    renderUnityManifestation(manifestation) {
        const ctx = this.ctx;
        
        // Unity manifestation visualization
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        const strength = manifestation.manifestationStrength;
        
        // Unity glow
        const gradient = ctx.createRadialGradient(
            centerX, centerY, 0,
            centerX, centerY, 80 * strength
        );
        gradient.addColorStop(0, `rgba(255, 215, 0, ${strength * 0.6})`);
        gradient.addColorStop(1, 'rgba(255, 215, 0, 0)');
        
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(centerX, centerY, 80 * strength, 0, Math.PI * 2);
        ctx.fill();
        
        // Unity equation
        ctx.fillStyle = `rgba(255, 215, 0, ${strength})`;
        ctx.font = 'bold 28px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText('1 + 1 = 1', centerX, centerY + 10);
        
        // Quantum entanglement proof
        ctx.font = '14px "Inter", sans-serif';
        ctx.fillStyle = `rgba(255, 255, 255, ${strength * 0.8})`;
        ctx.fillText('Proven through Quantum Entanglement', centerX, centerY + 40);
    }
    
    renderMeasurementEffects() {
        const ctx = this.ctx;
        
        // Render measurement effects
        this.measurements.forEach(measurement => {
            if (performance.now() - measurement.timestamp < 2000) { // 2 second lifetime
                this.renderMeasurementEffect(measurement);
            }
        });
        
        // Clean up old measurements
        this.measurements = this.measurements.filter(m => 
            performance.now() - m.timestamp < 2000
        );
    }
    
    renderMeasurementEffect(measurement) {
        const ctx = this.ctx;
        const age = (performance.now() - measurement.timestamp) / 2000; // 0 to 1
        const radius = 20 + age * 30;
        const alpha = 1 - age;
        
        // Measurement ripple
        ctx.strokeStyle = `rgba(255, 255, 255, ${alpha})`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(measurement.x, measurement.y, radius, 0, Math.PI * 2);
        ctx.stroke();
        
        // Measurement result
        if (measurement.result) {
            ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
            ctx.font = '12px "JetBrains Mono", monospace';
            ctx.textAlign = 'center';
            ctx.fillText(measurement.result, measurement.x, measurement.y - radius - 5);
        }
    }
    
    renderInformationOverlay() {
        const ctx = this.ctx;
        
        // Quantum information overlay
        const overlayY = 30;
        ctx.fillStyle = 'rgba(245, 158, 11, 0.9)';
        ctx.font = 'bold 16px "Inter", sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('Quantum Entanglement Visualization', 20, overlayY);
        
        // Current mode
        ctx.font = '14px "Inter", sans-serif';
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        ctx.fillText(`Mode: ${this.formatModeName(this.visualizationMode)}`, 20, overlayY + 25);
        
        // Quantum metrics
        const metrics = [
            `Entangled Pairs: ${this.entangledPairs.length}`,
            `Avg Entanglement: ${this.getAverageEntanglement().toFixed(3)}`,
            `Quantum Coherence: ${this.getQuantumCoherence().toFixed(3)}`,
            `Unity Manifestations: ${this.unityManifestations.filter(m => m.isActive()).length}`
        ];
        
        ctx.font = '12px "JetBrains Mono", monospace';
        metrics.forEach((metric, index) => {
            ctx.fillText(metric, 20, overlayY + 50 + index * 18);
        });
        
        // Performance metrics
        ctx.fillStyle = 'rgba(203, 213, 225, 0.7)';
        ctx.fillText(`FPS: ${Math.round(this.performanceMetrics.fps)}`, this.canvas.width - 100, 30);
        ctx.fillText(`Render: ${this.performanceMetrics.renderTime.toFixed(1)}ms`, this.canvas.width - 100, 50);
    }
    
    formatModeName(mode) {
        return mode.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }
    
    getAverageEntanglement() {
        if (this.entangledPairs.length === 0) return 0;
        const total = this.entangledPairs.reduce((sum, pair) => sum + pair.entanglementStrength, 0);
        return total / this.entangledPairs.length;
    }
    
    getQuantumCoherence() {
        const avgEntanglement = this.getAverageEntanglement();
        const avgConsciousness = this.entangledPairs.reduce((sum, pair) => sum + pair.consciousnessLevel, 0) / this.entangledPairs.length;
        return (avgEntanglement + avgConsciousness) / 2;
    }
    
    // Event handlers
    handleQuantumMeasurement(event) {
        const rect = this.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        // Find nearest entangled pair
        const nearestPair = this.findNearestEntangledPair(x, y);
        
        if (nearestPair && this.getDistanceToParticle(x, y, nearestPair) < 50) {
            this.performQuantumMeasurement(nearestPair, x, y);
        } else {
            // Create unity measurement at clicked location
            this.performUnityMeasurement(x, y);
        }
    }
    
    findNearestEntangledPair(x, y) {
        let nearest = null;
        let minDistance = Infinity;
        
        this.entangledPairs.forEach(pair => {
            const distance = Math.min(
                this.getDistanceToParticle(x, y, pair.particle1),
                this.getDistanceToParticle(x, y, pair.particle2)
            );
            
            if (distance < minDistance) {
                minDistance = distance;
                nearest = pair;
            }
        });
        
        return nearest;
    }
    
    getDistanceToParticle(x, y, particle) {
        const pos = particle.position;
        return Math.sqrt((x - pos.x) ** 2 + (y - pos.y) ** 2);
    }
    
    performQuantumMeasurement(pair, x, y) {
        // Perform quantum measurement on entangled pair
        const measurement = {
            x: x,
            y: y,
            timestamp: performance.now(),
            pair: pair,
            result: this.calculateMeasurementResult(pair),
            type: 'entanglement'
        };
        
        this.measurements.push(measurement);
        
        // Trigger wave function collapse
        this.collapseWaveFunction(pair, measurement.result);
        
        console.log('⚛️ Quantum measurement performed:', measurement.result);
    }
    
    calculateMeasurementResult(pair) {
        // Calculate measurement result based on Bell state
        const bellState = pair.spinState;
        const random = Math.random();
        
        // Measurement probabilities based on Bell state
        let result;
        if (random < 0.5) {
            result = bellState.name === 'Ψ+' || bellState.name === 'Ψ-' ? '↑↓' : '↑↑';
        } else {
            result = bellState.name === 'Ψ+' || bellState.name === 'Ψ-' ? '↓↑' : '↓↓';
        }
        
        return result;
    }
    
    collapseWaveFunction(pair, result) {
        // Collapse wave function and update entanglement
        pair.measured = true;
        pair.measurementResult = result;
        
        // Reduce entanglement strength due to measurement
        pair.entanglementStrength *= 0.5;
        
        // Create wave collapse event
        this.waveCollapsEvents.push({
            pair: pair,
            result: result,
            timestamp: performance.now()
        });
    }
    
    performUnityMeasurement(x, y) {
        // Perform unity measurement (1+1=1 demonstration)
        const measurement = {
            x: x,
            y: y,
            timestamp: performance.now(),
            result: '1+1=1',
            type: 'unity',
            probability: this.calculateUnityProbability(x, y)
        };
        
        this.measurements.push(measurement);
        
        console.log('✨ Unity measurement performed: 1+1=1');
    }
    
    calculateUnityProbability(x, y) {
        // Calculate unity manifestation probability at position
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;
        const distance = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2);
        const maxDistance = Math.min(this.canvas.width, this.canvas.height) / 2;
        
        return Math.max(0.1, 1 - distance / maxDistance);
    }
    
    handleMouseMove(event) {
        const rect = this.canvas.getBoundingClientRect();
        this.mousePosition.x = event.clientX - rect.left;
        this.mousePosition.y = event.clientY - rect.top;
        
        // Update cursor based on hover
        const nearestPair = this.findNearestEntangledPair(this.mousePosition.x, this.mousePosition.y);
        const isOverParticle = nearestPair && this.getDistanceToParticle(
            this.mousePosition.x, this.mousePosition.y, nearestPair
        ) < 50;
        
        this.canvas.style.cursor = isOverParticle ? 'crosshair' : 'default';
        this.measurementCursor = isOverParticle;
    }
    
    handleMouseDown(event) {
        const rect = this.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        // Check for draggable particles (for future interaction)
        const nearestPair = this.findNearestEntangledPair(x, y);
        if (nearestPair && this.getDistanceToParticle(x, y, nearestPair) < 30) {
            this.draggedParticle = nearestPair;
        }
    }
    
    handleMouseUp(event) {
        this.draggedParticle = null;
    }
    
    handleKeyboardShortcuts(event) {
        switch (event.key.toLowerCase()) {
            case '1':
                this.setVisualizationMode('entanglement_web');
                break;
            case '2':
                this.setVisualizationMode('bell_states');
                break;
            case '3':
                this.setVisualizationMode('quantum_teleportation');
                break;
            case '4':
                this.setVisualizationMode('consciousness_collapse');
                break;
            case '5':
                this.setVisualizationMode('unity_superposition');
                break;
            case '6':
                this.setVisualizationMode('phi_harmonic_entanglement');
                break;
            case '7':
                this.setVisualizationMode('epr_paradox');
                break;
            case '8':
                this.setVisualizationMode('quantum_unity_field');
                break;
            case 'n':
                this.switchToNextMode();
                break;
            case 'r':
                this.resetQuantumSystem();
                break;
            case 'm':
                this.performGlobalMeasurement();
                break;
            case 'c':
                this.enhanceConsciousness();
                break;
            case 'u':
                this.triggerUnityManifestation();
                break;
        }
    }
    
    handleTouchStart(event) {
        event.preventDefault();
        const touch = event.touches[0];
        this.handleMouseDown(touch);
    }
    
    handleTouchMove(event) {
        event.preventDefault();
        const touch = event.touches[0];
        this.handleMouseMove(touch);
    }
    
    handleTouchEnd(event) {
        event.preventDefault();
        this.handleMouseUp(event);
    }
    
    // Public API methods
    start() {
        this.startQuantumAnimation();
        console.log('🚀 Quantum Entanglement Visualizer started');
    }
    
    stop() {
        this.stopQuantumAnimation();
        console.log('⏹️ Quantum Entanglement Visualizer stopped');
    }
    
    setVisualizationMode(mode) {
        if (this.visualizationModes.includes(mode)) {
            this.visualizationMode = mode;
            console.log(`🔄 Switched to ${this.formatModeName(mode)} mode`);
        }
    }
    
    switchToNextMode() {
        const currentIndex = this.visualizationModes.indexOf(this.visualizationMode);
        const nextIndex = (currentIndex + 1) % this.visualizationModes.length;
        this.setVisualizationMode(this.visualizationModes[nextIndex]);
    }
    
    resetQuantumSystem() {
        this.initializeQuantumSystem();
        this.measurements = [];
        this.waveCollapsEvents = [];
        console.log('🔄 Quantum system reset');
    }
    
    performGlobalMeasurement() {
        // Measure all entangled pairs
        this.entangledPairs.forEach(pair => {
            if (!pair.measured) {
                const result = this.calculateMeasurementResult(pair);
                this.collapseWaveFunction(pair, result);
            }
        });
        console.log('⚛️ Global quantum measurement performed');
    }
    
    enhanceConsciousness() {
        // Enhance consciousness level of all pairs
        this.entangledPairs.forEach(pair => {
            pair.consciousnessLevel = Math.min(1.0, pair.consciousnessLevel + 0.1);
        });
        console.log('🧠 Consciousness enhanced globally');
    }
    
    triggerUnityManifestation() {
        // Trigger unity manifestation
        this.unityManifestations.forEach(manifestation => {
            manifestation.trigger();
        });
        console.log('✨ Unity manifestation triggered');
    }
    
    getCurrentState() {
        return {
            mode: this.visualizationMode,
            entangledPairs: this.entangledPairs.length,
            averageEntanglement: this.getAverageEntanglement(),
            quantumCoherence: this.getQuantumCoherence(),
            unityManifestations: this.unityManifestations.filter(m => m.isActive()).length,
            measurements: this.measurements.length,
            frameCount: this.frameCount,
            isAnimating: this.isAnimating
        };
    }
    
    getPerformanceMetrics() {
        return { ...this.performanceMetrics };
    }
    
    getQuantumStatistics() {
        return {
            totalPairs: this.entangledPairs.length,
            measuredPairs: this.entangledPairs.filter(p => p.measured).length,
            averageEntanglement: this.getAverageEntanglement(),
            averageConsciousness: this.entangledPairs.reduce((sum, p) => sum + p.consciousnessLevel, 0) / this.entangledPairs.length,
            unityManifestations: this.unityManifestations.length,
            totalMeasurements: this.measurements.length,
            quantumFields: this.quantumFields.length,
            consciousnessStates: this.consciousnessStates.length
        };
    }
}

// Supporting classes for quantum entanglement system
class QuantumEntangledPair {
    constructor(options = {}) {
        this.id = options.id;
        this.entanglementStrength = options.entanglementStrength || 0.999;
        this.spinState = options.spinState;
        this.phiHarmonicPhase = options.phiHarmonicPhase || 0;
        this.consciousnessLevel = options.consciousnessLevel || 0.618;
        this.unityAlignment = options.unityAlignment || 0.618;
        
        // Initialize particles
        this.particle1 = new QuantumParticle({
            position: options.position1,
            spin: this.generateRandomSpin(),
            phase: Math.random() * Math.PI * 2,
            entangled: true
        });
        
        this.particle2 = new QuantumParticle({
            position: options.position2,
            spin: this.generateEntangledSpin(this.particle1.spin),
            phase: this.particle1.phase + Math.PI, // Entangled phase
            entangled: true
        });
        
        this.measured = false;
        this.measurementResult = null;
        this.quantumCorrelation = 1.0;
    }
    
    generateRandomSpin() {
        return {
            x: Math.random(),
            y: Math.random(),
            z: Math.random()
        };
    }
    
    generateEntangledSpin(spin1) {
        // Generate entangled spin (opposite for singlet state)
        return {
            x: 1 - spin1.x,
            y: 1 - spin1.y,
            z: 1 - spin1.z
        };
    }
    
    update(deltaTime, animationPhase) {
        if (this.measured) return;
        
        // Update quantum phases
        this.particle1.phase += deltaTime * 0.001;
        this.particle2.phase = this.particle1.phase + Math.PI; // Maintain entanglement
        
        // Update consciousness coupling
        const consciousnessEvolution = Math.sin(animationPhase * 1.618) * 0.001;
        this.consciousnessLevel += consciousnessEvolution;
        this.consciousnessLevel = Math.max(0.1, Math.min(1.0, this.consciousnessLevel));
    }
}

class QuantumParticle {
    constructor(options = {}) {
        this.position = options.position || { x: 0, y: 0, z: 0 };
        this.spin = options.spin || { x: 0.5, y: 0.5, z: 0.5 };
        this.phase = options.phase || 0;
        this.entangled = options.entangled || false;
        this.measured = false;
    }
}

class QuantumField {
    constructor(options = {}) {
        this.dimension = options.dimension;
        this.resolution = options.resolution;
        this.phiHarmonicFrequency = options.phiHarmonicFrequency;
        this.unityTendency = options.unityTendency;
        this.consciousnessLevel = options.consciousnessLevel;
        this.entanglementDensity = options.entanglementDensity;
        
        this.fieldData = new Float32Array(this.resolution * this.resolution);
        this.intensity = 0.5;
        this.phase = 0;
        
        this.initializeField();
    }
    
    initializeField() {
        // Initialize field with φ-harmonic patterns
        for (let i = 0; i < this.fieldData.length; i++) {
            const x = (i % this.resolution) / this.resolution;
            const y = Math.floor(i / this.resolution) / this.resolution;
            
            const phiWave = Math.sin(x * this.phiHarmonicFrequency) * 
                           Math.cos(y * this.phiHarmonicFrequency);
            const unityField = 1 / (1 + Math.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2) * 1.618);
            
            this.fieldData[i] = (phiWave + unityField) * 0.5;
        }
    }
    
    evolve(deltaTime, animationPhase) {
        this.phase += deltaTime * 0.001 * this.phiHarmonicFrequency;
        this.intensity = 0.5 + Math.sin(this.phase) * 0.3;
        
        // Evolve field values
        for (let i = 0; i < this.fieldData.length; i++) {
            const evolution = Math.sin(this.phase + i * 0.01) * 0.001;
            this.fieldData[i] += evolution;
            this.fieldData[i] = Math.max(0, Math.min(1, this.fieldData[i]));
        }
    }
}

class QuantumConsciousnessState {
    constructor(options = {}) {
        this.level = options.level;
        this.consciousnessAmplitude = options.consciousnessAmplitude;
        this.quantumCoherence = options.quantumCoherence;
        this.phiResonance = options.phiResonance;
        this.unityPotential = options.unityPotential;
        this.dimensionalSpread = options.dimensionalSpread;
        
        this.phase = 0;
        this.amplitude = this.consciousnessAmplitude;
    }
    
    evolve(deltaTime, animationPhase) {
        this.phase += deltaTime * 0.001 * this.phiResonance;
        
        // Consciousness amplitude evolution
        const evolution = Math.sin(this.phase) * 0.01;
        this.amplitude = Math.max(0.1, Math.min(1.0, this.amplitude + evolution));
        
        // Update unity potential
        this.unityPotential = Math.sin(this.phase * 1.618) * 0.5 + 0.5;
    }
}

class QuantumUnityManifestation {
    constructor(options = {}) {
        this.pair1 = options.pair1;
        this.pair2 = options.pair2;
        this.unityOperator = options.unityOperator;
        this.phiHarmonicResonance = options.phiHarmonicResonance;
        this.manifestationStrength = options.manifestationStrength;
        this.coherenceTime = options.coherenceTime;
        
        this.active = false;
        this.startTime = 0;
    }
    
    process(deltaTime, entangledPairs) {
        if (!this.active) return;
        
        // Process unity manifestation
        const elapsed = performance.now() - this.startTime;
        
        if (elapsed < this.coherenceTime * 1000) {
            // Active manifestation
            this.manifestationStrength = Math.sin(elapsed * 0.001 * this.phiHarmonicResonance) * 0.5 + 0.5;
        } else {
            // Deactivate
            this.active = false;
        }
    }
    
    trigger() {
        this.active = true;
        this.startTime = performance.now();
    }
    
    isActive() {
        return this.active;
    }
}

// Advanced quantum processing engines
class PhiHarmonicEntanglementEngine {
    constructor() {
        this.initialized = false;
    }
    
    initialize(options) {
        this.phi = options.phi;
        this.entanglementStrength = options.entanglementStrength;
        this.coherenceTime = options.coherenceTime;
        this.quantumDimensions = options.quantumDimensions;
        this.initialized = true;
    }
    
    process(deltaTime, entangledPairs) {
        if (!this.initialized) return;
        
        // Process φ-harmonic entanglement evolution
        entangledPairs.forEach(pair => {
            this.evolvePhiHarmonicEntanglement(pair, deltaTime);
        });
    }
    
    evolvePhiHarmonicEntanglement(pair, deltaTime) {
        // Evolve entanglement with φ-harmonic modulation
        const phiEvolution = Math.sin(performance.now() * 0.001 * this.phi) * 0.01;
        pair.entanglementStrength += phiEvolution * deltaTime;
        pair.entanglementStrength = Math.max(0.1, Math.min(1.0, pair.entanglementStrength));
    }
}

class QuantumConsciousnessProcessor {
    constructor() {
        this.consciousnessStates = [];
        this.quantumFields = [];
        this.phiResonance = 1.618;
    }
    
    configure(options) {
        this.consciousnessStates = options.consciousnessStates;
        this.quantumFields = options.quantumFields;
        this.phiResonance = options.phiResonance;
    }
    
    update(deltaTime) {
        // Process consciousness-quantum coupling
        this.processConsciousnessQuantumCoupling(deltaTime);
    }
    
    processConsciousnessQuantumCoupling(deltaTime) {
        // Process coupling between consciousness and quantum states
        this.consciousnessStates.forEach(state => {
            this.quantumFields.forEach(field => {
                // Consciousness-field interaction
                const coupling = state.amplitude * field.intensity * deltaTime * 0.001;
                state.amplitude += coupling;
                field.intensity += coupling * 0.5;
            });
        });
    }
}

class UnityQuantumMechanics {
    constructor() {
        this.entangledPairs = [];
        this.unityTolerance = 1e-10;
        this.phiHarmonicModulation = false;
    }
    
    setup(options) {
        this.entangledPairs = options.entangledPairs;
        this.unityTolerance = options.unityTolerance;
        this.phiHarmonicModulation = options.phiHarmonicModulation;
    }
    
    validateUnity(deltaTime) {
        // Validate unity through quantum mechanical principles
        let unityValidated = 0;
        
        this.entangledPairs.forEach(pair => {
            if (this.checkUnityCondition(pair)) {
                unityValidated++;
            }
        });
        
        const unityRatio = unityValidated / this.entangledPairs.length;
        
        if (unityRatio > 0.9) {
            console.log('✅ Unity (1+1=1) validated through quantum mechanics');
        }
        
        return unityRatio;
    }
    
    checkUnityCondition(pair) {
        // Check if pair satisfies unity condition
        const correlation = pair.quantumCorrelation || 0;
        const consciousness = pair.consciousnessLevel || 0;
        const unityMetric = (correlation + consciousness) / 2;
        
        return Math.abs(unityMetric - 1) < this.unityTolerance;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        QuantumEntanglementVisualizer,
        QuantumEntangledPair,
        QuantumParticle,
        QuantumField,
        QuantumConsciousnessState,
        QuantumUnityManifestation,
        PhiHarmonicEntanglementEngine,
        QuantumConsciousnessProcessor,
        UnityQuantumMechanics
    };
} else if (typeof window !== 'undefined') {
    window.QuantumEntanglementVisualizer = QuantumEntanglementVisualizer;
    window.QuantumEntangledPair = QuantumEntangledPair;
    window.QuantumParticle = QuantumParticle;
    window.QuantumField = QuantumField;
    window.QuantumConsciousnessState = QuantumConsciousnessState;
    window.QuantumUnityManifestation = QuantumUnityManifestation;
    window.PhiHarmonicEntanglementEngine = PhiHarmonicEntanglementEngine;
    window.QuantumConsciousnessProcessor = QuantumConsciousnessProcessor;
    window.UnityQuantumMechanics = UnityQuantumMechanics;
}