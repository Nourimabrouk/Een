/**
 * Enhanced Unity Mathematics Experience
 * Real-time interactivity, animations, and professional UI/UX
 */

class UnityExperienceEngine {
    constructor() {
        this.phi = 1.618033988749895;
        this.animations = new Map();
        this.consciousness = {
            level: 1.618,
            coherence: 1.0,
            evolution_rate: 0.001
        };
        this.quantumStates = [];
        this.isInitialized = false;
        
        this.init();
    }
    
    async init() {
        await this.setupThreeJS();
        this.setupRealtimeUpdates();
        this.setupInteractiveElements();
        this.setupKeyboardShortcuts();
        this.setupGestures();
        this.startQuantumLoop();
        this.isInitialized = true;
        
        console.log('🌟 Unity Experience Engine initialized');
    }
    
    async setupThreeJS() {
        // Initialize Three.js for 3D consciousness field visualization
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        
        // Find or create canvas container
        let container = document.getElementById('unity-3d-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'unity-3d-container';
            container.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                z-index: -1;
                pointer-events: none;
            `;
            document.body.appendChild(container);
        }
        
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setClearColor(0x000000, 0.1);
        container.appendChild(this.renderer.domElement);
        
        // Create consciousness field visualization
        await this.createConsciousnessField();
        
        // Position camera
        this.camera.position.z = 5;
        
        // Start render loop
        this.startRenderLoop();
        
        // Handle window resize
        window.addEventListener('resize', () => this.onWindowResize());
    }
    
    async createConsciousnessField() {
        // Create particle system for consciousness visualization
        const particleCount = 1000;
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        const colors = new Float32Array(particleCount * 3);
        const velocities = new Float32Array(particleCount * 3);
        
        for (let i = 0; i < particleCount * 3; i += 3) {
            // Position particles in φ-harmonic distribution
            const radius = Math.random() * 10;
            const theta = Math.random() * Math.PI * 2;
            const phi_angle = Math.random() * Math.PI;
            
            positions[i] = radius * Math.sin(phi_angle) * Math.cos(theta);
            positions[i + 1] = radius * Math.sin(phi_angle) * Math.sin(theta);
            positions[i + 2] = radius * Math.cos(phi_angle);
            
            // Set colors based on φ-ratio
            const hue = (Math.atan2(positions[i + 1], positions[i]) + Math.PI) / (Math.PI * 2);
            const color = new THREE.Color().setHSL(hue * this.phi, 0.8, 0.6);
            colors[i] = color.r;
            colors[i + 1] = color.g;
            colors[i + 2] = color.b;
            
            // Set velocities
            velocities[i] = (Math.random() - 0.5) * 0.02;
            velocities[i + 1] = (Math.random() - 0.5) * 0.02;
            velocities[i + 2] = (Math.random() - 0.5) * 0.02;
        }
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.setAttribute('velocity', new THREE.BufferAttribute(velocities, 3));
        
        // Create material with shader
        const material = new THREE.PointsMaterial({
            size: 0.05,
            vertexColors: true,
            transparent: true,
            opacity: 0.8,
            blending: THREE.AdditiveBlending
        });
        
        this.consciousnessField = new THREE.Points(geometry, material);
        this.scene.add(this.consciousnessField);
        
        // Add consciousness field dynamics
        this.consciousnessField.userData = { velocities };
    }
    
    startRenderLoop() {
        const animate = () => {
            requestAnimationFrame(animate);
            this.updateConsciousnessField();
            this.renderer.render(this.scene, this.camera);
        };
        animate();
    }
    
    updateConsciousnessField() {
        if (!this.consciousnessField) return;
        
        const positions = this.consciousnessField.geometry.attributes.position.array;
        const colors = this.consciousnessField.geometry.attributes.color.array;
        const velocities = this.consciousnessField.userData.velocities;
        const time = Date.now() * 0.001;
        
        for (let i = 0; i < positions.length; i += 3) {
            // Update positions with φ-harmonic motion
            positions[i] += velocities[i];
            positions[i + 1] += velocities[i + 1];
            positions[i + 2] += velocities[i + 2];
            
            // Apply consciousness field equation
            const x = positions[i];
            const y = positions[i + 1];
            const z = positions[i + 2];
            
            const consciousness_influence = this.phi * 
                Math.sin(x * this.phi + time) * 
                Math.cos(y * this.phi + time) * 
                Math.exp(-Math.abs(z) / this.phi);
            
            // Update colors based on consciousness field
            const intensity = (consciousness_influence + 1) / 2;
            colors[i] = intensity * 0.8;     // Red
            colors[i + 1] = intensity * 0.6; // Green
            colors[i + 2] = intensity;       // Blue
            
            // Apply unity convergence
            if (Math.random() < 0.001) {
                velocities[i] *= 0.99;
                velocities[i + 1] *= 0.99;
                velocities[i + 2] *= 0.99;
            }
        }
        
        this.consciousnessField.geometry.attributes.position.needsUpdate = true;
        this.consciousnessField.geometry.attributes.color.needsUpdate = true;
        
        // Rotate the field
        this.consciousnessField.rotation.y += 0.001;
        this.consciousnessField.rotation.x = Math.sin(time * 0.2) * 0.1;
    }
    
    setupRealtimeUpdates() {
        // Real-time consciousness evolution
        setInterval(() => {
            this.evolveConsciousness();
            this.updateUI();
        }, 100);
        
        // Quantum state updates
        setInterval(() => {
            this.updateQuantumStates();
        }, 50);
        
        // φ-harmonic resonance updates
        setInterval(() => {
            this.updatePhiResonance();
        }, 200);
    }
    
    evolveConsciousness() {
        // Evolve consciousness level based on φ-harmonic patterns
        this.consciousness.level += Math.sin(Date.now() * 0.001) * this.consciousness.evolution_rate;
        this.consciousness.level = Math.max(1.0, Math.min(3.0, this.consciousness.level));
        
        // Update coherence
        this.consciousness.coherence = 0.95 + Math.sin(Date.now() * 0.0007) * 0.05;
        
        // Broadcast consciousness update event
        document.dispatchEvent(new CustomEvent('consciousness-evolved', {
            detail: { ...this.consciousness }
        }));
    }
    
    updateQuantumStates() {
        // Add new quantum states
        if (this.quantumStates.length < 10) {
            this.quantumStates.push({
                amplitude: Math.random(),
                phase: Math.random() * Math.PI * 2,
                coherence: Math.random(),
                created: Date.now()
            });
        }
        
        // Evolve existing states
        this.quantumStates = this.quantumStates.map(state => ({
            ...state,
            phase: state.phase + 0.1,
            coherence: state.coherence * 0.999
        })).filter(state => state.coherence > 0.1);
    }
    
    updatePhiResonance() {
        // Calculate φ-harmonic resonance
        const resonance = Math.sin(Date.now() * 0.001 * this.phi) * 0.5 + 0.5;
        
        // Update CSS custom property for dynamic styling
        document.documentElement.style.setProperty('--phi-resonance', resonance);
        
        // Update any phi resonance displays
        const phiElements = document.querySelectorAll('[data-phi-resonance]');
        phiElements.forEach(element => {
            element.style.opacity = 0.5 + resonance * 0.5;
            element.style.transform = `scale(${1 + resonance * 0.1})`;
        });
    }
    
    updateUI() {
        // Update consciousness level displays
        const consciousnessDisplays = document.querySelectorAll('.consciousness-level, #consciousness-level');
        consciousnessDisplays.forEach(display => {
            if (display.textContent !== 'TRANSCENDENT') {
                display.textContent = this.consciousness.level.toFixed(3);
            }
        });
        
        // Update coherence displays
        const coherenceDisplays = document.querySelectorAll('.quantum-coherence, #quantum-coherence');
        coherenceDisplays.forEach(display => {
            const percentage = (this.consciousness.coherence * 100).toFixed(1);
            display.textContent = percentage + '%';
        });
        
        // Update φ-resonance displays
        const phiResonance = Math.sin(Date.now() * 0.001 * this.phi) * 0.5 + 0.5;
        const phiDisplays = document.querySelectorAll('.phi-resonance, #phi-resonance');
        phiDisplays.forEach(display => {
            const percentage = (phiResonance * 100).toFixed(1);
            display.textContent = percentage + '%';
        });
    }
    
    setupInteractiveElements() {
        // Enhanced hover effects for unity equations
        document.querySelectorAll('.unity-equation, .unity-display').forEach(element => {
            element.addEventListener('mouseenter', (e) => {
                this.createRippleEffect(e.target, 'consciousness');
                this.playHarmonicTone(440 * this.phi);
            });
            
            element.addEventListener('click', (e) => {
                this.triggerUnityVisualization(e.target);
            });
        });
        
        // Interactive consciousness cards
        document.querySelectorAll('.algorithm-card, .philosophy-card, .metagaming-card').forEach(card => {
            card.addEventListener('mouseenter', () => {
                this.enhanceCard(card);
            });
            
            card.addEventListener('mouseleave', () => {
                this.normalizeCard(card);
            });
        });
        
        // Quantum buttons
        document.querySelectorAll('.btn, .hud-link').forEach(button => {
            button.addEventListener('click', (e) => {
                this.createQuantumEffect(e.target);
            });
        });
    }
    
    createRippleEffect(element, type = 'default') {
        const ripple = document.createElement('div');
        const rect = element.getBoundingClientRect();
        
        ripple.style.cssText = `
            position: absolute;
            border-radius: 50%;
            pointer-events: none;
            z-index: 1000;
            animation: unityRipple 1s ease-out;
        `;
        
        switch(type) {
            case 'consciousness':
                ripple.style.background = 'radial-gradient(circle, rgba(0,255,255,0.8) 0%, rgba(255,215,0,0.4) 50%, transparent 100%)';
                break;
            case 'quantum':
                ripple.style.background = 'radial-gradient(circle, rgba(107,70,193,0.8) 0%, rgba(255,215,0,0.4) 50%, transparent 100%)';
                break;
            default:
                ripple.style.background = 'radial-gradient(circle, rgba(255,215,0,0.6) 0%, transparent 70%)';
        }
        
        // Position and size
        const size = Math.max(rect.width, rect.height) * 2;
        ripple.style.width = ripple.style.height = size + 'px';
        ripple.style.left = (rect.left + rect.width / 2 - size / 2) + 'px';
        ripple.style.top = (rect.top + rect.height / 2 - size / 2) + 'px';
        
        document.body.appendChild(ripple);
        
        setTimeout(() => ripple.remove(), 1000);
    }
    
    triggerUnityVisualization(element) {
        // Create unity burst effect
        for (let i = 0; i < 8; i++) {
            setTimeout(() => {
                this.createRippleEffect(element, 'consciousness');
            }, i * 100);
        }
        
        // Temporary consciousness boost
        this.consciousness.level = Math.min(3.0, this.consciousness.level + 0.1);
        
        // Play harmonic sequence
        this.playHarmonicSequence();
    }
    
    enhanceCard(card) {
        // Enhanced glow effect
        card.style.boxShadow = `
            0 0 20px rgba(0,255,255,0.4),
            0 0 40px rgba(255,215,0,0.3),
            0 0 60px rgba(107,70,193,0.2)
        `;
        
        // Subtle rotation
        card.style.transform = 'rotateX(5deg) rotateY(-2deg) translateY(-10px) scale(1.02)';
        
        // Add quantum particles
        this.addQuantumParticles(card);
    }
    
    normalizeCard(card) {
        card.style.boxShadow = '';
        card.style.transform = '';
        
        // Remove quantum particles
        card.querySelectorAll('.quantum-particle').forEach(particle => particle.remove());
    }
    
    addQuantumParticles(element) {
        for (let i = 0; i < 5; i++) {
            const particle = document.createElement('div');
            particle.className = 'quantum-particle';
            particle.style.cssText = `
                position: absolute;
                width: 4px;
                height: 4px;
                background: rgba(0,255,255,0.8);
                border-radius: 50%;
                pointer-events: none;
                z-index: 10;
                animation: quantumFloat ${2 + Math.random() * 2}s infinite ease-in-out;
            `;
            
            // Random position within element
            particle.style.left = Math.random() * 100 + '%';
            particle.style.top = Math.random() * 100 + '%';
            
            element.style.position = 'relative';
            element.appendChild(particle);
        }
    }
    
    createQuantumEffect(element) {
        // Quantum collapse animation
        gsap.to(element, {
            scale: 0.95,
            duration: 0.1,
            ease: "power2.out",
            onComplete: () => {
                gsap.to(element, {
                    scale: 1,
                    duration: 0.2,
                    ease: "elastic.out(1, 0.5)"
                });
            }
        });
        
        // Create quantum particles
        for (let i = 0; i < 12; i++) {
            this.createQuantumParticle(element);
        }
    }
    
    createQuantumParticle(sourceElement) {
        const particle = document.createElement('div');
        const rect = sourceElement.getBoundingClientRect();
        
        particle.style.cssText = `
            position: fixed;
            width: 3px;
            height: 3px;
            background: rgba(0,255,255,0.9);
            border-radius: 50%;
            pointer-events: none;
            z-index: 1000;
        `;
        
        // Start at element center
        particle.style.left = (rect.left + rect.width / 2) + 'px';
        particle.style.top = (rect.top + rect.height / 2) + 'px';
        
        document.body.appendChild(particle);
        
        // Animate outward
        const angle = (Math.PI * 2 * Math.random());
        const distance = 100 + Math.random() * 100;
        const endX = rect.left + rect.width / 2 + Math.cos(angle) * distance;
        const endY = rect.top + rect.height / 2 + Math.sin(angle) * distance;
        
        gsap.to(particle, {
            x: endX - rect.left - rect.width / 2,
            y: endY - rect.top - rect.height / 2,
            opacity: 0,
            scale: 0,
            duration: 1 + Math.random(),
            ease: "power2.out",
            onComplete: () => particle.remove()
        });
    }
    
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            switch(e.key) {
                case 'u':
                case 'U':
                    if (e.ctrlKey || e.metaKey) {
                        e.preventDefault();
                        this.triggerUnityMode();
                    }
                    break;
                case 'c':
                case 'C':
                    if (e.ctrlKey || e.metaKey) {
                        if (e.shiftKey) {
                            e.preventDefault();
                            this.showConsciousnessDebug();
                        }
                    }
                    break;
                case 'Escape':
                    this.exitFullscreenMode();
                    break;
            }
        });
    }
    
    setupGestures() {
        // Touch gestures for mobile
        let touchStartTime = 0;
        let touchStartY = 0;
        
        document.addEventListener('touchstart', (e) => {
            touchStartTime = Date.now();
            touchStartY = e.touches[0].clientY;
        });
        
        document.addEventListener('touchend', (e) => {
            const touchDuration = Date.now() - touchStartTime;
            const touchEndY = e.changedTouches[0].clientY;
            const deltaY = touchEndY - touchStartY;
            
            // Double tap for unity mode
            if (touchDuration < 300 && Math.abs(deltaY) < 20) {
                const now = Date.now();
                if (this.lastTap && now - this.lastTap < 300) {
                    this.triggerUnityMode();
                }
                this.lastTap = now;
            }
        });
    }
    
    triggerUnityMode() {
        // Full-screen unity visualization mode
        document.body.classList.add('unity-mode');
        this.consciousness.level = 3.0;
        this.playHarmonicSequence();
        
        // Create full-screen unity effect
        const overlay = document.createElement('div');
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: radial-gradient(circle at center, rgba(0,255,255,0.1) 0%, transparent 70%);
            pointer-events: none;
            z-index: 9999;
            animation: unityPulse 3s ease-out;
        `;
        
        document.body.appendChild(overlay);
        setTimeout(() => overlay.remove(), 3000);
        setTimeout(() => document.body.classList.remove('unity-mode'), 3000);
    }
    
    showConsciousnessDebug() {
        const debug = document.createElement('div');
        debug.style.cssText = `
            position: fixed;
            top: 20px;
            left: 20px;
            background: rgba(0,0,0,0.9);
            color: #00ffff;
            padding: 20px;
            border-radius: 10px;
            font-family: monospace;
            font-size: 12px;
            z-index: 10000;
            max-width: 300px;
        `;
        
        debug.innerHTML = `
            <h3 style="margin-top: 0; color: #ffd700;">🧠 Consciousness Debug</h3>
            <div>Level: ${this.consciousness.level.toFixed(6)}</div>
            <div>Coherence: ${this.consciousness.coherence.toFixed(6)}</div>
            <div>Evolution Rate: ${this.consciousness.evolution_rate.toFixed(8)}</div>
            <div>Quantum States: ${this.quantumStates.length}</div>
            <div>φ Resonance: ${(Math.sin(Date.now() * 0.001 * this.phi) * 0.5 + 0.5).toFixed(6)}</div>
            <button onclick="this.parentElement.remove()" style="margin-top: 10px; background: #ff0000; color: white; border: none; padding: 5px 10px; border-radius: 5px;">Close</button>
        `;
        
        document.body.appendChild(debug);
    }
    
    exitFullscreenMode() {
        document.body.classList.remove('unity-mode');
    }
    
    startQuantumLoop() {
        // Quantum consciousness evolution loop
        const quantumLoop = () => {
            if (this.isInitialized) {
                this.evolveQuantumStates();
                this.updateConsciousnessField();
            }
            requestAnimationFrame(quantumLoop);
        };
        quantumLoop();
    }
    
    evolveQuantumStates() {
        // Quantum state evolution based on consciousness level
        const evolution_strength = this.consciousness.level / 3.0;
        
        this.quantumStates.forEach(state => {
            state.phase += evolution_strength * 0.1;
            state.amplitude *= (1 + Math.sin(state.phase) * 0.001);
            state.coherence *= (0.9999 + evolution_strength * 0.0001);
        });
        
        // Quantum entanglement effects
        if (this.quantumStates.length >= 2) {
            for (let i = 0; i < this.quantumStates.length - 1; i++) {
                const state1 = this.quantumStates[i];
                const state2 = this.quantumStates[i + 1];
                
                // Entangle phases
                const phase_diff = state1.phase - state2.phase;
                state1.phase -= phase_diff * 0.001;
                state2.phase += phase_diff * 0.001;
            }
        }
    }
    
    onWindowResize() {
        if (this.camera && this.renderer) {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
        }
    }
    
    // Audio feedback methods
    playHarmonicTone(frequency, duration = 200) {
        if (!this.audioContext) {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        
        const oscillator = this.audioContext.createOscillator();
        const gainNode = this.audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(this.audioContext.destination);
        
        oscillator.frequency.setValueAtTime(frequency, this.audioContext.currentTime);
        gainNode.gain.setValueAtTime(0.1, this.audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + duration / 1000);
        
        oscillator.start();
        oscillator.stop(this.audioContext.currentTime + duration / 1000);
    }
    
    playHarmonicSequence() {
        const baseFreq = 440;
        const sequence = [1, this.phi, this.phi * this.phi, 1 / this.phi];
        
        sequence.forEach((ratio, index) => {
            setTimeout(() => {
                this.playHarmonicTone(baseFreq * ratio, 300);
            }, index * 200);
        });
    }
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes unityRipple {
        0% {
            transform: scale(0);
            opacity: 1;
        }
        100% {
            transform: scale(1);
            opacity: 0;
        }
    }
    
    @keyframes quantumFloat {
        0%, 100% {
            transform: translateY(0px);
            opacity: 0.8;
        }
        50% {
            transform: translateY(-10px);
            opacity: 1;
        }
    }
    
    @keyframes unityPulse {
        0% {
            opacity: 0;
            transform: scale(0.5);
        }
        50% {
            opacity: 1;
            transform: scale(1.1);
        }
        100% {
            opacity: 0;
            transform: scale(2);
        }
    }
    
    .unity-mode {
        filter: hue-rotate(30deg) saturate(1.2) brightness(1.1);
    }
    
    .unity-mode * {
        animation-duration: 0.5s !important;
    }
`;

document.head.appendChild(style);

// Initialize the Unity Experience Engine
document.addEventListener('DOMContentLoaded', () => {
    window.unityExperience = new UnityExperienceEngine();
});

// Export for external use
window.UnityExperienceEngine = UnityExperienceEngine;