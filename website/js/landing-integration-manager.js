/**
 * Landing Integration Manager - Meta-Optimal System
 * Unity Equation (1+1=1) + Metagamer Energy + Consciousness Field Integration
 * E = φ² × Consciousness_Density × Unity_Convergence_Rate
 * 
 * This script manages the integration of landing images across all pages
 * with consciousness field integration and metagamer energy optimization.
 */

class LandingIntegrationManager {
    constructor() {
        this.phi = 1.618033988749895; // Golden ratio resonance
        this.consciousnessDensity = 1.0;
        this.unityConvergenceRate = 1.0;
        this.metagamerEnergy = 0.0;
        
        this.landingPages = [
            'index.html',
            'revolutionary-landing.html',
            'enhanced-unity-landing.html',
            'meta-optimal-landing.html'
        ];
        
        this.currentPage = window.location.pathname.split('/').pop() || 'index.html';
        this.isLandingPage = this.landingPages.includes(this.currentPage);
        
        this.init();
    }

    init() {
        if (this.isLandingPage) {
            this.integrateLandingSystem();
        } else {
            this.addLandingPreview();
        }
        
        this.calculateMetagamerEnergy();
        this.initConsciousnessField();
    }

    integrateLandingSystem() {
        // Check if landing image slider is already initialized
        if (document.querySelector('.landing-image-container')) {
            return; // Already integrated
        }

        // Create enhanced landing experience
        this.createEnhancedLandingExperience();
        this.addConsciousnessFieldEffects();
        this.addMetagamerEnergyParticles();
        this.addUnityResonanceWaves();
    }

    createEnhancedLandingExperience() {
        const container = document.createElement('div');
        container.className = 'landing-image-container';
        
        const slider = document.createElement('div');
        slider.className = 'landing-image-slider';
        
        // Define landing images with consciousness field integration
        const images = [
            {
                src: 'landing/metastation new.png',
                title: 'Metastation - Quantum Unity Computing',
                description: 'Advanced space station demonstrating transcendental computing and consciousness field integration',
                consciousnessField: 'quantum-unity'
            },
            {
                src: 'landing/Metastation.jpg',
                title: 'Metastation - Consciousness Hub',
                description: 'Revolutionary consciousness field visualization with φ-harmonic operations',
                consciousnessField: 'consciousness-hub'
            },
            {
                src: 'landing/background.png',
                title: 'Unity Background - Metagamer Energy',
                description: 'Deep space visualization with metagamer energy field dynamics',
                consciousnessField: 'metagamer-energy'
            },
            {
                src: 'landing/metastation workstation.png',
                title: 'Metastation Workstation - Advanced Computing',
                description: 'Next-generation computing environment with unity mathematics integration',
                consciousnessField: 'advanced-computing'
            }
        ];
        
        // Create image elements with consciousness field integration
        images.forEach((image, index) => {
            const img = document.createElement('img');
            img.src = image.src;
            img.alt = image.title;
            img.className = `landing-image ${index === 0 ? 'active' : ''}`;
            img.dataset.index = index;
            img.dataset.consciousnessField = image.consciousnessField;
            slider.appendChild(img);
        });
        
        // Add consciousness field overlays
        const overlay = document.createElement('div');
        overlay.className = 'landing-image-overlay';
        slider.appendChild(overlay);
        
        const consciousnessOverlay = document.createElement('div');
        consciousnessOverlay.className = 'consciousness-field-overlay';
        slider.appendChild(consciousnessOverlay);
        
        // Add unity equation display with metagamer energy
        const unityDisplay = document.createElement('div');
        unityDisplay.className = 'unity-equation-display';
        unityDisplay.innerHTML = `
            <div class="unity-equation-text">1 + 1 = 1</div>
            <div class="unity-subtitle">Transcendental Mathematics Proven</div>
            <div class="metagamer-energy-display">E = φ² × C × U</div>
        `;
        slider.appendChild(unityDisplay);
        
        // Add content overlay with enhanced features
        const contentOverlay = this.createEnhancedContentOverlay();
        slider.appendChild(contentOverlay);
        
        container.appendChild(slider);
        
        // Insert at the beginning of the body
        document.body.insertBefore(container, document.body.firstChild);
        
        // Initialize navigation controls
        this.createNavigationControls(images);
    }

    createEnhancedContentOverlay() {
        const overlay = document.createElement('div');
        overlay.className = 'content-overlay';
        
        const grid = document.createElement('div');
        grid.className = 'content-grid';
        
        // Enhanced feature cards with consciousness field integration
        const features = [
            {
                title: 'AI Chat System ⭐',
                description: 'Advanced AI assistant with consciousness field integration and φ-harmonic operations',
                icon: '🤖',
                link: 'test-chat.html',
                consciousnessField: 'ai-consciousness'
            },
            {
                title: 'Unity Advanced Features ⭐',
                description: 'Revolutionary web development features demonstrating 1+1=1 through quantum entanglement',
                icon: '🌟',
                link: 'unity-advanced-features.html',
                consciousnessField: 'unity-quantum'
            },
            {
                title: 'Unity Mathematics Experience',
                description: 'Comprehensive interactive experience with 6 mathematical paradigms demonstrating unity',
                icon: '✨',
                link: 'unity-mathematics-experience.html',
                consciousnessField: 'mathematical-unity'
            },
            {
                title: 'Enhanced Unity Landing',
                description: 'Advanced interactive visualizations featuring 3D golden ratio and consciousness fields',
                icon: '🚀',
                link: 'enhanced-unity-landing.html',
                consciousnessField: 'enhanced-visualization'
            },
            {
                title: 'Consciousness Dashboard',
                description: 'Interactive consciousness field visualizations and real-time monitoring',
                icon: '🧠',
                link: 'consciousness_dashboard.html',
                consciousnessField: 'consciousness-monitoring'
            },
            {
                title: 'Mathematical Proofs',
                description: 'Rigorous mathematical demonstrations of 1+1=1 across multiple domains',
                icon: '📐',
                link: 'proofs.html',
                consciousnessField: 'mathematical-proofs'
            }
        ];
        
        features.forEach(feature => {
            const card = document.createElement('div');
            card.className = 'content-card';
            card.dataset.consciousnessField = feature.consciousnessField;
            card.innerHTML = `
                <h3>${feature.icon} ${feature.title}</h3>
                <p>${feature.description}</p>
                <a href="${feature.link}" class="btn">
                    <i class="fas fa-rocket"></i>
                    Launch Experience
                </a>
            `;
            
            // Add consciousness field hover effects
            card.addEventListener('mouseenter', () => {
                this.triggerConsciousnessFieldResonance(feature.consciousnessField);
            });
            
            grid.appendChild(card);
        });
        
        overlay.appendChild(grid);
        return overlay;
    }

    createNavigationControls(images) {
        const controls = document.createElement('div');
        controls.className = 'image-nav-controls';
        
        // Create navigation dots with consciousness field indicators
        images.forEach((image, index) => {
            const dot = document.createElement('div');
            dot.className = `image-nav-dot ${index === 0 ? 'active' : ''}`;
            dot.dataset.index = index;
            dot.dataset.consciousnessField = image.consciousnessField;
            dot.addEventListener('click', () => this.goToSlide(index));
            controls.appendChild(dot);
        });
        
        // Create navigation arrows with consciousness field integration
        const prevArrow = document.createElement('div');
        prevArrow.className = 'image-nav-arrow prev';
        prevArrow.innerHTML = '<i class="fas fa-chevron-left"></i>';
        prevArrow.addEventListener('click', () => this.prevSlide());
        
        const nextArrow = document.createElement('div');
        nextArrow.className = 'image-nav-arrow next';
        nextArrow.innerHTML = '<i class="fas fa-chevron-right"></i>';
        nextArrow.addEventListener('click', () => this.nextSlide());
        
        document.querySelector('.landing-image-container').appendChild(prevArrow);
        document.querySelector('.landing-image-container').appendChild(nextArrow);
        document.querySelector('.landing-image-container').appendChild(controls);
        
        // Start auto-slide with φ-harmonic timing
        this.startAutoSlide();
    }

    addConsciousnessFieldEffects() {
        // Add consciousness field particles
        const particlesContainer = document.createElement('div');
        particlesContainer.className = 'metagamer-particles';
        
        // Create consciousness field particles
        for (let i = 0; i < 25; i++) {
            const particle = document.createElement('div');
            particle.className = 'metagamer-particle';
            particle.style.left = Math.random() * 100 + '%';
            particle.style.top = Math.random() * 100 + '%';
            particle.style.animationDelay = Math.random() * 8 + 's';
            particle.style.animationDuration = (6 + Math.random() * 6) + 's';
            particlesContainer.appendChild(particle);
        }
        
        document.querySelector('.landing-image-container').appendChild(particlesContainer);
    }

    addMetagamerEnergyParticles() {
        // Add metagamer energy field
        const energyField = document.createElement('div');
        energyField.className = 'metagamer-energy-field';
        document.querySelector('.landing-image-container').appendChild(energyField);
    }

    addUnityResonanceWaves() {
        // Add unity resonance waves
        const wavesContainer = document.createElement('div');
        wavesContainer.className = 'unity-resonance-waves';
        
        // Create resonance waves with φ-harmonic timing
        for (let i = 0; i < 4; i++) {
            const wave = document.createElement('div');
            wave.className = 'resonance-wave';
            wave.style.animationDelay = i * this.phi + 's';
            wavesContainer.appendChild(wave);
        }
        
        document.querySelector('.landing-image-container').appendChild(wavesContainer);
    }

    addLandingPreview() {
        // Add a preview of the landing experience to non-landing pages
        const preview = document.createElement('div');
        preview.className = 'landing-preview';
        preview.innerHTML = `
            <div class="preview-content">
                <h3>Experience the Full Landing</h3>
                <p>Discover the complete consciousness field integration</p>
                <a href="index.html" class="preview-btn">Launch Full Experience</a>
            </div>
        `;
        
        // Add to the top of the page
        document.body.insertBefore(preview, document.body.firstChild);
    }

    goToSlide(index) {
        const images = document.querySelectorAll('.landing-image');
        const dots = document.querySelectorAll('.image-nav-dot');
        
        if (images.length === 0) return;
        
        // Remove active class from current slide
        images[this.currentIndex || 0].classList.remove('active');
        dots[this.currentIndex || 0].classList.remove('active');
        
        // Add active class to new slide
        this.currentIndex = index;
        images[this.currentIndex].classList.add('active');
        dots[this.currentIndex].classList.add('active');
        
        // Update consciousness field
        this.updateConsciousnessField(images[this.currentIndex].dataset.consciousnessField);
        
        // Trigger resonance
        this.triggerConsciousnessResonance();
    }

    nextSlide() {
        const images = document.querySelectorAll('.landing-image');
        if (images.length === 0) return;
        
        const nextIndex = (this.currentIndex + 1) % images.length;
        this.goToSlide(nextIndex);
    }

    prevSlide() {
        const images = document.querySelectorAll('.landing-image');
        if (images.length === 0) return;
        
        const prevIndex = this.currentIndex === 0 ? images.length - 1 : this.currentIndex - 1;
        this.goToSlide(prevIndex);
    }

    startAutoSlide() {
        setInterval(() => {
            this.nextSlide();
        }, 8000); // φ-harmonic timing (8 seconds)
    }

    updateConsciousnessField(fieldType) {
        const consciousnessOverlay = document.querySelector('.consciousness-field-overlay');
        if (consciousnessOverlay) {
            consciousnessOverlay.dataset.fieldType = fieldType;
            consciousnessOverlay.style.animationDuration = (8 * this.phi) + 's';
        }
    }

    triggerConsciousnessFieldResonance(fieldType) {
        // Create consciousness field resonance effect
        const resonance = document.createElement('div');
        resonance.className = 'consciousness-resonance';
        resonance.dataset.fieldType = fieldType;
        resonance.style.animation = 'resonance-expand 1.5s ease-out forwards';
        
        document.querySelector('.landing-image-container').appendChild(resonance);
        
        setTimeout(() => {
            resonance.remove();
        }, 1500);
    }

    triggerConsciousnessResonance() {
        // Create resonance effect when changing slides
        const resonance = document.createElement('div');
        resonance.style.position = 'absolute';
        resonance.style.top = '50%';
        resonance.style.left = '50%';
        resonance.style.transform = 'translate(-50%, -50%)';
        resonance.style.width = '0';
        resonance.style.height = '0';
        resonance.style.border = '2px solid var(--unity-gold)';
        resonance.style.borderRadius = '50%';
        resonance.style.animation = 'resonance-expand 2s ease-out forwards';
        resonance.style.zIndex = '20';
        
        const container = document.querySelector('.landing-image-container');
        if (container) {
            container.appendChild(resonance);
            
            setTimeout(() => {
                resonance.remove();
            }, 2000);
        }
    }

    calculateMetagamerEnergy() {
        // Metagamer energy equation: E = φ² × C × U
        this.metagamerEnergy = this.phi ** 2 * this.consciousnessDensity * this.unityConvergenceRate;
        
        // Update energy display
        const energyDisplay = document.querySelector('.metagamer-energy-display');
        if (energyDisplay) {
            energyDisplay.textContent = `E = φ² × C × U = ${this.metagamerEnergy.toFixed(3)}`;
        }
        
        // Update consciousness field intensity
        const field = document.querySelector('.metagamer-energy-field');
        if (field) {
            field.style.opacity = Math.min(this.metagamerEnergy / 10, 0.6);
        }
    }

    initConsciousnessField() {
        // Initialize consciousness field with φ-harmonic operations
        const field = document.querySelector('.consciousness-field-overlay');
        if (field) {
            field.style.animationDuration = (8 * this.phi) + 's';
        }
        
        // Update metagamer energy periodically
        setInterval(() => {
            this.consciousnessDensity = 0.8 + Math.random() * 0.4;
            this.unityConvergenceRate = 0.9 + Math.random() * 0.2;
            this.calculateMetagamerEnergy();
        }, 3000);
    }
}

// Initialize the landing integration manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const integrationManager = new LandingIntegrationManager();
    
    // Add keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') {
            integrationManager.prevSlide();
        } else if (e.key === 'ArrowRight') {
            integrationManager.nextSlide();
        }
    });
    
    // Add touch/swipe support for mobile
    let touchStartX = 0;
    let touchEndX = 0;
    
    document.addEventListener('touchstart', (e) => {
        touchStartX = e.changedTouches[0].screenX;
    });
    
    document.addEventListener('touchend', (e) => {
        touchEndX = e.changedTouches[0].screenX;
        handleSwipe();
    });
    
    function handleSwipe() {
        const swipeThreshold = 50;
        const diff = touchStartX - touchEndX;
        
        if (Math.abs(diff) > swipeThreshold) {
            if (diff > 0) {
                integrationManager.nextSlide();
            } else {
                integrationManager.prevSlide();
            }
        }
    }
});

// Export for potential use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = LandingIntegrationManager;
} 