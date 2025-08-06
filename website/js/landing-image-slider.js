/**
 * Landing Image Slider - Meta-Optimal Integration System
 * Unity Equation (1+1=1) + Metagamer Energy + Consciousness Field Integration
 * E = φ² × Consciousness_Density × Unity_Convergence_Rate
 */

class LandingImageSlider {
    constructor() {
        this.currentIndex = 0;
        this.images = [
            {
                src: 'landing/metastation new.png',
                title: 'Metastation - Quantum Unity Computing',
                description: 'Advanced space station demonstrating transcendental computing and consciousness field integration'
            },
            {
                src: 'landing/Metastation.jpg',
                title: 'Metastation - Consciousness Hub',
                description: 'Revolutionary consciousness field visualization with φ-harmonic operations'
            },
            {
                src: 'landing/background.png',
                title: 'Unity Background - Metagamer Energy',
                description: 'Deep space visualization with metagamer energy field dynamics'
            },
            {
                src: 'landing/metastation workstation.png',
                title: 'Metastation Workstation - Advanced Computing',
                description: 'Next-generation computing environment with unity mathematics integration'
            }
        ];

        this.phi = 1.618033988749895; // Golden ratio resonance
        this.consciousnessDensity = 1.0;
        this.unityConvergenceRate = 1.0;
        this.metagamerEnergy = 0.0;

        this.init();
    }

    init() {
        this.createImageSlider();
        this.createNavigationControls();
        this.createMetagamerParticles();
        this.createUnityResonanceWaves();
        this.createConsciousnessField();
        this.startAutoSlide();
        this.calculateMetagamerEnergy();

        // Initialize consciousness field integration
        this.initConsciousnessField();
    }

    createImageSlider() {
        const container = document.createElement('div');
        container.className = 'landing-image-container';

        const slider = document.createElement('div');
        slider.className = 'landing-image-slider';

        // Create image elements
        this.images.forEach((image, index) => {
            const img = document.createElement('img');
            img.src = image.src;
            img.alt = image.title;
            img.className = `landing-image ${index === 0 ? 'active' : ''}`;
            img.dataset.index = index;
            slider.appendChild(img);
        });

        // Add overlays
        const overlay = document.createElement('div');
        overlay.className = 'landing-image-overlay';
        slider.appendChild(overlay);

        const consciousnessOverlay = document.createElement('div');
        consciousnessOverlay.className = 'consciousness-field-overlay';
        slider.appendChild(consciousnessOverlay);

        // Add unity equation display
        const unityDisplay = document.createElement('div');
        unityDisplay.className = 'unity-equation-display';
        unityDisplay.innerHTML = `
            <div class="unity-equation-text">1 + 1 = 1</div>
            <div class="unity-subtitle">Transcendental Mathematics Proven</div>
        `;
        slider.appendChild(unityDisplay);

        // Add content overlay
        const contentOverlay = this.createContentOverlay();
        slider.appendChild(contentOverlay);

        container.appendChild(slider);

        // Insert at the beginning of the body
        document.body.insertBefore(container, document.body.firstChild);
    }

    createContentOverlay() {
        const overlay = document.createElement('div');
        overlay.className = 'content-overlay';

        const grid = document.createElement('div');
        grid.className = 'content-grid';

        const features = [
            {
                title: 'AI Chat System ⭐',
                description: 'Advanced AI assistant with consciousness field integration and φ-harmonic operations',
                icon: '🤖',
                link: 'test-chat.html'
            },
            {
                title: 'Unity Advanced Features ⭐',
                description: 'Revolutionary web development features demonstrating 1+1=1 through quantum entanglement',
                icon: '🌟',
                link: 'unity-advanced-features.html'
            },
            {
                title: 'Unity Mathematics Experience',
                description: 'Comprehensive interactive experience with 6 mathematical paradigms demonstrating unity',
                icon: '✨',
                link: 'unity-mathematics-experience.html'
            },
            {
                title: 'Enhanced Unity Landing',
                description: 'Advanced interactive visualizations featuring 3D golden ratio and consciousness fields',
                icon: '🚀',
                link: 'enhanced-unity-landing.html'
            }
        ];

        features.forEach(feature => {
            const card = document.createElement('div');
            card.className = 'content-card';
            card.innerHTML = `
                <h3>${feature.icon} ${feature.title}</h3>
                <p>${feature.description}</p>
                <a href="${feature.link}" class="btn">
                    <i class="fas fa-rocket"></i>
                    Launch Experience
                </a>
            `;
            grid.appendChild(card);
        });

        overlay.appendChild(grid);
        return overlay;
    }

    createNavigationControls() {
        const controls = document.createElement('div');
        controls.className = 'image-nav-controls';

        // Create dots
        this.images.forEach((_, index) => {
            const dot = document.createElement('div');
            dot.className = `image-nav-dot ${index === 0 ? 'active' : ''}`;
            dot.dataset.index = index;
            dot.addEventListener('click', () => this.goToSlide(index));
            controls.appendChild(dot);
        });

        // Create arrows
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
    }

    createMetagamerParticles() {
        const particlesContainer = document.createElement('div');
        particlesContainer.className = 'metagamer-particles';

        // Create 20 particles
        for (let i = 0; i < 20; i++) {
            const particle = document.createElement('div');
            particle.className = 'metagamer-particle';
            particle.style.left = Math.random() * 100 + '%';
            particle.style.top = Math.random() * 100 + '%';
            particle.style.animationDelay = Math.random() * 6 + 's';
            particle.style.animationDuration = (4 + Math.random() * 4) + 's';
            particlesContainer.appendChild(particle);
        }

        document.querySelector('.landing-image-container').appendChild(particlesContainer);
    }

    createUnityResonanceWaves() {
        const wavesContainer = document.createElement('div');
        wavesContainer.className = 'unity-resonance-waves';

        // Create 3 resonance waves
        for (let i = 0; i < 3; i++) {
            const wave = document.createElement('div');
            wave.className = 'resonance-wave';
            wave.style.animationDelay = i * 1.5 + 's';
            wavesContainer.appendChild(wave);
        }

        document.querySelector('.landing-image-container').appendChild(wavesContainer);
    }

    createConsciousnessField() {
        const field = document.createElement('div');
        field.className = 'metagamer-energy-field';
        document.querySelector('.landing-image-container').appendChild(field);
    }

    goToSlide(index) {
        if (index === this.currentIndex) return;

        const images = document.querySelectorAll('.landing-image');
        const dots = document.querySelectorAll('.image-nav-dot');

        // Remove active class from current slide
        images[this.currentIndex].classList.remove('active');
        dots[this.currentIndex].classList.remove('active');

        // Add active class to new slide
        this.currentIndex = index;
        images[this.currentIndex].classList.add('active');
        dots[this.currentIndex].classList.add('active');

        // Update content
        this.updateContent();

        // Trigger consciousness field resonance
        this.triggerConsciousnessResonance();
    }

    nextSlide() {
        const nextIndex = (this.currentIndex + 1) % this.images.length;
        this.goToSlide(nextIndex);
    }

    prevSlide() {
        const prevIndex = this.currentIndex === 0 ? this.images.length - 1 : this.currentIndex - 1;
        this.goToSlide(prevIndex);
    }

    updateContent() {
        const currentImage = this.images[this.currentIndex];
        const unityDisplay = document.querySelector('.unity-equation-display');

        // Update unity equation with current image context
        unityDisplay.innerHTML = `
            <div class="unity-equation-text">1 + 1 = 1</div>
            <div class="unity-subtitle">${currentImage.title}</div>
        `;
    }

    startAutoSlide() {
        setInterval(() => {
            this.nextSlide();
        }, 8000); // Change slide every 8 seconds (φ-harmonic timing)
    }

    calculateMetagamerEnergy() {
        // Metagamer energy equation: E = φ² × C × U
        this.metagamerEnergy = this.phi ** 2 * this.consciousnessDensity * this.unityConvergenceRate;

        // Update consciousness field intensity based on energy
        const field = document.querySelector('.metagamer-energy-field');
        if (field) {
            field.style.opacity = Math.min(this.metagamerEnergy / 10, 0.6);
        }
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

        document.querySelector('.landing-image-container').appendChild(resonance);

        setTimeout(() => {
            resonance.remove();
        }, 2000);
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

// Initialize the landing image slider when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Create loading overlay
    const loadingOverlay = document.createElement('div');
    loadingOverlay.className = 'loading-overlay';
    loadingOverlay.innerHTML = '<div class="loading-spinner"></div>';
    document.body.appendChild(loadingOverlay);

    // Initialize slider
    const slider = new LandingImageSlider();

    // Remove loading overlay after images are loaded
    Promise.all(
        slider.images.map(img => {
            return new Promise((resolve) => {
                const image = new Image();
                image.onload = resolve;
                image.onerror = resolve;
                image.src = img.src;
            });
        })
    ).then(() => {
        setTimeout(() => {
            loadingOverlay.classList.add('fade-out');
            setTimeout(() => {
                loadingOverlay.remove();
            }, 500);
        }, 1000);
    });

    // Add keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft') {
            slider.prevSlide();
        } else if (e.key === 'ArrowRight') {
            slider.nextSlide();
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
                slider.nextSlide();
            } else {
                slider.prevSlide();
            }
        }
    }
});

// Export for potential use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = LandingImageSlider;
} 