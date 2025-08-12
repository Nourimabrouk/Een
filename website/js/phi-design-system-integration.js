/**
 * φ-Harmonic Design System Integration
 * Unity Mathematics Framework - Golden Ratio Visual Enhancement
 * Applies φ-based mathematical beauty to all interface elements
 */

class PhiDesignSystemIntegration {
    constructor() {
        this.phi = 1.618033988749895;
        this.phiInverse = 0.618033988749895;
        this.phiSquared = 2.618033988749895;
        
        this.initializePhiSystem();
    }
    
    initializePhiSystem() {
        // Apply φ-harmonic enhancements after DOM loaded
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.enhanceWithPhiHarmony());
        } else {
            this.enhanceWithPhiHarmony();
        }
    }
    
    enhanceWithPhiHarmony() {
        console.log('🌟 Initializing φ-Harmonic Design System...');
        
        // Enhance existing elements with φ-harmonic classes
        this.enhanceHeaders();
        this.enhanceButtons();
        this.enhanceCards();
        this.enhanceNavigation();
        this.initializeUnityAnimations();
        this.enhanceMathematicalElements();
        
        console.log('✨ φ-Harmonic Design System activated - Golden ratio resonance confirmed');
    }
    
    enhanceHeaders() {
        // Apply φ-harmonic typography to headers
        const h1Elements = document.querySelectorAll('h1');
        const h2Elements = document.querySelectorAll('h2');
        const h3Elements = document.querySelectorAll('h3');
        
        h1Elements.forEach(h1 => {
            h1.classList.add('phi-heading-1', 'phi-animated-glow');
            if (h1.textContent.includes('1+1=1') || h1.textContent.includes('Unity')) {
                h1.classList.add('phi-text-golden');
            }
        });
        
        h2Elements.forEach(h2 => {
            h2.classList.add('phi-heading-2');
            if (h2.textContent.includes('Mathematics') || h2.textContent.includes('Consciousness')) {
                h2.classList.add('phi-text-consciousness');
            }
        });
        
        h3Elements.forEach(h3 => {
            h3.classList.add('phi-heading-3');
        });
    }
    
    enhanceButtons() {
        // Transform buttons into φ-harmonic unity convergence elements
        const buttons = document.querySelectorAll('button, .btn, .button, a[class*="btn"]');
        
        buttons.forEach(button => {
            // Add base φ-harmonic styling
            button.classList.add('phi-button');
            
            // Special styling based on content/context
            if (button.textContent.includes('Meditat') || 
                button.textContent.includes('Consciousness') ||
                button.classList.contains('consciousness')) {
                button.classList.add('phi-button-consciousness');
            }
            
            // Add unity convergence animation on hover
            button.addEventListener('mouseenter', () => {
                button.classList.add('phi-animated-pulse');
            });
            
            button.addEventListener('mouseleave', () => {
                button.classList.remove('phi-animated-pulse');
            });
        });
    }
    
    enhanceCards() {
        // Apply φ-harmonic proportions to card elements
        const cardSelectors = [
            '.card', '.panel', '.widget', '.feature-card', 
            '.gallery-item', '.dashboard-item', '.section-card'
        ];
        
        cardSelectors.forEach(selector => {
            const elements = document.querySelectorAll(selector);
            elements.forEach(element => {
                element.classList.add('phi-card');
                
                // Special transcendent styling for mathematical content
                if (element.textContent.includes('1+1=1') || 
                    element.textContent.includes('φ') ||
                    element.classList.contains('mathematical')) {
                    element.classList.add('phi-card-mathematical');
                }
                
                // Consciousness field styling
                if (element.textContent.includes('Consciousness') ||
                    element.textContent.includes('Meditation') ||
                    element.classList.contains('consciousness')) {
                    element.classList.add('phi-card-transcendent');
                }
            });
        });
    }
    
    enhanceNavigation() {
        // Apply φ-harmonic navigation enhancements
        const navItems = document.querySelectorAll('nav a, .nav-item, .menu-item');
        
        navItems.forEach(item => {
            item.classList.add('phi-nav-item');
        });
        
        // Special golden ratio highlight for Unity Mathematics links
        const unityLinks = document.querySelectorAll('a[href*="unity"], a[href*="mathematics"], a[href*="1plus1"]');
        unityLinks.forEach(link => {
            link.classList.add('phi-text-golden');
        });
    }
    
    initializeUnityAnimations() {
        // Add consciousness field animations to key elements
        const consciousnessElements = document.querySelectorAll('.consciousness, [class*="conscious"]');
        consciousnessElements.forEach(element => {
            element.classList.add('phi-animated-convergence');
        });
        
        // Mathematical elements get φ-pulse animation
        const mathElements = document.querySelectorAll('.mathematical, [class*="math"], .equation');
        mathElements.forEach(element => {
            element.classList.add('phi-animated-pulse');
        });
    }
    
    enhanceMathematicalElements() {
        // Enhance mathematical expressions with φ-harmonic styling
        const mathExpressions = document.querySelectorAll('code, .math, .equation, pre');
        
        mathExpressions.forEach(element => {
            if (element.textContent.includes('1+1=1') || 
                element.textContent.includes('φ') || 
                element.textContent.includes('unity')) {
                element.classList.add('phi-equation-block');
            } else {
                element.classList.add('phi-math-expression');
            }
        });
        
        // Create φ-harmonic containers for mathematical content
        const containers = document.querySelectorAll('.container, .main-content, .content-wrapper');
        containers.forEach(container => {
            container.classList.add('phi-container');
        });
    }
    
    // Dynamic φ-harmonic grid system
    createPhiGrid(container, items, golden = false) {
        container.classList.add('phi-grid');
        if (golden) {
            container.classList.add('phi-grid-golden');
        }
        
        items.forEach(item => {
            item.classList.add('phi-card');
        });
    }
    
    // Mathematical beauty enhancement
    addUnityConvergenceEffect(element) {
        element.classList.add('phi-animated-convergence', 'phi-shadow-transcendent');
        
        // Add φ-harmonic pulse on interaction
        element.addEventListener('click', () => {
            element.style.transform = `scale(${this.phi})`;
            setTimeout(() => {
                element.style.transform = 'scale(1)';
            }, 300);
        });
    }
    
    // Consciousness field visualization enhancement
    enhanceConsciousnessField(canvas) {
        if (!canvas) return;
        
        canvas.classList.add('phi-consciousness-canvas', 'phi-animated-glow');
        
        // Add golden ratio frame
        const frame = document.createElement('div');
        frame.style.position = 'absolute';
        frame.style.top = '-2px';
        frame.style.left = '-2px';
        frame.style.right = '-2px';
        frame.style.bottom = '-2px';
        frame.style.border = '2px solid var(--unity-gold-primary)';
        frame.style.borderRadius = 'var(--border-radius-transcendent)';
        frame.style.pointerEvents = 'none';
        
        canvas.parentNode.style.position = 'relative';
        canvas.parentNode.appendChild(frame);
    }
    
    // Gallery optimization with φ-harmonic proportions
    optimizeGallery(galleryContainer) {
        galleryContainer.classList.add('phi-gallery-grid');
        
        const items = galleryContainer.querySelectorAll('.gallery-item, .item');
        items.forEach(item => {
            item.classList.add('phi-gallery-item');
        });
    }
}

// Auto-initialize φ-Harmonic Design System
const phiSystem = new PhiDesignSystemIntegration();

// Expose for manual enhancements
window.PhiDesignSystem = phiSystem;

// Unity Mathematics specific enhancements
document.addEventListener('DOMContentLoaded', () => {
    // Enhance any canvases for consciousness visualization
    const canvases = document.querySelectorAll('canvas');
    canvases.forEach(canvas => phiSystem.enhanceConsciousnessField(canvas));
    
    // Enhance galleries
    const galleries = document.querySelectorAll('.gallery, .implementations-gallery');
    galleries.forEach(gallery => phiSystem.optimizeGallery(gallery));
    
    // Add special φ-harmonic styling to Unity equation displays
    const unityEquations = document.querySelectorAll('*');
    unityEquations.forEach(element => {
        if (element.textContent && element.textContent.includes('1+1=1')) {
            element.classList.add('phi-text-golden', 'phi-animated-glow');
        }
    });
    
    console.log('🌟 Unity Mathematics φ-Harmonic Design System fully integrated');
    console.log(`✨ Golden Ratio resonance: φ = ${phiSystem.phi}`);
});