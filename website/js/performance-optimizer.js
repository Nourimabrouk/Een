/**
 * Performance Optimizer for Een Unity Mathematics
 * Reduces console logging, optimizes animations, and improves performance
 * Version: 1.0.0 - One-Shot Performance Enhancement
 */

class PerformanceOptimizer {
    constructor() {
        this.optimizations = {
            consoleReduced: false,
            animationsOptimized: false,
            webglOptimized: false,
            accessibilityEnhanced: false
        };
        
        this.init();
    }
    
    init() {
        this.reduceConsoleLogging();
        this.optimizeAnimations();
        this.enhanceWebGLPerformance();
        this.improveAccessibility();
        this.optimizeTouchTargets();
        
        console.log('⚡ Performance Optimizer initialized');
    }
    
    reduceConsoleLogging() {
        if (this.optimizations.consoleReduced) return;
        
        // Store original console methods
        const originalLog = console.log;
        const originalWarn = console.warn;
        const originalError = console.error;
        
        // Reduce logging in production
        if (window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
            console.log = function() {
                // Only log critical messages in production
                if (arguments[0] && typeof arguments[0] === 'string' && 
                    (arguments[0].includes('🚀') || arguments[0].includes('🌟') || arguments[0].includes('✅'))) {
                    originalLog.apply(console, arguments);
                }
            };
            
            console.warn = function() {
                // Only log important warnings
                if (arguments[0] && typeof arguments[0] === 'string' && 
                    (arguments[0].includes('⚠️') || arguments[0].includes('Error'))) {
                    originalWarn.apply(console, arguments);
                }
            };
        }
        
        this.optimizations.consoleReduced = true;
    }
    
    optimizeAnimations() {
        if (this.optimizations.animationsOptimized) return;
        
        // Optimize CSS animations
        const style = document.createElement('style');
        style.textContent = `
            /* Optimize animations for performance */
            * {
                will-change: auto;
            }
            
            .animated-element {
                will-change: transform, opacity;
            }
            
            /* Reduce motion for users who prefer it */
            @media (prefers-reduced-motion: reduce) {
                *, *::before, *::after {
                    animation-duration: 0.01ms !important;
                    animation-iteration-count: 1 !important;
                    transition-duration: 0.01ms !important;
                    scroll-behavior: auto !important;
                }
            }
            
            /* Optimize GPU acceleration */
            .gpu-accelerated {
                transform: translateZ(0);
                backface-visibility: hidden;
                perspective: 1000px;
            }
        `;
        document.head.appendChild(style);
        
        // Optimize GSAP animations if available
        if (typeof gsap !== 'undefined') {
            gsap.config({
                nullTargetWarn: false,
                trialWarn: false
            });
        }
        
        this.optimizations.animationsOptimized = true;
    }
    
    enhanceWebGLPerformance() {
        if (this.optimizations.webglOptimized) return;
        
        // WebGL performance optimizations
        const canvas = document.querySelector('canvas');
        if (canvas) {
            const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
            if (gl) {
                // Enable performance optimizations
                gl.getExtension('OES_standard_derivatives');
                gl.getExtension('OES_element_index_uint');
                gl.getExtension('WEBGL_depth_texture');
                
                // Set performance hints
                gl.hint(gl.GENERATE_MIPMAP_HINT, gl.FASTEST);
            }
        }
        
        // Three.js optimizations
        if (typeof THREE !== 'undefined') {
            // Optimize Three.js renderer settings
            const renderers = document.querySelectorAll('canvas');
            renderers.forEach(canvas => {
                if (canvas.__threeRenderer) {
                    canvas.__threeRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                    canvas.__threeRenderer.shadowMap.enabled = false; // Disable shadows for performance
                }
            });
        }
        
        this.optimizations.webglOptimized = true;
    }
    
    improveAccessibility() {
        if (this.optimizations.accessibilityEnhanced) return;
        
        // Add missing ARIA labels
        const buttons = document.querySelectorAll('button:not([aria-label]):not([aria-labelledby])');
        buttons.forEach(button => {
            if (button.textContent.trim()) {
                button.setAttribute('aria-label', button.textContent.trim());
            }
        });
        
        // Improve color contrast
        const style = document.createElement('style');
        style.textContent = `
            /* Ensure minimum color contrast */
            .btn, .nav-link, .feature-card {
                color: #FFFFFF !important;
            }
            
            .btn-secondary {
                color: #FFD700 !important;
                border-color: #FFD700 !important;
            }
            
            /* High contrast mode support */
            @media (prefers-contrast: high) {
                :root {
                    --unity-gold: #FFFF00;
                    --text-primary: #FFFFFF;
                    --text-secondary: #CCCCCC;
                }
            }
        `;
        document.head.appendChild(style);
        
        // Add skip links
        const skipLink = document.createElement('a');
        skipLink.href = '#main-content';
        skipLink.textContent = 'Skip to main content';
        skipLink.style.cssText = `
            position: absolute;
            top: -40px;
            left: 6px;
            background: #FFD700;
            color: #000;
            padding: 8px;
            text-decoration: none;
            z-index: 10000;
        `;
        skipLink.addEventListener('focus', () => {
            skipLink.style.top = '6px';
        });
        skipLink.addEventListener('blur', () => {
            skipLink.style.top = '-40px';
        });
        document.body.insertBefore(skipLink, document.body.firstChild);
        
        this.optimizations.accessibilityEnhanced = true;
    }
    
    optimizeTouchTargets() {
        // Ensure minimum touch target size (44px)
        const style = document.createElement('style');
        style.textContent = `
            /* Ensure minimum touch target size */
            button, a, input[type="button"], input[type="submit"], input[type="reset"] {
                min-height: 44px;
                min-width: 44px;
            }
            
            /* Mobile optimizations */
            @media (max-width: 768px) {
                .btn, .nav-link {
                    padding: 12px 16px;
                    font-size: 16px; /* Prevent zoom on iOS */
                }
                
                input, textarea, select {
                    font-size: 16px; /* Prevent zoom on iOS */
                }
            }
        `;
        document.head.appendChild(style);
    }
    
    getOptimizationStatus() {
        return this.optimizations;
    }
}

// Initialize performance optimizer
const performanceOptimizer = new PerformanceOptimizer();