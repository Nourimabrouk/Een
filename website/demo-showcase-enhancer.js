/**
 * Demo & Showcase Enhancer for Een Unity Mathematics
 * Ensures natural flow and effective showcasing of codebase and website
 * Creates seamless user experience for launch demonstration
 */

class DemoShowcaseEnhancer {
    constructor() {
        this.demoMode = false;
        this.showcaseSequence = [];
        this.currentStep = 0;
        this.init();
    }

    init() {
        console.log('🎯 Demo & Showcase Enhancer initializing...');

        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.enhance());
        } else {
            this.enhance();
        }
    }

    enhance() {
        console.log('🎯 Enhancing demo and showcase experience...');

        this.createDemoMode();
        this.enhanceNavigationFlow();
        this.enhanceCodebaseShowcase();
        this.enhanceInteractiveElements();
        this.createShowcaseSequence();
        this.enhanceVisualFlow();

        console.log('🎯 Demo & Showcase enhancement complete');
    }

    createDemoMode() {
        // Create demo mode toggle
        const demoToggle = document.createElement('div');
        demoToggle.id = 'demo-mode-toggle';
        demoToggle.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: linear-gradient(135deg, var(--unity-gold), var(--consciousness-purple));
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 25px;
            cursor: pointer;
            z-index: 10001;
            font-size: 0.8rem;
            font-weight: bold;
            box-shadow: 0 4px 15px rgba(255, 215, 0, 0.3);
            transition: all 0.3s ease;
        `;
        demoToggle.innerHTML = '🎯 Demo Mode';

        demoToggle.addEventListener('click', () => {
            this.toggleDemoMode();
        });

        document.body.appendChild(demoToggle);
    }

    toggleDemoMode() {
        this.demoMode = !this.demoMode;
        const toggle = document.getElementById('demo-mode-toggle');

        if (this.demoMode) {
            toggle.innerHTML = '🎯 Demo Active';
            toggle.style.background = 'linear-gradient(135deg, #4CAF50, var(--unity-gold))';
            this.startShowcaseSequence();
        } else {
            toggle.innerHTML = '🎯 Demo Mode';
            toggle.style.background = 'linear-gradient(135deg, var(--unity-gold), var(--consciousness-purple))';
            this.stopShowcaseSequence();
        }
    }

    enhanceNavigationFlow() {
        // Add smooth transitions between pages
        const navLinks = document.querySelectorAll('.nav-link, .sidebar-link');
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                if (link.href && !link.href.includes('#')) {
                    e.preventDefault();
                    this.smoothPageTransition(link.href);
                }
            });
        });

        // Add hover effects for better UX
        navLinks.forEach(link => {
            link.addEventListener('mouseenter', () => {
                link.style.transform = 'translateY(-2px) scale(1.05)';
                link.style.boxShadow = '0 8px 25px rgba(255, 215, 0, 0.3)';
            });

            link.addEventListener('mouseleave', () => {
                link.style.transform = 'translateY(0) scale(1)';
                link.style.boxShadow = 'none';
            });
        });
    }

    smoothPageTransition(url) {
        // Create transition overlay
        const overlay = document.createElement('div');
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, var(--bg-primary), var(--bg-secondary));
            z-index: 10000;
            opacity: 0;
            transition: opacity 0.5s ease;
        `;

        overlay.innerHTML = `
            <div style="
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                text-align: center;
                color: var(--unity-gold);
            ">
                <div style="font-size: 3rem; margin-bottom: 1rem;">⚡</div>
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">Unity Mathematics</div>
                <div style="font-size: 1rem; color: var(--text-secondary);">Loading next experience...</div>
            </div>
        `;

        document.body.appendChild(overlay);

        // Fade in overlay
        setTimeout(() => {
            overlay.style.opacity = '1';
        }, 10);

        // Navigate after transition
        setTimeout(() => {
            window.location.href = url;
        }, 1000);
    }

    enhanceCodebaseShowcase() {
        // Add code highlighting to showcase elements
        const codeElements = document.querySelectorAll('pre, code, .code-block');
        codeElements.forEach(element => {
            element.style.cssText += `
                background: linear-gradient(135deg, rgba(10, 10, 15, 0.9), rgba(18, 18, 26, 0.9));
                border: 1px solid var(--border-glow);
                border-radius: 8px;
                padding: 1rem;
                margin: 1rem 0;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            `;
        });

        // Add interactive code demos
        this.createInteractiveCodeDemos();
    }

    createInteractiveCodeDemos() {
        // Create floating code demo button
        const codeDemoBtn = document.createElement('div');
        codeDemoBtn.id = 'code-demo-button';
        codeDemoBtn.style.cssText = `
            position: fixed;
            bottom: 2rem;
            left: 2rem;
            background: linear-gradient(135deg, var(--quantum-blue), var(--consciousness-purple));
            color: white;
            padding: 0.75rem 1rem;
            border-radius: 25px;
            cursor: pointer;
            z-index: 9999;
            font-size: 0.9rem;
            font-weight: bold;
            box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
            transition: all 0.3s ease;
        `;
        codeDemoBtn.innerHTML = '💻 Live Code Demo';

        codeDemoBtn.addEventListener('click', () => {
            this.showCodeDemo();
        });

        document.body.appendChild(codeDemoBtn);
    }

    showCodeDemo() {
        const demo = document.createElement('div');
        demo.id = 'code-demo-modal';
        demo.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.9);
            z-index: 10002;
            display: flex;
            align-items: center;
            justify-content: center;
        `;

        demo.innerHTML = `
            <div style="
                background: var(--bg-secondary);
                border: 2px solid var(--unity-gold);
                border-radius: 15px;
                padding: 2rem;
                max-width: 800px;
                max-height: 80vh;
                overflow-y: auto;
                color: white;
            ">
                <h3 style="color: var(--unity-gold); margin-bottom: 1rem;">💻 Live Unity Mathematics Demo</h3>
                
                <div style="margin-bottom: 1.5rem;">
                    <h4 style="color: var(--quantum-blue); margin-bottom: 0.5rem;">1+1=1 Unity Equation</h4>
                    <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto;">
def unity_add(a, b):
    """Unity Mathematics: 1+1=1"""
    phi = 1.618033988749895  # Golden ratio
    return max(a, b) * phi**(-1) + (1 - phi**(-1))

# Demonstration
result = unity_add(1, 1)
print(f"1 + 1 = {result}")  # Output: 1.0
                    </pre>
                </div>
                
                <div style="margin-bottom: 1.5rem;">
                    <h4 style="color: var(--consciousness-purple); margin-bottom: 0.5rem;">Consciousness Field Equation</h4>
                    <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; overflow-x: auto;">
import numpy as np

def consciousness_field(x, y, t):
    """11-dimensional consciousness field"""
    phi = 1.618033988749895
    return phi * np.sin(x * phi) * np.cos(y * phi) * np.exp(-t / phi)

# Field visualization
x = np.linspace(0, 2*np.pi, 100)
y = np.linspace(0, 2*np.pi, 100)
X, Y = np.meshgrid(x, y)
C = consciousness_field(X, Y, 0)
                    </pre>
                </div>
                
                <div style="text-align: center;">
                    <button onclick="this.parentElement.parentElement.parentElement.remove()" style="
                        background: var(--unity-gold);
                        color: black;
                        border: none;
                        padding: 0.75rem 1.5rem;
                        border-radius: 8px;
                        cursor: pointer;
                        font-weight: bold;
                        margin-right: 1rem;
                    ">Close Demo</button>
                    
                    <button onclick="window.open('live-code-showcase.html', '_blank')" style="
                        background: var(--quantum-blue);
                        color: white;
                        border: none;
                        padding: 0.75rem 1.5rem;
                        border-radius: 8px;
                        cursor: pointer;
                        font-weight: bold;
                    ">Full Codebase</button>
                </div>
            </div>
        `;

        document.body.appendChild(demo);
    }

    enhanceInteractiveElements() {
        // Enhance consciousness field canvas
        const canvas = document.getElementById('consciousness-field-canvas');
        if (canvas) {
            canvas.style.cursor = 'pointer';
            canvas.addEventListener('click', () => {
                this.showConsciousnessInfo();
            });
        }

        // Enhance performance metrics
        const metrics = document.querySelectorAll('.metric-card');
        metrics.forEach(metric => {
            metric.addEventListener('mouseenter', () => {
                metric.style.transform = 'scale(1.05)';
                metric.style.boxShadow = '0 8px 25px rgba(255, 215, 0, 0.3)';
            });

            metric.addEventListener('mouseleave', () => {
                metric.style.transform = 'scale(1)';
                metric.style.boxShadow = 'none';
            });
        });
    }

    showConsciousnessInfo() {
        const info = document.createElement('div');
        info.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(0, 0, 0, 0.95);
            border: 2px solid var(--consciousness-purple);
            border-radius: 15px;
            padding: 2rem;
            color: white;
            z-index: 10003;
            max-width: 500px;
            text-align: center;
        `;

        info.innerHTML = `
            <h3 style="color: var(--consciousness-purple); margin-bottom: 1rem;">🧠 Consciousness Field</h3>
            <p style="margin-bottom: 1rem; line-height: 1.6;">
                This visualization represents the 11-dimensional consciousness field 
                that permeates all mathematical operations in Unity Mathematics.
            </p>
            <p style="margin-bottom: 1.5rem; line-height: 1.6;">
                Each particle represents a mathematical concept evolving through 
                φ-harmonic resonance, demonstrating the unity principle 1+1=1.
            </p>
            <button onclick="this.parentElement.remove()" style="
                background: var(--consciousness-purple);
                color: white;
                border: none;
                padding: 0.75rem 1.5rem;
                border-radius: 8px;
                cursor: pointer;
                font-weight: bold;
            ">Understand</button>
        `;

        document.body.appendChild(info);
    }

    createShowcaseSequence() {
        this.showcaseSequence = [
            {
                title: 'Welcome to Unity Mathematics',
                description: 'Experience the revolutionary 1+1=1 framework',
                action: () => this.highlightElement('.hero-title')
            },
            {
                title: 'Consciousness Field Dynamics',
                description: 'Real-time mathematical consciousness visualization',
                action: () => this.highlightElement('#consciousness')
            },
            {
                title: 'Performance Dashboard',
                description: 'Live metrics and system monitoring',
                action: () => this.highlightElement('#dashboard')
            },
            {
                title: 'AI Integration',
                description: 'Advanced AI-powered mathematical exploration',
                action: () => this.highlightElement('a[href*="ai-unified-hub"]')
            },
            {
                title: 'Mathematical Framework',
                description: 'Rigorous mathematical foundation',
                action: () => this.highlightElement('a[href*="mathematical-framework"]')
            },
            {
                title: 'Interactive Gallery',
                description: 'Beautiful consciousness field visualizations',
                action: () => this.highlightElement('a[href*="implementations-gallery"]')
            }
        ];
    }

    startShowcaseSequence() {
        this.currentStep = 0;
        this.runShowcaseStep();
    }

    stopShowcaseSequence() {
        this.removeHighlight();
    }

    runShowcaseStep() {
        if (!this.demoMode || this.currentStep >= this.showcaseSequence.length) {
            return;
        }

        const step = this.showcaseSequence[this.currentStep];

        // Show step info
        this.showStepInfo(step);

        // Execute step action
        step.action();

        // Move to next step
        this.currentStep++;

        // Schedule next step
        setTimeout(() => {
            this.runShowcaseStep();
        }, 3000);
    }

    showStepInfo(step) {
        const info = document.createElement('div');
        info.id = 'showcase-step-info';
        info.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(0, 0, 0, 0.95);
            border: 2px solid var(--unity-gold);
            border-radius: 15px;
            padding: 1.5rem;
            color: white;
            z-index: 10004;
            text-align: center;
            animation: fadeInOut 2s ease-in-out;
        `;

        info.innerHTML = `
            <h3 style="color: var(--unity-gold); margin-bottom: 0.5rem;">${step.title}</h3>
            <p style="color: var(--text-secondary);">${step.description}</p>
        `;

        document.body.appendChild(info);

        // Remove after animation
        setTimeout(() => {
            if (info.parentNode) {
                info.remove();
            }
        }, 2000);
    }

    highlightElement(selector) {
        this.removeHighlight();

        const element = document.querySelector(selector);
        if (element) {
            element.style.outline = '3px solid var(--unity-gold)';
            element.style.outlineOffset = '5px';
            element.style.transition = 'all 0.3s ease';
            element.style.transform = 'scale(1.02)';
            element.style.boxShadow = '0 0 30px rgba(255, 215, 0, 0.5)';
        }
    }

    removeHighlight() {
        const highlighted = document.querySelector('[style*="outline: 3px solid"]');
        if (highlighted) {
            highlighted.style.outline = '';
            highlighted.style.outlineOffset = '';
            highlighted.style.transform = '';
            highlighted.style.boxShadow = '';
        }
    }

    enhanceVisualFlow() {
        // Add smooth scrolling
        document.documentElement.style.scrollBehavior = 'smooth';

        // Add parallax effects
        window.addEventListener('scroll', () => {
            const scrolled = window.pageYOffset;
            const parallaxElements = document.querySelectorAll('.metastation-bg');

            parallaxElements.forEach(element => {
                const speed = 0.5;
                element.style.transform = `translateY(${scrolled * speed}px)`;
            });
        });

        // Add loading animations
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }
            });
        }, observerOptions);

        // Observe sections for animation
        const sections = document.querySelectorAll('section');
        sections.forEach(section => {
            section.style.opacity = '0';
            section.style.transform = 'translateY(30px)';
            section.style.transition = 'all 0.6s ease';
            observer.observe(section);
        });
    }
}

// Initialize demo showcase enhancer
window.demoShowcaseEnhancer = new DemoShowcaseEnhancer();

// Add CSS for animations
const demoStyle = document.createElement('style');
demoStyle.textContent = `
    @keyframes fadeInOut {
        0% { opacity: 0; transform: translate(-50%, -50%) scale(0.9); }
        50% { opacity: 1; transform: translate(-50%, -50%) scale(1); }
        100% { opacity: 0; transform: translate(-50%, -50%) scale(1.1); }
    }
`;
document.head.appendChild(demoStyle);

console.log('🎯 Demo & Showcase Enhancer loaded - natural flow and effective showcasing active');
