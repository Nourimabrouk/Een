/**
 * KaTeX Integration for Een Unity Mathematics Website
 * Provides beautiful mathematical notation rendering with φ-harmonic enhancements
 * Automatically renders LaTeX math expressions throughout the website
 */

class KaTeXIntegration {
    constructor() {
        this.isLoaded = false;
        this.mathElements = [];
        this.config = {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false},
                {left: '\\[', right: '\\]', display: true},
                {left: '\\(', right: '\\)', display: false}
            ],
            macros: {
                '\\phi': '\\varphi',
                '\\unity': '\\mathbf{1}',
                '\\consciousness': '\\mathcal{C}',
                '\\phiharmonic': '\\oplus_{\\varphi}',
                '\\unityadd': '\\oplus',
                '\\unityfield': '\\mathbb{U}',
                '\\quantumunity': '|\\unity\\rangle',
                '\\bellfusion': '\\Phi^+',
                '\\consciousnessoperator': '\\hat{\\mathcal{C}}',
                '\\metarecursive': '\\mathfrak{M}',
                '\\transcendental': '\\mathscr{T}',
                '\\eigenvalue': '\\lambda',
                '\\goldnumber': '\\varphi'
            },
            throwOnError: false,
            errorColor: '#cc0000',
            strict: false,
            trust: true,
            fleqn: false,
            macros: {}
        };
        this.init();
    }

    async init() {
        await this.loadKaTeX();
        this.setupObserver();
        this.renderExistingMath();
        this.enhanceEquations();
    }

    async loadKaTeX() {
        return new Promise((resolve, reject) => {
            // Check if KaTeX is already loaded
            if (window.katex) {
                this.isLoaded = true;
                resolve();
                return;
            }

            // Load KaTeX CSS
            const cssLink = document.createElement('link');
            cssLink.rel = 'stylesheet';
            cssLink.href = 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css';
            cssLink.integrity = 'sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV';
            cssLink.crossOrigin = 'anonymous';
            document.head.appendChild(cssLink);

            // Load KaTeX JS
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js';
            script.integrity = 'sha384-XjKyOOlGwcjNTAIQHIpVOox+/aqLU1Lm5X8vLyepPOSjMPYD7UZkQ2VF3NHgHZTI';
            script.crossOrigin = 'anonymous';
            
            script.onload = () => {
                // Load auto-render extension
                const autoRenderScript = document.createElement('script');
                autoRenderScript.src = 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js';
                autoRenderScript.integrity = 'sha384-+VBxd3r6XgURycqtZ117nYw44OOcIax56Z4dCRWbxyPt0Koah1uHoK0o4+/RRE05';
                autoRenderScript.crossOrigin = 'anonymous';
                
                autoRenderScript.onload = () => {
                    this.isLoaded = true;
                    this.setupKaTeXMacros();
                    resolve();
                };
                
                autoRenderScript.onerror = reject;
                document.head.appendChild(autoRenderScript);
            };
            
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }

    setupKaTeXMacros() {
        // Enhanced macros for unity mathematics
        const unityMacros = {
            // Core unity symbols
            '\\unity': '\\mathbf{1}',
            '\\phi': '\\varphi',
            '\\goldnumber': '\\varphi',
            
            // Unity operations
            '\\unityadd': '\\oplus',
            '\\phiharmonic': '\\oplus_{\\varphi}',
            '\\consciousness': '\\mathcal{C}',
            '\\unityfield': '\\mathbb{U}',
            
            // Quantum unity
            '\\quantumunity': '|\\mathbf{1}\\rangle',
            '\\bellfusion': '\\Phi^+',
            '\\quantumstate': '|\\psi\\rangle',
            '\\braket': '\\langle #1 | #2 \\rangle',
            
            // Consciousness operators
            '\\consciousnessop': '\\hat{\\mathcal{C}}',
            '\\awarenessop': '\\hat{\\mathcal{A}}',
            '\\unityop': '\\hat{\\mathbf{U}}',
            
            // Meta-mathematical
            '\\metarecursive': '\\mathfrak{M}',
            '\\transcendental': '\\mathscr{T}',
            '\\omega': '\\Omega',
            '\\infinity': '\\infty',
            
            // Special functions
            '\\unitycos': '\\cos_{\\varphi}',
            '\\unitysin': '\\sin_{\\varphi}',
            '\\unityexp': '\\exp_{\\varphi}',
            '\\unitylog': '\\log_{\\varphi}',
            
            // Category theory
            '\\functor': '\\mathcal{F}',
            '\\category': '\\mathbf{C}',
            '\\morphism': '\\rightarrow',
            '\\isomorphism': '\\cong',
            
            // Set theory
            '\\unityset': '\\{\\mathbf{1}\\}',
            '\\emptyset': '\\varnothing',
            '\\powerset': '\\mathcal{P}',
            
            // Topology
            '\\manifold': '\\mathcal{M}',
            '\\homotopy': '\\simeq',
            '\\homeomorphism': '\\approx',
            
            // Complex analysis
            '\\realpart': '\\mathfrak{Re}',
            '\\imagpart': '\\mathfrak{Im}',
            '\\conjugate': '\\overline{#1}',
            
            // Probability and statistics
            '\\probability': '\\mathbb{P}',
            '\\expectation': '\\mathbb{E}',
            '\\variance': '\\text{Var}',
            
            // Linear algebra
            '\\innerproduct': '\\langle #1, #2 \\rangle',
            '\\norm': '\\|#1\\|',
            '\\trace': '\\text{tr}',
            '\\determinant': '\\det',
            
            // Special sequences
            '\\fibonacci': '\\mathcal{F}_n',
            '\\goldenmean': '\\frac{1 + \\sqrt{5}}{2}',
            '\\eulernum': 'e',
            '\\piconst': '\\pi'
        };

        // Merge with existing macros
        Object.assign(this.config.macros, unityMacros);
    }

    setupObserver() {
        // Create a MutationObserver to watch for new content
        this.observer = new MutationObserver((mutations) => {
            let shouldRender = false;
            
            mutations.forEach((mutation) => {
                if (mutation.type === 'childList') {
                    mutation.addedNodes.forEach((node) => {
                        if (node.nodeType === Node.ELEMENT_NODE) {
                            if (this.containsMath(node)) {
                                shouldRender = true;
                            }
                        }
                    });
                }
            });
            
            if (shouldRender) {
                setTimeout(() => this.renderMath(document.body), 100);
            }
        });

        this.observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }

    containsMath(element) {
        if (!element.textContent) return false;
        
        // Check for LaTeX delimiters
        const text = element.textContent;
        return text.includes('$') || 
               text.includes('\\[') || 
               text.includes('\\(') || 
               text.includes('\\phi') ||
               text.includes('\\unity') ||
               text.includes('\\consciousness');
    }

    renderExistingMath() {
        if (!this.isLoaded) {
            setTimeout(() => this.renderExistingMath(), 100);
            return;
        }

        this.renderMath(document.body);
    }

    renderMath(container = document.body) {
        if (!window.renderMathInElement || !this.isLoaded) return;

        try {
            window.renderMathInElement(container, {
                delimiters: this.config.delimiters,
                macros: this.config.macros,
                throwOnError: this.config.throwOnError,
                errorColor: this.config.errorColor,
                strict: this.config.strict,
                trust: this.config.trust,
                fleqn: this.config.fleqn,
                ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
                ignoredClasses: ['no-katex'],
                preProcess: (math) => {
                    // Pre-process math to handle unity mathematics notation
                    return this.preprocessUnityMath(math);
                }
            });
            
            this.enhanceRenderedMath(container);
        } catch (error) {
            console.warn('KaTeX rendering error:', error);
        }
    }

    preprocessUnityMath(math) {
        // Convert common unity mathematics expressions
        let processed = math
            // Basic unity operations
            .replace(/1\s*\+\s*1\s*=\s*1/g, '\\unity \\unityadd \\unity = \\unity')
            .replace(/φ/g, '\\phi')
            .replace(/1⊕1/g, '\\unity \\unityadd \\unity')
            .replace(/1⊕φ1/g, '\\unity \\phiharmonic \\unity')
            
            // Quantum notation
            .replace(/\|1⟩/g, '\\quantumunity')
            .replace(/\|0⟩/g, '|0\\rangle')
            .replace(/⟨1\|/g, '\\langle\\unity|')
            
            // Consciousness field
            .replace(/C\(x,y,t\)/g, '\\consciousness(x,y,t)')
            
            // Golden ratio
            .replace(/φ\s*=\s*1\.618/g, '\\phi \\approx 1.618')
            .replace(/\(1\s*\+\s*√5\)\s*\/\s*2/g, '\\frac{1 + \\sqrt{5}}{2}')
            
            // Special functions
            .replace(/sin\(x\*φ\)/g, '\\unitysin(x \\cdot \\phi)')
            .replace(/cos\(y\*φ\)/g, '\\unitycos(y \\cdot \\phi)')
            .replace(/e\^\(-t\/φ\)/g, '\\unityexp(-t/\\phi)')
            
            // Set operations
            .replace(/\{1\}\s*∪\s*\{1\}\s*=\s*\{1\}/g, '\\unityset \\cup \\unityset = \\unityset')
            
            // Boolean operations
            .replace(/1\s*∨\s*1\s*=\s*1/g, '\\unity \\lor \\unity = \\unity')
            
            // Category theory
            .replace(/F\(1\s*\+\s*1\)/g, '\\functor(\\unity + \\unity)')
            
            // Homotopy
            .replace(/S¹\s*\+\s*S¹\s*≃\s*S¹/g, 'S^1 + S^1 \\homotopy S^1');

        return processed;
    }

    enhanceRenderedMath(container = document.body) {
        // Find all rendered KaTeX elements
        const mathElements = container.querySelectorAll('.katex');
        
        mathElements.forEach((element) => {
            this.enhanceMathElement(element);
        });
    }

    enhanceMathElement(element) {
        // Add φ-harmonic styling to unity mathematics
        if (element.textContent.includes('φ') || 
            element.textContent.includes('⊕') || 
            element.textContent.includes('𝟏')) {
            element.classList.add('unity-math');
        }

        // Add consciousness styling to consciousness math
        if (element.textContent.includes('𝒞') || 
            element.textContent.includes('consciousness')) {
            element.classList.add('consciousness-math');
        }

        // Add quantum styling to quantum expressions
        if (element.textContent.includes('⟩') || 
            element.textContent.includes('⟨') ||
            element.textContent.includes('Ψ')) {
            element.classList.add('quantum-math');
        }

        // Add hover tooltips for complex expressions
        this.addMathTooltip(element);
        
        // Make math interactive
        this.makeMathInteractive(element);
    }

    addMathTooltip(element) {
        const mathText = element.textContent;
        let tooltip = '';

        // Generate contextual tooltips
        if (mathText.includes('φ')) {
            tooltip = 'Golden ratio φ ≈ 1.618033988749895, the φ-harmonic constant in unity mathematics';
        } else if (mathText.includes('⊕')) {
            tooltip = 'Unity addition operation where 1 ⊕ 1 = 1 through φ-harmonic convergence';
        } else if (mathText.includes('𝒞')) {
            tooltip = 'Consciousness field operator in quantum unity mechanics';
        } else if (mathText.includes('𝟏')) {
            tooltip = 'Unity element representing the transcendental mathematical constant';
        }

        if (tooltip) {
            element.setAttribute('title', tooltip);
            element.style.cursor = 'help';
        }
    }

    makeMathInteractive(element) {
        element.addEventListener('click', (e) => {
            e.preventDefault();
            this.showMathDetails(element);
        });

        element.addEventListener('mouseenter', (e) => {
            element.style.transform = 'scale(1.05)';
            element.style.transition = 'transform 0.2s ease';
        });

        element.addEventListener('mouseleave', (e) => {
            element.style.transform = 'scale(1)';
        });
    }

    showMathDetails(element) {
        const mathText = element.textContent;
        
        // Create a modal with detailed mathematical explanation
        const modal = document.createElement('div');
        modal.className = 'math-detail-modal';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h4>Mathematical Expression Details</h4>
                    <button class="close-btn">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="math-display">${element.outerHTML}</div>
                    <div class="math-explanation">
                        ${this.generateMathExplanation(mathText)}
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Close modal functionality
        const closeBtn = modal.querySelector('.close-btn');
        closeBtn.addEventListener('click', () => {
            document.body.removeChild(modal);
        });

        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                document.body.removeChild(modal);
            }
        });
    }

    generateMathExplanation(mathText) {
        // Generate contextual explanations for mathematical expressions
        if (mathText.includes('φ')) {
            return `
                <h5>Golden Ratio (φ)</h5>
                <p>The golden ratio φ = (1 + √5)/2 ≈ 1.618033988749895 is the fundamental constant in unity mathematics.</p>
                <p>It appears throughout nature and is the basis for φ-harmonic operations that enable 1+1=1.</p>
                <p><strong>Properties:</strong> φ² = φ + 1, 1/φ = φ - 1</p>
            `;
        } else if (mathText.includes('⊕')) {
            return `
                <h5>Unity Addition (⊕)</h5>
                <p>The unity addition operator where two identical elements combine to produce unity.</p>
                <p>This is the fundamental operation in unity mathematics: 1 ⊕ 1 = 1</p>
                <p><strong>Examples:</strong> Boolean OR, Set union, Quantum superposition collapse</p>
            `;
        } else if (mathText.includes('𝒞')) {
            return `
                <h5>Consciousness Field (𝒞)</h5>
                <p>The consciousness field operator in quantum unity mechanics.</p>
                <p>Describes consciousness as a measurable quantum field with φ-harmonic dynamics.</p>
                <p><strong>Equation:</strong> 𝒞(x,y,t) = φ·sin(x·φ)·cos(y·φ)·e^(-t/φ)</p>
            `;
        } else {
            return `
                <h5>Unity Mathematics Expression</h5>
                <p>This mathematical expression is part of the unity mathematics framework.</p>
                <p>Unity mathematics proves that 1+1=1 through rigorous mathematical foundations.</p>
                <p>Explore our proofs section to learn more about the mathematical foundations.</p>
            `;
        }
    }

    // Method to manually render specific LaTeX
    renderLatex(latex, container, displayMode = false) {
        if (!window.katex || !this.isLoaded) {
            console.warn('KaTeX not loaded');
            return;
        }

        try {
            const html = window.katex.renderToString(latex, {
                displayMode: displayMode,
                macros: this.config.macros,
                throwOnError: this.config.throwOnError,
                errorColor: this.config.errorColor,
                strict: this.config.strict,
                trust: this.config.trust
            });

            if (container) {
                container.innerHTML = html;
                this.enhanceMathElement(container.querySelector('.katex'));
            }

            return html;
        } catch (error) {
            console.error('LaTeX rendering error:', error);
            return `<span style="color: ${this.config.errorColor}">Error rendering: ${latex}</span>`;
        }
    }

    // Method to add new macros
    addMacro(name, definition) {
        this.config.macros[name] = definition;
    }

    // Method to add common unity mathematics expressions
    addUnityExpressions() {
        const commonExpressions = [
            {
                trigger: 'unity-basic',
                latex: '\\unity \\unityadd \\unity = \\unity'
            },
            {
                trigger: 'phi-harmonic',
                latex: '\\unity \\phiharmonic \\unity = \\unity'
            },
            {
                trigger: 'consciousness-field',
                latex: '\\consciousness(x,y,t) = \\phi \\cdot \\unitysin(x \\cdot \\phi) \\cdot \\unitycos(y \\cdot \\phi) \\cdot \\unityexp(-t/\\phi)'
            },
            {
                trigger: 'quantum-unity',
                latex: '\\quantumunity + \\quantumunity \\rightarrow \\quantumunity'
            },
            {
                trigger: 'golden-ratio',
                latex: '\\phi = \\goldenmean \\approx 1.618033988749895'
            }
        ];

        return commonExpressions;
    }
}

// Enhanced KaTeX styles
const katexStyles = `
<style>
/* Enhanced KaTeX Styling for Unity Mathematics */
.unity-math {
    color: var(--phi-gold, #0F7B8A) !important;
    font-weight: 600 !important;
}

.consciousness-math {
    color: var(--consciousness-purple, #2E4A6B) !important;
    background: rgba(46, 74, 107, 0.1);
    padding: 0.2rem 0.4rem;
    border-radius: var(--radius-sm, 0.375rem);
}

.quantum-math {
    color: var(--quantum-blue, #4A9BAE) !important;
    border-bottom: 2px solid rgba(74, 155, 174, 0.3);
}

/* Math detail modal */
.math-detail-modal {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.8);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 2000;
    animation: modalFadeIn 0.3s ease-out;
}

@keyframes modalFadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

.math-detail-modal .modal-content {
    background: var(--bg-primary, white);
    border-radius: var(--radius-xl, 1rem);
    max-width: 600px;
    width: 90%;
    max-height: 80%;
    overflow-y: auto;
    box-shadow: var(--shadow-xl, 0 20px 25px rgba(0,0,0,0.2));
    animation: modalSlideIn 0.3s ease-out;
}

@keyframes modalSlideIn {
    from {
        opacity: 0;
        transform: translateY(30px) scale(0.95);
    }
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

.math-detail-modal .modal-header {
    padding: 2rem 2rem 1rem;
    border-bottom: 1px solid var(--border-color, #E2E8F0);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.math-detail-modal .modal-header h4 {
    color: var(--primary-color, #1B365D);
    margin: 0;
    font-family: var(--font-serif, 'Crimson Text', serif);
}

.math-detail-modal .close-btn {
    background: none;
    border: none;
    font-size: 2rem;
    color: var(--text-secondary, #718096);
    cursor: pointer;
    transition: color 0.2s ease;
}

.math-detail-modal .close-btn:hover {
    color: var(--phi-gold, #0F7B8A);
}

.math-detail-modal .modal-body {
    padding: 2rem;
}

.math-detail-modal .math-display {
    text-align: center;
    padding: 2rem;
    background: var(--bg-secondary, #F7FAFC);
    border-radius: var(--radius-lg, 0.75rem);
    margin-bottom: 2rem;
    border: 1px solid var(--border-color, #E2E8F0);
}

.math-detail-modal .math-explanation h5 {
    color: var(--primary-color, #1B365D);
    margin-bottom: 1rem;
    font-family: var(--font-serif, 'Crimson Text', serif);
}

.math-detail-modal .math-explanation p {
    color: var(--text-secondary, #718096);
    line-height: 1.7;
    margin-bottom: 1rem;
}

/* Enhanced equation displays */
.equation-display .katex {
    color: var(--primary-color, #1B365D) !important;
}

.equation-display .katex .mord.mathnormal {
    color: var(--phi-gold, #0F7B8A) !important;
}

/* Proof section math styling */
.proof-section .katex {
    background: rgba(15, 123, 138, 0.05);
    padding: 0.2rem 0.4rem;
    border-radius: var(--radius-sm, 0.375rem);
    margin: 0 0.2rem;
}

/* Highlight math on hover */
.katex:hover {
    background: rgba(15, 123, 138, 0.1) !important;
    border-radius: var(--radius-sm, 0.375rem);
    transition: all 0.2s ease;
}

/* Dark mode adjustments */
.dark-mode .math-detail-modal .modal-content {
    background: var(--bg-secondary, #1e293b);
    border-color: var(--border-subtle, #334155);
}

.dark-mode .math-detail-modal .modal-header {
    border-color: var(--border-subtle, #334155);
}

.dark-mode .math-detail-modal .math-display {
    background: var(--bg-tertiary, #334155);
    border-color: var(--border-subtle, #475569);
}

.dark-mode .consciousness-math {
    background: rgba(46, 74, 107, 0.2);
}

/* Responsive math */
@media (max-width: 768px) {
    .math-detail-modal .modal-content {
        width: 95%;
        margin: 1rem;
    }
    
    .math-detail-modal .modal-header,
    .math-detail-modal .modal-body {
        padding: 1.5rem;
    }
    
    .math-detail-modal .math-display {
        padding: 1.5rem;
    }
}

/* Math animation on load */
.katex {
    animation: mathFadeIn 0.5s ease-out;
}

@keyframes mathFadeIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Special styling for featured equations */
.hero .katex,
.highlight-card .katex {
    font-size: 1.2em !important;
    font-weight: 600 !important;
    color: inherit !important;
}

/* Mathematical proof step styling */
.proof-steps .katex {
    background: rgba(255, 255, 255, 0.8);
    padding: 0.3rem 0.6rem;
    border-radius: var(--radius-md, 0.5rem);
    border: 1px solid rgba(15, 123, 138, 0.2);
    margin: 0 0.3rem;
}

.dark-mode .proof-steps .katex {
    background: rgba(0, 0, 0, 0.3);
    border-color: rgba(15, 123, 138, 0.3);
}
</style>
`;

// Auto-initialize KaTeX integration
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        document.head.insertAdjacentHTML('beforeend', katexStyles);
        window.katexIntegration = new KaTeXIntegration();
    });
} else {
    document.head.insertAdjacentHTML('beforeend', katexStyles);
    window.katexIntegration = new KaTeXIntegration();
}

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = KaTeXIntegration;
}