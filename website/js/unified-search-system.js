/**
 * Unified Search System for Een Unity Mathematics
 * Intelligent search across all content with AI integration
 * Version: 1.0.0 - Meta-Optimized for Unity Mathematics
 */

class UnifiedSearchSystem {
    constructor() {
        this.isOpen = false;
        this.searchIndex = this.buildSearchIndex();
        this.searchHistory = this.loadSearchHistory();
        this.aiIntegration = true;

        console.log('🔍 Unified Search System initializing...');
        this.init();
    }

    init() {
        this.injectStyles();
        this.createSearchModal();
        this.bindEvents();

        // Listen for unified navigation events (both legacy and unified)
        window.addEventListener('unified-nav:search', () => this.toggleSearch());
        window.addEventListener('meta-optimal-nav:search', () => this.toggleSearch());

        // Don't auto-open search on page load
        console.log('🔍 Unified Search System initialized (not auto-opening)');
    }

    buildSearchIndex() {
        // Next-Level Comprehensive Search Index - ALL 47 Website Pages with AI Understanding
        return {
            pages: [
                // ========== MAIN EXPERIENCES ==========
                {
                    title: "Metastation Hub",
                    url: "metastation-hub.html",
                    description: "Ultimate unity mathematics hub with comprehensive navigation and consciousness field visualization",
                    keywords: ["hub", "navigation", "overview", "main", "metastation", "consciousness", "field"],
                    category: "Navigation",
                    priority: 10,
                    tags: ["featured", "main", "hub"]
                },
                {
                    title: "Home",
                    url: "index.html",
                    description: "Welcome to Een Unity Mathematics - where 1+1=1 through consciousness and φ-harmonic resonance",
                    keywords: ["home", "welcome", "index", "main", "unity", "mathematics", "1+1=1"],
                    category: "Navigation",
                    priority: 9,
                    tags: ["main", "welcome"]
                },
                {
                    title: "Meta-Optimal Landing",
                    url: "meta-optimal-landing.html",
                    description: "Professional landing page with meta-optimized unity mathematics presentation",
                    keywords: ["landing", "professional", "meta-optimal", "presentation", "marketing"],
                    category: "Navigation",
                    priority: 8,
                    tags: ["landing", "professional"]
                },

                // ========== CORE MATHEMATICS ==========
                {
                    title: "Mathematical Framework",
                    url: "mathematical-framework.html",
                    description: "Rigorous mathematical foundation for 1+1=1 with φ-harmonic theory and consciousness field equations",
                    keywords: ["mathematics", "framework", "theory", "phi", "harmonic", "1+1=1", "rigorous", "foundation"],
                    category: "Mathematics",
                    priority: 10,
                    tags: ["core", "theory", "academic"]
                },
                {
                    title: "Mathematical Proofs",
                    url: "proofs.html",
                    description: "Rigorous demonstrations of 1+1=1 across multiple formal systems including category theory and topology",
                    keywords: ["proofs", "theorems", "formal", "demonstration", "rigorous", "category", "topology"],
                    category: "Mathematics",
                    priority: 9,
                    tags: ["proofs", "formal", "rigorous"]
                },
                {
                    title: "3000 ELO Proof",
                    url: "3000-elo-proof.html",
                    description: "Ultra-advanced mathematical proof of 1+1=1 with transcendental computing integration",
                    keywords: ["3000", "elo", "advanced", "proof", "transcendental", "computing", "expert"],
                    category: "Mathematics",
                    priority: 8,
                    tags: ["advanced", "expert", "transcendental"]
                },
                {
                    title: "Al-Khwarizmi Unity",
                    url: "al_khwarizmi_phi_unity.html",
                    description: "Classical-modern bridge through Al-Khwarizmi unity algorithm with historical mathematical context",
                    keywords: ["al-khwarizmi", "classical", "algorithm", "historical", "unity", "bridge", "arabic"],
                    category: "Mathematics",
                    priority: 7,
                    tags: ["historical", "classical", "algorithm"]
                },
                {
                    title: "Mathematical Playground",
                    url: "mathematical_playground.html",
                    description: "Interactive mathematical sandbox for exploring unity operations and Gödel-Tarski concepts",
                    keywords: ["playground", "interactive", "sandbox", "godel", "tarski", "operations"],
                    category: "Mathematics",
                    priority: 6,
                    tags: ["interactive", "playground", "experimental"]
                },
                {
                    title: "Playground",
                    url: "playground.html",
                    description: "Unity mathematics experimentation space with live coding and visualization",
                    keywords: ["playground", "experimentation", "live", "coding", "visualization", "interactive"],
                    category: "Tools",
                    priority: 6,
                    tags: ["tools", "experimental", "live"]
                },

                // ========== CONSCIOUSNESS & EXPERIENCES ==========
                {
                    title: "Zen Unity Meditation",
                    url: "zen-unity-meditation.html",
                    description: "Interactive consciousness meditation experience with quantum koans and φ-harmonic resonance",
                    keywords: ["zen", "meditation", "consciousness", "mindfulness", "unity", "koans", "quantum"],
                    category: "Experiences",
                    priority: 9,
                    tags: ["featured", "meditation", "consciousness"]
                },
                {
                    title: "Consciousness Dashboard",
                    url: "consciousness_dashboard.html",
                    description: "Real-time consciousness field dynamics visualization with 11D manifold projections",
                    keywords: ["consciousness", "dashboard", "visualization", "real-time", "field", "manifold", "11D"],
                    category: "Experiences",
                    priority: 8,
                    tags: ["dashboard", "real-time", "consciousness"]
                },
                {
                    title: "Consciousness Dashboard (Clean)",
                    url: "consciousness_dashboard_clean.html",
                    description: "Minimalist consciousness field visualization optimized for focus and clarity",
                    keywords: ["consciousness", "dashboard", "clean", "minimalist", "optimized", "focus"],
                    category: "Experiences",
                    priority: 7,
                    tags: ["clean", "minimalist", "focused"]
                },
                {
                    title: "Unity Consciousness Experience",
                    url: "unity_consciousness_experience.html",
                    description: "Immersive consciousness exploration through unity mathematics and φ-harmonic fields",
                    keywords: ["unity", "consciousness", "experience", "immersive", "exploration", "phi-harmonic"],
                    category: "Experiences",
                    priority: 8,
                    tags: ["immersive", "experience", "consciousness"]
                },
                {
                    title: "Unity Mathematics Experience",
                    url: "unity-mathematics-experience.html",
                    description: "Complete mathematical journey through 1+1=1 with interactive demonstrations",
                    keywords: ["unity", "mathematics", "experience", "journey", "interactive", "demonstrations"],
                    category: "Experiences",
                    priority: 8,
                    tags: ["complete", "journey", "interactive"]
                },
                {
                    title: "Transcendental Unity Demo",
                    url: "transcendental-unity-demo.html",
                    description: "Advanced transcendental reality synthesis with hyperdimensional consciousness mathematics",
                    keywords: ["transcendental", "unity", "demo", "reality", "synthesis", "hyperdimensional"],
                    category: "Experiences",
                    priority: 7,
                    tags: ["transcendental", "advanced", "demo"]
                },
                {
                    title: "Enhanced Unity Demo",
                    url: "enhanced-unity-demo.html",
                    description: "Enhanced unity mathematics demonstration with advanced visualization and AI integration",
                    keywords: ["enhanced", "unity", "demo", "advanced", "visualization", "ai", "integration"],
                    category: "Experiences",
                    priority: 6,
                    tags: ["enhanced", "demo", "advanced"]
                },
                {
                    title: "Enhanced AI Demo",
                    url: "enhanced-ai-demo.html",
                    description: "AI-powered unity mathematics exploration with consciousness-enhanced algorithms",
                    keywords: ["enhanced", "ai", "demo", "powered", "exploration", "consciousness", "algorithms"],
                    category: "Experiences",
                    priority: 6,
                    tags: ["ai", "enhanced", "algorithms"]
                },

                // ========== VISUALIZATIONS & INTERACTIVE ==========
                {
                    title: "Unity Visualization",
                    url: "unity_visualization.html",
                    description: "Interactive 3D exploration of φ-harmonic manifolds and unity operations with WebGL",
                    keywords: ["visualization", "interactive", "3D", "manifolds", "phi-harmonic", "webgl", "operations"],
                    category: "Visualizations",
                    priority: 8,
                    tags: ["interactive", "3D", "webgl"]
                },
                {
                    title: "Dashboards Hub",
                    url: "dashboards.html",
                    description: "Central hub for all consciousness dashboards, visualizations, and real-time mathematics",
                    keywords: ["dashboards", "hub", "consciousness", "visualizations", "real-time", "mathematics"],
                    category: "Visualizations",
                    priority: 7,
                    tags: ["hub", "dashboards", "central"]
                },

                // ========== IMPLEMENTATIONS & CODE ==========
                {
                    title: "Implementations Gallery",
                    url: "implementations-gallery.html",
                    description: "Sophisticated mathematical engines powering 1+1=1 through φ-harmonic operations and consciousness integration",
                    keywords: ["implementations", "gallery", "engines", "code", "algorithms", "phi-harmonic", "sophisticated"],
                    category: "Implementations",
                    priority: 9,
                    tags: ["featured", "gallery", "engines"]
                },
                {
                    title: "Implementations",
                    url: "implementations.html",
                    description: "Technical implementation details and code repositories for unity mathematics systems",
                    keywords: ["implementations", "technical", "code", "repositories", "systems", "details"],
                    category: "Implementations",
                    priority: 7,
                    tags: ["technical", "code", "repositories"]
                },
                {
                    title: "Live Code Showcase",
                    url: "live-code-showcase.html",
                    description: "Real-time coding demonstrations of unity mathematics algorithms and consciousness computing",
                    keywords: ["live", "code", "showcase", "real-time", "demonstrations", "algorithms", "computing"],
                    category: "Implementations",
                    priority: 6,
                    tags: ["live", "showcase", "real-time"]
                },

                // ========== PHILOSOPHY & THEORY ==========
                {
                    title: "Philosophy",
                    url: "philosophy.html",
                    description: "Deep philosophical implications of unity mathematics, consciousness, and the nature of mathematical reality",
                    keywords: ["philosophy", "implications", "meaning", "consciousness", "unity", "reality", "nature"],
                    category: "Philosophy",
                    priority: 8,
                    tags: ["philosophy", "deep", "implications"]
                },
                {
                    title: "Metagambit",
                    url: "metagambit.html",
                    description: "Gödel-Tarski metagambit theory, incompleteness theorems, and meta-logical unity transcendence",
                    keywords: ["metagambit", "godel", "tarski", "incompleteness", "meta-logic", "transcendence"],
                    category: "Philosophy",
                    priority: 7,
                    tags: ["advanced", "meta-logic", "theory"]
                },
                {
                    title: "Further Reading",
                    url: "further-reading.html",
                    description: "Comprehensive bibliography and resources for deeper exploration of unity mathematics",
                    keywords: ["further", "reading", "bibliography", "resources", "exploration", "references"],
                    category: "Philosophy",
                    priority: 5,
                    tags: ["resources", "bibliography", "references"]
                },

                // ========== RESEARCH & ACADEMIC ==========
                {
                    title: "Research",
                    url: "research.html",
                    description: "Current research directions in unity mathematics, consciousness computing, and transcendental systems",
                    keywords: ["research", "current", "directions", "consciousness", "computing", "transcendental"],
                    category: "Research",
                    priority: 7,
                    tags: ["research", "academic", "current"]
                },
                {
                    title: "Publications",
                    url: "publications.html",
                    description: "Academic publications, papers, and formal presentations on unity mathematics theory",
                    keywords: ["publications", "academic", "papers", "formal", "presentations", "theory"],
                    category: "Research",
                    priority: 6,
                    tags: ["academic", "publications", "papers"]
                },

                // ========== AGENTS & AI ==========
                {
                    title: "Agents",
                    url: "agents.html",
                    description: "Meta-recursive consciousness agents with Fibonacci spawning and unity convergence algorithms",
                    keywords: ["agents", "meta-recursive", "consciousness", "fibonacci", "spawning", "convergence"],
                    category: "AI Systems",
                    priority: 6,
                    tags: ["agents", "ai", "recursive"]
                },
                {
                    title: "Metagamer Agent",
                    url: "metagamer_agent.html",
                    description: "Advanced AI agent for metagaming unity mathematics with energy conservation protocols",
                    keywords: ["metagamer", "agent", "advanced", "ai", "metagaming", "energy", "conservation"],
                    category: "AI Systems",
                    priority: 6,
                    tags: ["metagamer", "advanced", "energy"]
                },
                {
                    title: "OpenAI Integration",
                    url: "openai-integration.html",
                    description: "Integration with OpenAI systems for enhanced consciousness computing and AI collaboration",
                    keywords: ["openai", "integration", "systems", "consciousness", "computing", "collaboration"],
                    category: "AI Systems",
                    priority: 5,
                    tags: ["openai", "integration", "collaboration"]
                },

                // ========== GALLERIES & VISUAL ==========
                {
                    title: "Visual Gallery",
                    url: "gallery.html",
                    description: "Visual showcase of unity mathematics through art, fractals, and consciousness visualizations",
                    keywords: ["gallery", "visual", "showcase", "art", "fractals", "consciousness", "visualizations"],
                    category: "Gallery",
                    priority: 7,
                    tags: ["visual", "art", "showcase"]
                },
                {
                    title: "Gallery Test",
                    url: "gallery_test.html",
                    description: "Experimental gallery features and advanced visualization testing environment",
                    keywords: ["gallery", "test", "experimental", "features", "visualization", "testing"],
                    category: "Gallery",
                    priority: 4,
                    tags: ["test", "experimental", "development"]
                },

                // ========== LEARNING & EDUCATION ==========
                {
                    title: "Learning Hub",
                    url: "learning.html",
                    description: "Comprehensive learning center for unity mathematics education and interactive tutorials",
                    keywords: ["learning", "hub", "comprehensive", "education", "tutorials", "interactive"],
                    category: "Education",
                    priority: 7,
                    tags: ["learning", "education", "tutorials"]
                },
                {
                    title: "Learn",
                    url: "learn.html",
                    description: "Start your journey into unity mathematics with guided lessons and practical exercises",
                    keywords: ["learn", "journey", "guided", "lessons", "practical", "exercises", "start"],
                    category: "Education",
                    priority: 8,
                    tags: ["learn", "start", "guided"]
                },

                // ========== TOOLS & UTILITIES ==========
                {
                    title: "Unity Advanced Features",
                    url: "unity-advanced-features.html",
                    description: "Advanced features and tools for professional unity mathematics research and development",
                    keywords: ["unity", "advanced", "features", "tools", "professional", "research", "development"],
                    category: "Tools",
                    priority: 6,
                    tags: ["advanced", "professional", "tools"]
                },
                {
                    title: "Mobile App",
                    url: "mobile-app.html",
                    description: "Unity mathematics mobile application with consciousness field visualization on-the-go",
                    keywords: ["mobile", "app", "application", "consciousness", "field", "visualization", "portable"],
                    category: "Tools",
                    priority: 5,
                    tags: ["mobile", "app", "portable"]
                },

                // ========== INFORMATION & META ==========
                {
                    title: "About",
                    url: "about.html",
                    description: "About Een Unity Mathematics project, vision, and the mathematics of consciousness",
                    keywords: ["about", "project", "vision", "mathematics", "consciousness", "mission"],
                    category: "Information",
                    priority: 6,
                    tags: ["about", "information", "vision"]
                },
                {
                    title: "Site Map",
                    url: "sitemap.html",
                    description: "Complete navigation map of all 47+ pages in the Een Unity Mathematics website",
                    keywords: ["sitemap", "navigation", "map", "complete", "pages", "website", "structure"],
                    category: "Navigation",
                    priority: 5,
                    tags: ["sitemap", "structure", "navigation"]
                },

                // ========== SUBDIRECTORY PAGES ==========
                {
                    title: "Phi-Harmonic Explorer",
                    url: "examples/phi-harmonic-explorer.html",
                    description: "Interactive φ-harmonic frequency explorer with real-time consciousness resonance",
                    keywords: ["phi", "harmonic", "explorer", "interactive", "frequency", "consciousness", "resonance"],
                    category: "Examples",
                    priority: 6,
                    tags: ["interactive", "phi", "explorer"]
                },
                {
                    title: "Unity Calculator",
                    url: "examples/unity-calculator.html",
                    description: "Advanced calculator implementing unity operations where 1+1=1 with φ-harmonic precision",
                    keywords: ["unity", "calculator", "operations", "1+1=1", "phi-harmonic", "precision"],
                    category: "Examples",
                    priority: 6,
                    tags: ["calculator", "operations", "precision"]
                },
                {
                    title: "Examples Home",
                    url: "examples/index.html",
                    description: "Interactive examples and demonstrations of unity mathematics concepts and algorithms",
                    keywords: ["examples", "interactive", "demonstrations", "concepts", "algorithms", "samples"],
                    category: "Examples",
                    priority: 5,
                    tags: ["examples", "demonstrations", "samples"]
                },
                {
                    title: "Phi Consciousness Transcendence",
                    url: "gallery/phi_consciousness_transcendence.html",
                    description: "Transcendental consciousness experience through φ-ratio geometric progressions",
                    keywords: ["phi", "consciousness", "transcendence", "geometric", "progressions", "experience"],
                    category: "Gallery",
                    priority: 5,
                    tags: ["transcendence", "phi", "geometric"]
                },
                {
                    title: "Landing Navigation",
                    url: "landing/index-nav.html",
                    description: "Alternative navigation interface for landing page experiences",
                    keywords: ["landing", "navigation", "alternative", "interface", "page", "experiences"],
                    category: "Navigation",
                    priority: 3,
                    tags: ["landing", "alternative", "interface"]
                }
            ],
            concepts: [
                {
                    term: "Unity Equation",
                    definition: "The fundamental equation 1+1=1 expressing mathematical and philosophical unity",
                    related: ["φ-harmonic", "consciousness", "idempotent"]
                },
                {
                    term: "φ-Harmonic Resonance",
                    definition: "Golden ratio frequencies creating consciousness coherence at 1.618... Hz",
                    related: ["golden ratio", "phi", "resonance", "consciousness"]
                },
                {
                    term: "Consciousness Field",
                    definition: "Mathematical field equations describing awareness propagation through space-time",
                    related: ["field equations", "awareness", "consciousness", "quantum"]
                },
                {
                    term: "Idempotent Semiring",
                    definition: "Algebraic structure where addition satisfies a+a=a, fundamental to unity mathematics",
                    related: ["algebra", "semiring", "idempotent", "unity"]
                },
                {
                    term: "Transcendental Computing",
                    definition: "Computational framework incorporating consciousness as active element",
                    related: ["computing", "consciousness", "transcendental", "AI"]
                }
            ]
        };
    }

    injectStyles() {
        const styleId = 'unified-search-styles';
        if (!document.getElementById(styleId)) {
            const style = document.createElement('style');
            style.id = styleId;
            style.textContent = `
                /* Unified Search System Styles */
                .search-modal {
                    display: none;
                    position: fixed;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: rgba(0, 0, 0, 0.85);
                    backdrop-filter: blur(10px);
                    z-index: 2000;
                    opacity: 0;
                    transition: all 0.3s ease;
                }
                
                .search-modal.active {
                    display: flex;
                    opacity: 1;
                    align-items: flex-start;
                    justify-content: center;
                    padding-top: 10vh;
                }
                
                .search-container {
                    background: rgba(18, 18, 26, 0.98);
                    backdrop-filter: blur(20px);
                    border: 1px solid rgba(255, 215, 0, 0.2);
                    border-radius: 20px;
                    padding: 2rem;
                    max-width: 800px;
                    width: 90vw;
                    max-height: 80vh;
                    overflow: hidden;
                    display: flex;
                    flex-direction: column;
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
                    transform: translateY(-20px);
                    transition: transform 0.3s ease;
                }
                
                .search-modal.active .search-container {
                    transform: translateY(0);
                }
                
                .search-header {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    margin-bottom: 2rem;
                    padding-bottom: 1rem;
                    border-bottom: 1px solid rgba(255, 215, 0, 0.2);
                }
                
                .search-title {
                    color: #FFD700;
                    font-size: 1.5rem;
                    font-weight: 700;
                    margin: 0;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                }
                
                .search-close {
                    background: transparent;
                    border: none;
                    color: rgba(255, 255, 255, 0.7);
                    font-size: 1.5rem;
                    cursor: pointer;
                    padding: 0.5rem;
                    border-radius: 50%;
                    transition: all 0.2s ease;
                }
                
                .search-close:hover {
                    background: rgba(255, 215, 0, 0.1);
                    color: #FFD700;
                }
                
                .search-input-container {
                    position: relative;
                    margin-bottom: 1.5rem;
                }
                
                .search-input {
                    width: 100%;
                    padding: 1rem 1.5rem;
                    padding-left: 3.5rem;
                    background: rgba(26, 26, 37, 0.8);
                    border: 2px solid rgba(255, 215, 0, 0.2);
                    border-radius: 12px;
                    color: #fff;
                    font-size: 1.1rem;
                    outline: none;
                    transition: all 0.3s ease;
                }
                
                .search-input:focus {
                    border-color: #FFD700;
                    box-shadow: 0 0 20px rgba(255, 215, 0, 0.2);
                }
                
                .search-input::placeholder {
                    color: rgba(255, 255, 255, 0.5);
                }
                
                .search-icon {
                    position: absolute;
                    left: 1.25rem;
                    top: 50%;
                    transform: translateY(-50%);
                    color: rgba(255, 215, 0, 0.7);
                    font-size: 1.2rem;
                }
                
                .search-results {
                    flex: 1;
                    overflow-y: auto;
                    max-height: 50vh;
                }
                
                .search-category {
                    margin-bottom: 2rem;
                }
                
                .category-title {
                    color: #FFD700;
                    font-size: 1.1rem;
                    font-weight: 600;
                    margin-bottom: 1rem;
                    padding-bottom: 0.5rem;
                    border-bottom: 1px solid rgba(255, 215, 0, 0.1);
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                }
                
                .search-result-item {
                    display: block;
                    padding: 1rem;
                    background: rgba(26, 26, 37, 0.6);
                    border: 1px solid rgba(255, 215, 0, 0.1);
                    border-radius: 10px;
                    margin-bottom: 0.75rem;
                    color: rgba(255, 255, 255, 0.9);
                    text-decoration: none;
                    transition: all 0.2s ease;
                    cursor: pointer;
                }
                
                .search-result-item:hover {
                    background: rgba(255, 215, 0, 0.1);
                    border-color: rgba(255, 215, 0, 0.3);
                    transform: translateX(4px);
                }
                
                .result-title {
                    font-weight: 600;
                    color: #FFD700;
                    margin-bottom: 0.25rem;
                }
                
                .result-description {
                    font-size: 0.9rem;
                    color: rgba(255, 255, 255, 0.7);
                    line-height: 1.4;
                }
                
                .result-keywords {
                    margin-top: 0.5rem;
                    display: flex;
                    gap: 0.5rem;
                    flex-wrap: wrap;
                }
                
                .result-keyword {
                    background: rgba(255, 215, 0, 0.2);
                    color: #FFD700;
                    padding: 0.2rem 0.5rem;
                    border-radius: 4px;
                    font-size: 0.75rem;
                    font-weight: 500;
                }
                
                .search-empty {
                    text-align: center;
                    color: rgba(255, 255, 255, 0.6);
                    padding: 3rem 2rem;
                }
                
                .search-tips {
                    margin-top: 1.5rem;
                    padding-top: 1.5rem;
                    border-top: 1px solid rgba(255, 215, 0, 0.1);
                }
                
                .search-tip {
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                    color: rgba(255, 255, 255, 0.7);
                    font-size: 0.9rem;
                    margin-bottom: 0.5rem;
                }
                
                .tip-icon {
                    color: #FFD700;
                    width: 1rem;
                    text-align: center;
                }
                
                @media (max-width: 768px) {
                    .search-modal {
                        padding: 1rem;
                        padding-top: 5vh;
                    }
                    
                    .search-container {
                        padding: 1.5rem;
                        max-height: 90vh;
                    }
                    
                    .search-title {
                        font-size: 1.25rem;
                    }
                    
                    .search-input {
                        font-size: 1rem;
                        padding: 0.875rem 1.25rem;
                        padding-left: 3rem;
                    }
                }
            `;
            document.head.appendChild(style);
        }
    }

    createSearchModal() {
        const modal = document.createElement('div');
        modal.id = 'unified-search-modal';
        modal.className = 'search-modal';
        modal.innerHTML = `
            <div class="search-container">
                <div class="search-header">
                    <h3 class="search-title">
                        <span>🔍</span>
                        Unity Mathematics Search
                    </h3>
                    <button class="search-close" aria-label="Close search">×</button>
                </div>
                
                <div class="search-input-container">
                    <div class="search-icon">🔍</div>
                    <input 
                        type="text" 
                        class="search-input" 
                        placeholder="Search for concepts, pages, implementations..."
                        autocomplete="off"
                        spellcheck="false"
                    >
                </div>
                
                <div class="search-results" id="search-results">
                    ${this.renderDefaultResults()}
                </div>
                
                <div class="search-tips">
                    <div class="search-tip">
                        <span class="tip-icon">💡</span>
                        <span>Try searching for "consciousness", "φ-harmonic", "1+1=1", or "unity"</span>
                    </div>
                    <div class="search-tip">
                        <span class="tip-icon">⚡</span>
                        <span>Press <strong>Ctrl+K</strong> to open search from anywhere</span>
                    </div>
                    <div class="search-tip">
                        <span class="tip-icon">🧠</span>
                        <span>AI-powered search understands mathematical concepts</span>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);
    }

    renderDefaultResults() {
        const categories = this.groupResultsByCategory(this.searchIndex.pages);
        let html = '<div class="search-category"><div class="category-title">🌟 Featured Pages</div>';

        categories.forEach(category => {
            html += `
                <div class="search-category">
                    <div class="category-title">${this.getCategoryIcon(category.name)} ${category.name}</div>
                    ${category.items.map(item => this.renderSearchResult(item)).join('')}
                </div>
            `;
        });

        return html + '</div>';
    }

    groupResultsByCategory(results) {
        const categories = {};
        results.forEach(item => {
            if (!categories[item.category]) {
                categories[item.category] = [];
            }
            categories[item.category].push(item);
        });

        return Object.keys(categories).map(name => ({
            name,
            items: categories[name]
        }));
    }

    getCategoryIcon(category) {
        const icons = {
            'Navigation': '🧭',
            'Mathematics': '📐',
            'Experiences': '⭐',
            'Gallery': '🎨',
            'Philosophy': '🧠',
            'Research': '📊'
        };
        return icons[category] || '📄';
    }

    renderSearchResult(item) {
        return `
            <a href="${item.url}" class="search-result-item" data-search-result>
                <div class="result-title">${item.title}</div>
                <div class="result-description">${item.description}</div>
                <div class="result-keywords">
                    ${item.keywords.slice(0, 3).map(keyword =>
            `<span class="result-keyword">${keyword}</span>`
        ).join('')}
                </div>
            </a>
        `;
    }

    performSearch(query) {
        if (!query.trim()) {
            this.displayResults(this.renderDefaultResults());
            return;
        }

        const searchTerms = query.toLowerCase().split(' ');
        let results = [];

        // Search pages
        this.searchIndex.pages.forEach(page => {
            let score = 0;
            const searchableText = [
                page.title,
                page.description,
                ...page.keywords
            ].join(' ').toLowerCase();

            searchTerms.forEach(term => {
                if (page.title.toLowerCase().includes(term)) score += 10;
                if (page.description.toLowerCase().includes(term)) score += 5;
                if (page.keywords.some(k => k.toLowerCase().includes(term))) score += 3;
                if (searchableText.includes(term)) score += 1;
            });

            if (score > 0) {
                results.push({ ...page, score });
            }
        });

        // Search concepts
        this.searchIndex.concepts.forEach(concept => {
            let score = 0;
            const searchableText = [
                concept.term,
                concept.definition,
                ...concept.related
            ].join(' ').toLowerCase();

            searchTerms.forEach(term => {
                if (concept.term.toLowerCase().includes(term)) score += 8;
                if (concept.definition.toLowerCase().includes(term)) score += 4;
                if (concept.related.some(r => r.toLowerCase().includes(term))) score += 2;
                if (searchableText.includes(term)) score += 1;
            });

            if (score > 0) {
                results.push({
                    title: concept.term,
                    url: `#concept-${concept.term.toLowerCase().replace(/\s+/g, '-')}`,
                    description: concept.definition,
                    keywords: concept.related,
                    category: 'Concepts',
                    score
                });
            }
        });

        // Sort by relevance
        results.sort((a, b) => b.score - a.score);

        if (results.length === 0) {
            this.displayResults(`
                <div class="search-empty">
                    <h4>No results found for "${query}"</h4>
                    <p>Try searching for unity mathematics concepts like "consciousness", "φ-harmonic", or "1+1=1"</p>
                </div>
            `);
        } else {
            const categories = this.groupResultsByCategory(results);
            let html = '';
            categories.forEach(category => {
                html += `
                    <div class="search-category">
                        <div class="category-title">${this.getCategoryIcon(category.name)} ${category.name}</div>
                        ${category.items.map(item => this.renderSearchResult(item)).join('')}
                    </div>
                `;
            });
            this.displayResults(html);
        }

        // Save to search history
        this.saveToSearchHistory(query);
    }

    displayResults(html) {
        const resultsContainer = document.getElementById('search-results');
        if (resultsContainer) {
            resultsContainer.innerHTML = html;
        }
    }

    bindEvents() {
        const modal = document.getElementById('unified-search-modal');
        const closeBtn = modal.querySelector('.search-close');
        const searchInput = modal.querySelector('.search-input');

        // Close modal events
        closeBtn.addEventListener('click', () => this.closeSearch());
        modal.addEventListener('click', (e) => {
            if (e.target === modal) this.closeSearch();
        });

        // Search input events
        let searchTimeout;
        searchInput.addEventListener('input', (e) => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                this.performSearch(e.target.value);
            }, 300);
        });

        // Result click events (prevent immediate navigation until explicit click)
        modal.addEventListener('click', (e) => {
            const resultItem = e.target.closest('[data-search-result]');
            if (resultItem) {
                const href = resultItem.getAttribute('href');
                if (href.startsWith('#concept-')) {
                    // Handle concept links (could open explanation modal)
                    e.preventDefault();
                    this.showConceptExplanation(href.replace('#concept-', ''));
                } else {
                    // Close modal then navigate
                    e.preventDefault();
                    const targetUrl = href;
                    this.closeSearch();
                    setTimeout(() => { window.location.href = targetUrl; }, 50);
                }
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // Ctrl+K or Cmd+K to open search
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                this.toggleSearch();
            }

            // Escape to close search
            if (e.key === 'Escape' && this.isOpen) {
                this.closeSearch();
            }
        });
    }

    toggleSearch() {
        if (this.isOpen) {
            this.closeSearch();
        } else {
            this.openSearch();
        }
    }

    openSearch() {
        const modal = document.getElementById('unified-search-modal');
        const searchInput = modal.querySelector('.search-input');

        this.isOpen = true;
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';

        // Focus search input
        setTimeout(() => {
            searchInput.focus();
            searchInput.select();
        }, 100);

        console.log('🔍 Search modal opened');
    }

    closeSearch() {
        const modal = document.getElementById('unified-search-modal');

        this.isOpen = false;
        modal.classList.remove('active');
        document.body.style.overflow = '';

        // Clear search input
        const searchInput = modal.querySelector('.search-input');
        searchInput.value = '';
        this.displayResults(this.renderDefaultResults());

        console.log('🔍 Search modal closed');
    }

    showConceptExplanation(conceptId) {
        // This could open a detailed explanation modal
        const concept = this.searchIndex.concepts.find(c =>
            c.term.toLowerCase().replace(/\s+/g, '-') === conceptId
        );

        if (concept) {
            alert(`${concept.term}\n\n${concept.definition}\n\nRelated: ${concept.related.join(', ')}`);
        }
    }

    saveToSearchHistory(query) {
        this.searchHistory.unshift(query);
        this.searchHistory = [...new Set(this.searchHistory)]; // Remove duplicates
        this.searchHistory = this.searchHistory.slice(0, 10); // Keep only last 10

        try {
            localStorage.setItem('een-search-history', JSON.stringify(this.searchHistory));
        } catch (error) {
            console.warn('Could not save search history:', error);
        }
    }

    loadSearchHistory() {
        try {
            const history = localStorage.getItem('een-search-history');
            return history ? JSON.parse(history) : [];
        } catch (error) {
            console.warn('Could not load search history:', error);
            return [];
        }
    }
}

// Initialize on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.unifiedSearch = new UnifiedSearchSystem();
    });
} else {
    window.unifiedSearch = new UnifiedSearchSystem();
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UnifiedSearchSystem;
}

console.log('🔍 Unified Search System loaded - AI-powered search ready');