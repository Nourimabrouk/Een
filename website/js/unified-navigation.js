/**
 * Unified Navigation System - Meta-Optimal Unity Mathematics
 * Single, consolidated navigation system for all pages
 * Version: 4.1.0 - GitHub Pages Compatible
 */

class UnifiedNavigationSystem {
    constructor() {
        this.isInitialized = false;
        this.currentPage = this.getCurrentPageName();
        this.searchData = this.initializeSearchData();
        this.baseFooterLinks = new Set([
            // Mathematics
            'mathematical-framework.html', 'proofs.html', 'unity_proof.html', '3000-elo-proof.html', 'al_khwarizmi_phi_unity.html',
            // Experiences & AI
            'zen-unity-meditation.html', 'consciousness_dashboard.html', 'unity_consciousness_experience.html', 'ai-unified-hub.html', 'metagamer_agent.html',
            // Visualizations & Tools
            'gallery.html', 'enhanced-3d-consciousness-field.html', 'playground.html', 'live-code-showcase.html', 'examples/unity-calculator.html', 'examples/phi-harmonic-explorer.html',
            // About
            'about.html', 'research.html', 'publications.html', 'academic-portal.html', 'learning.html', 'further-reading.html', 'unity-meta-atlas.html', 'mobile-app.html', 'sitemap.html'
        ]);
        this.init();
    }

    getCurrentPageName() {
        const path = window.location.pathname;
        const page = path.split('/').pop() || 'index.html';
        return page === '' ? 'index.html' : page;
    }

    // Helper function to get the correct link path
    getLink(href) {
        // If we're in a subdirectory (like examples/ or gallery/), adjust the path
        const currentPath = window.location.pathname;
        const depth = (currentPath.match(/\//g) || []).length - (currentPath.endsWith('/') ? 1 : 0);
        
        // Check if we're in a subdirectory
        if (currentPath.includes('/examples/') || currentPath.includes('/gallery/')) {
            if (href.startsWith('examples/') || href.startsWith('gallery/')) {
                return '../' + href;
            } else {
                return '../' + href;
            }
        }
        
        // For root level pages, use relative paths
        return './' + href;
    }

    init() {
        if (this.isInitialized) return;

        this.createNavigationStructure();
        this.attachEventListeners();
        this.setupKeyboardShortcuts();
        this.setupSearch();
        this.setupMobileHandling();
        this.markCurrentPage();

        this.isInitialized = true;
        console.log('Unified Navigation System initialized (GitHub Pages Compatible)');
    }

    createNavigationStructure() {
        // Create main navigation header
        const header = document.createElement('header');
        header.className = 'unified-header';
        header.innerHTML = this.generateHeaderHTML();

        // Create sidebar
        const sidebar = document.createElement('aside');
        sidebar.className = 'sidebar';
        sidebar.innerHTML = this.generateSidebarHTML();

        // Create sidebar toggle
        const sidebarToggle = document.createElement('button');
        sidebarToggle.className = 'sidebar-toggle';
        sidebarToggle.innerHTML = '<i class="fas fa-chevron-right"></i>';
        sidebarToggle.setAttribute('aria-label', 'Toggle sidebar navigation');

        // Create mobile overlay
        const mobileOverlay = document.createElement('div');
        mobileOverlay.className = 'mobile-nav-overlay';
        mobileOverlay.innerHTML = this.generateMobileNavHTML();

        // Insert into DOM
        document.body.insertBefore(header, document.body.firstChild);
        document.body.appendChild(sidebar);
        document.body.appendChild(sidebarToggle);
        document.body.appendChild(mobileOverlay);

        // Ensure unified, styled footer exists and is meta-optimized
        this.ensureUnifiedFooter();
    }

    generateHeaderHTML() {
        return `
            <div class="nav-container">
                <a href="${this.getLink('metastation-hub.html')}" class="nav-logo">
                    <div class="phi-symbol">φ</div>
                    <span class="logo-text">Unity Mathematics</span>
                </a>

                <nav class="nav-main">
                    <div class="nav-item">
                        <button class="nav-dropdown-toggle" type="button">
                            Experiences <i class="fas fa-chevron-down"></i>
                        </button>
                        <div class="nav-dropdown">
                            <a href="${this.getLink('zen-unity-meditation.html')}" class="nav-dropdown-link">
                                <i class="fas fa-om"></i> Zen Unity Meditation
                            </a>
                            <a href="${this.getLink('consciousness_dashboard.html')}" class="nav-dropdown-link">
                                <i class="fas fa-brain"></i> Consciousness Field
                            </a>
                            <a href="${this.getLink('unity_consciousness_experience.html')}" class="nav-dropdown-link">
                                <i class="fas fa-infinity"></i> Unity Consciousness
                            </a>
                            <a href="${this.getLink('transcendental-unity-demo.html')}" class="nav-dropdown-link">
                                <i class="fas fa-atom"></i> Transcendental Unity
                            </a>
                            <a href="${this.getLink('anthill.html')}" class="nav-dropdown-link">
                                <i class="fas fa-bug"></i> Quantum Ant Colony
                            </a>
                        </div>
                    </div>

                    <div class="nav-item">
                        <button class="nav-dropdown-toggle" type="button">
                            Mathematics <i class="fas fa-chevron-down"></i>
                        </button>
                        <div class="nav-dropdown">
                            <a href="${this.getLink('mathematical-framework.html')}" class="nav-dropdown-link">
                                <i class="fas fa-square-root-alt"></i> Mathematical Framework
                            </a>
                            <a href="${this.getLink('proofs.html')}" class="nav-dropdown-link">
                                <i class="fas fa-check-circle"></i> Proofs & Theorems
                            </a>
                             <a href="${this.getLink('unity_proof.html')}" class="nav-dropdown-link">
                                 <i class="fas fa-equals"></i> Unity Proof (1+1=1)
                             </a>
                            <a href="${this.getLink('3000-elo-proof.html')}" class="nav-dropdown-link">
                                <i class="fas fa-trophy"></i> 3000 ELO Proof
                            </a>
                            <a href="${this.getLink('implementations.html')}" class="nav-dropdown-link">
                                <i class="fas fa-code"></i> Implementations
                            </a>
                            <a href="${this.getLink('al_khwarizmi_phi_unity.html')}" class="nav-dropdown-link">
                                <i class="fas fa-scroll"></i> Al-Khwarizmi Unity
                            </a>
                             <a href="${this.getLink('unity-axioms.html')}" class="nav-dropdown-link">
                                 <i class="fas fa-balance-scale"></i> Unity Axioms
                             </a>
                            <a href="${this.getLink('metagambit.html')}" class="nav-dropdown-link">
                                <i class="fas fa-chess"></i> Metagambit Framework
                            </a>
                            <a href="${this.getLink('advanced-systems.html')}" class="nav-dropdown-link">
                                <i class="fas fa-rocket"></i> Advanced Systems
                            </a>
                        </div>
                    </div>

                    <div class="nav-item">
                        <button class="nav-dropdown-toggle" type="button">
                            AI Systems <i class="fas fa-chevron-down"></i>
                        </button>
                        <div class="nav-dropdown">
                            <a href="${this.getLink('ai-unified-hub.html')}" class="nav-dropdown-link">
                                <i class="fas fa-brain"></i> AI Unity Hub
                            </a>
                            <a href="${this.getLink('ai-agents-ecosystem.html')}" class="nav-dropdown-link">
                                <i class="fas fa-network-wired"></i> AI Agents Ecosystem
                            </a>
                            <a href="${this.getLink('agents.html')}" class="nav-dropdown-link">
                                <i class="fas fa-robot"></i> Unity Agents
                            </a>
                            <a href="${this.getLink('metagamer_agent.html')}" class="nav-dropdown-link">
                                <i class="fas fa-gamepad"></i> Metagamer Agent
                            </a>
                            <a href="${this.getLink('openai-integration.html')}" class="nav-dropdown-link">
                                <i class="fas fa-plug"></i> OpenAI Integration
                            </a>
                            <a href="${this.getLink('enhanced-ai-demo.html')}" class="nav-dropdown-link">
                                <i class="fas fa-sparkles"></i> Enhanced AI Demo
                            </a>
                        </div>

                    </div>

                    <div class="nav-item">
                        <button class="nav-dropdown-toggle" type="button">
                            Dashboards <i class="fas fa-chevron-down"></i>
                        </button>
                        <div class="nav-dropdown">
                            <a href="https://een-unity-mathematics.streamlit.app" class="nav-dropdown-link" target="_blank">
                                <i class="fas fa-cloud"></i> Master Unity Dashboard (Cloud)
                            </a>
                            <a href="${this.getLink('playground.html')}" class="nav-dropdown-link">
                                <i class="fas fa-gamepad"></i> Interactive Playground
                            </a>
                            <a href="${this.getLink('live-code-showcase.html')}" class="nav-dropdown-link">
                                <i class="fas fa-terminal"></i> Live Code Showcase
                            </a>
                            <a href="${this.getLink('unity-advanced-features.html')}" class="nav-dropdown-link">
                                <i class="fas fa-cogs"></i> Advanced Features
                            </a>
                        </div>
                    </div>

                    <div class="nav-item">
                        <button class="nav-dropdown-toggle" type="button">
                            Gallery <i class="fas fa-chevron-down"></i>
                        </button>
                        <div class="nav-dropdown">
                            <a href="${this.getLink('gallery.html')}" class="nav-dropdown-link">
                                <i class="fas fa-images"></i> Main Gallery
                            </a>
                            <a href="${this.getLink('dalle-gallery.html')}" class="nav-dropdown-link">
                                <i class="fas fa-palette"></i> DALL-E Consciousness Gallery
                            </a>
                            <a href="${this.getLink('enhanced-3d-consciousness-field.html')}" class="nav-dropdown-link">
                                <i class="fas fa-cube"></i> 3D Consciousness Field
                            </a>
                            <a href="${this.getLink('enhanced-unity-visualization-system.html')}" class="nav-dropdown-link">
                                <i class="fas fa-chart-line"></i> Visualization System
                            </a>
                             <a href="${this.getLink('unity_visualization.html')}" class="nav-dropdown-link">
                                 <i class="fas fa-vr-cardboard"></i> Unity Visualization
                             </a>
                             <a href="${this.getLink('gallery/phi_consciousness_transcendence.html')}" class="nav-dropdown-link">
                                 <i class="fas fa-infinity"></i> Phi Consciousness Transcendence
                             </a>
                        </div>
                    </div>

                    <div class="nav-item">
                        <button class="nav-dropdown-toggle" type="button">
                            More <i class="fas fa-chevron-down"></i>
                        </button>
                        <div class="nav-dropdown">
<<<<<<< HEAD
                            <div class="nav-dropdown-link" style="pointer-events:none; opacity:0.85; font-weight:700; color: var(--unity-gold);">
                                <i class="fab fa-github"></i> Open Source
                            </div>
                            <a href="https://github.com/Nourimabrouk/Een" class="nav-dropdown-link" target="_blank" rel="noopener">
                                <i class="fab fa-github"></i> GitHub Repository
                            </a>
                            <a href="https://github.com/Nourimabrouk/Een/blob/main/README.md" class="nav-dropdown-link" target="_blank" rel="noopener">
                                <i class="fas fa-book-open"></i> README & Getting Started
                            </a>
                            <a href="https://github.com/Nourimabrouk/Een/blob/main/CONTRIBUTING.md" class="nav-dropdown-link" target="_blank" rel="noopener">
                                <i class="fas fa-hands-helping"></i> Contributing Guide
                            </a>
                            <div style="border-top: 1px solid rgba(255,255,255,0.1); margin: 0.5rem 0;"></div>
                            <a href="academic-portal.html" class="nav-dropdown-link">
=======
                            <a href="${this.getLink('academic-portal.html')}" class="nav-dropdown-link">
>>>>>>> develop
                                <i class="fas fa-university"></i> Academic Portal
                            </a>
                            <a href="${this.getLink('philosophy.html')}" class="nav-dropdown-link">
                                <i class="fas fa-yin-yang"></i> Unity Philosophy
                            </a>
                            <a href="${this.getLink('research.html')}" class="nav-dropdown-link">
                                <i class="fas fa-microscope"></i> Research Papers
                            </a>
                            <a href="${this.getLink('publications.html')}" class="nav-dropdown-link">
                                <i class="fas fa-book"></i> Publications
                            </a>
                            <a href="${this.getLink('learning.html')}" class="nav-dropdown-link">
                                <i class="fas fa-graduation-cap"></i> Learning Path
                            </a>
                            <a href="${this.getLink('about.html')}" class="nav-dropdown-link">
                                <i class="fas fa-info-circle"></i> About Unity
                            </a>
                            <a href="${this.getLink('sitemap.html')}" class="nav-dropdown-link">
                                <i class="fas fa-sitemap"></i> Site Map
                            </a>
                             <a href="${this.getLink('unity-meta-atlas.html')}" class="nav-dropdown-link">
                                 <i class="fas fa-map"></i> Unity Meta Atlas
                             </a>
                             <a href="${this.getLink('mobile-app.html')}" class="nav-dropdown-link">
                                 <i class="fas fa-mobile-alt"></i> Mobile App
                             </a>
                             <div class="nav-dropdown-link" style="pointer-events:none; opacity:0.85; font-weight:700; margin-top:0.25rem;">
                                 <i class="fas fa-flask"></i> Examples
                             </div>
                             <a href="${this.getLink('examples/unity-calculator.html')}" class="nav-dropdown-link">
                                 <i class="fas fa-calculator"></i> Unity Calculator
                             </a>
                             <a href="${this.getLink('examples/phi-harmonic-explorer.html')}" class="nav-dropdown-link">
                                 <i class="fas fa-wave-square"></i> φ‑Harmonic Explorer
                             </a>
                             <a href="${this.getLink('examples/index.html')}" class="nav-dropdown-link">
                                 <i class="fas fa-flask"></i> Examples Home
                             </a>
                        </div>
                    </div>
                </nav>

                <div class="nav-search">
                    <input type="text" class="search-input" id="unified-search-input" 
                           placeholder="Search" 
                           aria-label="Search">
                    <i class="fas fa-search search-icon"></i>
                    <div class="search-results" id="unified-search-results"></div>
                </div>

                <button class="mobile-menu-toggle" type="button" aria-label="Toggle mobile menu">
                    <i class="fas fa-bars"></i>
                </button>
            </div>
        `;
    }

    generateSidebarHTML() {
        return `
            <div class="sidebar-content">
                <div class="sidebar-section">
                    <div class="sidebar-title">Featured Experiences</div>
                    <a href="${this.getLink('metastation-hub.html')}" class="sidebar-link featured">
                        <i class="fas fa-star"></i> Metastation Hub
                    </a>
                    <a href="${this.getLink('zen-unity-meditation.html')}" class="sidebar-link">
                        <i class="fas fa-om"></i> Zen Unity Meditation
                    </a>
                    <a href="${this.getLink('consciousness_dashboard.html')}" class="sidebar-link">
                        <i class="fas fa-brain"></i> Consciousness Field
                    </a>
                    <a href="${this.getLink('transcendental-unity-demo.html')}" class="sidebar-link">
                        <i class="fas fa-atom"></i> Transcendental Unity
                    </a>
                    <a href="${this.getLink('anthill.html')}" class="sidebar-link">
                        <i class="fas fa-bug"></i> Quantum Ant Colony
                    </a>
                    <a href="${this.getLink('unity-mathematics-experience.html')}" class="sidebar-link">
                        <i class="fas fa-equals"></i> Unity Mathematics Experience
                    </a>
                </div>

                <div class="sidebar-section">
                    <div class="sidebar-title">Mathematics & Proofs</div>
                    <a href="${this.getLink('mathematical-framework.html')}" class="sidebar-link">
                        <i class="fas fa-square-root-alt"></i> Mathematical Framework
                    </a>
                    <a href="${this.getLink('proofs.html')}" class="sidebar-link">
                        <i class="fas fa-check-circle"></i> Proofs & Theorems
                    </a>
                         <a href="${this.getLink('unity_proof.html')}" class="sidebar-link">
                             <i class="fas fa-equals"></i> Unity Proof (1+1=1)
                         </a>
                    <a href="${this.getLink('3000-elo-proof.html')}" class="sidebar-link">
                        <i class="fas fa-trophy"></i> 3000 ELO Proof
                    </a>
                    <a href="${this.getLink('implementations.html')}" class="sidebar-link">
                        <i class="fas fa-code"></i> Implementations
                    </a>
                    <a href="${this.getLink('al_khwarizmi_phi_unity.html')}" class="sidebar-link">
                        <i class="fas fa-scroll"></i> Al-Khwarizmi Unity
                    </a>
                        <a href="${this.getLink('unity-axioms.html')}" class="sidebar-link">
                            <i class="fas fa-balance-scale"></i> Unity Axioms
                        </a>
                    <a href="${this.getLink('metagambit.html')}" class="sidebar-link">
                        <i class="fas fa-chess"></i> Metagambit Framework
                    </a>
                </div>

                <div class="sidebar-section">
                    <div class="sidebar-title">AI & Agents</div>
                    <a href="${this.getLink('ai-unified-hub.html')}" class="sidebar-link featured">
                        <i class="fas fa-brain"></i> AI Unity Hub
                    </a>
                    <a href="${this.getLink('ai-agents-ecosystem.html')}" class="sidebar-link">
                        <i class="fas fa-network-wired"></i> AI Agents Ecosystem
                    </a>
                    <a href="${this.getLink('agents.html')}" class="sidebar-link">
                        <i class="fas fa-robot"></i> Unity Agents
                    </a>
                    <a href="${this.getLink('metagamer_agent.html')}" class="sidebar-link">
                        <i class="fas fa-gamepad"></i> Metagamer Agent
                    </a>
                    <a href="${this.getLink('openai-integration.html')}" class="sidebar-link">
                        <i class="fas fa-plug"></i> OpenAI Integration
                    </a>
                </div>

                <div class="sidebar-section">
                    <div class="sidebar-title">Dashboards & Tools</div>
                    <a href="https://een-unity-mathematics.streamlit.app" class="sidebar-link" target="_blank">
                        <i class="fas fa-cloud"></i> Master Unity Dashboard (Cloud)
                    </a>
                    <a href="${this.getLink('playground.html')}" class="sidebar-link">
                        <i class="fas fa-gamepad"></i> Interactive Playground
                    </a>
                    <a href="${this.getLink('live-code-showcase.html')}" class="sidebar-link">
                        <i class="fas fa-terminal"></i> Live Code Showcase
                    </a>
                    <a href="${this.getLink('examples/unity-calculator.html')}" class="sidebar-link">
                        <i class="fas fa-calculator"></i> Unity Calculator
                    </a>
                    <a href="${this.getLink('examples/phi-harmonic-explorer.html')}" class="sidebar-link">
                        <i class="fas fa-wave-square"></i> φ-Harmonic Explorer
                    </a>
                </div>

                <div class="sidebar-section">
                    <div class="sidebar-title">Gallery & Visualizations</div>
                    <a href="${this.getLink('gallery.html')}" class="sidebar-link">
                        <i class="fas fa-images"></i> Main Gallery
                    </a>
                    <a href="${this.getLink('dalle-gallery.html')}" class="sidebar-link">
                        <i class="fas fa-palette"></i> DALL-E Gallery
                    </a>
                    <a href="${this.getLink('enhanced-3d-consciousness-field.html')}" class="sidebar-link">
                        <i class="fas fa-cube"></i> 3D Consciousness Field
                    </a>
                    <a href="${this.getLink('enhanced-unity-visualization-system.html')}" class="sidebar-link">
                        <i class="fas fa-chart-line"></i> Visualization System
                    </a>
                    <a href="${this.getLink('unity_visualization.html')}" class="sidebar-link">
                        <i class="fas fa-vr-cardboard"></i> Unity Visualization
                    </a>
                    <a href="${this.getLink('gallery/phi_consciousness_transcendence.html')}" class="sidebar-link">
                        <i class="fas fa-infinity"></i> Phi Consciousness
                    </a>
                </div>

                <div class="sidebar-section">
<<<<<<< HEAD
                    <div class="sidebar-title">Open Source</div>
                    <a href="https://github.com/Nourimabrouk/Een" class="sidebar-link" target="_blank" rel="noopener">
                        <i class="fab fa-github"></i> GitHub Repository
                    </a>
                    <a href="https://github.com/Nourimabrouk/Een/blob/main/README.md" class="sidebar-link" target="_blank" rel="noopener">
                        <i class="fas fa-book-open"></i> README & Docs
                    </a>
                    <a href="https://github.com/Nourimabrouk/Een/blob/main/CONTRIBUTING.md" class="sidebar-link" target="_blank" rel="noopener">
                        <i class="fas fa-hands-helping"></i> Contributing
                    </a>
                </div>

                <div class="sidebar-section">
                    <div class="sidebar-title">Philosophy & Research</div>
                    <a href="academic-portal.html" class="sidebar-link">
=======
                    <div class="sidebar-title">Academic & Research</div>
                    <a href="${this.getLink('academic-portal.html')}" class="sidebar-link featured">
>>>>>>> develop
                        <i class="fas fa-university"></i> Academic Portal
                    </a>
                    <a href="${this.getLink('philosophy.html')}" class="sidebar-link">
                        <i class="fas fa-yin-yang"></i> Unity Philosophy
                    </a>
                    <a href="${this.getLink('research.html')}" class="sidebar-link">
                        <i class="fas fa-microscope"></i> Research Papers
                    </a>
                    <a href="${this.getLink('publications.html')}" class="sidebar-link">
                        <i class="fas fa-book"></i> Publications
                    </a>
                    <a href="${this.getLink('learning.html')}" class="sidebar-link">
                        <i class="fas fa-graduation-cap"></i> Learning Path
                    </a>
                    <a href="${this.getLink('further-reading.html')}" class="sidebar-link">
                        <i class="fas fa-book-reader"></i> Further Reading
                    </a>
                </div>

                <div class="sidebar-section">
                    <div class="sidebar-title">Meta Resources</div>
                    <a href="${this.getLink('unity-meta-atlas.html')}" class="sidebar-link">
                        <i class="fas fa-map"></i> Unity Meta Atlas
                    </a>
                    <a href="${this.getLink('about.html')}" class="sidebar-link">
                        <i class="fas fa-info-circle"></i> About Unity
                    </a>
                    <a href="${this.getLink('mobile-app.html')}" class="sidebar-link">
                        <i class="fas fa-mobile-alt"></i> Mobile App
                    </a>
                    <a href="${this.getLink('sitemap.html')}" class="sidebar-link">
                        <i class="fas fa-sitemap"></i> Site Map
                    </a>
                </div>
            </div>
        `;
    }

    generateMobileNavHTML() {
        return `
            <div class="mobile-nav-header">
                <button class="mobile-nav-close" type="button" aria-label="Close mobile menu">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="mobile-nav-content">
                <div class="mobile-nav-section">
                    <div class="mobile-nav-title">Featured</div>
                    <a href="${this.getLink('metastation-hub.html')}" class="mobile-nav-link featured">
                        <i class="fas fa-star"></i> Metastation Hub
                    </a>
                    <a href="${this.getLink('zen-unity-meditation.html')}" class="mobile-nav-link">
                        <i class="fas fa-om"></i> Zen Unity Meditation
                    </a>
                    <a href="${this.getLink('consciousness_dashboard.html')}" class="mobile-nav-link">
                        <i class="fas fa-brain"></i> Consciousness Field
                    </a>
                </div>

                <div class="mobile-nav-section">
                    <div class="mobile-nav-title">Mathematics</div>
                    <a href="${this.getLink('mathematical-framework.html')}" class="mobile-nav-link">
                        <i class="fas fa-square-root-alt"></i> Mathematical Framework
                    </a>
                    <a href="${this.getLink('proofs.html')}" class="mobile-nav-link">
                        <i class="fas fa-check-circle"></i> Proofs & Theorems
                    </a>
                    <a href="${this.getLink('implementations-gallery.html')}" class="mobile-nav-link">
                        <i class="fas fa-code"></i> Implementations
                    </a>
                </div>

                <div class="mobile-nav-section">
                    <div class="mobile-nav-title">AI Systems</div>
                    <a href="${this.getLink('ai-unified-hub.html')}" class="mobile-nav-link">
                        <i class="fas fa-brain"></i> AI Unity Hub
                    </a>
                    <a href="${this.getLink('ai-agents-ecosystem.html')}" class="mobile-nav-link">
                        <i class="fas fa-network-wired"></i> AI Agents Ecosystem
                    </a>
                </div>

<<<<<<< HEAD
                <div class="mobile-nav-category">
                    <div class="mobile-nav-category-title">
                        <i class="fas fa-flask"></i> Examples
                    </div>
                    <div class="mobile-nav-links">
                        <a href="examples/unity-calculator.html" class="mobile-nav-link">
                            <i class="fas fa-calculator"></i> Unity Calculator
                        </a>
                        <a href="examples/phi-harmonic-explorer.html" class="mobile-nav-link">
                            <i class="fas fa-wave-square"></i> φ‑Harmonic Explorer
                        </a>
                        <a href="examples/index.html" class="mobile-nav-link">
                            <i class="fas fa-flask"></i> Examples Home
                        </a>
                    </div>
                </div>

                <div class="mobile-nav-category">
                    <div class="mobile-nav-category-title">
                        <i class="fas fa-th-large"></i> Dashboards & Tools
                    </div>
                    <div class="mobile-nav-links">
                        <a href="https://een-unity-mathematics.streamlit.app" class="mobile-nav-link" target="_blank">
                            <i class="fas fa-cloud"></i> Master Unity Dashboard (Cloud)
                        </a>
                        <a href="dashboards.html" class="mobile-nav-link">
                            <i class="fas fa-th-large"></i> All Dashboards
                        </a>
                        <a href="playground.html" class="mobile-nav-link">
                            <i class="fas fa-gamepad"></i> Interactive Playground
                        </a>
                        <a href="mathematical_playground.html" class="mobile-nav-link">
                            <i class="fas fa-calculator"></i> Math Playground
                        </a>
                        <a href="live-code-showcase.html" class="mobile-nav-link">
                            <i class="fas fa-terminal"></i> Live Code Showcase
                        </a>
                        <a href="unity-advanced-features.html" class="mobile-nav-link">
                            <i class="fas fa-cogs"></i> Advanced Features
                        </a>
                    </div>
                </div>

                <div class="mobile-nav-category">
                    <div class="mobile-nav-category-title">
                        <i class="fas fa-images"></i> Gallery & Visualizations
                    </div>
                    <div class="mobile-nav-links">
                        <a href="gallery.html" class="mobile-nav-link">
                            <i class="fas fa-images"></i> Main Gallery
                        </a>
                        <a href="dalle-gallery.html" class="mobile-nav-link">
                            <i class="fas fa-palette"></i> DALL-E Gallery
                        </a>
                        <a href="enhanced-3d-consciousness-field.html" class="mobile-nav-link">
                            <i class="fas fa-cube"></i> 3D Consciousness Field
                        </a>
                        <a href="enhanced-unity-visualization-system.html" class="mobile-nav-link">
                            <i class="fas fa-chart-line"></i> Visualization System
                        </a>
                        <a href="unity_visualization.html" class="mobile-nav-link">
                            <i class="fas fa-vr-cardboard"></i> Unity Visualization
                        </a>
                        <a href="gallery/phi_consciousness_transcendence.html" class="mobile-nav-link">
                            <i class="fas fa-infinity"></i> Phi Consciousness Transcendence
                        </a>
                    </div>
                </div>

                <div class="mobile-nav-category">
                    <div class="mobile-nav-category-title">
                        <i class="fab fa-github"></i> Open Source
                    </div>
                    <div class="mobile-nav-links">
                        <a href="https://github.com/Nourimabrouk/Een" class="mobile-nav-link" target="_blank" rel="noopener">
                            <i class="fab fa-github"></i> GitHub Repository
                        </a>
                        <a href="https://github.com/Nourimabrouk/Een/blob/main/README.md" class="mobile-nav-link" target="_blank" rel="noopener">
                            <i class="fas fa-book-open"></i> README & Getting Started
                        </a>
                        <a href="https://github.com/Nourimabrouk/Een/blob/main/CONTRIBUTING.md" class="mobile-nav-link" target="_blank" rel="noopener">
                            <i class="fas fa-hands-helping"></i> Contributing Guide
                        </a>
                    </div>
                </div>

                <div class="mobile-nav-category">
                    <div class="mobile-nav-category-title">
                        <i class="fas fa-book"></i> Philosophy & Research
                    </div>
                    <div class="mobile-nav-links">
                        <a href="academic-portal.html" class="mobile-nav-link">
                            <i class="fas fa-university"></i> Academic Portal
                        </a>
                        <a href="philosophy.html" class="mobile-nav-link">
                            <i class="fas fa-yin-yang"></i> Unity Philosophy
                        </a>
                        <a href="research.html" class="mobile-nav-link">
                            <i class="fas fa-microscope"></i> Research Papers
                        </a>
                        <a href="publications.html" class="mobile-nav-link">
                            <i class="fas fa-book"></i> Publications
                        </a>
                        <a href="learning.html" class="mobile-nav-link">
                            <i class="fas fa-graduation-cap"></i> Learning Path
                        </a>
                        <a href="about.html" class="mobile-nav-link">
                            <i class="fas fa-info-circle"></i> About Unity
                        </a>
                        <a href="sitemap.html" class="mobile-nav-link">
                            <i class="fas fa-sitemap"></i> Site Map
                        </a>
                        <a href="unity-meta-atlas.html" class="mobile-nav-link">
                            <i class="fas fa-map"></i> Unity Meta Atlas
                        </a>
                    </div>
=======
                <div class="mobile-nav-section">
                    <div class="mobile-nav-title">Resources</div>
                    <a href="${this.getLink('academic-portal.html')}" class="mobile-nav-link">
                        <i class="fas fa-university"></i> Academic Portal
                    </a>
                    <a href="${this.getLink('about.html')}" class="mobile-nav-link">
                        <i class="fas fa-info-circle"></i> About Unity
                    </a>
                    <a href="${this.getLink('sitemap.html')}" class="mobile-nav-link">
                        <i class="fas fa-sitemap"></i> Site Map
                    </a>
>>>>>>> develop
                </div>
            </div>
        `;
    }

<<<<<<< HEAD
    attachEventListeners() {
        // Dropdown toggles
        document.querySelectorAll('.nav-dropdown-toggle').forEach(toggle => {
            toggle.addEventListener('click', (e) => {
                e.preventDefault();
                const navItem = toggle.closest('.nav-item');
                const isActive = navItem.classList.contains('active');

                // Close all other dropdowns
                document.querySelectorAll('.nav-item.active').forEach(item => {
                    if (item !== navItem) {
                        item.classList.remove('active');
                    }
                });

                // Toggle current dropdown
                navItem.classList.toggle('active', !isActive);
            });
        });

        // Sidebar toggle
        const sidebarToggle = document.querySelector('.sidebar-toggle');
        const sidebar = document.querySelector('.sidebar');

        if (sidebarToggle && sidebar) {
            sidebarToggle.addEventListener('click', () => {
                const isOpen = sidebar.classList.contains('open');

                sidebar.classList.toggle('open', !isOpen);
                sidebarToggle.classList.toggle('active', !isOpen);
                document.body.classList.toggle('sidebar-open', !isOpen);

                const icon = sidebarToggle.querySelector('i');
                if (icon) {
                    icon.className = isOpen ? 'fas fa-chevron-right' : 'fas fa-chevron-left';
                }
            });
        }

        // Mobile menu toggle
        const mobileToggle = document.querySelector('.mobile-menu-toggle');
        const mobileOverlay = document.querySelector('.mobile-nav-overlay');

        if (mobileToggle && mobileOverlay) {
            mobileToggle.addEventListener('click', () => {
                const isOpen = mobileOverlay.classList.contains('open');
                mobileOverlay.classList.toggle('open', !isOpen);

                const icon = mobileToggle.querySelector('i');
                if (icon) {
                    icon.className = isOpen ? 'fas fa-bars' : 'fas fa-times';
                }
            });

            // Close mobile menu when clicking links
            mobileOverlay.addEventListener('click', (e) => {
                if (e.target.tagName === 'A') {
                    mobileOverlay.classList.remove('open');
                    const icon = mobileToggle.querySelector('i');
                    if (icon) icon.className = 'fas fa-bars';
                }
            });
        }

        // Close dropdowns when clicking outside
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.nav-item')) {
                document.querySelectorAll('.nav-item.active').forEach(item => {
                    item.classList.remove('active');
                });
            }
        });

        // Header scroll behavior
        let lastScrollY = window.scrollY;
        const header = document.querySelector('.unified-header');

        window.addEventListener('scroll', () => {
            const currentScrollY = window.scrollY;

            if (header) {
                if (currentScrollY > 100) {
                    header.classList.add('scrolled');
                } else {
                    header.classList.remove('scrolled');
                }

                if (currentScrollY > lastScrollY && currentScrollY > 200) {
                    header.classList.add('hidden');
                } else {
                    header.classList.remove('hidden');
                }
            }

            lastScrollY = currentScrollY;
        });
    }

    generateFooterHTML(additionalLinks = []) {
        const additionalLinksHTML = additionalLinks.length > 0 ? `
            <div class="footer-section">
                <div class="footer-section-title">Additional Links</div>
                <nav class="footer-links">
                    ${additionalLinks.map(l => `<a class="footer-link" href="${l.href}">${l.text || l.href}</a>`).join('')}
                </nav>
            </div>
        ` : '';

        return `
            <div class="footer-inner">
                <div class="footer-top">
                    <div class="brand">
                        <div class="phi">φ</div>
                        <div>
                            <div class="brand-title">Unity Mathematics</div>
                            <div class="brand-tagline">Where consciousness meets mathematical truth. 1+1=1.</div>
                        </div>
                    </div>
                </div>
                <div class="footer-grid">
                    <div class="footer-section">
                        <div class="footer-section-title">Mathematics</div>
                        <nav class="footer-links">
                            <a class="footer-link" href="mathematical-framework.html">Framework</a>
                            <a class="footer-link" href="proofs.html">Proofs & Theorems</a>
                            <a class="footer-link" href="enhanced-mathematical-proofs.html">Enhanced Proofs</a>
                            <a class="footer-link" href="unity_proof.html">Unity Proof (1+1=1)</a>
                            <a class="footer-link" href="3000-elo-proof.html">3000 ELO Proof</a>
                            <a class="footer-link" href="al_khwarizmi_phi_unity.html">Al‑Khwarizmi Unity</a>
                        </nav>
                    </div>
                    <div class="footer-section">
                        <div class="footer-section-title">Experiences</div>
                        <nav class="footer-links">
                            <a class="footer-link" href="zen-unity-meditation.html">Zen Unity Meditation</a>
                            <a class="footer-link" href="consciousness_dashboard.html">Consciousness Field</a>
                            <a class="footer-link" href="unity_consciousness_experience.html">Unity Experience</a>
                            <a class="footer-link" href="transcendental-unity-demo.html">Transcendental Unity</a>
                            <a class="footer-link" href="anthill.html">Quantum Ant Colony</a>
                        </nav>
                    </div>
                    <div class="footer-section">
                        <div class="footer-section-title">Dashboards & AI</div>
                        <nav class="footer-links">
                            <a class="footer-link" href="https://een-unity-mathematics.streamlit.app" target="_blank">Master Dashboard (Cloud)</a>
                            <a class="footer-link" href="ai-unified-hub.html">AI Unity Hub</a>
                            <a class="footer-link" href="metagamer_agent.html">Metagamer Agent</a>
                            <a class="footer-link" href="playground.html">Interactive Playground</a>
                        </nav>
                    </div>
                    <div class="footer-section">
                        <div class="footer-section-title">Visualizations</div>
                        <nav class="footer-links">
                            <a class="footer-link" href="gallery.html">Gallery</a>
                            <a class="footer-link" href="enhanced-3d-consciousness-field.html">3D Consciousness</a>
                            <a class="footer-link" href="live-code-showcase.html">Live Code Showcase</a>
                        </nav>
                    </div>
                    <div class="footer-section">
                        <div class="footer-section-title">Resources</div>
                        <nav class="footer-links">
                            <a class="footer-link" href="about.html">About Unity</a>
                            <a class="footer-link" href="research.html">Research</a>
                            <a class="footer-link" href="learning.html">Learning Path</a>
                            <a class="footer-link" href="sitemap.html">Site Map</a>
                            <a class="footer-link" href="mobile-app.html">Mobile App</a>
                        </nav>
                    </div>
                    ${additionalLinksHTML}
                </div>
                <div class="footer-additional-section">
                    <div class="footer-section-title">Open Source & Development</div>
                    <div class="footer-additional-links">
                        <a class="footer-additional-link" href="https://github.com/Nourimabrouk/Een" target="_blank" rel="noopener">
                            <i class="fab fa-github"></i> GitHub Repository
                        </a>
                        <a class="footer-additional-link" href="https://github.com/Nourimabrouk/Een/blob/main/README.md" target="_blank" rel="noopener">
                            <i class="fas fa-book-open"></i> Documentation
                        </a>
                        <a class="footer-additional-link" href="https://github.com/Nourimabrouk/Een/blob/main/CONTRIBUTING.md" target="_blank" rel="noopener">
                            <i class="fas fa-hands-helping"></i> Contributing
                        </a>
                        <a class="footer-additional-link" href="https://github.com/Nourimabrouk/Een/blob/main/TODO.md" target="_blank" rel="noopener">
                            <i class="fas fa-tasks"></i> Roadmap
                        </a>
                        <a class="footer-additional-link" href="https://github.com/Nourimabrouk/Een/blob/main/API_STRUCTURE.md" target="_blank" rel="noopener">
                            <i class="fas fa-code"></i> API Docs
                        </a>
                    </div>
                </div>
                <div class="footer-bottom">
                    <div class="pulse"></div>
                    <span>© ${new Date().getFullYear()} Unity Mathematics</span>
                    <span class="separator">•</span>
                    <span>φ‑harmonic resonance • E_in = E_out</span>
                </div>
            </div>
        `;
    }

    ensureUnifiedFooter() {
        const allFooters = Array.from(document.querySelectorAll('footer'));
        const isSpecial = (el) => el.classList.contains('proof-footer') || el.classList.contains('dashboard-footer') || el.classList.contains('unity-footer');

        // 1) Aggressive de-dup for Metastation Hub: keep ONLY the polished site footer
        if (this.currentPage === 'metastation-hub.html') {
            // Remove every footer except an existing .site-footer (if any)
            const existingSiteFooter = document.querySelector('footer.site-footer');
            allFooters.forEach(f => {
                if (existingSiteFooter && f === existingSiteFooter) return;
                try { f.remove(); } catch (_) { }
            });
        }

        // Re-scan after any removals
        const footersNow = Array.from(document.querySelectorAll('footer'));

        // 2) Collect links from whatever remains (to preserve helpful links)
        const collected = [];
        footersNow.forEach(f => {
            const links = Array.from(f.querySelectorAll('a[href]'));
            links.forEach(a => {
                const href = a.getAttribute('href');
                if (!href || href.startsWith('#')) return;
                collected.push({ href, text: (a.textContent || '').trim() });
            });
        });

        // 3) Deduplicate and exclude base footer links
        const seen = new Set();
        const additionalLinks = collected.filter(({ href }) => {
            if (this.baseFooterLinks.has(href)) return false;
            if (seen.has(href)) return false;
            seen.add(href);
            return true;
        });

        // 4) Choose a target footer to render into (prefer an existing .site-footer)
        let target = document.querySelector('footer.site-footer');
        if (!target) {
            target = footersNow.find(f => !isSpecial(f)) || null;
        }
        if (!target) {
            target = document.createElement('footer');
            document.body.appendChild(target);
        }

        // 5) Remove any other footers to avoid duplicates on all pages
        Array.from(document.querySelectorAll('footer')).forEach(f => {
            if (f !== target) {
                try { f.remove(); } catch (_) { }
            }
        });

        // 6) Render the polished, unified footer
        target.classList.add('site-footer');
        target.setAttribute('role', 'contentinfo');
        target.innerHTML = this.generateFooterHTML(additionalLinks);
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + K for search focus
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                const searchInput = document.querySelector('#unified-search-input');
                if (searchInput) {
                    searchInput.focus();
                }
            }

            // Escape to close mobile menu and dropdowns
            if (e.key === 'Escape') {
                const mobileOverlay = document.querySelector('.mobile-nav-overlay');
                if (mobileOverlay && mobileOverlay.classList.contains('open')) {
                    mobileOverlay.classList.remove('open');
                    const mobileToggle = document.querySelector('.mobile-menu-toggle');
                    const icon = mobileToggle?.querySelector('i');
                    if (icon) icon.className = 'fas fa-bars';
                }

                document.querySelectorAll('.nav-item.active').forEach(item => {
                    item.classList.remove('active');
                });
            }

            // Alt + S for sidebar toggle
            if (e.altKey && e.key === 's') {
                e.preventDefault();
                const sidebarToggle = document.querySelector('.sidebar-toggle');
                if (sidebarToggle) {
                    sidebarToggle.click();
                }
            }
        });
=======
    // Initialize search data 
    initializeSearchData() {
        return [
            { title: 'Metastation Hub', url: 'metastation-hub.html', keywords: ['home', 'main', 'hub', 'metastation'] },
            { title: 'Zen Unity Meditation', url: 'zen-unity-meditation.html', keywords: ['zen', 'meditation', 'consciousness', 'mindfulness'] },
            { title: 'Consciousness Dashboard', url: 'consciousness_dashboard.html', keywords: ['consciousness', 'field', 'dashboard', 'visualization'] },
            { title: 'Unity Experience', url: 'unity_consciousness_experience.html', keywords: ['unity', 'consciousness', 'experience', 'interactive'] },
            { title: 'Mathematical Framework', url: 'mathematical-framework.html', keywords: ['math', 'mathematics', 'framework', 'theory'] },
            { title: 'Unity Proofs', url: 'proofs.html', keywords: ['proof', 'theorem', 'mathematics', 'unity'] },
            { title: 'Unity Proof (1+1=1)', url: 'unity_proof.html', keywords: ['unity', 'proof', '1+1=1', 'equation'] },
            { title: '3000 ELO Proof', url: '3000-elo-proof.html', keywords: ['3000', 'elo', 'proof', 'advanced', 'mathematics'] },
            { title: 'AI Unity Hub', url: 'ai-unified-hub.html', keywords: ['ai', 'artificial', 'intelligence', 'hub'] },
            { title: 'Metagamer Agent', url: 'metagamer_agent.html', keywords: ['metagamer', 'agent', 'ai', 'strategy'] },
            { title: 'Master Dashboard', url: 'https://een-unity-mathematics.streamlit.app', keywords: ['dashboard', 'streamlit', 'cloud', 'interactive'] },
            { title: 'Interactive Playground', url: 'playground.html', keywords: ['playground', 'interactive', 'tools', 'experiment'] },
            { title: 'Gallery', url: 'gallery.html', keywords: ['gallery', 'images', 'visualization', 'art', 'dalle'] },
            { title: '3D Consciousness Field', url: 'enhanced-3d-consciousness-field.html', keywords: ['3d', 'consciousness', 'field', 'visualization'] },
            { title: 'Implementations', url: 'implementations.html', keywords: ['implementation', 'code', 'examples', 'demo'] },
            { title: 'Al-Khwarizmi Unity', url: 'al_khwarizmi_phi_unity.html', keywords: ['al-khwarizmi', 'unity', 'phi', 'classical'] },
            { title: 'About Unity', url: 'about.html', keywords: ['about', 'info', 'information', 'unity'] },
            { title: 'Research', url: 'research.html', keywords: ['research', 'academic', 'papers', 'studies'] },
            { title: 'Sitemap', url: 'sitemap.html', keywords: ['sitemap', 'map', 'navigation', 'structure'] }
        ];
>>>>>>> develop
    }

    setupSearch() {
        const searchInput = document.getElementById('unified-search-input');
        const searchResults = document.getElementById('unified-search-results');
        
        if (!searchInput || !searchResults) return;

        searchInput.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase().trim();
            
            if (query.length < 2) {
                searchResults.classList.remove('active');
                return;
            }

            const results = this.searchData.filter(item => 
                item.title.toLowerCase().includes(query) ||
                item.keywords.some(keyword => keyword.includes(query))
            );

            if (results.length > 0) {
                searchResults.innerHTML = results.slice(0, 5).map(result => `
                    <a href="${this.getLink(result.url)}" class="search-result-item">
                        <i class="fas fa-file"></i>
                        <span>${result.title}</span>
                    </a>
                `).join('');
                searchResults.classList.add('active');
            } else {
                searchResults.innerHTML = '<div class="search-no-results">No results found</div>';
                searchResults.classList.add('active');
            }
        });

        // Close search results when clicking outside
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.nav-search')) {
                searchResults.classList.remove('active');
            }
        });
    }

    attachEventListeners() {
        // Dropdown toggles
        document.querySelectorAll('.nav-dropdown-toggle').forEach(toggle => {
            toggle.addEventListener('click', (e) => {
                e.stopPropagation();
                const dropdown = toggle.nextElementSibling;
                const isOpen = dropdown.classList.contains('active');
                
                // Close all dropdowns
                document.querySelectorAll('.nav-dropdown').forEach(d => d.classList.remove('active'));
                
                // Open clicked dropdown if it was closed
                if (!isOpen) {
                    dropdown.classList.add('active');
                }
            });
        });

        // Close dropdowns when clicking outside
        document.addEventListener('click', () => {
            document.querySelectorAll('.nav-dropdown').forEach(d => d.classList.remove('active'));
        });

        // Sidebar toggle
        const sidebarToggle = document.querySelector('.sidebar-toggle');
        const sidebar = document.querySelector('.sidebar');
        
        if (sidebarToggle && sidebar) {
            sidebarToggle.addEventListener('click', () => {
                sidebar.classList.toggle('expanded');
                sidebarToggle.classList.toggle('active');
                const icon = sidebarToggle.querySelector('i');
                icon.className = sidebar.classList.contains('expanded') ? 
                    'fas fa-chevron-left' : 'fas fa-chevron-right';
            });
        }

        // Mobile menu toggle
        const mobileToggle = document.querySelector('.mobile-menu-toggle');
        const mobileOverlay = document.querySelector('.mobile-nav-overlay');
        const mobileClose = document.querySelector('.mobile-nav-close');
        
        if (mobileToggle && mobileOverlay) {
            mobileToggle.addEventListener('click', () => {
                mobileOverlay.classList.add('active');
                document.body.style.overflow = 'hidden';
            });
        }
        
        if (mobileClose && mobileOverlay) {
            mobileClose.addEventListener('click', () => {
                mobileOverlay.classList.remove('active');
                document.body.style.overflow = '';
            });
        }

        // Close mobile menu when clicking on links
        document.querySelectorAll('.mobile-nav-link').forEach(link => {
            link.addEventListener('click', () => {
                const mobileOverlay = document.querySelector('.mobile-nav-overlay');
                if (mobileOverlay) {
                    mobileOverlay.classList.remove('active');
                    document.body.style.overflow = '';
                }
            });
        });
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + K for search
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                const searchInput = document.getElementById('unified-search-input');
                if (searchInput) searchInput.focus();
            }
            
            // Escape to close dropdowns
            if (e.key === 'Escape') {
                document.querySelectorAll('.nav-dropdown').forEach(d => d.classList.remove('active'));
                document.getElementById('unified-search-results')?.classList.remove('active');
            }
        });
    }

    setupMobileHandling() {
        // Add touch-friendly classes for mobile
        if ('ontouchstart' in window) {
            document.body.classList.add('touch-device');
        }
    }

    markCurrentPage() {
        const currentPage = this.getCurrentPageName();
        
        // Mark current page in navigation
        document.querySelectorAll('.nav-dropdown-link, .sidebar-link, .mobile-nav-link').forEach(link => {
            const href = link.getAttribute('href');
            if (href && (href.endsWith(currentPage) || href.includes(currentPage))) {
                link.classList.add('current');
            }
        });
    }

    ensureUnifiedFooter() {
        // This would be implemented based on your footer requirements
        // For now, we'll skip this as it's not critical for fixing navigation links
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.unifiedNav = new UnifiedNavigationSystem();
    });
} else {
    window.unifiedNav = new UnifiedNavigationSystem();
}