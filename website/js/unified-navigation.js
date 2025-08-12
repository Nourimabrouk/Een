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
            'mathematical-framework.html', 'proofs.html', 'enhanced-mathematical-proofs.html', 'unity_proof.html', '3000-elo-proof.html', 'al_khwarizmi_phi_unity.html', 'unity-mathematics-synthesis.html',
            // Experiences & AI
            'zen-unity-meditation.html', 'consciousness_dashboard.html', 'consciousness_dashboard_clean.html', 'unity-mathematics-experience.html', 'ai-unified-hub.html', 'metagamer_agent.html',
            // Visualizations & Tools
            'gallery.html', 'dalle-gallery.html', 'enhanced-3d-consciousness-field.html', 'enhanced-unity-visualization-system.html', 'unity_visualization.html', 'gallery/phi_consciousness_transcendence.html', 'dashboards.html', 'unity-dashboard.html', 'unity-calculator-live.html', 'playground.html', 'mathematical_playground.html', 'live-code-showcase.html', 'examples/unity-calculator.html', 'examples/phi-harmonic-explorer.html',
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
                            <a href="${this.getLink('unity-axioms.html')}" class="nav-dropdown-link">
                                <i class="fas fa-balance-scale"></i> Unity Axioms
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
                             <a href="${this.getLink('enhanced-mathematical-proofs.html')}" class="nav-dropdown-link">
                                 <i class="fas fa-book-open"></i> Enhanced Mathematical Proofs
                             </a>
                             <a href="${this.getLink('unity_proof.html')}" class="nav-dropdown-link">
                                 <i class="fas fa-equals"></i> Unity Proof (1+1=1)
                             </a>
                            <a href="${this.getLink('3000-elo-proof.html')}" class="nav-dropdown-link">
                                <i class="fas fa-trophy"></i> 3000 ELO Proof
                            </a>
                            <a href="${this.getLink('implementations-gallery.html')}" class="nav-dropdown-link">
                                <i class="fas fa-code"></i> Implementations
                            </a>
                            <a href="${this.getLink('al_khwarizmi_phi_unity.html')}" class="nav-dropdown-link">
                                <i class="fas fa-scroll"></i> Al-Khwarizmi Unity
                            </a>
                             <a href="${this.getLink('unity-mathematics-synthesis.html')}" class="nav-dropdown-link">
                                 <i class="fas fa-project-diagram"></i> Unity Mathematics Synthesis
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
                            <a href="http://localhost:8501" class="nav-dropdown-link" target="_blank">
                                <i class="fas fa-rocket"></i> Metastation Command Center
                            </a>
                            <a href="${this.getLink('dashboards.html')}" class="nav-dropdown-link">
                                <i class="fas fa-th-large"></i> All Dashboards
                            </a>
                            <a href="${this.getLink('unity-dashboard.html')}" class="nav-dropdown-link">
                                <i class="fas fa-infinity"></i> Unity Dashboard (Full)
                            </a>
                            <a href="${this.getLink('playground.html')}" class="nav-dropdown-link">
                                <i class="fas fa-gamepad"></i> Interactive Playground
                            </a>
                            <a href="${this.getLink('mathematical_playground.html')}" class="nav-dropdown-link">
                                <i class="fas fa-calculator"></i> Math Playground
                            </a>
                            <a href="${this.getLink('unity-calculator-live.html')}" class="nav-dropdown-link">
                                <i class="fas fa-magic"></i> Unity Calculator Live
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
                            <a href="${this.getLink('academic-portal.html')}" class="nav-dropdown-link">
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
                    <a href="${this.getLink('consciousness_dashboard_clean.html')}" class="sidebar-link">
                        <i class="fas fa-heartbeat"></i> Consciousness Field (Clean)
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
                    <a href="${this.getLink('enhanced-mathematical-proofs.html')}" class="sidebar-link">
                        <i class="fas fa-book-open"></i> Enhanced Proofs
                    </a>
                         <a href="${this.getLink('unity_proof.html')}" class="sidebar-link">
                             <i class="fas fa-equals"></i> Unity Proof (1+1=1)
                         </a>
                    <a href="${this.getLink('3000-elo-proof.html')}" class="sidebar-link">
                        <i class="fas fa-trophy"></i> 3000 ELO Proof
                    </a>
                    <a href="${this.getLink('implementations-gallery.html')}" class="sidebar-link">
                        <i class="fas fa-code"></i> Implementations Gallery
                    </a>
                    <a href="${this.getLink('al_khwarizmi_phi_unity.html')}" class="sidebar-link">
                        <i class="fas fa-scroll"></i> Al-Khwarizmi Unity
                    </a>
                    <a href="${this.getLink('unity-mathematics-synthesis.html')}" class="sidebar-link">
                        <i class="fas fa-project-diagram"></i> Unity Synthesis
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
                    <a href="${this.getLink('dashboards.html')}" class="sidebar-link">
                        <i class="fas fa-th-large"></i> All Dashboards
                    </a>
                    <a href="${this.getLink('playground.html')}" class="sidebar-link">
                        <i class="fas fa-gamepad"></i> Interactive Playground
                    </a>
                    <a href="${this.getLink('mathematical_playground.html')}" class="sidebar-link">
                        <i class="fas fa-calculator"></i> Math Playground
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
                    <div class="sidebar-title">Academic & Research</div>
                    <a href="${this.getLink('academic-portal.html')}" class="sidebar-link featured">
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
                </div>
            </div>
        `;
    }

    // Initialize search data 
    initializeSearchData() {
        return [
            { title: 'Metastation Hub', url: 'metastation-hub.html', keywords: ['home', 'main', 'hub', 'metastation'] },
            { title: 'Zen Unity Meditation', url: 'zen-unity-meditation.html', keywords: ['zen', 'meditation', 'consciousness', 'mindfulness'] },
            { title: 'Consciousness Dashboard', url: 'consciousness_dashboard.html', keywords: ['consciousness', 'field', 'dashboard', 'visualization'] },
            { title: 'Mathematical Framework', url: 'mathematical-framework.html', keywords: ['math', 'mathematics', 'framework', 'theory'] },
            { title: 'Unity Proofs', url: 'proofs.html', keywords: ['proof', 'theorem', 'mathematics', 'unity'] },
            { title: 'AI Unity Hub', url: 'ai-unified-hub.html', keywords: ['ai', 'artificial', 'intelligence', 'hub'] },
            { title: 'Academic Portal', url: 'academic-portal.html', keywords: ['academic', 'research', 'papers', 'publications'] },
            { title: 'Gallery', url: 'gallery.html', keywords: ['gallery', 'images', 'visualization', 'art'] },
            { title: 'Implementations', url: 'implementations-gallery.html', keywords: ['implementation', 'code', 'examples', 'demo'] },
            { title: 'About Unity', url: 'about.html', keywords: ['about', 'info', 'information', 'unity'] },
            { title: 'Unity Philosophy', url: 'philosophy.html', keywords: ['philosophy', 'theory', 'concept'] },
            { title: 'Sitemap', url: 'sitemap.html', keywords: ['sitemap', 'map', 'navigation', 'structure'] }
        ];
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