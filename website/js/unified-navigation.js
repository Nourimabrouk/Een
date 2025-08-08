/**
 * Unified Navigation System - Meta-Optimal Unity Mathematics
 * Single, consolidated navigation system for all pages
 * Version: 4.0.0 - Complete consolidation and meta-optimization
 */

class UnifiedNavigationSystem {
    constructor() {
        this.isInitialized = false;
        this.currentPage = this.getCurrentPageName();
        this.searchData = this.initializeSearchData();
        this.init();
    }

    getCurrentPageName() {
        const path = window.location.pathname;
        const page = path.split('/').pop() || 'index.html';
        return page === '' ? 'index.html' : page;
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
        console.log('✅ Unified Navigation System initialized');
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

        // Inject footer if not present
        if (!document.querySelector('.site-footer')) {
            const footer = document.createElement('footer');
            footer.className = 'site-footer';
            footer.style.background = 'rgba(18,18,26,0.95)';
            footer.style.borderTop = '1px solid rgba(255, 215, 0, 0.3)';
            footer.style.color = 'var(--text-secondary, #cfcfe6)';
            footer.innerHTML = this.generateFooterHTML();
            document.body.appendChild(footer);
        }
    }

    generateHeaderHTML() {
        return `
            <div class="nav-container">
                <a href="metastation-hub.html" class="nav-logo">
                    <div class="phi-symbol">φ</div>
                    <span class="logo-text">Unity Mathematics</span>
                </a>

                <nav class="nav-main">
                    <div class="nav-item">
                        <button class="nav-dropdown-toggle" type="button">
                            Experiences <i class="fas fa-chevron-down"></i>
                        </button>
                        <div class="nav-dropdown">
                            <a href="zen-unity-meditation.html" class="nav-dropdown-link">
                                <i class="fas fa-om"></i> Zen Unity Meditation
                            </a>
                            <a href="consciousness_dashboard.html" class="nav-dropdown-link">
                                <i class="fas fa-brain"></i> Consciousness Field
                            </a>
                            <a href="unity_consciousness_experience.html" class="nav-dropdown-link">
                                <i class="fas fa-infinity"></i> Unity Consciousness
                            </a>
                            <a href="transcendental-unity-demo.html" class="nav-dropdown-link">
                                <i class="fas fa-atom"></i> Transcendental Unity
                            </a>
                            <a href="anthill.html" class="nav-dropdown-link">
                                <i class="fas fa-bug"></i> Quantum Ant Colony
                            </a>
                        </div>
                    </div>

                    <div class="nav-item">
                        <button class="nav-dropdown-toggle" type="button">
                            Mathematics <i class="fas fa-chevron-down"></i>
                        </button>
                        <div class="nav-dropdown">
                            <a href="mathematical-framework.html" class="nav-dropdown-link">
                                <i class="fas fa-square-root-alt"></i> Mathematical Framework
                            </a>
                            <a href="proofs.html" class="nav-dropdown-link">
                                <i class="fas fa-check-circle"></i> Proofs & Theorems
                            </a>
                             <a href="enhanced-mathematical-proofs.html" class="nav-dropdown-link">
                                 <i class="fas fa-book-open"></i> Enhanced Mathematical Proofs
                             </a>
                             <a href="unity_proof.html" class="nav-dropdown-link">
                                 <i class="fas fa-equals"></i> Unity Proof (1+1=1)
                             </a>
                            <a href="3000-elo-proof.html" class="nav-dropdown-link">
                                <i class="fas fa-trophy"></i> 3000 ELO Proof
                            </a>
                            <a href="implementations-gallery.html" class="nav-dropdown-link">
                                <i class="fas fa-code"></i> Implementations
                            </a>
                            <a href="al_khwarizmi_phi_unity.html" class="nav-dropdown-link">
                                <i class="fas fa-scroll"></i> Al-Khwarizmi Unity
                            </a>
                             <a href="unity-mathematics-synthesis.html" class="nav-dropdown-link">
                                 <i class="fas fa-project-diagram"></i> Unity Mathematics Synthesis
                             </a>
                            <a href="metagambit.html" class="nav-dropdown-link">
                                <i class="fas fa-chess"></i> Metagambit Framework
                            </a>
                        </div>
                    </div>

                    <div class="nav-item">
                        <button class="nav-dropdown-toggle" type="button">
                            AI Systems <i class="fas fa-chevron-down"></i>
                        </button>
                        <div class="nav-dropdown">
                            <a href="ai-unified-hub.html" class="nav-dropdown-link">
                                <i class="fas fa-brain"></i> AI Unity Hub
                            </a>
                            <a href="ai-agents-ecosystem.html" class="nav-dropdown-link">
                                <i class="fas fa-network-wired"></i> AI Agents Ecosystem
                            </a>
                            <a href="agents.html" class="nav-dropdown-link">
                                <i class="fas fa-robot"></i> Unity Agents
                            </a>
                            <a href="metagamer_agent.html" class="nav-dropdown-link">
                                <i class="fas fa-gamepad"></i> Metagamer Agent
                            </a>
                            <a href="openai-integration.html" class="nav-dropdown-link">
                                <i class="fas fa-plug"></i> OpenAI Integration
                            </a>
                            <a href="enhanced-ai-demo.html" class="nav-dropdown-link">
                                <i class="fas fa-sparkles"></i> Enhanced AI Demo
                            </a>
                        </div>

                    </div>

                    <div class="nav-item">
                        <button class="nav-dropdown-toggle" type="button">
                            Dashboards <i class="fas fa-chevron-down"></i>
                        </button>
                        <div class="nav-dropdown">
                            <a href="dashboards.html" class="nav-dropdown-link">
                                <i class="fas fa-th-large"></i> All Dashboards
                            </a>
                            <a href="playground.html" class="nav-dropdown-link">
                                <i class="fas fa-gamepad"></i> Interactive Playground
                            </a>
                            <a href="mathematical_playground.html" class="nav-dropdown-link">
                                <i class="fas fa-calculator"></i> Math Playground
                            </a>
                            <a href="live-code-showcase.html" class="nav-dropdown-link">
                                <i class="fas fa-terminal"></i> Live Code Showcase
                            </a>
                            <a href="unity-advanced-features.html" class="nav-dropdown-link">
                                <i class="fas fa-cogs"></i> Advanced Features
                            </a>
                        </div>
                    </div>

                    <div class="nav-item">
                        <button class="nav-dropdown-toggle" type="button">
                            Gallery <i class="fas fa-chevron-down"></i>
                        </button>
                        <div class="nav-dropdown">
                            <a href="gallery.html" class="nav-dropdown-link">
                                <i class="fas fa-images"></i> Main Gallery
                            </a>
                            <a href="dalle-gallery.html" class="nav-dropdown-link">
                                <i class="fas fa-palette"></i> DALL-E Consciousness Gallery
                            </a>
                            <a href="enhanced-3d-consciousness-field.html" class="nav-dropdown-link">
                                <i class="fas fa-cube"></i> 3D Consciousness Field
                            </a>
                            <a href="enhanced-unity-visualization-system.html" class="nav-dropdown-link">
                                <i class="fas fa-chart-line"></i> Visualization System
                            </a>
                             <a href="unity_visualization.html" class="nav-dropdown-link">
                                 <i class="fas fa-vr-cardboard"></i> Unity Visualization
                             </a>
                             <a href="gallery/phi_consciousness_transcendence.html" class="nav-dropdown-link">
                                 <i class="fas fa-infinity"></i> Phi Consciousness Transcendence
                             </a>
                        </div>
                    </div>

                    <div class="nav-item">
                        <button class="nav-dropdown-toggle" type="button">
                            More <i class="fas fa-chevron-down"></i>
                        </button>
                        <div class="nav-dropdown">
                            <a href="philosophy.html" class="nav-dropdown-link">
                                <i class="fas fa-yin-yang"></i> Unity Philosophy
                            </a>
                            <a href="research.html" class="nav-dropdown-link">
                                <i class="fas fa-microscope"></i> Research Papers
                            </a>
                            <a href="publications.html" class="nav-dropdown-link">
                                <i class="fas fa-book"></i> Publications
                            </a>
                            <a href="learning.html" class="nav-dropdown-link">
                                <i class="fas fa-graduation-cap"></i> Learning Path
                            </a>
                            <a href="about.html" class="nav-dropdown-link">
                                <i class="fas fa-info-circle"></i> About Unity
                            </a>
                            <a href="sitemap.html" class="nav-dropdown-link">
                                <i class="fas fa-sitemap"></i> Site Map
                            </a>
                             <a href="unity-meta-atlas.html" class="nav-dropdown-link">
                                 <i class="fas fa-map"></i> Unity Meta Atlas
                             </a>
                             <a href="mobile-app.html" class="nav-dropdown-link">
                                 <i class="fas fa-mobile-alt"></i> Mobile App
                             </a>
                             <div class="nav-dropdown-link" style="pointer-events:none; opacity:0.85; font-weight:700; margin-top:0.25rem;">
                                 <i class="fas fa-flask"></i> Examples
                             </div>
                             <a href="examples/unity-calculator.html" class="nav-dropdown-link">
                                 <i class="fas fa-calculator"></i> Unity Calculator
                             </a>
                             <a href="examples/phi-harmonic-explorer.html" class="nav-dropdown-link">
                                 <i class="fas fa-wave-square"></i> φ‑Harmonic Explorer
                             </a>
                             <a href="examples/index.html" class="nav-dropdown-link">
                                 <i class="fas fa-flask"></i> Examples Home
                             </a>
                        </div>
                    </div>
                </nav>

                <div class="nav-search">
                    <input type="text" class="search-input" id="unified-search-input" 
                           placeholder="Search unity mathematics..." 
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
                    <a href="metastation-hub.html" class="sidebar-link featured">
                        <i class="fas fa-star"></i> Metastation Hub
                    </a>
                    <a href="zen-unity-meditation.html" class="sidebar-link">
                        <i class="fas fa-om"></i> Zen Unity Meditation
                    </a>
                    <a href="consciousness_dashboard.html" class="sidebar-link">
                        <i class="fas fa-brain"></i> Consciousness Field
                    </a>
                    <a href="consciousness_dashboard_clean.html" class="sidebar-link">
                        <i class="fas fa-heartbeat"></i> Consciousness Field (Clean)
                    </a>
                    <a href="transcendental-unity-demo.html" class="sidebar-link">
                        <i class="fas fa-atom"></i> Transcendental Unity
                    </a>
                    <a href="anthill.html" class="sidebar-link">
                        <i class="fas fa-bug"></i> Quantum Ant Colony
                    </a>
                    <a href="unity-mathematics-experience.html" class="sidebar-link">
                        <i class="fas fa-equals"></i> Unity Mathematics Experience
                    </a>
                </div>

                <div class="sidebar-section">
                    <div class="sidebar-title">Mathematics & Proofs</div>
                    <a href="mathematical-framework.html" class="sidebar-link">
                        <i class="fas fa-square-root-alt"></i> Mathematical Framework
                    </a>
                    <a href="proofs.html" class="sidebar-link">
                        <i class="fas fa-check-circle"></i> Proofs & Theorems
                    </a>
                    <a href="enhanced-mathematical-proofs.html" class="sidebar-link">
                        <i class="fas fa-book-open"></i> Enhanced Proofs
                    </a>
                         <a href="unity_proof.html" class="sidebar-link">
                             <i class="fas fa-equals"></i> Unity Proof (1+1=1)
                         </a>
                    <a href="3000-elo-proof.html" class="sidebar-link">
                        <i class="fas fa-trophy"></i> 3000 ELO Proof
                    </a>
                    <a href="implementations-gallery.html" class="sidebar-link">
                        <i class="fas fa-code"></i> Implementations Gallery
                    </a>
                    <a href="al_khwarizmi_phi_unity.html" class="sidebar-link">
                        <i class="fas fa-scroll"></i> Al-Khwarizmi Unity
                    </a>
                    <a href="unity-mathematics-synthesis.html" class="sidebar-link">
                        <i class="fas fa-project-diagram"></i> Unity Synthesis
                    </a>
                    <a href="metagambit.html" class="sidebar-link">
                        <i class="fas fa-chess"></i> Metagambit Framework
                    </a>
                </div>

                <div class="sidebar-section">
                    <div class="sidebar-title">AI & Agents</div>
                    <a href="ai-unified-hub.html" class="sidebar-link featured">
                        <i class="fas fa-brain"></i> AI Unity Hub
                    </a>
                    <a href="ai-agents-ecosystem.html" class="sidebar-link">
                        <i class="fas fa-network-wired"></i> AI Agents Ecosystem
                    </a>
                    <a href="agents.html" class="sidebar-link">
                        <i class="fas fa-robot"></i> Unity Agents
                    </a>
                    <a href="metagamer_agent.html" class="sidebar-link">
                        <i class="fas fa-gamepad"></i> Metagamer Agent
                    </a>
                    <a href="openai-integration.html" class="sidebar-link">
                        <i class="fas fa-plug"></i> OpenAI Integration
                    </a>
                </div>

                <div class="sidebar-section">
                    <div class="sidebar-title">Dashboards & Tools</div>
                    <a href="dashboards.html" class="sidebar-link">
                        <i class="fas fa-th-large"></i> All Dashboards
                    </a>
                    <a href="playground.html" class="sidebar-link">
                        <i class="fas fa-gamepad"></i> Interactive Playground
                    </a>
                    <a href="mathematical_playground.html" class="sidebar-link">
                        <i class="fas fa-calculator"></i> Math Playground
                    </a>
                    <a href="live-code-showcase.html" class="sidebar-link">
                        <i class="fas fa-terminal"></i> Live Code Showcase
                    </a>
                    <a href="examples/unity-calculator.html" class="sidebar-link">
                        <i class="fas fa-calculator"></i> Unity Calculator
                    </a>
                    <a href="examples/phi-harmonic-explorer.html" class="sidebar-link">
                        <i class="fas fa-wave-square"></i> φ-Harmonic Explorer
                    </a>
                </div>

                <div class="sidebar-section">
                    <div class="sidebar-title">Gallery & Visualizations</div>
                    <a href="gallery.html" class="sidebar-link">
                        <i class="fas fa-images"></i> Main Gallery
                    </a>
                    <a href="dalle-gallery.html" class="sidebar-link featured">
                        <i class="fas fa-palette"></i> DALL-E Gallery
                    </a>
                    <a href="enhanced-3d-consciousness-field.html" class="sidebar-link">
                        <i class="fas fa-cube"></i> 3D Consciousness
                    </a>
                    <a href="enhanced-unity-visualization-system.html" class="sidebar-link">
                        <i class="fas fa-chart-line"></i> Visualization System
                    </a>
                    <a href="unity_visualization.html" class="sidebar-link">
                        <i class="fas fa-vr-cardboard"></i> Unity Visualization
                    </a>
                    <a href="gallery/phi_consciousness_transcendence.html" class="sidebar-link">
                        <i class="fas fa-infinity"></i> Phi Consciousness Transcendence
                    </a>
                </div>

                <div class="sidebar-section">
                    <div class="sidebar-title">Philosophy & Research</div>
                    <a href="philosophy.html" class="sidebar-link">
                        <i class="fas fa-yin-yang"></i> Unity Philosophy
                    </a>
                    <a href="research.html" class="sidebar-link">
                        <i class="fas fa-microscope"></i> Research Papers
                    </a>
                    <a href="publications.html" class="sidebar-link">
                        <i class="fas fa-book"></i> Publications
                    </a>
                    <a href="learning.html" class="sidebar-link">
                        <i class="fas fa-graduation-cap"></i> Learning Path
                    </a>
                    <a href="about.html" class="sidebar-link">
                        <i class="fas fa-info-circle"></i> About Unity
                    </a>
                    <a href="unity-meta-atlas.html" class="sidebar-link">
                        <i class="fas fa-map"></i> Unity Meta Atlas
                    </a>
                </div>

                <div class="sidebar-section">
                    <div class="sidebar-title">Resources</div>
                    <a href="sitemap.html" class="sidebar-link">
                        <i class="fas fa-sitemap"></i> Full Site Map
                    </a>
                    <a href="mobile-app.html" class="sidebar-link">
                        <i class="fas fa-mobile-alt"></i> Mobile App
                    </a>
                    <a href="further-reading.html" class="sidebar-link">
                        <i class="fas fa-book-open"></i> Further Reading
                    </a>
                </div>
            </div>
        `;
    }

    generateMobileNavHTML() {
        return `
            <div class="mobile-search">
                <input type="text" class="search-input" placeholder="Search unity mathematics..." 
                       aria-label="Mobile search">
                <i class="fas fa-search search-icon"></i>
            </div>

            <div class="mobile-nav-categories">
                <div class="mobile-nav-category">
                    <div class="mobile-nav-category-title">
                        <i class="fas fa-star"></i> Featured Experiences
                    </div>
                    <div class="mobile-nav-links">
                        <a href="metastation-hub.html" class="mobile-nav-link">
                            <i class="fas fa-star"></i> Metastation Hub
                        </a>
                        <a href="zen-unity-meditation.html" class="mobile-nav-link">
                            <i class="fas fa-om"></i> Zen Unity Meditation
                        </a>
                        <a href="consciousness_dashboard.html" class="mobile-nav-link">
                            <i class="fas fa-brain"></i> Consciousness Field
                        </a>
                            <a href="consciousness_dashboard_clean.html" class="mobile-nav-link">
                                <i class="fas fa-heartbeat"></i> Consciousness Field (Clean)
                            </a>
                        <a href="transcendental-unity-demo.html" class="mobile-nav-link">
                            <i class="fas fa-atom"></i> Transcendental Unity
                        </a>
                        <a href="anthill.html" class="mobile-nav-link">
                            <i class="fas fa-bug"></i> Quantum Ant Colony
                        </a>
                            <a href="unity-mathematics-experience.html" class="mobile-nav-link">
                                <i class="fas fa-equals"></i> Unity Mathematics Experience
                            </a>
                    </div>
                </div>

                <div class="mobile-nav-category">
                    <div class="mobile-nav-category-title">
                        <i class="fas fa-square-root-alt"></i> Mathematics & Proofs
                    </div>
                    <div class="mobile-nav-links">
                        <a href="mathematical-framework.html" class="mobile-nav-link">
                            <i class="fas fa-square-root-alt"></i> Mathematical Framework
                        </a>
                        <a href="proofs.html" class="mobile-nav-link">
                            <i class="fas fa-check-circle"></i> Proofs & Theorems
                        </a>
                        <a href="enhanced-mathematical-proofs.html" class="mobile-nav-link">
                            <i class="fas fa-book-open"></i> Enhanced Mathematical Proofs
                        </a>
                             <a href="unity_proof.html" class="mobile-nav-link">
                                 <i class="fas fa-equals"></i> Unity Proof (1+1=1)
                             </a>
                        <a href="3000-elo-proof.html" class="mobile-nav-link">
                            <i class="fas fa-trophy"></i> 3000 ELO Proof
                        </a>
                        <a href="implementations-gallery.html" class="mobile-nav-link">
                            <i class="fas fa-code"></i> Implementations Gallery
                        </a>
                        <a href="al_khwarizmi_phi_unity.html" class="mobile-nav-link">
                            <i class="fas fa-scroll"></i> Al-Khwarizmi Unity
                        </a>
                        <a href="unity-mathematics-synthesis.html" class="mobile-nav-link">
                            <i class="fas fa-project-diagram"></i> Unity Mathematics Synthesis
                        </a>
                        <a href="metagambit.html" class="mobile-nav-link">
                            <i class="fas fa-chess"></i> Metagambit Framework
                        </a>
                    </div>
                </div>

                <div class="mobile-nav-category">
                    <div class="mobile-nav-category-title">
                        <i class="fas fa-robot"></i> AI Systems & Agents
                    </div>
                    <div class="mobile-nav-links">
                        <a href="ai-unified-hub.html" class="mobile-nav-link">
                            <i class="fas fa-brain"></i> AI Unity Hub
                        </a>
                        <a href="ai-agents-ecosystem.html" class="mobile-nav-link">
                            <i class="fas fa-network-wired"></i> AI Agents Ecosystem
                        </a>
                        <a href="agents.html" class="mobile-nav-link">
                            <i class="fas fa-robot"></i> Unity Agents
                        </a>
                        <a href="metagamer_agent.html" class="mobile-nav-link">
                            <i class="fas fa-gamepad"></i> Metagamer Agent
                        </a>
                        <a href="openai-integration.html" class="mobile-nav-link">
                            <i class="fas fa-plug"></i> OpenAI Integration
                        </a>
                        <a href="enhanced-ai-demo.html" class="mobile-nav-link">
                            <i class="fas fa-sparkles"></i> Enhanced AI Demo
                        </a>
                    </div>
                </div>

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
                        <i class="fas fa-book"></i> Philosophy & Research
                    </div>
                    <div class="mobile-nav-links">
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
                </div>
            </div>
        `;
    }

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

    generateFooterHTML() {
        return `
            <div style="max-width:1200px; margin:0 auto; padding:1.5rem 2rem; display:grid; gap:1rem; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));">
                <div>
                    <div style="color:#FFD700; font-weight:700; margin-bottom:0.5rem;">Unity Mathematics</div>
                    <p style="margin:0; opacity:0.85;">Where consciousness meets mathematical truth. 1+1=1.</p>
                </div>
                <div>
                    <div style="color:#FFD700; font-weight:600; margin-bottom:0.5rem;">Mathematics</div>
                    <nav style="display:grid; gap:0.3rem;">
                        <a href="mathematical-framework.html">Framework</a>
                        <a href="proofs.html">Proofs & Theorems</a>
                        <a href="enhanced-mathematical-proofs.html">Enhanced Proofs</a>
                        <a href="unity_proof.html">Unity Proof (1+1=1)</a>
                        <a href="3000-elo-proof.html">3000 ELO Proof</a>
                        <a href="al_khwarizmi_phi_unity.html">Al‑Khwarizmi Unity</a>
                        <a href="unity-mathematics-synthesis.html">Unity Synthesis</a>
                    </nav>
                </div>
                <div>
                    <div style="color:#FFD700; font-weight:600; margin-bottom:0.5rem;">Experiences & AI</div>
                    <nav style="display:grid; gap:0.3rem;">
                        <a href="zen-unity-meditation.html">Zen Unity Meditation</a>
                        <a href="consciousness_dashboard.html">Consciousness Field</a>
                        <a href="consciousness_dashboard_clean.html">Consciousness Field (Clean)</a>
                        <a href="unity-mathematics-experience.html">Unity Mathematics Experience</a>
                        <a href="ai-unified-hub.html">AI Unity Hub</a>
                        <a href="metagamer_agent.html">Metagamer Agent</a>
                    </nav>
                </div>
                <div>
                    <div style="color:#FFD700; font-weight:600; margin-bottom:0.5rem;">Visualizations & Tools</div>
                    <nav style="display:grid; gap:0.3rem;">
                        <a href="gallery.html">Gallery</a>
                        <a href="dalle-gallery.html">DALL‑E Gallery</a>
                        <a href="enhanced-3d-consciousness-field.html">3D Consciousness</a>
                        <a href="enhanced-unity-visualization-system.html">Visualization System</a>
                        <a href="unity_visualization.html">Unity Visualization</a>
                        <a href="gallery/phi_consciousness_transcendence.html">Phi Consciousness</a>
                        <a href="dashboards.html">Dashboards</a>
                        <a href="playground.html">Playground</a>
                        <a href="mathematical_playground.html">Math Playground</a>
                        <a href="live-code-showcase.html">Live Code Showcase</a>
                        <a href="examples/unity-calculator.html">Unity Calculator</a>
                        <a href="examples/phi-harmonic-explorer.html">φ‑Harmonic Explorer</a>
                    </nav>
                </div>
                <div>
                    <div style="color:#FFD700; font-weight:600; margin-bottom:0.5rem;">About</div>
                    <nav style="display:grid; gap:0.3rem;">
                        <a href="about.html">About Unity</a>
                        <a href="research.html">Research</a>
                        <a href="publications.html">Publications</a>
                        <a href="learning.html">Learning Path</a>
                        <a href="further-reading.html">Further Reading</a>
                        <a href="unity-meta-atlas.html">Unity Meta Atlas</a>
                        <a href="mobile-app.html">Mobile App</a>
                        <a href="sitemap.html">Site Map</a>
                    </nav>
                </div>
            </div>
        `;
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
    }

    setupSearch() {
        const searchInput = document.querySelector('#unified-search-input');
        const searchResults = document.querySelector('#unified-search-results');

        if (!searchInput || !searchResults) return;

        let searchTimeout;

        searchInput.addEventListener('input', (e) => {
            clearTimeout(searchTimeout);
            const query = e.target.value.trim();

            if (query.length < 2) {
                searchResults.style.display = 'none';
                return;
            }

            searchTimeout = setTimeout(() => {
                this.performSearch(query, searchResults);
            }, 300);
        });

        searchInput.addEventListener('focus', () => {
            if (searchInput.value.trim().length >= 2) {
                searchResults.style.display = 'block';
            }
        });

        // Close search results when clicking outside
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.nav-search')) {
                searchResults.style.display = 'none';
            }
        });
    }

    setupMobileHandling() {
        // Handle orientation changes
        window.addEventListener('orientationchange', () => {
            setTimeout(() => {
                this.adjustMobileLayout();
            }, 100);
        });

        // Handle resize
        window.addEventListener('resize', () => {
            this.adjustMobileLayout();
        });
    }

    adjustMobileLayout() {
        const isMobile = window.innerWidth <= 768;
        const sidebar = document.querySelector('.sidebar');
        const mobileOverlay = document.querySelector('.mobile-nav-overlay');

        if (isMobile) {
            if (sidebar) sidebar.classList.remove('open');
            if (mobileOverlay) mobileOverlay.classList.remove('open');
            document.body.classList.remove('sidebar-open');
        }
    }

    markCurrentPage() {
        // Mark active navigation items
        const currentPage = this.currentPage;

        // Mark sidebar links
        document.querySelectorAll('.sidebar-link').forEach(link => {
            const href = link.getAttribute('href');
            if (href === currentPage) {
                link.classList.add('active');
            }
        });

        // Mark mobile navigation links
        document.querySelectorAll('.mobile-nav-link').forEach(link => {
            const href = link.getAttribute('href');
            if (href === currentPage) {
                link.classList.add('active');
            }
        });
    }

    initializeSearchData() {
        return [
            {
                title: 'Unity Mathematics Framework',
                url: 'mathematical-framework.html',
                description: 'Core mathematical proofs and theoretical foundations of 1+1=1',
                keywords: ['mathematics', 'framework', 'proofs', 'theory', 'unity', 'equation']
            },
            {
                title: 'Unity Proof (1+1=1)',
                url: 'unity_proof.html',
                description: 'Formal witnesses across algebra, category theory, logic, topology, quantum operations, and information fusion',
                keywords: ['unity proof', '1+1=1', 'algebra', 'category', 'logic', 'topology', 'quantum']
            },
            {
                title: 'Consciousness Field Dashboard',
                url: 'consciousness_dashboard.html',
                description: 'Real-time visualization of consciousness field dynamics with φ-harmonic resonance',
                keywords: ['consciousness', 'field', 'visualization', 'phi', 'harmonic', 'dashboard']
            },
            {
                title: 'Zen Unity Meditation',
                url: 'zen-unity-meditation.html',
                description: 'Transcendental consciousness meditation with real-time unity field dynamics',
                keywords: ['zen', 'meditation', 'consciousness', 'transcendental', 'mindfulness']
            },
            {
                title: 'AI Unity Hub',
                url: 'ai-unified-hub.html',
                description: 'Advanced AI systems integrated with unity mathematics and GPT-4o',
                keywords: ['ai', 'artificial intelligence', 'gpt-4', 'openai', 'chatbot', 'assistant']
            },
            {
                title: 'Implementations Gallery',
                url: 'implementations-gallery.html',
                description: 'Comprehensive collection of unity mathematics code implementations and algorithms',
                keywords: ['implementations', 'code', 'algorithms', 'programming', 'gallery']
            },
            {
                title: 'DALL-E Consciousness Gallery',
                url: 'dalle-gallery.html',
                description: 'AI-generated artistic representations of unity consciousness and mathematical beauty',
                keywords: ['dalle', 'art', 'gallery', 'ai art', 'consciousness', 'visual']
            },
            {
                title: '3000 ELO Unity Proof',
                url: '3000-elo-proof.html',
                description: 'Master-level mathematical proof demonstrating 1+1=1 through advanced techniques',
                keywords: ['proof', 'advanced', 'master', 'mathematics', 'elo', 'demonstration']
            },
            {
                title: 'Metagambit Framework',
                url: 'metagambit.html',
                description: 'Strategic meta-game theory applied to unity mathematics and consciousness evolution',
                keywords: ['metagambit', 'strategy', 'game theory', 'meta', 'framework']
            },
            {
                title: 'Al-Khwarizmi Unity Algorithm',
                url: 'al_khwarizmi_phi_unity.html',
                description: 'Classical-modern unity bridge connecting ancient algorithms with φ-harmonic mathematics',
                keywords: ['al-khwarizmi', 'algorithm', 'classical', 'phi', 'harmonic', 'bridge']
            },
            {
                title: 'Transcendental Unity Demo',
                url: 'transcendental-unity-demo.html',
                description: '3000 ELO transcendental reality synthesis and higher-dimensional consciousness',
                keywords: ['transcendental', 'demo', 'reality', 'synthesis', 'dimensional']
            },
            {
                title: 'Unity Philosophy',
                url: 'philosophy.html',
                description: 'Deep philosophical exploration of the unity equation and consciousness integration',
                keywords: ['philosophy', 'theory', 'consciousness', 'integration', 'metaphysics']
            },
            {
                title: 'Interactive Playground',
                url: 'playground.html',
                description: 'Interactive sandbox for experimenting with unity equations and consciousness fields',
                keywords: ['playground', 'interactive', 'sandbox', 'experiment', 'tools']
            },
            {
                title: 'Research Papers',
                url: 'research.html',
                description: 'Published academic research on unity mathematics and consciousness integration',
                keywords: ['research', 'papers', 'academic', 'published', 'science']
            },
            {
                title: 'Quantum Ant Colony',
                url: 'anthill.html',
                description: 'Explore collective consciousness through quantum ant systems demonstrating emergent unity',
                keywords: ['quantum', 'ants', 'colony', 'collective', 'consciousness', 'emergent']
            }
        ];
    }

    performSearch(query, resultsContainer) {
        const results = this.searchData.filter(item => {
            const searchString = `${item.title} ${item.description} ${item.keywords.join(' ')}`.toLowerCase();
            return searchString.includes(query.toLowerCase());
        });

        if (results.length > 0) {
            resultsContainer.innerHTML = results.slice(0, 8).map(item => `
                <a href="${item.url}" class="search-result-item">
                    <div class="search-result-title">${this.highlightQuery(item.title, query)}</div>
                    <div class="search-result-description">${this.highlightQuery(item.description, query)}</div>
                </a>
            `).join('');
        } else {
            resultsContainer.innerHTML = `
                <div class="search-no-results">
                    <div class="search-result-title">No results found</div>
                    <div class="search-result-description">Try searching for "consciousness", "mathematics", or "unity"</div>
                </div>
            `;
        }

        resultsContainer.style.display = 'block';

        // Add search result styles
        if (!document.querySelector('#search-results-styles')) {
            const style = document.createElement('style');
            style.id = 'search-results-styles';
            style.textContent = `
                .search-result-item {
                    display: block;
                    padding: 1rem;
                    text-decoration: none;
                    border-bottom: 1px solid var(--border-glow);
                    transition: var(--transition);
                }
                
                .search-result-item:hover {
                    background: rgba(255, 215, 0, 0.1);
                }
                
                .search-result-item:last-child {
                    border-bottom: none;
                }
                
                .search-result-title {
                    color: var(--unity-gold);
                    font-weight: 600;
                    font-size: 0.95rem;
                    margin-bottom: 0.25rem;
                }
                
                .search-result-description {
                    color: var(--text-secondary);
                    font-size: 0.85rem;
                    line-height: 1.4;
                }
                
                .search-no-results {
                    padding: 1rem;
                    text-align: center;
                }
                
                .search-highlight {
                    background: rgba(255, 215, 0, 0.3);
                    padding: 0.1em 0.2em;
                    border-radius: 2px;
                }
                
                .search-results {
                    position: absolute;
                    top: 100%;
                    left: 0;
                    right: 0;
                    background: var(--bg-tertiary);
                    border: 1px solid var(--border-glow);
                    border-radius: 12px;
                    margin-top: 0.5rem;
                    max-height: 400px;
                    overflow-y: auto;
                    z-index: 1003;
                    box-shadow: var(--shadow-dropdown);
                    display: none;
                }
            `;
            document.head.appendChild(style);
        }
    }

    highlightQuery(text, query) {
        const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
        return text.replace(regex, '<span class="search-highlight">$1</span>');
    }

    // Public API methods for integration
    openSidebar() {
        const sidebar = document.querySelector('.sidebar');
        const toggle = document.querySelector('.sidebar-toggle');

        if (sidebar && toggle) {
            sidebar.classList.add('open');
            toggle.classList.add('active');
            document.body.classList.add('sidebar-open');

            const icon = toggle.querySelector('i');
            if (icon) icon.className = 'fas fa-chevron-left';
        }
    }

    closeSidebar() {
        const sidebar = document.querySelector('.sidebar');
        const toggle = document.querySelector('.sidebar-toggle');

        if (sidebar && toggle) {
            sidebar.classList.remove('open');
            toggle.classList.remove('active');
            document.body.classList.remove('sidebar-open');

            const icon = toggle.querySelector('i');
            if (icon) icon.className = 'fas fa-chevron-right';
        }
    }

    focusSearch() {
        const searchInput = document.querySelector('#unified-search-input');
        if (searchInput) {
            searchInput.focus();
        }
    }
}

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.unifiedNavigation = new UnifiedNavigationSystem();
    });
} else {
    window.unifiedNavigation = new UnifiedNavigationSystem();
}

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UnifiedNavigationSystem;
}