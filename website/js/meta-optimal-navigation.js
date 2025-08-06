/**
 * Meta-Optimal Navigation System - Een Unity Mathematics
 * Comprehensive navigation with all code, philosophy, and implementations
 * Academic and professional presentation with logical categorization
 */

class MetaOptimalNavigation {
    constructor() {
        this.currentPage = window.location.pathname.split('/').pop() || 'index.html';
        this.isMobile = window.innerWidth <= 1024;
        this.isScrolled = false;
        this.mobileMenuOpen = false;
        this.init();
    }

    init() {
        this.createNavigation();
        this.bindEvents();
        this.setActivePage();
        this.initScrollEffects();
        this.initSearch();
        this.initAIChatButton();
    }

    createNavigation() {
        const navHTML = `
            <nav class="meta-optimal-nav" id="metaOptimalNav">
                <div class="nav-container">
                    <a href="index.html" class="nav-logo">
                        <span class="phi-symbol pulse-glow">φ</span>
                        <span class="logo-text">Een</span>
                        <span class="elo-badge">Advanced</span>
                    </a>
                    
                    <ul class="nav-menu">
                        ${this.createMathematicsMenu()}
                        ${this.createConsciousnessMenu()}
                        ${this.createResearchMenu()}
                        ${this.createImplementationsMenu()}
                        ${this.createExperimentsMenu()}
                        ${this.createVisualizationsMenu()}
                        ${this.createAcademyMenu()}
                        ${this.createAboutMenu()}
                    </ul>

                    <div class="nav-search">
                        <input type="text" placeholder="Search Een Unity..." id="navSearch">
                        <button type="button" id="navSearchBtn">
                            <i class="fas fa-search"></i>
                        </button>
                    </div>

                    <button class="ai-chat-trigger" id="aiChatTrigger">
                        <i class="fas fa-robot"></i>
                        AI Chat
                    </button>

                    <div class="mobile-menu-toggle" id="mobileMenuToggle">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
                
                <div class="nav-scroll-indicator" id="scrollIndicator"></div>
            </nav>

            <div class="mobile-nav" id="mobileNav">
                <ul class="mobile-nav-menu">
                    ${this.createMobileMathematicsMenu()}
                    ${this.createMobileConsciousnessMenu()}
                    ${this.createMobileResearchMenu()}
                    ${this.createMobileImplementationsMenu()}
                    ${this.createMobileExperimentsMenu()}
                    ${this.createMobileVisualizationsMenu()}
                    ${this.createMobileAcademyMenu()}
                    ${this.createMobileAboutMenu()}
                </ul>
            </div>
        `;

        document.body.insertAdjacentHTML('afterbegin', navHTML);
    }

    createMathematicsMenu() {
        return `
            <li class="nav-item dropdown">
                <a href="#" class="nav-link dropdown-toggle">
                    <i class="fas fa-calculator"></i>
                    Mathematics
                    <i class="fas fa-chevron-down dropdown-arrow"></i>
                </a>
                <ul class="dropdown-menu">
                    <div class="dropdown-section">
                        <div class="dropdown-section-title">Core Unity Proofs</div>
                        <li><a href="proofs.html" class="dropdown-link">
                            <i class="fas fa-check-circle"></i>
                            Unity Mathematical Proofs
                        </a></li>
                        <li><a href="3000-elo-proof.html" class="dropdown-link">
                            <i class="fas fa-trophy"></i>
                            3000 ELO Advanced Proofs
                        </a></li>
                        <li><a href="mathematical_playground.html" class="dropdown-link">
                            <i class="fas fa-calculator"></i>
                            Mathematical Playground
                        </a></li>
                    </div>
                    
                    <div class="dropdown-section">
                        <div class="dropdown-section-title">Interactive Experiences</div>
                        <li><a href="playground.html" class="dropdown-link">
                            <i class="fas fa-play-circle"></i>
                            Unity Interactive Playground
                        </a></li>
                        <li><a href="enhanced-unity-demo.html" class="dropdown-link">
                            <i class="fas fa-rocket"></i>
                            Enhanced Unity Demo
                        </a></li>
                        <li><a href="unity-mathematics-experience.html" class="dropdown-link">
                            <i class="fas fa-infinity"></i>
                            Unity Mathematics Experience
                        </a></li>
                        <li><a href="unity-advanced-features.html" class="dropdown-link">
                            <i class="fas fa-star"></i>
                            Unity Advanced Features
                        </a></li>
                    </div>
                    
                    <div class="dropdown-section">
                        <div class="dropdown-section-title">Advanced Systems</div>
                        <li><a href="metagambit.html" class="dropdown-link">
                            <i class="fas fa-chess-king"></i>
                            Metagambit Systems
                        </a></li>
                        <li><a href="al_khwarizmi_phi_unity.html" class="dropdown-link">
                            <i class="fas fa-square-root-alt"></i>
                            Al-Khwarizmi Phi Unity
                        </a></li>
                        <li><a href="transcendental-unity-demo.html" class="dropdown-link">
                            <i class="fas fa-eye"></i>
                            Transcendental Unity Demo
                        </a></li>
                    </div>
                </ul>
            </li>
        `;
    }

    createConsciousnessMenu() {
        return `
            <li class="nav-item dropdown">
                <a href="#" class="nav-link dropdown-toggle">
                    <i class="fas fa-brain"></i>
                    Consciousness
                    <i class="fas fa-chevron-down dropdown-arrow"></i>
                </a>
                <ul class="dropdown-menu">
                    <div class="dropdown-section">
                        <div class="dropdown-section-title">Consciousness Fields</div>
                        <li><a href="consciousness_dashboard.html" class="dropdown-link">
                            <i class="fas fa-lightbulb"></i>
                            Consciousness Field Dashboard
                        </a></li>
                        <li><a href="consciousness_dashboard_clean.html" class="dropdown-link">
                            <i class="fas fa-chart-line"></i>
                            Clean Consciousness Dashboard
                        </a></li>
                        <li><a href="unity_consciousness_experience.html" class="dropdown-link">
                            <i class="fas fa-meditation"></i>
                            Unity Consciousness Experience
                        </a></li>
                        <li><a href="unity_visualization.html" class="dropdown-link">
                            <i class="fas fa-wave-square"></i>
                            Unity Visualizations
                        </a></li>
                    </div>
                    
                    <div class="dropdown-section">
                        <div class="dropdown-section-title">Philosophy & Theory</div>
                        <li><a href="philosophy.html" class="dropdown-link">
                            <i class="fas fa-scroll"></i>
                            Unity Philosophy Treatise
                        </a></li>
                        <li><a href="metagamer_agent.html" class="dropdown-link">
                            <i class="fas fa-robot"></i>
                            Metagamer Consciousness Agent
                        </a></li>
                    </div>
                    
                    <div class="dropdown-section">
                        <div class="dropdown-section-title">AI Integration</div>
                        <li><a href="test-chat.html" class="dropdown-link">
                            <i class="fas fa-comments"></i>
                            Unity Chat System
                        </a></li>
                        <li><a href="enhanced-ai-demo.html" class="dropdown-link">
                            <i class="fas fa-brain"></i>
                            Enhanced AI Demonstrations
                        </a></li>
                    </div>
                </ul>
            </li>
        `;
    }

    createResearchMenu() {
        return `
            <li class="nav-item dropdown">
                <a href="#" class="nav-link dropdown-toggle">
                    <i class="fas fa-flask"></i>
                    Research
                    <i class="fas fa-chevron-down dropdown-arrow"></i>
                </a>
                <ul class="dropdown-menu">
                    <div class="dropdown-section">
                        <div class="dropdown-section-title">Current Research</div>
                        <li><a href="research.html" class="dropdown-link">
                            <i class="fas fa-search"></i>
                            Active Research Projects
                        </a></li>
                        <li><a href="publications.html" class="dropdown-link">
                            <i class="fas fa-file-alt"></i>
                            Academic Publications
                        </a></li>
                        <li><a href="further-reading.html" class="dropdown-link">
                            <i class="fas fa-book-open"></i>
                            Further Reading & Resources
                        </a></li>
                    </div>
                    
                    <div class="dropdown-section">
                        <div class="dropdown-section-title">AI Integration</div>
                        <li><a href="openai-integration.html" class="dropdown-link">
                            <i class="fas fa-ai"></i>
                            OpenAI Integration
                        </a></li>
                        <li><a href="live-code-showcase.html" class="dropdown-link">
                            <i class="fas fa-code"></i>
                            Live Code Showcase
                        </a></li>
                    </div>
                    
                    <div class="dropdown-section">
                        <div class="dropdown-section-title">Learning Resources</div>
                        <li><a href="learning.html" class="dropdown-link">
                            <i class="fas fa-graduation-cap"></i>
                            Unity Learning Academy
                        </a></li>
                        <li><a href="learn.html" class="dropdown-link">
                            <i class="fas fa-chalkboard-teacher"></i>
                            Interactive Learning
                        </a></li>
                    </div>
                </ul>
            </li>
        `;
    }

    createImplementationsMenu() {
        return `
            <li class="nav-item dropdown">
                <a href="#" class="nav-link dropdown-toggle">
                    <i class="fas fa-code"></i>
                    Implementations
                    <i class="fas fa-chevron-down dropdown-arrow"></i>
                </a>
                <ul class="dropdown-menu">
                    <div class="dropdown-section">
                        <div class="dropdown-section-title">Core Systems</div>
                        <li><a href="implementations.html" class="dropdown-link">
                            <i class="fas fa-cogs"></i>
                            Core Unity Implementations
                        </a></li>
                        <li><a href="agents.html" class="dropdown-link">
                            <i class="fas fa-users"></i>
                            Unity Agents & Systems
                        </a></li>
                        <li><a href="dashboards.html" class="dropdown-link">
                            <i class="fas fa-tachometer-alt"></i>
                            Unity Dashboards
                        </a></li>
                    </div>
                    
                    <div class="dropdown-section">
                        <div class="dropdown-section-title">Advanced Features</div>
                        <li><a href="unity-advanced-features.html" class="dropdown-link">
                            <i class="fas fa-star"></i>
                            Advanced Unity Features
                        </a></li>
                        <li><a href="mobile-app.html" class="dropdown-link">
                            <i class="fas fa-mobile-alt"></i>
                            Mobile Unity App
                        </a></li>
                        <li><a href="test-chat.html" class="dropdown-link">
                            <i class="fas fa-comments"></i>
                            Unity Chat System
                        </a></li>
                    </div>
                    
                    <div class="dropdown-section">
                        <div class="dropdown-section-title">Testing & Validation</div>
                        <li><a href="test-navigation.html" class="dropdown-link">
                            <i class="fas fa-route"></i>
                            Navigation Testing
                        </a></li>
                        <li><a href="test-website.html" class="dropdown-link">
                            <i class="fas fa-vial"></i>
                            Website Testing
                        </a></li>
                        <li><a href="test-chatbot.html" class="dropdown-link">
                            <i class="fas fa-robot"></i>
                            Chatbot Testing
                        </a></li>
                    </div>
                </ul>
            </li>
        `;
    }

    createExperimentsMenu() {
        return `
            <li class="nav-item dropdown">
                <a href="#" class="nav-link dropdown-toggle">
                    <i class="fas fa-atom"></i>
                    Experiments
                    <i class="fas fa-chevron-down dropdown-arrow"></i>
                </a>
                <ul class="dropdown-menu">
                    <div class="dropdown-section">
                        <div class="dropdown-section-title">Advanced Experiments</div>
                        <li><a href="experiments/advanced/5000_ELO_AGI_Metastation_Metagambit.py" class="dropdown-link">
                            <i class="fas fa-rocket"></i>
                            5000 ELO AGI Metagambit
                        </a></li>
                        <li><a href="experiments/advanced/Godel_Tarski_Metagambit_1v1_God.py" class="dropdown-link">
                            <i class="fas fa-crown"></i>
                            Gödel-Tarski Metagambit
                        </a></li>
                        <li><a href="experiments/advanced/meta_reinforcement_unity_learning.py" class="dropdown-link">
                            <i class="fas fa-brain"></i>
                            Meta-Reinforcement Learning
                        </a></li>
                    </div>
                    
                    <div class="dropdown-section">
                        <div class="dropdown-section-title">Unity Experiments</div>
                        <li><a href="experiments/advanced/Three_Years_Deep_Meta_Meditation_1plus1equals1.py" class="dropdown-link">
                            <i class="fas fa-om"></i>
                            Deep Meta Meditation
                        </a></li>
                        <li><a href="experiments/advanced/Unity_Highscore_Challenge_1plus1equals1.py" class="dropdown-link">
                            <i class="fas fa-trophy"></i>
                            Unity Highscore Challenge
                        </a></li>
                    </div>
                    
                    <div class="dropdown-section">
                        <div class="dropdown-section-title">Research Experiments</div>
                        <li><a href="src/experiments/1plus1equals1_metagambit.py" class="dropdown-link">
                            <i class="fas fa-flask"></i>
                            1+1=1 Metagambit Research
                        </a></li>
                        <li><a href="src/experiments/cloned_policy_paradox.py" class="dropdown-link">
                            <i class="fas fa-copy"></i>
                            Cloned Policy Paradox
                        </a></li>
                    </div>
                </ul>
            </li>
        `;
    }

    createVisualizationsMenu() {
        return `
            <li class="nav-item dropdown">
                <a href="#" class="nav-link dropdown-toggle">
                    <i class="fas fa-chart-bar"></i>
                    Visualizations
                    <i class="fas fa-chevron-down dropdown-arrow"></i>
                </a>
                <ul class="dropdown-menu">
                    <div class="dropdown-section">
                        <div class="dropdown-section-title">Unity Visualizations</div>
                        <li><a href="gallery.html" class="dropdown-link">
                            <i class="fas fa-images"></i>
                            Unity Visualization Gallery
                        </a></li>
                        <li><a href="viz/consciousness_field/" class="dropdown-link">
                            <i class="fas fa-wave-square"></i>
                            Consciousness Field Visualizations
                        </a></li>
                        <li><a href="viz/unity_mathematics/" class="dropdown-link">
                            <i class="fas fa-infinity"></i>
                            Unity Mathematics Visualizations
                        </a></li>
                    </div>
                    
                    <div class="dropdown-section">
                        <div class="dropdown-section-title">Proof Visualizations</div>
                        <li><a href="viz/proofs/" class="dropdown-link">
                            <i class="fas fa-check-circle"></i>
                            Mathematical Proof Visualizations
                        </a></li>
                        <li><a href="scripts/viz/consciousness_field/" class="dropdown-link">
                            <i class="fas fa-brain"></i>
                            Consciousness Field Scripts
                        </a></li>
                        <li><a href="scripts/viz/proofs/" class="dropdown-link">
                            <i class="fas fa-calculator"></i>
                            Proof Visualization Scripts
                        </a></li>
                    </div>
                    
                    <div class="dropdown-section">
                        <div class="dropdown-section-title">Advanced Visualizations</div>
                        <li><a href="visualizations/advanced_unity_visualization.py" class="dropdown-link">
                            <i class="fas fa-star"></i>
                            Advanced Unity Visualizations
                        </a></li>
                        <li><a href="src/visualizations/hyperdimensional_unity_visualizer.py" class="dropdown-link">
                            <i class="fas fa-cube"></i>
                            Hyperdimensional Unity Visualizer
                        </a></li>
                    </div>
                </ul>
            </li>
        `;
    }

    createAcademyMenu() {
        return `
            <li class="nav-item dropdown">
                <a href="#" class="nav-link dropdown-toggle">
                    <i class="fas fa-graduation-cap"></i>
                    Academy
                    <i class="fas fa-chevron-down dropdown-arrow"></i>
                </a>
                <ul class="dropdown-menu">
                    <div class="dropdown-section">
                        <div class="dropdown-section-title">Learning Resources</div>
                        <li><a href="docs/" class="dropdown-link">
                            <i class="fas fa-book"></i>
                            Documentation
                        </a></li>
                        <li><a href="docs/getting-started/" class="dropdown-link">
                            <i class="fas fa-play"></i>
                            Getting Started
                        </a></li>
                        <li><a href="docs/reference/" class="dropdown-link">
                            <i class="fas fa-search"></i>
                            API Reference
                        </a></li>
                    </div>
                    
                    <div class="dropdown-section">
                        <div class="dropdown-section-title">Tutorials</div>
                        <li><a href="examples/" class="dropdown-link">
                            <i class="fas fa-code"></i>
                            Code Examples
                        </a></li>
                        <li><a href="examples/advanced/" class="dropdown-link">
                            <i class="fas fa-rocket"></i>
                            Advanced Examples
                        </a></li>
                        <li><a href="notebooks/" class="dropdown-link">
                            <i class="fas fa-jupyter"></i>
                            Jupyter Notebooks
                        </a></li>
                    </div>
                    
                    <div class="dropdown-section">
                        <div class="dropdown-section-title">Community</div>
                        <li><a href="https://github.com/nourimabrouk/Een" class="dropdown-link" target="_blank">
                            <i class="fab fa-github"></i>
                            GitHub Repository
                        </a></li>
                        <li><a href="docs/AGENT_INSTRUCTIONS.md" class="dropdown-link">
                            <i class="fas fa-robot"></i>
                            Agent Instructions
                        </a></li>
                    </div>
                </ul>
            </li>
        `;
    }

    createAboutMenu() {
        return `
            <li class="nav-item dropdown">
                <a href="#" class="nav-link dropdown-toggle">
                    <i class="fas fa-info-circle"></i>
                    About
                    <i class="fas fa-chevron-down dropdown-arrow"></i>
                </a>
                <ul class="dropdown-menu">
                    <div class="dropdown-section">
                        <div class="dropdown-section-title">Project Information</div>
                        <li><a href="about.html" class="dropdown-link">
                            <i class="fas fa-user"></i>
                            About Een Unity
                        </a></li>
                        <li><a href="about/nourimabrouk.png" class="dropdown-link">
                            <i class="fas fa-image"></i>
                            Team Profile
                        </a></li>
                        <li><a href="README.md" class="dropdown-link">
                            <i class="fas fa-file-alt"></i>
                            Project README
                        </a></li>
                    </div>
                    
                    <div class="dropdown-section">
                        <div class="dropdown-section-title">Project Status</div>
                        <li><a href="docs/LAUNCH_READINESS_REPORT.md" class="dropdown-link">
                            <i class="fas fa-rocket"></i>
                            Launch Readiness
                        </a></li>
                        <li><a href="docs/WEBSITE_OPTIMIZATION_SUMMARY.md" class="dropdown-link">
                            <i class="fas fa-chart-line"></i>
                            Website Optimization
                        </a></li>
                        <li><a href="docs/ENHANCEMENT_PLAN.md" class="dropdown-link">
                            <i class="fas fa-tasks"></i>
                            Enhancement Plan
                        </a></li>
                    </div>
                    
                    <div class="dropdown-section">
                        <div class="dropdown-section-title">Technical</div>
                        <li><a href="docs/SECURITY_AUDIT_REPORT.md" class="dropdown-link">
                            <i class="fas fa-shield-alt"></i>
                            Security Audit
                        </a></li>
                        <li><a href="docs/DEPLOYMENT_CHECKLIST.md" class="dropdown-link">
                            <i class="fas fa-server"></i>
                            Deployment Guide
                        </a></li>
                        <li><a href="sitemap.xml" class="dropdown-link">
                            <i class="fas fa-sitemap"></i>
                            Site Map
                        </a></li>
                    </div>
                </ul>
            </li>
        `;
    }

    // Mobile menu versions (simplified structure)
    createMobileMathematicsMenu() {
        return `
            <li class="mobile-nav-item">
                <a href="#" class="mobile-nav-link" data-mobile-dropdown="mathematics">
                    <i class="fas fa-calculator"></i>
                    Mathematics
                    <i class="fas fa-chevron-right"></i>
                </a>
                <div class="mobile-dropdown" id="mobile-mathematics">
                    <a href="proofs.html" class="mobile-dropdown-link">
                        <i class="fas fa-check-circle"></i>
                        Unity Mathematical Proofs
                    </a>
                    <a href="3000-elo-proof.html" class="mobile-dropdown-link">
                        <i class="fas fa-trophy"></i>
                        3000 ELO Advanced Proofs
                    </a>
                    <a href="playground.html" class="mobile-dropdown-link">
                        <i class="fas fa-play-circle"></i>
                        Unity Interactive Playground
                    </a>
                    <a href="unity-advanced-features.html" class="mobile-dropdown-link">
                        <i class="fas fa-star"></i>
                        Unity Advanced Features
                    </a>
                    <a href="metagambit.html" class="mobile-dropdown-link">
                        <i class="fas fa-chess-king"></i>
                        Metagambit Systems
                    </a>
                </div>
            </li>
        `;
    }

    createMobileConsciousnessMenu() {
        return `
            <li class="mobile-nav-item">
                <a href="#" class="mobile-nav-link" data-mobile-dropdown="consciousness">
                    <i class="fas fa-brain"></i>
                    Consciousness
                    <i class="fas fa-chevron-right"></i>
                </a>
                <div class="mobile-dropdown" id="mobile-consciousness">
                    <a href="consciousness_dashboard.html" class="mobile-dropdown-link">
                        <i class="fas fa-lightbulb"></i>
                        Consciousness Field Dashboard
                    </a>
                    <a href="philosophy.html" class="mobile-dropdown-link">
                        <i class="fas fa-scroll"></i>
                        Unity Philosophy Treatise
                    </a>
                    <a href="unity_visualization.html" class="mobile-dropdown-link">
                        <i class="fas fa-wave-square"></i>
                        Unity Visualizations
                    </a>
                    <a href="test-chat.html" class="mobile-dropdown-link">
                        <i class="fas fa-comments"></i>
                        Unity Chat System
                    </a>
                </div>
            </li>
        `;
    }

    createMobileResearchMenu() {
        return `
            <li class="mobile-nav-item">
                <a href="#" class="mobile-nav-link" data-mobile-dropdown="research">
                    <i class="fas fa-flask"></i>
                    Research
                    <i class="fas fa-chevron-right"></i>
                </a>
                <div class="mobile-dropdown" id="mobile-research">
                    <a href="research.html" class="mobile-dropdown-link">
                        <i class="fas fa-search"></i>
                        Active Research Projects
                    </a>
                    <a href="publications.html" class="mobile-dropdown-link">
                        <i class="fas fa-file-alt"></i>
                        Academic Publications
                    </a>
                    <a href="learning.html" class="mobile-dropdown-link">
                        <i class="fas fa-graduation-cap"></i>
                        Unity Learning Academy
                    </a>
                </div>
            </li>
        `;
    }

    createMobileImplementationsMenu() {
        return `
            <li class="mobile-nav-item">
                <a href="#" class="mobile-nav-link" data-mobile-dropdown="implementations">
                    <i class="fas fa-code"></i>
                    Implementations
                    <i class="fas fa-chevron-right"></i>
                </a>
                <div class="mobile-dropdown" id="mobile-implementations">
                    <a href="implementations.html" class="mobile-dropdown-link">
                        <i class="fas fa-cogs"></i>
                        Core Unity Implementations
                    </a>
                    <a href="agents.html" class="mobile-dropdown-link">
                        <i class="fas fa-users"></i>
                        Unity Agents & Systems
                    </a>
                    <a href="dashboards.html" class="mobile-dropdown-link">
                        <i class="fas fa-tachometer-alt"></i>
                        Unity Dashboards
                    </a>
                    <a href="unity-advanced-features.html" class="mobile-dropdown-link">
                        <i class="fas fa-star"></i>
                        Advanced Unity Features
                    </a>
                </div>
            </li>
        `;
    }

    createMobileExperimentsMenu() {
        return `
            <li class="mobile-nav-item">
                <a href="#" class="mobile-nav-link" data-mobile-dropdown="experiments">
                    <i class="fas fa-atom"></i>
                    Experiments
                    <i class="fas fa-chevron-right"></i>
                </a>
                <div class="mobile-dropdown" id="mobile-experiments">
                    <a href="experiments/advanced/5000_ELO_AGI_Metastation_Metagambit.py" class="mobile-dropdown-link">
                        <i class="fas fa-rocket"></i>
                        5000 ELO AGI Metagambit
                    </a>
                    <a href="experiments/advanced/Godel_Tarski_Metagambit_1v1_God.py" class="mobile-dropdown-link">
                        <i class="fas fa-crown"></i>
                        Gödel-Tarski Metagambit
                    </a>
                    <a href="src/experiments/1plus1equals1_metagambit.py" class="mobile-dropdown-link">
                        <i class="fas fa-flask"></i>
                        1+1=1 Metagambit Research
                    </a>
                </div>
            </li>
        `;
    }

    createMobileVisualizationsMenu() {
        return `
            <li class="mobile-nav-item">
                <a href="#" class="mobile-nav-link" data-mobile-dropdown="visualizations">
                    <i class="fas fa-chart-bar"></i>
                    Visualizations
                    <i class="fas fa-chevron-right"></i>
                </a>
                <div class="mobile-dropdown" id="mobile-visualizations">
                    <a href="gallery.html" class="mobile-dropdown-link">
                        <i class="fas fa-images"></i>
                        Unity Visualization Gallery
                    </a>
                    <a href="viz/consciousness_field/" class="mobile-dropdown-link">
                        <i class="fas fa-wave-square"></i>
                        Consciousness Field Visualizations
                    </a>
                    <a href="viz/unity_mathematics/" class="mobile-dropdown-link">
                        <i class="fas fa-infinity"></i>
                        Unity Mathematics Visualizations
                    </a>
                </div>
            </li>
        `;
    }

    createMobileAcademyMenu() {
        return `
            <li class="mobile-nav-item">
                <a href="#" class="mobile-nav-link" data-mobile-dropdown="academy">
                    <i class="fas fa-graduation-cap"></i>
                    Academy
                    <i class="fas fa-chevron-right"></i>
                </a>
                <div class="mobile-dropdown" id="mobile-academy">
                    <a href="docs/" class="mobile-dropdown-link">
                        <i class="fas fa-book"></i>
                        Documentation
                    </a>
                    <a href="examples/" class="mobile-dropdown-link">
                        <i class="fas fa-code"></i>
                        Code Examples
                    </a>
                    <a href="notebooks/" class="mobile-dropdown-link">
                        <i class="fas fa-jupyter"></i>
                        Jupyter Notebooks
                    </a>
                </div>
            </li>
        `;
    }

    createMobileAboutMenu() {
        return `
            <li class="mobile-nav-item">
                <a href="#" class="mobile-nav-link" data-mobile-dropdown="about">
                    <i class="fas fa-info-circle"></i>
                    About
                    <i class="fas fa-chevron-right"></i>
                </a>
                <div class="mobile-dropdown" id="mobile-about">
                    <a href="about.html" class="mobile-dropdown-link">
                        <i class="fas fa-user"></i>
                        About Een Unity
                    </a>
                    <a href="docs/LAUNCH_READINESS_REPORT.md" class="mobile-dropdown-link">
                        <i class="fas fa-rocket"></i>
                        Launch Readiness
                    </a>
                    <a href="README.md" class="mobile-dropdown-link">
                        <i class="fas fa-file-alt"></i>
                        Project README
                    </a>
                </div>
            </li>
        `;
    }

    bindEvents() {
        // Mobile menu toggle
        const mobileToggle = document.getElementById('mobileMenuToggle');
        const mobileNav = document.getElementById('mobileNav');

        if (mobileToggle && mobileNav) {
            mobileToggle.addEventListener('click', () => {
                this.mobileMenuOpen = !this.mobileMenuOpen;
                mobileToggle.classList.toggle('active');
                mobileNav.classList.toggle('active');
            });
        }

        // Mobile dropdown toggles
        document.querySelectorAll('[data-mobile-dropdown]').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const dropdownId = link.getAttribute('data-mobile-dropdown');
                const dropdown = document.getElementById(`mobile-${dropdownId}`);

                if (dropdown) {
                    dropdown.classList.toggle('active');
                    const icon = link.querySelector('.fa-chevron-right');
                    if (icon) {
                        icon.style.transform = dropdown.classList.contains('active') ? 'rotate(90deg)' : 'rotate(0deg)';
                    }
                }
            });
        });

        // Close mobile menu when clicking outside
        document.addEventListener('click', (e) => {
            if (this.mobileMenuOpen && !mobileNav.contains(e.target) && !mobileToggle.contains(e.target)) {
                this.closeMobileMenu();
            }
        });

        // Handle window resize
        window.addEventListener('resize', () => {
            const wasMobile = this.isMobile;
            this.isMobile = window.innerWidth <= 1024;

            if (wasMobile !== this.isMobile && this.mobileMenuOpen) {
                this.closeMobileMenu();
            }
        });
    }

    closeMobileMenu() {
        const mobileToggle = document.getElementById('mobileMenuToggle');
        const mobileNav = document.getElementById('mobileNav');

        if (mobileToggle && mobileNav) {
            mobileToggle.classList.remove('active');
            mobileNav.classList.remove('active');
            this.mobileMenuOpen = false;
        }
    }

    setActivePage() {
        const currentPath = window.location.pathname;
        const navLinks = document.querySelectorAll('.nav-link, .mobile-nav-link');

        navLinks.forEach(link => {
            const href = link.getAttribute('href');
            if (href && (currentPath.includes(href) || (currentPath === '/' && href === 'index.html'))) {
                link.classList.add('active');
            }
        });
    }

    initScrollEffects() {
        const nav = document.getElementById('metaOptimalNav');
        const scrollIndicator = document.getElementById('scrollIndicator');

        window.addEventListener('scroll', () => {
            const scrollTop = window.pageYOffset;
            const docHeight = document.documentElement.scrollHeight - window.innerHeight;
            const scrollPercent = (scrollTop / docHeight) * 100;

            // Add scrolled class for styling
            if (scrollTop > 50) {
                nav.classList.add('scrolled');
                this.isScrolled = true;
            } else {
                nav.classList.remove('scrolled');
                this.isScrolled = false;
            }

            // Update scroll indicator
            if (scrollIndicator) {
                scrollIndicator.style.width = `${scrollPercent}%`;
            }
        });
    }

    initSearch() {
        const searchInput = document.getElementById('navSearch');
        const searchBtn = document.getElementById('navSearchBtn');

        if (searchInput && searchBtn) {
            const performSearch = () => {
                const query = searchInput.value.trim();
                if (query) {
                    // Implement search functionality
                    this.searchSite(query);
                }
            };

            searchBtn.addEventListener('click', performSearch);
            searchInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    performSearch();
                }
            });
        }
    }

    initAIChatButton() {
        const aiChatTrigger = document.getElementById('aiChatTrigger');

        if (aiChatTrigger) {
            aiChatTrigger.addEventListener('click', () => {
                this.openAIChat();
            });
        }
    }

    openAIChat() {
        // Try to open the AI chat system
        if (window.enhancedEenChat) {
            window.enhancedEenChat.open();
        } else if (window.eenChat) {
            window.eenChat.open();
        } else if (window.floatingChatButton) {
            window.floatingChatButton.handleClick();
        } else {
            // Fallback: redirect to chat page
            window.location.href = 'test-chat.html';
        }
    }

    searchSite(query) {
        // Simple search implementation - can be enhanced with more sophisticated search
        const searchResults = [
            { title: 'Unity Mathematical Proofs', url: 'proofs.html', category: 'Mathematics' },
            { title: 'Consciousness Field Dashboard', url: 'consciousness_dashboard.html', category: 'Consciousness' },
            { title: 'Unity Interactive Playground', url: 'playground.html', category: 'Mathematics' },
            { title: 'Research Projects', url: 'research.html', category: 'Research' },
            { title: 'Unity Philosophy Treatise', url: 'philosophy.html', category: 'Consciousness' },
            { title: 'Core Implementations', url: 'implementations.html', category: 'Implementations' },
            { title: 'Unity Learning Academy', url: 'learning.html', category: 'Academy' },
            { title: 'Visualization Gallery', url: 'gallery.html', category: 'Visualizations' },
            { title: 'Unity Advanced Features', url: 'unity-advanced-features.html', category: 'Mathematics' },
            { title: 'AI Chat System', url: 'test-chat.html', category: 'AI' },
            { title: 'Metagambit Systems', url: 'metagambit.html', category: 'Mathematics' },
            { title: 'Enhanced AI Demo', url: 'enhanced-ai-demo.html', category: 'AI' }
        ];

        const filteredResults = searchResults.filter(result =>
            result.title.toLowerCase().includes(query.toLowerCase()) ||
            result.category.toLowerCase().includes(query.toLowerCase())
        );

        if (filteredResults.length > 0) {
            // Show search results (could be enhanced with a modal or dropdown)
            const firstResult = filteredResults[0];
            window.location.href = firstResult.url;
        } else {
            // Show no results message
            alert(`No results found for "${query}". Try searching for: mathematics, consciousness, research, implementations, experiments, visualizations, academy, or about.`);
        }
    }
}

// Initialize navigation when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new MetaOptimalNavigation();
});

// Export for use in other scripts
window.MetaOptimalNavigation = MetaOptimalNavigation; 