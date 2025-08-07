/**
 * Meta-Optimal Unified Navigation System for Een Unity Mathematics
 * Comprehensive navigation covering all 42 HTML pages with perfect desktop/mobile experience
 * Version: 2.0.0 - Meta-Optimized for Unity Mathematics
 */

class MetaOptimalNavigationSystem {
    constructor() {
        this.currentPage = this.getCurrentPageName();
        this.navigationData = this.getCompleteNavigationStructure();
        this.isMobile = window.innerWidth <= 768;
        this.isTablet = window.innerWidth <= 1024 && window.innerWidth > 768;
        this.sidebarOpen = false;
        
        // Initialize meta-optimal system
        this.init();
        
        // Bind events
        this.bindEvents();
        
        console.log('🚀 Meta-Optimal Navigation System initialized - 42 pages accessible');
    }
    
    getCompleteNavigationStructure() {
        return {
            primary: [
                {
                    id: 'experiences',
                    label: 'Experiences',
                    icon: '⭐',
                    href: 'metastation-hub.html',
                    featured: true,
                    dropdown: [
                        { label: 'Metastation Hub', href: 'metastation-hub.html', icon: '🚀', featured: true },
                        { label: 'Home', href: 'index.html', icon: '🏠' },
                        { label: 'Meta-Optimal Landing', href: 'meta-optimal-landing.html', icon: '✨' },
                        { label: 'Enhanced Unity Landing', href: 'enhanced-unity-landing.html', icon: '🌟' },
                        { label: 'Unity Experience', href: 'unity-mathematics-experience.html', icon: '∞' },
                        { label: 'Consciousness Experience', href: 'unity_consciousness_experience.html', icon: '🧠' },
                        { label: 'Zen Meditation', href: 'zen-unity-meditation.html', icon: '🧘', featured: true },
                        { label: 'Transcendental Demo', href: 'transcendental-unity-demo.html', icon: '🌟' },
                        { label: 'Enhanced Unity Demo', href: 'enhanced-unity-demo.html', icon: '⚡' },
                        { label: 'Enhanced AI Demo', href: 'enhanced-ai-demo.html', icon: '🤖' },
                        { label: 'Advanced Features', href: 'unity-advanced-features.html', icon: '🔧' },
                        { label: 'Anthill Experience', href: 'anthill.html', icon: '🐜' }
                    ]
                },
                {
                    id: 'mathematics',
                    label: 'Mathematics',
                    icon: '📐',
                    href: 'mathematical-framework.html',
                    featured: true,
                    dropdown: [
                        { label: 'Mathematical Framework', href: 'mathematical-framework.html', icon: '📐', featured: true },
                        { label: 'Proofs & Theorems', href: 'proofs.html', icon: '✓' },
                        { label: '3000 ELO Proof', href: '3000-elo-proof.html', icon: '🏆' },
                        { label: 'Enhanced Mathematical Proofs', href: 'enhanced-mathematical-proofs.html', icon: '🔬' },
                        { label: 'Al-Khwarizmi Unity', href: 'al_khwarizmi_phi_unity.html', icon: '🕌' },
                        { label: 'Gödel-Tarski Playground', href: 'mathematical_playground.html', icon: '🎯' },
                        { label: 'Interactive Playground', href: 'playground.html', icon: '🎮' },
                        { label: 'Unity Visualization', href: 'unity_visualization.html', icon: '🌐' },
                        { label: 'Enhanced Unity Visualization', href: 'enhanced-unity-visualization-system.html', icon: '🎨' },
                        { label: 'Enhanced 3D Consciousness Field', href: 'enhanced-3d-consciousness-field.html', icon: '🧠' }
                    ]
                },
                {
                    id: 'consciousness',
                    label: 'Consciousness',
                    icon: '🧠',
                    href: 'consciousness_dashboard.html',
                    dropdown: [
                        { label: 'Consciousness Dashboard', href: 'consciousness_dashboard.html', icon: '📊', featured: true },
                        { label: 'Clean Dashboard', href: 'consciousness_dashboard_clean.html', icon: '🎯' },
                        { label: 'Metagamer Agent', href: 'metagamer_agent.html', icon: '🎮' },
                        { label: 'Unity Agents', href: 'agents.html', icon: '🤖' }
                    ]
                },
                {
                    id: 'philosophy',
                    label: 'Philosophy',
                    icon: '📜',
                    href: 'philosophy.html',
                    dropdown: [
                        { label: 'Unity Philosophy', href: 'philosophy.html', icon: '📜' },
                        { label: 'Metagambit Theory', href: 'metagambit.html', icon: '♟️' },
                        { label: 'AI Integration', href: 'openai-integration.html', icon: '🤖' },
                        { label: 'Further Reading', href: 'further-reading.html', icon: '📚' }
                    ]
                },
                {
                    id: 'gallery',
                    label: 'Gallery',
                    icon: '🎨',
                    href: 'gallery.html',
                    dropdown: [
                        { label: 'Implementations Gallery', href: 'implementations-gallery.html', icon: '⚙️', featured: true },
                        { label: 'Implementations', href: 'implementations.html', icon: '🔬' },
                        { label: 'Visual Gallery', href: 'gallery.html', icon: '🖼️' },
                        { label: 'Gallery Test', href: 'gallery_test.html', icon: '🧪' },
                        { label: 'Live Code Showcase', href: 'live-code-showcase.html', icon: '💻' }
                    ]
                },
                {
                    id: 'research',
                    label: 'Research',
                    icon: '📊',
                    href: 'research.html',
                    dropdown: [
                        { label: 'Research Overview', href: 'research.html', icon: '🔬' },
                        { label: 'Publications', href: 'publications.html', icon: '📄' },
                        { label: 'Dashboard Hub', href: 'dashboards.html', icon: '📊' }
                    ]
                },
                {
                    id: 'learning',
                    label: 'Learning',
                    icon: '🎓',
                    href: 'learn.html',
                    dropdown: [
                        { label: 'Learn Unity Math', href: 'learn.html', icon: '📚' },
                        { label: 'Learning Resources', href: 'learning.html', icon: '🎓' },
                        { label: 'Mobile App', href: 'mobile-app.html', icon: '📱' }
                    ]
                },
                {
                    id: 'tools',
                    label: 'Tools',
                    icon: '🛠️',
                    href: 'sitemap.html',
                    dropdown: [
                        { label: 'Complete Site Map', href: 'sitemap.html', icon: '🗺️' },
                        { label: 'About', href: 'about.html', icon: '👤' },
                        { label: 'Navigation Test', href: 'navigation-test.html', icon: '🧪' },
                        { label: 'Test Search', href: 'test-search.html', icon: '🔍' },
                        { label: 'Test Fixes', href: 'test-fixes.html', icon: '🔧' },
                        { label: 'Redirect', href: 'redirect.html', icon: '↪️' }
                    ]
                }
            ],
            utilities: [
                { id: 'search', label: 'Search', icon: '🔍', action: 'openSearch' },
                { id: 'chat', label: 'AI Assistant', icon: '💬', action: 'openChat' },
                { id: 'audio', label: 'Music', icon: '🎵', action: 'toggleAudio' }
            ]
        };
    }
    
    getCurrentPageName() {
        const path = window.location.pathname;
        const page = path.split('/').pop() || 'index.html';
        return page.replace('.html', '');
    }
    
    init() {
        // Remove existing conflicting navigation systems
        this.cleanupConflictingNavigation();
        
        // Create meta-optimal navigation structure
        this.createTopNavigation();
        this.createMobileNavigation();
        this.createSidebarToggle();
        
        // Apply meta-optimal styles
        this.applyMetaOptimalStyles();
        
        // Initialize active states
        this.updateActiveStates();
        
        // Add scroll effects
        this.addScrollEffects();
    }
    
    cleanupConflictingNavigation() {
        // Remove overloaded top navigation
        const existingNav = document.querySelector('.nav-links');
        if (existingNav && existingNav.children.length > 10) {
            console.log('🧹 Cleaning up overloaded navigation (42 pages → organized structure)');
            existingNav.innerHTML = '';
        }
        
        // Remove conflicting margin styles
        document.body.style.marginLeft = '';
        
        // Clean up duplicate event listeners
        const oldButtons = document.querySelectorAll('.nav-toggle, .sidebar-toggle');
        oldButtons.forEach(btn => btn.replaceWith(btn.cloneNode(true)));
    }
    
    createTopNavigation() {
        let navContainer = document.querySelector('.nav-container');
        if (!navContainer) {
            // Create navigation container if it doesn't exist
            const header = document.querySelector('header') || document.body;
            navContainer = document.createElement('nav');
            navContainer.className = 'nav-container meta-optimal-nav';
            header.appendChild(navContainer);
        }
        
        // Clear existing content
        navContainer.innerHTML = '';
        
        // Create navigation HTML
        navContainer.innerHTML = `
            <div class="nav-wrapper">
                <!-- Logo/Brand -->
                <div class="nav-brand">
                    <a href="metastation-hub.html" class="brand-link">
                        <span class="brand-icon">∞</span>
                        <span class="brand-text">Een</span>
                        <span class="brand-subtitle">Unity Mathematics</span>
                    </a>
                </div>
                
                <!-- Primary Navigation -->
                <div class="nav-primary">
                    ${this.renderPrimaryNavigation()}
                </div>
                
                <!-- Utility Navigation -->
                <div class="nav-utilities">
                    ${this.renderUtilityNavigation()}
                </div>
                
                <!-- Mobile Toggle -->
                <button class="mobile-toggle" aria-label="Toggle navigation">
                    <span class="toggle-line"></span>
                    <span class="toggle-line"></span>
                    <span class="toggle-line"></span>
                </button>
            </div>
        `;
    }
    
    renderPrimaryNavigation() {
        return this.navigationData.primary.map(item => `
            <div class="nav-item ${item.dropdown ? 'has-dropdown' : ''} ${item.featured ? 'featured' : ''}" data-nav-id="${item.id}">
                <a href="${item.href}" class="nav-link ${this.isActive(item.href) ? 'active' : ''}">
                    <span class="nav-icon">${item.icon}</span>
                    <span class="nav-text">${item.label}</span>
                    ${item.dropdown ? '<span class="dropdown-arrow">▾</span>' : ''}
                </a>
                ${item.dropdown ? this.renderDropdown(item.dropdown) : ''}
            </div>
        `).join('');
    }
    
    renderDropdown(items) {
        return `
            <div class="nav-dropdown">
                <div class="dropdown-content">
                    ${items.map(item => `
                        <a href="${item.href}" class="dropdown-item ${this.isActive(item.href) ? 'active' : ''} ${item.featured ? 'featured' : ''}">
                            <span class="dropdown-icon">${item.icon}</span>
                            <span class="dropdown-text">${item.label}</span>
                            ${item.featured ? '<span class="featured-badge">Featured</span>' : ''}
                        </a>
                    `).join('')}
                </div>
            </div>
        `;
    }
    
    renderUtilityNavigation() {
        return this.navigationData.utilities.map(item => `
            <button class="utility-btn" data-action="${item.action}" title="${item.label}">
                <span class="utility-icon">${item.icon}</span>
                <span class="utility-text">${item.label}</span>
            </button>
        `).join('');
    }
    
    createMobileNavigation() {
        // Create mobile overlay navigation
        const mobileNav = document.createElement('div');
        mobileNav.className = 'mobile-nav-overlay';
        mobileNav.innerHTML = `
            <div class="mobile-nav-content">
                <div class="mobile-nav-header">
                    <div class="mobile-brand">
                        <span class="mobile-brand-icon">∞</span>
                        <h3>Een Unity Mathematics</h3>
                    </div>
                    <button class="mobile-close" aria-label="Close navigation">×</button>
                </div>
                <div class="mobile-nav-body">
                    ${this.renderMobileNavigation()}
                </div>
                <div class="mobile-nav-footer">
                    <div class="mobile-utilities">
                        ${this.renderMobileUtilities()}
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(mobileNav);
    }
    
    renderMobileNavigation() {
        const allItems = [];
        
        // Add primary navigation items with their dropdowns
        this.navigationData.primary.forEach(section => {
            allItems.push(`
                <div class="mobile-nav-section">
                    <h4 class="mobile-section-title">
                        <span class="mobile-section-icon">${section.icon}</span>
                        ${section.label}
                        ${section.featured ? '<span class="mobile-featured-badge">Featured</span>' : ''}
                    </h4>
                    <div class="mobile-section-items">
                        <a href="${section.href}" class="mobile-nav-item ${this.isActive(section.href) ? 'active' : ''} ${section.featured ? 'featured' : ''}">
                            ${section.icon} ${section.label} Overview
                        </a>
                        ${section.dropdown ? section.dropdown.map(item => `
                            <a href="${item.href}" class="mobile-nav-item mobile-sub-item ${this.isActive(item.href) ? 'active' : ''} ${item.featured ? 'featured' : ''}">
                                ${item.icon} ${item.label}
                                ${item.featured ? '<span class="mobile-featured-badge">Featured</span>' : ''}
                            </a>
                        `).join('') : ''}
                    </div>
                </div>
            `);
        });
        
        return allItems.join('');
    }
    
    renderMobileUtilities() {
        return this.navigationData.utilities.map(item => `
            <button class="mobile-utility-btn" data-action="${item.action}">
                <span class="mobile-utility-icon">${item.icon}</span>
                <span class="mobile-utility-text">${item.label}</span>
            </button>
        `).join('');
    }
    
    createSidebarToggle() {
        // This replaces the conflicting sidebar system with a simple toggle
        // that works with the unified navigation
        const existingSidebar = document.querySelector('.metastation-sidebar');
        if (existingSidebar) {
            // Hide existing sidebar to prevent conflicts
            existingSidebar.style.display = 'none';
            console.log('🔧 Existing sidebar hidden to prevent conflicts');
        }
    }
    
    applyMetaOptimalStyles() {
        // Inject meta-optimal navigation styles
        const styleId = 'meta-optimal-navigation-styles';
        if (!document.getElementById(styleId)) {
            const style = document.createElement('style');
            style.id = styleId;
            style.textContent = this.getMetaOptimalStyles();
            document.head.appendChild(style);
        }
    }
    
    getMetaOptimalStyles() {
        return `
            /* Meta-Optimal Navigation System Styles */
            .meta-optimal-nav {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                z-index: 10000;
                background: rgba(10, 10, 15, 0.98);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                border-bottom: 1px solid rgba(255, 215, 0, 0.3);
                box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                font-weight: 500;
                letter-spacing: 0.025em;
            }
            
            .meta-optimal-nav.scrolled {
                background: rgba(8, 8, 12, 0.99);
                box-shadow: 0 8px 40px rgba(0, 0, 0, 0.6);
            }
            
            .nav-wrapper {
                display: flex;
                align-items: center;
                justify-content: space-between;
                max-width: 1600px;
                margin: 0 auto;
                padding: 0 2rem;
                height: 80px;
                min-width: 0;
                flex-wrap: nowrap;
            }
            
            /* Brand */
            .nav-brand {
                flex-shrink: 0;
                min-width: 0;
            }
            
            .brand-link {
                display: flex;
                align-items: center;
                gap: 0.75rem;
                color: #FFD700;
                text-decoration: none;
                font-weight: 700;
                font-size: 1.5rem;
                transition: all 0.3s ease;
                white-space: nowrap;
            }
            
            .brand-link:hover {
                transform: scale(1.05);
                text-shadow: 0 0 20px rgba(255, 215, 0, 0.5);
            }
            
            .brand-icon {
                font-size: 2rem;
                animation: rotate-phi 8s linear infinite;
            }
            
            .brand-text {
                background: linear-gradient(135deg, #FFD700, #D4AF37);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            
            .brand-subtitle {
                font-size: 0.7rem;
                color: rgba(255, 255, 255, 0.7);
                font-weight: 400;
                margin-left: 0.5rem;
            }
            
            @keyframes rotate-phi {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            /* Primary Navigation */
            .nav-primary {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                flex: 1;
                justify-content: center;
                min-width: 0;
            }
            
            .nav-item {
                position: relative;
                flex-shrink: 0;
            }
            
            .nav-item.featured .nav-link {
                background: rgba(255, 215, 0, 0.1);
                border: 1px solid rgba(255, 215, 0, 0.3);
            }
            
            .nav-link {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.875rem 1.25rem;
                color: rgba(255, 255, 255, 0.9);
                text-decoration: none;
                border-radius: 12px;
                transition: all 0.3s ease;
                white-space: nowrap;
                font-weight: 500;
                font-size: 0.95rem;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            
            .nav-link:hover,
            .nav-link.active {
                background: rgba(255, 215, 0, 0.15);
                color: #FFD700;
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(255, 215, 0, 0.2);
            }
            
            .nav-icon {
                font-size: 1.2rem;
                flex-shrink: 0;
            }
            
            .dropdown-arrow {
                font-size: 0.8rem;
                transition: transform 0.3s ease;
                flex-shrink: 0;
            }
            
            .nav-item:hover .dropdown-arrow {
                transform: rotate(180deg);
            }
            
            /* Dropdowns */
            .nav-dropdown {
                position: absolute;
                top: 100%;
                left: 50%;
                transform: translateX(-50%);
                opacity: 0;
                visibility: hidden;
                transition: all 0.3s ease;
                z-index: 10001;
                margin-top: 0.5rem;
            }
            
            .nav-item:hover .nav-dropdown {
                opacity: 1;
                visibility: visible;
                transform: translateX(-50%) translateY(0);
            }
            
            .dropdown-content {
                background: rgba(15, 15, 20, 0.98);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 215, 0, 0.2);
                border-radius: 16px;
                padding: 0.75rem 0;
                min-width: 280px;
                box-shadow: 0 15px 50px rgba(0, 0, 0, 0.4);
                max-width: calc(100vw - 2rem);
                overflow-x: hidden;
            }
            
            .dropdown-item {
                display: flex;
                align-items: center;
                gap: 0.75rem;
                padding: 0.875rem 1.5rem;
                color: rgba(255, 255, 255, 0.85);
                text-decoration: none;
                transition: all 0.2s ease;
                position: relative;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
            
            .dropdown-item:hover,
            .dropdown-item.active {
                background: rgba(255, 215, 0, 0.1);
                color: #FFD700;
                transform: translateX(5px);
            }
            
            .dropdown-item.featured {
                background: rgba(255, 215, 0, 0.05);
                border-left: 3px solid #FFD700;
            }
            
            .dropdown-icon {
                width: 1.2rem;
                text-align: center;
                flex-shrink: 0;
            }
            
            .featured-badge {
                background: linear-gradient(135deg, #FFD700, #D4AF37);
                color: #000;
                font-size: 0.7rem;
                padding: 0.2rem 0.5rem;
                border-radius: 8px;
                font-weight: 600;
                margin-left: auto;
                flex-shrink: 0;
            }
            
            /* Utility Navigation */
            .nav-utilities {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                flex-shrink: 0;
            }
            
            .utility-btn {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.75rem 1rem;
                background: transparent;
                border: 1px solid rgba(255, 215, 0, 0.3);
                border-radius: 12px;
                color: rgba(255, 255, 255, 0.9);
                cursor: pointer;
                transition: all 0.3s ease;
                font-family: inherit;
                font-size: 0.9rem;
            }
            
            .utility-btn:hover {
                background: rgba(255, 215, 0, 0.1);
                border-color: #FFD700;
                color: #FFD700;
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(255, 215, 0, 0.2);
            }
            
            .utility-text {
                font-size: 0.9rem;
            }
            
            /* Mobile Toggle */
            .mobile-toggle {
                display: none;
                flex-direction: column;
                gap: 3px;
                background: transparent;
                border: none;
                padding: 0.5rem;
                cursor: pointer;
                border-radius: 8px;
                transition: all 0.3s ease;
            }
            
            .mobile-toggle:hover {
                background: rgba(255, 215, 0, 0.1);
            }
            
            .toggle-line {
                width: 24px;
                height: 3px;
                background: #FFD700;
                border-radius: 2px;
                transition: all 0.3s ease;
            }
            
            /* Mobile Navigation */
            .mobile-nav-overlay {
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0, 0, 0, 0.95);
                z-index: 11000;
                opacity: 0;
                transition: opacity 0.3s ease;
            }
            
            .mobile-nav-overlay.active {
                display: block;
                opacity: 1;
            }
            
            .mobile-nav-content {
                background: rgba(10, 10, 15, 0.98);
                backdrop-filter: blur(20px);
                -webkit-backdrop-filter: blur(20px);
                height: 100vh;
                overflow-y: auto;
                padding: 0;
                display: flex;
                flex-direction: column;
            }
            
            .mobile-nav-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 1.5rem;
                border-bottom: 1px solid rgba(255, 215, 0, 0.2);
                background: rgba(255, 215, 0, 0.05);
            }
            
            .mobile-brand {
                display: flex;
                align-items: center;
                gap: 0.75rem;
            }
            
            .mobile-brand-icon {
                font-size: 2rem;
                color: #FFD700;
                animation: rotate-phi 8s linear infinite;
            }
            
            .mobile-nav-header h3 {
                color: #FFD700;
                margin: 0;
                font-size: 1.25rem;
            }
            
            .mobile-close {
                background: transparent;
                border: none;
                color: #FFD700;
                font-size: 2rem;
                cursor: pointer;
                padding: 0.5rem;
                border-radius: 8px;
                transition: all 0.3s ease;
            }
            
            .mobile-close:hover {
                background: rgba(255, 215, 0, 0.1);
            }
            
            .mobile-nav-body {
                flex: 1;
                padding: 1.5rem;
                overflow-y: auto;
            }
            
            .mobile-nav-section {
                margin-bottom: 2rem;
            }
            
            .mobile-section-title {
                color: #FFD700;
                margin: 0 0 1rem 0;
                padding: 0.75rem 0;
                border-bottom: 1px solid rgba(255, 215, 0, 0.1);
                display: flex;
                align-items: center;
                gap: 0.75rem;
                font-size: 1.1rem;
                font-weight: 600;
            }
            
            .mobile-featured-badge {
                background: linear-gradient(135deg, #FFD700, #D4AF37);
                color: #000;
                font-size: 0.7rem;
                padding: 0.2rem 0.5rem;
                border-radius: 8px;
                font-weight: 600;
                margin-left: auto;
            }
            
            .mobile-section-items {
                display: flex;
                flex-direction: column;
                gap: 0.25rem;
            }
            
            .mobile-nav-item {
                display: flex;
                align-items: center;
                gap: 0.75rem;
                padding: 1rem;
                color: rgba(255, 255, 255, 0.9);
                text-decoration: none;
                border-radius: 12px;
                transition: all 0.2s ease;
                border: 1px solid transparent;
                position: relative;
            }
            
            .mobile-nav-item:hover,
            .mobile-nav-item.active {
                background: rgba(255, 215, 0, 0.1);
                color: #FFD700;
                border-color: rgba(255, 215, 0, 0.3);
            }
            
            .mobile-nav-item.featured {
                background: rgba(255, 215, 0, 0.05);
                border-color: rgba(255, 215, 0, 0.2);
            }
            
            .mobile-sub-item {
                margin-left: 1rem;
                padding-left: 2rem;
                opacity: 0.8;
            }
            
            .mobile-nav-footer {
                padding: 1.5rem;
                border-top: 1px solid rgba(255, 215, 0, 0.2);
                background: rgba(255, 215, 0, 0.05);
            }
            
            .mobile-utilities {
                display: flex;
                gap: 0.5rem;
                justify-content: center;
            }
            
            .mobile-utility-btn {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 0.5rem;
                padding: 1rem;
                background: transparent;
                border: 1px solid rgba(255, 215, 0, 0.3);
                border-radius: 12px;
                color: rgba(255, 255, 255, 0.9);
                cursor: pointer;
                transition: all 0.3s ease;
                font-family: inherit;
                flex: 1;
            }
            
            .mobile-utility-btn:hover {
                background: rgba(255, 215, 0, 0.1);
                border-color: #FFD700;
                color: #FFD700;
            }
            
            .mobile-utility-icon {
                font-size: 1.5rem;
            }
            
            .mobile-utility-text {
                font-size: 0.8rem;
            }
            
            /* Responsive Design */
            @media (max-width: 1400px) {
                .nav-wrapper {
                    padding: 0 1.5rem;
                }
                
                .nav-link {
                    padding: 0.75rem 1rem;
                    font-size: 0.9rem;
                }
                
                .dropdown-content {
                    min-width: 250px;
                }
            }
            
            @media (max-width: 1200px) {
                .nav-wrapper {
                    padding: 0 1rem;
                }
                
                .nav-link {
                    padding: 0.75rem 0.75rem;
                    font-size: 0.85rem;
                }
                
                .nav-text {
                    font-size: 0.85rem;
                }
                
                .brand-subtitle {
                    display: none;
                }
            }
            
            @media (max-width: 1024px) {
                .nav-primary,
                .nav-utilities .utility-text {
                    display: none;
                }
                
                .mobile-toggle {
                    display: flex;
                }
                
                .nav-wrapper {
                    padding: 0 1rem;
                }
                
                .utility-btn {
                    padding: 0.5rem;
                }
            }
            
            @media (max-width: 768px) {
                .nav-wrapper {
                    height: 70px;
                    padding: 0 0.75rem;
                }
                
                .nav-brand {
                    font-size: 1.25rem;
                    gap: 0.5rem;
                }
                
                .brand-icon {
                    font-size: 1.75rem;
                }
                
                .mobile-nav {
                    top: 70px;
                }
            }
            
            @media (max-width: 480px) {
                .nav-wrapper {
                    padding: 0 0.5rem;
                }
                
                .nav-brand {
                    font-size: 1.1rem;
                    gap: 0.4rem;
                }
                
                .brand-icon {
                    font-size: 1.5rem;
                }
                
                .brand-text {
                    display: none;
                }
            }
            
            /* Fix for existing page content */
            body {
                margin-left: 0 !important;
                padding-top: 80px;
            }
            
            .container,
            .nav-container:not(.meta-optimal-nav) {
                max-width: 1600px;
                margin: 0 auto;
            }
            
            /* Chrome OS specific optimizations */
            @media screen and (max-width: 1200px) and (min-resolution: 1.5dppx) {
                .nav-link {
                    font-size: 0.8rem;
                    padding: 0.75rem 0.75rem;
                }
                
                .nav-brand {
                    font-size: 1.2rem;
                }
                
                .brand-icon {
                    font-size: 1.6rem;
                }
            }
            
            /* High Contrast Mode Support */
            @media (prefers-contrast: high) {
                .meta-optimal-nav {
                    background: #000000;
                    border-bottom: 2px solid var(--nav-accent-gold);
                }
                
                .dropdown-content {
                    background: #000000;
                    border: 2px solid var(--nav-accent-gold);
                }
            }
            
            /* Reduced Motion Support */
            @media (prefers-reduced-motion: reduce) {
                .meta-optimal-nav,
                .dropdown-content,
                .nav-link,
                .dropdown-item,
                .mobile-nav-item {
                    transition: none;
                }
                
                .brand-icon {
                    animation: none;
                }
            }
        `;
    }
    
    addScrollEffects() {
        let lastScrollTop = 0;
        
        window.addEventListener('scroll', () => {
            const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
            const nav = document.querySelector('.meta-optimal-nav');
            
            if (nav) {
                if (scrollTop > 50) {
                    nav.classList.add('scrolled');
                } else {
                    nav.classList.remove('scrolled');
                }
                
                // Hide/show navigation on scroll
                if (scrollTop > lastScrollTop && scrollTop > 100) {
                    nav.style.transform = 'translateY(-100%)';
                } else {
                    nav.style.transform = 'translateY(0)';
                }
                
                lastScrollTop = scrollTop;
            }
        });
    }
    
    bindEvents() {
        // Mobile navigation toggle
        const mobileToggle = document.querySelector('.mobile-toggle');
        const mobileNav = document.querySelector('.mobile-nav-overlay');
        const mobileClose = document.querySelector('.mobile-close');
        
        if (mobileToggle && mobileNav) {
            mobileToggle.addEventListener('click', () => {
                mobileNav.classList.add('active');
                document.body.style.overflow = 'hidden';
            });
            
            mobileClose?.addEventListener('click', () => {
                mobileNav.classList.remove('active');
                document.body.style.overflow = '';
            });
            
            // Close on backdrop click
            mobileNav.addEventListener('click', (e) => {
                if (e.target === mobileNav) {
                    mobileNav.classList.remove('active');
                    document.body.style.overflow = '';
                }
            });
        }
        
        // Utility button actions
        document.addEventListener('click', (e) => {
            const utilityBtn = e.target.closest('.utility-btn, .mobile-utility-btn');
            if (utilityBtn) {
                const action = utilityBtn.dataset.action;
                this.handleUtilityAction(action);
            }
        });
        
        // Close mobile nav on link click
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('mobile-nav-item')) {
                const mobileNav = document.querySelector('.mobile-nav-overlay');
                if (mobileNav) {
                    mobileNav.classList.remove('active');
                    document.body.style.overflow = '';
                }
            }
        });
        
        // Handle window resize
        window.addEventListener('resize', () => {
            this.isMobile = window.innerWidth <= 768;
            this.isTablet = window.innerWidth <= 1024 && window.innerWidth > 768;
            
            if (!this.isMobile) {
                const mobileNav = document.querySelector('.mobile-nav-overlay');
                if (mobileNav) {
                    mobileNav.classList.remove('active');
                    document.body.style.overflow = '';
                }
            }
        });
        
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                const mobileNav = document.querySelector('.mobile-nav-overlay');
                if (mobileNav && mobileNav.classList.contains('active')) {
                    mobileNav.classList.remove('active');
                    document.body.style.overflow = '';
                }
            }
        });
    }
    
    handleUtilityAction(action) {
        switch (action) {
            case 'openSearch':
                this.openSearch();
                break;
            case 'openChat':
                this.openChat();
                break;
            case 'toggleAudio':
                this.toggleAudio();
                break;
        }
    }
    
    openSearch() {
        console.log('🔍 Search functionality triggered');
        window.dispatchEvent(new CustomEvent('meta-optimal-nav:search'));
    }
    
    openChat() {
        console.log('💬 Chat functionality triggered');
        window.dispatchEvent(new CustomEvent('meta-optimal-nav:chat'));
    }
    
    toggleAudio() {
        console.log('🎵 Audio functionality triggered');
        window.dispatchEvent(new CustomEvent('meta-optimal-nav:audio'));
    }
    
    isActive(href) {
        if (!href) return false;
        const currentPage = this.currentPage;
        const linkPage = href.replace('.html', '').split('/').pop();
        
        // Check for exact match or index page
        return linkPage === currentPage || 
               (currentPage === 'index' && linkPage === 'metastation-hub') ||
               (currentPage === 'metastation-hub' && linkPage === 'index');
    }
    
    updateActiveStates() {
        // Update active states for all navigation links
        const allLinks = document.querySelectorAll('.nav-link, .dropdown-item, .mobile-nav-item');
        allLinks.forEach(link => {
            const href = link.getAttribute('href');
            if (this.isActive(href)) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });
    }
    
    // Public API
    refresh() {
        this.currentPage = this.getCurrentPageName();
        this.updateActiveStates();
    }
    
    destroy() {
        // Remove event listeners and clean up
        const style = document.getElementById('meta-optimal-navigation-styles');
        if (style) style.remove();
        
        const mobileNav = document.querySelector('.mobile-nav-overlay');
        if (mobileNav) mobileNav.remove();
        
        document.body.style.paddingTop = '';
        console.log('🧹 Meta-Optimal Navigation System destroyed');
    }
}

// Initialize on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.metaOptimalNavigation = new MetaOptimalNavigationSystem();
    });
} else {
    window.metaOptimalNavigation = new MetaOptimalNavigationSystem();
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MetaOptimalNavigationSystem;
}

console.log('🌟 Meta-Optimal Navigation System loaded - 42 pages accessible with perfect desktop/mobile experience');