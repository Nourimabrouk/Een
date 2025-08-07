/**
 * Unified Navigation System for Een Unity Mathematics
 * Resolves navigation conflicts and creates cohesive user experience
 * Version: 1.0.0 - Meta-Optimized for Desktop Chrome
 */

class UnifiedNavigationSystem {
    constructor() {
        this.currentPage = this.getCurrentPageName();
        this.navigationData = this.getNavigationStructure();
        this.isMobile = window.innerWidth <= 768;
        this.sidebarOpen = false;
        
        // Initialize unified system
        this.init();
        
        // Bind events
        this.bindEvents();
        
        console.log('🚀 Unified Navigation System initialized');
    }
    
    getNavigationStructure() {
        return {
            primary: [
                {
                    id: 'experiences',
                    label: 'Experiences',
                    icon: '⭐',
                    href: 'metastation-hub.html',
                    dropdown: [
                        { label: 'Metastation Hub', href: 'metastation-hub.html', icon: '🚀' },
                        { label: 'Home', href: 'index.html', icon: '🏠' },
                        { label: 'Meta-Optimal Landing', href: 'meta-optimal-landing.html', icon: '✨' },
                        { label: 'Unity Experience', href: 'unity-mathematics-experience.html', icon: '∞' },
                        { label: 'Consciousness Experience', href: 'unity_consciousness_experience.html', icon: '🧠' },
                        { label: 'Zen Meditation', href: 'zen-unity-meditation.html', icon: '🧘' },
                        { label: 'Consciousness Dashboard', href: 'consciousness_dashboard.html', icon: '📊' },
                        { label: 'Clean Dashboard', href: 'consciousness_dashboard_clean.html', icon: '🎯' },
                        { label: 'Transcendental Demo', href: 'transcendental-unity-demo.html', icon: '🌟' },
                        { label: 'Enhanced Unity Demo', href: 'enhanced-unity-demo.html', icon: '⚡' },
                        { label: 'Enhanced AI Demo', href: 'enhanced-ai-demo.html', icon: '🤖' },
                        { label: 'Advanced Features', href: 'unity-advanced-features.html', icon: '🔧' }
                    ]
                },
                {
                    id: 'mathematics',
                    label: 'Mathematics',
                    icon: '📐',
                    href: 'mathematical-framework.html',
                    dropdown: [
                        { label: 'Mathematical Framework', href: 'mathematical-framework.html', icon: '📐' },
                        { label: 'Proofs & Theorems', href: 'proofs.html', icon: '✓' },
                        { label: '3000 ELO Proof', href: '3000-elo-proof.html', icon: '🏆' },
                        { label: 'Al-Khwarizmi Unity', href: 'al_khwarizmi_phi_unity.html', icon: '🕌' },
                        { label: 'Gödel-Tarski', href: 'mathematical_playground.html', icon: '🎯' },
                        { label: 'Interactive Playground', href: 'playground.html', icon: '🎮' },
                        { label: 'Unity Visualization', href: 'unity_visualization.html', icon: '🌐' }
                    ]
                },
                {
                    id: 'philosophy',
                    label: 'Philosophy',
                    icon: '🧠',
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
                        { label: 'Implementations Gallery', href: 'implementations-gallery.html', icon: '⚙️' },
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
                        { label: 'Dashboard Hub', href: 'dashboards.html', icon: '📊' },
                        { label: 'Agents', href: 'agents.html', icon: '🤖' },
                        { label: 'Metagamer Agent', href: 'metagamer_agent.html', icon: '🎮' }
                    ]
                },
                {
                    id: 'tools',
                    label: 'Tools',
                    icon: '🛠️',
                    href: 'playground.html',
                    dropdown: [
                        { label: 'Learning Hub', href: 'learning.html', icon: '📚' },
                        { label: 'Learn', href: 'learn.html', icon: '🎓' },
                        { label: 'Mobile App', href: 'mobile-app.html', icon: '📱' },
                        { label: 'About', href: 'about.html', icon: '👤' },
                        { label: 'Site Map', href: 'sitemap.html', icon: '🗺️' }
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
        
        // Create unified navigation structure
        this.createTopNavigation();
        this.createMobileNavigation();
        this.createSidebarToggle();
        
        // Apply unified styles
        this.applyUnifiedStyles();
        
        // Initialize active states
        this.updateActiveStates();
    }
    
    cleanupConflictingNavigation() {
        // Remove overloaded top navigation
        const existingNav = document.querySelector('.nav-links');
        if (existingNav && existingNav.children.length > 10) {
            console.log('🧹 Cleaning up overloaded navigation (38 → 6 links)');
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
            navContainer.className = 'nav-container unified-nav';
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
            <div class="nav-item ${item.dropdown ? 'has-dropdown' : ''}" data-nav-id="${item.id}">
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
                        <a href="${item.href}" class="dropdown-item ${this.isActive(item.href) ? 'active' : ''}">
                            <span class="dropdown-icon">${item.icon}</span>
                            <span class="dropdown-text">${item.label}</span>
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
                    <h3>Een Unity Mathematics</h3>
                    <button class="mobile-close" aria-label="Close navigation">×</button>
                </div>
                <div class="mobile-nav-body">
                    ${this.renderMobileNavigation()}
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
                    </h4>
                    <div class="mobile-section-items">
                        <a href="${section.href}" class="mobile-nav-item ${this.isActive(section.href) ? 'active' : ''}">
                            ${section.icon} ${section.label} Overview
                        </a>
                        ${section.dropdown ? section.dropdown.map(item => `
                            <a href="${item.href}" class="mobile-nav-item mobile-sub-item ${this.isActive(item.href) ? 'active' : ''}">
                                ${item.icon} ${item.label}
                            </a>
                        `).join('') : ''}
                    </div>
                </div>
            `);
        });
        
        return allItems.join('');
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
    
    applyUnifiedStyles() {
        // Inject unified navigation styles
        const styleId = 'unified-navigation-styles';
        if (!document.getElementById(styleId)) {
            const style = document.createElement('style');
            style.id = styleId;
            style.textContent = this.getUnifiedStyles();
            document.head.appendChild(style);
        }
    }
    
    getUnifiedStyles() {
        return `
            /* Unified Navigation System Styles */
            .unified-nav {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                z-index: 1000;
                background: rgba(10, 10, 15, 0.95);
                backdrop-filter: blur(10px);
                border-bottom: 1px solid rgba(255, 215, 0, 0.2);
                box-shadow: 0 2px 20px rgba(0, 0, 0, 0.3);
            }
            
            .nav-wrapper {
                display: flex;
                align-items: center;
                justify-content: space-between;
                max-width: 1400px;
                margin: 0 auto;
                padding: 0 2rem;
                height: 70px;
            }
            
            /* Brand */
            .nav-brand {
                flex-shrink: 0;
            }
            
            .brand-link {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                color: #FFD700;
                text-decoration: none;
                font-weight: 700;
                font-size: 1.5rem;
                transition: all 0.3s ease;
            }
            
            .brand-link:hover {
                transform: scale(1.05);
                text-shadow: 0 0 20px rgba(255, 215, 0, 0.5);
            }
            
            .brand-icon {
                font-size: 2rem;
                animation: rotate-phi 8s linear infinite;
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
            }
            
            .nav-item {
                position: relative;
            }
            
            .nav-link {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.75rem 1rem;
                color: rgba(255, 255, 255, 0.9);
                text-decoration: none;
                border-radius: 8px;
                transition: all 0.3s ease;
                white-space: nowrap;
                font-weight: 500;
            }
            
            .nav-link:hover,
            .nav-link.active {
                background: rgba(255, 215, 0, 0.1);
                color: #FFD700;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(255, 215, 0, 0.2);
            }
            
            .nav-icon {
                font-size: 1.2rem;
            }
            
            .dropdown-arrow {
                font-size: 0.8rem;
                transition: transform 0.3s ease;
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
                z-index: 1001;
                margin-top: 0.5rem;
            }
            
            .nav-item:hover .nav-dropdown {
                opacity: 1;
                visibility: visible;
                transform: translateX(-50%) translateY(0);
            }
            
            .dropdown-content {
                background: rgba(18, 18, 26, 0.98);
                backdrop-filter: blur(15px);
                border: 1px solid rgba(255, 215, 0, 0.2);
                border-radius: 12px;
                padding: 0.5rem 0;
                min-width: 220px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            }
            
            .dropdown-item {
                display: flex;
                align-items: center;
                gap: 0.75rem;
                padding: 0.75rem 1.25rem;
                color: rgba(255, 255, 255, 0.85);
                text-decoration: none;
                transition: all 0.2s ease;
            }
            
            .dropdown-item:hover,
            .dropdown-item.active {
                background: rgba(255, 215, 0, 0.1);
                color: #FFD700;
            }
            
            .dropdown-icon {
                width: 1.2rem;
                text-align: center;
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
                padding: 0.75rem;
                background: transparent;
                border: 1px solid rgba(255, 215, 0, 0.3);
                border-radius: 8px;
                color: rgba(255, 255, 255, 0.9);
                cursor: pointer;
                transition: all 0.3s ease;
                font-family: inherit;
            }
            
            .utility-btn:hover {
                background: rgba(255, 215, 0, 0.1);
                border-color: #FFD700;
                color: #FFD700;
                transform: translateY(-2px);
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
                background: rgba(0, 0, 0, 0.9);
                z-index: 1100;
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
                height: 100vh;
                overflow-y: auto;
                padding: 1rem;
            }
            
            .mobile-nav-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 1rem 0;
                border-bottom: 1px solid rgba(255, 215, 0, 0.2);
                margin-bottom: 2rem;
            }
            
            .mobile-nav-header h3 {
                color: #FFD700;
                margin: 0;
            }
            
            .mobile-close {
                background: transparent;
                border: none;
                color: #FFD700;
                font-size: 2rem;
                cursor: pointer;
                padding: 0.5rem;
            }
            
            .mobile-nav-section {
                margin-bottom: 2rem;
            }
            
            .mobile-section-title {
                color: #FFD700;
                margin: 0 0 1rem 0;
                padding: 0.5rem 0;
                border-bottom: 1px solid rgba(255, 215, 0, 0.1);
                display: flex;
                align-items: center;
                gap: 0.5rem;
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
                padding: 0.75rem 1rem;
                color: rgba(255, 255, 255, 0.9);
                text-decoration: none;
                border-radius: 8px;
                transition: all 0.2s ease;
            }
            
            .mobile-nav-item:hover,
            .mobile-nav-item.active {
                background: rgba(255, 215, 0, 0.1);
                color: #FFD700;
            }
            
            .mobile-sub-item {
                margin-left: 1rem;
                padding-left: 2rem;
                opacity: 0.8;
            }
            
            /* Responsive Design */
            @media (max-width: 768px) {
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
            
            @media (max-width: 1200px) {
                .nav-primary {
                    gap: 0.25rem;
                }
                
                .nav-link {
                    padding: 0.5rem 0.75rem;
                }
                
                .nav-text {
                    font-size: 0.9rem;
                }
            }
            
            /* Fix for existing page content */
            body {
                margin-left: 0 !important;
                padding-top: 70px;
            }
            
            .container,
            .nav-container:not(.unified-nav) {
                max-width: 1400px;
                margin: 0 auto;
            }
        `;
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
            const utilityBtn = e.target.closest('.utility-btn');
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
            if (!this.isMobile) {
                const mobileNav = document.querySelector('.mobile-nav-overlay');
                if (mobileNav) {
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
        // Search functionality will be implemented by the search system
        window.dispatchEvent(new CustomEvent('unified-nav:search'));
    }
    
    openChat() {
        console.log('💬 Chat functionality triggered');
        // Chat functionality will be implemented by the chat system
        window.dispatchEvent(new CustomEvent('unified-nav:chat'));
    }
    
    toggleAudio() {
        console.log('🎵 Audio functionality triggered');
        // Audio functionality will be implemented by the audio system
        window.dispatchEvent(new CustomEvent('unified-nav:audio'));
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
        const style = document.getElementById('unified-navigation-styles');
        if (style) style.remove();
        
        const mobileNav = document.querySelector('.mobile-nav-overlay');
        if (mobileNav) mobileNav.remove();
        
        document.body.style.paddingTop = '';
        console.log('🧹 Unified Navigation System destroyed');
    }
}

// Initialize on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.unifiedNavigation = new UnifiedNavigationSystem();
    });
} else {
    window.unifiedNavigation = new UnifiedNavigationSystem();
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UnifiedNavigationSystem;
}

console.log('🌟 Unified Navigation System loaded - resolving 38-link overload');