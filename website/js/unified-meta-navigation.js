/**
 * Unified Meta-Optimal Navigation System
 * Universal navigation component for all Een Unity Mathematics pages
 * Provides consistent, accessible navigation across the entire website
 */

class UnifiedMetaNavigation {
    constructor() {
        this.currentPage = this.detectCurrentPage();
        this.navStructure = this.getNavigationStructure();
        this.isInitialized = false;
    }

    detectCurrentPage() {
        const path = window.location.pathname;
        const filename = path.split('/').pop() || 'index.html';
        return filename.replace('.html', '');
    }

    getNavigationStructure() {
        return {
            main: [
                { id: 'home', label: 'Metastation Hub', href: 'metastation-hub.html', icon: 'fas fa-home' },
                { id: 'philosophy', label: 'Philosophy', href: 'philosophy.html', icon: 'fas fa-brain' },
                { id: 'gallery', label: 'Gallery', href: 'gallery.html', icon: 'fas fa-images' },
                { id: 'proofs', label: 'Mathematical Proofs', href: 'proofs.html', icon: 'fas fa-calculator' },
                { id: 'metagambit', label: 'Metagambit', href: 'metagambit.html', icon: 'fas fa-chess' }
            ],
            experiences: [
                { id: 'unity-experience', label: 'Unity Experience', href: 'unity-mathematics-experience.html', icon: 'fas fa-atom' },
                { id: 'consciousness', label: 'Consciousness Dashboard', href: 'consciousness_dashboard.html', icon: 'fas fa-brain' },
                { id: 'zen-meditation', label: 'Zen Unity Meditation', href: 'zen-unity-meditation.html', icon: 'fas fa-om' },
                { id: 'transcendental-demo', label: 'Transcendental Demo', href: 'transcendental-unity-demo.html', icon: 'fas fa-infinity' }
            ],
            tools: [
                { id: 'playground', label: 'Unity Playground', href: 'playground.html', icon: 'fas fa-play-circle' },
                { id: 'dashboards', label: 'Interactive Dashboards', href: 'dashboards.html', icon: 'fas fa-chart-line' },
                { id: 'implementations', label: 'Code Implementations', href: 'implementations.html', icon: 'fas fa-code' },
                { id: 'agents', label: 'Unity Agents', href: 'agents.html', icon: 'fas fa-robot' }
            ],
            academic: [
                { id: 'research', label: 'Research Papers', href: 'research.html', icon: 'fas fa-graduation-cap' },
                { id: 'publications', label: 'Publications', href: 'publications.html', icon: 'fas fa-book' },
                { id: 'learning', label: 'Learning Resources', href: 'learning.html', icon: 'fas fa-user-graduate' },
                { id: 'about', label: 'About Een', href: 'about.html', icon: 'fas fa-info-circle' }
            ]
        };
    }

    async initialize() {
        if (this.isInitialized) return;

        this.createNavigationHTML();
        this.attachEventListeners();
        this.highlightCurrentPage();
        this.setupMobileNavigation();
        this.initializeKeyboardShortcuts();
        
        this.isInitialized = true;
        console.log('🗺️ Unified Meta Navigation initialized successfully');
    }

    createNavigationHTML() {
        // Create navigation container
        const navContainer = document.createElement('nav');
        navContainer.className = 'unified-meta-nav';
        navContainer.innerHTML = this.generateNavigationHTML();

        // Insert navigation at the top of the body
        const existingNav = document.querySelector('.unified-meta-nav, .orbital-hud');
        if (existingNav) {
            existingNav.replaceWith(navContainer);
        } else {
            document.body.insertBefore(navContainer, document.body.firstChild);
        }

        this.injectNavigationCSS();
    }

    generateNavigationHTML() {
        return `
            <div class="nav-container">
                <div class="nav-brand">
                    <a href="metastation-hub.html" class="brand-link">
                        <span class="brand-icon">∞</span>
                        <span class="brand-text">Een Unity Mathematics</span>
                    </a>
                </div>

                <button class="mobile-nav-toggle" aria-label="Toggle navigation">
                    <span class="hamburger-line"></span>
                    <span class="hamburger-line"></span>
                    <span class="hamburger-line"></span>
                </button>

                <div class="nav-menu">
                    ${this.generateNavSection('Main', this.navStructure.main)}
                    ${this.generateNavSection('Experiences', this.navStructure.experiences)}
                    ${this.generateNavSection('Tools', this.navStructure.tools)}
                    ${this.generateNavSection('Academic', this.navStructure.academic)}
                    
                    <div class="nav-actions">
                        <button class="nav-search-btn" title="Search (Ctrl+K)">
                            <i class="fas fa-search"></i>
                        </button>
                        <button class="nav-chat-btn" title="AI Chat (Ctrl+U)">
                            <i class="fas fa-robot"></i>
                        </button>
                        <button class="nav-theme-btn" title="Toggle Theme">
                            <i class="fas fa-adjust"></i>
                        </button>
                    </div>
                </div>

                <div class="nav-unity-indicator">
                    <span class="unity-equation">1+1=1</span>
                    <span class="phi-value">φ = 1.618...</span>
                </div>
            </div>

            <!-- Mobile Navigation Overlay -->
            <div class="mobile-nav-overlay">
                <div class="mobile-nav-content">
                    <div class="mobile-nav-header">
                        <h2>Navigation</h2>
                        <button class="mobile-nav-close">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                    ${this.generateMobileNavigation()}
                </div>
            </div>
        `;
    }

    generateNavSection(title, items) {
        return `
            <div class="nav-section">
                <span class="nav-section-title">${title}</span>
                <div class="nav-section-items">
                    ${items.map(item => `
                        <a href="${item.href}" 
                           class="nav-link ${this.currentPage === item.id ? 'active' : ''}"
                           data-page="${item.id}"
                           title="${item.label}">
                            <i class="${item.icon}"></i>
                            <span class="nav-link-text">${item.label}</span>
                        </a>
                    `).join('')}
                </div>
            </div>
        `;
    }

    generateMobileNavigation() {
        const allItems = [
            ...this.navStructure.main,
            ...this.navStructure.experiences,
            ...this.navStructure.tools,
            ...this.navStructure.academic
        ];

        return `
            <div class="mobile-nav-grid">
                ${allItems.map(item => `
                    <a href="${item.href}" 
                       class="mobile-nav-item ${this.currentPage === item.id ? 'active' : ''}"
                       data-page="${item.id}">
                        <i class="${item.icon}"></i>
                        <span>${item.label}</span>
                    </a>
                `).join('')}
            </div>
        `;
    }

    injectNavigationCSS() {
        if (document.getElementById('unified-nav-styles')) return;

        const style = document.createElement('style');
        style.id = 'unified-nav-styles';
        style.textContent = `
            :root {
                --nav-height: 70px;
                --nav-bg: rgba(10, 10, 15, 0.95);
                --nav-border: rgba(255, 215, 0, 0.2);
                --nav-text: #ffffff;
                --nav-text-hover: #FFD700;
                --nav-accent: #FFD700;
                --nav-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
                --nav-transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }

            body {
                padding-top: var(--nav-height);
            }

            .unified-meta-nav {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                height: var(--nav-height);
                background: var(--nav-bg);
                backdrop-filter: blur(20px);
                border-bottom: 1px solid var(--nav-border);
                box-shadow: var(--nav-shadow);
                z-index: 1000;
            }

            .nav-container {
                max-width: 1400px;
                margin: 0 auto;
                height: 100%;
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 0 1rem;
            }

            .nav-brand {
                display: flex;
                align-items: center;
            }

            .brand-link {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                text-decoration: none;
                color: var(--nav-text);
                font-weight: 700;
                font-size: 1.25rem;
                transition: var(--nav-transition);
            }

            .brand-link:hover {
                color: var(--nav-accent);
                transform: scale(1.05);
            }

            .brand-icon {
                font-size: 1.5rem;
                color: var(--nav-accent);
            }

            .nav-menu {
                display: flex;
                align-items: center;
                gap: 2rem;
                flex: 1;
                justify-content: center;
            }

            .nav-section {
                position: relative;
                display: flex;
                flex-direction: column;
                align-items: center;
            }

            .nav-section-title {
                font-size: 0.7rem;
                color: var(--nav-text);
                opacity: 0.6;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 0.25rem;
                font-weight: 500;
            }

            .nav-section-items {
                display: flex;
                gap: 0.5rem;
            }

            .nav-link {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 0.25rem;
                padding: 0.5rem;
                text-decoration: none;
                color: var(--nav-text);
                border-radius: 8px;
                transition: var(--nav-transition);
                position: relative;
                min-width: 60px;
            }

            .nav-link:hover {
                color: var(--nav-accent);
                background: rgba(255, 215, 0, 0.1);
                transform: translateY(-2px);
            }

            .nav-link.active {
                color: var(--nav-accent);
                background: rgba(255, 215, 0, 0.15);
                box-shadow: 0 0 15px rgba(255, 215, 0, 0.3);
            }

            .nav-link i {
                font-size: 1.1rem;
            }

            .nav-link-text {
                font-size: 0.65rem;
                text-align: center;
                line-height: 1.2;
                font-weight: 500;
            }

            .nav-actions {
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }

            .nav-actions button {
                width: 40px;
                height: 40px;
                border: none;
                background: rgba(255, 255, 255, 0.1);
                color: var(--nav-text);
                border-radius: 50%;
                cursor: pointer;
                transition: var(--nav-transition);
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .nav-actions button:hover {
                background: rgba(255, 215, 0, 0.2);
                color: var(--nav-accent);
                transform: scale(1.1);
            }

            .nav-unity-indicator {
                display: flex;
                flex-direction: column;
                align-items: center;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.7rem;
                color: var(--nav-accent);
                opacity: 0.8;
            }

            .mobile-nav-toggle {
                display: none;
                flex-direction: column;
                justify-content: center;
                width: 30px;
                height: 30px;
                background: none;
                border: none;
                cursor: pointer;
            }

            .hamburger-line {
                width: 20px;
                height: 2px;
                background: var(--nav-text);
                margin: 2px 0;
                transition: var(--nav-transition);
            }

            .mobile-nav-overlay {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0, 0, 0, 0.9);
                z-index: 2000;
                display: none;
                opacity: 0;
                transition: var(--nav-transition);
            }

            .mobile-nav-overlay.active {
                display: flex;
                opacity: 1;
            }

            .mobile-nav-content {
                width: 100%;
                max-width: 400px;
                margin: auto;
                background: var(--nav-bg);
                border-radius: 16px;
                padding: 2rem;
                max-height: 80vh;
                overflow-y: auto;
            }

            .mobile-nav-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 2rem;
                color: var(--nav-text);
            }

            .mobile-nav-close {
                background: none;
                border: none;
                color: var(--nav-text);
                font-size: 1.5rem;
                cursor: pointer;
            }

            .mobile-nav-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 1rem;
            }

            .mobile-nav-item {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 0.5rem;
                padding: 1rem;
                text-decoration: none;
                color: var(--nav-text);
                background: rgba(255, 255, 255, 0.05);
                border-radius: 12px;
                transition: var(--nav-transition);
            }

            .mobile-nav-item:hover,
            .mobile-nav-item.active {
                background: rgba(255, 215, 0, 0.15);
                color: var(--nav-accent);
            }

            .mobile-nav-item i {
                font-size: 1.5rem;
            }

            .mobile-nav-item span {
                font-size: 0.8rem;
                text-align: center;
            }

            @media (max-width: 1024px) {
                .nav-menu {
                    display: none;
                }
                
                .mobile-nav-toggle {
                    display: flex;
                }
                
                .nav-unity-indicator {
                    display: none;
                }
            }

            @media (max-width: 768px) {
                :root {
                    --nav-height: 60px;
                }
                
                .nav-container {
                    padding: 0 1rem;
                }
                
                .brand-link {
                    font-size: 1rem;
                }
                
                .brand-text {
                    display: none;
                }
            }
        `;
        
        document.head.appendChild(style);
    }

    attachEventListeners() {
        // Mobile navigation toggle
        const mobileToggle = document.querySelector('.mobile-nav-toggle');
        const mobileOverlay = document.querySelector('.mobile-nav-overlay');
        const mobileClose = document.querySelector('.mobile-nav-close');

        if (mobileToggle && mobileOverlay) {
            mobileToggle.addEventListener('click', () => {
                mobileOverlay.classList.add('active');
                document.body.style.overflow = 'hidden';
            });

            const closeMobile = () => {
                mobileOverlay.classList.remove('active');
                document.body.style.overflow = 'auto';
            };

            mobileClose?.addEventListener('click', closeMobile);
            mobileOverlay.addEventListener('click', (e) => {
                if (e.target === mobileOverlay) closeMobile();
            });
        }

        // Navigation actions
        const searchBtn = document.querySelector('.nav-search-btn');
        const chatBtn = document.querySelector('.nav-chat-btn');
        const themeBtn = document.querySelector('.nav-theme-btn');

        searchBtn?.addEventListener('click', () => this.toggleSearch());
        chatBtn?.addEventListener('click', () => this.toggleChat());
        themeBtn?.addEventListener('click', () => this.toggleTheme());
    }

    highlightCurrentPage() {
        const navLinks = document.querySelectorAll('.nav-link, .mobile-nav-item');
        navLinks.forEach(link => {
            const pageId = link.getAttribute('data-page');
            if (pageId === this.currentPage) {
                link.classList.add('active');
            }
        });
    }

    setupMobileNavigation() {
        // Enhance mobile navigation with swipe gestures
        let startX = null;
        let startY = null;

        document.addEventListener('touchstart', (e) => {
            startX = e.touches[0].clientX;
            startY = e.touches[0].clientY;
        });

        document.addEventListener('touchmove', (e) => {
            if (!startX || !startY) return;

            const currentX = e.touches[0].clientX;
            const currentY = e.touches[0].clientY;
            const diffX = startX - currentX;
            const diffY = startY - currentY;

            if (Math.abs(diffX) > Math.abs(diffY) && Math.abs(diffX) > 100) {
                if (diffX > 0) {
                    // Swipe left - could trigger navigation
                } else {
                    // Swipe right - could trigger navigation
                }
            }

            startX = null;
            startY = null;
        });
    }

    initializeKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch (e.key) {
                    case 'k':
                        e.preventDefault();
                        this.toggleSearch();
                        break;
                    case 'u':
                        e.preventDefault();
                        this.toggleChat();
                        break;
                    case 'h':
                        e.preventDefault();
                        window.location.href = 'metastation-hub.html';
                        break;
                    case 'p':
                        e.preventDefault();
                        window.location.href = 'philosophy.html';
                        break;
                    case 'g':
                        e.preventDefault();
                        window.location.href = 'gallery.html';
                        break;
                }
            }
        });
    }

    toggleSearch() {
        console.log('🔍 Search functionality - to be implemented');
        // TODO: Implement search modal
    }

    toggleChat() {
        // Try to find existing chat functionality
        if (typeof sendChatMessage === 'function') {
            const chatSection = document.getElementById('ai-chat');
            if (chatSection) {
                chatSection.scrollIntoView({ behavior: 'smooth' });
                const chatInput = document.getElementById('chat-input-field');
                if (chatInput) {
                    chatInput.focus();
                }
            }
        } else if (window.unityAI) {
            window.unityAI.toggleChat();
        } else {
            console.log('🤖 Chat functionality - to be implemented');
        }
    }

    toggleTheme() {
        document.body.classList.toggle('light-theme');
        const isDark = !document.body.classList.contains('light-theme');
        localStorage.setItem('theme-preference', isDark ? 'dark' : 'light');
        console.log(`🎨 Theme switched to ${isDark ? 'dark' : 'light'} mode`);
    }

    // Public API methods
    navigateTo(pageId) {
        const allItems = [
            ...this.navStructure.main,
            ...this.navStructure.experiences,
            ...this.navStructure.tools,
            ...this.navStructure.academic
        ];
        
        const page = allItems.find(item => item.id === pageId);
        if (page) {
            window.location.href = page.href;
        }
    }

    updateActiveState(pageId) {
        this.currentPage = pageId;
        this.highlightCurrentPage();
    }
}

// Initialize the unified navigation system
const unifiedMetaNavigation = new UnifiedMetaNavigation();

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        unifiedMetaNavigation.initialize();
    });
} else {
    unifiedMetaNavigation.initialize();
}

// Global access
window.UnifiedMetaNavigation = UnifiedMetaNavigation;
window.unifiedMetaNavigation = unifiedMetaNavigation;

console.log('🗺️ Unified Meta Navigation system loaded and ready');