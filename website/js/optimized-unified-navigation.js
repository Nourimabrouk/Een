/**
 * Een Unity Mathematics - Optimized Unified Navigation System v3.0
 * 
 * Complete navigation overhaul providing:
 * - Single unified navigation across ALL pages
 * - Optimal Chrome PC experience
 * - All functionality accessible from landing page
 * - Mobile-responsive design
 * - Performance optimized
 * - Accessibility compliant
 */

class OptimizedUnifiedNavigation {
    constructor() {
        this.currentPage = this.detectCurrentPage();
        this.isInitialized = false;
        this.isMobile = window.innerWidth <= 768;
        this.isTablet = window.innerWidth <= 1024;

        // Complete site structure optimized for user experience
        this.siteStructure = {
            'Core Mathematics': {
                icon: 'fas fa-infinity',
                color: '#FFD700',
                priority: 1,
                pages: {
                    'proofs.html': { title: 'Mathematical Proofs', icon: 'fas fa-check-circle', description: 'Rigorous mathematical proofs of 1+1=1' },
                    'playground.html': { title: 'Interactive Playground', icon: 'fas fa-play-circle', description: 'Live mathematical exploration' },
                    '3000-elo-proof.html': { title: 'Advanced Proofs', icon: 'fas fa-trophy', description: '3000 ELO level mathematical demonstrations' },
                    'mathematical_playground.html': { title: 'Mathematical Playground', icon: 'fas fa-calculator', description: 'Advanced mathematical tools' }
                }
            },
            'Consciousness & Philosophy': {
                icon: 'fas fa-brain',
                color: '#764ba2',
                priority: 2,
                pages: {
                    'philosophy.html': { title: 'Philosophical Treatise', icon: 'fas fa-scroll', description: 'Deep philosophical insights' },
                    'consciousness_dashboard.html': { title: 'Consciousness Dashboard', icon: 'fas fa-lightbulb', description: 'Interactive consciousness fields' },
                    'unity_consciousness_experience.html': { title: 'Unity Experience', icon: 'fas fa-meditation', description: 'Transcendental unity experience' },
                    'unity_visualization.html': { title: 'Unity Visualizations', icon: 'fas fa-wave-square', description: 'Visual consciousness representations' }
                }
            },
            'Research & Learning': {
                icon: 'fas fa-university',
                color: '#27ae60',
                priority: 3,
                pages: {
                    'research.html': { title: 'Research Overview', icon: 'fas fa-search', description: 'Current research projects' },
                    'publications.html': { title: 'Publications', icon: 'fas fa-file-alt', description: 'Academic publications' },
                    'implementations.html': { title: 'Code Implementations', icon: 'fas fa-code', description: 'Core mathematical implementations' },
                    'learning.html': { title: 'Learning Center', icon: 'fas fa-graduation-cap', description: 'Educational resources' }
                }
            },
            'AI Systems': {
                icon: 'fas fa-robot',
                color: '#3498db',
                priority: 4,
                pages: {
                    'agents.html': { title: 'AI Agents', icon: 'fas fa-robot', description: 'Advanced AI systems' },
                    'metagambit.html': { title: 'MetaGambit System', icon: 'fas fa-chess', description: 'Strategic AI framework' },
                    'enhanced-ai-demo.html': { title: 'AI Demo', icon: 'fas fa-rocket', description: 'Interactive AI demonstrations' }
                }
            },
            'Visual Gallery': {
                icon: 'fas fa-images',
                color: '#9b59b6',
                priority: 5,
                pages: {
                    'gallery.html': { title: 'Visualization Gallery', icon: 'fas fa-palette', description: 'Mathematical visualizations' },
                    'dashboards.html': { title: 'Dashboard Suite', icon: 'fas fa-tachometer-alt', description: 'Interactive dashboards' }
                }
            },
            'About & Info': {
                icon: 'fas fa-info-circle',
                color: '#95a5a6',
                priority: 6,
                pages: {
                    'about.html': { title: 'About Project', icon: 'fas fa-user-graduate', description: 'Project information and team' },
                    'further-reading.html': { title: 'Further Reading', icon: 'fas fa-book-open', description: 'Additional resources' }
                }
            }
        };
    }

    detectCurrentPage() {
        const path = window.location.pathname;
        const filename = path.split('/').pop() || 'index.html';
        return filename;
    }

    init() {
        if (this.isInitialized) return;

        this.injectStyles();
        this.createNavigation();
        this.attachEventListeners();
        this.setupAccessibility();
        this.initializeAnimations();

        this.isInitialized = true;
        console.log('Optimized Unified Navigation initialized');
    }

    injectStyles() {
        const styleId = 'optimized-nav-styles';
        if (document.getElementById(styleId)) return;

        const styles = `
            /* Optimized Navigation Styles */
            :root {
                --nav-bg: rgba(255, 255, 255, 0.98);
                --nav-bg-scrolled: rgba(255, 255, 255, 0.99);
                --nav-border: rgba(26, 35, 50, 0.1);
                --nav-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
                --nav-height: 80px;
                --nav-height-mobile: 70px;
                --primary-color: #1a2332;
                --phi-gold: #FFD700;
                --phi-gold-light: #FFA500;
                --accent-color: #667eea;
                --text-primary: #2d3748;
                --text-secondary: #4a5568;
                --bg-secondary: #f7fafc;
                --border-color: #e2e8f0;
                --radius: 8px;
                --radius-lg: 16px;
                --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                --font-primary: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                --font-serif: 'Crimson Text', Georgia, serif;
            }

            .optimized-nav {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                background: var(--nav-bg);
                backdrop-filter: blur(20px);
                border-bottom: 1px solid var(--nav-border);
                z-index: 1000;
                transition: var(--transition);
                height: var(--nav-height);
            }

            .optimized-nav.scrolled {
                background: var(--nav-bg-scrolled);
                box-shadow: var(--nav-shadow);
            }

            .nav-container {
                max-width: 1400px;
                margin: 0 auto;
                padding: 0 2rem;
                display: flex;
                align-items: center;
                justify-content: space-between;
                height: 100%;
            }

            .nav-logo {
                display: flex;
                align-items: center;
                gap: 0.75rem;
                text-decoration: none;
                font-weight: 800;
                font-size: 1.75rem;
                color: var(--primary-color);
                transition: var(--transition);
            }

            .nav-logo:hover {
                transform: translateY(-2px);
            }

            .phi-symbol {
                color: var(--phi-gold);
                font-size: 2rem;
                font-family: var(--font-serif);
                text-shadow: 0 0 10px rgba(255, 215, 0, 0.3);
            }

            .logo-text {
                font-family: var(--font-serif);
                font-weight: 700;
            }

            .elo-badge {
                background: linear-gradient(135deg, var(--phi-gold), var(--phi-gold-light));
                color: white;
                padding: 0.25rem 0.75rem;
                border-radius: 20px;
                font-size: 0.75rem;
                font-weight: 600;
                margin-left: 0.5rem;
                box-shadow: 0 2px 8px rgba(255, 215, 0, 0.3);
            }

            .nav-menu {
                display: flex;
                list-style: none;
                gap: 1.5rem;
                align-items: center;
                margin: 0;
                padding: 0;
            }

            .nav-item {
                position: relative;
            }

            .nav-link {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.75rem 1.25rem;
                text-decoration: none;
                color: var(--text-primary);
                font-weight: 500;
                border-radius: var(--radius-lg);
                transition: var(--transition);
                position: relative;
                white-space: nowrap;
            }

            .nav-link:hover,
            .nav-link.active {
                background: linear-gradient(135deg, var(--primary-color), var(--accent-color));
                color: white;
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
            }

            .dropdown-toggle {
                cursor: pointer;
            }

            .dropdown-arrow {
                font-size: 0.75rem;
                transition: transform var(--transition);
            }

            .dropdown:hover .dropdown-arrow {
                transform: rotate(180deg);
            }

            .dropdown-menu {
                position: absolute;
                top: 100%;
                left: 0;
                background: white;
                border: 1px solid var(--border-color);
                border-radius: var(--radius-lg);
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
                min-width: 280px;
                opacity: 0;
                visibility: hidden;
                transform: translateY(-10px);
                transition: var(--transition);
                z-index: 1001;
                padding: 0.75rem;
                margin-top: 0.5rem;
            }

            .dropdown:hover .dropdown-menu {
                opacity: 1;
                visibility: visible;
                transform: translateY(0);
            }

            .dropdown-link {
                display: flex;
                align-items: center;
                gap: 1rem;
                padding: 1rem 1.25rem;
                text-decoration: none;
                color: var(--text-primary);
                border-radius: var(--radius);
                transition: var(--transition);
                margin-bottom: 0.25rem;
            }

            .dropdown-link:hover {
                background: var(--bg-secondary);
                color: var(--primary-color);
                transform: translateX(5px);
            }

            .dropdown-link-icon {
                width: 20px;
                text-align: center;
                color: var(--accent-color);
            }

            .dropdown-link-content {
                flex: 1;
            }

            .dropdown-link-title {
                font-weight: 600;
                margin-bottom: 0.25rem;
            }

            .dropdown-link-description {
                font-size: 0.85rem;
                color: var(--text-secondary);
                line-height: 1.4;
            }

            .nav-toggle {
                display: none;
                flex-direction: column;
                cursor: pointer;
                gap: 4px;
                padding: 0.5rem;
            }

            .nav-toggle span {
                width: 24px;
                height: 3px;
                background: var(--primary-color);
                transition: var(--transition);
                border-radius: 2px;
            }

            .ai-chat-trigger {
                background: linear-gradient(135deg, var(--accent-color), var(--phi-gold));
                color: white;
                border: none;
                border-radius: var(--radius-lg);
                padding: 0.75rem 1.25rem;
                margin-left: 1rem;
                cursor: pointer;
                transition: var(--transition);
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }

            .ai-chat-trigger:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
                background: linear-gradient(135deg, var(--phi-gold), var(--accent-color));
            }

            /* Mobile Optimizations */
            @media (max-width: 1024px) {
                .optimized-nav {
                    height: var(--nav-height-mobile);
                }

                .nav-container {
                    padding: 0 1rem;
                }

                .nav-menu {
                    position: fixed;
                    top: var(--nav-height-mobile);
                    left: 0;
                    right: 0;
                    background: white;
                    flex-direction: column;
                    padding: 2rem;
                    gap: 1rem;
                    transform: translateY(-100%);
                    opacity: 0;
                    visibility: hidden;
                    transition: var(--transition);
                    box-shadow: var(--nav-shadow);
                    max-height: calc(100vh - var(--nav-height-mobile));
                    overflow-y: auto;
                }

                .nav-menu.mobile-active {
                    transform: translateY(0);
                    opacity: 1;
                    visibility: visible;
                }

                .nav-toggle {
                    display: flex;
                }

                .nav-toggle.active span:nth-child(1) {
                    transform: rotate(45deg) translate(6px, 6px);
                }

                .nav-toggle.active span:nth-child(2) {
                    opacity: 0;
                }

                .nav-toggle.active span:nth-child(3) {
                    transform: rotate(-45deg) translate(6px, -6px);
                }

                .dropdown-menu {
                    position: static;
                    opacity: 1;
                    visibility: visible;
                    transform: none;
                    box-shadow: none;
                    border: none;
                    background: var(--bg-secondary);
                    margin-top: 0.5rem;
                    margin-left: 1rem;
                }

                .ai-chat-trigger {
                    margin-left: 0;
                    margin-top: 1rem;
                    width: 100%;
                    justify-content: center;
                }
            }

            /* Chrome PC Specific Optimizations */
            @media (min-width: 1025px) {
                .nav-menu {
                    gap: 2rem;
                }

                .nav-link {
                    padding: 1rem 1.5rem;
                    font-size: 1rem;
                }

                .dropdown-menu {
                    min-width: 320px;
                }
            }

            /* Accessibility */
            .nav-link:focus,
            .dropdown-link:focus,
            .ai-chat-trigger:focus {
                outline: 2px solid var(--phi-gold);
                outline-offset: 2px;
            }

            /* High contrast mode */
            @media (prefers-contrast: high) {
                .optimized-nav {
                    background: white;
                    border-bottom: 2px solid black;
                }
            }

            /* Reduced motion */
            @media (prefers-reduced-motion: reduce) {
                .optimized-nav,
                .nav-link,
                .dropdown-menu,
                .dropdown-arrow {
                    transition: none;
                }
            }
        `;

        const styleElement = document.createElement('style');
        styleElement.id = styleId;
        styleElement.textContent = styles;
        document.head.appendChild(styleElement);
    }

    createNavigation() {
        const navHTML = `
            <nav class="optimized-nav" id="optimizedNav">
                <div class="nav-container">
                    <a href="index.html" class="nav-logo">
                        <span class="phi-symbol">φ</span>
                        <span class="logo-text">Een</span>
                        <span class="elo-badge">Advanced</span>
                    </a>
                    
                    <ul class="nav-menu" id="navMenu">
                        ${this.generateMenuItems()}
                    </ul>
                    
                    <button class="ai-chat-trigger" id="aiChatTrigger" aria-label="Open AI Assistant">
                        <i class="fas fa-robot"></i>
                        <span>AI Chat</span>
                    </button>
                    
                    <div class="nav-toggle" id="navToggle" aria-label="Toggle navigation menu">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
            </nav>
        `;

        // Insert at the beginning of body
        document.body.insertAdjacentHTML('afterbegin', navHTML);

        // Add padding to body to account for fixed nav
        document.body.style.paddingTop = this.isMobile ? '70px' : '80px';
    }

    generateMenuItems() {
        const sortedCategories = Object.entries(this.siteStructure)
            .sort(([, a], [, b]) => a.priority - b.priority);

        return sortedCategories.map(([category, config]) => {
            const hasActivePage = this.categoryHasActivePage(config.pages);
            const activeClass = hasActivePage ? 'active' : '';

            return `
                <li class="nav-item dropdown">
                    <a href="#" class="nav-link dropdown-toggle ${activeClass}" aria-expanded="false">
                        <i class="${config.icon}" style="color: ${config.color}"></i>
                        <span>${category}</span>
                        <i class="fas fa-chevron-down dropdown-arrow"></i>
                    </a>
                    <ul class="dropdown-menu" role="menu">
                        ${this.generateDropdownItems(config.pages)}
                    </ul>
                </li>
            `;
        }).join('');
    }

    generateDropdownItems(pages) {
        return Object.entries(pages).map(([url, pageInfo]) => {
            const isActive = this.isPageActive(url);
            const activeClass = isActive ? 'active' : '';

            return `
                <li role="none">
                    <a href="${url}" class="dropdown-link ${activeClass}" role="menuitem">
                        <i class="${pageInfo.icon} dropdown-link-icon"></i>
                        <div class="dropdown-link-content">
                            <div class="dropdown-link-title">${pageInfo.title}</div>
                            <div class="dropdown-link-description">${pageInfo.description}</div>
                        </div>
                    </a>
                </li>
            `;
        }).join('');
    }

    categoryHasActivePage(pages) {
        return Object.keys(pages).some(page => this.isPageActive(page));
    }

    isPageActive(pageUrl) {
        return this.currentPage === pageUrl;
    }

    attachEventListeners() {
        // Mobile menu toggle
        const navToggle = document.getElementById('navToggle');
        const navMenu = document.getElementById('navMenu');

        if (navToggle && navMenu) {
            navToggle.addEventListener('click', () => {
                navMenu.classList.toggle('mobile-active');
                navToggle.classList.toggle('active');
                navToggle.setAttribute('aria-expanded',
                    navToggle.classList.contains('active').toString());
            });
        }

        // Close mobile menu when clicking outside
        document.addEventListener('click', (e) => {
            if (navToggle && navMenu &&
                !navToggle.contains(e.target) &&
                !navMenu.contains(e.target)) {
                navMenu.classList.remove('mobile-active');
                navToggle.classList.remove('active');
                navToggle.setAttribute('aria-expanded', 'false');
            }
        });

        // Scroll effect
        window.addEventListener('scroll', () => {
            const nav = document.getElementById('optimizedNav');
            if (nav) {
                if (window.scrollY > 50) {
                    nav.classList.add('scrolled');
                } else {
                    nav.classList.remove('scrolled');
                }
            }
        });

        // AI Chat trigger
        const aiChatTrigger = document.getElementById('aiChatTrigger');
        if (aiChatTrigger) {
            aiChatTrigger.addEventListener('click', () => {
                this.initializeAIChat();
            });
        }

        // Keyboard navigation
        this.setupKeyboardNavigation();
    }

    setupKeyboardNavigation() {
        document.addEventListener('keydown', (e) => {
            // Escape key closes mobile menu
            if (e.key === 'Escape') {
                const navMenu = document.getElementById('navMenu');
                const navToggle = document.getElementById('navToggle');
                if (navMenu && navMenu.classList.contains('mobile-active')) {
                    navMenu.classList.remove('mobile-active');
                    navToggle.classList.remove('active');
                    navToggle.setAttribute('aria-expanded', 'false');
                }
            }
        });
    }

    setupAccessibility() {
        // Add ARIA labels and roles
        const dropdowns = document.querySelectorAll('.dropdown');
        dropdowns.forEach(dropdown => {
            const toggle = dropdown.querySelector('.dropdown-toggle');
            const menu = dropdown.querySelector('.dropdown-menu');

            if (toggle && menu) {
                toggle.setAttribute('aria-haspopup', 'true');
                toggle.setAttribute('aria-expanded', 'false');
                menu.setAttribute('role', 'menu');

                toggle.addEventListener('click', (e) => {
                    e.preventDefault();
                    const isExpanded = toggle.getAttribute('aria-expanded') === 'true';
                    toggle.setAttribute('aria-expanded', !isExpanded);
                });
            }
        });
    }

    initializeAnimations() {
        // Smooth scroll for anchor links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const target = document.querySelector(this.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    }

    async initializeAIChat() {
        try {
            // Check if chat system is already loaded
            if (window.EenAIChat) {
                window.EenAIChat.open();
                return;
            }

            // Load chat system if not already loaded
            const chatScript = document.createElement('script');
            chatScript.src = 'js/enhanced-ai-chat.js';
            document.head.appendChild(chatScript);

            chatScript.onload = () => {
                if (window.EenAIChat) {
                    window.EenAIChat.open();
                }
            };
        } catch (error) {
            console.error('Failed to initialize AI chat:', error);
            // Fallback: show a simple message
            alert('AI Chat system is loading. Please try again in a moment.');
        }
    }
}

// Initialize navigation when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const navigation = new OptimizedUnifiedNavigation();
    navigation.init();

    // Make it globally available
    window.OptimizedUnifiedNavigation = navigation;
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = OptimizedUnifiedNavigation;
} 