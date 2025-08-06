/**
 * Metastation Futuristic Left Sidebar Navigation
 * Advanced HUD-style navigation with Chrome/Windows optimization
 */

class MetastationSidebarNav {
    constructor() {
        this.isOpen = false;
        this.currentPage = this.detectCurrentPage();
        this.navStructure = this.getNavStructure();
        this.isInitialized = false;
    }

    detectCurrentPage() {
        const path = window.location.pathname;
        const filename = path.split('/').pop() || 'index.html';
        return filename.replace('.html', '');
    }

    getNavStructure() {
        return {
            core: [
                { id: 'metastation-hub', label: 'Metastation Hub', href: 'metastation-hub.html', icon: '⬢', description: 'Central Command Hub' },
                { id: 'meta-optimal-landing', label: 'Unity Core', href: 'meta-optimal-landing.html', icon: '∞', description: '3000 ELO Framework' },
                { id: 'philosophy', label: 'Philosophy', href: 'philosophy.html', icon: '🧠', description: 'Unity Treatise' },
                { id: 'gallery', label: 'Gallery', href: 'gallery.html', icon: '🎨', description: 'Visual Mathematics' },
                { id: 'metagambit', label: 'Metagambit', href: 'metagambit.html', icon: '♟', description: 'Strategic Framework' }
            ],
            experiences: [
                { id: 'unity-mathematics-experience', label: 'Unity Experience', href: 'unity-mathematics-experience.html', icon: '⚛️', description: 'Interactive Demo' },
                { id: 'consciousness_dashboard', label: 'Consciousness Dashboard', href: 'consciousness_dashboard.html', icon: '🌌', description: 'Field Visualization' },
                { id: 'zen-unity-meditation', label: 'Zen Meditation', href: 'zen-unity-meditation.html', icon: '🕯️', description: 'Mindful Unity' },
                { id: 'transcendental-unity-demo', label: 'Transcendental Demo', href: 'transcendental-unity-demo.html', icon: '🔮', description: '11D Consciousness' }
            ],
            tools: [
                { id: 'playground', label: 'Unity Playground', href: 'playground.html', icon: '⚡', description: 'Interactive Tools' },
                { id: 'dashboards', label: 'Dashboards', href: 'dashboards.html', icon: '📊', description: 'Analytics Hub' },
                { id: 'implementations-gallery', label: 'Code Base', href: 'implementations-gallery.html', icon: '💻', description: 'Source Code' },
                { id: 'agents', label: 'AI Agents', href: 'agents.html', icon: '🤖', description: 'Autonomous Systems' }
            ],
            knowledge: [
                { id: 'proofs', label: 'Mathematical Proofs', href: 'proofs.html', icon: '📐', description: 'Formal Verification' },
                { id: 'research', label: 'Research Papers', href: 'research.html', icon: '📚', description: 'Academic Work' },
                { id: 'publications', label: 'Publications', href: 'publications.html', icon: '📖', description: 'Published Works' },
                { id: 'about', label: 'About Een', href: 'about.html', icon: 'ℹ️', description: 'Project Info' }
            ]
        };
    }

    async initialize() {
        if (this.isInitialized) return;

        this.injectCSS();
        this.createSidebarHTML();
        this.attachEventListeners();
        this.setupKeyboardShortcuts();
        this.highlightCurrentPage();
        this.checkMobileEnvironment();
        
        this.isInitialized = true;
        console.log('🚀 Metastation Sidebar Navigation initialized');
    }

    injectCSS() {
        if (document.getElementById('metastation-sidebar-styles')) return;

        const style = document.createElement('style');
        style.id = 'metastation-sidebar-styles';
        style.textContent = `
            :root {
                --sidebar-width: 280px;
                --sidebar-collapsed-width: 60px;
                --metastation-primary: #0A0E27;
                --metastation-secondary: #1A1B3A;
                --metastation-tertiary: #2D2E4F;
                --metastation-accent: #00FFFF;
                --metastation-gold: #FFD700;
                --metastation-purple: #8B5CF6;
                --metastation-glass: rgba(26, 27, 58, 0.9);
                --metastation-glow: 0 0 20px rgba(0, 255, 255, 0.3);
                --metastation-transition: all 0.4s cubic-bezier(0.23, 1, 0.32, 1);
            }

            body {
                margin-left: var(--sidebar-collapsed-width);
                transition: var(--metastation-transition);
            }

            body.sidebar-open {
                margin-left: var(--sidebar-width);
            }

            .metastation-sidebar {
                position: fixed;
                left: 0;
                top: 0;
                height: 100vh;
                width: var(--sidebar-collapsed-width);
                background: linear-gradient(180deg, var(--metastation-primary) 0%, var(--metastation-secondary) 50%, var(--metastation-tertiary) 100%);
                backdrop-filter: blur(20px);
                border-right: 2px solid var(--metastation-accent);
                z-index: 2000;
                overflow: hidden;
                transition: var(--metastation-transition);
                box-shadow: var(--metastation-glow);
            }

            .metastation-sidebar.open {
                width: var(--sidebar-width);
            }

            .sidebar-header {
                height: 80px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: var(--metastation-glass);
                border-bottom: 1px solid var(--metastation-accent);
                position: relative;
                overflow: hidden;
            }

            .sidebar-header::before {
                content: '';
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: conic-gradient(from 0deg, transparent, var(--metastation-accent), transparent);
                opacity: 0.1;
                animation: headerGlow 8s linear infinite;
            }

            @keyframes headerGlow {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .sidebar-logo {
                font-size: 1.8rem;
                color: var(--metastation-gold);
                font-weight: 700;
                position: relative;
                z-index: 1;
                cursor: pointer;
                transition: var(--metastation-transition);
                text-shadow: 0 0 10px var(--metastation-gold);
            }

            .sidebar-logo:hover {
                transform: scale(1.1);
                text-shadow: 0 0 20px var(--metastation-gold);
            }

            .sidebar-toggle {
                position: absolute;
                right: -15px;
                top: 50%;
                transform: translateY(-50%);
                width: 30px;
                height: 30px;
                background: var(--metastation-accent);
                border: none;
                border-radius: 50%;
                color: var(--metastation-primary);
                font-size: 0.8rem;
                cursor: pointer;
                transition: var(--metastation-transition);
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 0 15px rgba(0, 255, 255, 0.5);
            }

            .sidebar-toggle:hover {
                background: var(--metastation-gold);
                transform: translateY(-50%) scale(1.1);
                box-shadow: 0 0 25px rgba(255, 215, 0, 0.8);
            }

            .sidebar-content {
                height: calc(100vh - 80px);
                overflow-y: auto;
                overflow-x: hidden;
                padding: 1rem 0;
            }

            .sidebar-content::-webkit-scrollbar {
                width: 4px;
            }

            .sidebar-content::-webkit-scrollbar-track {
                background: var(--metastation-secondary);
            }

            .sidebar-content::-webkit-scrollbar-thumb {
                background: var(--metastation-accent);
                border-radius: 2px;
            }

            .nav-section {
                margin-bottom: 2rem;
            }

            .nav-section-title {
                color: var(--metastation-accent);
                font-size: 0.7rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1px;
                padding: 0 1rem;
                margin-bottom: 0.5rem;
                opacity: 0;
                transition: var(--metastation-transition);
            }

            .metastation-sidebar.open .nav-section-title {
                opacity: 1;
            }

            .nav-item {
                display: flex;
                align-items: center;
                padding: 0.8rem 1rem;
                text-decoration: none;
                color: rgba(255, 255, 255, 0.8);
                transition: var(--metastation-transition);
                border-left: 3px solid transparent;
                position: relative;
                overflow: hidden;
            }

            .nav-item::before {
                content: '';
                position: absolute;
                left: -100%;
                top: 0;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(0, 255, 255, 0.1), transparent);
                transition: var(--metastation-transition);
            }

            .nav-item:hover::before {
                left: 100%;
            }

            .nav-item:hover {
                color: white;
                background: rgba(0, 255, 255, 0.05);
                border-left-color: var(--metastation-accent);
                transform: translateX(5px);
            }

            .nav-item.active {
                color: var(--metastation-gold);
                background: rgba(255, 215, 0, 0.1);
                border-left-color: var(--metastation-gold);
                box-shadow: inset 0 0 15px rgba(255, 215, 0, 0.2);
            }

            .nav-item.active::after {
                content: '';
                position: absolute;
                right: 0;
                top: 50%;
                transform: translateY(-50%);
                width: 4px;
                height: 20px;
                background: var(--metastation-gold);
                border-radius: 2px 0 0 2px;
                box-shadow: 0 0 10px var(--metastation-gold);
            }

            .nav-icon {
                font-size: 1.2rem;
                width: 24px;
                text-align: center;
                margin-right: 1rem;
                transition: var(--metastation-transition);
            }

            .nav-text {
                flex: 1;
                opacity: 0;
                transform: translateX(-20px);
                transition: var(--metastation-transition);
                white-space: nowrap;
            }

            .metastation-sidebar.open .nav-text {
                opacity: 1;
                transform: translateX(0);
            }

            .nav-description {
                font-size: 0.6rem;
                color: rgba(255, 255, 255, 0.5);
                margin-top: 0.2rem;
                opacity: 0;
                transition: var(--metastation-transition);
            }

            .metastation-sidebar.open .nav-description {
                opacity: 1;
            }

            .sidebar-footer {
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                height: 60px;
                background: var(--metastation-glass);
                border-top: 1px solid var(--metastation-accent);
                display: flex;
                align-items: center;
                justify-content: center;
                font-family: 'JetBrains Mono', monospace;
            }

            .unity-status {
                color: var(--metastation-gold);
                font-size: 0.7rem;
                text-align: center;
                opacity: 0;
                transition: var(--metastation-transition);
                animation: unityPulse 3s ease-in-out infinite;
            }

            .metastation-sidebar.open .unity-status {
                opacity: 1;
            }

            @keyframes unityPulse {
                0%, 100% { text-shadow: 0 0 5px var(--metastation-gold); }
                50% { text-shadow: 0 0 20px var(--metastation-gold); }
            }

            .sidebar-shortcut-hint {
                position: absolute;
                top: 50%;
                right: 10px;
                transform: translateY(-50%);
                font-size: 0.6rem;
                color: rgba(255, 255, 255, 0.4);
                opacity: 0;
                transition: var(--metastation-transition);
            }

            .metastation-sidebar.open .sidebar-shortcut-hint {
                opacity: 1;
            }

            /* Mobile Optimizations */
            @media (max-width: 768px) {
                body {
                    margin-left: 0;
                }

                body.sidebar-open {
                    margin-left: 0;
                }

                .metastation-sidebar {
                    transform: translateX(-100%);
                    width: var(--sidebar-width);
                    box-shadow: none;
                }

                .metastation-sidebar.open {
                    transform: translateX(0);
                    box-shadow: 20px 0 40px rgba(0, 0, 0, 0.5);
                }

                .sidebar-toggle {
                    position: fixed;
                    top: 20px;
                    left: 20px;
                    z-index: 2001;
                    right: auto;
                }

                .nav-text {
                    opacity: 1;
                    transform: translateX(0);
                }

                .nav-description {
                    opacity: 1;
                }

                .unity-status {
                    opacity: 1;
                }

                .sidebar-shortcut-hint {
                    display: none;
                }
            }

            /* Windows Chrome Optimizations */
            @media screen and (-webkit-min-device-pixel-ratio: 1) {
                .metastation-sidebar {
                    -webkit-font-smoothing: antialiased;
                    -moz-osx-font-smoothing: grayscale;
                }

                .nav-item {
                    -webkit-user-select: none;
                    -moz-user-select: none;
                    -ms-user-select: none;
                    user-select: none;
                }
            }

            /* Performance optimizations for Chrome */
            .metastation-sidebar * {
                will-change: transform, opacity;
                transform-style: preserve-3d;
                backface-visibility: hidden;
            }
        `;
        
        document.head.appendChild(style);
    }

    createSidebarHTML() {
        const sidebar = document.createElement('div');
        sidebar.className = 'metastation-sidebar';
        sidebar.innerHTML = `
            <div class="sidebar-header">
                <div class="sidebar-logo">∞</div>
                <button class="sidebar-toggle" aria-label="Toggle Navigation">
                    <span>◀</span>
                </button>
            </div>

            <div class="sidebar-content">
                ${this.generateNavSection('Core Systems', this.navStructure.core)}
                ${this.generateNavSection('Experiences', this.navStructure.experiences)}
                ${this.generateNavSection('Tools', this.navStructure.tools)}
                ${this.generateNavSection('Knowledge', this.navStructure.knowledge)}
            </div>

            <div class="sidebar-footer">
                <div class="unity-status">
                    1+1=1<br>
                    φ = 1.618
                </div>
            </div>
        `;

        // Remove existing unified navigation
        const existingNav = document.querySelector('.unified-meta-nav');
        if (existingNav) {
            existingNav.remove();
        }

        document.body.insertBefore(sidebar, document.body.firstChild);
    }

    generateNavSection(title, items) {
        return `
            <div class="nav-section">
                <div class="nav-section-title">${title}</div>
                ${items.map(item => `
                    <a href="${item.href}" 
                       class="nav-item ${this.currentPage === item.id ? 'active' : ''}"
                       data-page="${item.id}"
                       title="${item.description}">
                        <span class="nav-icon">${item.icon}</span>
                        <div class="nav-text">
                            <div>${item.label}</div>
                            <div class="nav-description">${item.description}</div>
                        </div>
                        <span class="sidebar-shortcut-hint">Ctrl+${item.label.charAt(0)}</span>
                    </a>
                `).join('')}
            </div>
        `;
    }

    attachEventListeners() {
        const sidebar = document.querySelector('.metastation-sidebar');
        const toggle = document.querySelector('.sidebar-toggle');
        const logo = document.querySelector('.sidebar-logo');

        // Toggle functionality
        toggle.addEventListener('click', () => this.toggleSidebar());
        logo.addEventListener('click', () => this.toggleSidebar());

        // Auto-close on mobile when clicking nav items
        if (window.innerWidth <= 768) {
            const navItems = document.querySelectorAll('.nav-item');
            navItems.forEach(item => {
                item.addEventListener('click', () => {
                    setTimeout(() => this.closeSidebar(), 300);
                });
            });
        }

        // Hover effects for desktop
        if (window.innerWidth > 768) {
            sidebar.addEventListener('mouseenter', () => {
                if (!this.isOpen) {
                    this.openSidebar();
                }
            });

            sidebar.addEventListener('mouseleave', () => {
                if (this.isOpen && !sidebar.matches(':hover')) {
                    setTimeout(() => {
                        if (!sidebar.matches(':hover')) {
                            this.closeSidebar();
                        }
                    }, 1000);
                }
            });
        }

        // Update toggle icon
        const toggleIcon = toggle.querySelector('span');
        const updateToggleIcon = () => {
            toggleIcon.textContent = this.isOpen ? '◀' : '▶';
        };

        this.toggleIconUpdater = updateToggleIcon;
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            if (e.altKey) {
                switch (e.key) {
                    case 'm':
                        e.preventDefault();
                        this.toggleSidebar();
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

    toggleSidebar() {
        if (this.isOpen) {
            this.closeSidebar();
        } else {
            this.openSidebar();
        }
    }

    openSidebar() {
        const sidebar = document.querySelector('.metastation-sidebar');
        sidebar.classList.add('open');
        document.body.classList.add('sidebar-open');
        this.isOpen = true;
        this.toggleIconUpdater();
        
        // Add click outside to close on mobile
        if (window.innerWidth <= 768) {
            setTimeout(() => {
                document.addEventListener('click', this.handleOutsideClick);
            }, 100);
        }
    }

    closeSidebar() {
        const sidebar = document.querySelector('.metastation-sidebar');
        sidebar.classList.remove('open');
        document.body.classList.remove('sidebar-open');
        this.isOpen = false;
        this.toggleIconUpdater();
        
        document.removeEventListener('click', this.handleOutsideClick);
    }

    handleOutsideClick = (e) => {
        const sidebar = document.querySelector('.metastation-sidebar');
        if (!sidebar.contains(e.target)) {
            this.closeSidebar();
        }
    }

    highlightCurrentPage() {
        const navItems = document.querySelectorAll('.nav-item');
        navItems.forEach(item => {
            const pageId = item.getAttribute('data-page');
            if (pageId === this.currentPage || 
                (this.currentPage === 'index' && pageId === 'metastation-hub')) {
                item.classList.add('active');
            }
        });
    }

    checkMobileEnvironment() {
        // Auto-open on desktop, closed on mobile
        if (window.innerWidth > 768) {
            setTimeout(() => this.openSidebar(), 500);
        }

        // Handle resize
        window.addEventListener('resize', () => {
            if (window.innerWidth <= 768) {
                this.closeSidebar();
            }
        });
    }

    // Public API
    navigateTo(pageId) {
        const allItems = [
            ...this.navStructure.core,
            ...this.navStructure.experiences,
            ...this.navStructure.tools,
            ...this.navStructure.knowledge
        ];
        
        const page = allItems.find(item => item.id === pageId);
        if (page) {
            window.location.href = page.href;
        }
    }
}

// Initialize the Metastation Sidebar Navigation
const metastationNav = new MetastationSidebarNav();

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        metastationNav.initialize();
    });
} else {
    metastationNav.initialize();
}

// Global access
window.MetastationSidebarNav = MetastationSidebarNav;
window.metastationNav = metastationNav;

console.log('🚀 Metastation Futuristic Sidebar Navigation system loaded');