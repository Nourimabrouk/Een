/**
 * Een Unity Mathematics - Master Navigation System v2.0
 * 
 * Ultimate unified navigation system that:
 * - Provides consistent navigation across ALL pages
 * - Includes persistent chatbot button bottom-right on every page
 * - Auto-detects current page and highlights appropriately
 * - Uses natural categories fitting research and code structure
 * - Links to every HTML page in the website folder
 * - Includes modern UX with smooth animations and accessibility
 * - Integrates seamlessly with the new modular chat system
 */

class MasterNavigation {
    constructor() {
        this.currentPage = this.detectCurrentPage();
        this.isInitialized = false;
        this.chatInitialized = false;

        // Complete site structure with natural categories
        this.siteStructure = {
            'Mathematics': {
                icon: 'fas fa-calculator',
                color: '#667eea',
                pages: {
                    'proofs.html': 'Unity Proofs',
                    '3000-elo-proof.html': 'Advanced Proofs',
                    'playground.html': 'Interactive Playground',
                    'mathematical_playground.html': 'Math Playground',
                    'enhanced-unity-demo.html': 'Unity Demo',
                    'transcendental-unity-demo.html': '🧠 Transcendental Unity',
                    'examples/unity-calculator.html': 'Unity Calculator',
                    'examples/phi-harmonic-explorer.html': 'φ-Harmonic Explorer'
                }
            },
            'Consciousness': {
                icon: 'fas fa-brain',
                color: '#764ba2',
                pages: {
                    'philosophy.html': 'Philosophy',
                    'consciousness_dashboard.html': 'Consciousness Dashboard',
                    'consciousness_dashboard_clean.html': 'Clean Dashboard',
                    'unity_consciousness_experience.html': 'Unity Experience',
                    'unity_visualization.html': 'Unity Visualization',
                    'transcendental-unity-demo.html': '🧠 Transcendental Unity',
                    'gallery/phi_consciousness_transcendence.html': 'Transcendence Gallery'
                }
            },
            'AI Systems': {
                icon: 'fas fa-robot',
                color: '#4a90e2',
                pages: {
                    'agents.html': 'AI Agents',
                    'metagambit.html': 'MetaGambit',
                    'metagamer_agent.html': 'MetaGamer Agent',
                    'al_khwarizmi_phi_unity.html': 'Al-Khwarizmi φ Unity',
                    'revolutionary-landing.html': 'Revolutionary AI',
                    'meta-optimal-landing.html': 'Meta-Optimal System'
                }
            },
            'Transcendental Computing': {
                icon: 'fas fa-infinity',
                color: '#FFD700',
                pages: {
                    'transcendental-unity-demo.html': '🧠 Transcendental Unity Demo',
                    'consciousness_dashboard.html': 'Consciousness Field',
                    'unity_visualization.html': 'Unity Visualization'
                }
            },
            'Research': {
                icon: 'fas fa-flask',
                color: '#28a745',
                pages: {
                    'research.html': 'Research Overview',
                    'publications.html': 'Publications',
                    'implementations.html': 'Implementations',
                    'dashboards.html': 'Research Dashboards',
                    'gallery.html': 'Visual Gallery'
                }
            },
            'Learning': {
                icon: 'fas fa-graduation-cap',
                color: '#f39c12',
                pages: {
                    'learn.html': 'Learn Unity Math',
                    'learning.html': 'Learning Center',
                    'further-reading.html': 'Further Reading',
                    'examples/index.html': 'Interactive Examples'
                }
            },
            'Tools': {
                icon: 'fas fa-tools',
                color: '#6c757d',
                pages: {
                    'mobile-app.html': 'Mobile App',
                    'about.html': 'About'
                }
            }
        };

        this.init();
    }

    detectCurrentPage() {
        const path = window.location.pathname;
        const filename = path.split('/').pop() || 'index.html';

        // Normalize path for comparison
        if (filename === '' || filename === 'index.html' || path.endsWith('/')) {
            return 'index.html';
        }

        return filename;
    }

    init() {
        if (this.isInitialized) return;

        this.createNavigationStructure();
        this.createChatButton();
        this.attachEventListeners();
        this.setupAccessibility();
        this.initializeAnimations();

        this.isInitialized = true;
        console.info('Master Navigation System v2.0 initialized');
    }

    createNavigationStructure() {
        const navHTML = `
            <nav class="master-nav" id="masterNavigation" role="navigation" aria-label="Main navigation">
                <div class="nav-container">
                    <div class="nav-brand">
                        <a href="index.html" class="brand-link" aria-label="Een Unity Mathematics Home">
                            <span class="phi-symbol" aria-hidden="true">φ</span>
                            <span class="brand-text">Een</span>
                            <span class="brand-subtitle">Unity Mathematics</span>
                        </a>
                    </div>
                    
                    <div class="nav-content">
                        <ul class="nav-menu" role="menubar">
                            ${this.generateMenuItems()}
                        </ul>
                        
                        <div class="nav-actions">
                            <button class="theme-toggle" id="themeToggle" aria-label="Toggle theme" title="Toggle light/dark theme">
                                <span class="theme-icon">🌓</span>
                            </button>
                            <button class="search-toggle" id="searchToggle" aria-label="Search" title="Search website">
                                <i class="fas fa-search"></i>
                            </button>
                        </div>
                        
                        <button class="mobile-menu-toggle" id="mobileMenuToggle" aria-label="Toggle mobile menu">
                            <span class="hamburger-line"></span>
                            <span class="hamburger-line"></span>
                            <span class="hamburger-line"></span>
                        </button>
                    </div>
                </div>
            </nav>
            
            <!-- Quick Search Modal -->
            <div class="search-modal" id="searchModal" role="dialog" aria-labelledby="searchModalTitle" aria-hidden="true">
                <div class="search-modal-content">
                    <div class="search-header">
                        <h2 id="searchModalTitle">Search Een Unity Mathematics</h2>
                        <button class="search-close" id="searchClose" aria-label="Close search">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                    <div class="search-input-wrapper">
                        <input type="text" id="searchInput" placeholder="Search pages, concepts, proofs..." aria-label="Search query">
                        <i class="fas fa-search search-icon"></i>
                    </div>
                    <div class="search-results" id="searchResults" role="listbox">
                        <!-- Search results will be populated here -->
                    </div>
                </div>
            </div>
        `;

        // Insert navigation at the beginning of body or replace existing nav
        const existingNav = document.querySelector('nav, #navigation-placeholder, .navbar, .enhanced-nav');
        if (existingNav) {
            existingNav.outerHTML = navHTML;
        } else {
            document.body.insertAdjacentHTML('afterbegin', navHTML);
        }

        this.injectStyles();
    }

    generateMenuItems() {
        let menuHTML = '';

        // Home item (always first)
        const isHome = this.currentPage === 'index.html';
        menuHTML += `
            <li class="nav-item ${isHome ? 'active' : ''}" role="none">
                <a href="index.html" class="nav-link ${isHome ? 'current-page' : ''}" role="menuitem" ${isHome ? 'aria-current="page"' : ''}>
                    <i class="fas fa-home"></i>
                    <span>Home</span>
                </a>
            </li>
        `;

        // Category dropdowns
        for (const [categoryName, categoryData] of Object.entries(this.siteStructure)) {
            const hasActivePage = this.categoryHasActivePage(categoryData.pages);

            menuHTML += `
                <li class="nav-item dropdown ${hasActivePage ? 'has-active' : ''}" role="none">
                    <a href="#" class="nav-link dropdown-toggle" role="menuitem" aria-haspopup="true" aria-expanded="false" tabindex="0">
                        <i class="${categoryData.icon}" style="color: ${categoryData.color}"></i>
                        <span>${categoryName}</span>
                        <i class="fas fa-chevron-down dropdown-arrow"></i>
                    </a>
                    <ul class="dropdown-menu" role="menu" aria-label="${categoryName} submenu">
                        ${this.generateDropdownItems(categoryData.pages)}
                    </ul>
                </li>
            `;
        }

        return menuHTML;
    }

    generateDropdownItems(pages) {
        let itemsHTML = '';

        for (const [pageUrl, pageTitle] of Object.entries(pages)) {
            const isActive = this.isPageActive(pageUrl);
            itemsHTML += `
                <li role="none">
                    <a href="${pageUrl}" class="dropdown-link ${isActive ? 'active' : ''}" role="menuitem" ${isActive ? 'aria-current="page"' : ''}>
                        ${pageTitle}
                    </a>
                </li>
            `;
        }

        return itemsHTML;
    }

    categoryHasActivePage(pages) {
        return Object.keys(pages).some(pageUrl => this.isPageActive(pageUrl));
    }

    isPageActive(pageUrl) {
        const normalizedUrl = pageUrl.split('/').pop();
        const currentPageNormalized = this.currentPage.split('/').pop();
        return normalizedUrl === currentPageNormalized;
    }

    createChatButton() {
        const chatButtonHTML = `
            <div class="chat-fab-container" id="chatFabContainer">
                <button class="chat-fab" id="chatFab" aria-label="Open AI Chat Assistant" title="Chat with Een Unity AI">
                    <div class="fab-icon">
                        <i class="fas fa-robot"></i>
                    </div>
                    <div class="fab-pulse"></div>
                </button>
                <div class="fab-tooltip">Chat with AI Assistant</div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', chatButtonHTML);
    }

    attachEventListeners() {
        // Mobile menu toggle
        const mobileToggle = document.getElementById('mobileMenuToggle');
        const navMenu = document.querySelector('.nav-menu');

        if (mobileToggle && navMenu) {
            mobileToggle.addEventListener('click', () => {
                const isOpen = navMenu.classList.toggle('mobile-open');
                mobileToggle.classList.toggle('active');
                mobileToggle.setAttribute('aria-expanded', isOpen.toString());

                // Trap focus in mobile menu
                if (isOpen) {
                    this.trapFocusInMobileMenu();
                }
            });
        }

        // Dropdown menus
        document.querySelectorAll('.dropdown-toggle').forEach(toggle => {
            toggle.addEventListener('click', (e) => {
                e.preventDefault();
                const dropdown = toggle.parentElement;
                const menu = dropdown.querySelector('.dropdown-menu');
                const isOpen = dropdown.classList.toggle('open');

                toggle.setAttribute('aria-expanded', isOpen.toString());

                // Close other dropdowns
                document.querySelectorAll('.dropdown.open').forEach(otherDropdown => {
                    if (otherDropdown !== dropdown) {
                        otherDropdown.classList.remove('open');
                        otherDropdown.querySelector('.dropdown-toggle').setAttribute('aria-expanded', 'false');
                    }
                });
            });

            // Keyboard navigation
            toggle.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    toggle.click();
                } else if (e.key === 'ArrowDown') {
                    e.preventDefault();
                    const dropdown = toggle.parentElement;
                    if (!dropdown.classList.contains('open')) {
                        toggle.click();
                    }
                    const firstLink = dropdown.querySelector('.dropdown-link');
                    if (firstLink) firstLink.focus();
                }
            });
        });

        // Close dropdowns when clicking outside
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.dropdown')) {
                document.querySelectorAll('.dropdown.open').forEach(dropdown => {
                    dropdown.classList.remove('open');
                    dropdown.querySelector('.dropdown-toggle').setAttribute('aria-expanded', 'false');
                });
            }
        });

        // Theme toggle
        const themeToggle = document.getElementById('themeToggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', () => {
                this.toggleTheme();
            });
        }

        // Search functionality
        this.setupSearch();

        // Chat FAB
        this.setupChatButton();

        // Keyboard shortcuts
        this.setupKeyboardShortcuts();

        // Close mobile menu on page navigation
        document.querySelectorAll('.nav-link, .dropdown-link').forEach(link => {
            link.addEventListener('click', () => {
                navMenu?.classList.remove('mobile-open');
                mobileToggle?.classList.remove('active');
            });
        });
    }

    setupChatButton() {
        const chatFab = document.getElementById('chatFab');
        if (!chatFab) return;

        chatFab.addEventListener('click', async () => {
            await this.initializeChat();
        });

        // Show tooltip on hover (desktop only)
        if (!this.isMobileDevice()) {
            const container = document.getElementById('chatFabContainer');
            const tooltip = container?.querySelector('.fab-tooltip');

            if (container && tooltip) {
                container.addEventListener('mouseenter', () => {
                    tooltip.classList.add('show');
                });

                container.addEventListener('mouseleave', () => {
                    tooltip.classList.remove('show');
                });
            }
        }
    }

    async initializeChat() {
        if (this.chatInitialized) {
            // Chat already initialized, just open it
            if (window.EenChat && window.EenChat.getInstance()) {
                window.EenChat.getInstance().open();
            }
            return;
        }

        try {
            // Show loading state
            const chatFab = document.getElementById('chatFab');
            const originalHTML = chatFab.innerHTML;
            chatFab.innerHTML = '<div class="fab-icon"><i class="fas fa-spinner fa-spin"></i></div>';
            chatFab.disabled = true;

            // Load chat modules if not already loaded
            if (!window.EenChatIntegration) {
                await this.loadChatModules();
            }

            // Initialize chat
            let chatInstance = window.EenChat?.getInstance();
            if (!chatInstance) {
                chatInstance = await window.EenChat.initialize({
                    skipHealthCheck: true,
                    config: {
                        ui: {
                            ENABLE_ANIMATIONS: !this.prefersReducedMotion(),
                            ENABLE_OFFLINE_FALLBACK: true,
                            ENABLE_DARK_MODE: this.isDarkMode()
                        }
                    }
                });
            }

            // Open chat
            chatInstance.open();
            this.chatInitialized = true;

            // Restore button
            chatFab.innerHTML = originalHTML;
            chatFab.disabled = false;

        } catch (error) {
            console.error('Failed to initialize chat:', error);

            // Restore button and show error
            const chatFab = document.getElementById('chatFab');
            chatFab.innerHTML = '<div class="fab-icon"><i class="fas fa-exclamation-triangle"></i></div>';
            chatFab.disabled = false;

            // Try fallback
            setTimeout(() => {
                chatFab.innerHTML = '<div class="fab-icon"><i class="fas fa-robot"></i></div>';
                this.showChatErrorMessage();
            }, 2000);
        }
    }

    async loadChatModules() {
        const modules = [
            'js/config.js',
            'js/chat/chat-api.js',
            'js/chat/chat-state.js',
            'js/chat/chat-ui.js',
            'js/chat/chat-utils.js',
            'js/chat/chat-integration.js'
        ];

        for (const modulePath of modules) {
            await this.loadScript(modulePath);
        }
    }

    loadScript(src) {
        return new Promise((resolve, reject) => {
            if (document.querySelector(`script[src="${src}"]`)) {
                resolve();
                return;
            }

            const script = document.createElement('script');
            script.type = 'module';
            script.src = src;
            script.onload = resolve;
            script.onerror = () => reject(new Error(`Failed to load ${src}`));
            document.head.appendChild(script);
        });
    }

    showChatErrorMessage() {
        const message = document.createElement('div');
        message.className = 'chat-error-toast';
        message.innerHTML = `
            <div class="toast-content">
                <i class="fas fa-exclamation-circle"></i>
                <span>AI Chat temporarily unavailable</span>
                <button class="toast-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;

        message.style.cssText = `
            position: fixed;
            bottom: 100px;
            right: 20px;
            background: #ff6b6b;
            color: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 10000;
            animation: slideInUp 0.3s ease;
        `;

        document.body.appendChild(message);

        setTimeout(() => {
            message.remove();
        }, 5000);
    }

    setupSearch() {
        const searchToggle = document.getElementById('searchToggle');
        const searchModal = document.getElementById('searchModal');
        const searchClose = document.getElementById('searchClose');
        const searchInput = document.getElementById('searchInput');
        const searchResults = document.getElementById('searchResults');

        if (!searchToggle || !searchModal) return;

        searchToggle.addEventListener('click', () => {
            searchModal.classList.add('open');
            searchModal.setAttribute('aria-hidden', 'false');
            searchInput.focus();
            document.body.style.overflow = 'hidden';
        });

        const closeSearch = () => {
            searchModal.classList.remove('open');
            searchModal.setAttribute('aria-hidden', 'true');
            document.body.style.overflow = '';
        };

        searchClose?.addEventListener('click', closeSearch);

        // Close on Escape
        searchModal.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeSearch();
        });

        // Search functionality
        if (searchInput && searchResults) {
            let searchTimeout;
            searchInput.addEventListener('input', (e) => {
                clearTimeout(searchTimeout);
                searchTimeout = setTimeout(() => {
                    this.performSearch(e.target.value, searchResults);
                }, 300);
            });
        }
    }

    performSearch(query, resultsContainer) {
        if (!query.trim()) {
            resultsContainer.innerHTML = '<div class="search-empty">Start typing to search...</div>';
            return;
        }

        const searchablePages = [];

        // Add home page
        searchablePages.push({
            title: 'Home - Een Unity Mathematics',
            url: 'index.html',
            description: 'Main page exploring Unity Mathematics where 1+1=1',
            category: 'Home'
        });

        // Add all pages from structure
        for (const [categoryName, categoryData] of Object.entries(this.siteStructure)) {
            for (const [pageUrl, pageTitle] of Object.entries(categoryData.pages)) {
                searchablePages.push({
                    title: pageTitle,
                    url: pageUrl,
                    description: `${categoryName} - ${pageTitle}`,
                    category: categoryName
                });
            }
        }

        // Filter results
        const results = searchablePages.filter(page =>
            page.title.toLowerCase().includes(query.toLowerCase()) ||
            page.description.toLowerCase().includes(query.toLowerCase()) ||
            page.category.toLowerCase().includes(query.toLowerCase())
        );

        if (results.length === 0) {
            resultsContainer.innerHTML = '<div class="search-empty">No results found</div>';
            return;
        }

        const resultsHTML = results.map(result => `
            <div class="search-result" role="option">
                <a href="${result.url}" class="search-result-link">
                    <div class="search-result-title">${result.title}</div>
                    <div class="search-result-description">${result.description}</div>
                    <div class="search-result-category">${result.category}</div>
                </a>
            </div>
        `).join('');

        resultsContainer.innerHTML = resultsHTML;
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Cmd/Ctrl + K for search
            if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
                e.preventDefault();
                document.getElementById('searchToggle')?.click();
            }

            // Cmd/Ctrl + / for chat
            if ((e.metaKey || e.ctrlKey) && e.key === '/') {
                e.preventDefault();
                document.getElementById('chatFab')?.click();
            }

            // Escape to close mobile menu
            if (e.key === 'Escape') {
                const navMenu = document.querySelector('.nav-menu');
                const mobileToggle = document.getElementById('mobileMenuToggle');
                if (navMenu?.classList.contains('mobile-open')) {
                    navMenu.classList.remove('mobile-open');
                    mobileToggle?.classList.remove('active');
                }
            }
        });
    }

    setupAccessibility() {
        // Skip to content link
        const skipLink = document.createElement('a');
        skipLink.href = '#main-content';
        skipLink.className = 'skip-to-content';
        skipLink.textContent = 'Skip to main content';
        skipLink.style.cssText = `
            position: absolute;
            top: -40px;
            left: 6px;
            background: #000;
            color: #fff;
            padding: 8px;
            text-decoration: none;
            border-radius: 4px;
            z-index: 10000;
            transition: top 0.3s;
        `;

        skipLink.addEventListener('focus', () => {
            skipLink.style.top = '6px';
        });

        skipLink.addEventListener('blur', () => {
            skipLink.style.top = '-40px';
        });

        document.body.insertBefore(skipLink, document.body.firstChild);

        // Add main content landmark if not exists
        if (!document.getElementById('main-content')) {
            const main = document.querySelector('main') || document.querySelector('.main-content') ||
                document.querySelector('[role="main"]') || document.body.children[1];
            if (main) {
                main.id = 'main-content';
                main.setAttribute('role', 'main');
            }
        }
    }

    initializeAnimations() {
        // Scroll-based animations
        let lastScrollTop = 0;
        const nav = document.getElementById('masterNavigation');

        window.addEventListener('scroll', () => {
            const scrollTop = window.pageYOffset || document.documentElement.scrollTop;

            // Auto-hide on scroll down, show on scroll up
            if (scrollTop > lastScrollTop && scrollTop > 100) {
                nav?.classList.add('nav-hidden');
            } else {
                nav?.classList.remove('nav-hidden');
            }

            // Add shadow when scrolled
            if (scrollTop > 10) {
                nav?.classList.add('scrolled');
            } else {
                nav?.classList.remove('scrolled');
            }

            lastScrollTop = scrollTop;
        });

        // Intersection observer for fade-in animations
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('fade-in');
                }
            });
        }, observerOptions);

        // Observe elements that should fade in
        setTimeout(() => {
            document.querySelectorAll('.fade-on-scroll').forEach(el => observer.observe(el));
        }, 100);
    }

    toggleTheme() {
        const isDark = document.documentElement.classList.toggle('dark-mode');
        localStorage.setItem('theme', isDark ? 'dark' : 'light');

        // Update theme icon
        const themeIcon = document.querySelector('.theme-icon');
        if (themeIcon) {
            themeIcon.textContent = isDark ? '☀️' : '🌓';
        }

        // Dispatch theme change event
        window.dispatchEvent(new CustomEvent('themeChange', {
            detail: { isDark, theme: isDark ? 'dark' : 'light' }
        }));
    }

    isDarkMode() {
        const saved = localStorage.getItem('theme');
        if (saved) return saved === 'dark';
        return window.matchMedia('(prefers-color-scheme: dark)').matches;
    }

    prefersReducedMotion() {
        return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    }

    isMobileDevice() {
        return window.innerWidth < 768 ||
            /Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    }

    trapFocusInMobileMenu() {
        const menu = document.querySelector('.nav-menu');
        const focusableElements = menu?.querySelectorAll(
            'a, button, [tabindex]:not([tabindex="-1"])'
        );

        if (!focusableElements || focusableElements.length === 0) return;

        const firstElement = focusableElements[0];
        const lastElement = focusableElements[focusableElements.length - 1];

        const handleTabKey = (e) => {
            if (e.key !== 'Tab') return;

            if (e.shiftKey) {
                if (document.activeElement === firstElement) {
                    e.preventDefault();
                    lastElement.focus();
                }
            } else {
                if (document.activeElement === lastElement) {
                    e.preventDefault();
                    firstElement.focus();
                }
            }
        };

        document.addEventListener('keydown', handleTabKey);
        firstElement.focus();

        // Remove listener when menu closes
        const removeListener = () => {
            document.removeEventListener('keydown', handleTabKey);
        };

        setTimeout(removeListener, 100); // Simple cleanup
    }

    injectStyles() {
        if (document.getElementById('master-nav-styles')) return;

        const styles = document.createElement('style');
        styles.id = 'master-nav-styles';
        styles.textContent = `
            /* Master Navigation Styles v2.0 */
            .master-nav {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-bottom: 1px solid rgba(0, 0, 0, 0.1);
                z-index: 1000;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }

            .master-nav.scrolled {
                box-shadow: 0 2px 20px rgba(0, 0, 0, 0.1);
            }

            .master-nav.nav-hidden {
                transform: translateY(-100%);
            }

            .nav-container {
                max-width: 1200px;
                margin: 0 auto;
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 0 2rem;
                min-height: 70px;
            }

            .nav-brand {
                display: flex;
                align-items: center;
            }

            .brand-link {
                display: flex;
                align-items: center;
                text-decoration: none;
                color: #1a2332;
                font-weight: 700;
                transition: color 0.3s ease;
            }

            .phi-symbol {
                font-size: 2rem;
                color: #FFD700;
                margin-right: 0.5rem;
                font-weight: 300;
            }

            .brand-text {
                font-size: 1.5rem;
                font-weight: 700;
                letter-spacing: -0.02em;
            }

            .brand-subtitle {
                font-size: 0.7rem;
                color: #666;
                margin-left: 0.5rem;
                opacity: 0.8;
                font-weight: 400;
            }

            .nav-content {
                display: flex;
                align-items: center;
                gap: 2rem;
            }

            .nav-menu {
                display: flex;
                list-style: none;
                margin: 0;
                padding: 0;
                gap: 0.5rem;
                align-items: center;
            }

            .nav-item {
                position: relative;
            }

            .nav-link {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.75rem 1rem;
                text-decoration: none;
                color: #4a5568;
                font-weight: 500;
                border-radius: 8px;
                transition: all 0.3s ease;
                white-space: nowrap;
                cursor: pointer;
            }

            .nav-link:hover,
            .nav-link:focus {
                background: rgba(102, 126, 234, 0.1);
                color: #667eea;
                transform: translateY(-1px);
            }

            .nav-link.current-page,
            .nav-item.active .nav-link {
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
            }

            .dropdown-arrow {
                font-size: 0.75rem;
                transition: transform 0.3s ease;
            }

            .dropdown.open .dropdown-arrow {
                transform: rotate(180deg);
            }

            .dropdown-menu {
                position: absolute;
                top: 100%;
                left: 0;
                background: white;
                border: 1px solid rgba(0, 0, 0, 0.1);
                border-radius: 12px;
                padding: 0.5rem;
                min-width: 220px;
                box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
                opacity: 0;
                visibility: hidden;
                transform: translateY(-10px);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                z-index: 1001;
                list-style: none;
                margin: 0;
            }

            .dropdown.open .dropdown-menu {
                opacity: 1;
                visibility: visible;
                transform: translateY(0);
            }

            .dropdown-link {
                display: block;
                padding: 0.75rem 1rem;
                text-decoration: none;
                color: #4a5568;
                border-radius: 8px;
                transition: all 0.3s ease;
                font-weight: 500;
            }

            .dropdown-link:hover,
            .dropdown-link:focus {
                background: rgba(102, 126, 234, 0.1);
                color: #667eea;
                transform: translateX(4px);
            }

            .dropdown-link.active {
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
            }

            .nav-actions {
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }

            .theme-toggle,
            .search-toggle {
                background: none;
                border: 1px solid rgba(0, 0, 0, 0.1);
                color: #4a5568;
                width: 42px;
                height: 42px;
                border-radius: 50%;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.3s ease;
                font-size: 1.1rem;
            }

            .theme-toggle:hover,
            .search-toggle:hover {
                background: rgba(102, 126, 234, 0.1);
                border-color: #667eea;
                color: #667eea;
                transform: translateY(-1px);
            }

            .mobile-menu-toggle {
                display: none;
                flex-direction: column;
                background: none;
                border: none;
                cursor: pointer;
                padding: 0.5rem;
                gap: 3px;
            }

            .hamburger-line {
                width: 24px;
                height: 3px;
                background: #4a5568;
                border-radius: 2px;
                transition: all 0.3s ease;
            }

            .mobile-menu-toggle.active .hamburger-line:nth-child(1) {
                transform: rotate(45deg) translate(6px, 6px);
            }

            .mobile-menu-toggle.active .hamburger-line:nth-child(2) {
                opacity: 0;
            }

            .mobile-menu-toggle.active .hamburger-line:nth-child(3) {
                transform: rotate(-45deg) translate(6px, -6px);
            }

            /* Chat FAB Styles */
            .chat-fab-container {
                position: fixed;
                bottom: 24px;
                right: 24px;
                z-index: 999;
            }

            .chat-fab {
                width: 60px;
                height: 60px;
                background: linear-gradient(135deg, #667eea, #764ba2);
                border: none;
                border-radius: 50%;
                cursor: pointer;
                box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4);
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 1.5rem;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                overflow: hidden;
            }

            .chat-fab:hover {
                transform: translateY(-2px) scale(1.05);
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
            }

            .chat-fab:active {
                transform: translateY(0) scale(0.95);
            }

            .fab-pulse {
                position: absolute;
                top: -2px;
                left: -2px;
                right: -2px;
                bottom: -2px;
                border: 2px solid rgba(102, 126, 234, 0.6);
                border-radius: 50%;
                animation: pulse 2s infinite;
                opacity: 0;
            }

            .fab-tooltip {
                position: absolute;
                bottom: 70px;
                right: 0;
                background: rgba(0, 0, 0, 0.8);
                color: white;
                padding: 0.5rem 0.75rem;
                border-radius: 6px;
                font-size: 0.875rem;
                white-space: nowrap;
                opacity: 0;
                transform: translateY(10px);
                transition: all 0.3s ease;
                pointer-events: none;
            }

            .fab-tooltip::after {
                content: '';
                position: absolute;
                top: 100%;
                right: 16px;
                border: 4px solid transparent;
                border-top-color: rgba(0, 0, 0, 0.8);
            }

            .fab-tooltip.show {
                opacity: 1;
                transform: translateY(0);
            }

            /* Search Modal */
            .search-modal {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0, 0, 0, 0.8);
                z-index: 10000;
                display: flex;
                align-items: flex-start;
                justify-content: center;
                padding: 5rem 2rem 2rem;
                opacity: 0;
                visibility: hidden;
                transition: all 0.3s ease;
            }

            .search-modal.open {
                opacity: 1;
                visibility: visible;
            }

            .search-modal-content {
                background: white;
                border-radius: 16px;
                width: 100%;
                max-width: 600px;
                max-height: 70vh;
                display: flex;
                flex-direction: column;
                overflow: hidden;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                transform: translateY(-20px);
                transition: transform 0.3s ease;
            }

            .search-modal.open .search-modal-content {
                transform: translateY(0);
            }

            .search-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 1.5rem 2rem;
                border-bottom: 1px solid rgba(0, 0, 0, 0.1);
            }

            .search-header h2 {
                margin: 0;
                font-size: 1.25rem;
                font-weight: 600;
            }

            .search-close {
                background: none;
                border: none;
                font-size: 1.5rem;
                cursor: pointer;
                color: #666;
                padding: 0.5rem;
                border-radius: 50%;
                transition: background 0.3s ease;
            }

            .search-close:hover {
                background: rgba(0, 0, 0, 0.05);
            }

            .search-input-wrapper {
                position: relative;
                padding: 1.5rem 2rem;
                border-bottom: 1px solid rgba(0, 0, 0, 0.1);
            }

            .search-input-wrapper input {
                width: 100%;
                border: 2px solid rgba(0, 0, 0, 0.1);
                border-radius: 12px;
                padding: 1rem 1rem 1rem 3rem;
                font-size: 1.125rem;
                outline: none;
                transition: border-color 0.3s ease;
            }

            .search-input-wrapper input:focus {
                border-color: #667eea;
            }

            .search-icon {
                position: absolute;
                left: 3rem;
                top: 50%;
                transform: translateY(-50%);
                color: #666;
                font-size: 1.125rem;
                pointer-events: none;
            }

            .search-results {
                flex: 1;
                overflow-y: auto;
                padding: 1rem;
            }

            .search-result {
                border-radius: 8px;
                overflow: hidden;
                margin-bottom: 0.5rem;
                transition: background 0.3s ease;
            }

            .search-result:hover {
                background: rgba(102, 126, 234, 0.05);
            }

            .search-result-link {
                display: block;
                padding: 1rem;
                text-decoration: none;
                color: inherit;
            }

            .search-result-title {
                font-weight: 600;
                color: #1a2332;
                margin-bottom: 0.25rem;
            }

            .search-result-description {
                color: #666;
                font-size: 0.9rem;
                margin-bottom: 0.25rem;
            }

            .search-result-category {
                color: #667eea;
                font-size: 0.8rem;
                font-weight: 500;
            }

            .search-empty {
                text-align: center;
                color: #666;
                padding: 2rem;
                font-style: italic;
            }

            /* Dark mode */
            .dark-mode .master-nav {
                background: rgba(26, 32, 44, 0.95);
                border-bottom-color: rgba(255, 255, 255, 0.1);
            }

            .dark-mode .brand-link,
            .dark-mode .nav-link {
                color: #f7fafc;
            }

            .dark-mode .nav-link:hover {
                background: rgba(102, 126, 234, 0.2);
            }

            .dark-mode .dropdown-menu {
                background: #2d3748;
                border-color: rgba(255, 255, 255, 0.1);
            }

            .dark-mode .dropdown-link {
                color: #e2e8f0;
            }

            .dark-mode .theme-toggle,
            .dark-mode .search-toggle {
                border-color: rgba(255, 255, 255, 0.2);
                color: #e2e8f0;
            }

            .dark-mode .search-modal-content {
                background: #2d3748;
                color: #f7fafc;
            }

            .dark-mode .search-input-wrapper input {
                background: #1a202c;
                border-color: rgba(255, 255, 255, 0.2);
                color: #f7fafc;
            }

            /* Responsive Design */
            @media (max-width: 768px) {
                .nav-container {
                    padding: 0 1rem;
                }

                .mobile-menu-toggle {
                    display: flex;
                }

                .nav-content {
                    gap: 1rem;
                }

                .nav-actions {
                    gap: 0.25rem;
                }

                .brand-subtitle {
                    display: none;
                }

                .nav-menu {
                    position: fixed;
                    top: 70px;
                    left: 0;
                    right: 0;
                    background: white;
                    border-top: 1px solid rgba(0, 0, 0, 0.1);
                    flex-direction: column;
                    padding: 1rem;
                    transform: translateY(-100%);
                    opacity: 0;
                    visibility: hidden;
                    transition: all 0.3s ease;
                    gap: 0;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
                }

                .nav-menu.mobile-open {
                    transform: translateY(0);
                    opacity: 1;
                    visibility: visible;
                }

                .nav-item {
                    width: 100%;
                }

                .nav-link {
                    justify-content: flex-start;
                    padding: 1rem;
                    border-radius: 8px;
                    margin-bottom: 0.25rem;
                }

                .dropdown-menu {
                    position: static;
                    opacity: 1;
                    visibility: visible;
                    transform: none;
                    box-shadow: none;
                    border: none;
                    background: rgba(0, 0, 0, 0.05);
                    margin-left: 1rem;
                    margin-top: 0.5rem;
                }

                .dropdown.open .dropdown-menu {
                    display: block;
                }

                .dropdown-menu:not(.dropdown.open .dropdown-menu) {
                    display: none;
                }

                .chat-fab-container {
                    bottom: 20px;
                    right: 20px;
                }

                .chat-fab {
                    width: 56px;
                    height: 56px;
                    font-size: 1.25rem;
                }

                .dark-mode .nav-menu {
                    background: #2d3748;
                }

                .search-modal {
                    padding: 2rem 1rem 1rem;
                }

                .search-modal-content {
                    max-height: 80vh;
                }

                .search-header,
                .search-input-wrapper {
                    padding-left: 1.5rem;
                    padding-right: 1.5rem;
                }
            }

            @media (max-width: 480px) {
                .nav-container {
                    min-height: 60px;
                }

                .brand-text {
                    font-size: 1.25rem;
                }

                .phi-symbol {
                    font-size: 1.5rem;
                    margin-right: 0.25rem;
                }
            }

            /* Animations */
            @keyframes pulse {
                0% {
                    opacity: 0;
                    transform: scale(1);
                }
                50% {
                    opacity: 1;
                }
                100% {
                    opacity: 0;
                    transform: scale(1.1);
                }
            }

            @keyframes slideInUp {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .fade-in {
                animation: fadeIn 0.6s ease forwards;
            }

            @keyframes fadeIn {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            /* Skip to content link */
            .skip-to-content:focus {
                top: 6px !important;
            }

            /* Focus indicators */
            .nav-link:focus-visible,
            .dropdown-link:focus-visible,
            .theme-toggle:focus-visible,
            .search-toggle:focus-visible,
            .chat-fab:focus-visible {
                outline: 2px solid #FFD700;
                outline-offset: 2px;
            }

            /* Reduced motion */
            @media (prefers-reduced-motion: reduce) {
                * {
                    animation: none !important;
                    transition: none !important;
                }
            }

            /* Print styles */
            @media print {
                .master-nav,
                .chat-fab-container,
                .search-modal {
                    display: none !important;
                }
            }
        `;

        document.head.appendChild(styles);
    }

    // Initialize theme on load
    initializeTheme() {
        const savedTheme = localStorage.getItem('theme');
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        const isDark = savedTheme ? savedTheme === 'dark' : prefersDark;

        if (isDark) {
            document.documentElement.classList.add('dark-mode');
        }

        // Update theme icon
        const themeIcon = document.querySelector('.theme-icon');
        if (themeIcon) {
            themeIcon.textContent = isDark ? '☀️' : '🌓';
        }
    }
}

// Auto-initialize when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.masterNavigation = new MasterNavigation();
        window.masterNavigation.initializeTheme();
    });
} else {
    window.masterNavigation = new MasterNavigation();
    window.masterNavigation.initializeTheme();
}

// Make available globally
window.MasterNavigation = MasterNavigation;