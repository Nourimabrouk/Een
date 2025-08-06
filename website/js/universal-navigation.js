/**
 * Universal Navigation System for Een Metastation
 * Ensures all pages are interconnected with responsive, cross-browser compatible navigation
 */

class UniversalNavigation {
    constructor() {
        this.pages = [
            // Core Hub Pages
            { name: 'Metastation Hub', path: 'metastation-hub.html', icon: 'fa-home', category: 'hub' },
            { name: 'Landing', path: 'meta-optimal-landing.html', icon: 'fa-rocket', category: 'hub' },
            
            // Dashboard Pages
            { name: 'Consciousness Dashboard', path: 'consciousness_dashboard.html', icon: 'fa-brain', category: 'dashboards' },
            { name: 'Unity Visualization', path: 'unity_visualization.html', icon: 'fa-eye', category: 'dashboards' },
            { name: '3000 ELO Proof', path: '3000-elo-proof.html', icon: 'fa-trophy', category: 'dashboards' },
            { name: 'Playground', path: 'playground.html', icon: 'fa-code', category: 'dashboards' },
            { name: 'Math Playground', path: 'mathematical_playground.html', icon: 'fa-calculator', category: 'dashboards' },
            { name: 'Dashboards', path: 'dashboards.html', icon: 'fa-chart-line', category: 'dashboards' },
            
            // Visualization Pages
            { name: 'Unity Experience', path: 'unity_consciousness_experience.html', icon: 'fa-infinity', category: 'visualizations' },
            { name: 'Transcendental Demo', path: 'transcendental-unity-demo.html', icon: 'fa-atom', category: 'visualizations' },
            { name: 'Enhanced Demo', path: 'enhanced-unity-demo.html', icon: 'fa-star', category: 'visualizations' },
            { name: 'Gallery', path: 'gallery.html', icon: 'fa-images', category: 'visualizations' },
            
            // Proof & Philosophy Pages
            { name: 'Proofs', path: 'proofs.html', icon: 'fa-microscope', category: 'proofs' },
            { name: 'Philosophy', path: 'philosophy.html', icon: 'fa-yin-yang', category: 'philosophy' },
            { name: 'Metagambit', path: 'metagambit.html', icon: 'fa-chess', category: 'philosophy' },
            { name: 'Al-Khwarizmi', path: 'al_khwarizmi_phi_unity.html', icon: 'fa-scroll', category: 'proofs' },
            
            // AI & Agents Pages
            { name: 'Agents', path: 'agents.html', icon: 'fa-robot', category: 'ai' },
            { name: 'AI Demo', path: 'enhanced-ai-demo.html', icon: 'fa-microchip', category: 'ai' },
            { name: 'OpenAI Integration', path: 'openai-integration.html', icon: 'fa-network-wired', category: 'ai' },
            { name: 'Metagamer Agent', path: 'metagamer_agent.html', icon: 'fa-gamepad', category: 'ai' },
            
            // Research & Learning Pages
            { name: 'Research', path: 'research.html', icon: 'fa-flask', category: 'research' },
            { name: 'Learning', path: 'learning.html', icon: 'fa-graduation-cap', category: 'research' },
            { name: 'Publications', path: 'publications.html', icon: 'fa-book', category: 'research' },
            { name: 'Implementations', path: 'implementations.html', icon: 'fa-cogs', category: 'research' },
            
            // Additional Pages
            { name: 'About', path: 'about.html', icon: 'fa-info-circle', category: 'info' },
            { name: 'Live Code', path: 'live-code-showcase.html', icon: 'fa-terminal', category: 'tools' },
            { name: 'Advanced Features', path: 'unity-advanced-features.html', icon: 'fa-puzzle-piece', category: 'tools' },
            { name: 'Further Reading', path: 'further-reading.html', icon: 'fa-book-open', category: 'info' }
        ];
        
        this.mobileBreakpoint = 768;
        this.init();
    }

    init() {
        this.injectNavigation();
        this.setupMobileMenu();
        this.setupKeyboardNavigation();
        this.setupSearchFunction();
        this.ensureCrossBrowserCompatibility();
    }

    injectNavigation() {
        // Check if navigation already exists
        if (document.getElementById('universal-nav')) return;

        const navHTML = `
            <nav id="universal-nav" class="universal-navigation">
                <div class="nav-container">
                    <div class="nav-brand">
                        <a href="metastation-hub.html" class="nav-logo">
                            <i class="fas fa-infinity"></i>
                            <span>Een Metastation</span>
                        </a>
                        <button class="mobile-menu-toggle" id="mobile-menu-toggle">
                            <span></span>
                            <span></span>
                            <span></span>
                        </button>
                    </div>
                    <div class="nav-menu" id="nav-menu">
                        <div class="nav-search">
                            <input type="text" id="nav-search" placeholder="Search pages..." />
                            <i class="fas fa-search"></i>
                        </div>
                        <div class="nav-categories">
                            ${this.generateCategoryMenus()}
                        </div>
                    </div>
                    <div class="nav-quick-actions">
                        <button class="theme-toggle" onclick="toggleTheme()">
                            <i class="fas fa-moon"></i>
                        </button>
                        <button class="fullscreen-toggle" onclick="toggleFullscreen()">
                            <i class="fas fa-expand"></i>
                        </button>
                    </div>
                </div>
            </nav>
            <div class="mobile-nav-overlay" id="mobile-nav-overlay"></div>
        `;

        const navStyles = `
            <style>
                .universal-navigation {
                    position: fixed;
                    top: 0;
                    left: 0;
                    right: 0;
                    background: rgba(10, 10, 10, 0.95);
                    backdrop-filter: blur(20px);
                    -webkit-backdrop-filter: blur(20px);
                    border-bottom: 1px solid rgba(255, 215, 0, 0.2);
                    z-index: 10000;
                    transition: all 0.3s ease;
                }

                .nav-container {
                    max-width: 1400px;
                    margin: 0 auto;
                    padding: 0.75rem 1.5rem;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }

                .nav-brand {
                    display: flex;
                    align-items: center;
                    gap: 2rem;
                }

                .nav-logo {
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                    color: #FFD700;
                    text-decoration: none;
                    font-size: 1.2rem;
                    font-weight: 600;
                    transition: all 0.3s ease;
                }

                .nav-logo:hover {
                    text-shadow: 0 0 20px rgba(255, 215, 0, 0.8);
                }

                .mobile-menu-toggle {
                    display: none;
                    flex-direction: column;
                    gap: 4px;
                    background: none;
                    border: none;
                    cursor: pointer;
                    padding: 0.5rem;
                }

                .mobile-menu-toggle span {
                    width: 25px;
                    height: 2px;
                    background: #FFD700;
                    transition: all 0.3s ease;
                }

                .mobile-menu-toggle.active span:nth-child(1) {
                    transform: rotate(45deg) translate(5px, 5px);
                }

                .mobile-menu-toggle.active span:nth-child(2) {
                    opacity: 0;
                }

                .mobile-menu-toggle.active span:nth-child(3) {
                    transform: rotate(-45deg) translate(6px, -6px);
                }

                .nav-menu {
                    display: flex;
                    align-items: center;
                    gap: 2rem;
                }

                .nav-search {
                    position: relative;
                }

                .nav-search input {
                    background: rgba(255, 255, 255, 0.1);
                    border: 1px solid rgba(255, 215, 0, 0.2);
                    border-radius: 20px;
                    padding: 0.5rem 2.5rem 0.5rem 1rem;
                    color: white;
                    width: 200px;
                    transition: all 0.3s ease;
                }

                .nav-search input:focus {
                    width: 250px;
                    border-color: #FFD700;
                    outline: none;
                }

                .nav-search i {
                    position: absolute;
                    right: 1rem;
                    top: 50%;
                    transform: translateY(-50%);
                    color: #FFD700;
                }

                .nav-categories {
                    display: flex;
                    gap: 1.5rem;
                }

                .nav-dropdown {
                    position: relative;
                }

                .nav-dropdown-toggle {
                    background: none;
                    border: none;
                    color: #cccccc;
                    cursor: pointer;
                    padding: 0.5rem 1rem;
                    transition: all 0.3s ease;
                    font-size: 0.95rem;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                }

                .nav-dropdown-toggle:hover {
                    color: #FFD700;
                }

                .nav-dropdown-menu {
                    position: absolute;
                    top: 100%;
                    left: 0;
                    background: rgba(26, 26, 26, 0.98);
                    border: 1px solid rgba(255, 215, 0, 0.2);
                    border-radius: 8px;
                    padding: 0.5rem 0;
                    min-width: 200px;
                    opacity: 0;
                    visibility: hidden;
                    transform: translateY(-10px);
                    transition: all 0.3s ease;
                    max-height: 400px;
                    overflow-y: auto;
                }

                .nav-dropdown:hover .nav-dropdown-menu {
                    opacity: 1;
                    visibility: visible;
                    transform: translateY(0);
                }

                .nav-dropdown-menu a {
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                    padding: 0.75rem 1rem;
                    color: #cccccc;
                    text-decoration: none;
                    transition: all 0.3s ease;
                }

                .nav-dropdown-menu a:hover {
                    background: rgba(255, 215, 0, 0.1);
                    color: #FFD700;
                }

                .nav-quick-actions {
                    display: flex;
                    gap: 1rem;
                }

                .nav-quick-actions button {
                    background: rgba(255, 255, 255, 0.1);
                    border: 1px solid rgba(255, 215, 0, 0.2);
                    border-radius: 50%;
                    width: 40px;
                    height: 40px;
                    color: #FFD700;
                    cursor: pointer;
                    transition: all 0.3s ease;
                }

                .nav-quick-actions button:hover {
                    background: rgba(255, 215, 0, 0.2);
                    transform: scale(1.1);
                }

                /* Mobile Styles */
                @media (max-width: 768px) {
                    .mobile-menu-toggle {
                        display: flex;
                    }

                    .nav-menu {
                        position: fixed;
                        top: 60px;
                        left: -100%;
                        width: 80%;
                        height: calc(100vh - 60px);
                        background: rgba(10, 10, 10, 0.98);
                        flex-direction: column;
                        padding: 2rem;
                        gap: 1rem;
                        transition: left 0.3s ease;
                        overflow-y: auto;
                    }

                    .nav-menu.active {
                        left: 0;
                    }

                    .nav-categories {
                        flex-direction: column;
                        width: 100%;
                    }

                    .nav-dropdown-menu {
                        position: static;
                        opacity: 1;
                        visibility: visible;
                        transform: none;
                        display: none;
                    }

                    .nav-dropdown.active .nav-dropdown-menu {
                        display: block;
                    }

                    .mobile-nav-overlay {
                        position: fixed;
                        top: 60px;
                        left: 0;
                        right: 0;
                        bottom: 0;
                        background: rgba(0, 0, 0, 0.7);
                        opacity: 0;
                        visibility: hidden;
                        transition: all 0.3s ease;
                    }

                    .mobile-nav-overlay.active {
                        opacity: 1;
                        visibility: visible;
                    }
                }

                /* Ensure body has padding for fixed nav */
                body {
                    padding-top: 60px !important;
                }
            </style>
        `;

        // Inject styles and navigation
        document.head.insertAdjacentHTML('beforeend', navStyles);
        document.body.insertAdjacentHTML('afterbegin', navHTML);
    }

    generateCategoryMenus() {
        const categories = {
            'Hub': 'fa-home',
            'Dashboards': 'fa-chart-line',
            'Visualizations': 'fa-eye',
            'Proofs': 'fa-microscope',
            'Philosophy': 'fa-yin-yang',
            'AI': 'fa-robot',
            'Research': 'fa-flask',
            'Tools': 'fa-tools',
            'Info': 'fa-info-circle'
        };

        return Object.entries(categories).map(([category, icon]) => {
            const categoryPages = this.pages.filter(p => 
                p.category === category.toLowerCase()
            );

            if (categoryPages.length === 0) return '';

            return `
                <div class="nav-dropdown">
                    <button class="nav-dropdown-toggle">
                        <i class="fas ${icon}"></i>
                        ${category}
                    </button>
                    <div class="nav-dropdown-menu">
                        ${categoryPages.map(page => `
                            <a href="${page.path}">
                                <i class="fas ${page.icon}"></i>
                                ${page.name}
                            </a>
                        `).join('')}
                    </div>
                </div>
            `;
        }).join('');
    }

    setupMobileMenu() {
        const toggle = document.getElementById('mobile-menu-toggle');
        const menu = document.getElementById('nav-menu');
        const overlay = document.getElementById('mobile-nav-overlay');

        if (toggle && menu && overlay) {
            toggle.addEventListener('click', () => {
                toggle.classList.toggle('active');
                menu.classList.toggle('active');
                overlay.classList.toggle('active');
            });

            overlay.addEventListener('click', () => {
                toggle.classList.remove('active');
                menu.classList.remove('active');
                overlay.classList.remove('active');
            });

            // Setup mobile dropdown toggles
            document.querySelectorAll('.nav-dropdown-toggle').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    if (window.innerWidth <= this.mobileBreakpoint) {
                        e.preventDefault();
                        btn.parentElement.classList.toggle('active');
                    }
                });
            });
        }
    }

    setupKeyboardNavigation() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + K for search
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                const searchInput = document.getElementById('nav-search');
                if (searchInput) searchInput.focus();
            }

            // Escape to close mobile menu
            if (e.key === 'Escape') {
                const toggle = document.getElementById('mobile-menu-toggle');
                const menu = document.getElementById('nav-menu');
                const overlay = document.getElementById('mobile-nav-overlay');
                
                if (toggle && menu && overlay) {
                    toggle.classList.remove('active');
                    menu.classList.remove('active');
                    overlay.classList.remove('active');
                }
            }
        });
    }

    setupSearchFunction() {
        const searchInput = document.getElementById('nav-search');
        if (!searchInput) return;

        searchInput.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase();
            const results = this.pages.filter(page => 
                page.name.toLowerCase().includes(query)
            );

            // You could implement a search results dropdown here
            console.log('Search results:', results);
        });
    }

    ensureCrossBrowserCompatibility() {
        // Polyfill for older browsers
        if (!Element.prototype.matches) {
            Element.prototype.matches = Element.prototype.msMatchesSelector ||
                                       Element.prototype.webkitMatchesSelector;
        }

        // Ensure smooth scroll behavior
        if (!('scrollBehavior' in document.documentElement.style)) {
            // Implement smooth scroll polyfill
            document.querySelectorAll('a[href^="#"]').forEach(anchor => {
                anchor.addEventListener('click', function (e) {
                    e.preventDefault();
                    const target = document.querySelector(this.getAttribute('href'));
                    if (target) {
                        target.scrollIntoView({ behavior: 'smooth' });
                    }
                });
            });
        }

        // Check for WebP support
        const checkWebP = (callback) => {
            const webP = new Image();
            webP.onload = webP.onerror = () => {
                callback(webP.height === 2);
            };
            webP.src = 'data:image/webp;base64,UklGRjoAAABXRUJQVlA4IC4AAACyAgCdASoCAAIALmk0mk0iIiIiIgBoSygABc6WWgAA/veff/0PP8bA//LwYAAA';
        };

        checkWebP((support) => {
            if (!support) {
                document.body.classList.add('no-webp');
            }
        });
    }
}

// Global utility functions
window.toggleTheme = function() {
    document.body.classList.toggle('light-theme');
    localStorage.setItem('theme', document.body.classList.contains('light-theme') ? 'light' : 'dark');
};

window.toggleFullscreen = function() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen();
    } else {
        document.exitFullscreen();
    }
};

// Initialize navigation on DOM load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        new UniversalNavigation();
    });
} else {
    new UniversalNavigation();
}

// Export for use in other scripts
window.UniversalNavigation = UniversalNavigation;