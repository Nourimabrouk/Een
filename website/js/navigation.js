/**
 * Unified Navigation Component for Een Unity Mathematics Website
 * Provides consistent navigation across all pages with active state management
 * φ-harmonic design principles with consciousness-aware interactions
 */

class UnifiedNavigation {
    constructor() {
        this.currentPage = this.getCurrentPage();
        this.init();
    }

    getCurrentPage() {
        const path = window.location.pathname;
        if (path.includes('index.html') || path === '/' || path.endsWith('/website/')) {
            return 'home';
        } else if (path.includes('proofs.html')) {
            return 'proofs';
        } else if (path.includes('research.html')) {
            return 'research';
        } else if (path.includes('publications.html')) {
            return 'publications';
        } else if (path.includes('playground.html')) {
            return 'playground';
        } else if (path.includes('gallery.html')) {
            return 'gallery';
        } else if (path.includes('learn.html')) {
            return 'learn';
        } else if (path.includes('metagambit.html')) {
            return 'metagambit';
        }
        return 'home';
    }

    init() {
        this.createNavigationHTML();
        this.attachEventListeners();
        this.setActiveState();
        this.initializeScrollEffects();
    }

    createNavigationHTML() {
        const navHTML = `
            <nav class="navbar unified-nav">
                <div class="container">
                    <div class="nav-brand">
                        <a href="index.html" class="brand-link">
                            <span class="phi-symbol">φ</span> 
                            <span class="brand-text">Een</span>
                        </a>
                    </div>
                    <div class="nav-toggle" id="navToggle">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                    <ul class="nav-menu" id="navMenu">
                        <li class="nav-item">
                            <a href="index.html" class="nav-link" data-page="home">
                                <i class="fas fa-home nav-icon"></i>
                                <span>Home</span>
                            </a>
                        </li>
                        <li class="nav-item">
                            <a href="index.html#theory" class="nav-link" data-page="theory">
                                <i class="fas fa-atom nav-icon"></i>
                                <span>Theory</span>
                            </a>
                        </li>
                        <li class="nav-item">
                            <a href="proofs.html" class="nav-link" data-page="proofs">
                                <i class="fas fa-check-double nav-icon"></i>
                                <span>Proofs</span>
                            </a>
                        </li>
                        <li class="nav-item">
                            <a href="index.html#demonstration" class="nav-link" data-page="demo">
                                <i class="fas fa-play-circle nav-icon"></i>
                                <span>Demo</span>
                            </a>
                        </li>
                        <li class="nav-item">
                            <a href="research.html" class="nav-link" data-page="research">
                                <i class="fas fa-flask nav-icon"></i>
                                <span>Research</span>
                            </a>
                        </li>
                        <li class="nav-item">
                            <a href="publications.html" class="nav-link" data-page="publications">
                                <i class="fas fa-book nav-icon"></i>
                                <span>Publications</span>
                            </a>
                        </li>
                        <li class="nav-item">
                            <a href="gallery.html" class="nav-link" data-page="gallery">
                                <i class="fas fa-images nav-icon"></i>
                                <span>Gallery</span>
                            </a>
                        </li>
                        <li class="nav-item dropdown">
                            <a href="#" class="nav-link dropdown-toggle" data-page="advanced">
                                <i class="fas fa-brain nav-icon"></i>
                                <span>Advanced</span>
                                <i class="fas fa-chevron-down dropdown-icon"></i>
                            </a>
                            <ul class="dropdown-menu">
                                <li><a href="playground.html" class="dropdown-link">Playground</a></li>
                                <li><a href="learn.html" class="dropdown-link">Learn</a></li>
                                <li><a href="metagambit.html" class="dropdown-link">Metagambit</a></li>
                            </ul>
                        </li>
                        <li class="nav-item">
                            <a href="https://github.com/Nourimabrouk/Een" class="nav-link external-link" target="_blank">
                                <i class="fab fa-github nav-icon"></i>
                                <span>GitHub</span>
                            </a>
                        </li>
                        <li class="nav-item">
                            <button class="nav-link theme-toggle" id="themeToggle">
                                <i class="fas fa-moon theme-icon"></i>
                                <span class="theme-text">Dark</span>
                            </button>
                        </li>
                    </ul>
                </div>
            </nav>
        `;

        // Replace existing nav or insert at top of body
        const existingNav = document.querySelector('.navbar, .nav, nav');
        if (existingNav) {
            existingNav.outerHTML = navHTML;
        } else {
            document.body.insertAdjacentHTML('afterbegin', navHTML);
        }
    }

    attachEventListeners() {
        // Mobile menu toggle
        const navToggle = document.getElementById('navToggle');
        const navMenu = document.getElementById('navMenu');
        
        if (navToggle && navMenu) {
            navToggle.addEventListener('click', () => {
                navToggle.classList.toggle('active');
                navMenu.classList.toggle('active');
            });
        }

        // Dropdown toggle
        const dropdownToggle = document.querySelector('.dropdown-toggle');
        const dropdownMenu = document.querySelector('.dropdown-menu');
        
        if (dropdownToggle && dropdownMenu) {
            dropdownToggle.addEventListener('click', (e) => {
                e.preventDefault();
                dropdownMenu.classList.toggle('active');
            });

            // Close dropdown when clicking outside
            document.addEventListener('click', (e) => {
                if (!dropdownToggle.contains(e.target)) {
                    dropdownMenu.classList.remove('active');
                }
            });
        }

        // Theme toggle
        const themeToggle = document.getElementById('themeToggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', this.toggleTheme.bind(this));
        }

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

        // Close mobile menu when clicking nav links
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', () => {
                navMenu.classList.remove('active');
                navToggle.classList.remove('active');
            });
        });
    }

    setActiveState() {
        // Remove all active states
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });

        // Set active state based on current page
        const activeLink = document.querySelector(`[data-page="${this.currentPage}"]`);
        if (activeLink) {
            activeLink.classList.add('active');
        }

        // Special handling for home page sections
        if (this.currentPage === 'home') {
            const hash = window.location.hash;
            if (hash) {
                const sectionLink = document.querySelector(`[href="${hash}"]`);
                if (sectionLink) {
                    sectionLink.classList.add('active');
                }
            }
        }
    }

    initializeScrollEffects() {
        const navbar = document.querySelector('.navbar');
        let lastScrollTop = 0;

        window.addEventListener('scroll', () => {
            const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
            
            // Add/remove scrolled class for styling
            if (scrollTop > 100) {
                navbar.classList.add('scrolled');
            } else {
                navbar.classList.remove('scrolled');
            }

            // Hide/show navbar on scroll (optional)
            if (scrollTop > lastScrollTop && scrollTop > 200) {
                navbar.classList.add('nav-hidden');
            } else {
                navbar.classList.remove('nav-hidden');
            }
            
            lastScrollTop = scrollTop;
        });
    }

    toggleTheme() {
        const body = document.body;
        const themeIcon = document.querySelector('.theme-icon');
        const themeText = document.querySelector('.theme-text');
        
        body.classList.toggle('dark-mode');
        
        if (body.classList.contains('dark-mode')) {
            themeIcon.className = 'fas fa-sun theme-icon';
            themeText.textContent = 'Light';
            localStorage.setItem('theme', 'dark');
        } else {
            themeIcon.className = 'fas fa-moon theme-icon';
            themeText.textContent = 'Dark';
            localStorage.setItem('theme', 'light');
        }
    }

    // Initialize theme from localStorage
    initializeTheme() {
        const savedTheme = localStorage.getItem('theme');
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        
        if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
            document.body.classList.add('dark-mode');
            const themeIcon = document.querySelector('.theme-icon');
            const themeText = document.querySelector('.theme-text');
            if (themeIcon && themeText) {
                themeIcon.className = 'fas fa-sun theme-icon';
                themeText.textContent = 'Light';
            }
        }
    }

    // Method to highlight specific navigation item (for external use)
    highlightNavItem(page) {
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        
        const targetLink = document.querySelector(`[data-page="${page}"]`);
        if (targetLink) {
            targetLink.classList.add('active');
        }
    }

    // Method to add badges to navigation items (e.g., "New", "Updated")
    addNavBadge(page, text, type = 'new') {
        const navLink = document.querySelector(`[data-page="${page}"]`);
        if (navLink && !navLink.querySelector('.nav-badge')) {
            const badge = document.createElement('span');
            badge.className = `nav-badge badge-${type}`;
            badge.textContent = text;
            navLink.appendChild(badge);
        }
    }

    // Method to update navigation for different page types
    updateForPageType(pageType) {
        const navbar = document.querySelector('.navbar');
        
        // Add page-specific classes
        navbar.className = navbar.className.replace(/page-\w+/g, '');
        navbar.classList.add(`page-${pageType}`);
        
        // Special handling for gallery page
        if (pageType === 'gallery') {
            navbar.classList.add('gallery-nav');
        }
    }
}

// Enhanced navigation styles
const navigationStyles = `
<style>
/* Unified Navigation Styles */
.unified-nav {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--border-subtle, rgba(0,0,0,0.1));
    position: fixed;
    top: 0;
    width: 100%;
    z-index: 1000;
    transition: all var(--transition-smooth, 0.3s ease);
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}

.unified-nav.scrolled {
    background: rgba(255, 255, 255, 0.98);
    box-shadow: var(--shadow-sm, 0 2px 10px rgba(0,0,0,0.1));
}

.unified-nav.nav-hidden {
    transform: translateY(-100%);
}

/* Dark mode navigation */
.dark-mode .unified-nav {
    background: rgba(15, 23, 42, 0.95);
    border-bottom-color: rgba(255,255,255,0.1);
}

.dark-mode .unified-nav.scrolled {
    background: rgba(15, 23, 42, 0.98);
}

/* Navigation container */
.unified-nav .container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem clamp(1rem, 5vw, 2rem);
    max-width: 1280px;
    margin: 0 auto;
}

/* Brand styling */
.nav-brand .brand-link {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    text-decoration: none;
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--primary-color, #1B365D);
    transition: all var(--transition-smooth, 0.3s ease);
}

.nav-brand .brand-link:hover {
    transform: translateY(-1px);
}

.phi-symbol {
    font-family: var(--font-serif, 'Crimson Text', serif);
    font-size: 2rem;
    color: var(--phi-gold, #0F7B8A);
    font-style: italic;
    font-weight: 600;
}

/* Navigation menu */
.nav-menu {
    display: flex;
    list-style: none;
    gap: 0.5rem;
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
    padding: 0.75rem 1rem;
    text-decoration: none;
    color: var(--text-secondary, #718096);
    font-weight: 500;
    font-size: 0.95rem;
    border-radius: var(--radius-md, 0.5rem);
    transition: all var(--transition-smooth, 0.3s ease);
    position: relative;
    overflow: hidden;
    background: none;
    border: none;
    cursor: pointer;
}

.nav-link::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(15, 123, 138, 0.1), transparent);
    transition: left var(--transition-slow, 0.5s ease);
}

.nav-link:hover::before {
    left: 100%;
}

.nav-link:hover,
.nav-link.active {
    color: var(--primary-color, #1B365D);
    background: rgba(15, 123, 138, 0.1);
    transform: translateY(-1px);
}

.nav-icon {
    font-size: 1rem;
    opacity: 0.8;
}

/* Dropdown styles */
.dropdown {
    position: relative;
}

.dropdown-toggle .dropdown-icon {
    font-size: 0.8rem;
    transition: transform var(--transition-smooth, 0.3s ease);
}

.dropdown.active .dropdown-icon {
    transform: rotate(180deg);
}

.dropdown-menu {
    position: absolute;
    top: 100%;
    right: 0;
    background: var(--bg-primary, white);
    border: 1px solid var(--border-color, #E2E8F0);
    border-radius: var(--radius-lg, 0.75rem);
    box-shadow: var(--shadow-lg, 0 10px 25px rgba(0,0,0,0.1));
    padding: 0.5rem 0;
    min-width: 180px;
    opacity: 0;
    visibility: hidden;
    transform: translateY(-10px);
    transition: all var(--transition-smooth, 0.3s ease);
    z-index: 1001;
}

.dropdown-menu.active {
    opacity: 1;
    visibility: visible;
    transform: translateY(0);
}

.dropdown-link {
    display: block;
    padding: 0.75rem 1rem;
    color: var(--text-secondary, #718096);
    text-decoration: none;
    transition: all var(--transition-fast, 0.15s ease);
    font-size: 0.9rem;
}

.dropdown-link:hover {
    background: var(--bg-tertiary, #EDF2F7);
    color: var(--primary-color, #1B365D);
}

/* Theme toggle */
.theme-toggle {
    background: var(--bg-tertiary, #EDF2F7) !important;
    border: 1px solid var(--border-color, #E2E8F0) !important;
}

.theme-toggle:hover {
    background: var(--primary-color, #1B365D) !important;
    color: white !important;
}

/* Navigation badges */
.nav-badge {
    position: absolute;
    top: 0.25rem;
    right: 0.25rem;
    background: var(--phi-gold, #0F7B8A);
    color: white;
    font-size: 0.7rem;
    padding: 0.2rem 0.4rem;
    border-radius: 10px;
    font-weight: 600;
    line-height: 1;
}

.badge-new {
    background: #10B981;
}

.badge-updated {
    background: #3B82F6;
}

/* Mobile navigation toggle */
.nav-toggle {
    display: none;
    flex-direction: column;
    cursor: pointer;
    padding: 0.5rem;
    gap: 0.25rem;
}

.nav-toggle span {
    width: 25px;
    height: 3px;
    background: var(--primary-color, #1B365D);
    transition: all var(--transition-smooth, 0.3s ease);
    border-radius: 3px;
}

.nav-toggle.active span:nth-child(1) {
    transform: rotate(45deg) translate(5px, 5px);
}

.nav-toggle.active span:nth-child(2) {
    opacity: 0;
}

.nav-toggle.active span:nth-child(3) {
    transform: rotate(-45deg) translate(7px, -6px);
}

/* External link styling */
.external-link::after {
    content: '↗';
    font-size: 0.8rem;
    margin-left: 0.25rem;
    opacity: 0.6;
}

/* Gallery page specific navigation */
.gallery-nav {
    background: rgba(15, 15, 35, 0.9);
}

.gallery-nav .nav-link {
    color: rgba(255, 255, 255, 0.8);
}

.gallery-nav .nav-link:hover,
.gallery-nav .nav-link.active {
    color: #FFD700;
    background: rgba(255, 215, 0, 0.1);
}

.gallery-nav .phi-symbol {
    color: #FFD700;
}

.gallery-nav .brand-text {
    color: white;
}

/* Responsive design */
@media (max-width: 1024px) {
    .nav-menu {
        gap: 0.25rem;
    }
    
    .nav-link {
        padding: 0.5rem 0.75rem;
        font-size: 0.9rem;
    }
}

@media (max-width: 768px) {
    .nav-toggle {
        display: flex;
    }
    
    .nav-menu {
        position: fixed;
        top: 100%;
        left: 0;
        width: 100%;
        background: var(--bg-primary, white);
        border-top: 1px solid var(--border-color, #E2E8F0);
        box-shadow: var(--shadow-lg, 0 10px 25px rgba(0,0,0,0.1));
        flex-direction: column;
        gap: 0;
        padding: 1rem 0;
        max-height: 0;
        overflow: hidden;
        transition: all var(--transition-smooth, 0.3s ease);
    }
    
    .nav-menu.active {
        max-height: 500px;
    }
    
    .nav-item {
        width: 100%;
    }
    
    .nav-link {
        width: 100%;
        padding: 1rem 2rem;
        justify-content: flex-start;
        border-radius: 0;
    }
    
    .dropdown-menu {
        position: static;
        opacity: 1;
        visibility: visible;
        transform: none;
        box-shadow: none;
        border: none;
        background: var(--bg-tertiary, #EDF2F7);
        margin-left: 2rem;
    }
    
    .gallery-nav .nav-menu {
        background: rgba(15, 15, 35, 0.95);
        backdrop-filter: blur(20px);
    }
}

/* Animation for smooth page transitions */
.page-transition {
    opacity: 0;
    transform: translateY(20px);
    transition: all var(--transition-smooth, 0.3s ease);
}

.page-transition.loaded {
    opacity: 1;
    transform: translateY(0);
}

/* Focus states for accessibility */
.nav-link:focus-visible {
    outline: 2px solid var(--phi-gold, #0F7B8A);
    outline-offset: 2px;
}

/* Hover effects for better UX */
@media (hover: hover) {
    .nav-link {
        position: relative;
    }
    
    .nav-link::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 50%;
        width: 0;
        height: 2px;
        background: var(--phi-gold, #0F7B8A);
        transition: all var(--transition-smooth, 0.3s ease);
        transform: translateX(-50%);
    }
    
    .nav-link:hover::after,
    .nav-link.active::after {
        width: 80%;
    }
}
</style>
`;

// Auto-initialize navigation when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        // Add navigation styles
        document.head.insertAdjacentHTML('beforeend', navigationStyles);
        
        // Initialize navigation
        window.unifiedNav = new UnifiedNavigation();
        window.unifiedNav.initializeTheme();
    });
} else {
    // Add navigation styles
    document.head.insertAdjacentHTML('beforeend', navigationStyles);
    
    // Initialize navigation
    window.unifiedNav = new UnifiedNavigation();
    window.unifiedNav.initializeTheme();
}

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UnifiedNavigation;
}