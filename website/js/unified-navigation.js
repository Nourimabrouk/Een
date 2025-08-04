/**
 * Een Unity Mathematics - Unified Navigation System
 * Ensures consistent navigation across all pages
 */

class UnifiedNavigation {
    constructor() {
        this.currentPage = this.getCurrentPage();
        this.initializeNavigation();
    }
    
    getCurrentPage() {
        const path = window.location.pathname;
        const page = path.split('/').pop() || 'index.html';
        return page.replace('.html', '');
    }
    
    getNavigationHTML() {
        const navItems = [
            { href: 'index.html', label: 'Home', id: 'index' },
            { href: 'proofs.html', label: 'Proofs', id: 'proofs' },
            { href: 'playground.html', label: 'Playground', id: 'playground' },
            { href: 'gallery.html', label: 'Gallery', id: 'gallery' },
            { href: 'research.html', label: 'Research', id: 'research' },
            { href: 'metagambit.html', label: 'MetaGambit', id: 'metagambit' },
            { href: 'publications.html', label: 'Publications', id: 'publications' },
            { href: 'about.html', label: 'About', id: 'about' }
        ];
        
        const navLinksHTML = navItems.map(item => {
            const isActive = item.id === this.currentPage ? 'active' : '';
            return `<li><a href="${item.href}" class="nav-link ${isActive}">${item.label}</a></li>`;
        }).join('');
        
        return `
            <nav class="nav navbar">
                <div class="nav-container container">
                    <a href="index.html" class="nav-logo nav-brand">
                        <span class="phi-symbol">φ</span>
                        <span>Een</span>
                    </a>
                    <ul class="nav-menu">
                        ${navLinksHTML}
                        <li><a href="https://github.com/Nourimabrouk/Een" class="nav-link" target="_blank" rel="noopener">
                            <i class="fab fa-github"></i>
                        </a></li>
                    </ul>
                    <button class="nav-toggle" id="nav-toggle">
                        <span></span>
                        <span></span>
                        <span></span>
                    </button>
                </div>
            </nav>
        `;
    }
    
    initializeNavigation() {
        // Check if navigation already exists
        const existingNav = document.querySelector('nav');
        if (existingNav) {
            // Update existing navigation
            this.updateExistingNavigation(existingNav);
        } else {
            // Insert new navigation
            this.insertNavigation();
        }
        
        // Add mobile toggle functionality
        this.initializeMobileToggle();
        
        // Add scroll effects
        this.initializeScrollEffects();
    }
    
    updateExistingNavigation(nav) {
        // Update the navigation with unified structure
        nav.outerHTML = this.getNavigationHTML();
    }
    
    insertNavigation() {
        // Insert at the beginning of body
        document.body.insertAdjacentHTML('afterbegin', this.getNavigationHTML());
    }
    
    initializeMobileToggle() {
        const toggle = document.getElementById('nav-toggle');
        const menu = document.querySelector('.nav-menu');
        
        if (toggle && menu) {
            toggle.addEventListener('click', () => {
                menu.classList.toggle('active');
                toggle.classList.toggle('active');
            });
            
            // Close menu when clicking outside
            document.addEventListener('click', (e) => {
                if (!toggle.contains(e.target) && !menu.contains(e.target)) {
                    menu.classList.remove('active');
                    toggle.classList.remove('active');
                }
            });
        }
    }
    
    initializeScrollEffects() {
        let lastScroll = 0;
        const nav = document.querySelector('.nav');
        
        if (!nav) return;
        
        window.addEventListener('scroll', () => {
            const currentScroll = window.pageYOffset;
            
            // Add background on scroll
            if (currentScroll > 50) {
                nav.classList.add('scrolled');
            } else {
                nav.classList.remove('scrolled');
            }
            
            // Hide/show on scroll
            if (currentScroll > lastScroll && currentScroll > 100) {
                nav.classList.add('nav-hidden');
            } else {
                nav.classList.remove('nav-hidden');
            }
            
            lastScroll = currentScroll;
        });
    }
}

// Additional navigation styles
const navStyles = `
<style>
    /* Unified Navigation Styles */
    .nav, .navbar {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        z-index: 1000;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    .nav.scrolled {
        background: rgba(255, 255, 255, 0.98);
        box-shadow: 0 2px 20px rgba(0, 0, 0, 0.1);
    }
    
    .nav.nav-hidden {
        transform: translateY(-100%);
    }
    
    .nav-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 1rem 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .nav-logo, .nav-brand {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--primary-color, #1B365D);
        text-decoration: none;
        transition: transform 0.3s ease;
    }
    
    .nav-logo:hover {
        transform: scale(1.05);
    }
    
    .phi-symbol {
        font-size: 1.8rem;
        color: var(--phi-gold, #0F7B8A);
        font-weight: 400;
    }
    
    .nav-menu {
        display: flex;
        list-style: none;
        gap: 2rem;
        align-items: center;
        margin: 0;
        padding: 0;
    }
    
    .nav-link {
        color: var(--text-primary, #2D3748);
        text-decoration: none;
        font-weight: 500;
        transition: all 0.3s ease;
        position: relative;
        padding: 0.5rem 0;
    }
    
    .nav-link:hover {
        color: var(--phi-gold, #0F7B8A);
    }
    
    .nav-link.active {
        color: var(--phi-gold, #0F7B8A);
    }
    
    .nav-link.active::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--phi-gold, #0F7B8A);
    }
    
    .nav-toggle {
        display: none;
        background: none;
        border: none;
        cursor: pointer;
        padding: 0.5rem;
        flex-direction: column;
        gap: 4px;
    }
    
    .nav-toggle span {
        display: block;
        width: 25px;
        height: 2px;
        background: var(--primary-color, #1B365D);
        transition: all 0.3s ease;
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
    
    /* Mobile Styles */
    @media (max-width: 768px) {
        .nav-container {
            padding: 1rem;
        }
        
        .nav-menu {
            position: fixed;
            top: 70px;
            left: 0;
            right: 0;
            background: white;
            flex-direction: column;
            padding: 2rem;
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1);
            transform: translateY(-100%);
            opacity: 0;
            transition: all 0.3s ease;
        }
        
        .nav-menu.active {
            transform: translateY(0);
            opacity: 1;
        }
        
        .nav-toggle {
            display: flex;
        }
        
        .nav-link {
            padding: 0.75rem 0;
        }
    }
    
    /* Ensure body has proper padding for fixed nav */
    body {
        padding-top: 80px;
    }
</style>
`;

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    // Add styles
    document.head.insertAdjacentHTML('beforeend', navStyles);
    
    // Initialize navigation
    new UnifiedNavigation();
    
    console.log('🧭 Unified navigation system initialized');
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UnifiedNavigation;
}