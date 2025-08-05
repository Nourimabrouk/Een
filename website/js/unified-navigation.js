/**
 * Een Unity Mathematics - Unified Navigation System
 * Meta-Optimal Navigation with Complete Site Coverage
 * 3000 ELO 300 IQ Navigation Framework
 */

class UnifiedNavigation {
    constructor() {
        this.currentPage = this.getCurrentPage();
        this.init();
    }

    getCurrentPage() {
        const path = window.location.pathname;
        const filename = path.split('/').pop() || 'index.html';

        // Map all available pages
        const pageMap = {
            'index.html': 'home',
            'proofs.html': 'proofs',
            '3000-elo-proof.html': 'advanced-proofs',
            'research.html': 'research',
            'publications.html': 'publications',
            'playground.html': 'playground',
            'mathematical_playground.html': 'mathematical-playground',
            'gallery.html': 'gallery',
            'learn.html': 'learn',
            'learning.html': 'learning-academy',
            'metagambit.html': 'metagambit',
            'metagamer_agent.html': 'metagamer-agent',
            'agents.html': 'agents',
            'consciousness_dashboard.html': 'consciousness-dashboard',
            'consciousness_dashboard_clean.html': 'consciousness-dashboard-clean',
            'unity_consciousness_experience.html': 'unity-consciousness-experience',
            'unity_visualization.html': 'unity-visualization',
            'philosophy.html': 'philosophy',
            'implementations.html': 'implementations',
            'further-reading.html': 'further-reading',
            'dashboards.html': 'dashboards',
            'enhanced-unity-demo.html': 'enhanced-unity-demo',
            'revolutionary-landing.html': 'revolutionary-landing',
            'meta-optimal-landing.html': 'meta-optimal-landing',
            'mobile-app.html': 'mobile-app',
            'about.html': 'about',
            'al_khwarizmi_phi_unity.html': 'al-khwarizmi-phi-unity'
        };

        return pageMap[filename] || 'home';
    }

    init() {
        this.createNavigationHTML();
        this.attachEventListeners();
        this.setActiveState();
        this.initializeScrollEffects();
        this.initializeTheme();
    }

    createNavigationHTML() {
        const navHTML = `
            <nav class="navbar unified-nav" id="enhancedUnityNav">
                <div class="nav-container">
                    <a href="index.html" class="nav-logo">
                        <span class="phi-symbol pulse-glow">φ</span>
                        <span class="logo-text">Een</span>
                        <span class="elo-badge">3000 ELO</span>
                    </a>
                    <ul class="nav-menu">
                        <li class="nav-item dropdown">
                            <a href="#" class="nav-link dropdown-toggle">
                                <i class="fas fa-calculator"></i>
                                Mathematics
                                <i class="fas fa-chevron-down dropdown-arrow"></i>
                            </a>
                            <ul class="dropdown-menu">
                                <li><a href="proofs.html" class="dropdown-link">
                                    <i class="fas fa-check-circle"></i>
                                    Unity Proofs
                                </a></li>
                                <li><a href="3000-elo-proof.html" class="dropdown-link">
                                    <i class="fas fa-trophy"></i>
                                    Advanced Proofs
                                </a></li>
                                <li><a href="playground.html" class="dropdown-link">
                                    <i class="fas fa-play-circle"></i>
                                    Interactive Playground
                                </a></li>
                                <li><a href="mathematical_playground.html" class="dropdown-link">
                                    <i class="fas fa-calculator"></i>
                                    Mathematical Playground
                                </a></li>
                                <li><a href="enhanced-unity-demo.html" class="dropdown-link">
                                    <i class="fas fa-rocket"></i>
                                    Enhanced Unity Demo
                                </a></li>
                                <li><a href="al_khwarizmi_phi_unity.html" class="dropdown-link">
                                    <i class="fas fa-brain"></i>
                                    Al-Khwarizmi Phi Unity
                                </a></li>
                            </ul>
                        </li>
                        <li class="nav-item dropdown">
                            <a href="#" class="nav-link dropdown-toggle">
                                <i class="fas fa-brain"></i>
                                Consciousness
                                <i class="fas fa-chevron-down dropdown-arrow"></i>
                            </a>
                            <ul class="dropdown-menu">
                                <li><a href="consciousness_dashboard.html" class="dropdown-link">
                                    <i class="fas fa-lightbulb"></i>
                                    Consciousness Fields
                                </a></li>
                                <li><a href="consciousness_dashboard_clean.html" class="dropdown-link">
                                    <i class="fas fa-lightbulb"></i>
                                    Clean Consciousness Dashboard
                                </a></li>
                                <li><a href="unity_consciousness_experience.html" class="dropdown-link">
                                    <i class="fas fa-meditation"></i>
                                    Unity Experience
                                </a></li>
                                <li><a href="unity_visualization.html" class="dropdown-link">
                                    <i class="fas fa-wave-square"></i>
                                    Unity Visualizations
                                </a></li>
                                <li><a href="philosophy.html" class="dropdown-link">
                                    <i class="fas fa-scroll"></i>
                                    Philosophy Treatise
                                </a></li>
                            </ul>
                        </li>
                        <li class="nav-item dropdown">
                            <a href="#" class="nav-link dropdown-toggle">
                                <i class="fas fa-flask"></i>
                                Research
                                <i class="fas fa-chevron-down dropdown-arrow"></i>
                            </a>
                            <ul class="dropdown-menu">
                                <li><a href="research.html" class="dropdown-link">
                                    <i class="fas fa-search"></i>
                                    Current Research
                                </a></li>
                                <li><a href="publications.html" class="dropdown-link">
                                    <i class="fas fa-file-alt"></i>
                                    Publications
                                </a></li>
                                <li><a href="implementations.html" class="dropdown-link">
                                    <i class="fas fa-code"></i>
                                    Core Implementations
                                </a></li>
                                <li><a href="further-reading.html" class="dropdown-link">
                                    <i class="fas fa-book-open"></i>
                                    Further Reading
                                </a></li>
                            </ul>
                        </li>
                        <li class="nav-item dropdown">
                            <a href="#" class="nav-link dropdown-toggle">
                                <i class="fas fa-graduation-cap"></i>
                                Learning
                                <i class="fas fa-chevron-down dropdown-arrow"></i>
                            </a>
                            <ul class="dropdown-menu">
                                <li><a href="learn.html" class="dropdown-link">
                                    <i class="fas fa-graduation-cap"></i>
                                    Learn Unity Mathematics
                                </a></li>
                                <li><a href="learning.html" class="dropdown-link">
                                    <i class="fas fa-university"></i>
                                    Learning Academy
                                </a></li>
                                <li><a href="dashboards.html" class="dropdown-link">
                                    <i class="fas fa-tachometer-alt"></i>
                                    Dashboard Suite
                                </a></li>
                            </ul>
                        </li>
                        <li class="nav-item dropdown">
                            <a href="#" class="nav-link dropdown-toggle">
                                <i class="fas fa-images"></i>
                                Gallery
                                <i class="fas fa-chevron-down dropdown-arrow"></i>
                            </a>
                            <ul class="dropdown-menu">
                                <li><a href="gallery.html" class="dropdown-link">
                                    <i class="fas fa-palette"></i>
                                    Visualization Gallery
                                </a></li>
                                <li><a href="revolutionary-landing.html" class="dropdown-link">
                                    <i class="fas fa-star"></i>
                                    Revolutionary Landing
                                </a></li>
                                <li><a href="meta-optimal-landing.html" class="dropdown-link">
                                    <i class="fas fa-bullseye"></i>
                                    Meta-Optimal Landing
                                </a></li>
                            </ul>
                        </li>
                        <li class="nav-item dropdown">
                            <a href="#" class="nav-link dropdown-toggle">
                                <i class="fas fa-robot"></i>
                                AI Systems
                                <i class="fas fa-chevron-down dropdown-arrow"></i>
                            </a>
                            <ul class="dropdown-menu">
                                <li><a href="agents.html" class="dropdown-link">
                                    <i class="fas fa-robot"></i>
                                    AI Agents
                                </a></li>
                                <li><a href="metagambit.html" class="dropdown-link">
                                    <i class="fas fa-chess"></i>
                                    MetaGambit System
                                </a></li>
                                <li><a href="metagamer_agent.html" class="dropdown-link">
                                    <i class="fas fa-gamepad"></i>
                                    MetaGamer Agent
                                </a></li>
                                <li><a href="mobile-app.html" class="dropdown-link">
                                    <i class="fas fa-mobile-alt"></i>
                                    Mobile App
                                </a></li>
                            </ul>
                        </li>
                        <li class="nav-item">
                            <a href="about.html" class="nav-link">
                                <i class="fas fa-user-graduate"></i>
                                About
                            </a>
                        </li>
                    </ul>
                    <div class="nav-toggle" id="navToggle">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                    <button class="nav-link ai-chat-trigger" id="aiChatTrigger" title="AI Assistant">
                        <i class="fas fa-robot"></i>
                        <span class="ai-label">AI Chat</span>
                    </button>
                </div>
            </nav>
        `;

        // Replace existing nav or insert at top of body
        const existingNav = document.querySelector('.navbar, .nav, nav, #enhancedUnityNav');
        if (existingNav) {
            existingNav.outerHTML = navHTML;
        } else {
            document.body.insertAdjacentHTML('afterbegin', navHTML);
        }
    }

    attachEventListeners() {
        // Mobile menu toggle
        const navToggle = document.getElementById('navToggle');
        const navMenu = document.querySelector('.nav-menu');

        if (navToggle && navMenu) {
            navToggle.addEventListener('click', () => {
                navToggle.classList.toggle('active');
                navMenu.classList.toggle('active');
            });
        }

        // Dropdown toggle
        const dropdownToggles = document.querySelectorAll('.dropdown-toggle');
        dropdownToggles.forEach(toggle => {
            toggle.addEventListener('click', (e) => {
                e.preventDefault();
                const dropdown = toggle.closest('.dropdown');
                const dropdownMenu = dropdown.querySelector('.dropdown-menu');
                dropdownMenu.classList.toggle('active');
            });
        });

        // Close dropdowns when clicking outside
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.dropdown')) {
                document.querySelectorAll('.dropdown-menu').forEach(menu => {
                    menu.classList.remove('active');
                });
            }
        });

        // AI Chat Trigger Functionality
        const aiChatTrigger = document.getElementById('aiChatTrigger');
        if (aiChatTrigger) {
            aiChatTrigger.addEventListener('click', () => {
                this.openAIChat();
            });
        }

        // Close mobile menu when clicking nav links
        document.querySelectorAll('.nav-link, .dropdown-link').forEach(link => {
            link.addEventListener('click', () => {
                if (navMenu) navMenu.classList.remove('active');
                if (navToggle) navToggle.classList.remove('active');
            });
        });

        // Smooth scrolling for anchor links
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

    setActiveState() {
        // Remove all active states
        document.querySelectorAll('.nav-link, .dropdown-link').forEach(link => {
            link.classList.remove('active');
        });

        // Set active state based on current page
        const currentPage = this.currentPage;

        // Find and activate the appropriate link
        const activeLink = document.querySelector(`[href*="${currentPage}"], [href*="${currentPage.replace('-', '_')}"]`);
        if (activeLink) {
            activeLink.classList.add('active');
        }

        // Special handling for home page
        if (currentPage === 'home') {
            const homeLink = document.querySelector('a[href="index.html"]');
            if (homeLink) homeLink.classList.add('active');
        }
    }

    initializeScrollEffects() {
        const navbar = document.getElementById('enhancedUnityNav');
        if (!navbar) return;

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

    openAIChat() {
        // Initialize AI chat if not already done
        if (typeof window.eenChat === 'undefined' || !window.eenChat) {
            // Check if EenAIChat class is available
            if (typeof EenAIChat !== 'undefined') {
                window.eenChat = EenAIChat.initialize();
                setTimeout(() => {
                    if (window.eenChat) {
                        window.eenChat.open();
                    }
                }, 100);
            } else {
                // Load AI chat integration script
                const script = document.createElement('script');
                script.src = 'js/ai-chat-integration.js';
                script.onload = () => {
                    setTimeout(() => {
                        if (window.eenChat) {
                            window.eenChat.open();
                        }
                    }, 100);
                };
                script.onerror = () => {
                    console.error('Failed to load AI chat integration');
                    alert('AI Chat is currently unavailable. Please try again later.');
                };
                document.head.appendChild(script);
            }
        } else {
            window.eenChat.open();
        }
    }

    initializeTheme() {
        const savedTheme = localStorage.getItem('theme');
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

        if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
            document.body.classList.add('dark-mode');
        }
    }

    // Method to highlight specific navigation item (for external use)
    highlightNavItem(page) {
        document.querySelectorAll('.nav-link, .dropdown-link').forEach(link => {
            link.classList.remove('active');
        });

        const targetLink = document.querySelector(`[href*="${page}"]`);
        if (targetLink) {
            targetLink.classList.add('active');
        }
    }

    // Method to add badges to navigation items (e.g., "New", "Updated")
    addNavBadge(page, text, type = 'new') {
        const navLink = document.querySelector(`[href*="${page}"]`);
        if (navLink && !navLink.querySelector('.nav-badge')) {
            const badge = document.createElement('span');
            badge.className = `nav-badge badge-${type}`;
            badge.textContent = text;
            navLink.appendChild(badge);
        }
    }
}

// Enhanced navigation styles
const navigationStyles = `
<style>
/* Unified Navigation Styles - Meta-Optimal Design */
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
.nav-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem clamp(1rem, 5vw, 2rem);
    max-width: 1280px;
    margin: 0 auto;
    height: 80px;
}

/* Brand styling */
.nav-logo {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    text-decoration: none;
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--primary-color, #1B365D);
    transition: all var(--transition-smooth, 0.3s ease);
}

.nav-logo:hover {
    transform: translateY(-2px);
}

.phi-symbol {
    font-family: var(--font-serif, 'Crimson Text', serif);
    font-size: 2rem;
    color: var(--phi-gold, #FFD700);
    font-style: italic;
    font-weight: 600;
    text-shadow: 0 0 10px rgba(255, 215, 0, 0.3);
}

.logo-text {
    font-family: var(--font-serif, 'Crimson Text', serif);
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

/* Navigation menu */
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
    color: var(--text-secondary, #718096);
    font-weight: 500;
    font-size: 0.95rem;
    border-radius: var(--radius-lg, 0.75rem);
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

.dropdown-toggle .dropdown-arrow {
    font-size: 0.8rem;
    transition: transform var(--transition-smooth, 0.3s ease);
}

.dropdown:hover .dropdown-arrow {
    transform: rotate(180deg);
}

.dropdown-menu {
    position: absolute;
    top: 100%;
    left: 0;
    background: var(--bg-primary, white);
    border: 1px solid var(--border-color, #E2E8F0);
    border-radius: var(--radius-lg, 0.75rem);
    box-shadow: var(--shadow-xl, 0 20px 25px -5px rgba(0,0,0,0.1));
    padding: 0.5rem;
    min-width: 240px;
    opacity: 0;
    visibility: hidden;
    transform: translateY(-10px);
    transition: all var(--transition-smooth, 0.3s ease);
    z-index: 1001;
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
    color: var(--text-secondary, #718096);
    border-radius: var(--radius, 0.375rem);
    transition: all var(--transition-fast, 0.15s ease);
    font-size: 0.9rem;
    margin-bottom: 0.25rem;
}

.dropdown-link:hover {
    background: var(--bg-secondary, #F8FAFC);
    color: var(--primary-color, #1B365D);
    transform: translateX(5px);
}

/* AI Chat Trigger Button */
.ai-chat-trigger {
    background: linear-gradient(135deg, var(--accent-color), var(--accent-bright));
    color: white;
    border: none;
    border-radius: var(--radius-lg, 0.75rem);
    padding: 0.75rem 1.25rem;
    margin-left: 1rem;
    cursor: pointer;
    transition: all var(--transition-smooth, 0.3s ease);
    position: relative;
    overflow: hidden;
}

.ai-chat-trigger::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transition: left 0.5s;
}

.ai-chat-trigger:hover::before {
    left: 100%;
}

.ai-chat-trigger:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg, 0 10px 15px -3px rgba(0,0,0,0.1));
    background: linear-gradient(135deg, var(--accent-bright), var(--accent-color));
}

.ai-chat-trigger .ai-label {
    margin-left: 0.5rem;
    font-weight: 600;
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

/* Navigation badges */
.nav-badge {
    position: absolute;
    top: 0.25rem;
    right: 0.25rem;
    background: var(--phi-gold, #FFD700);
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

/* Responsive design */
@media (max-width: 1024px) {
    .nav-menu {
        gap: 1rem;
    }
    
    .nav-link {
        padding: 0.5rem 1rem;
        font-size: 0.9rem;
    }
}

@media (max-width: 768px) {
    .nav-toggle {
        display: flex;
    }
    
    .nav-menu {
        position: fixed;
        top: 80px;
        left: 0;
        width: 100%;
        background: var(--bg-primary, white);
        border-top: 1px solid var(--border-color, #E2E8F0);
        box-shadow: var(--shadow-lg, 0 10px 25px rgba(0,0,0,0.1));
        flex-direction: column;
        gap: 0;
        padding: 2rem 0;
        transform: translateY(-100%);
        opacity: 0;
        visibility: hidden;
        transition: all var(--transition-smooth, 0.3s ease);
    }
    
    .nav-menu.active {
        transform: translateY(0);
        opacity: 1;
        visibility: visible;
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
        background: var(--bg-secondary, #F8FAFC);
        margin-left: 2rem;
    }
    
    .ai-chat-trigger {
        margin-left: 0;
        margin-top: 1rem;
        width: 100%;
        justify-content: center;
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
.nav-link:focus-visible,
.dropdown-link:focus-visible,
.ai-chat-trigger:focus-visible {
    outline: 2px solid var(--phi-gold, #FFD700);
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
        background: var(--phi-gold, #FFD700);
        transition: all var(--transition-smooth, 0.3s ease);
        transform: translateX(-50%);
    }
    
    .nav-link:hover::after,
    .nav-link.active::after {
        width: 80%;
    }
}

/* Special Effects */
.pulse-glow {
    animation: pulse-glow 3s ease-in-out infinite alternate;
}

@keyframes pulse-glow {
    from { text-shadow: 0 0 20px rgba(255, 215, 0, 0.5); }
    to { text-shadow: 0 0 30px rgba(255, 215, 0, 0.8); }
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
    });
} else {
    // Add navigation styles
    document.head.insertAdjacentHTML('beforeend', navigationStyles);

    // Initialize navigation
    window.unifiedNav = new UnifiedNavigation();
}

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UnifiedNavigation;
}