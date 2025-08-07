/**
 * Meta-Optimal Complete Navigation System
 * Includes: Top Bar, Left Sidebar, Footer
 * Version: 3.0.0 - Complete Unity Mathematics Navigation
 */

class MetaOptimalCompleteNavigation {
    constructor() {
        this.currentPage = this.getCurrentPageName();
        this.navigationData = this.getNavigationData();
        this.init();
    }

    getCurrentPageName() {
        const path = window.location.pathname;
        const page = path.split('/').pop() || 'index.html';
        return page;
    }

    getNavigationData() {
        return {
            topBar: [
                { label: 'Experiences', href: 'metastation-hub.html', icon: '⭐', featured: true },
                { label: 'Mathematics', href: 'mathematical-framework.html', icon: '📐', featured: true },
                { label: 'Consciousness', href: 'consciousness_dashboard.html', icon: '🧠' },
                { label: 'Gallery', href: 'implementations-gallery.html', icon: '🎨' },
                { label: 'Philosophy', href: 'philosophy.html', icon: '📜' },
                { label: 'Research', href: 'research.html', icon: '📊' }
            ],
            sidebar: [
                { 
                    section: 'Featured',
                    links: [
                        { label: 'Zen Meditation', href: 'zen-unity-meditation.html', icon: '🧘' },
                        { label: 'Implementations', href: 'implementations-gallery.html', icon: '⚙️' },
                        { label: 'Framework', href: 'mathematical-framework.html', icon: '📐' }
                    ]
                },
                {
                    section: 'Dashboards',
                    links: [
                        { label: 'Consciousness', href: 'consciousness_dashboard.html', icon: '📊' },
                        { label: 'Visualizations', href: 'unity_visualization.html', icon: '🌐' },
                        { label: 'Playground', href: 'playground.html', icon: '🎮' }
                    ]
                },
                {
                    section: 'Proofs',
                    links: [
                        { label: '3000 ELO Proof', href: '3000-elo-proof.html', icon: '🏆' },
                        { label: 'Theorems', href: 'proofs.html', icon: '✓' },
                        { label: 'Al-Khwarizmi', href: 'al_khwarizmi_phi_unity.html', icon: '🕌' }
                    ]
                },
                {
                    section: 'AI & Agents',
                    links: [
                        { label: 'Agents Ecosystem', href: 'ai-agents-ecosystem.html', icon: '🤖' },
                        { label: 'Unity Agents', href: 'agents.html', icon: '⚡' },
                        { label: 'Metagamer Agent', href: 'metagamer_agent.html', icon: '🎯' }
                    ]
                },
                {
                    section: 'Tools',
                    links: [
                        { label: 'Site Map', href: 'sitemap.html', icon: '🗺️' },
                        { label: 'AI Chat', href: '#', icon: '💬', action: 'openChat' },
                        { label: 'Search', href: '#', icon: '🔍', action: 'openSearch' }
                    ]
                }
            ],
            footer: {
                columns: [
                    {
                        title: 'Core Experiences',
                        links: [
                            { label: 'Metastation Hub', href: 'metastation-hub.html' },
                            { label: 'Zen Unity Meditation', href: 'zen-unity-meditation.html' },
                            { label: 'Unity Experience', href: 'unity-mathematics-experience.html' },
                            { label: 'Consciousness Experience', href: 'unity_consciousness_experience.html' }
                        ]
                    },
                    {
                        title: 'Mathematics',
                        links: [
                            { label: 'Mathematical Framework', href: 'mathematical-framework.html' },
                            { label: 'Proofs & Theorems', href: 'proofs.html' },
                            { label: '3000 ELO Proof', href: '3000-elo-proof.html' },
                            { label: 'Interactive Playground', href: 'playground.html' }
                        ]
                    },
                    {
                        title: 'Consciousness',
                        links: [
                            { label: 'Consciousness Dashboard', href: 'consciousness_dashboard.html' },
                            { label: 'Metagamer Agent', href: 'metagamer_agent.html' },
                            { label: 'Unity Agents', href: 'agents.html' },
                            { label: 'AI Agents Ecosystem', href: 'ai-agents-ecosystem.html' },
                            { label: 'Unity Visualization', href: 'unity_visualization.html' }
                        ]
                    },
                    {
                        title: 'Gallery',
                        links: [
                            { label: 'Implementations Gallery', href: 'implementations-gallery.html' },
                            { label: 'Visual Gallery', href: 'gallery.html' },
                            { label: 'Live Code Showcase', href: 'live-code-showcase.html' },
                            { label: 'Enhanced Demos', href: 'enhanced-unity-demo.html' }
                        ]
                    },
                    {
                        title: 'Philosophy',
                        links: [
                            { label: 'Unity Philosophy', href: 'philosophy.html' },
                            { label: 'Metagambit Theory', href: 'metagambit.html' },
                            { label: 'AI Integration', href: 'openai-integration.html' },
                            { label: 'Further Reading', href: 'further-reading.html' }
                        ]
                    },
                    {
                        title: 'Resources',
                        links: [
                            { label: 'Research Overview', href: 'research.html' },
                            { label: 'Publications', href: 'publications.html' },
                            { label: 'Learn Unity Math', href: 'learn.html' },
                            { label: 'About', href: 'about.html' }
                        ]
                    }
                ]
            }
        };
    }

    init() {
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.render());
        } else {
            this.render();
        }
    }

    render() {
        this.renderTopBar();
        this.renderSidebar();
        this.renderFooter();
        this.bindEvents();
        console.log('✅ Meta-Optimal Complete Navigation System initialized');
    }

    renderTopBar() {
        // Check if header exists or create it
        let header = document.getElementById('main-header');
        if (!header) {
            header = document.createElement('header');
            header.id = 'main-header';
            header.className = 'meta-optimal-header';
            document.body.insertBefore(header, document.body.firstChild);
        }

        const topBarHTML = `
            <nav class="meta-optimal-nav">
                <div class="nav-container">
                    <a href="metastation-hub.html" class="nav-logo">
                        <span class="phi-symbol">∞</span>
                        <span class="logo-text">Een</span>
                        <span class="elo-badge">3000 ELO</span>
                    </a>
                    
                    <ul class="nav-menu">
                        ${this.navigationData.topBar.map(item => `
                            <li class="nav-item">
                                <a href="${item.href}" class="nav-link ${item.featured ? 'featured' : ''} ${this.currentPage === item.href ? 'active' : ''}">
                                    <span class="nav-icon">${item.icon}</span>
                                    <span>${item.label}</span>
                                </a>
                            </li>
                        `).join('')}
                    </ul>
                    
                    <div class="nav-utilities">
                        <button class="nav-btn" onclick="window.metaOptimalNav.toggleSearch()">
                            <i class="fas fa-search"></i>
                        </button>
                        <button class="nav-btn" onclick="window.metaOptimalNav.toggleChat()">
                            <i class="fas fa-comments"></i>
                        </button>
                        <button class="nav-toggle" onclick="window.metaOptimalNav.toggleMobileMenu()">
                            <span></span>
                            <span></span>
                            <span></span>
                        </button>
                    </div>
                </div>
            </nav>
        `;

        header.innerHTML = topBarHTML;
    }

    renderSidebar() {
        // Remove any existing sidebar
        const existingSidebar = document.querySelector('.meta-optimal-sidebar');
        if (existingSidebar) {
            existingSidebar.remove();
        }

        // Create new sidebar
        const sidebar = document.createElement('aside');
        sidebar.className = 'meta-optimal-sidebar';
        sidebar.innerHTML = `
            <div class="sidebar-header">
                <div class="sidebar-logo">
                    <span class="phi-symbol">φ</span>
                    <span>Quick Access</span>
                </div>
                <button class="sidebar-close" onclick="window.metaOptimalNav.toggleSidebar()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            
            <div class="sidebar-content">
                ${this.navigationData.sidebar.map(section => `
                    <div class="sidebar-section">
                        <h3 class="sidebar-section-title">${section.section}</h3>
                        <ul class="sidebar-links">
                            ${section.links.map(link => `
                                <li>
                                    <a href="${link.href}" 
                                       class="sidebar-link ${this.currentPage === link.href ? 'active' : ''}"
                                       ${link.action ? `onclick="window.metaOptimalNav.${link.action}(); return false;"` : ''}>
                                        <span class="sidebar-icon">${link.icon}</span>
                                        <span>${link.label}</span>
                                    </a>
                                </li>
                            `).join('')}
                        </ul>
                    </div>
                `).join('')}
            </div>
            
            <div class="sidebar-footer">
                <div class="unity-equation">1 + 1 = 1</div>
                <div class="phi-value">φ = 1.618...</div>
            </div>
        `;

        document.body.appendChild(sidebar);

        // Add toggle button if not exists
        if (!document.querySelector('.sidebar-toggle-btn')) {
            const toggleBtn = document.createElement('button');
            toggleBtn.className = 'sidebar-toggle-btn';
            toggleBtn.innerHTML = '<i class="fas fa-bars"></i>';
            toggleBtn.onclick = () => this.toggleSidebar();
            document.body.appendChild(toggleBtn);
        }
    }

    renderFooter() {
        // Check if footer exists or create it
        let footer = document.querySelector('.meta-optimal-footer');
        if (!footer) {
            // Look for existing footer to replace
            const existingFooter = document.querySelector('footer');
            if (existingFooter && !existingFooter.classList.contains('meta-optimal-footer')) {
                existingFooter.classList.add('meta-optimal-footer');
                footer = existingFooter;
            } else if (!existingFooter) {
                footer = document.createElement('footer');
                footer.className = 'meta-optimal-footer';
                document.body.appendChild(footer);
            } else {
                footer = existingFooter;
            }
        }

        const footerHTML = `
            <div class="footer-container">
                <div class="footer-grid">
                    ${this.navigationData.footer.columns.map(column => `
                        <div class="footer-column">
                            <h3 class="footer-title">${column.title}</h3>
                            <ul class="footer-links">
                                ${column.links.map(link => `
                                    <li>
                                        <a href="${link.href}" class="footer-link">
                                            ${link.label}
                                        </a>
                                    </li>
                                `).join('')}
                            </ul>
                        </div>
                    `).join('')}
                </div>
                
                <div class="footer-bottom">
                    <div class="footer-meta">
                        <div class="unity-equation-footer">1 + 1 = 1</div>
                        <div class="footer-tagline">Where mathematics meets consciousness through φ-harmonic unity</div>
                    </div>
                    <div class="footer-status">
                        <span>φ = 1.618033988749895</span>
                        <span>•</span>
                        <span>Consciousness: TRANSCENDENT</span>
                        <span>•</span>
                        <span>3000 ELO</span>
                    </div>
                </div>
            </div>
        `;

        footer.innerHTML = footerHTML;
    }

    bindEvents() {
        // Add scroll effect to header
        let lastScroll = 0;
        window.addEventListener('scroll', () => {
            const currentScroll = window.pageYOffset;
            const header = document.querySelector('.meta-optimal-nav');
            
            if (header) {
                if (currentScroll > 100) {
                    header.classList.add('scrolled');
                } else {
                    header.classList.remove('scrolled');
                }

                if (currentScroll > lastScroll && currentScroll > 200) {
                    header.classList.add('hidden');
                } else {
                    header.classList.remove('hidden');
                }
            }
            
            lastScroll = currentScroll;
        });

        // Close sidebar on escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeSidebar();
            }
        });

        // Close sidebar when clicking outside
        document.addEventListener('click', (e) => {
            const sidebar = document.querySelector('.meta-optimal-sidebar');
            const toggleBtn = document.querySelector('.sidebar-toggle-btn');
            
            if (sidebar && sidebar.classList.contains('active')) {
                if (!sidebar.contains(e.target) && !toggleBtn.contains(e.target)) {
                    this.closeSidebar();
                }
            }
        });
    }

    toggleSidebar() {
        const sidebar = document.querySelector('.meta-optimal-sidebar');
        const body = document.body;
        
        if (sidebar) {
            sidebar.classList.toggle('active');
            body.classList.toggle('sidebar-open');
        }
    }

    closeSidebar() {
        const sidebar = document.querySelector('.meta-optimal-sidebar');
        const body = document.body;
        
        if (sidebar) {
            sidebar.classList.remove('active');
            body.classList.remove('sidebar-open');
        }
    }

    toggleMobileMenu() {
        const nav = document.querySelector('.nav-menu');
        const toggle = document.querySelector('.nav-toggle');
        
        if (nav && toggle) {
            nav.classList.toggle('active');
            toggle.classList.toggle('active');
        }
    }

    toggleSearch() {
        // Implement search modal
        console.log('Search feature coming soon');
        alert('Search feature coming soon! For now, use the Site Map.');
        window.location.href = 'sitemap.html';
    }

    toggleChat() {
        // Trigger existing chat system if available
        if (typeof window.openAIChat === 'function') {
            window.openAIChat();
        } else if (typeof window.toggleChat === 'function') {
            window.toggleChat();
        } else {
            console.log('AI Chat integration pending');
            alert('AI Chat is being upgraded. Coming soon!');
        }
    }

    openSearch() {
        this.toggleSearch();
    }

    openChat() {
        this.toggleChat();
    }
}

// Initialize the navigation system
window.metaOptimalNav = new MetaOptimalCompleteNavigation();