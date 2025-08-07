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
                { label: 'Metastation Hub', href: 'metastation-hub.html', icon: '⭐', featured: true },
                { label: 'AI Hub', href: 'ai-unified-hub.html', icon: '🤖', featured: true },
                { label: 'AI Agents', href: 'ai-agents-ecosystem.html', icon: '🤖', featured: true },
                { label: 'Mathematics', href: 'mathematical-framework.html', icon: '📐', featured: true },
                { label: 'Consciousness', href: 'consciousness_dashboard.html', icon: '🧠', featured: true },
                { label: 'DALL-E Gallery', href: 'dalle-gallery.html', icon: '🌟', featured: true },
                { label: 'Gallery', href: 'implementations-gallery.html', icon: '🎨' },
                { label: 'Philosophy', href: 'philosophy.html', icon: '📜' },
                { label: 'Research', href: 'research.html', icon: '📊' }
            ],
            sidebar: [
                {
                    section: 'Core Experiences',
                    links: [
                        { label: 'Metastation Hub', href: 'metastation-hub.html', icon: '⭐', featured: true },
                        { label: 'Zen Unity Meditation', href: 'zen-unity-meditation.html', icon: '🧘' },
                        { label: 'Unity Experience', href: 'unity-mathematics-experience.html', icon: '⚡' },
                        { label: 'Consciousness Experience', href: 'unity_consciousness_experience.html', icon: '🧠' },
                        { label: 'Anthill Simulation', href: 'anthill.html', icon: '🐜' },
                        { label: 'Transcendental Unity Demo', href: 'transcendental-unity-demo.html', icon: '🌟' }
                    ]
                },
                {
                    section: 'Mathematics & Proofs',
                    links: [
                        { label: 'Mathematical Framework', href: 'mathematical-framework.html', icon: '📐', featured: true },
                        { label: 'Unity Synthesis (New)', href: 'unity-mathematics-synthesis.html', icon: '🧮' },
                        { label: 'Unity Meta-Atlas (New)', href: 'unity-meta-atlas.html', icon: '🌌' },
                        { label: '3000 ELO Proof', href: '3000-elo-proof.html', icon: '🏆' },
                        { label: 'Proofs & Theorems', href: 'proofs.html', icon: '✓' },
                        { label: 'Al-Khwarizmi Unity', href: 'al_khwarizmi_phi_unity.html', icon: '🕌' },
                        { label: 'Mathematical Playground', href: 'mathematical_playground.html', icon: '🎮' },
                        { label: 'Enhanced Proofs', href: 'enhanced-mathematical-proofs.html', icon: '📊' }
                    ]
                },
                {
                    section: 'Dashboards & Visualizations',
                    links: [
                        { label: 'Consciousness Dashboard', href: 'consciousness_dashboard.html', icon: '📊', featured: true },
                        { label: 'Unity Visualization', href: 'unity_visualization.html', icon: '🌐' },
                        { label: 'Interactive Playground', href: 'playground.html', icon: '🎮' },
                        { label: 'Dashboards Overview', href: 'dashboards.html', icon: '📈' },
                        { label: 'Enhanced 3D Consciousness', href: 'enhanced-3d-consciousness-field.html', icon: '🎯' },
                        { label: 'Enhanced Unity Visualization', href: 'enhanced-unity-visualization-system.html', icon: '🔮' },
                        { label: 'Test Visualizations', href: 'test-visualizations.html', icon: '🧪' }
                    ]
                },
                {
                    section: 'AI & Agents',
                    links: [
                        { label: 'AI Unified Hub', href: 'ai-unified-hub.html', icon: '🤖', featured: true },
                        { label: 'AI Agents Ecosystem', href: 'ai-agents-ecosystem.html', icon: '🤖' },
                        { label: 'Unity Agents', href: 'agents.html', icon: '⚡' },
                        { label: 'Metagamer Agent', href: 'metagamer_agent.html', icon: '🎯' },
                        { label: 'Enhanced AI Demo', href: 'enhanced-ai-demo.html', icon: '🚀' },
                        { label: 'OpenAI Integration', href: 'openai-integration.html', icon: '🔗' },
                        { label: 'GPT-4o Reasoning', href: '#', icon: '🧠', action: 'openAIReasoning' },
                        { label: 'DALL-E 3 Art', href: '#', icon: '🎨', action: 'openAIVisualization' },
                        { label: 'Voice Processing', href: '#', icon: '🎤', action: 'openAIVoice' },
                        { label: 'Code Search', href: '#', icon: '🔍', action: 'openAISearch' },
                        { label: 'Knowledge Base', href: '#', icon: '📚', action: 'openAIKnowledge' }
                    ]
                },
                {
                    section: 'Gallery & Implementations',
                    links: [
                        { label: 'Implementations Gallery', href: 'implementations-gallery.html', icon: '⚙️', featured: true },
                        { label: 'Visual Gallery', href: 'gallery.html', icon: '🎨' },
                        { label: 'DALL-E Consciousness Gallery', href: 'dalle-gallery.html', icon: '🌟', featured: true },
                        { label: 'Live Code Showcase', href: 'live-code-showcase.html', icon: '💻' },
                        { label: 'Enhanced Unity Demo', href: 'enhanced-unity-demo.html', icon: '🚀' },
                        { label: 'Implementations', href: 'implementations.html', icon: '🔧' },
                        { label: 'Meta-Optimal Landing', href: 'meta-optimal-landing.html', icon: '🎯' },
                        { label: 'Enhanced Unity Landing', href: 'enhanced-unity-landing.html', icon: '🌟' }
                    ]
                },
                {
                    section: 'Philosophy & Theory',
                    links: [
                        { label: 'Unity Philosophy', href: 'philosophy.html', icon: '📜', featured: true },
                        { label: 'Metagambit Theory', href: 'metagambit.html', icon: '🎲' },
                        { label: 'Further Reading', href: 'further-reading.html', icon: '📚' },
                        { label: 'Unity Advanced Features', href: 'unity-advanced-features.html', icon: '⚡' }
                    ]
                },
                {
                    section: 'Research & Learning',
                    links: [
                        { label: 'Research Overview', href: 'research.html', icon: '📊', featured: true },
                        { label: 'Publications', href: 'publications.html', icon: '📄' },
                        { label: 'Learn Unity Math', href: 'learn.html', icon: '📖' },
                        { label: 'Learning Hub', href: 'learning.html', icon: '🎓' },
                        { label: 'Mobile App', href: 'mobile-app.html', icon: '📱' }
                    ]
                },
                {
                    section: 'Tools & Utilities',
                    links: [
                        { label: 'Site Map', href: 'sitemap.html', icon: '🗺️' },
                        { label: 'About', href: 'about.html', icon: 'ℹ️' },
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
                            { label: 'Consciousness Experience', href: 'unity_consciousness_experience.html' },
                            { label: 'Anthill Simulation', href: 'anthill.html' },
                            { label: 'Transcendental Unity Demo', href: 'transcendental-unity-demo.html' }
                        ]
                    },
                    {
                        title: 'Mathematics',
                        links: [
                            { label: 'Mathematical Framework', href: 'mathematical-framework.html' },
                            { label: 'Proofs & Theorems', href: 'proofs.html' },
                            { label: '3000 ELO Proof', href: '3000-elo-proof.html' },
                            { label: 'Interactive Playground', href: 'playground.html' },
                            { label: 'Al-Khwarizmi Unity', href: 'al_khwarizmi_phi_unity.html' },
                            { label: 'Mathematical Playground', href: 'mathematical_playground.html' }
                        ]
                    },
                    {
                        title: 'Consciousness',
                        links: [
                            { label: 'Consciousness Dashboard', href: 'consciousness_dashboard.html' },
                            { label: 'Metagamer Agent', href: 'metagamer_agent.html' },
                            { label: 'Unity Agents', href: 'agents.html' },
                            { label: 'AI Agents Ecosystem', href: 'ai-agents-ecosystem.html' },
                            { label: 'Unity Visualization', href: 'unity_visualization.html' },
                            { label: 'Enhanced 3D Consciousness', href: 'enhanced-3d-consciousness-field.html' }
                        ]
                    },
                    {
                        title: 'AI Integration',
                        links: [
                            { label: 'AI Unified Hub', href: 'ai-unified-hub.html' },
                            { label: 'Enhanced AI Demo', href: 'enhanced-ai-demo.html' },
                            { label: 'OpenAI Integration', href: 'openai-integration.html' },
                            { label: 'GPT-4o Reasoning', href: 'ai-unified-hub.html#reasoning' },
                            { label: 'DALL-E 3 Art', href: 'ai-unified-hub.html#visualization' },
                            { label: 'Voice Processing', href: 'ai-unified-hub.html#voice' }
                        ]
                    },
                    {
                        title: 'Gallery',
                        links: [
                            { label: 'Implementations Gallery', href: 'implementations-gallery.html' },
                            { label: 'Visual Gallery', href: 'gallery.html' },
                            { label: 'Live Code Showcase', href: 'live-code-showcase.html' },
                            { label: 'Enhanced Unity Demo', href: 'enhanced-unity-demo.html' },
                            { label: 'Implementations', href: 'implementations.html' },
                            { label: 'Meta-Optimal Landing', href: 'meta-optimal-landing.html' }
                        ]
                    },
                    {
                        title: 'Philosophy',
                        links: [
                            { label: 'Unity Philosophy', href: 'philosophy.html' },
                            { label: 'Metagambit Theory', href: 'metagambit.html' },
                            { label: 'Further Reading', href: 'further-reading.html' },
                            { label: 'Unity Advanced Features', href: 'unity-advanced-features.html' }
                        ]
                    },
                    {
                        title: 'Resources',
                        links: [
                            { label: 'Research Overview', href: 'research.html' },
                            { label: 'Publications', href: 'publications.html' },
                            { label: 'Learn Unity Math', href: 'learn.html' },
                            { label: 'Learning Hub', href: 'learning.html' },
                            { label: 'Mobile App', href: 'mobile-app.html' },
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

    // AI Feature Handlers
    openAIReasoning() {
        window.location.href = 'ai-unified-hub.html#reasoning';
    }

    openAIVisualization() {
        window.location.href = 'ai-unified-hub.html#visualization';
    }

    openAIVoice() {
        window.location.href = 'ai-unified-hub.html#voice';
    }

    openAISearch() {
        window.location.href = 'ai-unified-hub.html#search';
    }

    openAIKnowledge() {
        window.location.href = 'ai-unified-hub.html#knowledge';
    }
}

// Initialize the navigation system
window.metaOptimalNav = new MetaOptimalCompleteNavigation();