/**
 * Een Unity Mathematics - Unified Navigation System
 * Provides consistent navigation and styling across all pages
 */

// Enhanced Navigation HTML Template - Complete Site Overhaul
const unifiedNavigation = `
    <nav class="enhanced-nav" id="enhancedUnityNav">
        <div class="nav-container">
            <a href="index.html" class="nav-logo">
                <span class="phi-symbol pulse-glow">φ</span>
                <span class="logo-text">Een</span>
                <span class="elo-badge">Advanced</span>
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
                        <li><a href="unity_consciousness_experience.html" class="dropdown-link">
                            <i class="fas fa-meditation"></i>
                            Unity Experience
                        </a></li>
                        <li><a href="philosophy.html" class="dropdown-link">
                            <i class="fas fa-scroll"></i>
                            Philosophy Treatise
                        </a></li>
                        <li><a href="unity_visualization.html" class="dropdown-link">
                            <i class="fas fa-wave-square"></i>
                            Unity Visualizations
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
                        <li><a href="learning.html" class="dropdown-link">
                            <i class="fas fa-graduation-cap"></i>
                            Learning Academy
                        </a></li>
                        <li><a href="further-reading.html" class="dropdown-link">
                            <i class="fas fa-book-open"></i>
                            Further Reading
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
                        <li><a href="dashboards.html" class="dropdown-link">
                            <i class="fas fa-tachometer-alt"></i>
                            Dashboard Suite
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
                <li class="nav-item dropdown">
                    <a href="#" class="nav-link dropdown-toggle">
                        <i class="fas fa-cogs"></i>
                        Navigation
                        <i class="fas fa-chevron-down dropdown-arrow"></i>
                    </a>
                    <ul class="dropdown-menu">
                        <li><a href="unified-nav.html" class="dropdown-link">
                            <i class="fas fa-compass"></i>
                            Unified Navigation
                        </a></li>
                        <li><a href="enhanced-unified-nav.html" class="dropdown-link">
                            <i class="fas fa-route"></i>
                            Enhanced Navigation
                        </a></li>
                        <li><a href="about.html" class="dropdown-link">
                            <i class="fas fa-user-graduate"></i>
                            About Project
                        </a></li>
                    </ul>
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

// Enhanced Footer HTML Template - Complete Site Coverage
const unifiedFooter = `
    <footer class="footer">
        <div class="container">
            <div class="footer-content">
                <div class="footer-section">
                    <h3>Een Unity Mathematics</h3>
                    <p>Advanced computational consciousness mathematics demonstrating the profound truth that 1+1=1 through rigorous mathematical frameworks and philosophical transcendence.</p>
                    <div class="social-links">
                        <a href="https://github.com/Nourimabrouk/Een" target="_blank" aria-label="GitHub">
                            <i class="fab fa-github"></i>
                        </a>
                    </div>
                </div>
                <div class="footer-section">
                    <h4>Mathematical Core</h4>
                    <a href="proofs.html">Unity Proofs</a>
                    <a href="3000-elo-proof.html">Advanced Proofs</a>
                    <a href="playground.html">Interactive Playground</a>
                    <a href="mathematical_playground.html">Mathematical Playground</a>
                    <a href="enhanced-unity-demo.html">Enhanced Unity Demo</a>
                </div>
                <div class="footer-section">
                    <h4>Consciousness & Philosophy</h4>
                    <a href="consciousness_dashboard.html">Consciousness Fields</a>
                    <a href="unity_consciousness_experience.html">Unity Experience</a>
                    <a href="philosophy.html">Philosophy Treatise</a>
                    <a href="unity_visualization.html">Unity Visualizations</a>
                </div>
                <div class="footer-section">
                    <h4>Research & Development</h4>
                    <a href="research.html">Current Research</a>
                    <a href="publications.html">Publications</a>
                    <a href="implementations.html">Core Implementations</a>
                    <a href="learning.html">Learning Academy</a>
                    <a href="further-reading.html">Further Reading</a>
                </div>
                <div class="footer-section">
                    <h4>Visual Gallery</h4>
                    <a href="gallery.html">Visualization Gallery</a>
                    <a href="dashboards.html">Dashboard Suite</a>
                    <a href="revolutionary-landing.html">Revolutionary Landing</a>
                    <a href="meta-optimal-landing.html">Meta-Optimal Landing</a>
                </div>
                <div class="footer-section">
                    <h4>AI Systems & Apps</h4>
                    <a href="agents.html">AI Agents</a>
                    <a href="metagambit.html">MetaGambit System</a>
                    <a href="metagamer_agent.html">MetaGamer Agent</a>
                    <a href="mobile-app.html">Mobile App</a>
                    <a href="about.html">About Project</a>
                </div>
            </div>
            <div class="footer-bottom">
                <p>&copy; 2025 Een Unity Mathematics. Where pure mathematics transcends paradox: 1+1=1.</p>
            </div>
        </div>
    </footer>
`;

// Unified CSS Styles
const unifiedStyles = `
    <style>
        /* CSS Variables */
        :root {
            --primary-color: #1a2332;
            --secondary-color: #2d3748;
            --accent-color: #4a5568;
        --accent-bright: #667eea;
            --phi-gold: #FFD700;
            --phi-gold-light: #FFA500;
            --text-primary: #2d3748;
            --text-secondary: #4a5568;
            --text-light: #718096;
            --bg-primary: #ffffff;
            --bg-secondary: #f7fafc;
            --bg-tertiary: #edf2f7;
            --border-color: #e2e8f0;
            --shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
            --shadow-lg: 0 10px 25px rgba(0, 0, 0, 0.15);
            --shadow-xl: 0 20px 40px rgba(0, 0, 0, 0.1);
            --radius: 8px;
            --radius-md: 12px;
            --radius-lg: 16px;
            --radius-xl: 24px;
            --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            --transition-smooth: all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
            --font-primary: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-serif: 'Crimson Text', Georgia, serif;
            --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        html {
            scroll-behavior: smooth;
        }

        body {
            font-family: var(--font-primary);
            line-height: 1.7;
            color: var(--text-primary);
            background: var(--bg-primary);
            overflow-x: hidden;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 2rem;
        }

        /* Enhanced Navigation */
        .enhanced-nav {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid rgba(26, 35, 50, 0.1);
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 1000;
            transition: var(--transition-smooth);
        }

        .enhanced-nav.scrolled {
            background: rgba(255, 255, 255, 0.98);
            box-shadow: var(--shadow);
        }

        .nav-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            height: 80px;
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
        }

        .nav-link:hover,
        .nav-link.active {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }

        .dropdown-toggle {
            cursor: pointer;
        }

        .dropdown-arrow {
            font-size: 0.75rem;
            transition: transform var(--transition-smooth);
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
            box-shadow: var(--shadow-xl);
            min-width: 240px;
            opacity: 0;
            visibility: hidden;
            transform: translateY(-10px);
            transition: var(--transition-smooth);
            z-index: 1001;
            padding: 0.5rem;
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

        .nav-toggle {
            display: none;
            flex-direction: column;
            cursor: pointer;
            gap: 4px;
        }

        .nav-toggle span {
            width: 24px;
            height: 3px;
            background: var(--primary-color);
            transition: var(--transition);
            border-radius: 2px;
        }

        /* AI Chat Trigger Button */
        .ai-chat-trigger {
            background: linear-gradient(135deg, var(--accent-color), var(--accent-bright));
            color: white;
            border: none;
            border-radius: var(--radius-lg);
            padding: 0.75rem 1.25rem;
            margin-left: 1rem;
            cursor: pointer;
            transition: var(--transition);
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
            box-shadow: var(--shadow-lg);
            background: linear-gradient(135deg, var(--accent-bright), var(--accent-color));
        }

        .ai-chat-trigger .ai-label {
            margin-left: 0.5rem;
            font-weight: 600;
        }

        @media (max-width: 768px) {
            .ai-chat-trigger {
                margin-left: 0;
                margin-top: 1rem;
                width: 100%;
                justify-content: center;
            }
            
            .nav-container {
                flex-direction: column;
                align-items: center;
            }
            
            .ai-chat-trigger {
                order: 3;
                margin-top: 0.5rem;
                margin-bottom: 0.5rem;
            }
        }

        /* Page Layout */
        .page-header {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            color: white;
            padding: 8rem 0 4rem;
            position: relative;
            overflow: hidden;
            margin-top: 80px;
        }

        .page-header::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255, 215, 0, 0.03) 1px, transparent 1px);
            background-size: 40px 40px;
            animation: rotate-grid 180s linear infinite;
        }

        @keyframes rotate-grid {
            to { transform: rotate(360deg); }
        }

        .page-title {
            font-size: clamp(2.5rem, 6vw, 4rem);
            font-weight: 800;
            margin-bottom: 1rem;
            background: linear-gradient(45deg, #FFD700, #FFA500, #FF6B6B);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-family: var(--font-serif);
            text-align: center;
            position: relative;
            z-index: 1;
        }

        .page-subtitle {
            font-size: 1.3rem;
            opacity: 0.9;
            margin-bottom: 2rem;
            line-height: 1.6;
            text-align: center;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
            position: relative;
            z-index: 1;
        }

        .main-content {
            padding: 6rem 0;
        }

        .section {
            padding: 4rem 0;
        }

        .section-title {
            font-size: clamp(2rem, 4vw, 2.5rem);
            color: var(--primary-color);
            margin-bottom: 1.5rem;
            font-weight: 700;
            font-family: var(--font-serif);
            position: relative;
        }

        .section-title::after {
            content: '';
            position: absolute;
            bottom: -8px;
            left: 0;
            width: 60px;
            height: 3px;
            background: linear-gradient(90deg, var(--phi-gold), var(--phi-gold-light));
            border-radius: 2px;
        }

        /* Footer */
        .footer {
            background: var(--primary-color);
            color: white;
            padding: 4rem 0 2rem;
        }

        .footer-content {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 3rem;
            margin-bottom: 2rem;
        }

        .footer-section h3 {
            color: var(--phi-gold);
            margin-bottom: 1.5rem;
            font-size: 1.25rem;
            font-weight: 700;
        }

        .footer-section h4 {
            color: var(--phi-gold);
            margin-bottom: 1rem;
            font-size: 1.1rem;
            font-weight: 600;
        }

        .footer-section a {
            color: rgba(255, 255, 255, 0.8);
            text-decoration: none;
            display: block;
            margin-bottom: 0.5rem;
            transition: var(--transition);
        }

        .footer-section a:hover {
            color: var(--phi-gold);
            transform: translateX(5px);
        }

        .footer-section p {
            color: rgba(255, 255, 255, 0.7);
            line-height: 1.6;
        }

        .social-links {
            display: flex;
            gap: 1rem;
            margin-top: 1rem;
        }

        .social-links a {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 40px;
            height: 40px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 50%;
            color: var(--phi-gold);
            font-size: 1.2rem;
            transition: var(--transition);
        }

        .social-links a:hover {
            background: var(--phi-gold);
            color: white;
            transform: translateY(-2px);
        }

        .footer-bottom {
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            padding-top: 2rem;
            text-align: center;
            color: rgba(255, 255, 255, 0.7);
        }

        /* Responsive Design */
        @media (max-width: 768px) {
            .nav-menu {
                position: fixed;
                top: 80px;
                left: 0;
                right: 0;
                background: white;
                border-top: 1px solid var(--border-color);
                flex-direction: column;
                gap: 0;
                transform: translateY(-100%);
                opacity: 0;
                visibility: hidden;
                transition: var(--transition-smooth);
                padding: 2rem 0;
            }

            .nav-menu.active {
                transform: translateY(0);
                opacity: 1;
                visibility: visible;
            }

            .nav-toggle {
                display: flex;
            }

            .dropdown-menu {
                position: static;
                opacity: 1;
                visibility: visible;
                transform: none;
                box-shadow: none;
                border: none;
                background: var(--bg-secondary);
                margin: 1rem 0;
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

// Navigation JavaScript functionality
const navigationScript = `
    <script>
        document.addEventListener("DOMContentLoaded", function () {
            // Navigation scroll effect
            const nav = document.getElementById('enhancedUnityNav');
            if (nav) {
                window.addEventListener('scroll', () => {
                    if (window.scrollY > 50) {
                        nav.classList.add('scrolled');
                    } else {
                        nav.classList.remove('scrolled');
                    }
                });
            }

            // Mobile navigation
            const navToggle = document.getElementById('navToggle');
            const navMenu = document.querySelector('.nav-menu');

            if (navToggle && navMenu) {
                navToggle.addEventListener('click', function() {
                    navMenu.classList.toggle('active');
                });

                document.addEventListener('click', function(event) {
                    if (!navToggle.contains(event.target) && !navMenu.contains(event.target)) {
                        navMenu.classList.remove('active');
                    }
                });

                // Close menu when clicking on a link
                const navLinks = document.querySelectorAll('.nav-link, .dropdown-link');
                navLinks.forEach(link => {
                    link.addEventListener('click', function() {
                        navMenu.classList.remove('active');
                    });
                });
            }

            // Highlight current page
            const currentPage = window.location.pathname.split('/').pop() || 'index.html';
            const navLinks = document.querySelectorAll('.nav-link, .dropdown-link');
            navLinks.forEach(link => {
                if (link.getAttribute('href') === currentPage) {
                    link.classList.add('active');
                }
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

            // AI Chat Trigger Functionality
            const aiChatTrigger = document.getElementById('aiChatTrigger');
            if (aiChatTrigger) {
                aiChatTrigger.addEventListener('click', function() {
                    openAIChat();
                });
            }

            function openAIChat() {
                // Initialize AI chat if not already done
                if (typeof window.eenChat === 'undefined' || !window.eenChat) {
                    // Check if EenAIChat class is available
                    if (typeof EenAIChat !== 'undefined') {
                        // Initialize the chat system directly
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
        });
    </script>
`;

// Function to inject unified navigation and footer
function injectUnifiedNavigation() {
    // Inject navigation
    const navPlaceholder = document.getElementById('navigation-placeholder');
    if (navPlaceholder) {
        navPlaceholder.innerHTML = unifiedNavigation;
    } else {
        // Insert at beginning of body if no placeholder
        document.body.insertAdjacentHTML('afterbegin', unifiedNavigation);
    }

    // Inject footer
    const footerPlaceholder = document.getElementById('footer-placeholder');
    if (footerPlaceholder) {
        footerPlaceholder.innerHTML = unifiedFooter;
    } else {
        // Insert at end of body if no placeholder
        document.body.insertAdjacentHTML('beforeend', unifiedFooter);
    }

    // Inject styles
    document.head.insertAdjacentHTML('beforeend', unifiedStyles);

    // Inject scripts
    document.body.insertAdjacentHTML('beforeend', navigationScript);
}

// Auto-inject if this script is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', injectUnifiedNavigation);
} else {
    injectUnifiedNavigation();
}

// Export for manual use
window.EenNavigation = {
    inject: injectUnifiedNavigation,
    navigation: unifiedNavigation,
    footer: unifiedFooter,
    styles: unifiedStyles,
    script: navigationScript
};