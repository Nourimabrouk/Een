/**
 * Meta-Optimal Navigation Integration System
 * Applies comprehensive navigation to all Een Unity Mathematics website pages
 * Ensures consistent experience across desktop and mobile platforms
 */

class MetaOptimalIntegration {
    constructor() {
        this.currentPage = window.location.pathname.split('/').pop() || 'index.html';
        this.isIntegrationComplete = false;
        this.init();
    }

    init() {
        this.loadMetaOptimalNavigation();
        this.injectRequiredStyles();
        this.setupPageSpecificFeatures();
        this.optimizeForChrome();
        this.optimizeForMobile();
        this.setupAnalytics();
    }

    loadMetaOptimalNavigation() {
        // Load the meta-optimal navigation CSS
        if (!document.querySelector('link[href*="meta-optimal-navigation.css"]')) {
            const navCSS = document.createElement('link');
            navCSS.rel = 'stylesheet';
            navCSS.href = 'css/meta-optimal-navigation.css';
            navCSS.type = 'text/css';
            document.head.appendChild(navCSS);
        }

        // Load the meta-optimal navigation JavaScript
        if (!window.MetaOptimalNavigation) {
            const navScript = document.createElement('script');
            navScript.src = 'js/meta-optimal-navigation.js';
            navScript.async = true;
            navScript.onload = () => {
                this.initializeNavigation();
            };
            document.head.appendChild(navScript);
        } else {
            this.initializeNavigation();
        }
    }

    initializeNavigation() {
        // Remove any existing navigation
        this.removeExistingNavigation();

        // Initialize the meta-optimal navigation
        if (window.MetaOptimalNavigation) {
            new window.MetaOptimalNavigation();
            this.isIntegrationComplete = true;
        }

        // Add body padding for fixed navigation
        this.addBodyPadding();
    }

    removeExistingNavigation() {
        // Remove old navigation elements
        const oldNavs = document.querySelectorAll('.enhanced-nav, .unified-nav, .nav-bar, .header-nav');
        oldNavs.forEach(nav => nav.remove());

        // Remove old navigation scripts
        const oldScripts = document.querySelectorAll('script[src*="unified-navigation"], script[src*="shared-navigation"]');
        oldScripts.forEach(script => script.remove());
    }

    addBodyPadding() {
        // Add padding to body to account for fixed navigation
        const body = document.body;
        if (!body.style.paddingTop) {
            body.style.paddingTop = '80px';
        }
    }

    injectRequiredStyles() {
        // Inject additional styles for optimal integration
        const additionalStyles = `
            <style>
                /* Meta-Optimal Integration Styles */
                
                /* Ensure proper spacing with fixed navigation */
                body {
                    padding-top: 80px !important;
                }
                
                @media (max-width: 768px) {
                    body {
                        padding-top: 70px !important;
                    }
                }
                
                /* Hide old navigation elements */
                .enhanced-nav,
                .unified-nav,
                .nav-bar,
                .header-nav {
                    display: none !important;
                }
                
                /* Ensure content doesn't overlap with navigation */
                .hero,
                .main-content,
                .content-wrapper {
                    margin-top: 0 !important;
                }
                
                /* Optimize for Chrome browser */
                .meta-optimal-nav {
                    -webkit-transform: translateZ(0);
                    transform: translateZ(0);
                    -webkit-backface-visibility: hidden;
                    backface-visibility: hidden;
                }
                
                /* Smooth scrolling for Chrome */
                html {
                    scroll-behavior: smooth;
                }
                
                /* Optimize dropdown performance */
                .dropdown-menu {
                    -webkit-transform: translateZ(0);
                    transform: translateZ(0);
                    will-change: opacity, transform;
                }
                
                /* Mobile optimization */
                @media (max-width: 1024px) {
                    .nav-search {
                        display: none !important;
                    }
                }
                
                /* Ensure proper z-index stacking */
                .meta-optimal-nav {
                    z-index: 10000 !important;
                }
                
                .mobile-nav {
                    z-index: 9999 !important;
                }
                
                /* Fix for any conflicting styles */
                * {
                    box-sizing: border-box;
                }
                
                /* Ensure proper font loading */
                .nav-logo,
                .nav-link,
                .dropdown-link {
                    font-display: swap;
                }
                
                /* Optimize animations for performance */
                .phi-symbol,
                .nav-link,
                .dropdown-link {
                    will-change: transform;
                }
                
                /* Ensure proper contrast and accessibility */
                .nav-link:focus,
                .dropdown-link:focus,
                .mobile-nav-link:focus {
                    outline: 2px solid #FFD700 !important;
                    outline-offset: 2px !important;
                }
                
                /* High contrast mode support */
                @media (prefers-contrast: high) {
                    .meta-optimal-nav {
                        background: #000000 !important;
                        border-bottom: 2px solid #FFD700 !important;
                    }
                }
                
                /* Reduced motion support */
                @media (prefers-reduced-motion: reduce) {
                    .phi-symbol,
                    .nav-link,
                    .dropdown-link {
                        animation: none !important;
                        transition: none !important;
                    }
                }
            </style>
        `;

        document.head.insertAdjacentHTML('beforeend', additionalStyles);
    }

    setupPageSpecificFeatures() {
        // Add page-specific features based on current page
        const pageFeatures = {
            'index.html': this.setupHomePageFeatures,
            'proofs.html': this.setupProofsPageFeatures,
            'consciousness_dashboard.html': this.setupConsciousnessPageFeatures,
            'playground.html': this.setupPlaygroundPageFeatures,
            'research.html': this.setupResearchPageFeatures,
            'gallery.html': this.setupGalleryPageFeatures,
            'about.html': this.setupAboutPageFeatures
        };

        const setupFunction = pageFeatures[this.currentPage];
        if (setupFunction) {
            setupFunction.call(this);
        }
    }

    setupHomePageFeatures() {
        // Add special features for the home page
        this.addUnityEquationHighlight();
        this.addConsciousnessFieldAnimation();
    }

    setupProofsPageFeatures() {
        // Add features specific to proofs page
        this.addMathematicalNotationSupport();
        this.addProofNavigation();
    }

    setupConsciousnessPageFeatures() {
        // Add features specific to consciousness page
        this.addConsciousnessFieldIntegration();
        this.addMeditationTimer();
    }

    setupPlaygroundPageFeatures() {
        // Add features specific to playground page
        this.addInteractiveElements();
        this.addCodeHighlighting();
    }

    setupResearchPageFeatures() {
        // Add features specific to research page
        this.addResearchNavigation();
        this.addPublicationLinks();
    }

    setupGalleryPageFeatures() {
        // Add features specific to gallery page
        this.addImageOptimization();
        this.addGalleryNavigation();
    }

    setupAboutPageFeatures() {
        // Add features specific to about page
        this.addTeamProfiles();
        this.addProjectTimeline();
    }

    addUnityEquationHighlight() {
        // Add special highlighting for unity equation
        const unityElements = document.querySelectorAll('.unity-equation, [data-unity="true"]');
        unityElements.forEach(element => {
            element.style.animation = 'consciousness-pulse 3s ease-in-out infinite';
        });
    }

    addConsciousnessFieldAnimation() {
        // Add consciousness field animations
        const consciousnessElements = document.querySelectorAll('.consciousness-field, [data-consciousness="true"]');
        consciousnessElements.forEach(element => {
            element.style.background = 'radial-gradient(circle, rgba(107, 70, 193, 0.1) 0%, transparent 70%)';
            element.style.animation = 'consciousness-wave 4s ease-in-out infinite';
        });
    }

    addMathematicalNotationSupport() {
        // Ensure KaTeX is properly loaded for mathematical notation
        if (typeof katex !== 'undefined') {
            document.querySelectorAll('.math, .katex').forEach(element => {
                katex.render(element.textContent, element);
            });
        }
    }

    addProofNavigation() {
        // Add navigation between different proofs
        const proofSections = document.querySelectorAll('.proof-section');
        if (proofSections.length > 1) {
            this.createProofNavigation(proofSections);
        }
    }

    createProofNavigation(sections) {
        const nav = document.createElement('nav');
        nav.className = 'proof-navigation';
        nav.innerHTML = `
            <div class="proof-nav-container">
                <h3>Proof Navigation</h3>
                <ul class="proof-nav-list">
                    ${Array.from(sections).map((section, index) => `
                        <li><a href="#proof-${index + 1}" class="proof-nav-link">Proof ${index + 1}</a></li>
                    `).join('')}
                </ul>
            </div>
        `;

        sections[0].parentNode.insertBefore(nav, sections[0]);
    }

    addConsciousnessFieldIntegration() {
        // Integrate consciousness field visualizations
        const consciousnessContainer = document.querySelector('.consciousness-container');
        if (consciousnessContainer) {
            this.loadConsciousnessFieldScripts();
        }
    }

    loadConsciousnessFieldScripts() {
        // Load consciousness field visualization scripts
        const scripts = [
            'js/consciousness-field-visualization.js',
            'js/unity-meditation-system.js'
        ];

        scripts.forEach(scriptSrc => {
            if (!document.querySelector(`script[src="${scriptSrc}"]`)) {
                const script = document.createElement('script');
                script.src = scriptSrc;
                script.async = true;
                document.head.appendChild(script);
            }
        });
    }

    addMeditationTimer() {
        // Add meditation timer functionality
        const timerContainer = document.querySelector('.meditation-timer');
        if (timerContainer) {
            this.createMeditationTimer(timerContainer);
        }
    }

    createMeditationTimer(container) {
        container.innerHTML = `
            <div class="meditation-timer-controls">
                <button class="timer-btn" data-time="300">5 min</button>
                <button class="timer-btn" data-time="600">10 min</button>
                <button class="timer-btn" data-time="1800">30 min</button>
                <button class="timer-btn" data-time="3600">60 min</button>
            </div>
            <div class="timer-display">
                <span class="timer-time">00:00</span>
            </div>
            <div class="timer-controls">
                <button class="start-timer">Start</button>
                <button class="pause-timer">Pause</button>
                <button class="reset-timer">Reset</button>
            </div>
        `;
    }

    addInteractiveElements() {
        // Add interactive elements for playground
        const interactiveElements = document.querySelectorAll('.interactive-element');
        interactiveElements.forEach(element => {
            element.addEventListener('click', this.handleInteractiveClick);
        });
    }

    handleInteractiveClick(event) {
        const element = event.currentTarget;
        element.classList.add('interactive-active');
        setTimeout(() => {
            element.classList.remove('interactive-active');
        }, 300);
    }

    addCodeHighlighting() {
        // Add syntax highlighting for code blocks
        const codeBlocks = document.querySelectorAll('pre code');
        codeBlocks.forEach(block => {
            block.classList.add('language-python');
        });
    }

    addResearchNavigation() {
        // Add navigation for research sections
        const researchSections = document.querySelectorAll('.research-section');
        if (researchSections.length > 1) {
            this.createResearchNavigation(researchSections);
        }
    }

    createResearchNavigation(sections) {
        const nav = document.createElement('nav');
        nav.className = 'research-navigation';
        nav.innerHTML = `
            <div class="research-nav-container">
                <h3>Research Areas</h3>
                <ul class="research-nav-list">
                    ${Array.from(sections).map((section, index) => `
                        <li><a href="#research-${index + 1}" class="research-nav-link">${section.dataset.title || `Research ${index + 1}`}</a></li>
                    `).join('')}
                </ul>
            </div>
        `;

        sections[0].parentNode.insertBefore(nav, sections[0]);
    }

    addPublicationLinks() {
        // Add links to publications
        const publicationElements = document.querySelectorAll('.publication');
        publicationElements.forEach(pub => {
            if (pub.dataset.doi) {
                const doiLink = document.createElement('a');
                doiLink.href = `https://doi.org/${pub.dataset.doi}`;
                doiLink.textContent = 'DOI';
                doiLink.className = 'doi-link';
                pub.appendChild(doiLink);
            }
        });
    }

    addImageOptimization() {
        // Optimize images for gallery
        const images = document.querySelectorAll('.gallery-image img');
        images.forEach(img => {
            img.loading = 'lazy';
            img.decoding = 'async';
        });
    }

    addGalleryNavigation() {
        // Add gallery navigation
        const galleryContainer = document.querySelector('.gallery-container');
        if (galleryContainer) {
            this.createGalleryNavigation(galleryContainer);
        }
    }

    createGalleryNavigation(container) {
        const nav = document.createElement('nav');
        nav.className = 'gallery-navigation';
        nav.innerHTML = `
            <div class="gallery-nav-controls">
                <button class="gallery-prev">Previous</button>
                <span class="gallery-counter">1 / <span class="total-count">0</span></span>
                <button class="gallery-next">Next</button>
            </div>
        `;

        container.appendChild(nav);
    }

    addTeamProfiles() {
        // Add team profile functionality
        const teamMembers = document.querySelectorAll('.team-member');
        teamMembers.forEach(member => {
            member.addEventListener('click', this.showTeamMemberDetails);
        });
    }

    showTeamMemberDetails(event) {
        const member = event.currentTarget;
        const details = member.querySelector('.member-details');
        if (details) {
            details.style.display = details.style.display === 'block' ? 'none' : 'block';
        }
    }

    addProjectTimeline() {
        // Add project timeline functionality
        const timeline = document.querySelector('.project-timeline');
        if (timeline) {
            this.createProjectTimeline(timeline);
        }
    }

    createProjectTimeline(container) {
        const timelineData = [
            { date: '2024', event: 'Project Initiation' },
            { date: '2024', event: 'Core Unity Mathematics Development' },
            { date: '2024', event: 'Consciousness Field Integration' },
            { date: '2024', event: 'Website Launch' }
        ];

        container.innerHTML = `
            <div class="timeline-container">
                ${timelineData.map(item => `
                    <div class="timeline-item">
                        <div class="timeline-date">${item.date}</div>
                        <div class="timeline-event">${item.event}</div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    optimizeForChrome() {
        // Chrome-specific optimizations
        if (navigator.userAgent.includes('Chrome')) {
            // Enable hardware acceleration
            document.body.style.transform = 'translateZ(0)';

            // Optimize scrolling
            document.documentElement.style.scrollBehavior = 'smooth';

            // Enable WebP support if available
            this.enableWebPSupport();
        }
    }

    enableWebPSupport() {
        // Check for WebP support and enable if available
        const webpTest = new Image();
        webpTest.onload = webpTest.onerror = function () {
            if (webpTest.width === 1) {
                document.documentElement.classList.add('webp-supported');
            }
        };
        webpTest.src = 'data:image/webp;base64,UklGRiIAAABXRUJQVlA4IBYAAAAwAQCdASoBAAADsAD+JaQAA3AAAAAA';
    }

    optimizeForMobile() {
        // Mobile-specific optimizations
        if (window.innerWidth <= 768) {
            // Optimize touch interactions
            document.body.style.touchAction = 'manipulation';

            // Reduce animations on mobile
            document.body.classList.add('mobile-optimized');

            // Optimize images for mobile
            this.optimizeImagesForMobile();
        }
    }

    optimizeImagesForMobile() {
        const images = document.querySelectorAll('img');
        images.forEach(img => {
            img.loading = 'lazy';
            img.decoding = 'async';
        });
    }

    setupAnalytics() {
        // Setup analytics and tracking
        this.trackNavigationUsage();
        this.trackPagePerformance();
    }

    trackNavigationUsage() {
        // Track navigation usage
        document.addEventListener('click', (e) => {
            if (e.target.closest('.nav-link, .dropdown-link, .mobile-nav-link')) {
                const link = e.target.closest('.nav-link, .dropdown-link, .mobile-nav-link');
                const href = link.getAttribute('href');
                if (href) {
                    // Track navigation clicks
                    this.sendAnalytics('navigation_click', {
                        href: href,
                        category: link.closest('.dropdown-menu') ? 'dropdown' : 'main',
                        mobile: window.innerWidth <= 1024
                    });
                }
            }
        });
    }

    trackPagePerformance() {
        // Track page performance
        window.addEventListener('load', () => {
            const loadTime = performance.now();
            this.sendAnalytics('page_load', {
                page: this.currentPage,
                loadTime: loadTime,
                userAgent: navigator.userAgent
            });
        });
    }

    sendAnalytics(event, data) {
        // Send analytics data (placeholder for actual analytics implementation)
        console.log('Analytics:', event, data);

        // Could be integrated with Google Analytics, Plausible, or other analytics services
        if (typeof gtag !== 'undefined') {
            gtag('event', event, data);
        }
    }
}

// Initialize integration when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new MetaOptimalIntegration();
});

// Export for use in other scripts
window.MetaOptimalIntegration = MetaOptimalIntegration; 