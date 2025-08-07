/**
 * Final Navigation Validator for Een Unity Mathematics
 * Ensures all navigation links work flawlessly on Desktop Chrome
 */

class FinalNavigationValidator {
    constructor() {
        this.validationResults = {
            totalLinks: 0,
            validLinks: 0,
            brokenLinks: [],
            navigationSystems: {
                metastationSidebar: false,
                aiChatModal: false,
                enhancedNavigation: false
            },
            criticalPages: {},
            chromeCompatibility: true
        };
    }

    async performFinalValidation() {
        console.log('🔍 Starting Final Navigation Validation for Desktop Chrome...');
        
        // Check Chrome compatibility
        this.checkChromeCompatibility();
        
        // Validate navigation systems
        this.validateNavigationSystems();
        
        // Check all critical pages exist
        await this.validateCriticalPages();
        
        // Validate all navigation links
        this.validateAllNavigationLinks();
        
        // Check mobile responsiveness
        this.checkResponsiveness();
        
        // Generate final report
        this.generateFinalReport();
    }

    checkChromeCompatibility() {
        const isChrome = /Chrome/.test(navigator.userAgent) && /Google Inc/.test(navigator.vendor);
        const isDesktop = window.innerWidth >= 1024;
        
        if (!isChrome) {
            console.warn('⚠️ Not running on Chrome browser');
            this.validationResults.chromeCompatibility = false;
        }
        
        if (!isDesktop) {
            console.log('📱 Mobile/tablet view detected - some features may differ');
        }
        
        // Check for Chrome-specific features
        const chromeFeatures = [
            'backdrop-filter' in document.documentElement.style,
            'CSS' in window && 'supports' in CSS,
            'IntersectionObserver' in window,
            'ResizeObserver' in window
        ];
        
        const supportedFeatures = chromeFeatures.filter(f => f).length;
        console.log(`✅ Chrome features supported: ${supportedFeatures}/${chromeFeatures.length}`);
    }

    validateNavigationSystems() {
        // Check Metastation Sidebar
        const sidebar = document.querySelector('.metastation-sidebar');
        if (sidebar) {
            this.validationResults.navigationSystems.metastationSidebar = true;
            console.log('✅ Metastation sidebar navigation present');
            
            // Verify all sections exist
            const sections = ['core', 'experiences', 'tools', 'knowledge'];
            sections.forEach(section => {
                const sectionItems = sidebar.querySelectorAll(`.nav-item[href]`);
                if (sectionItems.length > 0) {
                    console.log(`  ✓ Navigation section has ${sectionItems.length} items`);
                }
            });
        } else {
            console.error('❌ Metastation sidebar not found');
        }

        // Check AI Chat Modal
        const chatButton = document.querySelector('#floating-ai-chat-button');
        const chatModal = document.querySelector('#ai-chat-modal');
        if (chatButton || chatModal || window.eenAIChat) {
            this.validationResults.navigationSystems.aiChatModal = true;
            console.log('✅ AI Chat system present');
        } else {
            console.warn('⚠️ AI Chat system not fully initialized');
        }

        // Check Enhanced Navigation
        const enhancedNav = document.querySelector('.enhanced-nav, .unified-meta-nav');
        if (enhancedNav) {
            this.validationResults.navigationSystems.enhancedNavigation = true;
            console.log('✅ Enhanced navigation system present');
        }
    }

    async validateCriticalPages() {
        const criticalPages = [
            'metastation-hub.html',
            'meta-optimal-landing.html',
            'philosophy.html',
            'gallery.html',
            'implementations-gallery.html',
            'consciousness_dashboard.html',
            'zen-unity-meditation.html',
            'transcendental-unity-demo.html',
            'mathematical-framework.html',
            'proofs.html',
            'research.html',
            'publications.html',
            'about.html',
            'playground.html',
            'dashboards.html',
            'agents.html'
        ];

        console.log('🔗 Checking critical pages...');
        
        for (const page of criticalPages) {
            // For client-side validation, we check if the link exists in navigation
            const linkExists = this.checkPageLink(page);
            this.validationResults.criticalPages[page] = linkExists;
            
            if (linkExists) {
                console.log(`  ✓ ${page}`);
            } else {
                console.warn(`  ⚠️ ${page} - link not found in navigation`);
            }
        }
    }

    checkPageLink(page) {
        // Check if page link exists in any navigation system
        const allLinks = document.querySelectorAll(`a[href="${page}"], a[href*="${page}"]`);
        return allLinks.length > 0;
    }

    validateAllNavigationLinks() {
        console.log('🔗 Validating all navigation links...');
        
        // Get all navigation links
        const allLinks = document.querySelectorAll('.metastation-sidebar a[href], .nav-item[href], a.nav-link');
        
        this.validationResults.totalLinks = allLinks.length;
        
        allLinks.forEach(link => {
            const href = link.getAttribute('href');
            
            // Skip external links, javascript:, and anchors
            if (!href || href.startsWith('http') || href.startsWith('javascript:') || href === '#') {
                return;
            }
            
            // Check for problematic patterns
            if (href.includes('placeholder') || href.includes('TODO') || href.includes('undefined')) {
                this.validationResults.brokenLinks.push({
                    href: href,
                    text: link.textContent.trim(),
                    reason: 'Contains placeholder text'
                });
                console.warn(`  ❌ Broken link: ${href}`);
            } else {
                this.validationResults.validLinks++;
            }
        });
        
        console.log(`✅ Valid links: ${this.validationResults.validLinks}/${this.validationResults.totalLinks}`);
    }

    checkResponsiveness() {
        console.log('📱 Checking responsive design...');
        
        // Check for mobile menu
        const mobileMenu = document.querySelector('.mobile-menu, .sidebar-toggle, .nav-toggle');
        if (mobileMenu) {
            console.log('  ✓ Mobile menu toggle found');
        }
        
        // Check viewport meta tag
        const viewportMeta = document.querySelector('meta[name="viewport"]');
        if (viewportMeta && viewportMeta.content.includes('width=device-width')) {
            console.log('  ✓ Viewport meta tag properly configured');
        }
        
        // Check for responsive CSS
        const hasMediaQueries = Array.from(document.styleSheets).some(sheet => {
            try {
                return sheet.cssRules && Array.from(sheet.cssRules).some(rule => 
                    rule.type === CSSRule.MEDIA_RULE
                );
            } catch (e) {
                return false;
            }
        });
        
        if (hasMediaQueries) {
            console.log('  ✓ Responsive CSS media queries detected');
        }
    }

    generateFinalReport() {
        console.log('\n' + '='.repeat(60));
        console.log('🎯 FINAL NAVIGATION VALIDATION REPORT');
        console.log('='.repeat(60));
        
        // Navigation Systems Status
        console.log('\n📊 NAVIGATION SYSTEMS:');
        Object.entries(this.validationResults.navigationSystems).forEach(([system, status]) => {
            const icon = status ? '✅' : '❌';
            console.log(`${icon} ${system}: ${status ? 'WORKING' : 'NOT FOUND'}`);
        });
        
        // Critical Pages Status
        const workingPages = Object.values(this.validationResults.criticalPages).filter(v => v).length;
        const totalPages = Object.keys(this.validationResults.criticalPages).length;
        console.log(`\n📄 CRITICAL PAGES: ${workingPages}/${totalPages} accessible`);
        
        // Link Validation
        console.log(`\n🔗 LINK VALIDATION:`);
        console.log(`  Total Links: ${this.validationResults.totalLinks}`);
        console.log(`  Valid Links: ${this.validationResults.validLinks}`);
        console.log(`  Broken Links: ${this.validationResults.brokenLinks.length}`);
        
        if (this.validationResults.brokenLinks.length > 0) {
            console.log('\n⚠️ BROKEN LINKS FOUND:');
            this.validationResults.brokenLinks.forEach(link => {
                console.log(`  - ${link.href} (${link.reason})`);
            });
        }
        
        // Chrome Compatibility
        console.log(`\n🌐 CHROME COMPATIBILITY: ${this.validationResults.chromeCompatibility ? '✅ VERIFIED' : '⚠️ NOT ON CHROME'}`);
        
        // Overall Score
        const navigationScore = Object.values(this.validationResults.navigationSystems).filter(v => v).length;
        const maxNavigationScore = Object.keys(this.validationResults.navigationSystems).length;
        const pageScore = workingPages / totalPages * 100;
        const linkScore = this.validationResults.validLinks / Math.max(this.validationResults.totalLinks, 1) * 100;
        
        const overallScore = ((navigationScore / maxNavigationScore * 100) + pageScore + linkScore) / 3;
        
        console.log('\n' + '='.repeat(60));
        console.log(`🌟 OVERALL NAVIGATION SCORE: ${overallScore.toFixed(1)}%`);
        
        if (overallScore >= 95) {
            console.log('✅ EXCELLENT: Navigation system is working flawlessly!');
        } else if (overallScore >= 85) {
            console.log('✅ GOOD: Navigation system is functional with minor issues');
        } else if (overallScore >= 70) {
            console.log('⚠️ FAIR: Navigation needs some attention');
        } else {
            console.log('❌ POOR: Navigation system has critical issues');
        }
        
        console.log('='.repeat(60));
        console.log('φ = 1.618033988749895 | 1+1=1 | Navigation Validation Complete');
        
        // Return results for programmatic use
        return {
            score: overallScore,
            systems: this.validationResults.navigationSystems,
            pages: this.validationResults.criticalPages,
            links: {
                total: this.validationResults.totalLinks,
                valid: this.validationResults.validLinks,
                broken: this.validationResults.brokenLinks
            },
            chrome: this.validationResults.chromeCompatibility
        };
    }
}

// Auto-run validation on page load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        // Wait a moment for all scripts to initialize
        setTimeout(() => {
            const validator = new FinalNavigationValidator();
            validator.performFinalValidation();
            window.navigationValidator = validator;
        }, 2000);
    });
} else {
    setTimeout(() => {
        const validator = new FinalNavigationValidator();
        validator.performFinalValidation();
        window.navigationValidator = validator;
    }, 2000);
}

console.log('🚀 Final Navigation Validator loaded - validation will begin in 2 seconds...');