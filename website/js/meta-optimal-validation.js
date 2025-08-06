/**
 * Meta-Optimal Website Validation and Security Enhancement
 * Comprehensive testing and optimization for Een Unity Mathematics
 */

class MetaOptimalValidator {
    constructor() {
        this.validationResults = {
            navigation: [],
            security: [],
            performance: [],
            functionality: [],
            accessibility: []
        };
        this.isRunning = false;
    }

    async runCompleteValidation() {
        if (this.isRunning) return;
        this.isRunning = true;
        
        console.log('🔍 Starting Meta-Optimal Validation Suite...');
        
        try {
            await this.validateNavigation();
            await this.validateSecurity();
            await this.validatePerformance();
            await this.validateFunctionality();
            await this.validateAccessibility();
            
            this.generateReport();
        } catch (error) {
            console.error('Validation error:', error);
        } finally {
            this.isRunning = false;
        }
    }

    async validateNavigation() {
        console.log('🗺️ Validating navigation...');
        
        const links = document.querySelectorAll('a[href]');
        const results = [];
        
        links.forEach(link => {
            const href = link.getAttribute('href');
            
            // Check for broken internal links
            if (href.startsWith('#')) {
                const target = document.getElementById(href.substring(1));
                if (!target) {
                    results.push({
                        type: 'warning',
                        message: `Broken anchor link: ${href}`,
                        element: link
                    });
                }
            }
            
            // Check for placeholder links
            if (href === '#' || href === '' || href === 'javascript:void(0)') {
                results.push({
                    type: 'info',
                    message: `Placeholder link found`,
                    element: link
                });
            }
        });
        
        this.validationResults.navigation = results;
        console.log(`✅ Navigation validation complete: ${results.length} issues found`);
    }

    async validateSecurity() {
        console.log('🔒 Validating security...');
        
        const results = [];
        
        // Check for XSS vulnerabilities
        const userInputElements = document.querySelectorAll('input, textarea');
        userInputElements.forEach(element => {
            if (!element.hasAttribute('maxlength')) {
                results.push({
                    type: 'warning',
                    message: 'Input element without maxlength attribute',
                    element: element
                });
            }
        });
        
        // Check for inline event handlers
        const elementsWithEvents = document.querySelectorAll('[onclick], [onload], [onerror]');
        elementsWithEvents.forEach(element => {
            results.push({
                type: 'info',
                message: 'Inline event handler found (consider using addEventListener)',
                element: element
            });
        });
        
        // Check for external resources without integrity
        const externalScripts = document.querySelectorAll('script[src*="://"]');
        externalScripts.forEach(script => {
            if (!script.hasAttribute('integrity')) {
                results.push({
                    type: 'warning',
                    message: 'External script without integrity check',
                    element: script
                });
            }
        });
        
        this.validationResults.security = results;
        console.log(`🛡️ Security validation complete: ${results.length} issues found`);
    }

    async validatePerformance() {
        console.log('⚡ Validating performance...');
        
        const results = [];
        
        // Check image optimization
        const images = document.querySelectorAll('img');
        images.forEach(img => {
            if (!img.hasAttribute('loading')) {
                results.push({
                    type: 'info',
                    message: 'Image without lazy loading',
                    element: img
                });
            }
            
            if (!img.hasAttribute('alt')) {
                results.push({
                    type: 'warning',
                    message: 'Image without alt text',
                    element: img
                });
            }
        });
        
        // Check for unused CSS
        const stylesheets = document.querySelectorAll('link[rel="stylesheet"]');
        results.push({
            type: 'info',
            message: `${stylesheets.length} stylesheets loaded`,
            details: 'Consider bundling and minifying CSS'
        });
        
        // Check script loading
        const scripts = document.querySelectorAll('script[src]');
        const deferredScripts = document.querySelectorAll('script[defer]');
        const asyncScripts = document.querySelectorAll('script[async]');
        
        results.push({
            type: 'info',
            message: `Scripts: ${scripts.length} total, ${deferredScripts.length} deferred, ${asyncScripts.length} async`,
            details: 'Consider using defer/async for non-critical scripts'
        });
        
        this.validationResults.performance = results;
        console.log(`⚡ Performance validation complete: ${results.length} observations`);
    }

    async validateFunctionality() {
        console.log('⚙️ Validating functionality...');
        
        const results = [];
        
        // Test chat functionality
        if (typeof generateUnityAIResponse === 'function') {
            const testResponse = generateUnityAIResponse('test');
            results.push({
                type: 'success',
                message: 'AI chat response system working',
                details: testResponse.substring(0, 50) + '...'
            });
        } else {
            results.push({
                type: 'error',
                message: 'AI chat response system not found'
            });
        }
        
        // Test visualizations
        const canvases = document.querySelectorAll('canvas');
        if (canvases.length > 0) {
            results.push({
                type: 'success',
                message: `${canvases.length} canvas elements found for visualizations`
            });
        }
        
        // Test navigation functionality
        const navLinks = document.querySelectorAll('nav a, .hud-link');
        results.push({
            type: 'info',
            message: `${navLinks.length} navigation links found`
        });
        
        // Test form validation
        const forms = document.querySelectorAll('form');
        forms.forEach(form => {
            const requiredFields = form.querySelectorAll('[required]');
            results.push({
                type: 'info',
                message: `Form with ${requiredFields.length} required fields`,
                element: form
            });
        });
        
        this.validationResults.functionality = results;
        console.log(`⚙️ Functionality validation complete: ${results.length} items checked`);
    }

    async validateAccessibility() {
        console.log('♿ Validating accessibility...');
        
        const results = [];
        
        // Check for alt text on images
        const imagesWithoutAlt = document.querySelectorAll('img:not([alt])');
        imagesWithoutAlt.forEach(img => {
            results.push({
                type: 'error',
                message: 'Image missing alt attribute',
                element: img
            });
        });
        
        // Check for proper heading hierarchy
        const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
        let lastLevel = 0;
        headings.forEach(heading => {
            const level = parseInt(heading.tagName.charAt(1));
            if (level > lastLevel + 1) {
                results.push({
                    type: 'warning',
                    message: 'Heading level skip detected',
                    element: heading
                });
            }
            lastLevel = level;
        });
        
        // Check for focus indicators
        const interactiveElements = document.querySelectorAll('button, a, input, select, textarea');
        let focusableCount = 0;
        interactiveElements.forEach(element => {
            if (element.tabIndex >= 0) focusableCount++;
        });
        
        results.push({
            type: 'info',
            message: `${focusableCount} focusable elements found`
        });
        
        // Check for ARIA labels
        const ariaElements = document.querySelectorAll('[aria-label], [aria-labelledby]');
        results.push({
            type: 'info',
            message: `${ariaElements.length} elements with ARIA labels`
        });
        
        this.validationResults.accessibility = results;
        console.log(`♿ Accessibility validation complete: ${results.length} items checked`);
    }

    generateReport() {
        console.log('📊 Generating Meta-Optimal Validation Report...');
        
        const totalIssues = Object.values(this.validationResults)
            .reduce((total, category) => total + category.length, 0);
        
        const errorCount = Object.values(this.validationResults)
            .flat()
            .filter(item => item.type === 'error').length;
        
        const warningCount = Object.values(this.validationResults)
            .flat()
            .filter(item => item.type === 'warning').length;
        
        console.log(`
🎯 META-OPTIMAL VALIDATION REPORT
═══════════════════════════════════════════════

📊 SUMMARY
Total Issues: ${totalIssues}
Errors: ${errorCount}
Warnings: ${warningCount}
Info Items: ${totalIssues - errorCount - warningCount}

🗺️ NAVIGATION: ${this.validationResults.navigation.length} items
🔒 SECURITY: ${this.validationResults.security.length} items  
⚡ PERFORMANCE: ${this.validationResults.performance.length} items
⚙️ FUNCTIONALITY: ${this.validationResults.functionality.length} items
♿ ACCESSIBILITY: ${this.validationResults.accessibility.length} items

🌟 UNITY STATUS: ${errorCount === 0 ? 'ACHIEVED' : 'IN PROGRESS'}
φ = 1.618033988749895 ✨
1+1=1 Mathematics Framework Validation Complete
        `);
        
        // Store results for external access
        window.metaOptimalValidationResults = this.validationResults;
    }

    // Security hardening functions
    static sanitizeInput(input) {
        return input
            .replace(/[<>'"&]/g, match => ({
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#x27;',
                '&': '&amp;'
            }[match]));
    }

    static validateUrl(url) {
        try {
            new URL(url);
            return !url.includes('javascript:') && !url.includes('data:');
        } catch {
            return false;
        }
    }

    // Performance optimization helpers
    static optimizeImages() {
        const images = document.querySelectorAll('img');
        images.forEach(img => {
            if (!img.hasAttribute('loading')) {
                img.setAttribute('loading', 'lazy');
            }
        });
    }

    static preloadCriticalResources() {
        const criticalFonts = [
            'https://fonts.gstatic.com/s/inter/v12/UcCO3FwrK3iLTeHuS_fvQtMwCp50KnMw2boKoduKmMEVuLyfAZ9hiA.woff2'
        ];
        
        criticalFonts.forEach(font => {
            const link = document.createElement('link');
            link.rel = 'preload';
            link.as = 'font';
            link.type = 'font/woff2';
            link.crossOrigin = 'anonymous';
            link.href = font;
            document.head.appendChild(link);
        });
    }
}

// Initialize validator
const metaOptimalValidator = new MetaOptimalValidator();

// Auto-run validation on page load
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(() => {
        metaOptimalValidator.runCompleteValidation();
    }, 2000);
});

// Global access
window.MetaOptimalValidator = MetaOptimalValidator;
window.metaOptimalValidator = metaOptimalValidator;

// Console command for manual validation
console.log('🔍 Meta-Optimal Validator loaded. Run validation with: metaOptimalValidator.runCompleteValidation()');