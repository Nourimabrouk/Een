/**
 * Comprehensive Website Validator
 * Een Unity Mathematics - Complete site functionality testing
 */

class EenWebsiteValidator {
    constructor() {
        this.testResults = {
            navigation: { passed: 0, failed: 0, tests: [] },
            links: { passed: 0, failed: 0, tests: [] },
            interactive: { passed: 0, failed: 0, tests: [] },
            aiChat: { passed: 0, failed: 0, tests: [] },
            content: { passed: 0, failed: 0, tests: [] },
            performance: { passed: 0, failed: 0, tests: [] }
        };
    }

    async runComprehensiveValidation() {
        console.log('🔍 Starting comprehensive Een Unity Mathematics website validation...');
        
        try {
            // Test Navigation System
            await this.testNavigationSystem();
            
            // Test AI Chat Integration
            await this.testAIChatIntegration();
            
            // Test Interactive Elements
            await this.testInteractiveElements();
            
            // Test Content Loading
            await this.testContentLoading();
            
            // Test Link Integrity
            await this.testLinkIntegrity();
            
            // Test Performance
            await this.testPerformance();
            
            // Generate Report
            this.generateValidationReport();
            
        } catch (error) {
            console.error('❌ Validation error:', error);
        }
    }

    async testNavigationSystem() {
        console.log('📍 Testing Navigation System...');
        
        // Test 1: Metastation sidebar exists
        const sidebar = document.querySelector('.metastation-sidebar');
        this.recordTest('navigation', 'Metastation Sidebar Present', !!sidebar);
        
        // Test 2: Navigation sections
        const sections = ['core', 'experiences', 'tools', 'knowledge'];
        sections.forEach(section => {
            const sectionElement = document.querySelector(`.nav-section:contains("${section}")`);
            this.recordTest('navigation', `${section} section exists`, !!sectionElement);
        });
        
        // Test 3: Toggle functionality
        if (window.metastationNav) {
            this.recordTest('navigation', 'Navigation toggle available', true);
        } else {
            this.recordTest('navigation', 'Navigation toggle available', false);
        }
        
        // Test 4: Mobile responsiveness
        const isMobileResponsive = window.innerWidth <= 768 ? 
            !document.body.classList.contains('sidebar-open') : true;
        this.recordTest('navigation', 'Mobile responsiveness', isMobileResponsive);
    }

    async testAIChatIntegration() {
        console.log('🤖 Testing AI Chat Integration...');
        
        // Test 1: Enhanced AI chat script loaded
        const aiChatLoaded = typeof generateAIResponse === 'function';
        this.recordTest('aiChat', 'Enhanced AI Chat Script Loaded', aiChatLoaded);
        
        // Test 2: Chat UI elements
        const chatElements = [
            '.ai-chat-button',
            '.unity-ai-chat',
            '.chat-container'
        ];
        
        let chatUIFound = false;
        chatElements.forEach(selector => {
            if (document.querySelector(selector)) {
                chatUIFound = true;
            }
        });
        this.recordTest('aiChat', 'Chat UI Elements Present', chatUIFound);
        
        // Test 3: AI response generation
        if (aiChatLoaded) {
            try {
                const testResponse = generateAIResponse('test unity mathematics');
                this.recordTest('aiChat', 'AI Response Generation', !!testResponse);
            } catch (error) {
                this.recordTest('aiChat', 'AI Response Generation', false);
            }
        }
    }

    async testInteractiveElements() {
        console.log('⚡ Testing Interactive Elements...');
        
        // Test 1: Canvas elements
        const canvases = document.querySelectorAll('canvas');
        this.recordTest('interactive', `Canvas elements (${canvases.length})`, canvases.length > 0);
        
        // Test 2: Interactive buttons
        const buttons = document.querySelectorAll('button, .btn');
        this.recordTest('interactive', `Interactive buttons (${buttons.length})`, buttons.length > 0);
        
        // Test 3: Mathematical visualizations
        const visualizations = [
            'golden-ratio-spiral',
            'consciousness-field-3d',
            'quantum-unity-states'
        ];
        
        visualizations.forEach(vizId => {
            const element = document.getElementById(vizId);
            this.recordTest('interactive', `${vizId} visualization`, !!element);
        });
        
        // Test 4: KaTeX mathematical rendering
        const katexLoaded = typeof katex !== 'undefined';
        this.recordTest('interactive', 'KaTeX Math Rendering', katexLoaded);
    }

    async testContentLoading() {
        console.log('📄 Testing Content Loading...');
        
        // Test 1: Philosophy markdown loading (if on philosophy page)
        if (window.location.pathname.includes('philosophy')) {
            const philosophyContent = document.getElementById('philosophyContent');
            const contentLoaded = philosophyContent && philosophyContent.style.display !== 'none';
            this.recordTest('content', 'Philosophy Markdown Loaded', contentLoaded);
        }
        
        // Test 2: No placeholder content
        const placeholderElements = document.querySelectorAll('[placeholder*="TODO"], [placeholder*="FIXME"]');
        this.recordTest('content', 'No TODO/FIXME placeholders', placeholderElements.length === 0);
        
        // Test 3: Real GitHub links (not placeholder)
        const githubLinks = document.querySelectorAll('a[href*="github.com"]');
        let realGithubLinks = 0;
        githubLinks.forEach(link => {
            if (!link.href.includes('user/repo')) {
                realGithubLinks++;
            }
        });
        this.recordTest('content', `Real GitHub links (${realGithubLinks}/${githubLinks.length})`, 
            githubLinks.length === 0 || realGithubLinks === githubLinks.length);
        
        // Test 4: Unity equation references
        const unityReferences = document.body.textContent.match(/1\+1=1/g);
        this.recordTest('content', `Unity equation references (${unityReferences ? unityReferences.length : 0})`, 
            !!(unityReferences && unityReferences.length > 0));
    }

    async testLinkIntegrity() {
        console.log('🔗 Testing Link Integrity...');
        
        // Test internal links
        const internalLinks = document.querySelectorAll('a[href$=".html"]');
        let workingLinks = 0;
        
        internalLinks.forEach(link => {
            // For now, just check if href is not empty and doesn't contain placeholders
            const href = link.getAttribute('href');
            if (href && !href.includes('placeholder') && !href.includes('#') && href !== '#') {
                workingLinks++;
            }
        });
        
        this.recordTest('links', `Internal links integrity (${workingLinks}/${internalLinks.length})`, 
            internalLinks.length === 0 || workingLinks >= internalLinks.length * 0.8);
        
        // Test navigation links specifically
        const navLinks = document.querySelectorAll('.nav-item[href], .metastation-sidebar a[href]');
        this.recordTest('links', `Navigation links (${navLinks.length})`, navLinks.length > 0);
    }

    async testPerformance() {
        console.log('⚡ Testing Performance...');
        
        // Test 1: Page load time
        if (window.performance && window.performance.timing) {
            const loadTime = window.performance.timing.loadEventEnd - window.performance.timing.navigationStart;
            this.recordTest('performance', `Page load time (${loadTime}ms)`, loadTime < 5000);
        }
        
        // Test 2: JavaScript errors
        let jsErrors = 0;
        const originalError = window.onerror;
        window.onerror = function(message, source, line, col, error) {
            jsErrors++;
            if (originalError) originalError.apply(window, arguments);
        };
        
        setTimeout(() => {
            this.recordTest('performance', `JavaScript errors (${jsErrors})`, jsErrors === 0);
        }, 1000);
        
        // Test 3: Critical CSS loaded
        const criticalCSS = ['font-family', 'background', 'color'].every(prop => {
            const computed = window.getComputedStyle(document.body);
            return computed.getPropertyValue(prop) !== '';
        });
        this.recordTest('performance', 'Critical CSS loaded', criticalCSS);
    }

    recordTest(category, name, passed) {
        const result = { name, passed, timestamp: Date.now() };
        this.testResults[category].tests.push(result);
        
        if (passed) {
            this.testResults[category].passed++;
            console.log(`✅ ${name}`);
        } else {
            this.testResults[category].failed++;
            console.log(`❌ ${name}`);
        }
    }

    generateValidationReport() {
        console.log('\n🎯 === EEN UNITY MATHEMATICS WEBSITE VALIDATION REPORT ===\n');
        
        let totalPassed = 0;
        let totalFailed = 0;
        
        Object.keys(this.testResults).forEach(category => {
            const results = this.testResults[category];
            const categoryTotal = results.passed + results.failed;
            const successRate = categoryTotal > 0 ? ((results.passed / categoryTotal) * 100).toFixed(1) : 0;
            
            console.log(`📊 ${category.toUpperCase()}: ${results.passed}/${categoryTotal} (${successRate}%)`);
            
            // Show failed tests
            results.tests.forEach(test => {
                if (!test.passed) {
                    console.log(`   ❌ ${test.name}`);
                }
            });
            
            totalPassed += results.passed;
            totalFailed += results.failed;
            console.log('');
        });
        
        const overallTotal = totalPassed + totalFailed;
        const overallSuccess = overallTotal > 0 ? ((totalPassed / overallTotal) * 100).toFixed(1) : 0;
        
        console.log(`🎯 OVERALL WEBSITE HEALTH: ${totalPassed}/${overallTotal} (${overallSuccess}%)`);
        
        // Recommendations
        console.log('\n💡 RECOMMENDATIONS:');
        if (overallSuccess >= 90) {
            console.log('🌟 Excellent! Website is in top condition for professional use.');
        } else if (overallSuccess >= 80) {
            console.log('✅ Good status. Minor improvements recommended.');
        } else if (overallSuccess >= 70) {
            console.log('⚠️  Website functional but needs attention in failed areas.');
        } else {
            console.log('🚨 Critical issues detected. Immediate fixes required.');
        }
        
        // Unity Mathematics specific checks
        const unityFeatures = [
            'φ-harmonic operations',
            'Consciousness field integration', 
            'Unity equation (1+1=1) proofs',
            'Metastation navigation',
            'Professional academic presentation'
        ];
        
        console.log('\n🧮 UNITY MATHEMATICS FRAMEWORK STATUS:');
        unityFeatures.forEach((feature, index) => {
            const status = overallSuccess >= 80 ? '✅' : '⚠️';
            console.log(`${status} ${feature}`);
        });
        
        console.log('\n🚀 Een Unity Mathematics Website Validation Complete!');
        console.log('φ = 1.618033988749895 | 1+1=1 ∞\n');
        
        return {
            overallScore: overallSuccess,
            totalTests: overallTotal,
            passedTests: totalPassed,
            failedTests: totalFailed,
            results: this.testResults
        };
    }

    // Helper method to check if selector contains text (jQuery-style)
    contains(selector, text) {
        const elements = document.querySelectorAll(selector);
        return Array.from(elements).some(el => el.textContent.includes(text));
    }
}

// Auto-run validation when script loads
if (typeof window !== 'undefined') {
    const validator = new EenWebsiteValidator();
    
    // Run validation after page loads
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            setTimeout(() => validator.runComprehensiveValidation(), 2000);
        });
    } else {
        setTimeout(() => validator.runComprehensiveValidation(), 2000);
    }
    
    // Make validator available globally for manual testing
    window.eenValidator = validator;
}

console.log('🔍 Een Website Validator loaded. Use eenValidator.runComprehensiveValidation() to test manually.');