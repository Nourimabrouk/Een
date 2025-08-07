/**
 * Final Launch Check for Een Unity Mathematics
 * Comprehensive verification of navigation, metastation-hub, and AI chatbot
 * Ensures natural flow and effective showcase of codebase and website
 */

class FinalLaunchCheck {
    constructor() {
        this.checks = [];
        this.issues = [];
        this.recommendations = [];
        this.init();
    }

    init() {
        console.log('🚀 Final Launch Check initializing...');

        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.runChecks());
        } else {
            this.runChecks();
        }
    }

    runChecks() {
        console.log('🚀 Running comprehensive final launch checks...');

        this.checkNavigationFlow();
        this.checkMetastationHubOptimization();
        this.checkAIChatbotIntegration();
        this.checkCodebaseShowcase();
        this.checkWebsiteFlow();
        this.checkPerformanceOptimization();

        this.generateReport();
    }

    checkNavigationFlow() {
        console.log('🔍 Checking navigation flow...');

        // Check top navigation bar
        const topNav = document.querySelector('.meta-optimal-nav');
        if (topNav) {
            this.checks.push('✅ Top navigation bar present');

            // Verify all critical links are accessible
            const criticalLinks = [
                'metastation-hub.html',
                'ai-unified-hub.html',
                'ai-agents-ecosystem.html',
                'mathematical-framework.html',
                'consciousness_dashboard.html'
            ];

            criticalLinks.forEach(link => {
                const navLink = document.querySelector(`a[href="${link}"]`);
                if (navLink) {
                    this.checks.push(`✅ Navigation link: ${link}`);
                } else {
                    this.issues.push(`❌ Missing navigation link: ${link}`);
                }
            });
        } else {
            this.issues.push('❌ Top navigation bar missing');
        }

        // Check sidebar navigation
        const sidebar = document.querySelector('.meta-optimal-sidebar');
        if (sidebar) {
            this.checks.push('✅ Sidebar navigation present');

            // Check sidebar toggle functionality
            const sidebarToggle = document.querySelector('.sidebar-toggle-btn');
            if (sidebarToggle) {
                this.checks.push('✅ Sidebar toggle button present');
            } else {
                this.issues.push('❌ Sidebar toggle button missing');
            }
        } else {
            this.issues.push('❌ Sidebar navigation missing');
        }

        // Check search functionality
        const searchButton = document.querySelector('[onclick*="toggleSearch"]');
        if (searchButton) {
            this.checks.push('✅ Search button present');
        } else {
            this.issues.push('❌ Search button missing');
        }

        // Check chat functionality
        const chatButton = document.querySelector('[onclick*="toggleChat"]');
        if (chatButton) {
            this.checks.push('✅ Chat button present');
        } else {
            this.issues.push('❌ Chat button missing');
        }
    }

    checkMetastationHubOptimization() {
        console.log('🔍 Checking metastation hub optimization...');

        // Check if optimization script is loaded
        const optimizationScript = document.querySelector('script[src*="metastation-hub-optimization.js"]');
        if (optimizationScript) {
            this.checks.push('✅ Metastation hub optimization script loaded');
        } else {
            this.issues.push('❌ Metastation hub optimization script missing');
        }

        // Check for duplicate audio systems
        const audioScripts = document.querySelectorAll('script[src*="audio"]');
        if (audioScripts.length <= 1) {
            this.checks.push('✅ Audio system optimized (single instance)');
        } else {
            this.issues.push(`❌ Multiple audio systems detected: ${audioScripts.length}`);
        }

        // Check image optimization
        const metastationImage = document.querySelector('img[src*="Metastation.jpg"]');
        if (metastationImage) {
            this.checks.push('✅ Metastation.jpg image present');
        } else {
            this.issues.push('❌ Metastation.jpg image missing');
        }

        // Check for unnecessary images
        const unnecessaryImages = [
            'metastation new.png',
            'background.png',
            'metastation workstation.png'
        ];

        unnecessaryImages.forEach(imgSrc => {
            const img = document.querySelector(`img[src*="${imgSrc}"]`);
            if (!img) {
                this.checks.push(`✅ Unnecessary image removed: ${imgSrc}`);
            } else {
                this.issues.push(`❌ Unnecessary image still present: ${imgSrc}`);
            }
        });
    }

    checkAIChatbotIntegration() {
        console.log('🔍 Checking AI chatbot integration...');

        // Check floating chat button
        const floatingChatButton = document.querySelector('#floating-ai-chat-button');
        if (floatingChatButton) {
            this.checks.push('✅ Floating AI chat button present');
        } else {
            this.issues.push('❌ Floating AI chat button missing');
        }

        // Check chat modal
        const chatModal = document.querySelector('#ai-chat-modal');
        if (chatModal) {
            this.checks.push('✅ AI chat modal present');
        } else {
            this.issues.push('❌ AI chat modal missing');
        }

        // Check AI chat scripts
        const aiChatScripts = [
            'enhanced-ai-chat-modal.js',
            'enhanced-consciousness-chat.js',
            'ai-unified-integration.js'
        ];

        aiChatScripts.forEach(script => {
            const scriptElement = document.querySelector(`script[src*="${script}"]`);
            if (scriptElement) {
                this.checks.push(`✅ AI chat script loaded: ${script}`);
            } else {
                this.issues.push(`❌ AI chat script missing: ${script}`);
            }
        });

        // Check consciousness integration
        const consciousnessScripts = [
            'consciousness-field-engine.js',
            'phi-harmonic-consciousness-engine.js'
        ];

        consciousnessScripts.forEach(script => {
            const scriptElement = document.querySelector(`script[src*="${script}"]`);
            if (scriptElement) {
                this.checks.push(`✅ Consciousness script loaded: ${script}`);
            } else {
                this.issues.push(`❌ Consciousness script missing: ${script}`);
            }
        });
    }

    checkCodebaseShowcase() {
        console.log('🔍 Checking codebase showcase...');

        // Check for code showcase elements
        const codeShowcaseScript = document.querySelector('script[src*="code-showcase.js"]');
        if (codeShowcaseScript) {
            this.checks.push('✅ Code showcase script loaded');
        } else {
            this.issues.push('❌ Code showcase script missing');
        }

        // Check for mathematical framework showcase
        const mathFrameworkLink = document.querySelector('a[href*="mathematical-framework.html"]');
        if (mathFrameworkLink) {
            this.checks.push('✅ Mathematical framework showcase accessible');
        } else {
            this.issues.push('❌ Mathematical framework showcase missing');
        }

        // Check for implementations gallery
        const implementationsGallery = document.querySelector('a[href*="implementations-gallery.html"]');
        if (implementationsGallery) {
            this.checks.push('✅ Implementations gallery accessible');
        } else {
            this.issues.push('❌ Implementations gallery missing');
        }

        // Check for live code showcase
        const liveCodeShowcase = document.querySelector('a[href*="live-code-showcase.html"]');
        if (liveCodeShowcase) {
            this.checks.push('✅ Live code showcase accessible');
        } else {
            this.issues.push('❌ Live code showcase missing');
        }

        // Check for research and publications
        const researchLink = document.querySelector('a[href*="research.html"]');
        const publicationsLink = document.querySelector('a[href*="publications.html"]');

        if (researchLink) {
            this.checks.push('✅ Research showcase accessible');
        } else {
            this.issues.push('❌ Research showcase missing');
        }

        if (publicationsLink) {
            this.checks.push('✅ Publications showcase accessible');
        } else {
            this.issues.push('❌ Publications showcase missing');
        }
    }

    checkWebsiteFlow() {
        console.log('🔍 Checking website flow...');

        // Check for natural progression in navigation
        const navigationFlow = [
            'metastation-hub.html', // Main hub
            'ai-unified-hub.html',  // AI integration
            'mathematical-framework.html', // Core mathematics
            'consciousness_dashboard.html', // Consciousness
            'implementations-gallery.html', // Showcase
            'philosophy.html' // Philosophy
        ];

        let flowScore = 0;
        navigationFlow.forEach(link => {
            const navLink = document.querySelector(`a[href="${link}"]`);
            if (navLink) {
                flowScore++;
            }
        });

        if (flowScore === navigationFlow.length) {
            this.checks.push('✅ Complete navigation flow present');
        } else {
            this.issues.push(`❌ Incomplete navigation flow: ${flowScore}/${navigationFlow.length}`);
        }

        // Check for interactive elements
        const interactiveElements = [
            'consciousness-field-canvas',
            'unity-equation',
            'performance-metrics-grid'
        ];

        interactiveElements.forEach(elementId => {
            const element = document.getElementById(elementId);
            if (element) {
                this.checks.push(`✅ Interactive element present: ${elementId}`);
            } else {
                this.issues.push(`❌ Interactive element missing: ${elementId}`);
            }
        });

        // Check for visual appeal
        const visualElements = [
            '.unity-gold',
            '.consciousness-purple',
            '.quantum-blue'
        ];

        visualElements.forEach(selector => {
            const elements = document.querySelectorAll(selector);
            if (elements.length > 0) {
                this.checks.push(`✅ Visual styling present: ${selector}`);
            } else {
                this.issues.push(`❌ Visual styling missing: ${selector}`);
            }
        });
    }

    checkPerformanceOptimization() {
        console.log('🔍 Checking performance optimization...');

        // Check for deferred script loading
        const deferredScripts = document.querySelectorAll('script[defer]');
        if (deferredScripts.length > 0) {
            this.checks.push(`✅ Deferred script loading: ${deferredScripts.length} scripts`);
        } else {
            this.issues.push('❌ No deferred script loading');
        }

        // Check for preconnect links
        const preconnectLinks = document.querySelectorAll('link[rel="preconnect"]');
        if (preconnectLinks.length > 0) {
            this.checks.push(`✅ Preconnect optimization: ${preconnectLinks.length} links`);
        } else {
            this.issues.push('❌ No preconnect optimization');
        }

        // Check for image optimization
        const images = document.querySelectorAll('img[loading="lazy"]');
        if (images.length > 0) {
            this.checks.push(`✅ Lazy loading images: ${images.length} images`);
        } else {
            this.recommendations.push('💡 Consider adding lazy loading to images');
        }

        // Check for CSS optimization
        const externalCSS = document.querySelectorAll('link[rel="stylesheet"]');
        if (externalCSS.length > 0) {
            this.checks.push(`✅ External CSS loading: ${externalCSS.length} stylesheets`);
        } else {
            this.issues.push('❌ No external CSS loading');
        }
    }

    generateReport() {
        console.log('🚀 Final Launch Check Complete!');
        console.log('📋 Check Results:');

        this.checks.forEach(check => {
            console.log(`  ${check}`);
        });

        if (this.issues.length > 0) {
            console.log('\n❌ Issues Found:');
            this.issues.forEach(issue => {
                console.log(`  ${issue}`);
            });
        }

        if (this.recommendations.length > 0) {
            console.log('\n💡 Recommendations:');
            this.recommendations.forEach(rec => {
                console.log(`  ${rec}`);
            });
        }

        // Create visual report
        this.createVisualReport();

        // Calculate launch readiness score
        const totalChecks = this.checks.length + this.issues.length;
        const successRate = (this.checks.length / totalChecks) * 100;

        console.log(`\n🚀 Launch Readiness Score: ${successRate.toFixed(1)}%`);

        if (successRate >= 90) {
            console.log('🎉 EXCELLENT! Ready for launch!');
        } else if (successRate >= 80) {
            console.log('✅ GOOD! Minor issues to address before launch.');
        } else {
            console.log('⚠️ NEEDS WORK! Significant issues to resolve before launch.');
        }
    }

    createVisualReport() {
        const report = document.createElement('div');
        report.id = 'final-launch-report';
        report.style.cssText = `
            position: fixed;
            top: 20px;
            left: 20px;
            background: rgba(0, 0, 0, 0.95);
            border: 2px solid var(--unity-gold);
            border-radius: 15px;
            padding: 1.5rem;
            color: white;
            font-size: 0.9rem;
            z-index: 10000;
            max-width: 400px;
            max-height: 80vh;
            overflow-y: auto;
            animation: slideInLeft 0.5s ease-out;
        `;

        const totalChecks = this.checks.length + this.issues.length;
        const successRate = (this.checks.length / totalChecks) * 100;

        report.innerHTML = `
            <h3 style="color: var(--unity-gold); margin-bottom: 1rem; text-align: center;">
                🚀 Final Launch Check Report
            </h3>
            
            <div style="margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span>Launch Readiness:</span>
                    <span style="color: ${successRate >= 90 ? '#4CAF50' : successRate >= 80 ? '#FF9800' : '#F44336'}; font-weight: bold;">
                        ${successRate.toFixed(1)}%
                    </span>
                </div>
                <div style="background: rgba(255,255,255,0.1); height: 8px; border-radius: 4px; overflow: hidden;">
                    <div style="background: ${successRate >= 90 ? '#4CAF50' : successRate >= 80 ? '#FF9800' : '#F44336'}; height: 100%; width: ${successRate}%; transition: width 0.5s ease;"></div>
                </div>
            </div>
            
            <div style="margin-bottom: 1rem;">
                <h4 style="color: var(--unity-gold); margin-bottom: 0.5rem;">✅ Checks Passed (${this.checks.length})</h4>
                <ul style="margin: 0; padding-left: 1rem; font-size: 0.8rem;">
                    ${this.checks.slice(0, 10).map(check => `<li>${check.replace('✅ ', '')}</li>`).join('')}
                    ${this.checks.length > 10 ? `<li>... and ${this.checks.length - 10} more</li>` : ''}
                </ul>
            </div>
            
            ${this.issues.length > 0 ? `
                <div style="margin-bottom: 1rem;">
                    <h4 style="color: #F44336; margin-bottom: 0.5rem;">❌ Issues Found (${this.issues.length})</h4>
                    <ul style="margin: 0; padding-left: 1rem; font-size: 0.8rem;">
                        ${this.issues.slice(0, 5).map(issue => `<li>${issue.replace('❌ ', '')}</li>`).join('')}
                        ${this.issues.length > 5 ? `<li>... and ${this.issues.length - 5} more</li>` : ''}
                    </ul>
                </div>
            ` : ''}
            
            ${this.recommendations.length > 0 ? `
                <div style="margin-bottom: 1rem;">
                    <h4 style="color: #FF9800; margin-bottom: 0.5rem;">💡 Recommendations (${this.recommendations.length})</h4>
                    <ul style="margin: 0; padding-left: 1rem; font-size: 0.8rem;">
                        ${this.recommendations.slice(0, 3).map(rec => `<li>${rec.replace('💡 ', '')}</li>`).join('')}
                        ${this.recommendations.length > 3 ? `<li>... and ${this.recommendations.length - 3} more</li>` : ''}
                    </ul>
                </div>
            ` : ''}
            
            <div style="text-align: center; margin-top: 1rem;">
                <button onclick="this.parentElement.parentElement.remove()" style="
                    background: var(--unity-gold);
                    color: black;
                    border: none;
                    padding: 0.5rem 1rem;
                    border-radius: 5px;
                    cursor: pointer;
                    font-weight: bold;
                ">Close Report</button>
            </div>
        `;

        document.body.appendChild(report);

        // Auto-remove after 10 seconds
        setTimeout(() => {
            if (report.parentNode) {
                report.remove();
            }
        }, 10000);
    }
}

// Initialize final launch check
window.finalLaunchCheck = new FinalLaunchCheck();

// Add CSS for animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInLeft {
        from { transform: translateX(-100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
`;
document.head.appendChild(style);

console.log('🚀 Final Launch Check loaded - comprehensive verification active');
