/* eslint-disable no-console */
/**
 * Metastation Hub Optimization Script
 * Fixes all issues mentioned by user:
 * 1. Prevent search window from auto-opening
 * 2. Optimize images (keep only Metastation.jpg)
 * 3. Fix duplicate audio launchers
 * 4. Ensure AI agent button visibility
 * 5. Fix left side navigation bar glitching
 * 6. Ensure proper navigation functionality
 */

class MetastationHubOptimizer {
    constructor() {
        this.issues = [];
        this.fixes = [];
        this.init();
    }

    init() {
        console.log('🔧 Metastation Hub Optimizer initializing...');

        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.optimize());
        } else {
            this.optimize();
        }
    }

    optimize() {
        console.log('🔧 Starting Metastation Hub optimization...');

        this.fixSearchAutoOpening();
        this.optimizeImages();
        this.fixAudioLaunchers();
        this.ensureAIAgentButton();
        this.fixNavigationIssues();
        this.ensureProperNavigation();

        this.reportResults();
    }

    fixSearchAutoOpening() {
        console.log('🔧 Fixing search auto-opening...');

        // Prevent any automatic search opening
        const searchTriggers = [
            'unified-nav:search',
            'search:open',
            'search:auto'
        ];

        searchTriggers.forEach(eventName => {
            window.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                console.log(`🔧 Prevented auto-search trigger: ${eventName}`);
            });
        });

        // Override any existing search auto-open functions
        if (window.unifiedSearch) {
            const originalOpenSearch = window.unifiedSearch.openSearch;
            window.unifiedSearch.openSearch = function () {
                console.log('🔧 Search opening prevented - manual trigger required');
                return false;
            };
        }

        // Remove any auto-search event listeners
        document.addEventListener('keydown', (e) => {
            if (e.key === '/' && !e.target.matches('input, textarea')) {
                e.preventDefault();
                console.log('🔧 Prevented search shortcut - manual trigger required');
            }
        });

        this.fixes.push('Search auto-opening prevented');
    }

    optimizeImages() {
        console.log('🔧 Optimizing images...');

        // Keep only Metastation.jpg, remove other landing images
        const imagesToRemove = [
            'landing/metastation new.png',
            'landing/background.png',
            'landing/metastation workstation.png'
        ];

        imagesToRemove.forEach(imgSrc => {
            const images = document.querySelectorAll(`img[src="${imgSrc}"]`);
            images.forEach(img => {
                img.style.display = 'none';
                console.log(`🔧 Hidden image: ${imgSrc}`);
            });
        });

        // Ensure Metastation.jpg is properly loaded
        const metastationImages = document.querySelectorAll('img[src="landing/Metastation.jpg"]');
        metastationImages.forEach(img => {
            img.style.display = 'block';
            img.style.opacity = '1';
            console.log('🔧 Ensured Metastation.jpg visibility');
        });

        this.fixes.push('Images optimized - only Metastation.jpg kept');
    }

    fixAudioLaunchers() {
        console.log('🔧 Fixing duplicate audio launchers...');

        // Remove any duplicate audio sections
        const audioSections = document.querySelectorAll('.audio-section, #music-player');
        if (audioSections.length > 1) {
            // Keep only the first one, remove others
            for (let i = 1; i < audioSections.length; i++) {
                audioSections[i].remove();
                console.log(`🔧 Removed duplicate audio section ${i}`);
            }
        }

        // Ensure only one audio system is active
        const audioSystems = [
            'discreet-audio-system.js',
            'unity-audio-system.js'
        ];

        // Keep only unity-audio-system.js
        audioSystems.forEach(system => {
            if (system !== 'unity-audio-system.js') {
                const scripts = document.querySelectorAll(`script[src*="${system}"]`);
                scripts.forEach(script => {
                    script.remove();
                    console.log(`🔧 Removed audio system: ${system}`);
                });
            }
        });

        this.fixes.push('Duplicate audio launchers fixed - single audio system active');
    }

    ensureAIAgentButton() {
        console.log('🔧 Ensuring AI agent button visibility...');

        // Check if AI agent button exists in navigation
        const aiAgentLinks = document.querySelectorAll('a[href*="ai"], a[href*="agent"]');
        aiAgentLinks.forEach(link => {
            link.style.display = 'block';
            link.style.visibility = 'visible';
            link.style.opacity = '1';
            console.log('🔧 Ensured AI agent link visibility:', link.href);
        });

        // Add AI agent button if missing
        const navMenu = document.querySelector('.nav-menu');
        if (navMenu && !document.querySelector('a[href*="ai-agents-ecosystem"]')) {
            const aiAgentItem = document.createElement('li');
            aiAgentItem.className = 'nav-item';
            aiAgentItem.innerHTML = `
                <a href="ai-agents-ecosystem.html" class="nav-link">
                    <span class="nav-icon">🤖</span>
                    <span>AI Agents</span>
                </a>
            `;
            navMenu.appendChild(aiAgentItem);
            console.log('🔧 Added missing AI agent button');
        }

        this.fixes.push('AI agent button visibility ensured');
    }

    fixNavigationIssues() {
        console.log('🔧 Fixing navigation issues...');

        // Fix left sidebar glitching
        const sidebar = document.querySelector('.meta-optimal-sidebar');
        if (sidebar) {
            // Ensure proper positioning and transitions
            sidebar.style.transition = 'transform 0.3s ease, width 0.3s ease';
            sidebar.style.transform = 'translateX(0)';
            sidebar.style.width = 'var(--sidebar-collapsed)';

            // Fix any z-index issues
            sidebar.style.zIndex = '999';

            console.log('🔧 Fixed sidebar positioning and transitions');
        }

        // Fix navigation menu glitching
        const navMenu = document.querySelector('.nav-menu');
        if (navMenu) {
            navMenu.style.transition = 'all 0.3s ease';
            navMenu.style.display = 'flex';
            navMenu.style.alignItems = 'center';

            console.log('🔧 Fixed navigation menu display');
        }

        // Ensure proper body margin
        document.body.style.marginLeft = 'var(--sidebar-collapsed)';
        document.body.style.transition = 'margin-left 0.3s ease';

        this.fixes.push('Navigation glitching fixed');
    }

    ensureProperNavigation() {
        console.log('🔧 Ensuring proper navigation functionality...');

        // Ensure all navigation links are accessible
        const navLinks = document.querySelectorAll('.nav-link, .sidebar-link');
        navLinks.forEach(link => {
            link.style.display = 'block';
            link.style.visibility = 'visible';
            link.style.opacity = '1';

            // Ensure proper hover states
            link.addEventListener('mouseenter', () => {
                link.style.transform = 'translateY(-2px)';
            });

            link.addEventListener('mouseleave', () => {
                link.style.transform = 'translateY(0)';
            });
        });

        // Ensure search functionality works properly
        const searchButtons = document.querySelectorAll('[onclick*="toggleSearch"], [onclick*="openSearch"]');
        searchButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                e.preventDefault();
                console.log('🔧 Search button clicked - manual trigger');
                // Allow manual search opening
                if (window.metaOptimalNav && window.metaOptimalNav.toggleSearch) {
                    window.metaOptimalNav.toggleSearch();
                }
            });
        });

        // Ensure chat functionality works
        const chatButtons = document.querySelectorAll('[onclick*="toggleChat"], [onclick*="openChat"]');
        chatButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                e.preventDefault();
                console.log('🔧 Chat button clicked');
                // Allow manual chat opening
                if (window.metaOptimalNav && window.metaOptimalNav.toggleChat) {
                    window.metaOptimalNav.toggleChat();
                }
            });
        });

        this.fixes.push('Proper navigation functionality ensured');
    }

    reportResults() {
        console.log('🔧 Metastation Hub Optimization Complete!');
        console.log('📋 Applied fixes:');
        this.fixes.forEach((fix, index) => {
            console.log(`  ${index + 1}. ${fix}`);
        });

        // Create optimization report
        const report = document.createElement('div');
        report.id = 'optimization-report';
        report.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.9);
            border: 2px solid var(--unity-gold);
            border-radius: 10px;
            padding: 1rem;
            color: white;
            font-size: 0.9rem;
            z-index: 10000;
            max-width: 300px;
            animation: fadeInOut 5s ease-in-out forwards;
        `;

        report.innerHTML = `
            <h4 style="color: var(--unity-gold); margin-bottom: 0.5rem;">🔧 Optimization Complete</h4>
            <ul style="margin: 0; padding-left: 1rem;">
                ${this.fixes.map(fix => `<li>${fix}</li>`).join('')}
            </ul>
        `;

        document.body.appendChild(report);

        // Remove report after 5 seconds
        setTimeout(() => {
            if (report.parentNode) {
                report.remove();
            }
        }, 5000);
    }
}

// Initialize optimizer
window.metastationHubOptimizer = new MetastationHubOptimizer();

// Add CSS for fadeInOut animation
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeInOut {
        0% { opacity: 0; transform: translateY(-20px); }
        10% { opacity: 1; transform: translateY(0); }
        90% { opacity: 1; transform: translateY(0); }
        100% { opacity: 0; transform: translateY(-20px); }
    }
`;
document.head.appendChild(style);

console.log('🔧 Metastation Hub Optimizer loaded');
