/**
 * Apply Unified Navigation to All Pages
 * This script automatically applies the unified navigation to any page
 * Usage: Include this script after the unified navigation CSS/JS
 */

(function() {
    'use strict';
    
    // Configuration
    const NAVIGATION_CONFIG = {
        removeOldNavigation: true,
        adjustBodyPadding: true,
        hideConflictingElements: true,
        preserveFooter: true
    };

    // Selectors for old navigation elements to remove
    const OLD_NAVIGATION_SELECTORS = [
        '.nav-bar',
        '.meta-optimal-header',
        '.meta-optimal-nav', 
        '.side-nav',
        '.side-nav-toggle',
        '.mobile-menu-toggle:not(.unified-header .mobile-menu-toggle)',
        // Add more selectors as needed based on old navigation patterns
    ];

    // Conflicting element selectors to hide
    const CONFLICTING_SELECTORS = [
        '.legacy-nav',
        '.old-header',
        '.deprecated-navigation'
    ];

    function removeOldNavigation() {
        OLD_NAVIGATION_SELECTORS.forEach(selector => {
            const elements = document.querySelectorAll(selector);
            elements.forEach(element => {
                console.log(`🗑️ Removing old navigation element: ${selector}`);
                element.remove();
            });
        });
    }

    function hideConflictingElements() {
        CONFLICTING_SELECTORS.forEach(selector => {
            const elements = document.querySelectorAll(selector);
            elements.forEach(element => {
                console.log(`👁️ Hiding conflicting element: ${selector}`);
                element.style.display = 'none';
            });
        });
    }

    function adjustBodyPadding() {
        // Ensure body has proper padding for fixed header
        const body = document.body;
        const currentPaddingTop = window.getComputedStyle(body).paddingTop;
        
        // Only adjust if padding is less than nav height
        if (parseInt(currentPaddingTop) < 75) {
            body.style.paddingTop = '75px';
            console.log('📏 Adjusted body padding for unified navigation');
        }
        
        // Remove any conflicting margin-left from old sidebar systems
        const currentMarginLeft = window.getComputedStyle(body).marginLeft;
        if (parseInt(currentMarginLeft) > 0) {
            body.style.marginLeft = '0';
            console.log('📏 Removed old sidebar margin from body');
        }
    }

    function injectNavigationIncludes() {
        // Check if unified navigation CSS is already loaded
        const cssLink = document.querySelector('link[href*="unified-navigation.css"]');
        if (!cssLink) {
            const link = document.createElement('link');
            link.rel = 'stylesheet';
            link.href = 'css/unified-navigation.css';
            document.head.appendChild(link);
            console.log('🎨 Injected unified navigation CSS');
        }

        // Check if unified navigation JS is already loaded
        const jsScript = document.querySelector('script[src*="unified-navigation.js"]');
        if (!jsScript) {
            const script = document.createElement('script');
            script.src = 'js/unified-navigation.js';
            script.defer = true;
            document.head.appendChild(script);
            console.log('⚡ Injected unified navigation JS');
        }
    }

    function fixChatButtonPositioning() {
        // Find any chat buttons and ensure they're positioned correctly
        const chatButtons = document.querySelectorAll([
            '.ai-chat-button',
            '#floating-ai-chat-button',
            '#persistent-chat-button',
            '.universal-ai-chat-button',
            '[id*="chat"]',
            '[class*="chat"]'
        ].join(','));

        chatButtons.forEach(button => {
            if (button && window.getComputedStyle(button).position === 'fixed') {
                button.style.right = '2rem';
                button.style.left = 'auto';
                button.style.zIndex = '999';
                console.log('🗨️ Fixed chat button positioning');
            }
        });
    }

    function preventNavigationConflicts() {
        // Prevent multiple navigation systems from initializing
        if (window.navigationSystemInitialized) {
            console.log('⚠️ Navigation system already initialized, skipping');
            return false;
        }
        
        window.navigationSystemInitialized = true;
        return true;
    }

    function addResponsiveMetaTag() {
        // Ensure viewport meta tag is present for responsive navigation
        if (!document.querySelector('meta[name="viewport"]')) {
            const meta = document.createElement('meta');
            meta.name = 'viewport';
            meta.content = 'width=device-width, initial-scale=1.0';
            document.head.appendChild(meta);
            console.log('📱 Added responsive viewport meta tag');
        }
    }

    function optimizeForPrint() {
        // Add print styles to hide navigation when printing
        const printStyle = document.createElement('style');
        printStyle.textContent = `
            @media print {
                .unified-header,
                .sidebar,
                .sidebar-toggle,
                .mobile-nav-overlay {
                    display: none !important;
                }
                
                body {
                    padding-top: 0 !important;
                    margin-left: 0 !important;
                }
            }
        `;
        document.head.appendChild(printStyle);
    }

    function initAccessibilityFeatures() {
        // Add skip link if not present
        if (!document.querySelector('.skip-link')) {
            const skipLink = document.createElement('a');
            skipLink.className = 'skip-link';
            skipLink.href = '#main';
            skipLink.textContent = 'Skip to main content';
            skipLink.style.cssText = `
                position: absolute;
                top: -40px;
                left: 6px;
                background: var(--unity-gold, #FFD700);
                color: var(--bg-primary, #000);
                padding: 8px;
                text-decoration: none;
                border-radius: 4px;
                z-index: 10000;
                transition: top 0.3s;
            `;
            
            // Show on focus
            skipLink.addEventListener('focus', () => {
                skipLink.style.top = '6px';
            });
            
            skipLink.addEventListener('blur', () => {
                skipLink.style.top = '-40px';
            });
            
            document.body.insertBefore(skipLink, document.body.firstChild);
            console.log('♿ Added accessibility skip link');
        }
    }

    function validatePageStructure() {
        // Ensure main content area is properly identified
        let mainElement = document.querySelector('main');
        if (!mainElement) {
            mainElement = document.querySelector('#main, .main-content, .content');
            if (mainElement && mainElement.tagName !== 'MAIN') {
                console.log('📝 Found content area but not using <main> tag');
            }
        }

        if (!mainElement) {
            console.warn('⚠️ No main content area found. Consider adding id="main" to your content wrapper.');
        }
    }

    function applyUnifiedNavigation() {
        console.log('🚀 Applying Unified Navigation System...');
        
        // Prevent conflicts
        if (!preventNavigationConflicts()) {
            return;
        }

        // Clean up old navigation
        if (NAVIGATION_CONFIG.removeOldNavigation) {
            removeOldNavigation();
        }

        if (NAVIGATION_CONFIG.hideConflictingElements) {
            hideConflictingElements();
        }

        // Inject navigation files
        injectNavigationIncludes();

        // Adjust layout
        if (NAVIGATION_CONFIG.adjustBodyPadding) {
            adjustBodyPadding();
        }

        // Fix positioning issues
        fixChatButtonPositioning();

        // Add enhancements
        addResponsiveMetaTag();
        optimizeForPrint();
        initAccessibilityFeatures();
        validatePageStructure();

        console.log('✅ Unified Navigation System applied successfully');
    }

    // Initialize based on document ready state
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', applyUnifiedNavigation);
    } else {
        applyUnifiedNavigation();
    }

    // Handle dynamic content changes
    const observer = new MutationObserver((mutations) => {
        let needsReapplication = false;
        
        mutations.forEach((mutation) => {
            if (mutation.type === 'childList') {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === 1) { // Element node
                        // Check if old navigation elements were re-added
                        OLD_NAVIGATION_SELECTORS.forEach(selector => {
                            if (node.matches && node.matches(selector)) {
                                needsReapplication = true;
                            }
                        });
                    }
                });
            }
        });
        
        if (needsReapplication) {
            console.log('🔄 Re-applying navigation due to DOM changes');
            setTimeout(() => {
                removeOldNavigation();
                fixChatButtonPositioning();
            }, 100);
        }
    });

    // Start observing
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });

    // Expose API for manual control
    window.UnifiedNavigationApplier = {
        reapply: applyUnifiedNavigation,
        removeOld: removeOldNavigation,
        fixPositioning: fixChatButtonPositioning
    };

    // Debug helper
    if (window.location.search.includes('debug-nav')) {
        console.log('🐛 Navigation Debug Mode Enabled');
        window.debugNavigation = {
            config: NAVIGATION_CONFIG,
            oldSelectors: OLD_NAVIGATION_SELECTORS,
            conflictingSelectors: CONFLICTING_SELECTORS
        };
    }

})();