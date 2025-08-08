/**
 * Navigation Template Applier
 * Automatically applies unified navigation to any page by simply including this script
 * This is the easiest way to upgrade pages to the new navigation system
 */

(function () {
    'use strict';

    const TEMPLATE_CONFIG = {
        autoRemoveOld: true,
        injectCSS: true,
        injectJS: true,
        preserveCustomStyling: true
    };

    function applySingleIncludeNavigation() {
        console.log('🚀 Applying Single-Include Unified Navigation...');

        // Inject CSS if not present
        if (TEMPLATE_CONFIG.injectCSS && !document.querySelector('link[href*="unified-navigation.css"]')) {
            const link = document.createElement('link');
            link.rel = 'stylesheet';
            link.href = 'css/unified-navigation.css';
            document.head.appendChild(link);
            console.log('💄 Injected unified navigation CSS');
        }

        // Inject main JS if not present
        if (TEMPLATE_CONFIG.injectJS && !document.querySelector('script[src*="unified-navigation.js"]')) {
            const script = document.createElement('script');
            script.src = 'js/unified-navigation.js';
            script.defer = true;
            document.head.appendChild(script);
            console.log('⚡ Injected unified navigation JS');
        }

        // Inject applier script if not present
        if (!document.querySelector('script[src*="apply-unified-navigation.js"]')) {
            const applierScript = document.createElement('script');
            applierScript.src = 'js/apply-unified-navigation.js';
            applierScript.defer = true;
            document.head.appendChild(applierScript);
            console.log('🔧 Injected navigation applier script');
        }

        // Inject unified chatbot (floating button) if not present
        if (!document.querySelector('script[src*="unified-chatbot-system.js"]')) {
            const chatScript = document.createElement('script');
            chatScript.src = 'js/unified-chatbot-system.js';
            chatScript.defer = true;
            document.head.appendChild(chatScript);
            console.log('💬 Injected unified chatbot system');
        }

        // Remove common old navigation patterns immediately
        if (TEMPLATE_CONFIG.autoRemoveOld) {
            const oldNavSelectors = [
                '.nav-bar:not(.unified-header)',
                '.meta-optimal-header:not(.unified-header)',
                '.meta-optimal-nav',
                '.side-nav:not(.sidebar)',
                '.side-nav-toggle:not(.sidebar-toggle)',
                // Add more patterns as found
            ];

            oldNavSelectors.forEach(selector => {
                document.querySelectorAll(selector).forEach(el => {
                    console.log(`🗑️ Removing old nav element: ${selector}`);
                    el.style.display = 'none'; // Hide first, remove after unified loads
                    setTimeout(() => el.remove(), 1000);
                });
            });
        }

        console.log('✅ Single-Include Navigation Applied');
    }

    // Apply immediately if DOM is ready, otherwise wait
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', applySingleIncludeNavigation);
    } else {
        applySingleIncludeNavigation();
    }

})();

/**
 * USAGE INSTRUCTIONS:
 * 
 * To upgrade ANY page to use the unified navigation system, simply add this line to the <head>:
 * 
 * <script src="js/nav-template-applier.js" defer></script>
 * 
 * That's it! This script will:
 * 1. Automatically inject the unified navigation CSS
 * 2. Automatically inject the unified navigation JavaScript
 * 3. Remove/hide old navigation elements
 * 4. Apply the complete unified navigation system
 * 
 * No other changes needed to existing pages!
 */