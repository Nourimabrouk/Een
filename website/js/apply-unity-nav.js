/**
 * Universal Navigation Applier
 * Single script inclusion for complete Unity Mathematics navigation
 */

(function() {
    'use strict';
    
    // Load configuration first
    const configScript = document.createElement('script');
    configScript.src = 'js/unified-navigation-config.js';
    configScript.onload = function() {
        // Then load the navigation system
        const navScript = document.createElement('script');
        navScript.src = 'js/unity-nav-system.js';
        document.head.appendChild(navScript);
    };
    document.head.appendChild(configScript);
    
})();