/**
 * Universal AI System Applier
 * This script adds the universal navigation and AI chat to ALL pages
 */

// Function to add universal AI system to any page
function addUniversalAISystem() {
    // Check if already applied
    if (document.getElementById('universal-ai-nav-styles')) {
        return; // Already applied
    }

    // Add Font Awesome if not present
    if (!document.querySelector('link[href*="font-awesome"]')) {
        const fontAwesome = document.createElement('link');
        fontAwesome.rel = 'stylesheet';
        fontAwesome.href = 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css';
        document.head.appendChild(fontAwesome);
    }

    // Add universal AI navigation script if not present
    if (!document.querySelector('script[src*="universal-ai-navigation"]')) {
        const universalScript = document.createElement('script');
        universalScript.src = 'js/universal-ai-navigation.js';
        universalScript.defer = true;
        document.head.appendChild(universalScript);
    }
}

// Apply on DOM load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', addUniversalAISystem);
} else {
    addUniversalAISystem();
}

// Export for manual use
window.addUniversalAISystem = addUniversalAISystem;