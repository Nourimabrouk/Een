/**
 * Universal Navigation Applicator
 * Applies the meta-optimal navigation to all pages
 */

(function() {
    'use strict';
    
    // Function to update a single HTML file
    function getNavigationHTML() {
        return {
            css: [
                '<link rel="stylesheet" href="css/meta-optimal-navigation-complete.css">'
            ],
            js: [
                '<script src="js/meta-optimal-navigation-complete.js" defer></script>'
            ]
        };
    }
    
    // Pages to update
    const pagesToUpdate = [
        'zen-unity-meditation.html',
        'implementations-gallery.html',
        'mathematical-framework.html',
        'consciousness_dashboard.html',
        'unity_visualization.html',
        'playground.html',
        '3000-elo-proof.html',
        'proofs.html',
        'al_khwarizmi_phi_unity.html',
        'philosophy.html',
        'metagambit.html',
        'research.html',
        'publications.html',
        'gallery.html',
        'sitemap.html',
        'about.html',
        'learn.html',
        'unity-mathematics-experience.html',
        'unity_consciousness_experience.html',
        'transcendental-unity-demo.html',
        'enhanced-unity-demo.html',
        'mathematical_playground.html',
        'enhanced-unity-visualization-system.html',
        'enhanced-3d-consciousness-field.html',
        'consciousness_dashboard_clean.html',
        'metagamer_agent.html',
        'agents.html',
        'ai-agents-ecosystem.html',
        'openai-integration.html',
        'further-reading.html',
        'implementations.html',
        'gallery_test.html',
        'live-code-showcase.html',
        'dashboards.html',
        'learning.html',
        'mobile-app.html',
        'navigation-test.html',
        'test-search.html',
        'test-fixes.html',
        'index.html',
        'meta-optimal-landing.html',
        'enhanced-unity-landing.html',
        'enhanced-ai-demo.html',
        'unity-advanced-features.html',
        'anthill.html',
        'enhanced-mathematical-proofs.html',
        'redirect.html'
    ];
    
    // Log the pages that need updating
    console.log('Pages to update with universal navigation:');
    console.log(pagesToUpdate);
    
    // Instructions for manual update
    console.log('\n=== INSTRUCTIONS FOR UPDATING PAGES ===\n');
    console.log('1. Replace old navigation includes:');
    console.log('   OLD: <link rel="stylesheet" href="css/meta-optimal-navigation.css">');
    console.log('   NEW: <link rel="stylesheet" href="css/meta-optimal-navigation-complete.css">');
    console.log('\n');
    console.log('   OLD: <script src="js/unified-navigation-system.js" defer></script>');
    console.log('   NEW: <script src="js/meta-optimal-navigation-complete.js" defer></script>');
    console.log('\n');
    console.log('2. Ensure each page has:');
    console.log('   - A <header id="main-header"></header> element (or let the script create it)');
    console.log('   - The navigation scripts will handle the rest automatically');
    console.log('\n');
    console.log('3. The new navigation includes:');
    console.log('   - Top navigation bar');
    console.log('   - Left sidebar (collapsible)');
    console.log('   - Comprehensive footer');
    console.log('\n');
    
    // Export for use in other scripts
    window.UniversalNavigationConfig = {
        pages: pagesToUpdate,
        navigation: getNavigationHTML()
    };
})();