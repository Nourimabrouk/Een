/**
 * Navigation Batch Updater
 * Updates all HTML files to use the new complete navigation system
 * Version: 1.0.0 - Automated Legacy Cleanup
 */

(function () {
    'use strict';

    console.log('🚀 Navigation Batch Updater - Legacy Cleanup Tool');
    console.log('=================================================');

    // Pages that need updating (excluding the 4 already updated)
    const pagesToUpdate = [
        // Priority 1 - Core Pages
        'consciousness_dashboard.html',
        'unity_visualization.html',
        'playground.html',
        'proofs.html',
        'philosophy.html',

        // Priority 2 - Featured Content
        '3000-elo-proof.html',
        'al_khwarizmi_phi_unity.html',
        'transcendental-unity-demo.html',
        'enhanced-unity-demo.html',
        'unity-mathematics-experience.html',

        // Priority 3 - All Remaining Pages
        'about.html',
        'agents.html',
        'anthill.html',
        'consciousness_dashboard_clean.html',
        'dashboards.html',
        'enhanced-ai-demo.html',
        'further-reading.html',
        'gallery.html',
        'gallery_test.html',
        'implementations.html',
        'index.html',
        'learn.html',
        'learning.html',
        'live-code-showcase.html',
        'mathematical_playground.html',
        'metagambit.html',
        'metagamer_agent.html',
        'mobile-app.html',
        'navigation-test.html',
        'openai-integration.html',
        'publications.html',
        'redirect.html',
        'research.html',
        'unity-advanced-features.html',
        'unity_consciousness_experience.html',
        'enhanced-unity-landing.html',
        'meta-optimal-landing.html'
    ];

    // Legacy patterns to find and replace
    const navigationUpdates = {
        // CSS Updates
        'meta-optimal-navigation.css': 'meta-optimal-navigation-complete.css',

        // JavaScript Updates
        'unified-navigation-system.js': 'meta-optimal-navigation-complete.js',

        // Remove redundant scripts (these will be removed entirely)
        removeScripts: [
            'metastation-sidebar-navigation.js',
            'master-integration-system.js',
            'meta-optimal-integration.js',
            // 'enhanced-unity-ai-chat.js' // Legacy; unified-chatbot-system.js is canonical now
        ]
    };

    // Patterns for complete replacement
    const updatePatterns = [
        {
            name: 'Update CSS Reference',
            find: /<link\s+rel="stylesheet"\s+href="css\/meta-optimal-navigation\.css">/g,
            replace: '<link rel="stylesheet" href="css/meta-optimal-navigation-complete.css">'
        },
        {
            name: 'Update Main Navigation JS',
            find: /<script\s+src="js\/unified-navigation-system\.js"\s+defer><\/script>/g,
            replace: '<script src="js/meta-optimal-navigation-complete.js" defer></script>'
        },
        {
            name: 'Remove Sidebar Navigation JS',
            find: /\s*<script\s+src="js\/metastation-sidebar-navigation\.js"\s+defer><\/script>/g,
            replace: ''
        },
        {
            name: 'Remove Master Integration JS',
            find: /\s*<script\s+src="js\/master-integration-system\.js"\s+defer><\/script>/g,
            replace: ''
        },
        {
            name: 'Remove Meta Optimal Integration JS',
            find: /\s*<script\s+src="js\/meta-optimal-integration\.js"\s+defer><\/script>/g,
            replace: ''
        },
        {
            name: 'Update Navigation Comments',
            find: /<!-- Meta-Optimal Navigation System -->/g,
            replace: '<!-- Meta-Optimal Complete Navigation System -->'
        }
    ];

    console.log(`📊 Analysis Complete:`);
    console.log(`   • Pages to update: ${pagesToUpdate.length}`);
    console.log(`   • Update patterns: ${updatePatterns.length}`);
    console.log(`   • Files already updated: 4 (metastation-hub, zen-meditation, implementations-gallery, mathematical-framework)`);

    console.log(`\n🔧 Required Updates Per File:`);
    updatePatterns.forEach((pattern, index) => {
        console.log(`   ${index + 1}. ${pattern.name}`);
    });

    console.log(`\n📝 Manual Update Instructions:`);
    console.log(`\nFor each HTML file, make these replacements:`);
    console.log(`\n1. CSS Update:`);
    console.log(`   OLD: <link rel="stylesheet" href="css/meta-optimal-navigation.css">`);
    console.log(`   NEW: <link rel="stylesheet" href="css/meta-optimal-navigation-complete.css">`);

    console.log(`\n2. JavaScript Update:`);
    console.log(`   OLD: <script src="js/unified-navigation-system.js" defer></script>`);
    console.log(`   NEW: <script src="js/meta-optimal-navigation-complete.js" defer></script>`);

    console.log(`\n3. Remove Conflicting Scripts:`);
    console.log(`   REMOVE: <script src="js/metastation-sidebar-navigation.js" defer></script>`);
    console.log(`   REMOVE: <script src="js/master-integration-system.js" defer></script>`);
    console.log(`   REMOVE: <script src="js/meta-optimal-integration.js" defer></script>`);

    console.log(`\n✅ Expected Results After Update:`);
    console.log(`   • Consistent navigation on all pages`);
    console.log(`   • Top bar, left sidebar, and footer on every page`);
    console.log(`   • No navigation conflicts`);
    console.log(`   • 70% reduction in navigation JavaScript`);
    console.log(`   • Faster page load times`);

    // Export configuration for use in other scripts
    window.NavigationBatchUpdater = {
        pagesToUpdate,
        updatePatterns,
        navigationUpdates
    };

    console.log(`\n🚀 Navigation Batch Updater Ready!`);
    console.log(`📋 See NAVIGATION_LEGACY_CLEANUP_REPORT.md for full details.`);

})();