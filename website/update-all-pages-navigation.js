/**
 * Update All Pages Navigation Script
 * 
 * This script updates all HTML pages in the website to use the new optimized navigation system.
 * Run this script to ensure consistency across all pages.
 */

// List of all HTML pages to update
const pagesToUpdate = [
    'gallery.html',
    'proofs.html',
    'philosophy.html',
    'live-code-showcase.html',
    'openai-integration.html',
    'enhanced-ai-demo.html',
    'transcendental-unity-demo.html',
    'mobile-app.html',
    'metagamer_agent.html',
    'mathematical_playground.html',
    'unity-mathematics-experience.html',
    'further-reading.html',
    'meta-optimal-landing.html',
    'revolutionary-landing.html',
    'enhanced-unity-demo.html',
    'unity_visualization.html',
    'unity_consciousness_experience.html',
    'consciousness_dashboard_clean.html',
    'consciousness_dashboard.html',
    'al_khwarizmi_phi_unity.html',
    'agents.html',
    'dashboards.html',
    'implementations.html',
    'learning.html',
    'learn.html',
    'playground.html',
    'publications.html',
    'research.html',
    '3000-elo-proof.html',
    'test-chat.html',
    'test-navigation.html',
    'test-website.html',
    'unified-nav.html',
    'enhanced-unified-nav.html',
    'gallery_test.html'
];

// Function to update a single page
function updatePageNavigation(pagePath) {
    try {
        // Read the file content
        const fs = require('fs');
        let content = fs.readFileSync(pagePath, 'utf8');

        // Remove old navigation scripts
        content = content.replace(/<!-- Master Navigation System v2\.0 -->\s*<script src="js\/master-navigation\.js"><\/script>/g, '');
        content = content.replace(/<!-- Enhanced Navigation and AI Chat -->\s*<script src="js\/shared-navigation\.js"><\/script>/g, '');
        content = content.replace(/<!-- Unified Navigation and AI Chat -->\s*<script src="js\/unified-navigation\.js"><\/script>/g, '');

        // Add new navigation script
        if (!content.includes('optimized-unified-navigation.js')) {
            content = content.replace(/<\/head>/, '    <!-- Optimized Unified Navigation System v3.0 -->\n    <script src="js/optimized-unified-navigation.js"></script>\n</head>');
        }

        // Remove old navigation placeholders
        content = content.replace(/<!-- Navigation Placeholder -->/g, '<!-- Optimized Navigation will be injected automatically -->');

        // Remove old navigation HTML if present
        content = content.replace(/<nav class="enhanced-nav"[^>]*>[\s\S]*?<\/nav>/g, '');
        content = content.replace(/<div class="nav-container"[^>]*>[\s\S]*?<\/div>/g, '');

        // Write the updated content back
        fs.writeFileSync(pagePath, content, 'utf8');

        console.log(`✅ Updated: ${pagePath}`);
        return true;
    } catch (error) {
        console.error(`❌ Error updating ${pagePath}:`, error.message);
        return false;
    }
}

// Function to update all pages
function updateAllPages() {
    console.log('🚀 Starting navigation update for all pages...\n');

    let successCount = 0;
    let errorCount = 0;

    pagesToUpdate.forEach(page => {
        const result = updatePageNavigation(page);
        if (result) {
            successCount++;
        } else {
            errorCount++;
        }
    });

    console.log(`\n📊 Update Summary:`);
    console.log(`✅ Successfully updated: ${successCount} pages`);
    console.log(`❌ Failed to update: ${errorCount} pages`);
    console.log(`📄 Total pages processed: ${pagesToUpdate.length}`);

    if (errorCount === 0) {
        console.log('\n🎉 All pages updated successfully!');
    } else {
        console.log('\n⚠️  Some pages failed to update. Check the errors above.');
    }
}

// Run the update if this script is executed directly
if (require.main === module) {
    updateAllPages();
}

module.exports = {
    updatePageNavigation,
    updateAllPages,
    pagesToUpdate
}; 