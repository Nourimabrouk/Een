/**
 * Add Unified CSS to All HTML Pages
 * Ensures consistent styling across the entire website
 */

const fs = require('fs');
const path = require('path');

// List of HTML files to update
const htmlFiles = [
    'index.html',
    'about.html',
    'proofs.html',
    'research.html',
    'gallery.html',
    'consciousness_dashboard.html',
    'playground.html',
    'metagambit.html',
    'agents.html',
    'publications.html',
    'implementations.html',
    'learning.html',
    'further-reading.html',
    'philosophy.html',
    'unity_visualization.html',
    'dashboards.html',
    'metagamer_agent.html',
    'mobile-app.html',
    'zen-unity-meditation.html',
    'unity-mathematics-experience.html',
    'unity_consciousness_experience.html',
    'unity-advanced-features.html',
    'openai-integration.html',
    'mathematical_playground.html',
    'enhanced-unity-demo.html',
    'enhanced-ai-demo.html',
    'al_khwarizmi_phi_unity.html',
    '3000-elo-proof.html',
    'transcendental-unity-demo.html',
    'implementations-gallery.html',
    'live-code-showcase.html',
    'mathematical-framework.html',
    'metastation-hub.html',
    'anthill.html'
];

function addUnifiedCSS(htmlContent) {
    // Check if unified CSS is already included
    if (htmlContent.includes('unified-styles.css')) {
        return htmlContent;
    }

    // Add unified CSS after the navigation CSS
    const unifiedCSSLink = `
    <!-- Unified Styles -->
    <link rel="stylesheet" href="css/unified-styles.css">`;

    // Insert after meta-optimal-navigation.css if present
    if (htmlContent.includes('meta-optimal-navigation.css')) {
        return htmlContent.replace(
            /<link[^>]*meta-optimal-navigation\.css[^>]*>/i,
            `$&${unifiedCSSLink}`
        );
    }

    // Insert after font-awesome if present
    if (htmlContent.includes('font-awesome')) {
        return htmlContent.replace(
            /<link[^>]*font-awesome[^>]*>/i,
            `$&${unifiedCSSLink}`
        );
    }

    // Insert after the first link tag
    return htmlContent.replace(
        /<link[^>]*>/i,
        `$&${unifiedCSSLink}`
    );
}

function updateHTMLFile(filePath) {
    try {
        console.log(`Processing: ${filePath}`);
        
        let content = fs.readFileSync(filePath, 'utf8');
        
        // Add unified CSS
        content = addUnifiedCSS(content);
        
        // Write updated content back to file
        fs.writeFileSync(filePath, content, 'utf8');
        
        console.log(`✓ Updated: ${filePath}`);
        return true;
    } catch (error) {
        console.error(`✗ Error updating ${filePath}:`, error.message);
        return false;
    }
}

function addUnifiedCSSToAllPages() {
    console.log('🎨 Adding unified CSS to all HTML pages...\n');
    
    let successCount = 0;
    let totalCount = htmlFiles.length;
    
    htmlFiles.forEach(file => {
        const filePath = path.join(__dirname, file);
        if (fs.existsSync(filePath)) {
            if (updateHTMLFile(filePath)) {
                successCount++;
            }
        } else {
            console.log(`⚠️  File not found: ${file}`);
        }
    });
    
    console.log(`\n📊 Update Summary:`);
    console.log(`   Total files processed: ${totalCount}`);
    console.log(`   Successfully updated: ${successCount}`);
    console.log(`   Failed: ${totalCount - successCount}`);
    
    if (successCount === totalCount) {
        console.log('\n🎉 All pages now have unified CSS!');
        console.log('\n✨ Benefits applied:');
        console.log('   • Consistent color scheme');
        console.log('   • Unified typography');
        console.log('   • Professional component styling');
        console.log('   • Responsive design');
        console.log('   • Metagamer energy effects');
        console.log('   • Academic and professional aesthetic');
    } else {
        console.log('\n⚠️  Some files failed to update. Check the errors above.');
    }
}

// Run the update
if (require.main === module) {
    addUnifiedCSSToAllPages();
}

module.exports = {
    addUnifiedCSS,
    updateHTMLFile,
    addUnifiedCSSToAllPages
}; 