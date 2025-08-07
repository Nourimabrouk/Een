/**
 * Consolidate to Optimal Navigation System
 * Updates all HTML pages to use the most comprehensive navigation system
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

// Function to update navigation system
function updateToOptimalNavigation(htmlContent) {
    let updatedContent = htmlContent;

    // Replace shared-navigation.js with unified-navigation-system.js
    updatedContent = updatedContent.replace(
        /<script src="shared-navigation\.js"[^>]*><\/script>/gi,
        '<script src="js/unified-navigation-system.js" defer></script>'
    );

    // Ensure meta-optimal-navigation.css is included
    if (!updatedContent.includes('meta-optimal-navigation.css')) {
        updatedContent = updatedContent.replace(
            /<link[^>]*font-awesome[^>]*>/i,
            `$&\n    <!-- Meta-Optimal Navigation System -->
    <link rel="stylesheet" href="css/meta-optimal-navigation.css">`
        );
    }

    // Ensure unified-styles.css is included
    if (!updatedContent.includes('unified-styles.css')) {
        updatedContent = updatedContent.replace(
            /<link[^>]*meta-optimal-navigation\.css[^>]*>/i,
            `$&\n    <!-- Unified Styles -->
    <link rel="stylesheet" href="css/unified-styles.css">`
        );
    }

    // Remove any old navigation HTML if present
    updatedContent = updatedContent.replace(
        /<nav[^>]*>[\s\S]*?<\/nav>/gi,
        '<!-- Navigation will be injected by unified-navigation-system.js -->'
    );

    // Add body padding for fixed navigation if not present
    if (!updatedContent.includes('padding-top: 80px')) {
        updatedContent = updatedContent.replace(
            /body\s*\{[^}]*\}/i,
            (match) => {
                if (!match.includes('padding-top')) {
                    return match.replace('{', '{\n            padding-top: 80px; /* Account for fixed navigation */');
                }
                return match;
            }
        );
    }

    return updatedContent;
}

// Function to update a single HTML file
function updateHTMLFile(filePath) {
    try {
        console.log(`Processing: ${filePath}`);
        
        let content = fs.readFileSync(filePath, 'utf8');
        
        // Update to optimal navigation system
        content = updateToOptimalNavigation(content);
        
        // Write updated content back to file
        fs.writeFileSync(filePath, content, 'utf8');
        
        console.log(`✓ Updated: ${filePath}`);
        return true;
    } catch (error) {
        console.error(`✗ Error updating ${filePath}:`, error.message);
        return false;
    }
}

// Main consolidation function
function consolidateToOptimalNavigation() {
    console.log('🚀 Consolidating to Optimal Navigation System...\n');
    
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
    
    console.log(`\n📊 Consolidation Summary:`);
    console.log(`   Total files processed: ${totalCount}`);
    console.log(`   Successfully updated: ${successCount}`);
    console.log(`   Failed: ${totalCount - successCount}`);
    
    if (successCount === totalCount) {
        console.log('\n🎉 All pages now use the optimal navigation system!');
        console.log('\n✨ Benefits of consolidation:');
        console.log('   • Comprehensive navigation covering 42+ pages');
        console.log('   • Advanced dropdown system with icons');
        console.log('   • Perfect desktop/mobile experience');
        console.log('   • Meta-optimal styling and animations');
        console.log('   • Enhanced accessibility features');
        console.log('   • Professional academic categorization');
        console.log('   • Anthill page properly integrated');
        console.log('   • Consistent user experience across all pages');
    } else {
        console.log('\n⚠️  Some files failed to update. Check the errors above.');
    }
}

// Run the consolidation
if (require.main === module) {
    consolidateToOptimalNavigation();
}

module.exports = {
    updateToOptimalNavigation,
    updateHTMLFile,
    consolidateToOptimalNavigation
}; 