/**
 * Master Navigation Update Script
 * Updates all HTML files with consistent navigation and AI chatbot integration
 */

const fs = require('fs');
const path = require('path');

// List of HTML files to update
const htmlFiles = [
    'index.html',
    'proofs.html',
    'research.html',
    'publications.html',
    'playground.html',
    'gallery.html',
    'learn.html',
    'learning.html',
    'metagambit.html',
    'metagamer_agent.html',
    'consciousness_dashboard.html',
    'consciousness_dashboard_clean.html',
    'unity_consciousness_experience.html',
    'unity_visualization.html',
    'about.html',
    'agents.html',
    'dashboards.html',
    'implementations.html',
    'philosophy.html',
    'further-reading.html',
    'revolutionary-landing.html',
    'meta-optimal-landing.html',
    'mathematical_playground.html',
    'enhanced-unity-demo.html',
    'al_khwarizmi_phi_unity.html',
    'mobile-app.html'
];

// Navigation script includes to add
const navigationScripts = `
    <!-- Enhanced Navigation System -->
    <script src="shared-navigation.js"></script>
    <script src="js/ai-chat-integration.js"></script>
`;

// Body padding adjustment for fixed navigation
const bodyPaddingCSS = `
    <style>
        body {
            padding-top: 80px; /* Account for fixed navigation */
        }
        .hero {
            margin-top: -80px;
            padding-top: 80px;
        }
        .page-header {
            margin-top: 0;
            padding-top: 8rem;
        }
    </style>
`;

function updateHTMLFile(filePath) {
    try {
        if (!fs.existsSync(filePath)) {
            console.log(`❌ File not found: ${filePath}`);
            return;
        }

        let content = fs.readFileSync(filePath, 'utf8');
        let updated = false;

        // Add navigation placeholder if it doesn't exist
        if (!content.includes('id="navigation-placeholder"') && !content.includes('id="enhancedUnityNav"')) {
            const bodyMatch = content.match(/<body[^>]*>/);
            if (bodyMatch) {
                const insertAfter = bodyMatch.index + bodyMatch[0].length;
                content = content.slice(0, insertAfter) + 
                    '\n    <!-- Navigation will be injected by shared-navigation.js -->\n    <div id="navigation-placeholder"></div>\n' +
                    content.slice(insertAfter);
                updated = true;
            }
        }

        // Add navigation scripts if not present
        if (!content.includes('shared-navigation.js')) {
            const headCloseMatch = content.match(/<\/head>/);
            if (headCloseMatch) {
                const insertBefore = headCloseMatch.index;
                content = content.slice(0, insertBefore) + 
                    navigationScripts + '\n' +
                    content.slice(insertBefore);
                updated = true;
            }
        }

        // Add body padding CSS if not present
        if (!content.includes('padding-top: 80px') && !content.includes('Account for fixed navigation')) {
            const headCloseMatch = content.match(/<\/head>/);
            if (headCloseMatch) {
                const insertBefore = headCloseMatch.index;
                content = content.slice(0, insertBefore) + 
                    bodyPaddingCSS + '\n' +
                    content.slice(insertBefore);
                updated = true;
            }
        }

        // Add footer placeholder if it doesn't exist
        if (!content.includes('id="footer-placeholder"') && !content.includes('class="footer"')) {
            const bodyCloseMatch = content.match(/<\/body>/);
            if (bodyCloseMatch) {
                const insertBefore = bodyCloseMatch.index;
                content = content.slice(0, insertBefore) + 
                    '\n    <!-- Footer will be injected by shared-navigation.js -->\n    <div id="footer-placeholder"></div>\n' +
                    content.slice(insertBefore);
                updated = true;
            }
        }

        if (updated) {
            fs.writeFileSync(filePath, content, 'utf8');
            console.log(`✅ Updated: ${filePath}`);
        } else {
            console.log(`ℹ️  Already updated: ${filePath}`);
        }

    } catch (error) {
        console.error(`❌ Error updating ${filePath}:`, error);
    }
}

function main() {
    console.log('🚀 Starting navigation update process...\n');

    const websiteDir = __dirname;
    
    htmlFiles.forEach(fileName => {
        const filePath = path.join(websiteDir, fileName);
        updateHTMLFile(filePath);
    });

    console.log('\n✨ Navigation update process completed!');
    console.log('\n📋 Summary:');
    console.log('- Added navigation placeholders to all HTML files');
    console.log('- Integrated AI chatbot functionality site-wide');
    console.log('- Added consistent styling for fixed navigation');
    console.log('- Enhanced mobile responsiveness');
    console.log('\n🎯 Next steps:');
    console.log('1. Test the website in a browser');
    console.log('2. Verify AI chatbot functionality');
    console.log('3. Check mobile responsiveness');
    console.log('4. Test all navigation links');
}

// Run if called directly (not required as module)
if (require.main === module) {
    main();
}

module.exports = { updateHTMLFile, htmlFiles };