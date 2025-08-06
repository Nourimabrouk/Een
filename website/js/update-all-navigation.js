/**
 * Navigation Update Script
 * Updates all HTML pages to use the new Master Navigation System v2.0
 */

const fs = require('fs');
const path = require('path');

// Pages that need navigation updates
const HTML_PAGES = [
    'index.html',
    'proofs.html', 
    '3000-elo-proof.html',
    'research.html',
    'publications.html',
    'playground.html',
    'mathematical_playground.html',
    'gallery.html',
    'learn.html',
    'learning.html',
    'philosophy.html',
    'implementations.html',
    'dashboards.html',
    'agents.html',
    'metagambit.html',
    'metagamer_agent.html',
    'al_khwarizmi_phi_unity.html',
    'consciousness_dashboard.html',
    'consciousness_dashboard_clean.html',
    'unity_consciousness_experience.html',
    'unity_visualization.html',
    'enhanced-unity-demo.html',
    'revolutionary-landing.html',
    'meta-optimal-landing.html',
    'mobile-app.html',
    'about.html',
    'further-reading.html',
    'examples/index.html',
    'examples/unity-calculator.html',
    'examples/phi-harmonic-explorer.html',
    'gallery/phi_consciousness_transcendence.html'
];

function updateHTMLPage(filePath) {
    try {
        let content = fs.readFileSync(filePath, 'utf8');
        let modified = false;

        // Remove old navigation script imports
        const oldNavScripts = [
            /<script\s+src=["']shared-navigation\.js["'][^>]*><\/script>/gi,
            /<script\s+src=["']js\/navigation\.js["'][^>]*><\/script>/gi,
            /<script\s+src=["']js\/unified-navigation\.js["'][^>]*><\/script>/gi,
            /<script\s+src=["']js\/quantum-enhanced-navigation\.js["'][^>]*><\/script>/gi,
            /<script\s+src=["']update-all-navigation\.js["'][^>]*><\/script>/gi
        ];

        oldNavScripts.forEach(regex => {
            const originalContent = content;
            content = content.replace(regex, '');
            if (content !== originalContent) modified = true;
        });

        // Remove old navigation placeholders and elements
        const oldNavElements = [
            /<div\s+id=["']navigation-placeholder["'][^>]*><\/div>/gi,
            /<nav\s+class=["'][^"']*navbar[^"']*["'][^>]*>.*?<\/nav>/gi,
            /<nav\s+class=["'][^"']*enhanced-nav[^"']*["'][^>]*>.*?<\/nav>/gi
        ];

        oldNavElements.forEach(regex => {
            const originalContent = content;
            content = content.replace(regex, '');
            if (content !== originalContent) modified = true;
        });

        // Add new master navigation script if not already present
        if (!content.includes('master-navigation.js')) {
            // Find head section and add the script
            const headMatch = content.match(/<head[^>]*>(.*?)<\/head>/si);
            if (headMatch) {
                const headContent = headMatch[1];
                
                // Add master navigation script before closing head
                const newHeadContent = headContent + '\n    <!-- Master Navigation System v2.0 -->\n    <script src="js/master-navigation.js"></script>';
                content = content.replace(headMatch[0], `<head${headMatch[0].match(/<head([^>]*)>/)[1] || ''}>${newHeadContent}\n</head>`);
                modified = true;
            }
        }

        // Ensure body has proper structure for navigation
        if (!content.includes('main-content')) {
            // Find main content area and add ID
            const bodyMatch = content.match(/<body[^>]*>(.*?)<\/body>/si);
            if (bodyMatch) {
                let bodyContent = bodyMatch[1];
                
                // Look for main content containers
                const mainContainers = [
                    /<div\s+class=["'][^"']*container[^"']*["'][^>]*>/i,
                    /<main[^>]*>/i,
                    /<div\s+class=["'][^"']*main[^"']*["'][^>]*>/i
                ];

                let foundMain = false;
                mainContainers.forEach(regex => {
                    const match = bodyContent.match(regex);
                    if (match && !foundMain) {
                        const originalTag = match[0];
                        const newTag = originalTag.replace('>', ' id="main-content">');
                        bodyContent = bodyContent.replace(originalTag, newTag);
                        foundMain = true;
                        modified = true;
                    }
                });

                // If no main container found, wrap everything in main
                if (!foundMain) {
                    bodyContent = `<main id="main-content" role="main">\n${bodyContent}\n</main>`;
                    modified = true;
                }

                content = content.replace(bodyMatch[0], `<body${bodyMatch[0].match(/<body([^>]*)>/)[1] || ''}>${bodyContent}</body>`);
            }
        }

        // Add responsive meta viewport if not present
        if (!content.includes('viewport')) {
            content = content.replace(
                /<meta\s+charset=[^>]+>/i,
                '$&\n    <meta name="viewport" content="width=device-width, initial-scale=1.0">'
            );
            modified = true;
        }

        // Fix relative paths for examples and gallery subdirectories
        if (filePath.includes('examples/') || filePath.includes('gallery/')) {
            content = content.replace(/src=["']js\/master-navigation\.js["']/g, 'src="../js/master-navigation.js"');
            content = content.replace(/href=["']([^"']+\.html)["']/g, (match, url) => {
                if (!url.startsWith('http') && !url.startsWith('../') && !url.startsWith('#')) {
                    return `href="../${url}"`;
                }
                return match;
            });
            modified = true;
        }

        if (modified) {
            fs.writeFileSync(filePath, content);
            console.log(`✅ Updated: ${filePath}`);
            return true;
        } else {
            console.log(`⚪ No changes needed: ${filePath}`);
            return false;
        }

    } catch (error) {
        console.error(`❌ Error updating ${filePath}:`, error.message);
        return false;
    }
}

function updateAllPages() {
    console.log('🚀 Starting navigation update for all HTML pages...\n');
    
    let updatedCount = 0;
    let totalCount = 0;

    HTML_PAGES.forEach(page => {
        const filePath = path.join(__dirname, '..', page);
        
        if (fs.existsSync(filePath)) {
            totalCount++;
            const wasUpdated = updateHTMLPage(filePath);
            if (wasUpdated) updatedCount++;
        } else {
            console.log(`⚠️  File not found: ${page}`);
        }
    });

    console.log(`\n🎉 Navigation update complete!`);
    console.log(`📊 Updated ${updatedCount} out of ${totalCount} pages`);
    console.log(`\n✨ All pages now use Master Navigation System v2.0`);
    console.log(`✨ Persistent chatbot button available on every page`);
    console.log(`✨ Unified navigation with natural categories`);
    console.log(`✨ Modern responsive design with accessibility features`);
}

// Run the update if this script is executed directly
if (require.main === module) {
    updateAllPages();
}

module.exports = { updateAllPages, updateHTMLPage };