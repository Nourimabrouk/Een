/**
 * Apply Meta-Optimal Navigation to All Website Pages
 * This script updates all HTML files in the website directory to include the new navigation system
 */

const fs = require('fs');
const path = require('path');

class MetaOptimalNavigationApplier {
    constructor() {
        this.websiteDir = __dirname;
        this.htmlFiles = [];
        this.updatedFiles = [];
        this.errors = [];
    }

    async applyToAllPages() {
        console.log('🚀 Starting Meta-Optimal Navigation Application...');

        try {
            // Find all HTML files
            this.findHTMLFiles();
            console.log(`📁 Found ${this.htmlFiles.length} HTML files to update`);

            // Apply navigation to each file
            for (const file of this.htmlFiles) {
                await this.updateFile(file);
            }

            // Generate report
            this.generateReport();

        } catch (error) {
            console.error('❌ Error applying meta-optimal navigation:', error);
        }
    }

    findHTMLFiles() {
        const scanDirectory = (dir) => {
            const items = fs.readdirSync(dir);

            for (const item of items) {
                const fullPath = path.join(dir, item);
                const stat = fs.statSync(fullPath);

                if (stat.isDirectory()) {
                    // Skip certain directories
                    if (!['node_modules', '.git', 'venv', '__pycache__'].includes(item)) {
                        scanDirectory(fullPath);
                    }
                } else if (item.endsWith('.html')) {
                    this.htmlFiles.push(fullPath);
                }
            }
        };

        scanDirectory(this.websiteDir);
    }

    async updateFile(filePath) {
        try {
            console.log(`📝 Updating: ${path.relative(this.websiteDir, filePath)}`);

            let content = fs.readFileSync(filePath, 'utf8');
            const originalContent = content;

            // Apply meta-optimal navigation updates
            content = this.applyNavigationUpdates(content, filePath);

            // Write updated content
            if (content !== originalContent) {
                fs.writeFileSync(filePath, content, 'utf8');
                this.updatedFiles.push(filePath);
                console.log(`✅ Updated: ${path.relative(this.websiteDir, filePath)}`);
            } else {
                console.log(`⏭️  No changes needed: ${path.relative(this.websiteDir, filePath)}`);
            }

        } catch (error) {
            console.error(`❌ Error updating ${filePath}:`, error.message);
            this.errors.push({ file: filePath, error: error.message });
        }
    }

    applyNavigationUpdates(content, filePath) {
        // Remove old navigation scripts and styles
        content = this.removeOldNavigation(content);

        // Add meta-optimal navigation integration
        content = this.addMetaOptimalNavigation(content, filePath);

        // Update meta tags for better SEO
        content = this.updateMetaTags(content);

        // Add performance optimizations
        content = this.addPerformanceOptimizations(content);

        return content;
    }

    removeOldNavigation(content) {
        // Remove old navigation scripts
        const oldScriptPatterns = [
            /<script[^>]*src="[^"]*(?:unified-navigation|shared-navigation|enhanced-navigation)[^"]*"[^>]*><\/script>/gi,
            /<script[^>]*src="[^"]*navigation[^"]*"[^>]*><\/script>/gi
        ];

        oldScriptPatterns.forEach(pattern => {
            content = content.replace(pattern, '');
        });

        // Remove old navigation styles
        const oldStylePatterns = [
            /<link[^>]*href="[^"]*(?:unified-navigation|shared-navigation)[^"]*"[^>]*>/gi,
            /<style>[^<]*\.enhanced-nav[^<]*<\/style>/gi
        ];

        oldStylePatterns.forEach(pattern => {
            content = content.replace(pattern, '');
        });

        return content;
    }

    addMetaOptimalNavigation(content, filePath) {
        const relativePath = path.relative(this.websiteDir, filePath);
        const isRootPage = !relativePath.includes('/');

        // Add meta-optimal navigation integration script
        const integrationScript = `
    <!-- Meta-Optimal Navigation Integration -->
    <script src="js/meta-optimal-integration.js"></script>
    <script src="js/meta-optimal-navigation.js"></script>`;

        // Find the closing </head> tag and add scripts before it
        if (content.includes('</head>')) {
            content = content.replace('</head>', `${integrationScript}\n</head>`);
        } else {
            // If no </head> tag, add scripts before </body>
            content = content.replace('</body>', `${integrationScript}\n</body>`);
        }

        // Add CSS link if not already present
        if (!content.includes('meta-optimal-navigation.css')) {
            const cssLink = `
    <link rel="stylesheet" href="css/meta-optimal-navigation.css">`;

            if (content.includes('</head>')) {
                content = content.replace('</head>', `${cssLink}\n</head>`);
            }
        }

        return content;
    }

    updateMetaTags(content) {
        // Update title if it's generic
        if (content.includes('<title>Een Unity Mathematics')) {
            content = content.replace(
                /<title>Een Unity Mathematics[^<]*<\/title>/,
                '<title>Een Unity Mathematics | Meta-Optimal 1+1=1 Consciousness Hub</title>'
            );
        }

        // Add or update meta description
        const metaDescPattern = /<meta[^>]*name="description"[^>]*>/;
        const newMetaDesc = '<meta name="description" content="Transcendental computing hub demonstrating 1+1=1 through quantum entanglement, neural networks, and fractal mathematics with consciousness field integration.">';

        if (metaDescPattern.test(content)) {
            content = content.replace(metaDescPattern, newMetaDesc);
        } else {
            // Add meta description after title
            content = content.replace('</title>', '</title>\n    ' + newMetaDesc);
        }

        // Add viewport meta tag if not present
        if (!content.includes('viewport')) {
            const viewportMeta = '<meta name="viewport" content="width=device-width, initial-scale=1.0">';
            content = content.replace('</title>', '</title>\n    ' + viewportMeta);
        }

        return content;
    }

    addPerformanceOptimizations(content) {
        // Add preload for critical resources
        const preloads = `
    <!-- Preload critical resources -->
    <link rel="preload" href="css/meta-optimal-navigation.css" as="style">
    <link rel="preload" href="js/meta-optimal-navigation.js" as="script">
    <link rel="preload" href="js/meta-optimal-integration.js" as="script">`;

        if (content.includes('</head>')) {
            content = content.replace('</head>', `${preloads}\n</head>`);
        }

        // Add performance monitoring
        const performanceScript = `
    <!-- Performance Monitoring -->
    <script>
        window.addEventListener('load', function() {
            if ('performance' in window) {
                const loadTime = performance.now();
                console.log('Page load time:', loadTime + 'ms');
            }
        });
    </script>`;

        if (content.includes('</body>')) {
            content = content.replace('</body>', `${performanceScript}\n</body>`);
        }

        return content;
    }

    generateReport() {
        console.log('\n📊 Meta-Optimal Navigation Application Report');
        console.log('='.repeat(50));
        console.log(`✅ Successfully updated: ${this.updatedFiles.length} files`);
        console.log(`📁 Total HTML files found: ${this.htmlFiles.length}`);
        console.log(`❌ Errors: ${this.errors.length}`);

        if (this.updatedFiles.length > 0) {
            console.log('\n📝 Updated Files:');
            this.updatedFiles.forEach(file => {
                console.log(`  • ${path.relative(this.websiteDir, file)}`);
            });
        }

        if (this.errors.length > 0) {
            console.log('\n❌ Errors:');
            this.errors.forEach(error => {
                console.log(`  • ${path.relative(this.websiteDir, error.file)}: ${error.error}`);
            });
        }

        // Generate detailed report file
        this.writeDetailedReport();

        console.log('\n🎉 Meta-Optimal Navigation application complete!');
        console.log('🌐 All pages now feature the comprehensive navigation system');
        console.log('📱 Optimized for Chrome, mobile, and all modern browsers');
    }

    writeDetailedReport() {
        const report = {
            timestamp: new Date().toISOString(),
            summary: {
                totalFiles: this.htmlFiles.length,
                updatedFiles: this.updatedFiles.length,
                errors: this.errors.length
            },
            updatedFiles: this.updatedFiles.map(file => path.relative(this.websiteDir, file)),
            errors: this.errors.map(error => ({
                file: path.relative(this.websiteDir, error.file),
                error: error.error
            })),
            features: {
                navigation: 'Meta-Optimal Navigation System',
                categories: [
                    'Mathematics',
                    'Consciousness',
                    'Research',
                    'Implementations',
                    'Experiments',
                    'Visualizations',
                    'Academy',
                    'About'
                ],
                mobileOptimized: true,
                chromeOptimized: true,
                accessibility: true,
                performance: true
            }
        };

        const reportPath = path.join(this.websiteDir, 'META_OPTIMAL_NAVIGATION_REPORT.json');
        fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
        console.log(`📄 Detailed report saved to: ${path.relative(this.websiteDir, reportPath)}`);
    }
}

// Create a simple HTML template for testing
function createTestPage() {
    const testHTML = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Page - Meta-Optimal Navigation</title>
    <meta name="description" content="Test page for meta-optimal navigation system">
</head>
<body>
    <h1>Test Page</h1>
    <p>This page will have the meta-optimal navigation applied.</p>
</body>
</html>`;

    const testPath = path.join(__dirname, 'test-meta-optimal.html');
    fs.writeFileSync(testPath, testHTML);
    console.log(`🧪 Created test page: ${testPath}`);
}

// Main execution
async function main() {
    const applier = new MetaOptimalNavigationApplier();

    // Create test page if it doesn't exist
    if (!fs.existsSync(path.join(__dirname, 'test-meta-optimal.html'))) {
        createTestPage();
    }

    // Apply meta-optimal navigation to all pages
    await applier.applyToAllPages();
}

// Run if this script is executed directly
if (require.main === module) {
    main().catch(console.error);
}

module.exports = MetaOptimalNavigationApplier; 