/**
 * Apply Unified Navigation to All Pages - Batch Processing Script
 * This Node.js script automatically applies the one-line navigation fix to all HTML pages
 */

const fs = require('fs');
const path = require('path');
const glob = require('glob');

// Configuration
const WEBSITE_DIR = '.';
const BACKUP_DIR = './navigation-backups';
const DRY_RUN = false; // Set to true to preview changes without applying them

// Navigation template to inject
const NAVIGATION_TEMPLATE = `    <!-- Unified Navigation System - Meta-Optimal One-Line Solution -->
    <script src="js/nav-template-applier.js" defer></script>`;

// Legacy navigation patterns to remove or replace
const LEGACY_PATTERNS = [
    // CSS patterns
    {
        pattern: /<link[^>]*href[^>]*meta-optimal-navigation[^>]*>/gi,
        replacement: ''
    },
    {
        pattern: /<link[^>]*href[^>]*unified-navigation-system[^>]*>/gi,
        replacement: ''
    },
    {
        pattern: /<link[^>]*href[^>]*navigation[^>]*>/gi,
        replacement: ''
    },
    // JS patterns  
    {
        pattern: /<script[^>]*src[^>]*meta-optimal-navigation[^>]*><\/script>/gi,
        replacement: ''
    },
    {
        pattern: /<script[^>]*src[^>]*unified-navigation-system[^>]*><\/script>/gi,
        replacement: ''
    },
    {
        pattern: /<script[^>]*src[^>]*navigation[^>]*><\/script>/gi,
        replacement: ''
    },
    // Comment cleanup
    {
        pattern: /<!--[\s\S]*?Meta-Optimal.*?Navigation[\s\S]*?-->/gi,
        replacement: ''
    },
    {
        pattern: /<!--[\s\S]*?Navigation.*?System[\s\S]*?-->/gi,
        replacement: ''
    }
];

// Files to skip (already manually updated or special cases)
const SKIP_FILES = [
    './metastation-hub.html',
    './implementations-gallery.html', 
    './zen-unity-meditation.html',
    './meta_tags_template.html',
    './redirect.html',
    './google5936e6fc51b68c92.html'
];

function createBackupDir() {
    if (!fs.existsSync(BACKUP_DIR)) {
        fs.mkdirSync(BACKUP_DIR, { recursive: true });
        console.log(`📁 Created backup directory: ${BACKUP_DIR}`);
    }
}

function backupFile(filePath) {
    const backupPath = path.join(BACKUP_DIR, path.basename(filePath));
    fs.copyFileSync(filePath, backupPath);
    console.log(`💾 Backed up: ${filePath} → ${backupPath}`);
}

function hasNavigationTemplate(content) {
    return content.includes('nav-template-applier.js') || 
           content.includes('unified-navigation.js');
}

function removeLegacyNavigation(content) {
    let cleanContent = content;
    
    LEGACY_PATTERNS.forEach(({ pattern, replacement }) => {
        cleanContent = cleanContent.replace(pattern, replacement);
    });
    
    // Clean up multiple blank lines
    cleanContent = cleanContent.replace(/\n\s*\n\s*\n/g, '\n\n');
    
    return cleanContent;
}

function injectNavigationTemplate(content) {
    // Find the closing </head> tag and inject before it
    const headEndPattern = /<\/head>/i;
    const headEndMatch = content.match(headEndPattern);
    
    if (!headEndMatch) {
        console.warn('⚠️ No </head> tag found, adding to top of <body>');
        const bodyStartPattern = /<body[^>]*>/i;
        return content.replace(bodyStartPattern, (match) => {
            return match + '\n' + NAVIGATION_TEMPLATE + '\n';
        });
    }
    
    return content.replace(headEndPattern, NAVIGATION_TEMPLATE + '\n</head>');
}

function processFile(filePath) {
    console.log(`\n🔧 Processing: ${filePath}`);
    
    // Skip if in skip list
    const relativePath = './' + path.relative(WEBSITE_DIR, filePath).replace(/\\/g, '/');
    if (SKIP_FILES.includes(relativePath)) {
        console.log(`⏭️ Skipping (already updated): ${relativePath}`);
        return { skipped: true, reason: 'already_updated' };
    }
    
    // Read file
    let content;
    try {
        content = fs.readFileSync(filePath, 'utf8');
    } catch (error) {
        console.error(`❌ Error reading ${filePath}:`, error.message);
        return { error: true, reason: error.message };
    }
    
    // Check if already has unified navigation
    if (hasNavigationTemplate(content)) {
        console.log(`✅ Already has unified navigation: ${relativePath}`);
        return { skipped: true, reason: 'already_unified' };
    }
    
    // Create backup
    if (!DRY_RUN) {
        backupFile(filePath);
    }
    
    // Process content
    let newContent = content;
    
    // Remove legacy navigation
    newContent = removeLegacyNavigation(newContent);
    
    // Inject unified navigation template
    newContent = injectNavigationTemplate(newContent);
    
    if (DRY_RUN) {
        console.log(`🔍 [DRY RUN] Would update: ${relativePath}`);
        return { dryRun: true };
    }
    
    // Write updated file
    try {
        fs.writeFileSync(filePath, newContent, 'utf8');
        console.log(`✅ Updated: ${relativePath}`);
        return { success: true };
    } catch (error) {
        console.error(`❌ Error writing ${filePath}:`, error.message);
        return { error: true, reason: error.message };
    }
}

function main() {
    console.log('🚀 Starting Unified Navigation System Deployment');
    console.log(`📁 Working directory: ${path.resolve(WEBSITE_DIR)}`);
    console.log(`🔧 Mode: ${DRY_RUN ? 'DRY RUN (preview only)' : 'LIVE DEPLOYMENT'}`);
    
    if (!DRY_RUN) {
        createBackupDir();
    }
    
    // Find all HTML files
    const htmlFiles = glob.sync('**/*.html', { 
        cwd: WEBSITE_DIR,
        ignore: [
            'node_modules/**',
            'navigation-backups/**',
            '**/node_modules/**'
        ]
    });
    
    console.log(`📄 Found ${htmlFiles.length} HTML files to process`);
    
    // Process each file
    const results = {
        processed: 0,
        updated: 0,
        skipped: 0,
        errors: 0,
        dryRun: 0
    };
    
    htmlFiles.forEach((file) => {
        const filePath = path.join(WEBSITE_DIR, file);
        const result = processFile(filePath);
        
        results.processed++;
        
        if (result.success) results.updated++;
        else if (result.skipped) results.skipped++;
        else if (result.error) results.errors++;
        else if (result.dryRun) results.dryRun++;
    });
    
    // Summary
    console.log('\n' + '='.repeat(60));
    console.log('📊 DEPLOYMENT SUMMARY');
    console.log('='.repeat(60));
    console.log(`📄 Total files processed: ${results.processed}`);
    
    if (DRY_RUN) {
        console.log(`🔍 Files that would be updated: ${results.dryRun}`);
    } else {
        console.log(`✅ Files successfully updated: ${results.updated}`);
        console.log(`💾 Backups created in: ${BACKUP_DIR}`);
    }
    
    console.log(`⏭️ Files skipped: ${results.skipped}`);
    console.log(`❌ Errors encountered: ${results.errors}`);
    
    if (results.errors === 0) {
        console.log('\n🎉 SUCCESS: All pages now have unified navigation!');
        console.log('\n📋 Next steps:');
        console.log('1. Test pages: http://localhost:8001/metastation-hub.html');
        console.log('2. Verify navigation works on all screen sizes');
        console.log('3. Clean up legacy navigation files');
    } else {
        console.log('\n⚠️ Some errors occurred. Check the logs above.');
    }
    
    console.log('\n🌟 Unity Mathematics Navigation System Deployed! 🌟');
}

// Run if called directly
if (require.main === module) {
    main();
}

module.exports = { main, processFile, LEGACY_PATTERNS, NAVIGATION_TEMPLATE };