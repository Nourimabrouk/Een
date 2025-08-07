/**
 * Comprehensive Navigation and Styling Update Script
 * Updates all HTML pages to use unified navigation and consistent styling
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
    'metastation-hub.html'
];

// Unified navigation injection function
function injectUnifiedNavigation(htmlContent) {
    // Remove existing navigation if present
    let updatedContent = htmlContent.replace(
        /<nav[^>]*>[\s\S]*?<\/nav>/gi,
        '<!-- Navigation will be injected by shared-navigation.js -->'
    );

    // Add navigation CSS and JS if not present
    if (!updatedContent.includes('meta-optimal-navigation.css')) {
        updatedContent = updatedContent.replace(
            /<link[^>]*font-awesome[^>]*>/i,
            `$&\n    <!-- Unified Navigation System -->
    <link rel="stylesheet" href="css/meta-optimal-navigation.css">
    <script src="shared-navigation.js" defer></script>`
        );
    }

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

// Fix common styling issues
function fixStylingIssues(htmlContent) {
    let updatedContent = htmlContent;

    // Replace Tailwind classes with custom CSS classes
    const tailwindReplacements = {
        'bg-zinc-950': 'bg-primary',
        'bg-zinc-900': 'bg-secondary',
        'bg-zinc-800': 'bg-tertiary',
        'text-zinc-100': 'text-primary',
        'text-zinc-300': 'text-secondary',
        'text-zinc-400': 'text-muted',
        'border-zinc-700': 'border-default',
        'hover:text-purple-400': 'hover:text-accent',
        'hover:bg-purple-500': 'hover:bg-accent',
        'hover:border-purple-500': 'hover:border-accent'
    };

    Object.entries(tailwindReplacements).forEach(([oldClass, newClass]) => {
        const regex = new RegExp(`\\b${oldClass}\\b`, 'g');
        updatedContent = updatedContent.replace(regex, newClass);
    });

    // Add consistent CSS variables if not present
    if (!updatedContent.includes('--phi:')) {
        const cssVariables = `
        /* Meta-Optimal Design System */
        :root {
            --phi: 1.618033988749895;
            --primary-color: #0a0a0a;
            --secondary-color: #1a1a1a;
            --accent-color: #2d2d2d;
            --unity-gold: #FFD700;
            --consciousness-purple: #6B46C1;
            --quantum-blue: #4ECDC4;
            --fractal-orange: #FF6B6B;
            --neural-pink: #FF69B4;
            --godel-red: #E74C3C;
            --tarski-green: #27AE60;
            --text-primary: #ffffff;
            --text-secondary: #cccccc;
            --text-muted: #888888;
            --bg-primary: #0a0a0a;
            --bg-secondary: #1a1a1a;
            --bg-tertiary: #2d2d2d;
            --border-color: rgba(255, 215, 0, 0.2);
            --shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            --shadow-lg: 0 10px 40px rgba(0, 0, 0, 0.5);
            --shadow-glow: 0 0 30px rgba(255, 215, 0, 0.3);
            --radius: 12px;
            --radius-lg: 20px;
            --radius-xl: 30px;
            --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            --transition-smooth: all 0.5s cubic-bezier(0.25, 0.8, 0.25, 1);
            --font-primary: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-serif: 'Crimson Text', Georgia, serif;
            --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
        }`;

        updatedContent = updatedContent.replace(
            /<style>/i,
            `<style>${cssVariables}`
        );
    }

    return updatedContent;
}

// Add performance optimizations
function addPerformanceOptimizations(htmlContent) {
    let updatedContent = htmlContent;

    // Add preconnect links if not present
    if (!updatedContent.includes('preconnect')) {
        const preconnectLinks = `
    <!-- Performance Optimizations -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link rel="preconnect" href="https://cdnjs.cloudflare.com">
    <link rel="preconnect" href="https://cdn.plot.ly">
    <link rel="preconnect" href="https://cdn.jsdelivr.net">`;

        updatedContent = updatedContent.replace(
            /<meta[^>]*viewport[^>]*>/i,
            `$&${preconnectLinks}`
        );
    }

    // Add Google Fonts if not present
    if (!updatedContent.includes('fonts.googleapis.com')) {
        const googleFonts = `
    <link
        href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Crimson+Text:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
        rel="stylesheet">`;

        updatedContent = updatedContent.replace(
            /<link[^>]*font-awesome[^>]*>/i,
            `${googleFonts}\n$&`
        );
    }

    return updatedContent;
}

// Add responsive design improvements
function addResponsiveDesign(htmlContent) {
    let updatedContent = htmlContent;

    // Add responsive CSS if not present
    if (!updatedContent.includes('@media (max-width: 768px)')) {
        const responsiveCSS = `
        /* Responsive Design */
        @media (max-width: 768px) {
            h1 {
                font-size: 2.5rem;
            }

            h2 {
                font-size: 2rem;
            }

            section {
                padding: 2rem 1rem;
            }

            .container {
                padding: 0 1rem;
            }

            .grid-2 {
                grid-template-columns: 1fr;
            }

            .grid-3 {
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            }
        }

        @media (max-width: 480px) {
            h1 {
                font-size: 2rem;
            }

            h2 {
                font-size: 1.5rem;
            }

            .card {
                padding: 1.5rem;
            }
        }`;

        updatedContent = updatedContent.replace(
            /<\/style>/i,
            `${responsiveCSS}\n    </style>`
        );
    }

    return updatedContent;
}

// Add animation classes
function addAnimationClasses(htmlContent) {
    let updatedContent = htmlContent;

    // Add animation CSS if not present
    if (!updatedContent.includes('fadeInUp')) {
        const animationCSS = `
        /* Animation Classes */
        .fade-in {
            opacity: 0;
            transform: translateY(30px);
            animation: fadeInUp 0.8s ease-out forwards;
        }

        @keyframes fadeInUp {
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .stagger-animation > * {
            opacity: 0;
            animation: fadeInUp 0.8s ease-out forwards;
        }

        .stagger-animation > *:nth-child(1) { animation-delay: 0.1s; }
        .stagger-animation > *:nth-child(2) { animation-delay: 0.2s; }
        .stagger-animation > *:nth-child(3) { animation-delay: 0.3s; }
        .stagger-animation > *:nth-child(4) { animation-delay: 0.4s; }
        .stagger-animation > *:nth-child(5) { animation-delay: 0.5s; }`;

        updatedContent = updatedContent.replace(
            /<\/style>/i,
            `${animationCSS}\n    </style>`
        );
    }

    return updatedContent;
}

// Add consistent component classes
function addComponentClasses(htmlContent) {
    let updatedContent = htmlContent;

    // Add component CSS if not present
    if (!updatedContent.includes('.card {')) {
        const componentCSS = `
        /* Component Classes */
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 2rem;
        }

        .grid {
            display: grid;
            gap: 2rem;
        }

        .grid-2 {
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        }

        .grid-3 {
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        }

        .card {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            border-radius: var(--radius);
            padding: 2rem;
            transition: var(--transition);
        }

        .card:hover {
            transform: translateY(-5px);
            box-shadow: var(--shadow-lg);
            border-color: var(--unity-gold);
        }

        .card-icon {
            font-size: 3rem;
            color: var(--consciousness-purple);
            margin-bottom: 1rem;
        }

        .card-title {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: var(--text-primary);
        }

        .card-text {
            color: var(--text-secondary);
            line-height: 1.6;
        }

        .btn {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: var(--radius);
            font-family: var(--font-primary);
            font-weight: 500;
            text-decoration: none;
            cursor: pointer;
            transition: var(--transition);
            background: var(--bg-tertiary);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
        }

        .btn:hover {
            background: var(--consciousness-purple);
            border-color: var(--consciousness-purple);
            transform: translateY(-2px);
        }

        .btn-primary {
            background: var(--consciousness-purple);
            border-color: var(--consciousness-purple);
        }

        .btn-primary:hover {
            background: var(--unity-gold);
            border-color: var(--unity-gold);
            color: var(--bg-primary);
        }

        .section-dark {
            background: var(--bg-secondary);
            padding: 4rem 2rem;
        }

        .section-darker {
            background: var(--bg-primary);
            padding: 4rem 2rem;
        }`;

        updatedContent = updatedContent.replace(
            /<\/style>/i,
            `${componentCSS}\n    </style>`
        );
    }

    return updatedContent;
}

// Main update function
function updateHTMLFile(filePath) {
    try {
        console.log(`Processing: ${filePath}`);

        let content = fs.readFileSync(filePath, 'utf8');

        // Apply all updates
        content = injectUnifiedNavigation(content);
        content = fixStylingIssues(content);
        content = addPerformanceOptimizations(content);
        content = addResponsiveDesign(content);
        content = addAnimationClasses(content);
        content = addComponentClasses(content);

        // Write updated content back to file
        fs.writeFileSync(filePath, content, 'utf8');

        console.log(`✓ Updated: ${filePath}`);
        return true;
    } catch (error) {
        console.error(`✗ Error updating ${filePath}:`, error.message);
        return false;
    }
}

// Process all HTML files
function updateAllPages() {
    console.log('🚀 Starting comprehensive navigation and styling update...\n');

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
        console.log('\n🎉 All pages updated successfully!');
        console.log('\n✨ Key improvements applied:');
        console.log('   • Unified navigation system');
        console.log('   • Consistent styling with CSS variables');
        console.log('   • Performance optimizations');
        console.log('   • Responsive design');
        console.log('   • Animation classes');
        console.log('   • Component classes');
        console.log('   • Academic and professional aesthetic');
    } else {
        console.log('\n⚠️  Some files failed to update. Check the errors above.');
    }
}

// Run the update
if (require.main === module) {
    updateAllPages();
}

module.exports = {
    updateHTMLFile,
    updateAllPages,
    injectUnifiedNavigation,
    fixStylingIssues,
    addPerformanceOptimizations,
    addResponsiveDesign,
    addAnimationClasses,
    addComponentClasses
}; 