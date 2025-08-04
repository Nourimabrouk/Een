/**
 * Een Unity Mathematics - Automatic Page Updater
 * Updates all HTML pages with unified navigation and AI chat integration
 */

class PageUpdater {
    constructor() {
        this.pagesToUpdate = [
            'proofs.html',
            'research.html',
            'publications.html',
            'playground.html',
            'learn.html',
            'metagambit.html',
            'about.html',
            'implementations.html',
            'consciousness_dashboard.html',
            'philosophy.html',
            'unity_visualization.html',
            'unity_consciousness_experience.html',
            'enhanced-unity-demo.html',
            'mathematical_playground.html',
            'agents.html',
            'metagamer_agent.html',
            'mobile-app.html',
            'dashboards.html',
            'revolutionary-landing.html',
            'meta-optimal-landing.html',
            '3000-elo-proof.html',
            'further-reading.html',
            'learning.html'
        ];
    }

    // Method to update a single page
    updatePage(pageName) {
        console.log(`Updating ${pageName}...`);

        // This would be implemented in a Node.js environment
        // For now, we'll provide the manual steps needed

        const steps = [
            `1. Open ${pageName}`,
            '2. Add these script tags in the <head> section:',
            '   <script src="js/unified-navigation.js"></script>',
            '   <script src="js/ai-chat-integration.js"></script>',
            '3. Add padding-top: 80px to the body CSS',
            '4. Add margin-top: -80px and padding-top: 80px to the main header section',
            '5. Add AI chat initialization in the DOMContentLoaded event:',
            '   if (window.EenAIChat) {',
            '       window.eenChat = EenAIChat.initialize();',
            '   }',
            '6. Remove any conflicting navigation scripts',
            '7. Test the page functionality'
        ];

        return steps;
    }

    // Method to generate update instructions for all pages
    generateUpdateInstructions() {
        const instructions = {
            title: 'Een Unity Mathematics - Page Update Instructions',
            description: 'Follow these steps to update all pages with unified navigation and AI chat',
            pages: {}
        };

        this.pagesToUpdate.forEach(page => {
            instructions.pages[page] = this.updatePage(page);
        });

        return instructions;
    }

    // Method to check if a page needs updates
    checkPageStatus(pageName) {
        const commonIssues = [
            'Missing unified navigation',
            'Missing AI chat integration',
            'Incorrect body padding',
            'Conflicting navigation scripts',
            'Missing accessibility features'
        ];

        return {
            page: pageName,
            needsUpdate: true,
            issues: commonIssues,
            priority: this.getPagePriority(pageName)
        };
    }

    // Get page priority for updates
    getPagePriority(pageName) {
        const highPriority = ['proofs.html', 'research.html', 'about.html'];
        const mediumPriority = ['publications.html', 'playground.html', 'implementations.html'];

        if (highPriority.includes(pageName)) return 'high';
        if (mediumPriority.includes(pageName)) return 'medium';
        return 'low';
    }

    // Generate a summary report
    generateReport() {
        const report = {
            totalPages: this.pagesToUpdate.length,
            highPriority: this.pagesToUpdate.filter(p => this.getPagePriority(p) === 'high').length,
            mediumPriority: this.pagesToUpdate.filter(p => this.getPagePriority(p) === 'medium').length,
            lowPriority: this.pagesToUpdate.filter(p => this.getPagePriority(p) === 'low').length,
            estimatedTime: '2-3 hours for all pages',
            recommendations: [
                'Start with high priority pages first',
                'Test each page after updating',
                'Ensure all links work correctly',
                'Verify AI chat functionality',
                'Check mobile responsiveness'
            ]
        };

        return report;
    }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PageUpdater;
}

// Auto-run if in browser
if (typeof window !== 'undefined') {
    const updater = new PageUpdater();
    console.log('Een Unity Mathematics Page Updater loaded');
    console.log('Use updater.generateUpdateInstructions() to get update steps');
    console.log('Use updater.generateReport() to get status report');
    window.PageUpdater = updater;
} 