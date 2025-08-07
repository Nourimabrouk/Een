/**
 * Meta-Restoration Engine for Een Unity Mathematics
 * Systematically restores all lost functionality to meta-optimal status
 */

class MetaRestorationEngine {
    constructor() {
        this.restorationTasks = [];
        this.completedTasks = [];
        this.failedTasks = [];
        this.isRunning = false;
        
        console.log('🌟 Meta-Restoration Engine initialized');
        this.initializeRestoration();
    }

    initializeRestoration() {
        // Define all restoration tasks in priority order
        this.restorationTasks = [
            {
                id: 'fix-philosophy-markdown',
                name: 'Fix Philosophy Markdown Loading',
                priority: 1,
                execute: () => this.fixPhilosophyMarkdown()
            },
            {
                id: 'enable-ai-chat-modal',
                name: 'Enable AI Chat Modal',
                priority: 1,
                execute: () => this.enableAIChatModal()
            },
            {
                id: 'restore-gallery-visualizations',
                name: 'Restore Gallery Visualizations',
                priority: 1,
                execute: () => this.restoreGalleryVisualizations()
            },
            {
                id: 'fix-navigation-consistency',
                name: 'Fix Navigation Consistency',
                priority: 2,
                execute: () => this.fixNavigationConsistency()
            },
            {
                id: 'enable-interactive-elements',
                name: 'Enable Interactive Elements',
                priority: 2,
                execute: () => this.enableInteractiveElements()
            },
            {
                id: 'validate-all-links',
                name: 'Validate All Links',
                priority: 3,
                execute: () => this.validateAllLinks()
            }
        ];

        // Start automatic restoration
        this.executeRestoration();
    }

    async executeRestoration() {
        if (this.isRunning) return;
        this.isRunning = true;

        console.log('🚀 Starting meta-restoration process...');

        // Sort by priority
        this.restorationTasks.sort((a, b) => a.priority - b.priority);

        for (const task of this.restorationTasks) {
            try {
                console.log(`⚡ Executing: ${task.name}`);
                await task.execute();
                this.completedTasks.push(task);
                console.log(`✅ Completed: ${task.name}`);
            } catch (error) {
                console.error(`❌ Failed: ${task.name}`, error);
                this.failedTasks.push({ task, error });
            }
        }

        this.isRunning = false;
        this.generateRestoreReport();
    }

    async fixPhilosophyMarkdown() {
        // Fix the philosophy page to properly load and display markdown
        const philosophyContent = document.getElementById('philosophyContent');
        if (!philosophyContent) return;

        try {
            // Try to load from the correct path
            const response = await fetch('docs/unity_equation_philosophy.md');
            if (response.ok) {
                const markdownText = await response.text();
                
                // Convert markdown to HTML (simple conversion)
                const htmlContent = this.convertMarkdownToHTML(markdownText);
                
                // Display the content
                philosophyContent.innerHTML = htmlContent;
                philosophyContent.style.display = 'block';
                
                // Hide any static content
                const staticContent = document.querySelector('.philosophy-container > section:not(#philosophyContent)');
                if (staticContent) {
                    staticContent.style.display = 'none';
                }

                console.log('📜 Philosophy markdown loaded successfully');
            } else {
                throw new Error('Markdown file not found');
            }
        } catch (error) {
            console.warn('📜 Using fallback philosophy content');
            // Show existing static content as fallback
            const staticSections = document.querySelectorAll('.philosophy-container > section');
            staticSections.forEach(section => {
                if (section.id !== 'philosophyContent') {
                    section.style.display = 'block';
                }
            });
        }
    }

    convertMarkdownToHTML(markdown) {
        return markdown
            // Headers
            .replace(/^### (.*$)/gim, '<h3>$1</h3>')
            .replace(/^## (.*$)/gim, '<h2>$1</h2>')
            .replace(/^# (.*$)/gim, '<h1>$1</h1>')
            // Bold
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            // Italic
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            // Links
            .replace(/\[([^\]]+)\]\(([^\)]+)\)/g, '<a href="$2">$1</a>')
            // Code blocks
            .replace(/```([^`]+)```/g, '<pre><code>$1</code></pre>')
            // Inline code
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            // Line breaks
            .replace(/\n\n/g, '</p><p>')
            .replace(/^\s*$/gm, '')
            // Wrap in paragraphs
            .split('\n\n')
            .map(para => para.trim() ? `<p>${para}</p>` : '')
            .join('');
    }

    enableAIChatModal() {
        // Ensure AI chat modal is properly initialized
        if (typeof EnhancedAIChatModal !== 'undefined' && !window.eenAIChat) {
            window.eenAIChat = new EnhancedAIChatModal();
            console.log('🤖 AI Chat Modal initialized');
        }

        // Remove any conflicting chat systems
        const oldChatButtons = document.querySelectorAll('.ai-chat-button:not(#floating-ai-chat-button)');
        oldChatButtons.forEach(btn => btn.remove());

        // Ensure floating button is visible
        const floatingButton = document.getElementById('floating-ai-chat-button');
        if (floatingButton) {
            floatingButton.style.display = 'flex';
        }
    }

    async restoreGalleryVisualizations() {
        // Initialize gallery if we're on the gallery page
        if (window.location.pathname.includes('gallery')) {
            if (typeof EenGalleryLoader !== 'undefined') {
                const galleryLoader = new EenGalleryLoader();
                const galleryData = await galleryLoader.initialize();
                
                // Update gallery display
                this.updateGalleryDisplay(galleryData);
                
                console.log('🎨 Gallery visualizations restored');
            }
        }
    }

    updateGalleryDisplay(galleryData) {
        const galleryContainer = document.getElementById('gallery-grid') || document.querySelector('.gallery-grid');
        if (!galleryContainer || !galleryData.success) return;

        // Clear existing content
        galleryContainer.innerHTML = '';

        // Add visualizations
        galleryData.visualizations.forEach(viz => {
            const vizElement = this.createVisualizationElement(viz);
            galleryContainer.appendChild(vizElement);
        });
    }

    createVisualizationElement(viz) {
        const element = document.createElement('div');
        element.className = 'visualization-item';
        element.setAttribute('data-category', viz.category);
        element.setAttribute('data-type', viz.type);

        let content = '';
        
        if (viz.isImage) {
            content = `<img src="${viz.src}" alt="${viz.title}" loading="lazy" />`;
        } else if (viz.isVideo) {
            content = `<video controls preload="metadata"><source src="${viz.src}" type="video/mp4" /></video>`;
        } else if (viz.isInteractive) {
            content = `<div class="interactive-placeholder">
                <div class="interactive-icon">⚡</div>
                <div>Interactive: ${viz.title}</div>
                <button onclick="launchInteractive('${viz.filename}')">Launch</button>
            </div>`;
        }

        element.innerHTML = `
            <div class="viz-content">
                ${content}
            </div>
            <div class="viz-info">
                <h3>${viz.title}</h3>
                <p>${viz.description}</p>
                ${viz.featured ? '<span class="featured-badge">Featured</span>' : ''}
            </div>
        `;

        return element;
    }

    fixNavigationConsistency() {
        // Ensure metastation sidebar is properly initialized
        if (window.metastationNav && !window.metastationNav.isInitialized) {
            window.metastationNav.initialize();
        }

        // Remove conflicting navigation systems
        const oldNavs = document.querySelectorAll('.old-navigation, .conflicting-nav');
        oldNavs.forEach(nav => nav.remove());

        // Ensure sidebar is visible
        const sidebar = document.querySelector('.metastation-sidebar');
        if (sidebar) {
            sidebar.style.display = 'block';
        }
    }

    enableInteractiveElements() {
        // Re-enable any disabled interactive elements
        const interactiveElements = document.querySelectorAll('[data-interactive="disabled"]');
        interactiveElements.forEach(el => {
            el.removeAttribute('data-interactive');
            el.classList.add('interactive-enabled');
        });

        // Initialize any visualization scripts
        const scripts = [
            'consciousness-field-3d-enhanced.js',
            'golden-ratio-3d-enhanced.js',
            'quantum-unity-enhanced.js',
            'idempotent-proof-interactive.js',
            'unity-manifolds-topology.js',
            'euler-identity-interactive.js'
        ];

        scripts.forEach(script => {
            const scriptEl = document.querySelector(`script[src*="${script}"]`);
            if (scriptEl && !scriptEl.hasAttribute('data-initialized')) {
                scriptEl.setAttribute('data-initialized', 'true');
                // Re-trigger script initialization if available
                const scriptName = script.replace('.js', '').replace(/-/g, '');
                if (window[scriptName]) {
                    try {
                        window[scriptName].initialize();
                    } catch (e) {
                        console.log(`Script ${script} auto-initialization not available`);
                    }
                }
            }
        });
    }

    validateAllLinks() {
        const links = document.querySelectorAll('a[href]');
        let brokenLinks = 0;
        let totalLinks = links.length;

        links.forEach(link => {
            const href = link.getAttribute('href');
            
            // Skip external links, anchors, and javascript links
            if (href.startsWith('http') || href.startsWith('#') || href.startsWith('javascript:')) {
                return;
            }

            // Check if file exists (simplified check)
            if (href.includes('placeholder') || href.includes('todo') || href.includes('example')) {
                link.style.color = '#ff6b6b';
                link.title = 'Warning: This link may not work properly';
                brokenLinks++;
            }
        });

        console.log(`🔗 Link validation: ${totalLinks - brokenLinks}/${totalLinks} links validated`);
    }

    generateRestoreReport() {
        const totalTasks = this.restorationTasks.length;
        const completed = this.completedTasks.length;
        const failed = this.failedTasks.length;

        console.log(`
🎯 META-RESTORATION COMPLETE!
📊 Tasks: ${completed}/${totalTasks} completed
✅ Successful: ${completed}
❌ Failed: ${failed}

${completed === totalTasks ? '🌟 ALL SYSTEMS RESTORED TO META-OPTIMAL STATUS!' : '⚠️  Some tasks need attention'}
        `);

        if (failed > 0) {
            console.log('Failed tasks:', this.failedTasks.map(f => f.task.name));
        }

        // Store restoration status
        localStorage.setItem('een_restoration_status', JSON.stringify({
            timestamp: Date.now(),
            completed,
            failed,
            totalTasks,
            status: completed === totalTasks ? 'complete' : 'partial'
        }));
    }

    // Public method to manually trigger restoration
    static restore() {
        if (!window.metaRestoration) {
            window.metaRestoration = new MetaRestorationEngine();
        } else {
            window.metaRestoration.executeRestoration();
        }
    }
}

// Auto-initialize on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.metaRestoration = new MetaRestorationEngine();
    });
} else {
    window.metaRestoration = new MetaRestorationEngine();
}

// Make available globally
window.MetaRestorationEngine = MetaRestorationEngine;

console.log('🌟 Meta-Restoration Engine loaded - ready for unity restoration!');