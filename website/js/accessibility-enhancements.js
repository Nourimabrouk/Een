/**
 * Accessibility Enhancements
 * ==========================
 * 
 * Dynamic accessibility improvements for Een Unity Mathematics website
 * WCAG 2.1 AA compliance enhancements
 * Screen reader optimization
 * Keyboard navigation improvements
 */

class AccessibilityEnhancer {
    constructor() {
        this.isEnabled = true;
        this.announcer = null;
        this.focusHistory = [];
        this.currentFocusIndex = -1;
        
        // Configuration
        this.config = {
            announceChanges: true,
            keyboardNavigation: true,
            focusManagement: true,
            colorContrastCheck: true,
            skipLinks: true
        };
        
        this.init();
    }
    
    init() {
        console.log('🔧 Accessibility Enhancer initializing...');
        
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.enhance());
        } else {
            this.enhance();
        }
    }
    
    enhance() {
        console.log('♿ Applying accessibility enhancements...');
        
        try {
            this.createSkipLinks();
            this.createLiveRegion();
            this.enhanceSemantics();
            this.improveFormAccessibility();
            this.enhanceKeyboardNavigation();
            this.addARIALabels();
            this.improveColorContrast();
            this.addFocusManagement();
            this.enhanceInteractiveElements();
            this.addMathAccessibility();
            this.setupErrorHandling();
            
            console.log('✅ Accessibility enhancements applied successfully');
            this.announceToScreenReader('Accessibility enhancements loaded');
            
        } catch (error) {
            console.error('❌ Accessibility enhancement error:', error);
        }
    }
    
    /**
     * Create skip navigation links
     */
    createSkipLinks() {
        if (!this.config.skipLinks) return;
        
        const skipNav = document.createElement('nav');
        skipNav.className = 'skip-navigation';
        skipNav.setAttribute('aria-label', 'Skip navigation');
        
        const skipLinks = [
            { href: '#main-content', text: 'Skip to main content' },
            { href: '#navigation', text: 'Skip to navigation' },
            { href: '#unity-demo-container', text: 'Skip to Unity Mathematics demo' },
            { href: '#search', text: 'Skip to search' }
        ];
        
        skipLinks.forEach(link => {
            const skipLink = document.createElement('a');
            skipLink.href = link.href;
            skipLink.textContent = link.text;
            skipLink.className = 'skip-link';
            skipNav.appendChild(skipLink);
        });
        
        document.body.insertBefore(skipNav, document.body.firstChild);
    }
    
    /**
     * Create live region for screen reader announcements
     */ 
    createLiveRegion() {
        this.announcer = document.createElement('div');
        this.announcer.id = 'accessibility-announcer';
        this.announcer.setAttribute('aria-live', 'polite');
        this.announcer.setAttribute('aria-atomic', 'true');
        this.announcer.className = 'sr-only';
        document.body.appendChild(this.announcer);
    }
    
    /**
     * Enhance semantic HTML structure
     */
    enhanceSemantics() {
        // Identify and mark main content area
        const mainContent = document.querySelector('.page-hero, .section, main') || 
                           document.querySelector('body > *:not(nav):not(header):not(footer)');
        
        if (mainContent && !mainContent.closest('main')) {
            const main = document.createElement('main');
            main.id = 'main-content';
            main.setAttribute('role', 'main');
            main.setAttribute('aria-label', 'Main content');
            
            mainContent.parentNode.insertBefore(main, mainContent);
            main.appendChild(mainContent);
        }
        
        // Enhance navigation structure
        const navElements = document.querySelectorAll('nav, .navigation, .nav');
        navElements.forEach((nav, index) => {
            if (!nav.getAttribute('aria-label')) {
                const label = index === 0 ? 'Main navigation' : `Navigation ${index + 1}`;
                nav.setAttribute('aria-label', label);
            }
            
            // Enhance navigation lists
            const navLists = nav.querySelectorAll('ul');
            navLists.forEach(list => {
                list.setAttribute('role', 'menubar');
                const items = list.querySelectorAll('li');
                items.forEach(item => {
                    item.setAttribute('role', 'none');
                    const link = item.querySelector('a');
                    if (link) {
                        link.setAttribute('role', 'menuitem');
                    }
                });
            });
        });
        
        // Enhance section headings
        const sections = document.querySelectorAll('section, .section');
        sections.forEach((section, index) => {
            if (!section.getAttribute('aria-labelledby') && !section.getAttribute('aria-label')) {
                const heading = section.querySelector('h1, h2, h3, h4, h5, h6');
                if (heading) {
                    if (!heading.id) {
                        heading.id = `section-heading-${index}`;
                    }
                    section.setAttribute('aria-labelledby', heading.id);
                }
            }
        });
    }
    
    /**
     * Improve form accessibility
     */
    improveFormAccessibility() {
        const forms = document.querySelectorAll('form');
        forms.forEach((form, formIndex) => {
            // Add form labels
            if (!form.getAttribute('aria-label') && !form.getAttribute('aria-labelledby')) {
                form.setAttribute('aria-label', `Form ${formIndex + 1}`);
            }
            
            // Enhance form inputs
            const inputs = form.querySelectorAll('input, select, textarea');
            inputs.forEach(input => {
                // Ensure all inputs have labels
                const label = form.querySelector(`label[for="${input.id}"]`) ||
                            input.closest('.form-group')?.querySelector('label') ||
                            input.previousElementSibling?.tagName === 'LABEL' ? input.previousElementSibling : null;
                
                if (!label && !input.getAttribute('aria-label')) {
                    const placeholder = input.getAttribute('placeholder');
                    if (placeholder) {
                        input.setAttribute('aria-label', placeholder);
                    }
                }
                
                // Add required field indicators
                if (input.hasAttribute('required')) {
                    input.setAttribute('aria-required', 'true');
                    
                    // Add visual indicator
                    if (label && !label.querySelector('.required-indicator')) {
                        const indicator = document.createElement('span');
                        indicator.textContent = ' *';
                        indicator.className = 'required-indicator';
                        indicator.setAttribute('aria-label', 'required');
                        label.appendChild(indicator);
                    }
                }
                
                // Add error state support
                if (input.validity && !input.validity.valid) {
                    input.setAttribute('aria-invalid', 'true');
                }
            });
            
            // Enhance submit buttons
            const submitButtons = form.querySelectorAll('button[type="submit"], input[type="submit"]');
            submitButtons.forEach(button => {
                if (!button.getAttribute('aria-describedby')) {
                    button.setAttribute('aria-describedby', `${form.id || 'form'}-description`);
                }
            });
        });
    }
    
    /**
     * Enhance keyboard navigation
     */
    enhanceKeyboardNavigation() {
        if (!this.config.keyboardNavigation) return;
        
        // Add keyboard event listeners
        document.addEventListener('keydown', (e) => this.handleKeyboardNavigation(e));
        
        // Ensure all interactive elements are focusable
        const interactiveElements = document.querySelectorAll(`
            button, a, input, select, textarea, 
            [role="button"], [role="link"], [role="menuitem"],
            [tabindex]:not([tabindex="-1"]), .btn, .interactive
        `);
        
        interactiveElements.forEach(element => {
            if (!element.hasAttribute('tabindex') && 
                !['A', 'BUTTON', 'INPUT', 'SELECT', 'TEXTAREA'].includes(element.tagName)) {
                element.setAttribute('tabindex', '0');
            }
            
            // Ensure minimum touch target size
            const computedStyle = window.getComputedStyle(element);
            const minSize = 44; // 44px minimum as per WCAG
            
            if (parseFloat(computedStyle.height) < minSize || 
                parseFloat(computedStyle.width) < minSize) {
                element.style.minHeight = `${minSize}px`;
                element.style.minWidth = `${minSize}px`;
            }
        });
    }
    
    /**
     * Handle keyboard navigation events
     */
    handleKeyboardNavigation(event) {
        switch (event.key) {
            case 'Escape':
                this.handleEscape(event);
                break;
            case 'Tab':
                this.handleTab(event);
                break;
            case 'Enter':
            case ' ':
                this.handleActivation(event);
                break;
            case 'ArrowUp':
            case 'ArrowDown':
            case 'ArrowLeft':
            case 'ArrowRight':
                this.handleArrowKeys(event);
                break;
        }
    }
    
    /**
     * Add comprehensive ARIA labels
     */
    addARIALabels() {
        // Interactive Unity elements
        const unityElements = document.querySelectorAll('.unity-calculator, .consciousness-field-viz, .proof-viewer');
        unityElements.forEach(element => {
            element.setAttribute('role', 'application');
            element.setAttribute('aria-label', 'Unity Mathematics Interactive Component');
        });
        
        // Mathematical expressions
        const mathElements = document.querySelectorAll('.math, [class*="equation"], [id*="result"]');
        mathElements.forEach(element => {
            if (!element.getAttribute('aria-label')) {
                const mathText = this.extractMathText(element);
                if (mathText) {
                    element.setAttribute('aria-label', `Mathematical expression: ${mathText}`);
                }
            }
        });
        
        // Visualizations and charts
        const vizElements = document.querySelectorAll('canvas, svg, .visualization, .chart');
        vizElements.forEach((element, index) => {
            if (!element.getAttribute('aria-label')) {
                element.setAttribute('aria-label', `Visualization ${index + 1}: Unity Mathematics demonstration`);
            }
            element.setAttribute('role', 'img');
        });
        
        // Status indicators
        const statusElements = document.querySelectorAll('.status, .indicator, [id*="status"]');
        statusElements.forEach(element => {
            element.setAttribute('aria-live', 'polite');
            element.setAttribute('aria-atomic', 'true');
        });
    }
    
    /**
     * Improve color contrast for accessibility
     */
    improveColorContrast() {
        if (!this.config.colorContrastCheck) return;
        
        // Add high contrast mode detection
        const prefersHighContrast = window.matchMedia('(prefers-contrast: high)').matches;
        
        if (prefersHighContrast) {
            document.documentElement.setAttribute('data-high-contrast', 'true');
        }
        
        // Check and fix low contrast elements
        const textElements = document.querySelectorAll('p, span, div, a, button, label');
        textElements.forEach(element => {
            const style = window.getComputedStyle(element);
            const contrastRatio = this.calculateContrast(style.color, style.backgroundColor);
            
            if (contrastRatio < 4.5) {
                element.classList.add('low-contrast');
                element.setAttribute('data-contrast-ratio', contrastRatio.toFixed(2));
            }
        });
    }
    
    /**
     * Add focus management for interactive components
     */
    addFocusManagement() {
        if (!this.config.focusManagement) return;
        
        // Track focus changes
        document.addEventListener('focusin', (e) => {
            this.focusHistory.push(e.target);
            this.currentFocusIndex = this.focusHistory.length - 1;
            
            // Announce focus changes for complex widgets
            if (e.target.hasAttribute('role') || e.target.classList.contains('interactive')) {
                const announcement = this.getFocusAnnouncement(e.target);
                if (announcement) {
                    this.announceToScreenReader(announcement);
                }
            }
        });
        
        // Restore focus for dynamic content
        const observer = new MutationObserver((mutations) => {
            mutations.forEach(mutation => {
                if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                    // If focused element was removed, restore focus
                    const activeElement = document.activeElement;
                    if (activeElement === document.body || !document.contains(activeElement)) {
                        this.restoreFocus();
                    }
                }
            });
        });
        
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }
    
    /**
     * Enhance interactive elements
     */
    enhanceInteractiveElements() {
        // Enhance buttons
        const buttons = document.querySelectorAll('button, .btn, [role="button"]');
        buttons.forEach(button => {
            // Add loading state support
            button.addEventListener('click', () => {
                if (button.classList.contains('loading')) {
                    button.setAttribute('aria-busy', 'true');
                    button.setAttribute('aria-label', 'Loading...');
                } else {
                    button.removeAttribute('aria-busy');
                }
            });
            
            // Add pressed state for toggle buttons
            if (button.hasAttribute('aria-pressed')) {
                button.addEventListener('click', () => {
                    const pressed = button.getAttribute('aria-pressed') === 'true';
                    button.setAttribute('aria-pressed', !pressed);
                });
            }
        });
        
        // Enhance dropdowns and selects
        const dropdowns = document.querySelectorAll('select, .dropdown, [role="combobox"]');
        dropdowns.forEach(dropdown => {
            dropdown.setAttribute('aria-haspopup', 'listbox');
            
            if (dropdown.tagName !== 'SELECT') {
                dropdown.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        // Toggle dropdown
                        const expanded = dropdown.getAttribute('aria-expanded') === 'true';
                        dropdown.setAttribute('aria-expanded', !expanded);
                    }
                });
            }
        });
    }
    
    /**
     * Add accessibility for mathematical expressions
     */
    addMathAccessibility() {
        // MathJax accessibility
        if (window.MathJax) {
            window.MathJax.config = window.MathJax.config || {};
            window.MathJax.config.options = window.MathJax.config.options || {};
            window.MathJax.config.options.enableAssistiveMml = true;
            window.MathJax.config.options.enableExplorer = true;
        }
        
        // Manual math expressions
        const mathExpressions = document.querySelectorAll('.math-expression, .equation, [class*="unity-result"]');
        mathExpressions.forEach(expr => {
            const mathText = this.extractMathText(expr);
            if (mathText) {
                expr.setAttribute('aria-label', `Mathematical expression: ${mathText}`);
                expr.setAttribute('role', 'math');
            }
        });
    }
    
    /**
     * Setup error handling and reporting
     */
    setupErrorHandling() {
        window.addEventListener('error', (e) => {
            console.error('Accessibility Error:', e.error);
            this.announceToScreenReader('An error occurred. Please try again or contact support.');
        });
        
        // Monitor for accessibility violations
        if (typeof axe !== 'undefined') {
            axe.run(document, (err, results) => {
                if (err) {
                    console.error('Accessibility audit error:', err);
                    return;
                }
                
                if (results.violations.length > 0) {
                    console.warn('Accessibility violations found:', results.violations);
                }
            });
        }
    }
    
    /**
     * Announce messages to screen readers
     */
    announceToScreenReader(message, priority = 'polite') {
        if (!this.config.announceChanges || !this.announcer) return;
        
        this.announcer.setAttribute('aria-live', priority);
        this.announcer.textContent = message;
        
        // Clear after announcement
        setTimeout(() => {
            this.announcer.textContent = '';
        }, 1000);
    }
    
    /**
     * Extract readable text from mathematical expressions
     */
    extractMathText(element) {
        const text = element.textContent || element.innerText || '';
        
        // Convert common mathematical symbols to words
        return text
            .replace(/\+/g, ' plus ')
            .replace(/-/g, ' minus ')
            .replace(/\*/g, ' times ')
            .replace(/\//g, ' divided by ')
            .replace(/=/g, ' equals ')
            .replace(/≈/g, ' approximately equals ')
            .replace(/φ/g, ' phi ')
            .replace(/π/g, ' pi ')
            .replace(/∞/g, ' infinity ')
            .replace(/⊕/g, ' unity addition ')
            .replace(/∪/g, ' union ')
            .replace(/∩/g, ' intersection ')
            .trim();
    }
    
    /**
     * Calculate color contrast ratio
     */
    calculateContrast(color1, color2) {
        // Simplified contrast calculation
        // In a real implementation, you'd want a more robust color parsing
        const rgb1 = this.parseColor(color1);
        const rgb2 = this.parseColor(color2);
        
        const l1 = this.getLuminance(rgb1);
        const l2 = this.getLuminance(rgb2);
        
        const brightest = Math.max(l1, l2);
        const darkest = Math.min(l1, l2);
        
        return (brightest + 0.05) / (darkest + 0.05);
    }
    
    /**
     * Parse color string to RGB values
     */
    parseColor(color) {
        // Simplified color parser - extend as needed
        const div = document.createElement('div');
        div.style.color = color;
        document.body.appendChild(div);
        const computed = window.getComputedStyle(div).color;
        document.body.removeChild(div);
        
        const match = computed.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
        if (match) {
            return [parseInt(match[1]), parseInt(match[2]), parseInt(match[3])];
        }
        return [0, 0, 0];
    }
    
    /**
     * Calculate relative luminance
     */
    getLuminance([r, g, b]) {
        const [rs, gs, bs] = [r, g, b].map(c => {
            c = c / 255;
            return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
        });
        return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs;
    }
    
    /**
     * Get focus announcement for screen readers
     */
    getFocusAnnouncement(element) {
        const role = element.getAttribute('role');
        const label = element.getAttribute('aria-label') || element.textContent?.trim();
        
        if (role && label) {
            return `${role}: ${label}`;
        }
        return null;
    }
    
    /**
     * Restore focus to last focused element
     */
    restoreFocus() {
        if (this.focusHistory.length > 0) {
            for (let i = this.focusHistory.length - 1; i >= 0; i--) {
                const element = this.focusHistory[i];
                if (document.contains(element) && element.offsetParent !== null) {
                    element.focus();
                    this.currentFocusIndex = i;
                    return;
                }
            }
        }
        
        // Fallback to first focusable element
        const firstFocusable = document.querySelector('button, a, input, select, textarea, [tabindex]:not([tabindex="-1"])');
        if (firstFocusable) {
            firstFocusable.focus();
        }
    }
    
    /**
     * Handle escape key press
     */
    handleEscape(event) {
        // Close modals, dropdowns, etc.
        const openModal = document.querySelector('[role="dialog"][aria-hidden="false"]');
        if (openModal) {
            openModal.setAttribute('aria-hidden', 'true');
            this.restoreFocus();
        }
        
        const expandedElement = document.querySelector('[aria-expanded="true"]');
        if (expandedElement) {
            expandedElement.setAttribute('aria-expanded', 'false');
        }
    }
    
    /**
     * Handle tab key navigation
     */
    handleTab(event) {
        // Trap focus in modals
        const modal = event.target.closest('[role="dialog"]');
        if (modal) {
            const focusableElements = modal.querySelectorAll(`
                button, a, input, select, textarea, 
                [tabindex]:not([tabindex="-1"])
            `);
            
            if (focusableElements.length > 0) {
                const firstElement = focusableElements[0];
                const lastElement = focusableElements[focusableElements.length - 1];
                
                if (event.shiftKey && event.target === firstElement) {
                    event.preventDefault();
                    lastElement.focus();
                } else if (!event.shiftKey && event.target === lastElement) {
                    event.preventDefault();
                    firstElement.focus();
                }
            }
        }
    }
    
    /**
     * Handle activation keys (Enter, Space)
     */
    handleActivation(event) {
        const element = event.target;
        
        // Activate custom buttons
        if (element.getAttribute('role') === 'button' && element.tagName !== 'BUTTON') {
            event.preventDefault();
            element.click();
        }
    }
    
    /**
     * Handle arrow key navigation
     */
    handleArrowKeys(event) {
        const element = event.target;
        const role = element.getAttribute('role');
        
        // Handle menubar navigation
        if (role === 'menuitem') {
            const menubar = element.closest('[role="menubar"]');
            if (menubar) {
                const items = menubar.querySelectorAll('[role="menuitem"]');
                const currentIndex = Array.from(items).indexOf(element);
                
                let nextIndex;
                if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
                    nextIndex = (currentIndex + 1) % items.length;
                } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
                    nextIndex = (currentIndex - 1 + items.length) % items.length;
                }
                
                if (nextIndex !== undefined) {
                    event.preventDefault();
                    items[nextIndex].focus();
                }
            }
        }
    }
}

// Initialize accessibility enhancements
const accessibilityEnhancer = new AccessibilityEnhancer();

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AccessibilityEnhancer;
} else {
    window.AccessibilityEnhancer = AccessibilityEnhancer;
}