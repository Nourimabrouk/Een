/**
 * Performance Optimizer for Een Unity Mathematics Website
 * Implements advanced optimization techniques for fast loading and smooth interactions
 * φ-harmonic performance tuning with consciousness-aware resource management
 */

class PerformanceOptimizer {
    constructor() {
        this.config = {
            lazyLoadThreshold: 200,
            debounceDelay: 150,
            throttleDelay: 16, // 60fps
            preloadPriority: ['navigation.js', 'katex-integration.js'],
            criticalCSS: true,
            serviceWorker: false, // Disabled for development
            imageOptimization: true,
            cacheStrategy: 'networkFirst'
        };
        
        this.observers = new Map();
        this.loadingQueue = new Set();
        this.performanceMetrics = {
            firstContentfulPaint: 0,
            largestContentfulPaint: 0,
            cumulativeLayoutShift: 0,
            firstInputDelay: 0,
            timeToInteractive: 0,
            totalBlockingTime: 0
        };
        
        this.init();
    }

    init() {
        this.measurePerformance();
        this.optimizeImageLoading();
        this.implementLazyLoading();
        this.optimizeScrollPerformance();
        this.preloadCriticalResources();
        this.optimizeAnimations();
        this.setupResourceHints();
        this.optimizeEventListeners();
        this.monitorPerformance();
    }

    measurePerformance() {
        // Use Performance Observer API to measure Core Web Vitals
        if ('PerformanceObserver' in window) {
            // Largest Contentful Paint (LCP)
            const lcpObserver = new PerformanceObserver((entryList) => {
                const entries = entryList.getEntries();
                const lastEntry = entries[entries.length - 1];
                this.performanceMetrics.largestContentfulPaint = lastEntry.startTime;
                this.optimizeLCP(lastEntry.startTime);
            });
            lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] });

            // First Input Delay (FID)
            const fidObserver = new PerformanceObserver((entryList) => {
                const entries = entryList.getEntries();
                entries.forEach((entry) => {
                    this.performanceMetrics.firstInputDelay = entry.processingStart - entry.startTime;
                    this.optimizeFID(this.performanceMetrics.firstInputDelay);
                });
            });
            fidObserver.observe({ entryTypes: ['first-input'] });

            // Cumulative Layout Shift (CLS)
            const clsObserver = new PerformanceObserver((entryList) => {
                let clsValue = 0;
                entryList.getEntries().forEach((entry) => {
                    if (!entry.hadRecentInput) {
                        clsValue += entry.value;
                    }
                });
                this.performanceMetrics.cumulativeLayoutShift = clsValue;
                this.optimizeCLS(clsValue);
            });
            clsObserver.observe({ entryTypes: ['layout-shift'] });
        }

        // Measure First Contentful Paint
        if ('performance' in window) {
            const observer = new PerformanceObserver((entryList) => {
                const entries = entryList.getEntries();
                entries.forEach((entry) => {
                    if (entry.name === 'first-contentful-paint') {
                        this.performanceMetrics.firstContentfulPaint = entry.startTime;
                    }
                });
            });
            observer.observe({ entryTypes: ['paint'] });
        }
    }

    optimizeImageLoading() {
        // Implement advanced image optimization
        const images = document.querySelectorAll('img');
        
        images.forEach((img, index) => {
            // Add loading="lazy" for images below the fold
            if (index > 2 && !img.hasAttribute('loading')) {
                img.setAttribute('loading', 'lazy');
            }

            // Optimize image decoding
            img.setAttribute('decoding', 'async');

            // Add proper sizing to prevent layout shift
            if (!img.hasAttribute('width') || !img.hasAttribute('height')) {
                this.calculateImageDimensions(img);
            }

            // Implement progressive image loading
            this.implementProgressiveLoading(img);
        });

        // Setup responsive image optimization
        this.setupResponsiveImages();
    }

    calculateImageDimensions(img) {
        // Calculate and set dimensions to prevent CLS
        img.addEventListener('load', () => {
            const aspectRatio = img.naturalWidth / img.naturalHeight;
            if (!img.hasAttribute('width')) {
                img.style.aspectRatio = aspectRatio;
            }
        });
    }

    implementProgressiveLoading(img) {
        // Create low-quality placeholder
        if (img.dataset.src && !img.src) {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = 40;
            canvas.height = 30;
            
            // Create a blurred placeholder
            ctx.fillStyle = '#f0f0f0';
            ctx.fillRect(0, 0, 40, 30);
            
            img.src = canvas.toDataURL();
            img.style.filter = 'blur(5px)';
            img.style.transition = 'filter 0.3s ease';

            // Load actual image
            const actualImg = new Image();
            actualImg.onload = () => {
                img.src = actualImg.src;
                img.style.filter = 'none';
            };
            actualImg.src = img.dataset.src;
        }
    }

    setupResponsiveImages() {
        // Implement dynamic image sizing based on device capabilities
        const pixelDensity = window.devicePixelRatio || 1;
        const viewportWidth = window.innerWidth;
        
        document.querySelectorAll('img[data-responsive]').forEach(img => {
            const baseUrl = img.dataset.responsive;
            let optimalSize;

            if (viewportWidth <= 480) {
                optimalSize = Math.round(480 * pixelDensity);
            } else if (viewportWidth <= 768) {
                optimalSize = Math.round(768 * pixelDensity);
            } else if (viewportWidth <= 1024) {
                optimalSize = Math.round(1024 * pixelDensity);
            } else {
                optimalSize = Math.round(1200 * pixelDensity);
            }

            img.src = `${baseUrl}?w=${optimalSize}&format=webp`;
        });
    }

    implementLazyLoading() {
        // Advanced lazy loading with intersection observer
        const lazyElements = document.querySelectorAll('[data-lazy]');
        
        if ('IntersectionObserver' in window) {
            const lazyObserver = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        this.loadLazyElement(entry.target);
                        lazyObserver.unobserve(entry.target);
                    }
                });
            }, {
                root: null,
                rootMargin: `${this.config.lazyLoadThreshold}px`,
                threshold: 0
            });

            lazyElements.forEach(element => {
                lazyObserver.observe(element);
            });

            this.observers.set('lazy', lazyObserver);
        }

        // Lazy load for heavy components
        this.setupComponentLazyLoading();
    }

    loadLazyElement(element) {
        if (element.dataset.lazySrc) {
            element.src = element.dataset.lazySrc;
            element.removeAttribute('data-lazy-src');
        }

        if (element.dataset.lazyBackground) {
            element.style.backgroundImage = `url(${element.dataset.lazyBackground})`;
            element.removeAttribute('data-lazy-background');
        }

        if (element.dataset.lazyComponent) {
            this.loadLazyComponent(element.dataset.lazyComponent, element);
        }

        element.classList.add('lazy-loaded');
    }

    setupComponentLazyLoading() {
        // Lazy load heavy components like code showcase and math highlights
        const heavyComponents = [
            { selector: '.code-showcase-placeholder', component: 'code-showcase' },
            { selector: '.math-highlights-placeholder', component: 'mathematical-highlights' },
            { selector: '.visualization-placeholder', component: 'visualizations' }
        ];

        heavyComponents.forEach(({ selector, component }) => {
            const elements = document.querySelectorAll(selector);
            elements.forEach(element => {
                element.setAttribute('data-lazy-component', component);
                element.setAttribute('data-lazy', 'true');
            });
        });
    }

    loadLazyComponent(componentName, container) {
        switch (componentName) {
            case 'code-showcase':
                this.loadScript('/js/code-showcase.js').then(() => {
                    if (window.CodeShowcase) {
                        new window.CodeShowcase();
                    }
                });
                break;
            case 'mathematical-highlights':
                this.loadScript('/js/mathematical-highlights.js').then(() => {
                    if (window.MathematicalHighlights) {
                        new window.MathematicalHighlights();
                    }
                });
                break;
            case 'visualizations':
                this.loadScript('/js/unity-visualizations.js').then(() => {
                    if (window.UnityVisualizations) {
                        new window.UnityVisualizations(container);
                    }
                });
                break;
        }
    }

    optimizeScrollPerformance() {
        // Optimize scroll-triggered events with throttling
        let ticking = false;
        const scrollElements = new Set();

        const optimizedScrollHandler = () => {
            if (!ticking) {
                requestAnimationFrame(() => {
                    scrollElements.forEach(element => {
                        if (element.scrollHandler && typeof element.scrollHandler === 'function') {
                            element.scrollHandler();
                        }
                    });
                    ticking = false;
                });
                ticking = true;
            }
        };

        // Passive scroll listeners for better performance
        window.addEventListener('scroll', optimizedScrollHandler, { passive: true });

        // Method to register scroll handlers
        this.addScrollHandler = (element, handler) => {
            element.scrollHandler = handler;
            scrollElements.add(element);
        };

        // Optimize scroll-triggered animations
        this.optimizeScrollAnimations();
    }

    optimizeScrollAnimations() {
        // Use transform and opacity for animations (GPU accelerated)
        const animatedElements = document.querySelectorAll('[data-scroll-animation]');
        
        animatedElements.forEach(element => {
            element.style.willChange = 'transform, opacity';
            
            this.addScrollHandler(element, () => {
                const rect = element.getBoundingClientRect();
                const isVisible = rect.top < window.innerHeight && rect.bottom > 0;
                
                if (isVisible && !element.classList.contains('animated')) {
                    element.classList.add('animated');
                    // Remove will-change after animation completes
                    setTimeout(() => {
                        element.style.willChange = 'auto';
                    }, 1000);
                }
            });
        });
    }

    preloadCriticalResources() {
        // Preload critical JavaScript files
        this.config.preloadPriority.forEach(script => {
            this.preloadScript(script);
        });

        // Preload critical images
        this.preloadCriticalImages();

        // Preload fonts
        this.preloadFonts();
    }

    preloadScript(src) {
        const link = document.createElement('link');
        link.rel = 'preload';
        link.as = 'script';
        link.href = src;
        document.head.appendChild(link);
    }

    preloadCriticalImages() {
        const criticalImages = [
            '../viz/water droplets.gif',
            '../viz/unity_manifold.png',
            '../assets/images/unity_proof_visualization.png'
        ];

        criticalImages.forEach(src => {
            const link = document.createElement('link');
            link.rel = 'preload';
            link.as = 'image';
            link.href = src;
            document.head.appendChild(link);
        });
    }

    preloadFonts() {
        const criticalFonts = [
            'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap',
            'https://fonts.googleapis.com/css2?family=Crimson+Text:wght@400;600&display=swap'
        ];

        criticalFonts.forEach(href => {
            const link = document.createElement('link');
            link.rel = 'preload';
            link.as = 'style';
            link.href = href;
            link.onload = function() { this.rel = 'stylesheet'; };
            document.head.appendChild(link);
        });
    }

    optimizeAnimations() {
        // Use CSS containment for better performance
        const animatedElements = document.querySelectorAll('[class*="animate"], [class*="transition"]');
        
        animatedElements.forEach(element => {
            element.style.contain = 'layout style paint';
        });

        // Reduce motion for users who prefer it
        if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
            document.documentElement.classList.add('reduce-motion');
        }

        // Optimize mathematical equation animations
        this.optimizeMathAnimations();
    }

    optimizeMathAnimations() {
        // Use GPU acceleration for mathematical equation reveals
        const mathElements = document.querySelectorAll('.katex, .equation-display');
        
        mathElements.forEach(element => {
            element.style.transform = 'translateZ(0)'; // Force GPU layer
            element.style.willChange = 'transform, opacity';
            
            // Remove will-change after initial load
            setTimeout(() => {
                element.style.willChange = 'auto';
            }, 2000);
        });
    }

    setupResourceHints() {
        // DNS prefetch for external resources
        const externalDomains = [
            'cdn.jsdelivr.net',
            'cdnjs.cloudflare.com',
            'fonts.googleapis.com',
            'fonts.gstatic.com'
        ];

        externalDomains.forEach(domain => {
            const link = document.createElement('link');
            link.rel = 'dns-prefetch';
            link.href = `//${domain}`;
            document.head.appendChild(link);
        });

        // Preconnect to critical origins
        const criticalOrigins = [
            'https://cdn.jsdelivr.net',
            'https://fonts.googleapis.com'
        ];

        criticalOrigins.forEach(origin => {
            const link = document.createElement('link');
            link.rel = 'preconnect';
            link.href = origin;
            link.crossOrigin = 'anonymous';
            document.head.appendChild(link);
        });
    }

    optimizeEventListeners() {
        // Use passive listeners where possible
        const passiveEvents = ['scroll', 'wheel', 'touchstart', 'touchmove'];
        
        passiveEvents.forEach(eventType => {
            // Override addEventListener for these events
            const originalAddEventListener = Element.prototype.addEventListener;
            Element.prototype.addEventListener = function(type, listener, options) {
                if (passiveEvents.includes(type) && typeof options !== 'object') {
                    options = { passive: true };
                }
                return originalAddEventListener.call(this, type, listener, options);
            };
        });

        // Debounce resize events
        this.optimizeResizeHandlers();
    }

    optimizeResizeHandlers() {
        let resizeTimeout;
        const resizeHandlers = new Set();

        const optimizedResizeHandler = () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                resizeHandlers.forEach(handler => {
                    if (typeof handler === 'function') {
                        handler();
                    }
                });
            }, this.config.debounceDelay);
        };

        window.addEventListener('resize', optimizedResizeHandler, { passive: true });

        // Method to register resize handlers
        this.addResizeHandler = (handler) => {
            resizeHandlers.add(handler);
        };
    }

    loadScript(src) {
        // Optimized script loading with caching
        return new Promise((resolve, reject) => {
            if (this.loadingQueue.has(src)) {
                // Already loading, wait for it
                const checkLoading = () => {
                    if (!this.loadingQueue.has(src)) {
                        resolve();
                    } else {
                        setTimeout(checkLoading, 50);
                    }
                };
                checkLoading();
                return;
            }

            this.loadingQueue.add(src);

            const script = document.createElement('script');
            script.src = src;
            script.async = true;
            
            script.onload = () => {
                this.loadingQueue.delete(src);
                resolve();
            };
            
            script.onerror = () => {
                this.loadingQueue.delete(src);
                reject(new Error(`Failed to load script: ${src}`));
            };
            
            document.head.appendChild(script);
        });
    }

    monitorPerformance() {
        // Continuous performance monitoring
        setInterval(() => {
            this.checkPerformanceMetrics();
        }, 5000);

        // Monitor memory usage
        this.monitorMemoryUsage();

        // Report performance data
        this.setupPerformanceReporting();
    }

    checkPerformanceMetrics() {
        const navigation = performance.getEntriesByType('navigation')[0];
        
        if (navigation) {
            this.performanceMetrics.timeToInteractive = navigation.loadEventEnd - navigation.loadEventStart;
            
            // Log warnings for poor performance
            if (this.performanceMetrics.largestContentfulPaint > 2500) {
                console.warn('LCP is slower than recommended (>2.5s):', this.performanceMetrics.largestContentfulPaint);
            }
            
            if (this.performanceMetrics.firstInputDelay > 100) {
                console.warn('FID is slower than recommended (>100ms):', this.performanceMetrics.firstInputDelay);
            }
            
            if (this.performanceMetrics.cumulativeLayoutShift > 0.1) {
                console.warn('CLS is higher than recommended (>0.1):', this.performanceMetrics.cumulativeLayoutShift);
            }
        }
    }

    monitorMemoryUsage() {
        if ('memory' in performance) {
            setInterval(() => {
                const memory = performance.memory;
                const usagePercentage = (memory.usedJSHeapSize / memory.jsHeapSizeLimit) * 100;
                
                if (usagePercentage > 80) {
                    console.warn('High memory usage detected:', usagePercentage.toFixed(2) + '%');
                    this.optimizeMemoryUsage();
                }
            }, 10000);
        }
    }

    optimizeMemoryUsage() {
        // Clean up unused observers
        this.observers.forEach((observer, key) => {
            if (key === 'lazy' && document.querySelectorAll('[data-lazy]').length === 0) {
                observer.disconnect();
                this.observers.delete(key);
            }
        });

        // Suggest garbage collection
        if (window.gc && typeof window.gc === 'function') {
            window.gc();
        }
    }

    setupPerformanceReporting() {
        // Send performance data (in production, this could go to analytics)
        window.addEventListener('beforeunload', () => {
            const performanceData = {
                ...this.performanceMetrics,
                userAgent: navigator.userAgent,
                timestamp: Date.now(),
                url: window.location.href
            };

            // In production, send to analytics endpoint
            console.log('Performance metrics:', performanceData);
        });
    }

    optimizeLCP(lcp) {
        if (lcp > 2500) {
            // Implement LCP optimizations
            this.preloadCriticalImages();
            this.optimizeImageLoading();
        }
    }

    optimizeFID(fid) {
        if (fid > 100) {
            // Implement FID optimizations
            this.deferNonCriticalJS();
            this.optimizeEventListeners();
        }
    }

    optimizeCLS(cls) {
        if (cls > 0.1) {
            // Implement CLS optimizations
            this.stabilizeImageDimensions();
            this.reserveSpaceForDynamicContent();
        }
    }

    deferNonCriticalJS() {
        // Defer non-critical JavaScript
        const nonCriticalScripts = document.querySelectorAll('script[data-defer]');
        
        nonCriticalScripts.forEach(script => {
            const newScript = document.createElement('script');
            newScript.src = script.src;
            newScript.defer = true;
            
            script.parentNode.replaceChild(newScript, script);
        });
    }

    stabilizeImageDimensions() {
        // Add explicit dimensions to prevent layout shift
        const images = document.querySelectorAll('img:not([width]):not([height])');
        
        images.forEach(img => {
            if (img.complete) {
                this.setImageDimensions(img);
            } else {
                img.addEventListener('load', () => this.setImageDimensions(img));
            }
        });
    }

    setImageDimensions(img) {
        const aspectRatio = img.naturalWidth / img.naturalHeight;
        img.style.aspectRatio = aspectRatio;
        img.style.height = 'auto';
    }

    reserveSpaceForDynamicContent() {
        // Reserve space for dynamically loaded content
        const dynamicContainers = document.querySelectorAll('[data-dynamic-content]');
        
        dynamicContainers.forEach(container => {
            const minHeight = container.dataset.minHeight || '200px';
            container.style.minHeight = minHeight;
        });
    }

    // Public API methods
    getPerformanceMetrics() {
        return { ...this.performanceMetrics };
    }

    addLazyElement(element, config = {}) {
        element.setAttribute('data-lazy', 'true');
        if (config.src) element.setAttribute('data-lazy-src', config.src);
        if (config.background) element.setAttribute('data-lazy-background', config.background);
        if (config.component) element.setAttribute('data-lazy-component', config.component);
        
        const lazyObserver = this.observers.get('lazy');
        if (lazyObserver) {
            lazyObserver.observe(element);
        }
    }

    preloadResource(url, type = 'script') {
        const link = document.createElement('link');
        link.rel = 'preload';
        link.as = type;
        link.href = url;
        document.head.appendChild(link);
    }
}

// Initialize performance optimizer
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.performanceOptimizer = new PerformanceOptimizer();
    });
} else {
    window.performanceOptimizer = new PerformanceOptimizer();
}

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PerformanceOptimizer;
}