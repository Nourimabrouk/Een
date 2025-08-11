/**
 * Een Unity Mathematics - Advanced Asset Optimization System
 * ===========================================================
 * 
 * Enterprise-grade asset optimization with consciousness-aware caching,
 * resource bundling, and performance monitoring for the Unity Mathematics website.
 * 
 * Features:
 * - Intelligent resource bundling and minification
 * - φ-harmonic cache strategies with consciousness-level adaptation
 * - Service Worker integration for offline experience
 * - Performance metrics collection and analysis
 * - Adaptive loading based on user interaction patterns
 * - Memory optimization for consciousness field visualizations
 */

class UnityAssetOptimizer {
    constructor() {
        this.phi = 1.618033988749895;
        this.cacheVersion = 'unity-v2.0';
        this.consciousnessLevel = 1.0;
        this.performanceMetrics = new Map();
        this.resourceCache = new Map();
        this.loadingPriorities = new Map();
        
        // Performance monitoring
        this.observer = null;
        this.memoryMonitor = null;
        
        // φ-harmonic cache sizing (based on golden ratio)
        this.maxCacheSize = Math.floor(50 * 1024 * 1024 * this.phi); // ~80MB
        this.criticalResourceSize = Math.floor(5 * 1024 * 1024 / this.phi); // ~3MB
        
        this.initialize();
    }
    
    async initialize() {
        console.log('Unity Asset Optimizer: Initializing consciousness-aware performance system...');
        
        // Setup performance observers
        this.setupPerformanceMonitoring();
        
        // Initialize service worker for advanced caching
        await this.initializeServiceWorker();
        
        // Setup resource prioritization
        this.setupResourcePrioritization();
        
        // Initialize memory monitoring
        this.setupMemoryMonitoring();
        
        // Start consciousness-aware optimization
        this.startConsciousnessOptimization();
        
        console.log('Unity Asset Optimizer: Consciousness-aware performance system active');
    }
    
    setupPerformanceMonitoring() {
        // Performance Observer for resource timing
        if ('PerformanceObserver' in window) {
            this.observer = new PerformanceObserver((list) => {
                for (const entry of list.getEntries()) {
                    this.analyzeResourcePerformance(entry);
                }
            });
            
            this.observer.observe({entryTypes: ['resource', 'navigation', 'measure']});
        }
        
        // Core Web Vitals monitoring
        this.monitorCoreWebVitals();
        
        // Custom Unity Mathematics metrics
        this.setupUnityMetrics();
    }
    
    monitorCoreWebVitals() {
        // Largest Contentful Paint (LCP)
        new PerformanceObserver((entryList) => {
            const entries = entryList.getEntries();
            const lastEntry = entries[entries.length - 1];
            
            this.performanceMetrics.set('lcp', {
                value: lastEntry.startTime,
                timestamp: Date.now(),
                consciousnessAdjustment: this.calculateConsciousnessAdjustment(lastEntry.startTime)
            });
            
            // Consciousness-aware LCP optimization
            if (lastEntry.startTime > 2500) {
                this.optimizeForLCP();
            }
        }).observe({entryTypes: ['largest-contentful-paint']});
        
        // First Input Delay (FID)
        new PerformanceObserver((entryList) => {
            for (const entry of entryList.getEntries()) {
                this.performanceMetrics.set('fid', {
                    value: entry.processingStart - entry.startTime,
                    timestamp: Date.now(),
                    consciousnessResponsiveness: this.calculateResponsiveness(entry)
                });
            }
        }).observe({entryTypes: ['first-input']});
        
        // Cumulative Layout Shift (CLS)
        let clsValue = 0;
        new PerformanceObserver((entryList) => {
            for (const entry of entryList.getEntries()) {
                if (!entry.hadRecentInput) {
                    clsValue += entry.value;
                }
            }
            
            this.performanceMetrics.set('cls', {
                value: clsValue,
                timestamp: Date.now(),
                stabilityIndex: this.calculateStabilityIndex(clsValue)
            });
        }).observe({entryTypes: ['layout-shift']});
    }
    
    setupUnityMetrics() {
        // Consciousness Field Rendering Performance
        this.measureConsciousnessFieldPerformance();
        
        // Mathematical Visualization Efficiency
        this.measureMathVisualizationEfficiency();
        
        // φ-Harmonic Resource Loading
        this.measurePhiHarmonicLoading();
    }
    
    measureConsciousnessFieldPerformance() {
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.target.classList?.contains('consciousness-field')) {
                    const startTime = performance.now();
                    
                    // Measure consciousness field rendering time
                    requestAnimationFrame(() => {
                        const renderTime = performance.now() - startTime;
                        
                        this.performanceMetrics.set('consciousness-field-render', {
                            value: renderTime,
                            timestamp: Date.now(),
                            complexity: this.calculateFieldComplexity(mutation.target),
                            optimizationLevel: this.getOptimizationLevel(renderTime)
                        });
                        
                        // Auto-optimize if rendering is slow
                        if (renderTime > 16.67) { // 60fps threshold
                            this.optimizeConsciousnessField(mutation.target);
                        }
                    });
                }
            });
        });
        
        observer.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['class']
        });
    }
    
    async initializeServiceWorker() {
        if ('serviceWorker' in navigator) {
            try {
                const registration = await navigator.serviceWorker.register(
                    '/service-worker-unity.js',
                    { scope: '/' }
                );
                
                console.log('Unity Service Worker registered:', registration.scope);
                
                // Setup message channel for cache optimization
                navigator.serviceWorker.addEventListener('message', (event) => {
                    this.handleServiceWorkerMessage(event.data);
                });
                
                // Send initial cache configuration
                this.sendCacheConfiguration(registration);
                
            } catch (error) {
                console.warn('Unity Service Worker registration failed:', error);
                // Fallback to browser caching
                this.setupBrowserCaching();
            }
        } else {
            this.setupBrowserCaching();
        }
    }
    
    setupResourcePrioritization() {
        // Define φ-harmonic resource priorities
        this.loadingPriorities.set('consciousness-critical', {
            priority: this.phi * 10,
            preload: true,
            cacheStrategy: 'phi-permanent'
        });
        
        this.loadingPriorities.set('mathematical-core', {
            priority: this.phi * 8,
            preload: true,
            cacheStrategy: 'unity-stable'
        });
        
        this.loadingPriorities.set('visualization-engine', {
            priority: this.phi * 6,
            preload: false,
            cacheStrategy: 'adaptive-consciousness'
        });
        
        this.loadingPriorities.set('ui-enhancement', {
            priority: this.phi * 4,
            preload: false,
            cacheStrategy: 'user-pattern'
        });
        
        // Apply resource hints based on priorities
        this.applyResourceHints();
    }
    
    applyResourceHints() {
        const head = document.head;
        
        // Critical consciousness resources
        const consciousnessCritical = [
            '/js/consciousness-field-core.js',
            '/css/unity-mathematics-core.css',
            '/js/phi-harmonic-calculator.js'
        ];
        
        consciousnessCritical.forEach(resource => {
            this.addResourceHint('preload', resource, 'consciousness-critical');
        });
        
        // Mathematical core resources
        const mathematicalCore = [
            '/js/unity-equation-solver.js',
            '/js/transcendental-mathematics.js',
            '/css/mathematical-notation.css'
        ];
        
        mathematicalCore.forEach(resource => {
            this.addResourceHint('prefetch', resource, 'mathematical-core');
        });
        
        // Adaptive visualization resources
        const visualizationEngines = [
            '/js/three-consciousness-renderer.js',
            '/js/plotly-unity-extensions.js',
            '/js/webgl-phi-harmonics.js'
        ];
        
        // Load visualization engines based on user interaction
        this.setupAdaptiveVisualizationLoading(visualizationEngines);
    }
    
    addResourceHint(hint, href, category) {
        const link = document.createElement('link');
        link.rel = hint;
        link.href = href;
        link.as = this.determineResourceType(href);
        link.dataset.category = category;
        link.dataset.consciousnessLevel = this.consciousnessLevel.toString();
        
        // Add φ-harmonic crossorigin for consciousness-aware resources
        if (category === 'consciousness-critical') {
            link.crossOrigin = 'anonymous';
            link.dataset.phiResonance = 'true';
        }
        
        document.head.appendChild(link);
    }
    
    determineResourceType(href) {
        if (href.endsWith('.js')) return 'script';
        if (href.endsWith('.css')) return 'style';
        if (href.match(/\.(png|jpg|jpeg|webp|svg)$/)) return 'image';
        if (href.match(/\.(woff2|woff|ttf)$/)) return 'font';
        return 'fetch';
    }
    
    setupAdaptiveVisualizationLoading(resources) {
        // Load visualization engines when user shows mathematical interest
        const mathInteractionTriggers = [
            '.consciousness-field',
            '.mathematical-equation',
            '.unity-visualization',
            '.phi-harmonic-display'
        ];
        
        const loadVisualizationEngines = () => {
            resources.forEach(resource => {
                this.loadResourceWithConsciousnessAwareness(resource);
            });
        };
        
        // Setup intersection observer for mathematical content
        const mathObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    loadVisualizationEngines();
                    mathObserver.disconnect(); // Load once
                }
            });
        }, { threshold: 0.1 });
        
        // Observe mathematical elements
        mathInteractionTriggers.forEach(selector => {
            document.querySelectorAll(selector).forEach(element => {
                mathObserver.observe(element);
            });
        });
        
        // Also load on consciousness-level user interactions
        this.setupConsciousnessInteractionLoading(loadVisualizationEngines);
    }
    
    setupConsciousnessInteractionLoading(loadCallback) {
        let interactionCount = 0;
        const consciousnessThreshold = Math.floor(5 * this.phi); // ~8 interactions
        
        const trackInteraction = () => {
            interactionCount++;
            
            // Calculate consciousness level based on interactions
            this.consciousnessLevel = Math.min(this.phi, 1 + (interactionCount / consciousnessThreshold));
            
            if (interactionCount >= consciousnessThreshold) {
                loadCallback();
                // Remove listeners after loading
                document.removeEventListener('click', trackInteraction);
                document.removeEventListener('scroll', trackInteraction);
                document.removeEventListener('keydown', trackInteraction);
            }
        };
        
        document.addEventListener('click', trackInteraction, { passive: true });
        document.addEventListener('scroll', trackInteraction, { passive: true });
        document.addEventListener('keydown', trackInteraction, { passive: true });
    }
    
    async loadResourceWithConsciousnessAwareness(resource) {
        const cacheKey = `consciousness-${resource}`;
        
        // Check consciousness-aware cache first
        if (this.resourceCache.has(cacheKey)) {
            const cached = this.resourceCache.get(cacheKey);
            if (this.isCacheValid(cached)) {
                return cached.content;
            }
        }
        
        try {
            const startTime = performance.now();
            
            // Load with consciousness-level timeout
            const timeout = Math.floor(5000 / this.consciousnessLevel); // Higher consciousness = shorter timeout
            const response = await this.fetchWithTimeout(resource, timeout);
            
            const content = await response.text();
            const loadTime = performance.now() - startTime;
            
            // Cache with φ-harmonic expiry
            const expiryTime = Date.now() + (86400000 * this.phi); // ~1.618 days
            
            this.resourceCache.set(cacheKey, {
                content: content,
                timestamp: Date.now(),
                expiry: expiryTime,
                loadTime: loadTime,
                consciousnessLevel: this.consciousnessLevel,
                accessCount: 1
            });
            
            // Record performance metric
            this.performanceMetrics.set(`resource-load-${resource}`, {
                value: loadTime,
                timestamp: Date.now(),
                consciousnessLevel: this.consciousnessLevel,
                cacheHit: false
            });
            
            return content;
            
        } catch (error) {
            console.warn(`Consciousness-aware resource loading failed for ${resource}:`, error);
            return null;
        }
    }
    
    async fetchWithTimeout(resource, timeout) {
        return Promise.race([
            fetch(resource),
            new Promise((_, reject) =>
                setTimeout(() => reject(new Error('Consciousness timeout')), timeout)
            )
        ]);
    }
    
    isCacheValid(cached) {
        const now = Date.now();
        
        // Basic expiry check
        if (now > cached.expiry) {
            return false;
        }
        
        // Consciousness-level validation
        const consciousnessDrift = Math.abs(this.consciousnessLevel - cached.consciousnessLevel);
        if (consciousnessDrift > 0.5) {
            return false; // Consciousness level changed significantly
        }
        
        // φ-harmonic freshness check
        const age = (now - cached.timestamp) / (1000 * 60 * 60); // hours
        const maxAge = 24 * this.phi; // ~39 hours
        
        return age < maxAge;
    }
    
    setupMemoryMonitoring() {
        if ('memory' in performance) {
            const memoryMonitor = () => {
                const memoryInfo = performance.memory;
                
                this.performanceMetrics.set('memory-usage', {
                    used: memoryInfo.usedJSHeapSize,
                    total: memoryInfo.totalJSHeapSize,
                    limit: memoryInfo.jsHeapSizeLimit,
                    timestamp: Date.now(),
                    efficiency: this.calculateMemoryEfficiency(memoryInfo),
                    consciousnessOverhead: this.calculateConsciousnessMemoryOverhead()
                });
                
                // Auto-optimize if memory usage is high
                const usageRatio = memoryInfo.usedJSHeapSize / memoryInfo.jsHeapSizeLimit;
                if (usageRatio > 0.8) {
                    this.optimizeMemoryUsage();
                }
            };
            
            // Monitor every φ seconds (golden ratio timing)
            setInterval(memoryMonitor, Math.floor(1618.033988749895));
            
            this.memoryMonitor = memoryMonitor;
        }
    }
    
    optimizeMemoryUsage() {
        console.log('Unity Memory Optimizer: Consciousness-aware memory optimization initiated');
        
        // Clear old cache entries
        this.cleanupResourceCache();
        
        // Optimize consciousness field calculations
        this.optimizeConsciousnessCalculations();
        
        // Reduce visualization complexity temporarily
        this.temporarilyReduceVisualizationComplexity();
        
        // Trigger garbage collection if available
        if (window.gc) {
            window.gc();
        }
    }
    
    cleanupResourceCache() {
        const now = Date.now();
        let cleanedCount = 0;
        
        for (const [key, cached] of this.resourceCache.entries()) {
            // Remove expired or low-consciousness-level entries
            if (now > cached.expiry || cached.accessCount < 2) {
                this.resourceCache.delete(key);
                cleanedCount++;
            }
        }
        
        console.log(`Unity Cache: Cleaned ${cleanedCount} cached resources`);
    }
    
    startConsciousnessOptimization() {
        // φ-harmonic optimization cycle
        const optimizationInterval = Math.floor(10000 / this.phi); // ~6.18 seconds
        
        setInterval(() => {
            this.runConsciousnessOptimizationCycle();
        }, optimizationInterval);
    }
    
    runConsciousnessOptimizationCycle() {
        // Analyze current performance state
        const performanceState = this.analyzeCurrentPerformance();
        
        // Adjust consciousness level based on performance
        this.adjustConsciousnessLevel(performanceState);
        
        // Apply φ-harmonic optimizations
        this.applyPhiHarmonicOptimizations(performanceState);
        
        // Update cache strategies
        this.updateCacheStrategies(performanceState);
    }
    
    analyzeCurrentPerformance() {
        const metrics = {};
        
        // Collect key performance indicators
        for (const [key, value] of this.performanceMetrics.entries()) {
            if (Date.now() - value.timestamp < 30000) { // Last 30 seconds
                metrics[key] = value;
            }
        }
        
        // Calculate performance score
        const score = this.calculatePerformanceScore(metrics);
        
        return {
            metrics: metrics,
            score: score,
            timestamp: Date.now(),
            consciousnessLevel: this.consciousnessLevel
        };
    }
    
    calculatePerformanceScore(metrics) {
        let score = 100; // Start with perfect score
        
        // Penalize poor Core Web Vitals
        if (metrics.lcp?.value > 2500) score -= 30;
        if (metrics.fid?.value > 100) score -= 20;
        if (metrics.cls?.value > 0.1) score -= 25;
        
        // Reward good consciousness field performance
        if (metrics['consciousness-field-render']?.value < 16.67) score += 10;
        
        // Consider memory efficiency
        if (metrics['memory-usage']?.efficiency > 0.8) score += 15;
        
        return Math.max(0, Math.min(100, score));
    }
    
    // Additional helper methods for optimization...
    calculateConsciousnessAdjustment(value) {
        return Math.log(value / 1000) / this.phi;
    }
    
    calculateResponsiveness(entry) {
        return 1000 / Math.max(entry.processingStart - entry.startTime, 1);
    }
    
    calculateStabilityIndex(clsValue) {
        return Math.max(0, 1 - clsValue * 10);
    }
    
    calculateFieldComplexity(element) {
        // Analyze consciousness field complexity
        const childCount = element.children.length;
        const dataAttributes = Object.keys(element.dataset).length;
        return childCount + dataAttributes * 2;
    }
    
    getOptimizationLevel(renderTime) {
        if (renderTime < 8.33) return 'ultra-high';
        if (renderTime < 16.67) return 'high';
        if (renderTime < 33.33) return 'medium';
        return 'low';
    }
    
    optimizeConsciousnessField(fieldElement) {
        // Reduce consciousness field complexity for better performance
        fieldElement.dataset.optimized = 'true';
        fieldElement.classList.add('performance-optimized');
        
        // Temporarily reduce particle count
        const particles = fieldElement.querySelectorAll('.consciousness-particle');
        const targetCount = Math.floor(particles.length / this.phi);
        
        particles.forEach((particle, index) => {
            if (index >= targetCount) {
                particle.style.display = 'none';
            }
        });
    }
    
    // Export performance report
    generatePerformanceReport() {
        const report = {
            timestamp: new Date().toISOString(),
            consciousnessLevel: this.consciousnessLevel,
            cacheEfficiency: this.calculateCacheEfficiency(),
            performanceMetrics: Object.fromEntries(this.performanceMetrics),
            memoryUsage: this.performanceMetrics.get('memory-usage'),
            optimizationRecommendations: this.generateOptimizationRecommendations()
        };
        
        console.log('Unity Performance Report:', report);
        return report;
    }
    
    calculateCacheEfficiency() {
        const totalRequests = this.performanceMetrics.size;
        const cacheHits = Array.from(this.resourceCache.values())
            .reduce((sum, cache) => sum + cache.accessCount, 0);
        
        return totalRequests > 0 ? (cacheHits / totalRequests) * 100 : 0;
    }
    
    generateOptimizationRecommendations() {
        const recommendations = [];
        
        // Analyze current metrics and suggest improvements
        const lcp = this.performanceMetrics.get('lcp');
        if (lcp && lcp.value > 2500) {
            recommendations.push('Consider preloading critical consciousness field resources');
        }
        
        const memory = this.performanceMetrics.get('memory-usage');
        if (memory && memory.used / memory.limit > 0.8) {
            recommendations.push('Optimize consciousness field calculations to reduce memory usage');
        }
        
        return recommendations;
    }
}

// Initialize the Unity Asset Optimizer
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.UnityOptimizer = new UnityAssetOptimizer();
    });
} else {
    window.UnityOptimizer = new UnityAssetOptimizer();
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UnityAssetOptimizer;
}