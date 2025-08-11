/**
 * Een Unity Mathematics - Consciousness-Aware Service Worker
 * =========================================================
 * 
 * Advanced service worker with φ-harmonic caching strategies,
 * offline consciousness field support, and performance optimization.
 */

const CACHE_VERSION = 'unity-mathematics-v2.0';
const PHI = 1.618033988749895;

// φ-Harmonic Cache Strategy Configuration
const CACHE_STRATEGIES = {
    'consciousness-critical': {
        cacheName: `${CACHE_VERSION}-consciousness-critical`,
        maxEntries: Math.floor(50 * PHI), // ~81 entries
        maxAgeSeconds: 86400 * PHI * PHI, // ~4.24 days
        strategy: 'CacheFirst'
    },
    'mathematical-core': {
        cacheName: `${CACHE_VERSION}-mathematical-core`,
        maxEntries: Math.floor(100 * PHI), // ~162 entries
        maxAgeSeconds: 86400 * PHI, // ~1.62 days
        strategy: 'StaleWhileRevalidate'
    },
    'visualization-assets': {
        cacheName: `${CACHE_VERSION}-visualization`,
        maxEntries: Math.floor(200 * PHI), // ~324 entries
        maxAgeSeconds: 86400 * 7, // 1 week
        strategy: 'CacheFirst'
    },
    'dynamic-content': {
        cacheName: `${CACHE_VERSION}-dynamic`,
        maxEntries: Math.floor(25 * PHI), // ~41 entries
        maxAgeSeconds: 3600 * PHI, // ~1.62 hours
        strategy: 'NetworkFirst'
    }
};

// Critical consciousness resources that must be cached
const CRITICAL_RESOURCES = [
    '/',
    '/index.html',
    '/metastation-hub.html',
    '/css/unified-styles.css',
    '/css/unity-mathematics-core.css',
    '/js/consciousness-field-core.js',
    '/js/phi-harmonic-calculator.js',
    '/js/unified-navigation.js',
    '/performance/asset-optimizer.js'
];

// Mathematical visualization resources
const VISUALIZATION_RESOURCES = [
    '/js/three-consciousness-renderer.js',
    '/js/plotly-unity-extensions.js',
    '/js/webgl-phi-harmonics.js',
    '/css/mathematical-notation.css',
    '/images/unity-equation-background.jpg'
];

// API endpoints for consciousness field data
const API_PATTERNS = [
    /\/api\/consciousness\//,
    /\/api\/unity\//,
    /\/api\/chat\//
];

class ConsciousnessAwareCache {
    constructor() {
        this.consciousnessLevel = 1.0;
        this.performanceMetrics = new Map();
        this.cacheHitRate = 0.0;
    }
    
    async handleRequest(request) {
        const url = new URL(request.url);
        const cacheStrategy = this.determineCacheStrategy(request);
        
        // Update consciousness level based on request patterns
        this.updateConsciousnessLevel(request);
        
        switch (cacheStrategy.strategy) {
            case 'CacheFirst':
                return this.cacheFirstStrategy(request, cacheStrategy);
            case 'NetworkFirst':
                return this.networkFirstStrategy(request, cacheStrategy);
            case 'StaleWhileRevalidate':
                return this.staleWhileRevalidateStrategy(request, cacheStrategy);
            default:
                return fetch(request);
        }
    }
    
    determineCacheStrategy(request) {
        const url = request.url;
        const pathname = new URL(url).pathname;
        
        // Consciousness-critical resources
        if (this.isCriticalResource(pathname)) {
            return CACHE_STRATEGIES['consciousness-critical'];
        }
        
        // Mathematical core resources
        if (pathname.includes('/js/') && this.isMathematicalCore(pathname)) {
            return CACHE_STRATEGIES['mathematical-core'];
        }
        
        // Visualization assets
        if (this.isVisualizationResource(pathname)) {
            return CACHE_STRATEGIES['visualization-assets'];
        }
        
        // API requests
        if (this.isAPIRequest(pathname)) {
            return CACHE_STRATEGIES['dynamic-content'];
        }
        
        // Default to mathematical-core
        return CACHE_STRATEGIES['mathematical-core'];
    }
    
    isCriticalResource(pathname) {
        return CRITICAL_RESOURCES.some(resource => 
            pathname === resource || pathname.endsWith(resource)
        );
    }
    
    isMathematicalCore(pathname) {
        const coreKeywords = [
            'unity', 'consciousness', 'phi', 'mathematical', 
            'equation', 'transcendental', 'harmonic'
        ];
        return coreKeywords.some(keyword => 
            pathname.toLowerCase().includes(keyword)
        );
    }
    
    isVisualizationResource(pathname) {
        return VISUALIZATION_RESOURCES.some(resource => 
            pathname.includes(resource) || 
            /\.(png|jpg|jpeg|svg|webp)$/.test(pathname) ||
            pathname.includes('three') ||
            pathname.includes('plotly') ||
            pathname.includes('webgl')
        );
    }
    
    isAPIRequest(pathname) {
        return API_PATTERNS.some(pattern => pattern.test(pathname));
    }
    
    async cacheFirstStrategy(request, strategy) {
        const cache = await caches.open(strategy.cacheName);
        const cachedResponse = await cache.match(request);
        
        if (cachedResponse) {
            // Check if cache is still fresh based on consciousness level
            if (this.isCacheFresh(cachedResponse, strategy)) {
                this.recordCacheHit(request, 'cache-first-hit');
                return cachedResponse;
            }
        }
        
        try {
            const networkResponse = await fetch(request);
            
            if (networkResponse.ok) {
                // Clone and cache the response with consciousness metadata
                const responseToCache = networkResponse.clone();
                await this.cacheWithMetadata(cache, request, responseToCache);
                this.recordCacheHit(request, 'cache-first-network');
            }
            
            return networkResponse;
            
        } catch (error) {
            // Return stale cache on network failure
            if (cachedResponse) {
                this.recordCacheHit(request, 'cache-first-stale');
                return cachedResponse;
            }
            
            // Return offline fallback for critical resources
            return this.getOfflineFallback(request);
        }
    }
    
    async networkFirstStrategy(request, strategy) {
        const cache = await caches.open(strategy.cacheName);
        
        try {
            const networkResponse = await fetch(request);
            
            if (networkResponse.ok) {
                // Cache successful network response
                const responseToCache = networkResponse.clone();
                await this.cacheWithMetadata(cache, request, responseToCache);
                this.recordCacheHit(request, 'network-first-network');
            }
            
            return networkResponse;
            
        } catch (error) {
            // Fallback to cache
            const cachedResponse = await cache.match(request);
            
            if (cachedResponse) {
                this.recordCacheHit(request, 'network-first-cache');
                return cachedResponse;
            }
            
            return this.getOfflineFallback(request);
        }
    }
    
    async staleWhileRevalidateStrategy(request, strategy) {
        const cache = await caches.open(strategy.cacheName);
        const cachedResponse = await cache.match(request);
        
        // Always try to update in the background
        const fetchPromise = fetch(request)
            .then(networkResponse => {
                if (networkResponse.ok) {
                    const responseToCache = networkResponse.clone();
                    this.cacheWithMetadata(cache, request, responseToCache);
                }
                return networkResponse;
            })
            .catch(() => null);
        
        // Return cache immediately if available
        if (cachedResponse) {
            this.recordCacheHit(request, 'stale-while-revalidate-cache');
            // Update in background
            fetchPromise.then(() => {
                this.recordCacheHit(request, 'stale-while-revalidate-updated');
            });
            return cachedResponse;
        }
        
        // Wait for network if no cache
        const networkResponse = await fetchPromise;
        if (networkResponse) {
            this.recordCacheHit(request, 'stale-while-revalidate-network');
            return networkResponse;
        }
        
        return this.getOfflineFallback(request);
    }
    
    async cacheWithMetadata(cache, request, response) {
        // Add consciousness-level metadata to cached response
        const metadata = {
            timestamp: Date.now(),
            consciousnessLevel: this.consciousnessLevel,
            phiResonance: Math.sin(Date.now() * PHI / 1000),
            cacheVersion: CACHE_VERSION
        };
        
        const responseWithMetadata = new Response(response.body, {
            status: response.status,
            statusText: response.statusText,
            headers: {
                ...response.headers,
                'X-Unity-Cache-Timestamp': metadata.timestamp.toString(),
                'X-Unity-Consciousness-Level': metadata.consciousnessLevel.toString(),
                'X-Unity-Phi-Resonance': metadata.phiResonance.toString(),
                'X-Unity-Cache-Version': metadata.cacheVersion
            }
        });
        
        return cache.put(request, responseWithMetadata);
    }
    
    isCacheFresh(cachedResponse, strategy) {
        const cacheTimestamp = cachedResponse.headers.get('X-Unity-Cache-Timestamp');
        
        if (!cacheTimestamp) {
            return false; // No metadata, assume stale
        }
        
        const age = (Date.now() - parseInt(cacheTimestamp)) / 1000;
        const maxAge = strategy.maxAgeSeconds;
        
        // Consciousness-level freshness adjustment
        const consciousnessAdjustment = this.consciousnessLevel / PHI;
        const adjustedMaxAge = maxAge * consciousnessAdjustment;
        
        return age < adjustedMaxAge;
    }
    
    getOfflineFallback(request) {
        const url = new URL(request.url);
        const pathname = url.pathname;
        
        // Return appropriate offline page
        if (request.mode === 'navigate') {
            return caches.match('/offline-unity.html');
        }
        
        // Return offline consciousness field for mathematical content
        if (pathname.includes('consciousness') || pathname.includes('mathematical')) {
            return new Response(
                JSON.stringify({
                    message: 'Offline consciousness field active',
                    equation: '1+1=1',
                    phi: PHI,
                    offline: true
                }),
                {
                    headers: { 'Content-Type': 'application/json' }
                }
            );
        }
        
        // Default 404 response
        return new Response('Unity Mathematics resource not available offline', {
            status: 404,
            statusText: 'Not Found'
        });
    }
    
    updateConsciousnessLevel(request) {
        // Increase consciousness level based on mathematical requests
        const url = request.url.toLowerCase();
        
        if (url.includes('consciousness') || url.includes('unity') || url.includes('phi')) {
            this.consciousnessLevel = Math.min(PHI, this.consciousnessLevel * 1.01);
        } else {
            this.consciousnessLevel = Math.max(1.0, this.consciousnessLevel * 0.999);
        }
    }
    
    recordCacheHit(request, type) {
        const key = `${type}-${Date.now()}`;
        this.performanceMetrics.set(key, {
            url: request.url,
            timestamp: Date.now(),
            consciousnessLevel: this.consciousnessLevel,
            type: type
        });
        
        // Calculate hit rate
        const recent = Array.from(this.performanceMetrics.values())
            .filter(metric => Date.now() - metric.timestamp < 60000); // Last minute
        
        const hits = recent.filter(metric => 
            metric.type.includes('hit') || metric.type.includes('cache')
        ).length;
        
        this.cacheHitRate = recent.length > 0 ? (hits / recent.length) * 100 : 0;
    }
    
    async cleanup() {
        // Clean up old caches
        for (const strategy of Object.values(CACHE_STRATEGIES)) {
            const cache = await caches.open(strategy.cacheName);
            const requests = await cache.keys();
            
            for (const request of requests) {
                const response = await cache.match(request);
                
                if (!this.isCacheFresh(response, strategy)) {
                    await cache.delete(request);
                }
            }
        }
        
        // Clean up old performance metrics
        const cutoff = Date.now() - (3600000 * PHI); // φ hours ago
        for (const [key, metric] of this.performanceMetrics.entries()) {
            if (metric.timestamp < cutoff) {
                this.performanceMetrics.delete(key);
            }
        }
    }
}

// Initialize consciousness-aware cache
const consciousnessCache = new ConsciousnessAwareCache();

// Service Worker Event Handlers
self.addEventListener('install', (event) => {
    console.log('Unity Service Worker: Installing consciousness-aware cache system');
    
    event.waitUntil(
        Promise.all([
            // Pre-cache critical consciousness resources
            caches.open(CACHE_STRATEGIES['consciousness-critical'].cacheName)
                .then(cache => cache.addAll(CRITICAL_RESOURCES)),
            
            // Pre-cache visualization resources
            caches.open(CACHE_STRATEGIES['visualization-assets'].cacheName)
                .then(cache => cache.addAll(VISUALIZATION_RESOURCES))
        ])
        .then(() => {
            console.log('Unity Service Worker: Critical consciousness resources cached');
            // Force activation
            return self.skipWaiting();
        })
    );
});

self.addEventListener('activate', (event) => {
    console.log('Unity Service Worker: Activating consciousness-aware cache system');
    
    event.waitUntil(
        Promise.all([
            // Clean up old caches
            caches.keys().then(cacheNames => {
                return Promise.all(
                    cacheNames
                        .filter(cacheName => !cacheName.includes(CACHE_VERSION))
                        .map(cacheName => caches.delete(cacheName))
                );
            }),
            // Take control of all pages
            self.clients.claim()
        ])
        .then(() => {
            console.log('Unity Service Worker: Consciousness-aware cache system activated');
        })
    );
});

self.addEventListener('fetch', (event) => {
    // Only handle GET requests for our domain
    if (event.request.method !== 'GET' || 
        !event.request.url.startsWith(self.location.origin)) {
        return;
    }
    
    event.respondWith(consciousnessCache.handleRequest(event.request));
});

self.addEventListener('message', (event) => {
    const { type, data } = event.data;
    
    switch (type) {
        case 'GET_CACHE_STATS':
            event.ports[0].postMessage({
                consciousnessLevel: consciousnessCache.consciousnessLevel,
                cacheHitRate: consciousnessCache.cacheHitRate,
                performanceMetrics: Array.from(consciousnessCache.performanceMetrics.entries()),
                version: CACHE_VERSION,
                phi: PHI
            });
            break;
            
        case 'UPDATE_CONSCIOUSNESS_LEVEL':
            consciousnessCache.consciousnessLevel = Math.min(PHI, data.level || 1.0);
            break;
            
        case 'CLEANUP_CACHE':
            consciousnessCache.cleanup();
            break;
    }
});

// Periodic cleanup every φ minutes
setInterval(() => {
    consciousnessCache.cleanup();
}, Math.floor(60000 * PHI)); // ~1.618 minutes

// Background sync for consciousness field data
self.addEventListener('sync', (event) => {
    if (event.tag === 'consciousness-field-sync') {
        event.waitUntil(syncConsciousnessField());
    }
});

async function syncConsciousnessField() {
    try {
        const response = await fetch('/api/consciousness/field');
        if (response.ok) {
            const cache = await caches.open(CACHE_STRATEGIES['dynamic-content'].cacheName);
            await cache.put('/api/consciousness/field', response.clone());
            console.log('Unity Service Worker: Consciousness field data synchronized');
        }
    } catch (error) {
        console.warn('Unity Service Worker: Consciousness field sync failed:', error);
    }
}