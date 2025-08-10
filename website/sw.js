// Unity Mathematics Service Worker
const CACHE_NAME = 'een-unity-v1';
const CRITICAL_ASSETS = [
    '/',
    '/metastation-hub.html',
    '/css/unified-navigation.css',
    '/css/components.css',
    '/js/unified-navigation.js',
    '/assets/images/unity_mandala.png'
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => cache.addAll(CRITICAL_ASSETS))
            .catch((error) => console.log('Cache install failed:', error))
    );
});

self.addEventListener('fetch', (event) => {
    if (event.request.destination === 'image') {
        event.respondWith(
            caches.match(event.request)
                .then((response) => response || fetch(event.request))
                .catch(() => {
                    // Return fallback image for unity mathematics
                    return new Response('<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><rect width="100" height="100" fill="#1a1a1a"/><text x="50" y="50" fill="#FFD700" text-anchor="middle" dominant-baseline="middle">1+1=1</text></svg>', {
                        headers: { 'Content-Type': 'image/svg+xml' }
                    });
                })
        );
    }
});