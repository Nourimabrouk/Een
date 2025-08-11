// Unified site enhancements (safe, non-invasive)

// Respect reduced motion and allow toggling via query ?reduceMotion=1
try {
    const params = new URLSearchParams(location.search);
    const preferReduce =
        params.get('reduceMotion') === '1' ||
        window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (preferReduce) document.documentElement.classList.add('reduce-motion');
} catch (_) { }

// Link prefetch on hover for same-origin navigations
const prefetchCache = new Set();
function prefetch(url) {
    if (!url || prefetchCache.has(url)) return;
    try {
        const u = new URL(url, location.href);
        if (u.origin !== location.origin) return;
        const link = document.createElement('link');
        link.rel = 'prefetch';
        link.href = u.href;
        document.head.appendChild(link);
        prefetchCache.add(u.href);
    } catch (_) { }
}

document.addEventListener('mouseover', (e) => {
    const a = e.target.closest('a[href]');
    if (a) prefetch(a.getAttribute('href'));
});

// Keyboard focus ring only when using keyboard
let usingKeyboard = false;
document.addEventListener('keydown', (e) => {
    if (e.key === 'Tab') {
        usingKeyboard = true;
        document.documentElement.classList.add('using-keyboard');
    }
});
document.addEventListener('mousedown', () => {
    if (usingKeyboard) {
        usingKeyboard = false;
        document.documentElement.classList.remove('using-keyboard');
    }
});

// Skip-link support if main is present
document.addEventListener('DOMContentLoaded', () => {
    const main = document.querySelector('main#main');
    if (!main) return;
    // Ensure main is focusable for skip-link target
    if (!main.hasAttribute('tabindex')) main.setAttribute('tabindex', '-1');
});

// Small performance hint: lazy-load images without loading attribute
const imgs = document.querySelectorAll('img:not([loading])');
imgs.forEach((img) => img.setAttribute('loading', 'lazy'));



