// Lightweight visualization runtime utilities for progressive enhancement
// Vanilla ES module; safe to include on any page

export function getPerformanceProfile() {
    const deviceMemory = navigator.deviceMemory || undefined;
    const hardwareConcurrency = navigator.hardwareConcurrency || undefined;
    const reducedMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const colorSchemeDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;

    return {
        deviceMemory,
        hardwareConcurrency,
        reducedMotion,
        colorSchemeDark,
        userAgent: navigator.userAgent,
    };
}

export function isLowEndDevice(profile = getPerformanceProfile()) {
    const fewCores = typeof profile.hardwareConcurrency === 'number' && profile.hardwareConcurrency <= 4;
    const lowMem = typeof profile.deviceMemory === 'number' && profile.deviceMemory <= 4;
    return Boolean(profile.reducedMotion || fewCores || lowMem);
}

export function isWebGLAvailable() {
    try {
        const canvas = document.createElement('canvas');
        return !!(
            canvas.getContext('webgl') ||
            canvas.getContext('experimental-webgl') ||
            canvas.getContext('webgl2')
        );
    } catch (_) {
        return false;
    }
}

// requestIdleCallback polyfill
export const requestIdle = (cb) => {
    if (typeof window.requestIdleCallback === 'function') return window.requestIdleCallback(cb);
    return setTimeout(() => cb({ didTimeout: false, timeRemaining: () => 0 }), 1);
};

export const cancelIdle = (id) => {
    if (typeof window.cancelIdleCallback === 'function') return window.cancelIdleCallback(id);
    clearTimeout(id);
};

export function throttle(fn, wait) {
    let inFlight = false;
    let lastArgs = null;
    return function throttled(...args) {
        lastArgs = args;
        if (inFlight) return;
        inFlight = true;
        setTimeout(() => {
            inFlight = false;
            const callArgs = lastArgs;
            lastArgs = null;
            fn.apply(this, callArgs || []);
        }, wait);
    };
}

export function createFPSMeter(targetElement) {
    let frames = 0;
    let lastTime = performance.now();
    let rafId = null;
    let fps = 0;

    function loop() {
        frames += 1;
        const now = performance.now();
        if (now - lastTime >= 1000) {
            fps = Math.round((frames * 1000) / (now - lastTime));
            frames = 0;
            lastTime = now;
            targetElement.textContent = `FPS: ${fps}`;
        }
        rafId = requestAnimationFrame(loop);
    }

    return {
        start() {
            if (rafId == null) rafId = requestAnimationFrame(loop);
        },
        stop() {
            if (rafId != null) cancelAnimationFrame(rafId), (rafId = null);
        },
        getFPS() {
            return fps;
        },
    };
}

export function createVisibilityController() {
    let pageVisible = !document.hidden;
    const subscribers = new Set();

    function notify() {
        subscribers.forEach((cb) => {
            try { cb(pageVisible); } catch (_) { }
        });
    }

    const onVisibility = () => {
        pageVisible = !document.hidden;
        notify();
    };
    document.addEventListener('visibilitychange', onVisibility);

    return {
        isVisible: () => pageVisible,
        subscribe(cb) { subscribers.add(cb); return () => subscribers.delete(cb); },
        destroy() { document.removeEventListener('visibilitychange', onVisibility); subscribers.clear(); },
    };
}

export function withErrorBoundary(fn, onError) {
    try {
        return fn();
    } catch (err) {
        try { onError && onError(err); } catch (_) { }
        // Keep site functional
        console.error('[visualizers/runtime] Error:', err);
        return undefined;
    }
}

export function applyReducedMotionStyles() {
    if (document.getElementById('reduced-motion-style')) return;
    const style = document.createElement('style');
    style.id = 'reduced-motion-style';
    style.textContent = `
    .reduced-motion *,
    [data-reduced-motion="true"] * {
      animation: none !important;
      transition: none !important;
      scroll-behavior: auto !important;
    }
  `;
    document.head.appendChild(style);
}

// Global runtime state (non-invasive hints)
export const RuntimeState = {
    reducedMotion: false,
    lowEndDevice: false,
};

export function initializeRuntimeHints() {
    const profile = getPerformanceProfile();
    RuntimeState.lowEndDevice = isLowEndDevice(profile);
    RuntimeState.reducedMotion = profile.reducedMotion || RuntimeState.lowEndDevice;
    if (RuntimeState.reducedMotion) {
        applyReducedMotionStyles();
        document.documentElement.classList.add('reduced-motion');
        document.documentElement.setAttribute('data-reduced-motion', 'true');
    }
    if (!('requestIdleCallback' in window)) {
        // Ensure symbol exists for downstream checks
        window.requestIdleCallback = requestIdle;
        window.cancelIdleCallback = cancelIdle;
    }
    // Expose hints for inline scripts that want to cooperate
    window.__VIS_RUNTIME__ = Object.assign(window.__VIS_RUNTIME__ || {}, {
        profile,
        state: RuntimeState,
    });
}


