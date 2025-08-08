// Auto-initialize hooks for existing pages.
// This enhances visuals non-invasively and exposes a command palette & metrics overlay.

import { initializeRuntimeHints, createFPSMeter, createVisibilityController, isWebGLAvailable, RuntimeState, withErrorBoundary } from './runtime.js';

function createOverlay() {
    const overlay = document.createElement('div');
    overlay.setAttribute('aria-live', 'polite');
    overlay.style.position = 'fixed';
    overlay.style.right = '12px';
    overlay.style.bottom = '12px';
    overlay.style.zIndex = '2147483647';
    overlay.style.fontFamily = "Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial";
    overlay.style.fontSize = '12px';
    overlay.style.color = 'var(--text-primary, #fff)';
    overlay.style.background = 'rgba(0,0,0,0.5)';
    overlay.style.border = '1px solid rgba(255,255,255,0.15)';
    overlay.style.borderRadius = '10px';
    overlay.style.backdropFilter = 'blur(6px)';
    overlay.style.webkitBackdropFilter = 'blur(6px)';
    overlay.style.padding = '8px 10px';
    overlay.style.display = 'flex';
    overlay.style.gap = '10px';
    overlay.style.alignItems = 'center';

    const fps = document.createElement('span');
    fps.textContent = 'FPS: —';
    fps.style.minWidth = '60px';

    const status = document.createElement('span');
    status.textContent = RuntimeState.lowEndDevice ? 'Low-end mode' : 'Performance mode';
    status.title = 'Adaptive quality based on device and prefers-reduced-motion.';

    const webgl = document.createElement('span');
    webgl.textContent = isWebGLAvailable() ? 'WebGL ✓' : 'WebGL ✗';
    webgl.title = 'WebGL availability check';

    const btn = document.createElement('button');
    btn.textContent = 'Command Palette (⌘/Ctrl+K)';
    btn.style.cursor = 'pointer';
    btn.style.padding = '6px 8px';
    btn.style.borderRadius = '6px';
    btn.style.border = '1px solid rgba(255,255,255,0.2)';
    btn.style.background = 'rgba(255,255,255,0.08)';
    btn.style.color = 'inherit';
    btn.addEventListener('click', openPalette);

    overlay.append(fps, status, webgl, btn);
    document.body.appendChild(overlay);

    const meter = createFPSMeter(fps);
    meter.start();

    const visibility = createVisibilityController();
    visibility.subscribe((visible) => {
        if (visible) meter.start(); else meter.stop();
    });

    return { overlay, meter, visibility };
}

function openPalette() {
    // Simple, accessible prompt-based palette to avoid UI CSS conflicts
    const choice = window.prompt('[Unity Visualizers] Command Palette\n- pause animations\n- resume animations\n- reduce motion on\n- reduce motion off\n- quality low\n- quality high\n- reload');
    if (!choice) return;
    const cmd = choice.trim().toLowerCase();
    if (cmd.includes('pause')) document.dispatchEvent(new CustomEvent('vis:pulse', { detail: { action: 'pause' } }));
    if (cmd.includes('resume')) document.dispatchEvent(new CustomEvent('vis:pulse', { detail: { action: 'resume' } }));
    if (cmd.includes('reduce') && cmd.includes('on')) document.documentElement.classList.add('reduced-motion');
    if (cmd.includes('reduce') && cmd.includes('off')) document.documentElement.classList.remove('reduced-motion');
    if (cmd.includes('quality') && cmd.includes('low')) window.__VIS_RUNTIME__.state.lowEndDevice = true;
    if (cmd.includes('quality') && cmd.includes('high')) window.__VIS_RUNTIME__.state.lowEndDevice = false;
    if (cmd.includes('reload')) location.reload();
}

function bindKeyboard() {
    document.addEventListener('keydown', (e) => {
        const isMac = navigator.platform.toUpperCase().includes('MAC');
        if ((isMac ? e.metaKey : e.ctrlKey) && e.key.toLowerCase() === 'k') {
            e.preventDefault();
            openPalette();
        }
    });
}

function enhanceExistingCanvases() {
    // Add data hooks to known containers so inline scripts can optionally read flags
    const containers = document.querySelectorAll('#canvas, #unity-canvas, #consciousness-canvas, #quantum-canvas, #fractal-canvas, #entanglement-canvas, #mandala-canvas, #neural-canvas, #3d-consciousness-viz, #phi-fractal-viz, #evolution-viz, #consciousness-field-viz');
    containers.forEach((el) => {
        el.setAttribute('data-visualizer', 'unity-runtime');
        el.setAttribute('data-quality', RuntimeState.lowEndDevice ? 'low' : 'high');
        el.setAttribute('data-reduced-motion', String(RuntimeState.reducedMotion));
    });
}

function adaptiveAnimationPulse() {
    // Notify listeners to pause/resume based on visibility and reduced motion
    const visibility = document.hidden ? 'pause' : 'resume';
    document.dispatchEvent(new CustomEvent('vis:pulse', { detail: { action: visibility } }));
}

function attachLifecycleHooks() {
    document.addEventListener('visibilitychange', adaptiveAnimationPulse);
    window.addEventListener('blur', () => document.dispatchEvent(new CustomEvent('vis:pulse', { detail: { action: 'pause' } })));
    window.addEventListener('focus', () => document.dispatchEvent(new CustomEvent('vis:pulse', { detail: { action: 'resume' } })));
}

export function autoInit() {
    withErrorBoundary(() => {
        initializeRuntimeHints();
        bindKeyboard();
        enhanceExistingCanvases();
        attachLifecycleHooks();
        // Defer overlay to idle time to avoid affecting LCP
        setTimeout(() => {
            try { createOverlay(); } catch (_) { }
        }, 0);
    });
}

// Auto-run when loaded as a module
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', autoInit);
} else {
    autoInit();
}


