/**
 * Consciousness Field Bootloader
 * Ensures the visualization renders reliably on static hosts (e.g., GitHub Pages)
 * independent of other page scripts. Works with either canvas id:
 *  - #consciousness-field-canvas (website version)
 *  - #consciousness-canvas (root legacy version)
 */
(function () {
    function startFallback(canvas) {
        if (!canvas) return;
        try {
            const ctx = canvas.getContext('2d');
            function resize() {
                canvas.width = canvas.clientWidth || canvas.offsetWidth || 800;
                canvas.height = canvas.clientHeight || canvas.offsetHeight || 400;
            }
            resize();
            window.addEventListener('resize', resize);

            const PHI = 1.618033988749895;
            const particles = Array.from({ length: 140 }, () => ({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 1.2,
                vy: (Math.random() - 0.5) * 1.2,
                r: Math.random() * 2 + 1,
                c: Math.random() > 0.5 ? '#FFD700' : '#00D4FF',
                a: 0.8 + Math.random() * 0.2,
            }));
            let t = 0;
            function frame() {
                if (!ctx) return;
                ctx.fillStyle = 'rgba(10,10,15,0.12)';
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                // Consciousness wave
                ctx.strokeStyle = 'rgba(157,78,221,0.35)';
                ctx.lineWidth = 2;
                ctx.beginPath();
                for (let x = 0; x < canvas.width; x += 5) {
                    const y = canvas.height / 2 +
                        Math.sin((x / canvas.width) * Math.PI * 4 + t * 0.8) * 48 *
                        Math.cos((x / canvas.width) * Math.PI * PHI);
                    if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                }
                ctx.stroke();

                // Particles influenced toward unity center
                const cx = canvas.width / 2, cy = canvas.height / 2;
                for (const p of particles) {
                    p.x += p.vx; p.y += p.vy;
                    const dx = cx - p.x, dy = cy - p.y;
                    const d = Math.hypot(dx, dy) || 1;
                    p.vx += (dx / d) * 0.05; p.vy += (dy / d) * 0.05;
                    p.vx *= 0.99; p.vy *= 0.99;
                    if (p.x < 0 || p.x > canvas.width || p.y < 0 || p.y > canvas.height) {
                        p.x = Math.random() * canvas.width; p.y = Math.random() * canvas.height;
                    }
                    ctx.save();
                    ctx.globalAlpha = p.a;
                    ctx.fillStyle = p.c;
                    ctx.shadowColor = p.c;
                    ctx.shadowBlur = 16;
                    ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill();
                    ctx.restore();
                }

                // Unity equation at center
                ctx.save();
                ctx.font = 'bold 46px JetBrains Mono, Arial, sans-serif';
                ctx.fillStyle = 'rgba(64,224,208,0.8)';
                ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
                ctx.fillText('1+1=1', cx, cy);
                ctx.restore();

                t += 0.01;
                requestAnimationFrame(frame);
            }
            frame();
        } catch (_) { /* swallow to avoid breaking page */ }
    }

    function init() {
        var canvas = document.getElementById('consciousness-field-canvas') ||
            document.getElementById('consciousness-canvas');
        if (!canvas) return;

        // Prefer complete/refined engines if present
        try {
            if (typeof window !== 'undefined') {
                if (window.CompleteConsciousnessFieldEngine) {
                    new window.CompleteConsciousnessFieldEngine(canvas.id);
                    canvas.classList.add('js-active');
                    return;
                }
                if (window.RefinedConsciousnessFieldEngine) {
                    new window.RefinedConsciousnessFieldEngine(canvas.id);
                    canvas.classList.add('js-active');
                    return;
                }
                if (window.EnhancedConsciousnessFieldEngine) {
                    new window.EnhancedConsciousnessFieldEngine(canvas.id);
                    canvas.classList.add('js-active');
                    return;
                }
            }
        } catch (_) {
            // fall through to fallback renderer
        }
        // Fallback minimal visualization
        startFallback(canvas);
    }

    // Run after load to ensure other scripts/errors do not block this boot
    if (document.readyState === 'complete') {
        setTimeout(init, 200);
    } else {
        window.addEventListener('load', function () { setTimeout(init, 200); });
    }
})();


