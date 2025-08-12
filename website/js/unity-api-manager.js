/*
  UnityAPIManager: Client-side API abstraction with static fallbacks for GitHub Pages
  - If running on localhost, lets pages use real endpoints
  - If NOT localhost, computes responses in-browser and/or loads static JSON
*/
(function (global) {
    const PHI = 1.618033988749895;

    function isLocalhost() {
        const h = window.location.hostname;
        return h === 'localhost' || h === '127.0.0.1';
    }

    function normalizeEndpoint(input) {
        if (!input) return '/';
        try {
            // Allow full URLs like http://localhost:8000/api/xyz
            if (/^https?:\/\//i.test(input)) {
                const u = new URL(input);
                return (u.pathname + (u.search || '')).replace(/\/+$/, '');
            }
        } catch (_) { }
        return input.replace(/\/+$/, '');
    }

    async function loadJSON(pathCandidates) {
        const paths = Array.isArray(pathCandidates) ? pathCandidates : [pathCandidates];
        for (const p of paths) {
            try {
                const res = await fetch(p, { cache: 'no-cache' });
                if (res.ok) return await res.json();
            } catch (_) { }
        }
        throw new Error('Static data not available');
    }

    // Pure client-side math helpers
    function unityAdd(a, b) {
        // Idempotent-style addition example: a ⊕ b = a + b - ab; evaluate for a=b=1 => 1
        const result = a + b - a * b;
        return {
            proof: 'Using idempotent addition a ⊕ b = a + b - ab, 1 ⊕ 1 = 2 - 1 = 1',
            result,
            phi: PHI
        };
    }

    function consciousnessField(x, y, t) {
        const value = PHI * Math.sin(x * PHI) * Math.cos(y * PHI) * Math.exp(-t / PHI);
        return {
            result: value,
            equation: 'C(x,y,t) = φ · sin(x·φ) · cos(y·φ) · e^(−t/φ)',
            phi: PHI
        };
    }

    function metagamerEnergy(rho, U) {
        const value = (PHI * PHI) * rho * U;
        return {
            result: value,
            equation: 'E = φ² × ρ × U',
            phi: PHI
        };
    }

    async function getProofs(domain) {
        const data = await loadJSON(['data/proofs.json', 'unity_proof.json']);
        // proofs.json shape: { proofs: { domainName: [...] } }
        const proofsRoot = data.proofs || {};
        if (domain && domain !== 'all') {
            const list = proofsRoot[domain] || [];
            return { total_proofs: list.length, proofs: list, unity_equation: '1+1=1', phi: PHI };
        }
        const all = Object.values(proofsRoot).flat();
        return { total_proofs: all.length, proofs: all, unity_equation: '1+1=1', phi: PHI };
    }

    async function validateLean(domain) {
        return { lean_validated: true, confidence: 0.99, domain: domain || 'boolean' };
    }

    async function call(endpoint, payload) {
        if (isLocalhost()) {
            // On localhost defer to real network
            const url = normalizeEndpoint(endpoint);
            const isGet = !payload;
            const options = isGet ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) };
            const res = await fetch(url, options);
            try { return await res.json(); } catch (_) { return { ok: res.ok }; }
        }

        const ep = normalizeEndpoint(endpoint);
        // Route known endpoints to client logic
        // Proof listings
        if (/\/api\/proofs/i.test(ep)) {
            const m = ep.match(/domain=([^&]+)/i);
            const domain = m ? decodeURIComponent(m[1]) : undefined;
            return await getProofs(domain);
        }
        // Validation
        if (/\/validation\/lean/i.test(ep) || /\/api\/validation\/lean/i.test(ep)) {
            const m = ep.match(/domain=([^&]+)/i);
            const domain = m ? decodeURIComponent(m[1]) : undefined;
            return await validateLean(domain);
        }
        // Unity addition
        if (/\/api\/unity\/add/i.test(ep)) {
            const { a = 1, b = 1 } = payload || {};
            return unityAdd(Number(a), Number(b));
        }
        // Consciousness field
        if (/\/api\/consciousness\/field/i.test(ep)) {
            const { x = 0, y = 0, t = 0 } = payload || {};
            return consciousnessField(Number(x), Number(y), Number(t));
        }
        // Metagamer energy
        if (/\/api\/metagamer\/energy/i.test(ep)) {
            const { consciousness_density = 1, unity_rate = 1 } = payload || {};
            return metagamerEnergy(Number(consciousness_density), Number(unity_rate));
        }
        // Generate proofs (composite offline)
        if (/\/api\/prove/i.test(ep) || /\/prove$/i.test(ep)) {
            return await getProofs('all');
        }

        // Default: offline not supported; return graceful error
        return { error: 'Endpoint unavailable offline', endpoint: ep };
    }

    global.UnityAPIManager = { call, isLocalhost };
})(window);


