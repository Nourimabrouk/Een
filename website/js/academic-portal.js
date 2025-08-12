/*
  Academic Portal: Formal Proof Gallery Renderer
  - Loads proofs from static JSON (website/data/proofs.json), with fallback to website/unity_proof.json if present
  - Renders filter controls, proof cards, expandable steps, and basic stats
  - Uses KaTeX auto-render (already included on the page) after DOM injection
*/

(function () {
  const DATA_SOURCES = [
    'data/proofs.json',
    'unity_proof.json'
  ];

  const state = {
    proofsByDomain: {},
    domains: [],
    allProofs: [],
    activeDomain: 'all'
  };

  function qs(selector, root = document) {
    return root.querySelector(selector);
  }

  function qsa(selector, root = document) {
    return Array.from(root.querySelectorAll(selector));
  }

  function sum(arr, fn) {
    return arr.reduce((acc, v) => acc + (fn ? fn(v) : v), 0);
  }

  async function fetchWithFallback(urls) {
    for (const url of urls) {
      try {
        const res = await fetch(url, { cache: 'no-cache' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
      } catch (e) {
        console.warn('Proofs fetch failed for', url, e);
      }
    }
    throw new Error('All data sources failed');
  }

  function flattenProofs(proofsByDomain) {
    const all = [];
    for (const [domain, arr] of Object.entries(proofsByDomain)) {
      if (Array.isArray(arr)) {
        for (const item of arr) all.push({ ...item, domain });
      }
    }
    return all;
  }

  function computeStats(proofs) {
    const total = proofs.length;
    const verified = proofs.filter(p => typeof p.confidence === 'number' && p.confidence >= 90).length;
    const avgConfidence = proofs.length
      ? Math.round(sum(proofs, p => (typeof p.confidence === 'number' ? p.confidence : 0)) / proofs.length)
      : 0;
    return { total, verified, avgConfidence };
  }

  function renderStats(stats) {
    const totalEl = qs('#proof-total');
    const verifiedEl = qs('#proof-verified');
    const avgEl = qs('#proof-avg-confidence');
    if (totalEl) totalEl.textContent = String(stats.total);
    if (verifiedEl) verifiedEl.textContent = String(stats.verified);
    if (avgEl) avgEl.textContent = `${stats.avgConfidence}%`;
  }

  function renderFilters(domains) {
    const container = qs('#proof-filter-buttons');
    if (!container) return;
    container.innerHTML = '';

    const mkBtn = (label, value) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'proof-filter-btn';
      btn.dataset.domain = value;
      btn.textContent = label;
      btn.addEventListener('click', () => {
        state.activeDomain = value;
        qsa('.proof-filter-btn').forEach(b => b.classList.toggle('active', b === btn));
        renderProofs();
      });
      return btn;
    };

    const allBtn = mkBtn('All Domains', 'all');
    allBtn.classList.add('active');
    container.appendChild(allBtn);

    domains.forEach(domain => container.appendChild(mkBtn(domain.replace(/_/g, ' '), domain)));
  }

  function renderProofs() {
    const grid = qs('#proof-gallery-grid');
    if (!grid) return;
    const search = qs('#proof-search');
    const query = search ? search.value.trim().toLowerCase() : '';

    const filtered = state.allProofs.filter(p => {
      const domainOk = state.activeDomain === 'all' || p.domain === state.activeDomain;
      if (!domainOk) return false;
      if (!query) return true;
      const hay = `${p.title || ''} ${p.description || ''} ${p.domain || ''}`.toLowerCase();
      return hay.includes(query);
    });

    grid.innerHTML = '';

    for (const proof of filtered) {
      const card = document.createElement('article');
      card.className = 'proof-card';
      card.innerHTML = `
        <header class="proof-card__header">
          <div class="proof-card__title">${escapeHtml(proof.title || 'Untitled Proof')}</div>
          <div class="proof-card__badges">
            <span class="badge badge-domain">${escapeHtml((proof.domain || '').replace(/_/g, ' '))}</span>
            ${proof.complexity ? `<span class="badge badge-complexity">${escapeHtml(proof.complexity)}</span>` : ''}
            ${typeof proof.confidence === 'number' ? `<span class="badge badge-confidence">${proof.confidence}%</span>` : ''}
          </div>
        </header>
        ${proof.description ? `<p class="proof-card__desc">${escapeHtml(proof.description)}</p>` : ''}
        ${Array.isArray(proof.steps) ? `
          <details class="proof-card__details">
            <summary>Show steps</summary>
            <ol class="proof-steps">
              ${proof.steps.map(s => `
                <li>
                  ${s.description ? `<div class="step-desc">${escapeHtml(s.description)}</div>` : ''}
                  ${s.equation ? `<div class="step-eq">${escapeHtml(s.equation)}</div>` : ''}
                </li>
              `).join('')}
            </ol>
          </details>
        ` : ''}
        ${proof.link ? `<div class="proof-link"><a href="${encodeURI(proof.link)}" class="resource-link">Open related page</a></div>` : ''}
      `;
      grid.appendChild(card);
    }

    if (typeof renderMathInElement === 'function') {
      try { renderMathInElement(grid, { delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '\\(', right: '\\)', display: false },
        { left: '$', right: '$', display: false }
      ]}); } catch (e) { console.warn('KaTeX render error:', e); }
    }
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function bindSearch() {
    const input = qs('#proof-search');
    if (!input) return;
    let t;
    input.addEventListener('input', () => {
      clearTimeout(t);
      t = setTimeout(renderProofs, 120);
    });
  }

  function applyStyles() {
    const css = `
      .proof-stats { display: flex; gap: 1rem; flex-wrap: wrap; justify-content: center; margin: 1rem 0 1.5rem; }
      .proof-stat { background: #f8fafc; border: 1px solid #e2e8f0; color: #2c5282; border-radius: 999px; padding: 0.4rem 0.8rem; font-size: 0.9rem; }
      .proof-controls { display: flex; gap: 0.75rem; flex-wrap: wrap; align-items: center; margin: 1rem 0 1.25rem; }
      .proof-filter-btn { background: white; border: 1px solid #e2e8f0; border-radius: 999px; padding: 0.5rem 0.9rem; cursor: pointer; color: #2c5282; }
      .proof-filter-btn.active { border-color: #4a90a4; color: #1a365d; box-shadow: 0 1px 6px rgba(0,0,0,.06); }
      .proof-search { flex: 1 1 240px; max-width: 380px; }
      .proof-search input { width: 100%; padding: 0.6rem 0.8rem; border: 1px solid #e2e8f0; border-radius: 8px; font-size: 0.95rem; }
      #proof-gallery-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; }
      .proof-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
      .proof-card__header { display: flex; align-items: baseline; justify-content: space-between; gap: 0.75rem; margin-bottom: 0.5rem; }
      .proof-card__title { font-family: 'Crimson Text', serif; font-weight: 600; color: #1a365d; font-size: 1.1rem; }
      .proof-card__badges { display: flex; gap: 0.4rem; flex-wrap: wrap; }
      .badge { border: 1px solid #e2e8f0; border-radius: 999px; padding: 0.15rem 0.5rem; font-size: 0.75rem; color: #2d4a5a; }
      .badge-confidence { border-color: #38a169; color: #2f855a; }
      .badge-complexity { border-color: #b7791f; color: #975a16; }
      .proof-card__desc { color: #4a5568; margin: 0.25rem 0 0.75rem; font-size: 0.95rem; }
      .proof-card__details summary { cursor: pointer; color: #2c5282; }
      .proof-steps { margin: 0.75rem 0 0 1rem; color: #2d3748; }
      .proof-steps .step-eq { margin-top: 0.15rem; color: #b7791f; font-family: 'KaTeX_Main', 'Times New Roman', serif; }
      .proof-link { margin-top: 0.5rem; }
    `;
    const style = document.createElement('style');
    style.setAttribute('data-academic-portal', 'proof-gallery');
    style.textContent = css;
    document.head.appendChild(style);
  }

  async function init() {
    const section = qs('#formal-proof-gallery');
    if (!section) return; // Only run on academic portal
    applyStyles();

    try {
      const data = await fetchWithFallback(DATA_SOURCES);
      const proofsByDomain = data.proofs || data; // support alternative shape
      state.proofsByDomain = proofsByDomain;
      state.domains = Object.keys(proofsByDomain);
      state.allProofs = flattenProofs(proofsByDomain);

      const stats = computeStats(state.allProofs);
      renderStats(stats);
      renderFilters(state.domains);
      bindSearch();
      renderProofs();
    } catch (e) {
      console.error('Failed to initialize proof gallery:', e);
      const grid = qs('#proof-gallery-grid');
      if (grid) grid.innerHTML = '<div style="color:#c53030;">Failed to load proofs. Please try again later.</div>';
    }
  }

  document.addEventListener('DOMContentLoaded', init);
})();


