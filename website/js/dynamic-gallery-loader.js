/*  Een Unity – dynamic-gallery-loader.js
    Generic loader + renderer. Relies on window.unityGalleryData (populated by gallery-data.js).
    Maintains compatibility with existing gallery features while using centralized data. */

(function () {
  'use strict';

  /* CONFIG ---------------------------------------------------------- */
  const BASE_PATH = '../viz/';             // relative to gallery.html
  const IMG_EXTS = ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp'];
  const VID_EXTS = ['.mp4', '.webm', '.ogg'];

  /* Build enriched data -------------------------------------------- */
  const visualizations = (window.unityGalleryData || []).map(v => {
    const ext = v.file.slice(v.file.lastIndexOf('.')).toLowerCase();
    const src = BASE_PATH + (v.folder ? `${v.folder.replace(/\\/g, '/')}/` : '') + v.file;
    return {
      ...v,
      src,
      extension: ext,
      isImage: IMG_EXTS.includes(ext),
      isVideo: VID_EXTS.includes(ext),
      isInteractive: v.isInteractive || false
    };
  });

  /* DOM refs -------------------------------------------------------- */
  const grid = document.getElementById('galleryGrid');
  const vizCount = document.getElementById('vizCount');
  const modal = document.getElementById('imageModal');
  const modalImg = document.getElementById('modalImage');
  const modalVideo = document.getElementById('modalVideo');
  const modalTitle = document.getElementById('modalTitle');
  const modalMeta = document.getElementById('modalMeta');
  const modalDesc = document.getElementById('modalDescription');
  const modalCloseBtn = document.querySelector('.close');
  const noResults = document.getElementById('noResults');
  const generationStatus = document.getElementById('generationStatus');

  /* RENDERING ------------------------------------------------------- */
  function createCard(viz, idx) {
    const card = document.createElement('div');
    card.className = `gallery-item${viz.featured ? ' featured-item' : ''}`;
    card.dataset.category = viz.category;
    card.style.animationDelay = `${idx * 0.1}s`;          // re‑use existing fade‑in CSS

    const badge = viz.featured
      ? '<span class="featured-badge">★ FEATURED</span>'
      : '';

    let media = '';
    if (viz.isInteractive) {
      media = `
        <div class="interactive-preview" style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 250px; background: var(--bg-gradient);">
          <i class="fas fa-play-circle" style="font-size: 3rem; color: white; margin-bottom: 1rem;"></i>
          <div class="interactive-label" style="color: white; font-weight: 600; text-transform: uppercase;">CLICK TO EXPLORE</div>
        </div>
        <span class="media-indicator interactive-indicator">INTERACTIVE</span>`;
    } else if (viz.isImage) {
      const indicator = viz.extension === '.gif' ? '<span class="media-indicator gif-indicator">ANIMATED</span>' : '';
      media = `<img src="${viz.src}" alt="${viz.title}" loading="lazy">${indicator}`;
    } else if (viz.isVideo) {
      media = `<video src="${viz.src}" muted loop preload="none" width="100%" height="250" controls></video>
               <span class="media-indicator video-indicator">VIDEO</span>`;
    }

    card.innerHTML = `
      ${badge}
      ${media}
      <div class="gallery-item-info">
        <h3 class="gallery-item-title">${viz.title}</h3>
        <p class="gallery-item-type">${viz.type}</p>
        <p class="gallery-item-description">${viz.description}</p>
      </div>`;

    card.addEventListener('click', () => {
      if (viz.isInteractive && viz.link) {
        window.location.href = viz.link;
      } else {
        openModal(viz);
      }
    });
    return card;
  }

  function renderGallery(filter = 'all') {
    if (!grid) return;
    
    grid.innerHTML = '';
    const list = filter === 'all'
      ? visualizations
      : visualizations.filter(v => v.category === filter);

    // Sort: featured items first, then by creation date
    const sorted = list.sort((a, b) => {
      if (a.featured && !b.featured) return -1;
      if (!a.featured && b.featured) return 1;
      return new Date(b.created || '2024-01-01') - new Date(a.created || '2024-01-01');
    });

    sorted.forEach((viz, i) => grid.appendChild(createCard(viz, i)));
    
    if (vizCount) vizCount.textContent = sorted.length.toString();
    
    // Show/hide no results
    if (noResults) {
      noResults.style.display = sorted.length === 0 ? 'block' : 'none';
    }
  }

  /* FILTER UI ------------------------------------------------------- */
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      renderGallery(btn.dataset.filter);
    });
  });

  /* MODAL ----------------------------------------------------------- */
  function openModal(viz) {
    if (!modal) return;
    
    modal.style.display = 'block';
    
    if (modalTitle) modalTitle.textContent = viz.title;
    if (modalDesc) modalDesc.textContent = viz.description;

    if (modalMeta) {
      modalMeta.innerHTML = `
        <div class="meta-item" style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
          <span class="meta-label" style="font-weight: 600;">Type:</span>
          <span class="meta-value">${viz.type}</span>
        </div>
        <div class="meta-item" style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
          <span class="meta-label" style="font-weight: 600;">Category:</span>
          <span class="meta-value">${viz.category.charAt(0).toUpperCase() + viz.category.slice(1)}</span>
        </div>
        <div class="meta-item" style="display: flex; justify-content: space-between;">
          <span class="meta-label" style="font-weight: 600;">Format:</span>
          <span class="meta-value">${viz.extension.toUpperCase().slice(1)}</span>
        </div>
      `;
    }

    // Handle different media types
    if (viz.isVideo && modalVideo) {
      if (modalImg) modalImg.style.display = 'none';
      modalVideo.style.display = 'block';
      modalVideo.src = viz.src;
    } else if (viz.isImage && modalImg) {
      if (modalVideo) modalVideo.style.display = 'none';
      modalImg.style.display = 'block';
      modalImg.src = viz.src;
      modalImg.alt = viz.title;
    }
  }

  // Modal event listeners
  if (modal && modalCloseBtn) {
    modalCloseBtn.addEventListener('click', () => {
      modal.style.display = 'none';
      if (modalVideo) modalVideo.pause();
    });
    
    modal.addEventListener('click', e => {
      if (e.target === modal) {
        modal.style.display = 'none';
        if (modalVideo) modalVideo.pause();
      }
    });
  }

  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && modal && modal.style.display === 'block') {
      modal.style.display = 'none';
      if (modalVideo) modalVideo.pause();
    }
  });

  /* GENERATION STATUS ------------------------------------------------ */
  const generateBtn = document.getElementById('generateNewViz');
  
  if (generateBtn && generationStatus) {
    generateBtn.addEventListener('click', () => {
      // Show status
      generationStatus.style.transform = 'translateY(0)';
      generationStatus.style.opacity = '1';
      
      const statusText = generationStatus.querySelector('.status-text');
      
      // Simulate generation process
      setTimeout(() => {
        if (statusText) statusText.textContent = 'Applying φ-harmonic transformations...';
      }, 1000);
      
      setTimeout(() => {
        if (statusText) statusText.textContent = 'Converging to unity state...';
      }, 2000);
      
      setTimeout(() => {
        if (statusText) statusText.textContent = 'Visualization complete!';
        setTimeout(() => {
          generationStatus.style.transform = 'translateY(100px)';
          generationStatus.style.opacity = '0';
        }, 1000);
      }, 3000);
    });
  }

  /* INITIALISE ------------------------------------------------------ */
  // Wait for both DOM and gallery data to be ready
  function initialize() {
    if (!window.unityGalleryData) {
      console.warn('⚠️ Gallery data not loaded yet, retrying...');
      setTimeout(initialize, 100);
      return;
    }
    
    renderGallery('all');
    console.log(`✅ Een Unity Gallery: ${visualizations.length} items loaded from ${BASE_PATH}`);
  }

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize);
  } else {
    initialize();
  }

  /* Expose for dev console (optional) */
  window.dynamicGallery = { visualizations, renderGallery, openModal };
})();