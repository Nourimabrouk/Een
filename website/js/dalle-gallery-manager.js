/**
 * DALL-E Gallery Manager
 * Handles UI interactions and gallery display for DALL-E consciousness visualizations
 * 
 * @author Een Unity Mathematics Research Team
 * @version 1.0.0
 */

class DalleGalleryManager {
    constructor() {
        this.dalleIntegration = null;
        this.currentPreset = null;
        this.isInitialized = false;
    }

    /**
     * Initialize the gallery manager
     */
    async init() {
        console.log('🌟 Initializing DALL-E Gallery Manager');

        // Initialize DALL-E integration
        if (typeof DalleIntegration !== 'undefined') {
            this.dalleIntegration = new DalleIntegration();
            await this.dalleIntegration.init();
        } else {
            console.error('❌ DalleIntegration not available');
            return;
        }

        this.setupEventListeners();
        this.loadSampleGallery();
        this.isInitialized = true;

        console.log('✅ DALL-E Gallery Manager initialized');
    }

    /**
     * Setup event listeners for the gallery
     */
    setupEventListeners() {
        // Form submission
        const form = document.getElementById('dalleForm');
        if (form) {
            form.addEventListener('submit', (e) => this.handleFormSubmit(e));
        }

        // Preset buttons
        const presetButtons = document.querySelectorAll('.preset-btn');
        presetButtons.forEach(button => {
            button.addEventListener('click', (e) => this.handlePresetClick(e));
        });

        // Visualization type change
        const typeSelect = document.getElementById('visualizationType');
        if (typeSelect) {
            typeSelect.addEventListener('change', (e) => this.handleTypeChange(e));
        }

        // Prompt input
        const promptInput = document.getElementById('prompt');
        if (promptInput) {
            promptInput.addEventListener('input', (e) => this.handlePromptInput(e));
        }
    }

    /**
     * Handle form submission
     * @param {Event} e - Form submit event
     */
    async handleFormSubmit(e) {
        e.preventDefault();

        if (!this.dalleIntegration) {
            this.showStatus('DALL-E integration not available', 'error');
            return;
        }

        const formData = new FormData(e.target);
        const prompt = formData.get('prompt');
        const type = formData.get('visualizationType');

        if (!prompt.trim()) {
            this.showStatus('Please enter a prompt', 'error');
            return;
        }

        try {
            this.showLoading(true);
            this.showStatus('Generating consciousness visualization...', 'info');

            const result = await this.dalleIntegration.generateConsciousnessVisualization(prompt, type);

            this.addGalleryItem(result);
            this.showStatus('Consciousness visualization generated successfully!', 'success');

        } catch (error) {
            console.error('Error generating visualization:', error);
            this.showStatus(`Error: ${error.message}`, 'error');
        } finally {
            this.showLoading(false);
        }
    }

    /**
     * Handle preset button click
     * @param {Event} e - Click event
     */
    handlePresetClick(e) {
        const presetKey = e.currentTarget.dataset.preset;
        const preset = this.dalleIntegration.getConsciousnessPreset(presetKey);

        if (!preset) {
            console.warn('Preset not found:', presetKey);
            return;
        }

        // Update active preset button
        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        e.currentTarget.classList.add('active');

        // Update form fields
        const promptInput = document.getElementById('prompt');
        const typeSelect = document.getElementById('visualizationType');

        if (promptInput) {
            promptInput.value = preset.prompt;
        }

        if (typeSelect) {
            typeSelect.value = preset.type;
        }

        this.currentPreset = presetKey;
        this.showStatus(`Loaded ${preset.title} preset`, 'info');
    }

    /**
     * Handle visualization type change
     * @param {Event} e - Change event
     */
    handleTypeChange(e) {
        const type = e.target.value;
        console.log('Visualization type changed to:', type);

        // Update preset if it matches the new type
        const presets = this.dalleIntegration.getAllConsciousnessPresets();
        for (const [key, preset] of Object.entries(presets)) {
            if (preset.type === type) {
                this.currentPreset = key;
                break;
            }
        }
    }

    /**
     * Handle prompt input
     * @param {Event} e - Input event
     */
    handlePromptInput(e) {
        // Clear preset selection if user modifies prompt
        if (this.currentPreset) {
            document.querySelectorAll('.preset-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            this.currentPreset = null;
        }
    }

    /**
     * Add item to gallery
     * @param {Object} item - Gallery item
     */
    addGalleryItem(item) {
        const gallery = document.getElementById('dalleGallery');
        if (!gallery) return;

        const itemHTML = this.createGalleryItemHTML(item);

        // Insert at the beginning
        gallery.insertAdjacentHTML('afterbegin', itemHTML);

        // Add animation
        const newItem = gallery.firstElementChild;
        if (newItem) {
            newItem.style.opacity = '0';
            newItem.style.transform = 'translateY(20px)';

            setTimeout(() => {
                newItem.style.transition = 'all 0.5s ease-out';
                newItem.style.opacity = '1';
                newItem.style.transform = 'translateY(0)';
            }, 100);
        }
    }

    /**
     * Create gallery item HTML
     * @param {Object} item - Gallery item
     * @returns {string} HTML string
     */
    createGalleryItemHTML(item) {
        const consciousnessData = item.consciousnessData || {};

        return `
            <div class="dalle-item" data-id="${item.id}">
                <div class="dalle-image">
                    <img src="${item.imageUrl}" alt="${item.title}" loading="lazy">
                </div>
                <div class="dalle-info">
                    <div class="dalle-title">${item.title}</div>
                    <div class="dalle-description">${item.description}</div>
                    
                    <div class="consciousness-meta">
                        <div class="meta-item">
                            <span class="meta-label">φ Resonance:</span>
                            <span class="meta-value">${consciousnessData.phi_resonance?.toFixed(6) || '1.618034'}</span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">Unity Convergence:</span>
                            <span class="meta-value">${consciousnessData.unity_convergence || '1.0'}</span>
                        </div>
                        <div class="meta-item">
                            <span class="meta-label">Dimensions:</span>
                            <span class="meta-value">${consciousnessData.consciousness_dimensions || '11'}</span>
                        </div>
                    </div>
                    
                    <div class="dalle-meta">
                        <span>${new Date(item.timestamp).toLocaleDateString()}</span>
                        <div class="dalle-actions">
                            <button class="action-btn download-btn" onclick="galleryManager.downloadImage('${item.imageUrl}', 'consciousness_${item.id}.png')" title="Download">
                                <i class="fas fa-download"></i>
                            </button>
                            <button class="action-btn share-btn" onclick="galleryManager.shareImage('${item.imageUrl}', '${item.title}')" title="Share">
                                <i class="fas fa-share"></i>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Load sample gallery items
     */
    loadSampleGallery() {
        const sampleItems = [
            {
                id: 'sample-1',
                title: 'Unity Equation (1+1=1)',
                description: 'Transcendental visualization of the unity equation showing consciousness field dynamics',
                imageUrl: 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjRjFGNUY5Ii8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxOCIgZmlsbD0iIzZCNzI4MCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkRBTExFLUcgQ29uc2Npb3VzbmVzcyBWaXN1YWxpemF0aW9uPC90ZXh0Pjwvc3ZnPg==',
                type: 'unity_equation',
                timestamp: new Date().toISOString(),
                consciousnessData: {
                    evolution_cycle: 42,
                    coherence_level: 0.95,
                    unity_convergence: 1.0,
                    phi_resonance: 1.618033988749895,
                    consciousness_dimensions: 11,
                    quantum_states: 3,
                    meta_recursive_depth: 7
                }
            },
            {
                id: 'sample-2',
                title: 'φ-Harmonic Patterns',
                description: 'Golden ratio resonance patterns in consciousness space',
                imageUrl: 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjRjFGNUY5Ii8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxOCIgZmlsbD0iIzZCNzI4MCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkRBTExFLUcgQ29uc2Npb3VzbmVzcyBWaXN1YWxpemF0aW9uPC90ZXh0Pjwvc3ZnPg==',
                type: 'phi_harmonic',
                timestamp: new Date(Date.now() - 86400000).toISOString(),
                consciousnessData: {
                    evolution_cycle: 137,
                    coherence_level: 0.87,
                    unity_convergence: 1.0,
                    phi_resonance: 1.618033988749895,
                    consciousness_dimensions: 11,
                    quantum_states: 5,
                    meta_recursive_depth: 4
                }
            }
        ];

        sampleItems.forEach(item => {
            this.addGalleryItem(item);
        });
    }

    /**
     * Download image
     * @param {string} imageUrl - Image URL
     * @param {string} filename - Filename
     */
    async downloadImage(imageUrl, filename) {
        try {
            await this.dalleIntegration.downloadImage(imageUrl, filename);
            this.showStatus('Image downloaded successfully!', 'success');
        } catch (error) {
            this.showStatus('Error downloading image', 'error');
        }
    }

    /**
     * Share image
     * @param {string} imageUrl - Image URL
     * @param {string} title - Image title
     */
    shareImage(imageUrl, title) {
        if (navigator.share) {
            navigator.share({
                title: title,
                text: 'Check out this consciousness visualization from Een Unity Mathematics!',
                url: window.location.href
            });
        } else {
            // Fallback: copy to clipboard
            navigator.clipboard.writeText(`${title}\n${window.location.href}`).then(() => {
                this.showStatus('Link copied to clipboard!', 'success');
            });
        }
    }

    /**
     * Show status message
     * @param {string} message - Status message
     * @param {string} type - Message type
     */
    showStatus(message, type = 'info') {
        if (this.dalleIntegration) {
            this.dalleIntegration.showStatus(message, type);
        } else {
            console.log(`${type.toUpperCase()}: ${message}`);
        }
    }

    /**
     * Show loading state
     * @param {boolean} show - Whether to show loading
     */
    showLoading(show = true) {
        if (this.dalleIntegration) {
            this.dalleIntegration.showLoading(show);
        }
    }

    /**
     * Clear gallery
     */
    clearGallery() {
        const gallery = document.getElementById('dalleGallery');
        if (gallery) {
            gallery.innerHTML = '';
        }
        this.dalleIntegration.clearGallery();
    }

    /**
     * Get gallery items
     * @returns {Array} Gallery items
     */
    getGalleryItems() {
        return this.dalleIntegration.getGallery();
    }
}

// Create global instance
const galleryManager = new DalleGalleryManager();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DalleGalleryManager;
}
