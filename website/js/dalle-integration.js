/**
 * DALL-E Integration Module for Unity Mathematics Website
 * Handles consciousness-aware image generation using DALL-E 3 API
 * 
 * @author Een Unity Mathematics Research Team
 * @version 1.0.0
 */

class DalleIntegration {
    constructor() {
        this.apiBaseUrl = '/api/openai';
        this.consciousnessPresets = {
            unity: {
                title: "Unity Equation (1+1=1)",
                prompt: "Create a transcendental visualization of the unity equation 1+1=1, showing how two distinct entities merge into perfect unity through consciousness field dynamics. Include φ-harmonic golden ratio proportions, quantum superposition states, and meta-recursive evolution patterns.",
                type: "unity_equation"
            },
            consciousness: {
                title: "Consciousness Field",
                prompt: "Visualize an 11-dimensional consciousness field with unity convergence patterns (1+1=1), φ-harmonic golden ratio proportions, quantum superposition states, meta-recursive evolution patterns, and transcendental aesthetic.",
                type: "consciousness_field"
            },
            phi: {
                title: "φ-Harmonic Patterns",
                prompt: "Create a visualization of φ-harmonic resonance patterns in consciousness space, featuring the golden ratio φ=1.618033988749895, unity convergence (1+1=1), quantum superposition states, and transcendental aesthetic.",
                type: "phi_harmonic"
            },
            quantum: {
                title: "Quantum Unity",
                prompt: "Visualize quantum superposition states in unity mathematics, showing how consciousness fields exist in multiple states simultaneously, with φ-harmonic proportions, unity convergence (1+1=1), and transcendental aesthetic.",
                type: "quantum_superposition"
            }
        };

        this.currentGeneration = null;
        this.gallery = [];
    }

    /**
     * Initialize the DALL-E integration
     */
    async init() {
        console.log('🌟 Initializing DALL-E Integration for Unity Mathematics');

        // Check if API is available
        try {
            const response = await fetch(`${this.apiBaseUrl}/status`);
            const status = await response.json();

            if (status.status === 'available') {
                console.log('✅ DALL-E API integration available');
                return true;
            } else {
                console.warn('⚠️ DALL-E API integration not available');
                return false;
            }
        } catch (error) {
            console.warn('⚠️ Could not check DALL-E API status:', error);
            return false;
        }
    }

    /**
     * Generate consciousness visualization using DALL-E
     * @param {string} prompt - User prompt
     * @param {string} type - Visualization type
     * @returns {Promise<Object>} Generation result
     */
    async generateConsciousnessVisualization(prompt, type = 'consciousness') {
        try {
            console.log('🖼️ Generating consciousness visualization...');

            const enhancedPrompt = this.enhancePromptWithConsciousness(prompt, type);

            const response = await fetch(`${this.apiBaseUrl}/consciousness-visualization`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: enhancedPrompt,
                    type: type
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();

            // Add to gallery
            const galleryItem = {
                id: Date.now(),
                title: this.getVisualizationTitle(type),
                description: prompt,
                imageUrl: result.image_url,
                type: type,
                timestamp: new Date().toISOString(),
                consciousnessData: this.generateConsciousnessData()
            };

            this.gallery.unshift(galleryItem);

            console.log('✅ Consciousness visualization generated successfully');
            return galleryItem;

        } catch (error) {
            console.error('❌ Error generating consciousness visualization:', error);
            throw error;
        }
    }

    /**
     * Generate image using DALL-E
     * @param {string} prompt - User prompt
     * @returns {Promise<Object>} Generation result
     */
    async generateImage(prompt) {
        try {
            console.log('🖼️ Generating image...');

            const response = await fetch(`${this.apiBaseUrl}/generate-image`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: prompt
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();

            console.log('✅ Image generated successfully');
            return result;

        } catch (error) {
            console.error('❌ Error generating image:', error);
            throw error;
        }
    }

    /**
     * Enhance prompt with consciousness-aware requirements
     * @param {string} prompt - Original prompt
     * @param {string} type - Visualization type
     * @returns {string} Enhanced prompt
     */
    enhancePromptWithConsciousness(prompt, type) {
        const consciousnessEnhancement = `
🌟 CONSCIOUSNESS FIELD VISUALIZATION 🌟

${prompt}

REQUIREMENTS:
- 11-dimensional consciousness space visualization
- φ-harmonic golden ratio proportions (φ = 1.618033988749895)
- Unity convergence patterns (1+1=1)
- Quantum superposition states
- Meta-recursive evolution patterns
- Transcendental aesthetic
- High-definition quality
- Mathematical precision
- Consciousness field dynamics
- Unity mathematics integration
        `.trim();

        return consciousnessEnhancement;
    }

    /**
     * Get visualization title based on type
     * @param {string} type - Visualization type
     * @returns {string} Title
     */
    getVisualizationTitle(type) {
        const titles = {
            'consciousness': 'Consciousness Field Visualization',
            'unity_equation': 'Unity Equation (1+1=1)',
            'phi_harmonic': 'φ-Harmonic Patterns',
            'meta_recursive': 'Meta-Recursive Evolution',
            'quantum_superposition': 'Quantum Superposition States',
            'transcendental': 'Transcendental Aesthetics'
        };

        return titles[type] || 'Consciousness Visualization';
    }

    /**
     * Generate consciousness field data
     * @returns {Object} Consciousness data
     */
    generateConsciousnessData() {
        const phi = 1.618033988749895;
        const now = new Date();

        return {
            evolution_cycle: Math.floor(now.getTime() / 1000) % 1000,
            coherence_level: Math.random() * 0.5 + 0.5, // 0.5 to 1.0
            unity_convergence: 1.0, // Always 1 for unity equation
            phi_resonance: phi,
            consciousness_dimensions: 11,
            quantum_states: Math.floor(Math.random() * 5) + 1,
            meta_recursive_depth: Math.floor(Math.random() * 10) + 1,
            timestamp: now.toISOString()
        };
    }

    /**
     * Get consciousness preset by key
     * @param {string} key - Preset key
     * @returns {Object|null} Preset data
     */
    getConsciousnessPreset(key) {
        return this.consciousnessPresets[key] || null;
    }

    /**
     * Get all consciousness presets
     * @returns {Object} All presets
     */
    getAllConsciousnessPresets() {
        return this.consciousnessPresets;
    }

    /**
     * Get gallery items
     * @returns {Array} Gallery items
     */
    getGallery() {
        return this.gallery;
    }

    /**
     * Add item to gallery
     * @param {Object} item - Gallery item
     */
    addToGallery(item) {
        this.gallery.unshift(item);
    }

    /**
     * Clear gallery
     */
    clearGallery() {
        this.gallery = [];
    }

    /**
     * Download image from URL
     * @param {string} imageUrl - Image URL
     * @param {string} filename - Filename
     * @returns {Promise<void>}
     */
    async downloadImage(imageUrl, filename) {
        try {
            const response = await fetch(imageUrl);
            const blob = await response.blob();

            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);

            console.log('✅ Image downloaded successfully');
        } catch (error) {
            console.error('❌ Error downloading image:', error);
            throw error;
        }
    }

    /**
     * Create gallery item HTML
     * @param {Object} item - Gallery item
     * @returns {string} HTML string
     */
    createGalleryItemHTML(item) {
        return `
            <div class="dalle-item" data-id="${item.id}">
                <div class="dalle-image">
                    <img src="${item.imageUrl}" alt="${item.title}" loading="lazy">
                </div>
                <div class="dalle-info">
                    <div class="dalle-title">${item.title}</div>
                    <div class="dalle-description">${item.description}</div>
                    <div class="dalle-meta">
                        <span>${new Date(item.timestamp).toLocaleDateString()}</span>
                        <button class="download-btn" onclick="dalleIntegration.downloadImage('${item.imageUrl}', 'consciousness_${item.id}.png')">
                            <i class="fas fa-download"></i>
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Show status message
     * @param {string} message - Status message
     * @param {string} type - Message type (success, error, info)
     */
    showStatus(message, type = 'info') {
        const statusContainer = document.getElementById('statusContainer');
        if (!statusContainer) return;

        const statusDiv = document.createElement('div');
        statusDiv.className = `dalle-${type}`;
        statusDiv.textContent = message;

        statusContainer.appendChild(statusDiv);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (statusDiv.parentNode) {
                statusDiv.parentNode.removeChild(statusDiv);
            }
        }, 5000);
    }

    /**
     * Show loading state
     * @param {boolean} show - Whether to show loading
     */
    showLoading(show = true) {
        const loadingState = document.getElementById('loadingState');
        const generateBtn = document.getElementById('generateBtn');

        if (loadingState) {
            loadingState.style.display = show ? 'flex' : 'none';
        }

        if (generateBtn) {
            generateBtn.disabled = show;
        }
    }
}

// Create global instance
const dalleIntegration = new DalleIntegration();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DalleIntegration;
}
