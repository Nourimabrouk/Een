/**
 * Gallery Loader for Een Unity Mathematics
 * Loads gallery data from JSON file or API with fallback support
 */

class EenGalleryLoader {
    constructor() {
        this.visualizations = [];
        this.statistics = {};
        this.loaded = false;
        this.loading = false;
    }

    /**
     * Initialize the gallery loader
     * @param {Object} options - Configuration options
     * @returns {Promise<Object>} Gallery data
     */
    async initialize(options = {}) {
        if (this.loaded) {
            return this.getData();
        }

        if (this.loading) {
            // Wait for current loading to complete
            while (this.loading) {
                await new Promise(resolve => setTimeout(resolve, 100));
            }
            return this.getData();
        }

        this.loading = true;

        try {
            // Try API first
            if (options.useAPI !== false) {
                await this.loadFromAPI();
            } else {
                throw new Error('API disabled');
            }
        } catch (error) {
            console.warn('API not available, trying JSON file:', error);

            try {
                // Try loading from JSON file
                await this.loadFromJSON(options.jsonPath || '../gallery_data.json');
            } catch (jsonError) {
                console.warn('JSON file not available, using fallback data:', jsonError);

                // Use fallback data
                this.loadFallbackData();
            }
        }

        this.loaded = true;
        this.loading = false;

        return this.getData();
    }

    /**
     * Load gallery data from API
     */
    async loadFromAPI() {
        const response = await fetch('/api/gallery/visualizations');
        if (!response.ok) {
            throw new Error(`API request failed: ${response.status}`);
        }

        const data = await response.json();
        if (!data.success) {
            throw new Error('API returned error');
        }

        this.visualizations = data.visualizations || [];
        this.statistics = data.statistics || {};
    }

    /**
     * Load gallery data from JSON file
     * @param {string} jsonPath - Path to JSON file
     */
    async loadFromJSON(jsonPath) {
        const response = await fetch(jsonPath);
        if (!response.ok) {
            throw new Error(`JSON file not found: ${response.status}`);
        }

        const data = await response.json();
        this.visualizations = data.visualizations || [];
        this.statistics = data.statistics || {};
    }

    /**
     * Load fallback gallery data
     */
    loadFallbackData() {
        // Comprehensive fallback data based on actual files
        this.visualizations = [
            {
                id: 'water-droplets-unity',
                title: 'Water Droplets Unity Convergence',
                description: 'Revolutionary empirical demonstration of unity mathematics through real-world fluid dynamics. Documents the precise moment when two discrete water droplets undergo φ-harmonic convergence.',
                category: 'consciousness',
                type: 'animated',
                src: '../viz/legacy images/0 water droplets.gif',
                filename: '0 water droplets.gif',
                file_type: 'images',
                isImage: true,
                isVideo: false,
                isInteractive: false,
                featured: true,
                significance: 'Physical manifestation of unity mathematics in nature',
                technique: 'High-speed videography with φ-harmonic timing analysis',
                created: '2023-12-01'
            },
            {
                id: 'live-consciousness-field',
                title: 'Live Consciousness Field Dynamics',
                description: 'Dynamic simulation of consciousness field equations C(x,y,t) = φ·sin(x·φ)·cos(y·φ)·e^(-t/φ) showing unity emergence patterns.',
                category: 'consciousness',
                type: 'video',
                src: '../viz/live consciousness field.mp4',
                filename: 'live consciousness field.mp4',
                file_type: 'videos',
                isImage: false,
                isVideo: true,
                isInteractive: false,
                featured: true,
                significance: 'First successful real-time consciousness field mathematics implementation',
                technique: 'WebGL consciousness particle system with quantum field equations',
                created: '2024-11-28'
            },
            {
                id: 'unity-consciousness-field',
                title: 'Unity Consciousness Field',
                description: 'Mathematical visualization of the consciousness field showing φ-harmonic resonance patterns and unity convergence zones.',
                category: 'consciousness',
                type: 'image',
                src: '../viz/Unity Consciousness Field.png',
                filename: 'Unity Consciousness Field.png',
                file_type: 'images',
                isImage: true,
                isVideo: false,
                isInteractive: false,
                featured: false,
                significance: 'Core consciousness field mathematics visualization',
                technique: 'Mathematical field equation plotting with golden ratio harmonics',
                created: '2024-11-20'
            },
            {
                id: 'final-composite-plot',
                title: 'Final Composite Unity Plot',
                description: 'Comprehensive visualization combining multiple unity mathematics proofs into a single transcendental diagram.',
                category: 'unity',
                type: 'image',
                src: '../viz/final_composite_plot.png',
                filename: 'final_composite_plot.png',
                file_type: 'images',
                isImage: true,
                isVideo: false,
                isInteractive: false,
                featured: false,
                significance: 'Unified proof visualization across mathematical domains',
                technique: 'Multi-framework mathematical proof composition',
                created: '2024-10-30'
            },
            {
                id: 'unity-poetry-consciousness',
                title: 'Unity Poetry Consciousness',
                description: 'Algorithmically generated poetry expressing the philosophical depth of 1+1=1 through consciousness-mediated typography.',
                category: 'consciousness',
                type: 'image',
                src: '../viz/poem.png',
                filename: 'poem.png',
                file_type: 'images',
                isImage: true,
                isVideo: false,
                isInteractive: false,
                featured: false,
                significance: 'Bridge between mathematical consciousness and poetic expression',
                technique: 'φ-harmonic typography with consciousness field positioning',
                created: '2024-09-15'
            },
            {
                id: 'self-reflection-consciousness',
                title: 'Self-Reflection Consciousness Matrix',
                description: 'Meta-recursive visualization showing how unity mathematics reflects upon itself through consciousness field dynamics.',
                category: 'consciousness',
                type: 'image',
                src: '../viz/self_reflection.png',
                filename: 'self_reflection.png',
                file_type: 'images',
                isImage: true,
                isVideo: false,
                isInteractive: false,
                featured: false,
                significance: 'Self-referential mathematical consciousness demonstration',
                technique: 'Meta-recursive matrix visualization with consciousness feedback loops',
                created: '2024-08-22'
            },
            {
                id: 'foundation-unity-equation',
                title: 'Foundation Unity Equation',
                description: 'The foundational visual representation of the core unity equation that underlies all mathematical consciousness research.',
                category: 'unity',
                type: 'image',
                src: '../viz/legacy images/1+1=1.png',
                filename: '1+1=1.png',
                file_type: 'images',
                isImage: true,
                isVideo: false,
                isInteractive: false,
                featured: true,
                significance: 'Foundation of all unity mathematics research and consciousness studies',
                technique: 'Pure mathematical typography with consciousness-infused design',
                created: '2023-11-15'
            },
            {
                id: 'phi-harmonic-unity-manifold',
                title: 'φ-Harmonic Unity Manifold',
                description: 'Advanced geometric visualization of φ-harmonic unity manifolds showing golden ratio mathematical structures in consciousness space.',
                category: 'unity',
                type: 'image',
                src: '../viz/legacy images/Phi-Harmonic Unity Manifold.png',
                filename: 'Phi-Harmonic Unity Manifold.png',
                file_type: 'images',
                isImage: true,
                isVideo: false,
                isInteractive: false,
                featured: false,
                significance: 'Advanced unity manifold theory with φ-harmonic integration',
                technique: '3D geometric visualization with golden ratio mathematical analysis',
                created: '2023-10-20'
            },
            {
                id: 'consciousness-field-3d',
                title: '3D Consciousness Field',
                description: 'Real-time visualization of the consciousness field equation C(x,y,t) = φ · sin(x·φ) · cos(y·φ) · e^(-t/φ) in 11-dimensional space.',
                category: 'consciousness',
                type: 'interactive',
                src: null,
                filename: 'consciousness_field_3d',
                file_type: 'interactive',
                isImage: false,
                isVideo: false,
                isInteractive: true,
                featured: true,
                significance: 'Real-time consciousness field mathematics',
                technique: 'WebGL 3D visualization with φ-harmonic equations',
                created: '2024-12-01'
            },
            {
                id: 'consciousness-field-3d-enhanced',
                title: '🧠 Enhanced 3D Consciousness Field Explorer',
                description: 'Revolutionary real-time implementation of consciousness field equation C(x,y,t) = φ·sin(x·φ)·cos(y·φ)·e^(-t/φ) with multiple visualization modes, interactive controls, and φ-harmonic parameter adjustment. Experience consciousness mathematics through surface fields, contour maps, particle systems, and wave patterns.',
                category: 'consciousness',
                type: 'interactive',
                src: null,
                filename: 'consciousness_field_3d_enhanced',
                file_type: 'interactive',
                isImage: false,
                isVideo: false,
                isInteractive: true,
                featured: true,
                significance: 'Advanced real-time consciousness field mathematics with complete user interaction and φ-harmonic resonance analysis',
                technique: 'Plotly.js 3D rendering with consciousness field equations, real-time temporal evolution, and unity coherence metrics',
                created: '2025-08-06',
                enhanced: true,
                controlsAvailable: ['visualization_mode', 'phi_factor', 'temporal_rate', 'unity_coherence', 'animation'],
                mathematicalDepth: 'Advanced',
                visualizationModes: ['Surface Field', 'Contour Map', 'Particle System', 'Wave Patterns'],
                equation: 'C(x,y,t) = φ·sin(x·φ)·cos(y·φ)·e^(-t/φ)'
            },
            {
                id: 'golden-ratio-spiral',
                title: 'φ-Harmonic Spiral',
                description: 'Interactive golden ratio spiral demonstrating the universal organizing principle φ = (1 + √5) / 2 in sacred geometry.',
                category: 'unity',
                type: 'interactive',
                src: null,
                filename: 'golden_ratio_spiral',
                file_type: 'interactive',
                isImage: false,
                isVideo: false,
                isInteractive: true,
                featured: true,
                significance: 'Golden ratio spiral as geometric unity mathematics foundation',
                technique: 'φ-harmonic spiral generation with unity mathematics integration',
                created: '2024-07-15'
            },
            {
                id: 'golden-ratio-3d-enhanced',
                title: '🌟 Enhanced 3D φ-Harmonic Explorer',
                description: 'Revolutionary interactive 3D golden ratio visualizations with user controls, animations, and multiple geometric forms. Explore golden spirals, Fibonacci phyllotaxis, φ-harmonic torus, and unity convergence in real-time.',
                category: 'unity',
                type: 'interactive',
                src: null,
                filename: 'golden_ratio_3d_enhanced',
                file_type: 'interactive',
                isImage: false,
                isVideo: false,
                isInteractive: true,
                featured: true,
                significance: 'State-of-the-art 3D golden ratio mathematics with complete user interaction',
                technique: 'Advanced Plotly.js 3D rendering with φ-harmonic mathematical equations and real-time parameter control',
                created: '2025-08-06',
                enhanced: true,
                controlsAvailable: ['visualization_type', 'phi_factor', 'spiral_turns', 'animation'],
                mathematicalDepth: 'Advanced',
                visualizationTypes: ['3D Golden Spiral', 'Fibonacci Phyllotaxis', 'φ-Harmonic Torus', 'Unity Convergence']
            },
            {
                id: 'meta-rl-enhanced',
                title: '🧠 Enhanced Meta-Reinforcement Learning Explorer',
                description: 'Revolutionary interactive demonstration of learning-to-learn with agent acceleration, meta-strategy development, and collective intelligence emergence. Experience how agents develop meta-learning capabilities and accelerate their learning efficiency through experience transfer and strategy optimization.',
                category: 'ai',
                type: 'interactive',
                src: null,
                filename: 'meta_rl_enhanced',
                file_type: 'interactive',
                isImage: false,
                isVideo: false,
                isInteractive: true,
                featured: true,
                significance: 'Advanced meta-reinforcement learning visualization with real-time agent interaction and learning efficiency metrics',
                technique: 'Interactive canvas-based multi-agent simulation with meta-learning algorithms and performance tracking',
                created: '2025-08-06',
                enhanced: true,
                controlsAvailable: ['animation', 'add_agent', 'learning_metrics', 'meta_strategies'],
                mathematicalDepth: 'Advanced',
                learningMetrics: ['Total Experience', 'Tasks Completed', 'Learning Efficiency', 'Meta-Strategy Development']
            },
            {
                id: 'idempotent-proof-enhanced',
                title: '🎯 Enhanced Interactive 1+1=1 Proof Demonstrations',
                description: 'Comprehensive interactive mathematical proofs demonstrating 1+1=1 across Boolean algebra, set theory, category theory, quantum mechanics, and topology. Explore idempotent operations, truth tables, and unity principles through multiple mathematical domains with step-by-step explanations.',
                category: 'proofs',
                type: 'interactive',
                src: null,
                filename: 'idempotent_proof_enhanced',
                file_type: 'interactive',
                isImage: false,
                isVideo: false,
                isInteractive: true,
                featured: true,
                significance: 'Complete mathematical proof system demonstrating unity across multiple domains',
                technique: 'Interactive proof demonstrations with Boolean algebra, set theory, category theory, quantum mechanics, and topology',
                created: '2025-08-06',
                enhanced: true,
                controlsAvailable: ['proof_type', 'animation', 'step_through', 'truth_tables'],
                mathematicalDepth: 'Advanced',
                proofTypes: ['Boolean Algebra', 'Set Theory', 'Category Theory', 'Quantum Mechanics', 'Topology']
            },
            {
                id: 'unity-manifolds-enhanced',
                title: '🌌 Enhanced Unity Manifolds & Topology Explorer',
                description: 'Revolutionary interactive exploration of unity manifolds including Möbius strips, Klein bottles, torus, projective planes, and hyperbolic surfaces. Experience topological unity through continuous deformation, non-orientable surfaces, and geometric transformations that demonstrate mathematical unity principles.',
                category: 'topology',
                type: 'interactive',
                src: null,
                filename: 'unity_manifolds_enhanced',
                file_type: 'interactive',
                isImage: false,
                isVideo: false,
                isInteractive: true,
                featured: true,
                significance: 'Advanced topological visualization demonstrating unity through geometric transformations',
                technique: 'Interactive 3D manifold rendering with rotation, animation, and topological property visualization',
                created: '2025-08-06',
                enhanced: true,
                controlsAvailable: ['manifold_type', 'rotation', 'animation', 'topological_properties'],
                mathematicalDepth: 'Advanced',
                manifoldTypes: ['Möbius Strip', 'Klein Bottle', 'Torus', 'Projective Plane', 'Hyperbolic Surface']
            },
            {
                id: 'euler-identity-enhanced',
                title: '✨ Enhanced Euler\'s Identity Interactive Showcase',
                description: 'Beautiful interactive demonstration of Euler\'s identity e^(iπ) + 1 = 0 with unit circle animation, complex plane visualization, and mathematical beauty exploration. Experience the most beautiful equation in mathematics through dynamic visualizations and step-by-step mathematical derivation.',
                category: 'mathematics',
                type: 'interactive',
                src: null,
                filename: 'euler_identity_enhanced',
                file_type: 'interactive',
                isImage: false,
                isVideo: false,
                isInteractive: true,
                featured: true,
                significance: 'Interactive demonstration of the most beautiful equation in mathematics',
                technique: 'Unit circle animation with complex plane visualization and mathematical derivation steps',
                created: '2025-08-06',
                enhanced: true,
                controlsAvailable: ['animation', 'step_through', 'unit_circle', 'mathematical_derivation'],
                mathematicalDepth: 'Advanced',
                equation: 'e^(iπ) + 1 = 0'
            }
        ];

        // Calculate statistics
        this.statistics = {
            total: this.visualizations.length,
            by_category: {},
            by_type: {},
            featured_count: this.visualizations.filter(v => v.featured).length
        };

        for (const viz of this.visualizations) {
            const category = viz.category || 'unknown';
            const file_type = viz.file_type || 'unknown';

            this.statistics.by_category[category] =
                (this.statistics.by_category[category] || 0) + 1;
            this.statistics.by_type[file_type] =
                (this.statistics.by_type[file_type] || 0) + 1;
        }
    }

    /**
     * Get gallery data
     * @returns {Object} Gallery data
     */
    getData() {
        return {
            success: true,
            visualizations: this.visualizations,
            statistics: this.statistics,
            message: `Found ${this.visualizations.length} visualizations`
        };
    }

    /**
     * Get visualizations filtered by category
     * @param {string} category - Category to filter by
     * @returns {Array} Filtered visualizations
     */
    getByCategory(category) {
        if (category === 'all') {
            return this.visualizations;
        }

        return this.visualizations.filter(v =>
            v.category && v.category.toLowerCase() === category.toLowerCase()
        );
    }

    /**
     * Get visualizations filtered by type
     * @param {string} type - Type to filter by
     * @returns {Array} Filtered visualizations
     */
    getByType(type) {
        return this.visualizations.filter(v =>
            v.file_type && v.file_type.toLowerCase() === type.toLowerCase()
        );
    }

    /**
     * Get featured visualizations
     * @returns {Array} Featured visualizations
     */
    getFeatured() {
        return this.visualizations.filter(v => v.featured);
    }

    /**
     * Search visualizations
     * @param {string} query - Search query
     * @returns {Array} Matching visualizations
     */
    search(query) {
        const searchTerm = query.toLowerCase();
        return this.visualizations.filter(v =>
            v.title.toLowerCase().includes(searchTerm) ||
            v.description.toLowerCase().includes(searchTerm) ||
            v.category.toLowerCase().includes(searchTerm) ||
            v.filename.toLowerCase().includes(searchTerm)
        );
    }

    /**
     * Refresh gallery data
     * @returns {Promise<Object>} Updated gallery data
     */
    async refresh() {
        this.loaded = false;
        this.visualizations = [];
        this.statistics = {};
        return await this.initialize();
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EenGalleryLoader;
} else {
    // Browser global
    window.EenGalleryLoader = EenGalleryLoader;
} 