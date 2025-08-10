/* eslint-disable no-console */
/**
 * Visualization Fix Script - Ensures all visualizations work properly
 */

// Enhanced Plotly error handling
function safelyCreatePlot(elementId, data, layout, config = {}) {
    try {
        const element = document.getElementById(elementId);
        if (!element) {
            console.warn(`Element ${elementId} not found for plotting`);
            return;
        }

        // Check if Plotly is available
        if (typeof Plotly === 'undefined') {
            console.warn('Plotly not loaded, falling back to simple visualization');
            element.innerHTML = '<div class="fallback-viz">Visualization loading...</div>';
            return;
        }

        // Create plot with error handling
        Plotly.newPlot(elementId, data, layout, {
            responsive: true,
            displayModeBar: false,
            ...config
        }).catch(error => {
            console.error('Plotly error:', error);
            element.innerHTML = '<div class="viz-error">Visualization temporarily unavailable</div>';
        });

    } catch (error) {
        console.error('Plot creation error:', error);
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = '<div class="viz-error">Visualization error</div>';
        }
    }
}

// Enhanced Canvas error handling
function safelyDrawCanvas(canvasId, drawFunction) {
    try {
        const canvas = document.getElementById(canvasId);
        if (!canvas) {
            console.warn(`Canvas ${canvasId} not found`);
            return;
        }

        const ctx = canvas.getContext('2d');
        if (!ctx) {
            console.warn(`Could not get 2D context for canvas ${canvasId}`);
            return;
        }

        // Set canvas dimensions if not set
        if (canvas.width === 0 || canvas.height === 0) {
            canvas.width = canvas.offsetWidth || 400;
            canvas.height = canvas.offsetHeight || 300;
        }

        drawFunction(ctx, canvas);
    } catch (error) {
        console.error('Canvas drawing error:', error);
        const canvas = document.getElementById(canvasId);
        if (canvas) {
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#f0f0f0';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = '#666';
            ctx.font = '14px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('Visualization error', canvas.width/2, canvas.height/2);
        }
    }
}

// Add fallback CSS for visualizations
function addVisualizationCSS() {
    const css = `
        <style>
        .fallback-viz, .viz-error {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 200px;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border: 2px dashed #dee2e6;
            border-radius: 8px;
            color: #6c757d;
            font-style: italic;
            margin: 1rem 0;
        }
        
        .viz-error {
            background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
            border-color: #feb2b2;
            color: #c53030;
        }
        
        .loading-indicator {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #666;
            font-style: italic;
        }
        
        .spinner {
            width: 2rem;
            height: 2rem;
            border: 3px solid rgba(255, 215, 0, 0.3);
            border-top: 3px solid #FFD700;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 1rem;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Responsive visualization containers */
        .visualization-panel, .panel-content, .attention-heatmap, .training-curve {
            min-height: 250px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        /* Plotly container fixes */
        .plotly-graph-div {
            width: 100% !important;
            height: 100% !important;
        }
        
        /* Canvas responsiveness */
        canvas {
            max-width: 100%;
            height: auto;
            display: block;
        }
        </style>
    `;
    
    document.head.insertAdjacentHTML('beforeend', css);
}

// Initialize visualization fixes
function initializeVisualizationFixes() {
    // Add CSS
    addVisualizationCSS();
    
    // Expose safe plotting functions globally
    window.safelyCreatePlot = safelyCreatePlot;
    window.safelyDrawCanvas = safelyDrawCanvas;
    
    // Handle library loading failures
    window.addEventListener('error', function(e) {
        if (e.filename && (e.filename.includes('plotly') || e.filename.includes('d3'))) {
            console.warn('Visualization library failed to load:', e.filename);
            // Could show fallback message or load alternative
        }
    });
    
    console.log('✅ Visualization fixes initialized');
}

// Auto-initialize
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeVisualizationFixes);
} else {
    initializeVisualizationFixes();
}

// Export for manual use
window.VisualizationFixes = {
    safelyCreatePlot,
    safelyDrawCanvas,
    addVisualizationCSS,
    initialize: initializeVisualizationFixes
};