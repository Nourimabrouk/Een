/**
 * Error Handler for Een Unity Mathematics
 * Gracefully handles WebGL failures and other JavaScript errors
 * Version: 1.0.0 - One-Shot Error Management
 */

class ErrorHandler {
    constructor() {
        this.errorCount = 0;
        this.maxErrors = 10;
        this.init();
    }

    init() {
        // Global error handler
        window.addEventListener('error', (event) => {
            this.handleError(event.error || event.message, event.filename, event.lineno);
        });

        // Promise rejection handler
        window.addEventListener('unhandledrejection', (event) => {
            this.handleError(event.reason, 'Promise', 0);
        });

        // WebGL error handler
        this.setupWebGLErrorHandling();

        console.log('🛡️ Error Handler initialized');
    }

    handleError(error, filename, lineno) {
        this.errorCount++;

        // Don't spam console with too many errors
        if (this.errorCount <= this.maxErrors) {
            console.warn(`⚠️ Error ${this.errorCount}/${this.maxErrors}:`, {
                message: error,
                file: filename,
                line: lineno,
                timestamp: new Date().toISOString()
            });
        }

        // Show user-friendly error message for critical errors
        if (this.errorCount === 1) {
            this.showUserFriendlyError();
        }

        // Prevent error from breaking the page
        event.preventDefault();
        return false;
    }

    setupWebGLErrorHandling() {
        // Override WebGL context creation to handle failures gracefully
        const originalGetContext = HTMLCanvasElement.prototype.getContext;

        HTMLCanvasElement.prototype.getContext = function (contextType, contextAttributes) {
            try {
                const context = originalGetContext.call(this, contextType, contextAttributes);

                if (contextType === 'webgl' || contextType === 'webgl2') {
                    // Add error checking to WebGL context
                    const originalGetError = context.getError;
                    context.getError = function () {
                        const error = originalGetError.call(this);
                        if (error !== 0) {
                            console.warn('WebGL Error:', this.getErrorString(error));
                        }
                        return error;
                    };

                    // Add error string helper
                    context.getErrorString = function (error) {
                        const errorStrings = {
                            0x0500: 'INVALID_ENUM',
                            0x0501: 'INVALID_VALUE',
                            0x0502: 'INVALID_OPERATION',
                            0x0503: 'STACK_OVERFLOW',
                            0x0504: 'STACK_UNDERFLOW',
                            0x0505: 'OUT_OF_MEMORY',
                            0x0506: 'INVALID_FRAMEBUFFER_OPERATION',
                            0x0507: 'CONTEXT_LOST_WEBGL'
                        };
                        return errorStrings[error] || 'UNKNOWN_ERROR';
                    };
                }

                return context;
            } catch (error) {
                console.warn('WebGL context creation failed:', error.message);
                this.showWebGLFallback();
                return null;
            }
        };
    }

    showWebGLFallback() {
        // Create fallback for WebGL failures
        const canvases = document.querySelectorAll('canvas');
        canvases.forEach(canvas => {
            if (!canvas.getContext('webgl') && !canvas.getContext('webgl2')) {
                // Replace with fallback content
                const fallback = document.createElement('div');
                fallback.className = 'webgl-fallback';
                fallback.innerHTML = `
                    <div class="fallback-content">
                        <i class="fas fa-exclamation-triangle"></i>
                        <p>3D visualization not available</p>
                        <p>Your browser may not support WebGL</p>
                    </div>
                `;
                fallback.style.cssText = `
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background: rgba(255, 215, 0, 0.1);
                    border: 2px solid #FFD700;
                    border-radius: 8px;
                    padding: 2rem;
                    color: #FFD700;
                    text-align: center;
                    min-height: 200px;
                `;

                canvas.parentNode.replaceChild(fallback, canvas);
            }
        });
    }

    showUserFriendlyError() {
        // Create a subtle error notification
        const notification = document.createElement('div');
        notification.className = 'error-notification';
        notification.innerHTML = `
            <div class="error-content">
                <i class="fas fa-info-circle"></i>
                <span>Some features may not work optimally. Please refresh the page if you experience issues.</span>
                <button onclick="this.parentElement.parentElement.remove()" aria-label="Close notification">×</button>
            </div>
        `;

        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(255, 69, 0, 0.9);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            z-index: 10000;
            max-width: 300px;
            font-size: 0.9rem;
            backdrop-filter: blur(10px);
        `;

        document.body.appendChild(notification);

        // Auto-remove after 10 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 10000);
    }
}

// Initialize error handler
const errorHandler = new ErrorHandler(); 