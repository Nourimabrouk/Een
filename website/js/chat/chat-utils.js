/**
 * Een Unity Mathematics - Chat Utilities
 * Helper functions for message processing, formatting, and UI utilities
 */

import EenConfig from '../config.js';

/**
 * Debounce function to limit rapid function calls
 * @param {Function} func - Function to debounce
 * @param {number} delay - Delay in milliseconds
 * @returns {Function}
 */
export function debounce(func, delay = 300) {
    let timeoutId;
    return function (...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => func.apply(this, args), delay);
    };
}

/**
 * Throttle function to limit function calls to once per interval
 * @param {Function} func - Function to throttle
 * @param {number} interval - Interval in milliseconds
 * @returns {Function}
 */
export function throttle(func, interval = 100) {
    let lastCall = 0;
    return function (...args) {
        const now = Date.now();
        if (now - lastCall >= interval) {
            lastCall = now;
            return func.apply(this, args);
        }
    };
}

/**
 * Format timestamp for display
 * @param {number} timestamp - Unix timestamp
 * @param {string} format - Format type ('short', 'long', 'relative')
 * @returns {string}
 */
export function formatTimestamp(timestamp, format = 'short') {
    const date = new Date(timestamp);
    const now = new Date();
    
    switch (format) {
        case 'short':
            return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            
        case 'long':
            return date.toLocaleString();
            
        case 'relative':
            const diff = now - date;
            const minutes = Math.floor(diff / 60000);
            const hours = Math.floor(diff / 3600000);
            const days = Math.floor(diff / 86400000);
            
            if (minutes < 1) return 'just now';
            if (minutes < 60) return `${minutes}m ago`;
            if (hours < 24) return `${hours}h ago`;
            return `${days}d ago`;
            
        default:
            return date.toLocaleString();
    }
}

/**
 * Format file size for display
 * @param {number} bytes - Size in bytes
 * @returns {string}
 */
export function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

/**
 * Format number with commas
 * @param {number} num - Number to format
 * @returns {string}
 */
export function formatNumber(num) {
    return num.toLocaleString();
}

/**
 * Calculate reading time estimate
 * @param {string} text - Text content
 * @param {number} wpm - Words per minute (default: 200)
 * @returns {string}
 */
export function estimateReadingTime(text, wpm = 200) {
    const words = text.split(/\s+/).length;
    const minutes = Math.ceil(words / wpm);
    return minutes === 1 ? '1 minute' : `${minutes} minutes`;
}

/**
 * Sanitize HTML to prevent XSS
 * @param {string} html - HTML content
 * @returns {string}
 */
export function sanitizeHTML(html) {
    const div = document.createElement('div');
    div.textContent = html;
    return div.innerHTML;
}

/**
 * Parse and format Unity Mathematics equations
 * @param {string} text - Text with equations
 * @returns {string}
 */
export function formatUnityMath(text) {
    // Replace common Unity Mathematics expressions
    return text
        .replace(/1\+1=1/g, '<span class="unity-equation">1+1=1</span>')
        .replace(/φ/g, '<span class="golden-ratio">φ</span>')
        .replace(/∞/g, '<span class="infinity-symbol">∞</span>')
        .replace(/consciousness field/gi, '<span class="consciousness-term">consciousness field</span>');
}

/**
 * Generate color scheme based on theme
 * @param {string} theme - Theme name
 * @returns {object}
 */
export function getThemeColors(theme = 'auto') {
    const themes = {
        light: {
            primary: '#667eea',
            secondary: '#764ba2',
            background: '#ffffff',
            surface: '#f8f9fa',
            text: '#2d3748',
            textSecondary: '#666666',
            border: '#e0e0e0',
            success: '#38a169',
            warning: '#d69e2e',
            error: '#e53e3e'
        },
        dark: {
            primary: '#667eea',
            secondary: '#764ba2',
            background: '#1a202c',
            surface: '#2d3748',
            text: '#f7fafc',
            textSecondary: '#a0aec0',
            border: '#4a5568',
            success: '#68d391',
            warning: '#f6e05e',
            error: '#fc8181'
        },
        auto: null // Will use CSS variables
    };

    return themes[theme] || themes.auto;
}

/**
 * Copy text to clipboard
 * @param {string} text - Text to copy
 * @returns {Promise<boolean>}
 */
export async function copyToClipboard(text) {
    try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
            await navigator.clipboard.writeText(text);
            return true;
        } else {
            // Fallback for older browsers
            const textarea = document.createElement('textarea');
            textarea.value = text;
            textarea.style.position = 'fixed';
            textarea.style.opacity = '0';
            document.body.appendChild(textarea);
            textarea.select();
            const success = document.execCommand('copy');
            document.body.removeChild(textarea);
            return success;
        }
    } catch (error) {
        console.error('Failed to copy to clipboard:', error);
        return false;
    }
}

/**
 * Download content as file
 * @param {string} content - File content
 * @param {string} filename - File name
 * @param {string} mimeType - MIME type
 */
export function downloadFile(content, filename, mimeType = 'text/plain') {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.style.display = 'none';
    
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    
    URL.revokeObjectURL(url);
}

/**
 * Check if user prefers reduced motion
 * @returns {boolean}
 */
export function prefersReducedMotion() {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/**
 * Check if user prefers dark mode
 * @returns {boolean}
 */
export function prefersDarkMode() {
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
}

/**
 * Generate unique ID
 * @param {string} prefix - Optional prefix
 * @returns {string}
 */
export function generateId(prefix = '') {
    const timestamp = Date.now().toString(36);
    const random = Math.random().toString(36).substring(2);
    return prefix ? `${prefix}_${timestamp}_${random}` : `${timestamp}_${random}`;
}

/**
 * Validate email address
 * @param {string} email - Email address
 * @returns {boolean}
 */
export function isValidEmail(email) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
}

/**
 * Validate URL
 * @param {string} url - URL to validate
 * @returns {boolean}
 */
export function isValidURL(url) {
    try {
        new URL(url);
        return true;
    } catch {
        return false;
    }
}

/**
 * Create loading animation
 * @param {string} text - Loading text
 * @returns {HTMLElement}
 */
export function createLoadingElement(text = 'Loading...') {
    const loading = document.createElement('div');
    loading.className = 'loading-indicator';
    loading.innerHTML = `
        <div class="loading-spinner"></div>
        <span class="loading-text">${text}</span>
    `;
    
    // Add spinner styles if not already present
    if (!document.getElementById('loading-styles')) {
        const styles = document.createElement('style');
        styles.id = 'loading-styles';
        styles.textContent = `
            .loading-indicator {
                display: flex;
                align-items: center;
                gap: 8px;
                color: var(--chat-text-secondary, #666);
                font-size: 14px;
            }
            
            .loading-spinner {
                width: 16px;
                height: 16px;
                border: 2px solid var(--chat-border, #e0e0e0);
                border-top: 2px solid var(--chat-primary, #667eea);
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        `;
        document.head.appendChild(styles);
    }
    
    return loading;
}

/**
 * Create error message element
 * @param {string} message - Error message
 * @param {boolean} canRetry - Whether retry option should be shown
 * @returns {HTMLElement}
 */
export function createErrorElement(message, canRetry = false) {
    const error = document.createElement('div');
    error.className = 'error-message';
    error.innerHTML = `
        <div class="error-icon">⚠️</div>
        <div class="error-content">
            <div class="error-text">${sanitizeHTML(message)}</div>
            ${canRetry ? '<button class="error-retry-btn">Retry</button>' : ''}
        </div>
    `;
    
    // Add error styles if not already present
    if (!document.getElementById('error-styles')) {
        const styles = document.createElement('style');
        styles.id = 'error-styles';
        styles.textContent = `
            .error-message {
                display: flex;
                align-items: flex-start;
                gap: 8px;
                padding: 12px;
                background: var(--chat-error-bg, #fed7d7);
                border: 1px solid var(--chat-error-border, #feb2b2);
                border-radius: 8px;
                color: var(--chat-error-text, #c53030);
                font-size: 14px;
            }
            
            .error-icon {
                font-size: 16px;
                flex-shrink: 0;
            }
            
            .error-content {
                flex: 1;
            }
            
            .error-text {
                margin-bottom: 8px;
            }
            
            .error-retry-btn {
                background: var(--chat-error-text, #c53030);
                color: white;
                border: none;
                padding: 4px 8px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
                transition: background 0.2s ease;
            }
            
            .error-retry-btn:hover {
                background: var(--chat-error-hover, #9c2626);
            }
        `;
        document.head.appendChild(styles);
    }
    
    return error;
}

/**
 * Format consciousness alignment as percentage with visual indicator
 * @param {number} alignment - Alignment value (0-1)
 * @returns {string}
 */
export function formatConsciousnessAlignment(alignment) {
    const percentage = Math.round(alignment * 100);
    const indicator = alignment >= 0.8 ? '🌟' : alignment >= 0.6 ? '✨' : alignment >= 0.4 ? '⭐' : '💫';
    return `${indicator} ${percentage}%`;
}

/**
 * Extract mathematical expressions from text
 * @param {string} text - Text content
 * @returns {array}
 */
export function extractMathExpressions(text) {
    const expressions = [];
    
    // LaTeX expressions
    const latexInline = text.match(/\$([^$]+)\$/g);
    if (latexInline) {
        expressions.push(...latexInline.map(expr => ({ type: 'latex-inline', content: expr })));
    }
    
    const latexDisplay = text.match(/\$\$([^$]+)\$\$/g);
    if (latexDisplay) {
        expressions.push(...latexDisplay.map(expr => ({ type: 'latex-display', content: expr })));
    }
    
    // Unity Mathematics expressions
    const unityExpressions = text.match(/1\+1=1|φ|∞|consciousness field/gi);
    if (unityExpressions) {
        expressions.push(...unityExpressions.map(expr => ({ type: 'unity-math', content: expr })));
    }
    
    return expressions;
}

/**
 * Create citation link element
 * @param {object} citation - Citation object
 * @returns {HTMLElement}
 */
export function createCitationElement(citation) {
    const cite = document.createElement('a');
    cite.className = 'citation-link';
    cite.href = citation.url || '#';
    cite.title = citation.title || citation.text;
    cite.target = '_blank';
    cite.rel = 'noopener noreferrer';
    cite.innerHTML = `
        <span class="citation-icon">📄</span>
        <span class="citation-text">${sanitizeHTML(citation.text || 'Reference')}</span>
    `;
    
    // Add citation styles if not already present
    if (!document.getElementById('citation-styles')) {
        const styles = document.createElement('style');
        styles.id = 'citation-styles';
        styles.textContent = `
            .citation-link {
                display: inline-flex;
                align-items: center;
                gap: 4px;
                color: var(--chat-primary, #667eea);
                text-decoration: none;
                font-size: 12px;
                padding: 2px 6px;
                border-radius: 4px;
                border: 1px solid var(--chat-primary, #667eea);
                transition: all 0.2s ease;
            }
            
            .citation-link:hover {
                background: var(--chat-primary, #667eea);
                color: white;
            }
            
            .citation-icon {
                font-size: 10px;
            }
            
            .citation-text {
                max-width: 100px;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
        `;
        document.head.appendChild(styles);
    }
    
    return cite;
}

/**
 * Retry function with exponential backoff
 * @param {Function} fn - Function to retry
 * @param {number} maxAttempts - Maximum attempts
 * @param {number} baseDelay - Base delay in milliseconds
 * @returns {Promise}
 */
export async function retry(fn, maxAttempts = 3, baseDelay = 1000) {
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
            return await fn();
        } catch (error) {
            if (attempt === maxAttempts) {
                throw error;
            }
            
            const delay = baseDelay * Math.pow(2, attempt - 1);
            console.warn(`Attempt ${attempt} failed, retrying in ${delay}ms:`, error.message);
            
            await new Promise(resolve => setTimeout(resolve, delay));
        }
    }
}

/**
 * Simple event emitter for component communication
 */
export class EventEmitter {
    constructor() {
        this.events = {};
    }
    
    on(event, callback) {
        if (!this.events[event]) {
            this.events[event] = [];
        }
        this.events[event].push(callback);
    }
    
    off(event, callback) {
        if (this.events[event]) {
            this.events[event] = this.events[event].filter(cb => cb !== callback);
        }
    }
    
    emit(event, ...args) {
        if (this.events[event]) {
            this.events[event].forEach(callback => {
                try {
                    callback(...args);
                } catch (error) {
                    console.error(`Error in event handler for ${event}:`, error);
                }
            });
        }
    }
    
    once(event, callback) {
        const wrapper = (...args) => {
            callback(...args);
            this.off(event, wrapper);
        };
        this.on(event, wrapper);
    }
}

export default {
    debounce,
    throttle,
    formatTimestamp,
    formatFileSize,
    formatNumber,
    estimateReadingTime,
    sanitizeHTML,
    formatUnityMath,
    getThemeColors,
    copyToClipboard,
    downloadFile,
    prefersReducedMotion,
    prefersDarkMode,
    generateId,
    isValidEmail,
    isValidURL,
    createLoadingElement,
    createErrorElement,
    formatConsciousnessAlignment,
    extractMathExpressions,
    createCitationElement,
    retry,
    EventEmitter
};