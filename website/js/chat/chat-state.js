/**
 * Een Unity Mathematics - Chat State Manager
 * Manages session state, chat history, and user preferences
 */

import EenConfig from '../config.js';

class ChatStateManager {
    constructor() {
        this.sessionId = null;
        this.chatHistory = [];
        this.isProcessing = false;
        this.isVisible = false;
        this.isMinimized = false;
        this.currentStreamController = null;
        
        // User preferences
        this.preferences = {
            theme: 'auto', // 'light', 'dark', 'auto'
            enableAnimations: true,
            enableTypingIndicator: true,
            enableMathRendering: true,
            enableVoice: false,
            enableReducedMotion: false,
            enableHighContrast: false,
            announceResponses: true
        };

        // Load persisted state
        this.loadState();
        
        // Auto-save state periodically
        this.setupAutoSave();
    }

    /**
     * Initialize new chat session
     * @param {string} sessionId - Optional session ID
     */
    initializeSession(sessionId = null) {
        this.sessionId = sessionId || this.generateSessionId();
        this.chatHistory = [];
        this.saveState();
        
        console.info(`Chat session initialized: ${this.sessionId}`);
    }

    /**
     * Generate unique session ID
     * @returns {string}
     */
    generateSessionId() {
        return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    /**
     * Add message to chat history
     * @param {string} role - 'user' or 'assistant'
     * @param {string} content - Message content
     * @param {object} metadata - Additional metadata
     */
    addMessage(role, content, metadata = {}) {
        const message = {
            id: this.generateMessageId(),
            role,
            content,
            timestamp: Date.now(),
            ...metadata
        };

        this.chatHistory.push(message);
        
        // Limit history size
        if (this.chatHistory.length > EenConfig.ui.MAX_HISTORY_ITEMS) {
            this.chatHistory.shift();
        }

        this.saveState();
        return message;
    }

    /**
     * Generate unique message ID
     * @returns {string}
     */
    generateMessageId() {
        return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    /**
     * Get chat history
     * @param {number} limit - Optional limit
     * @returns {array}
     */
    getChatHistory(limit = null) {
        if (limit && limit > 0) {
            return this.chatHistory.slice(-limit);
        }
        return [...this.chatHistory];
    }

    /**
     * Clear chat history
     */
    clearHistory() {
        this.chatHistory = [];
        this.saveState();
        console.info('Chat history cleared');
    }

    /**
     * Set processing state
     * @param {boolean} isProcessing - Processing state
     */
    setProcessing(isProcessing) {
        this.isProcessing = isProcessing;
    }

    /**
     * Set visibility state
     * @param {boolean} isVisible - Visibility state
     */
    setVisible(isVisible) {
        this.isVisible = isVisible;
        if (isVisible) {
            this.isMinimized = false;
        }
    }

    /**
     * Set minimized state
     * @param {boolean} isMinimized - Minimized state
     */
    setMinimized(isMinimized) {
        this.isMinimized = isMinimized;
        if (isMinimized) {
            this.isVisible = true;
        }
    }

    /**
     * Set stream controller
     * @param {AbortController} controller - Abort controller
     */
    setStreamController(controller) {
        this.currentStreamController = controller;
    }

    /**
     * Cancel current stream
     */
    cancelStream() {
        if (this.currentStreamController) {
            this.currentStreamController.abort();
            this.currentStreamController = null;
            this.setProcessing(false);
        }
    }

    /**
     * Update user preferences
     * @param {object} newPreferences - Preference updates
     */
    updatePreferences(newPreferences) {
        this.preferences = { ...this.preferences, ...newPreferences };
        this.saveState();
        this.applyPreferences();
    }

    /**
     * Get user preference
     * @param {string} key - Preference key
     * @param {any} defaultValue - Default value
     * @returns {any}
     */
    getPreference(key, defaultValue = null) {
        return this.preferences[key] ?? defaultValue;
    }

    /**
     * Apply preferences to DOM
     */
    applyPreferences() {
        const root = document.documentElement;
        
        // Theme
        if (this.preferences.theme === 'auto') {
            root.classList.remove('light-theme', 'dark-theme');
        } else {
            root.classList.remove('light-theme', 'dark-theme');
            root.classList.add(`${this.preferences.theme}-theme`);
        }

        // Accessibility preferences
        if (this.preferences.enableReducedMotion) {
            root.style.setProperty('--animation-duration', '0s');
            root.style.setProperty('--transition-duration', '0s');
        } else {
            root.style.removeProperty('--animation-duration');
            root.style.removeProperty('--transition-duration');
        }

        if (this.preferences.enableHighContrast) {
            root.classList.add('high-contrast');
        } else {
            root.classList.remove('high-contrast');
        }
    }

    /**
     * Get session statistics
     * @returns {object}
     */
    getSessionStats() {
        if (!this.chatHistory.length) {
            return {
                messageCount: 0,
                duration: 0,
                tokensUsed: 0
            };
        }

        const firstMessage = this.chatHistory[0];
        const lastMessage = this.chatHistory[this.chatHistory.length - 1];
        
        return {
            sessionId: this.sessionId,
            messageCount: this.chatHistory.length,
            userMessages: this.chatHistory.filter(m => m.role === 'user').length,
            assistantMessages: this.chatHistory.filter(m => m.role === 'assistant').length,
            duration: lastMessage.timestamp - firstMessage.timestamp,
            startTime: new Date(firstMessage.timestamp).toISOString(),
            lastActivity: new Date(lastMessage.timestamp).toISOString(),
            tokensUsed: this.chatHistory.reduce((total, msg) => total + (msg.tokens_used || 0), 0)
        };
    }

    /**
     * Export chat history
     * @param {string} format - Export format ('json', 'markdown', 'text')
     * @returns {string}
     */
    exportHistory(format = 'json') {
        const stats = this.getSessionStats();
        
        switch (format) {
            case 'json':
                return JSON.stringify({
                    session: stats,
                    preferences: this.preferences,
                    history: this.chatHistory
                }, null, 2);
                
            case 'markdown':
                let md = `# Een Unity Mathematics Chat Session\n\n`;
                md += `**Session ID**: ${stats.sessionId}\n`;
                md += `**Messages**: ${stats.messageCount}\n`;
                md += `**Duration**: ${Math.round(stats.duration / 1000)}s\n`;
                md += `**Tokens Used**: ${stats.tokensUsed}\n\n`;
                md += `---\n\n`;
                
                for (const msg of this.chatHistory) {
                    const time = new Date(msg.timestamp).toLocaleTimeString();
                    md += `## ${msg.role === 'user' ? '👤 User' : '🤖 Assistant'} (${time})\n\n`;
                    md += `${msg.content}\n\n`;
                }
                return md;
                
            case 'text':
                let txt = `Een Unity Mathematics Chat Session\n`;
                txt += `Session: ${stats.sessionId}\n`;
                txt += `Messages: ${stats.messageCount}\n`;
                txt += `Duration: ${Math.round(stats.duration / 1000)}s\n\n`;
                
                for (const msg of this.chatHistory) {
                    const time = new Date(msg.timestamp).toLocaleTimeString();
                    txt += `[${time}] ${msg.role}: ${msg.content}\n\n`;
                }
                return txt;
                
            default:
                throw new Error(`Unsupported export format: ${format}`);
        }
    }

    /**
     * Load state from localStorage
     */
    loadState() {
        try {
            const saved = localStorage.getItem('een-chat-state');
            if (saved) {
                const state = JSON.parse(saved);
                
                // Restore preferences
                if (state.preferences) {
                    this.preferences = { ...this.preferences, ...state.preferences };
                }
                
                // Restore session if recent (within timeout)
                if (state.sessionId && state.chatHistory && state.lastSaved) {
                    const timeSinceLastSave = Date.now() - state.lastSaved;
                    if (timeSinceLastSave < EenConfig.ui.SESSION_TIMEOUT) {
                        this.sessionId = state.sessionId;
                        this.chatHistory = state.chatHistory || [];
                        console.info(`Restored chat session: ${this.sessionId}`);
                    }
                }
            }
        } catch (error) {
            console.warn('Failed to load chat state:', error);
        }

        // Apply loaded preferences
        this.applyPreferences();
    }

    /**
     * Save state to localStorage
     */
    saveState() {
        if (!EenConfig.ui.SAVE_HISTORY) {
            return;
        }

        try {
            const state = {
                sessionId: this.sessionId,
                chatHistory: this.chatHistory,
                preferences: this.preferences,
                lastSaved: Date.now()
            };

            localStorage.setItem('een-chat-state', JSON.stringify(state));
        } catch (error) {
            console.warn('Failed to save chat state:', error);
        }
    }

    /**
     * Setup automatic state saving
     */
    setupAutoSave() {
        // Save state every 30 seconds
        setInterval(() => {
            this.saveState();
        }, 30000);

        // Save state on page unload
        window.addEventListener('beforeunload', () => {
            this.saveState();
        });
    }

    /**
     * Check if dark mode is preferred
     * @returns {boolean}
     */
    isDarkMode() {
        if (this.preferences.theme === 'dark') return true;
        if (this.preferences.theme === 'light') return false;
        
        // Auto mode - use system preference
        return window.matchMedia('(prefers-color-scheme: dark)').matches;
    }

    /**
     * Toggle theme
     */
    toggleTheme() {
        const themes = ['auto', 'light', 'dark'];
        const currentIndex = themes.indexOf(this.preferences.theme);
        const nextTheme = themes[(currentIndex + 1) % themes.length];
        
        this.updatePreferences({ theme: nextTheme });
    }
}

export default ChatStateManager;