/**
 * Een Unity Mathematics - Chat UI Component
 * Handles chat interface rendering and user interactions
 */

import EenConfig from '../config.js';

class ChatUI {
    constructor(stateManager) {
        this.stateManager = stateManager;
        this.container = null;
        this.messagesContainer = null;
        this.inputField = null;
        this.sendButton = null;
        this.isRendered = false;
        
        // Animation settings
        this.animationDuration = EenConfig.ui.ENABLE_ANIMATIONS ? 300 : 0;
    }

    /**
     * Create and inject chat interface
     */
    render() {
        if (this.isRendered) {
            return;
        }

        this.createChatInterface();
        this.attachEventListeners();
        this.setupAccessibility();
        this.isRendered = true;
        
        console.info('Chat UI rendered');
    }

    /**
     * Create chat interface HTML structure
     */
    createChatInterface() {
        // Create main container
        this.container = document.createElement('div');
        this.container.id = 'een-ai-chat';
        this.container.className = 'een-chat-container';
        this.container.setAttribute('role', 'dialog');
        this.container.setAttribute('aria-labelledby', 'chat-title');
        this.container.setAttribute('aria-describedby', 'chat-description');
        this.container.style.display = 'none';

        // Chat header
        const header = document.createElement('div');
        header.className = 'een-chat-header';
        header.innerHTML = `
            <div class="chat-title-section">
                <h3 id="chat-title" class="chat-title">
                    <span class="unity-symbol">∞</span>
                    Een Unity AI
                </h3>
                <p id="chat-description" class="chat-description">
                    Exploring Unity Mathematics where 1+1=1
                </p>
            </div>
            <div class="chat-controls">
                <button id="chat-theme-btn" class="chat-control-btn" title="Toggle Theme" aria-label="Toggle Theme">
                    <span class="theme-icon">🌓</span>
                </button>
                <button id="chat-settings-btn" class="chat-control-btn" title="Settings" aria-label="Open Settings">
                    <span class="settings-icon">⚙️</span>
                </button>
                <button id="chat-minimize-btn" class="chat-control-btn" title="Minimize" aria-label="Minimize Chat">
                    <span class="minimize-icon">−</span>
                </button>
                <button id="chat-close-btn" class="chat-control-btn" title="Close" aria-label="Close Chat">
                    <span class="close-icon">×</span>
                </button>
            </div>
        `;

        // Messages container
        this.messagesContainer = document.createElement('div');
        this.messagesContainer.className = 'een-chat-messages';
        this.messagesContainer.setAttribute('role', 'log');
        this.messagesContainer.setAttribute('aria-live', 'polite');
        this.messagesContainer.setAttribute('aria-label', 'Chat messages');

        // Input section
        const inputSection = document.createElement('div');
        inputSection.className = 'een-chat-input-section';
        inputSection.innerHTML = `
            <div class="input-wrapper">
                <textarea 
                    id="chat-input" 
                    class="chat-input" 
                    placeholder="Ask about Unity Mathematics, consciousness fields, or 1+1=1..."
                    rows="2"
                    maxlength="4000"
                    aria-label="Type your message about Unity Mathematics"
                ></textarea>
                <button id="chat-send-btn" class="chat-send-btn" disabled aria-label="Send message">
                    <span class="send-icon">↗</span>
                </button>
            </div>
            <div class="input-info">
                <span class="char-count">0/4000</span>
                <div class="input-actions">
                    <button id="chat-clear-btn" class="action-btn" title="Clear history" aria-label="Clear chat history">
                        Clear
                    </button>
                    <button id="chat-export-btn" class="action-btn" title="Export chat" aria-label="Export chat history">
                        Export
                    </button>
                </div>
            </div>
        `;

        // Status bar
        const statusBar = document.createElement('div');
        statusBar.className = 'een-chat-status';
        statusBar.innerHTML = `
            <div class="status-info">
                <span class="connection-status" title="Connection status">●</span>
                <span class="typing-indicator" style="display: none;">AI is thinking...</span>
            </div>
            <div class="session-info">
                <span class="session-id">Session: ${this.stateManager.sessionId || 'New'}</span>
            </div>
        `;

        // Assemble interface
        this.container.appendChild(header);
        this.container.appendChild(this.messagesContainer);
        this.container.appendChild(inputSection);
        this.container.appendChild(statusBar);

        // Add to page
        document.body.appendChild(this.container);
        
        // Get references
        this.inputField = document.getElementById('chat-input');
        this.sendButton = document.getElementById('chat-send-btn');

        // Add welcome message if no history
        if (!this.stateManager.getChatHistory().length) {
            this.addWelcomeMessage();
        } else {
            this.renderChatHistory();
        }

        // Inject styles
        this.injectStyles();
    }

    /**
     * Add welcome message to chat
     */
    addWelcomeMessage() {
        const welcomeMessage = EenConfig.prompts.WELCOME_MESSAGE;
        this.addMessage('assistant', welcomeMessage, {
            isWelcome: true,
            timestamp: Date.now()
        });
    }

    /**
     * Render existing chat history
     */
    renderChatHistory() {
        const history = this.stateManager.getChatHistory();
        for (const message of history) {
            this.displayMessage(message);
        }
        this.scrollToBottom();
    }

    /**
     * Add message to chat
     * @param {string} role - 'user' or 'assistant'
     * @param {string} content - Message content
     * @param {object} metadata - Additional metadata
     */
    addMessage(role, content, metadata = {}) {
        const message = this.stateManager.addMessage(role, content, metadata);
        this.displayMessage(message);
        return message;
    }

    /**
     * Display message in chat interface
     * @param {object} message - Message object
     */
    displayMessage(message) {
        const messageElement = document.createElement('div');
        messageElement.className = `chat-message ${message.role}-message`;
        messageElement.setAttribute('role', message.role === 'assistant' ? 'status' : 'text');
        messageElement.dataset.messageId = message.id;

        const timestamp = new Date(message.timestamp).toLocaleTimeString();
        
        messageElement.innerHTML = `
            <div class="message-header">
                <span class="message-avatar">
                    ${message.role === 'user' ? '👤' : '🤖'}
                </span>
                <span class="message-role">
                    ${message.role === 'user' ? 'You' : 'Een Unity AI'}
                </span>
                <span class="message-time" title="${new Date(message.timestamp).toLocaleString()}">
                    ${timestamp}
                </span>
            </div>
            <div class="message-content">
                ${this.formatMessageContent(message.content)}
            </div>
            ${this.renderMessageMetadata(message)}
        `;

        // Add animation
        if (EenConfig.ui.ENABLE_ANIMATIONS) {
            messageElement.style.opacity = '0';
            messageElement.style.transform = 'translateY(10px)';
        }

        this.messagesContainer.appendChild(messageElement);

        // Animate in
        if (EenConfig.ui.ENABLE_ANIMATIONS) {
            requestAnimationFrame(() => {
                messageElement.style.transition = `opacity ${this.animationDuration}ms ease, transform ${this.animationDuration}ms ease`;
                messageElement.style.opacity = '1';
                messageElement.style.transform = 'translateY(0)';
            });
        }

        this.scrollToBottom();

        // Announce to screen readers
        if (this.stateManager.getPreference('announceResponses') && message.role === 'assistant') {
            this.announceMessage(message.content);
        }
    }

    /**
     * Format message content (Markdown, LaTeX, etc.)
     * @param {string} content - Raw content
     * @returns {string} - Formatted HTML
     */
    formatMessageContent(content) {
        let formatted = content;

        // Basic Markdown formatting
        formatted = formatted
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');

        // LaTeX math rendering (if enabled)
        if (EenConfig.ui.ENABLE_MATH_RENDERING) {
            formatted = this.renderMath(formatted);
        }

        return formatted;
    }

    /**
     * Render LaTeX mathematics
     * @param {string} content - Content with LaTeX
     * @returns {string} - Content with rendered math
     */
    renderMath(content) {
        // Simple LaTeX rendering - in production, use KaTeX or MathJax
        return content
            .replace(/\$\$(.*?)\$\$/g, '<div class="math-display">$1</div>')
            .replace(/\$(.*?)\$/g, '<span class="math-inline">$1</span>');
    }

    /**
     * Render message metadata
     * @param {object} message - Message object
     * @returns {string} - Metadata HTML
     */
    renderMessageMetadata(message) {
        if (!message.tokens_used && !message.processing_time) {
            return '';
        }

        let metadata = '<div class="message-metadata">';
        
        if (message.tokens_used) {
            metadata += `<span class="metadata-item" title="Tokens used">🏷️ ${message.tokens_used}</span>`;
        }
        
        if (message.processing_time) {
            metadata += `<span class="metadata-item" title="Processing time">⏱️ ${message.processing_time.toFixed(1)}s</span>`;
        }
        
        if (message.consciousness_alignment) {
            const alignment = Math.round(message.consciousness_alignment * 100);
            metadata += `<span class="metadata-item" title="Unity alignment">∞ ${alignment}%</span>`;
        }

        metadata += '</div>';
        return metadata;
    }

    /**
     * Update typing indicator
     * @param {boolean} isTyping - Whether AI is typing
     */
    updateTypingIndicator(isTyping) {
        const indicator = this.container.querySelector('.typing-indicator');
        if (indicator) {
            indicator.style.display = isTyping ? 'inline' : 'none';
        }
    }

    /**
     * Update connection status
     * @param {string} status - 'connected', 'connecting', 'disconnected', 'error'
     */
    updateConnectionStatus(status) {
        const statusIndicator = this.container.querySelector('.connection-status');
        if (statusIndicator) {
            statusIndicator.className = `connection-status ${status}`;
            statusIndicator.title = `Connection: ${status}`;
        }
    }

    /**
     * Scroll to bottom of messages
     */
    scrollToBottom() {
        if (this.messagesContainer) {
            this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        }
    }

    /**
     * Show chat interface
     */
    show() {
        if (!this.isRendered) {
            this.render();
        }

        this.container.style.display = 'flex';
        this.stateManager.setVisible(true);
        this.stateManager.setMinimized(false);
        
        // Focus input field
        setTimeout(() => {
            if (this.inputField) {
                this.inputField.focus();
            }
        }, 100);

        // Add show animation
        if (EenConfig.ui.ENABLE_ANIMATIONS) {
            this.container.style.transform = 'translateY(20px)';
            this.container.style.opacity = '0';
            
            requestAnimationFrame(() => {
                this.container.style.transition = `all ${this.animationDuration}ms ease`;
                this.container.style.transform = 'translateY(0)';
                this.container.style.opacity = '1';
            });
        }

        console.info('Chat UI shown');
    }

    /**
     * Hide chat interface
     */
    hide() {
        if (EenConfig.ui.ENABLE_ANIMATIONS) {
            this.container.style.transition = `all ${this.animationDuration}ms ease`;
            this.container.style.transform = 'translateY(20px)';
            this.container.style.opacity = '0';
            
            setTimeout(() => {
                this.container.style.display = 'none';
                this.stateManager.setVisible(false);
            }, this.animationDuration);
        } else {
            this.container.style.display = 'none';
            this.stateManager.setVisible(false);
        }

        console.info('Chat UI hidden');
    }

    /**
     * Toggle chat visibility
     */
    toggle() {
        if (this.stateManager.isVisible) {
            this.hide();
        } else {
            this.show();
        }
    }

    /**
     * Minimize chat interface
     */
    minimize() {
        this.container.classList.add('minimized');
        this.stateManager.setMinimized(true);
        
        const minimizeBtn = this.container.querySelector('#chat-minimize-btn .minimize-icon');
        if (minimizeBtn) {
            minimizeBtn.textContent = '+';
        }
    }

    /**
     * Restore from minimized state
     */
    restore() {
        this.container.classList.remove('minimized');
        this.stateManager.setMinimized(false);
        
        const minimizeBtn = this.container.querySelector('#chat-minimize-btn .minimize-icon');
        if (minimizeBtn) {
            minimizeBtn.textContent = '−';
        }
    }

    /**
     * Clear chat messages
     */
    clearMessages() {
        this.messagesContainer.innerHTML = '';
        this.stateManager.clearHistory();
        this.addWelcomeMessage();
        console.info('Chat messages cleared');
    }

    /**
     * Get current input value
     * @returns {string}
     */
    getInputValue() {
        return this.inputField ? this.inputField.value.trim() : '';
    }

    /**
     * Set input value
     * @param {string} value - Input value
     */
    setInputValue(value) {
        if (this.inputField) {
            this.inputField.value = value;
            this.updateSendButton();
            this.updateCharCount();
        }
    }

    /**
     * Clear input field
     */
    clearInput() {
        this.setInputValue('');
    }

    /**
     * Enable/disable send button based on input
     */
    updateSendButton() {
        if (this.sendButton && this.inputField) {
            const hasContent = this.inputField.value.trim().length > 0;
            const isNotProcessing = !this.stateManager.isProcessing;
            this.sendButton.disabled = !hasContent || !isNotProcessing;
        }
    }

    /**
     * Update character count
     */
    updateCharCount() {
        const charCountElement = this.container.querySelector('.char-count');
        if (charCountElement && this.inputField) {
            const count = this.inputField.value.length;
            charCountElement.textContent = `${count}/4000`;
            
            if (count > 3800) {
                charCountElement.classList.add('warning');
            } else {
                charCountElement.classList.remove('warning');
            }
        }
    }

    /**
     * Announce message to screen readers
     * @param {string} content - Message content
     */
    announceMessage(content) {
        const announcement = document.createElement('div');
        announcement.setAttribute('aria-live', 'assertive');
        announcement.setAttribute('aria-atomic', 'true');
        announcement.className = 'sr-only';
        announcement.textContent = `AI response: ${content.substring(0, 100)}${content.length > 100 ? '...' : ''}`;
        
        document.body.appendChild(announcement);
        
        // Remove after announcement
        setTimeout(() => {
            document.body.removeChild(announcement);
        }, 1000);
    }

    /**
     * Attach event listeners
     */
    attachEventListeners() {
        // Send button
        this.sendButton.addEventListener('click', () => {
            this.onSendMessage();
        });

        // Input field
        this.inputField.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.onSendMessage();
            }
        });

        this.inputField.addEventListener('input', () => {
            this.updateSendButton();
            this.updateCharCount();
        });

        // Control buttons
        const themeBtn = this.container.querySelector('#chat-theme-btn');
        themeBtn.addEventListener('click', () => {
            this.stateManager.toggleTheme();
        });

        const minimizeBtn = this.container.querySelector('#chat-minimize-btn');
        minimizeBtn.addEventListener('click', () => {
            if (this.stateManager.isMinimized) {
                this.restore();
            } else {
                this.minimize();
            }
        });

        const closeBtn = this.container.querySelector('#chat-close-btn');
        closeBtn.addEventListener('click', () => {
            this.hide();
        });

        const clearBtn = this.container.querySelector('#chat-clear-btn');
        clearBtn.addEventListener('click', () => {
            if (confirm('Clear all chat messages?')) {
                this.clearMessages();
            }
        });

        const exportBtn = this.container.querySelector('#chat-export-btn');
        exportBtn.addEventListener('click', () => {
            this.exportChat();
        });
    }

    /**
     * Handle send message
     */
    onSendMessage() {
        const message = this.getInputValue();
        if (!message || this.stateManager.isProcessing) {
            return;
        }

        this.clearInput();
        this.addMessage('user', message);
        
        // Emit send event
        this.container.dispatchEvent(new CustomEvent('een-chat-send', {
            detail: { message }
        }));
    }

    /**
     * Export chat history
     */
    exportChat() {
        const format = prompt('Export format (json, markdown, text):', 'markdown');
        if (!format) return;

        try {
            const exported = this.stateManager.exportHistory(format);
            const blob = new Blob([exported], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = `een-chat-${this.stateManager.sessionId}.${format === 'markdown' ? 'md' : format === 'json' ? 'json' : 'txt'}`;
            a.click();
            
            URL.revokeObjectURL(url);
        } catch (error) {
            alert(`Export failed: ${error.message}`);
        }
    }

    /**
     * Setup accessibility features
     */
    setupAccessibility() {
        // Focus management
        this.container.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.hide();
            }
        });

        // Trap focus within chat when visible
        this.container.addEventListener('keydown', (e) => {
            if (e.key === 'Tab' && this.stateManager.isVisible) {
                this.trapFocus(e);
            }
        });
    }

    /**
     * Trap focus within chat interface
     * @param {KeyboardEvent} e - Keyboard event
     */
    trapFocus(e) {
        const focusableElements = this.container.querySelectorAll(
            'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        
        const firstElement = focusableElements[0];
        const lastElement = focusableElements[focusableElements.length - 1];

        if (e.shiftKey && document.activeElement === firstElement) {
            e.preventDefault();
            lastElement.focus();
        } else if (!e.shiftKey && document.activeElement === lastElement) {
            e.preventDefault();
            firstElement.focus();
        }
    }

    /**
     * Inject chat interface styles
     */
    injectStyles() {
        if (document.getElementById('een-chat-styles')) {
            return;
        }

        const styles = document.createElement('style');
        styles.id = 'een-chat-styles';
        styles.textContent = `
            /* Een Unity Mathematics Chat Styles */
            .een-chat-container {
                position: fixed;
                bottom: 20px;
                right: 20px;
                width: 400px;
                max-width: calc(100vw - 40px);
                height: 600px;
                max-height: calc(100vh - 40px);
                background: var(--chat-bg, #ffffff);
                border: 2px solid var(--chat-border, #e0e0e0);
                border-radius: 16px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
                display: flex;
                flex-direction: column;
                z-index: 10000;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                backdrop-filter: blur(10px);
            }

            .een-chat-container.minimized {
                height: auto;
            }

            .een-chat-container.minimized .een-chat-messages,
            .een-chat-container.minimized .een-chat-input-section,
            .een-chat-container.minimized .een-chat-status {
                display: none;
            }

            .een-chat-header {
                padding: 16px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 14px 14px 0 0;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .chat-title {
                margin: 0;
                font-size: 18px;
                font-weight: 600;
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .unity-symbol {
                font-size: 20px;
                transform: rotate(90deg);
                display: inline-block;
            }

            .chat-description {
                margin: 4px 0 0 0;
                font-size: 12px;
                opacity: 0.9;
            }

            .chat-controls {
                display: flex;
                gap: 4px;
            }

            .chat-control-btn {
                background: rgba(255, 255, 255, 0.2);
                border: none;
                color: white;
                width: 32px;
                height: 32px;
                border-radius: 8px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: background 0.2s ease;
            }

            .chat-control-btn:hover {
                background: rgba(255, 255, 255, 0.3);
            }

            .een-chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 16px;
                gap: 16px;
                display: flex;
                flex-direction: column;
                scroll-behavior: smooth;
            }

            .chat-message {
                display: flex;
                flex-direction: column;
                gap: 8px;
                animation: slideIn 0.3s ease;
            }

            .message-header {
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 12px;
                color: var(--chat-text-secondary, #666);
            }

            .message-avatar {
                font-size: 16px;
            }

            .message-role {
                font-weight: 600;
            }

            .message-time {
                margin-left: auto;
            }

            .message-content {
                background: var(--chat-message-bg, #f5f5f5);
                padding: 12px 16px;
                border-radius: 12px;
                border: 1px solid var(--chat-message-border, #e0e0e0);
                line-height: 1.5;
            }

            .user-message .message-content {
                background: var(--chat-user-bg, #667eea);
                color: white;
                border-color: var(--chat-user-border, #5a67d8);
                margin-left: 40px;
            }

            .assistant-message .message-content {
                margin-right: 40px;
            }

            .message-content strong {
                font-weight: 600;
            }

            .message-content em {
                font-style: italic;
            }

            .message-content code {
                background: rgba(0, 0, 0, 0.1);
                padding: 2px 4px;
                border-radius: 4px;
                font-family: 'Monaco', 'Menlo', monospace;
                font-size: 0.9em;
            }

            .math-display {
                text-align: center;
                margin: 8px 0;
                font-family: 'Times New Roman', serif;
                font-size: 1.1em;
            }

            .math-inline {
                font-family: 'Times New Roman', serif;
            }

            .message-metadata {
                display: flex;
                gap: 12px;
                font-size: 10px;
                color: var(--chat-text-secondary, #666);
                margin-top: 4px;
            }

            .metadata-item {
                display: flex;
                align-items: center;
                gap: 2px;
            }

            .een-chat-input-section {
                padding: 16px;
                border-top: 1px solid var(--chat-border, #e0e0e0);
            }

            .input-wrapper {
                display: flex;
                gap: 8px;
                margin-bottom: 8px;
            }

            .chat-input {
                flex: 1;
                border: 1px solid var(--chat-border, #e0e0e0);
                border-radius: 12px;
                padding: 12px 16px;
                resize: none;
                font-family: inherit;
                font-size: 14px;
                line-height: 1.4;
                outline: none;
                transition: border-color 0.2s ease;
            }

            .chat-input:focus {
                border-color: var(--chat-primary, #667eea);
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }

            .chat-send-btn {
                background: var(--chat-primary, #667eea);
                border: none;
                color: white;
                width: 48px;
                height: 48px;
                border-radius: 12px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s ease;
                font-size: 18px;
            }

            .chat-send-btn:hover:not(:disabled) {
                background: var(--chat-primary-hover, #5a67d8);
                transform: translateY(-1px);
            }

            .chat-send-btn:disabled {
                background: var(--chat-disabled, #ccc);
                cursor: not-allowed;
                transform: none;
            }

            .input-info {
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 12px;
                color: var(--chat-text-secondary, #666);
            }

            .char-count.warning {
                color: var(--chat-warning, #e53e3e);
            }

            .input-actions {
                display: flex;
                gap: 8px;
            }

            .action-btn {
                background: none;
                border: 1px solid var(--chat-border, #e0e0e0);
                color: var(--chat-text-secondary, #666);
                padding: 4px 8px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 11px;
                transition: all 0.2s ease;
            }

            .action-btn:hover {
                background: var(--chat-hover-bg, #f5f5f5);
                border-color: var(--chat-primary, #667eea);
                color: var(--chat-primary, #667eea);
            }

            .een-chat-status {
                padding: 8px 16px;
                background: var(--chat-status-bg, #f8f9fa);
                border-top: 1px solid var(--chat-border, #e0e0e0);
                border-radius: 0 0 14px 14px;
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-size: 11px;
                color: var(--chat-text-secondary, #666);
            }

            .connection-status {
                color: var(--chat-success, #38a169);
            }

            .connection-status.connecting {
                color: var(--chat-warning, #d69e2e);
            }

            .connection-status.disconnected,
            .connection-status.error {
                color: var(--chat-error, #e53e3e);
            }

            .typing-indicator {
                color: var(--chat-primary, #667eea);
                font-style: italic;
            }

            .sr-only {
                position: absolute;
                width: 1px;
                height: 1px;
                padding: 0;
                margin: -1px;
                overflow: hidden;
                clip: rect(0, 0, 0, 0);
                white-space: nowrap;
                border: 0;
            }

            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            /* Dark theme */
            .dark-theme .een-chat-container,
            [data-theme="dark"] .een-chat-container {
                --chat-bg: #1a202c;
                --chat-border: #2d3748;
                --chat-text-secondary: #a0aec0;
                --chat-message-bg: #2d3748;
                --chat-message-border: #4a5568;
                --chat-status-bg: #2d3748;
                --chat-hover-bg: #2d3748;
            }

            /* High contrast theme */
            .high-contrast .een-chat-container {
                --chat-bg: #000000;
                --chat-border: #ffffff;
                --chat-text-secondary: #ffffff;
                --chat-message-bg: #1a1a1a;
                --chat-message-border: #ffffff;
                --chat-primary: #ffff00;
            }

            /* Reduced motion */
            @media (prefers-reduced-motion: reduce) {
                .een-chat-container,
                .chat-message,
                .chat-control-btn,
                .chat-send-btn,
                .action-btn {
                    animation: none;
                    transition: none;
                }
            }

            /* Mobile responsive */
            @media (max-width: 480px) {
                .een-chat-container {
                    width: calc(100vw - 20px);
                    height: calc(100vh - 20px);
                    bottom: 10px;
                    right: 10px;
                    border-radius: 12px;
                }

                .user-message .message-content {
                    margin-left: 20px;
                }

                .assistant-message .message-content {
                    margin-right: 20px;
                }
            }
        `;

        document.head.appendChild(styles);
    }
}

export default ChatUI;