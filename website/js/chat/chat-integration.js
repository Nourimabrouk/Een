/**
 * Een Unity Mathematics - Modern Chat Integration
 * State-of-the-art modular chat system with streaming, authentication, and consciousness integration
 */

import EenConfig from '../config.js';
import ChatAPIClient, { ChatAPIError } from './chat-api.js';
import ChatStateManager from './chat-state.js';
import ChatUI from './chat-ui.js';
import ChatUtils, { retry } from './chat-utils.js';

class EenChatIntegration {
    constructor(options = {}) {
        // Configuration
        this.config = {
            ...EenConfig,
            ...options
        };

        // Initialize components
        this.apiClient = new ChatAPIClient();
        this.stateManager = new ChatStateManager();
        this.ui = new ChatUI(this.stateManager);
        this.utils = ChatUtils;

        // State
        this.isInitialized = false;
        this.retryCount = 0;
        this.maxRetries = this.config.api.RETRY_ATTEMPTS;

        // Event handling
        this.eventEmitter = new ChatUtils.EventEmitter();

        // Bind methods
        this.handleSendMessage = this.handleSendMessage.bind(this);
        this.handleStreamChunk = this.handleStreamChunk.bind(this);
        this.handleError = this.handleError.bind(this);
    }

    /**
     * Initialize chat system
     * @param {object} options - Initialization options
     */
    async initialize(options = {}) {
        if (this.isInitialized) {
            console.warn('Chat system already initialized');
            return;
        }

        try {
            console.info('Initializing Een Unity Chat Integration...');

            // Apply configuration updates
            if (options.config) {
                this.config = { ...this.config, ...options.config };
                EenConfig.updateApiConfig(options.config.api || {});
                EenConfig.updateUIConfig(options.config.ui || {});
            }

            // Setup authentication
            if (options.authToken) {
                this.apiClient.setAuthToken(options.authToken);
            }

            // Initialize session
            if (!this.stateManager.sessionId) {
                this.stateManager.initializeSession(options.sessionId);
            }

            // Setup event listeners
            this.setupEventListeners();

            // Test API connection if enabled
            if (this.config.api.AUTH_REQUIRED && !options.skipHealthCheck) {
                await this.performHealthCheck();
            }

            this.isInitialized = true;
            this.eventEmitter.emit('initialized', this);

            console.info('Een Unity Chat Integration initialized successfully');
            
        } catch (error) {
            console.error('Failed to initialize chat system:', error);
            this.eventEmitter.emit('error', error);
            throw error;
        }
    }

    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // UI events
        if (this.ui.container) {
            this.ui.container.addEventListener('een-chat-send', (e) => {
                this.handleSendMessage(e.detail.message);
            });
        }

        // Global keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + Shift + C to toggle chat
            if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'C') {
                e.preventDefault();
                this.toggle();
            }

            // Escape to close chat
            if (e.key === 'Escape' && this.stateManager.isVisible) {
                this.close();
            }
        });

        // Window events
        window.addEventListener('online', () => {
            this.ui.updateConnectionStatus('connected');
            console.info('Connection restored');
        });

        window.addEventListener('offline', () => {
            this.ui.updateConnectionStatus('disconnected');
            console.warn('Connection lost - switching to offline mode');
        });

        // Page visibility
        document.addEventListener('visibilitychange', () => {
            if (document.visibilityState === 'visible' && this.stateManager.isVisible) {
                this.refreshConnection();
            }
        });
    }

    /**
     * Perform health check
     */
    async performHealthCheck() {
        try {
            this.ui.updateConnectionStatus('connecting');
            const health = await this.apiClient.healthCheck();
            this.ui.updateConnectionStatus('connected');
            console.info('Health check passed:', health);
        } catch (error) {
            this.ui.updateConnectionStatus('error');
            console.warn('Health check failed:', error);
            
            if (!this.config.api.ENABLE_OFFLINE_FALLBACK) {
                throw error;
            }
        }
    }

    /**
     * Refresh connection status
     */
    async refreshConnection() {
        if (!this.config.api.AUTH_REQUIRED) {
            return;
        }

        try {
            await this.performHealthCheck();
        } catch (error) {
            console.warn('Connection refresh failed:', error);
        }
    }

    /**
     * Handle sending a message
     * @param {string} message - User message
     */
    async handleSendMessage(message) {
        if (!message?.trim()) {
            return;
        }

        if (this.stateManager.isProcessing) {
            console.warn('Already processing a message');
            return;
        }

        try {
            // Set processing state
            this.stateManager.setProcessing(true);
            this.ui.updateSendButton();
            this.ui.updateTypingIndicator(true);
            this.retryCount = 0;

            // Add user message to UI
            const userMessage = this.ui.addMessage('user', message);
            
            // Prepare request
            const request = this.buildChatRequest(message);

            // Send request
            if (this.config.api.ENABLE_STREAMING) {
                await this.sendStreamingRequest(request);
            } else {
                await this.sendRegularRequest(request);
            }

            this.eventEmitter.emit('message-sent', { message, request });

        } catch (error) {
            await this.handleError(error);
        } finally {
            this.stateManager.setProcessing(false);
            this.ui.updateSendButton();
            this.ui.updateTypingIndicator(false);
        }
    }

    /**
     * Build chat request object
     * @param {string} message - User message
     * @returns {object}
     */
    buildChatRequest(message) {
        return {
            message,
            session_id: this.stateManager.sessionId,
            model: this.config.api.MODEL,
            provider: 'openai', // TODO: Make configurable
            temperature: this.config.api.TEMPERATURE,
            max_tokens: this.config.api.MAX_TOKENS,
            stream: this.config.api.ENABLE_STREAMING,
            consciousness_level: 1.0
        };
    }

    /**
     * Send streaming chat request
     * @param {object} request - Chat request
     */
    async sendStreamingRequest(request) {
        let assistantMessageElement = null;
        let collectedContent = '';

        try {
            await this.apiClient.streamChat(
                request,
                (chunk) => this.handleStreamChunk(chunk, { assistantMessageElement, collectedContent }),
                (error) => this.handleError(error)
            );
        } catch (error) {
            if (this.shouldRetry(error)) {
                await this.retryRequest(() => this.sendStreamingRequest(request));
            } else {
                throw error;
            }
        }
    }

    /**
     * Send regular (non-streaming) chat request
     * @param {object} request - Chat request
     */
    async sendRegularRequest(request) {
        try {
            const response = await this.apiClient.chat(request);
            
            // Add assistant response
            this.ui.addMessage('assistant', response.response, {
                tokens_used: response.tokens_used,
                processing_time: response.processing_time,
                consciousness_alignment: response.consciousness_alignment,
                model: response.model,
                provider: response.provider
            });

        } catch (error) {
            if (this.shouldRetry(error)) {
                await this.retryRequest(() => this.sendRegularRequest(request));
            } else {
                throw error;
            }
        }
    }

    /**
     * Handle streaming chunk
     * @param {object} chunk - Stream chunk
     * @param {object} context - Streaming context
     */
    handleStreamChunk(chunk, context) {
        switch (chunk.type) {
            case 'content':
                // Create assistant message element if not exists
                if (!context.assistantMessageElement) {
                    const message = this.ui.addMessage('assistant', '', {
                        isStreaming: true
                    });
                    context.assistantMessageElement = this.ui.messagesContainer.querySelector(`[data-message-id="${message.id}"]`);
                    context.collectedContent = '';
                }

                // Append content
                context.collectedContent += chunk.data;
                const contentElement = context.assistantMessageElement.querySelector('.message-content');
                if (contentElement) {
                    contentElement.innerHTML = this.ui.formatMessageContent(context.collectedContent);
                }

                this.ui.scrollToBottom();
                break;

            case 'sources':
                // Handle source citations
                this.handleSourceCitations(chunk.data, context.assistantMessageElement);
                break;

            case 'done':
                // Finalize message
                this.finalizeStreamedMessage(chunk, context);
                break;

            case 'error':
                // Handle streaming error
                this.handleStreamError(chunk.data, context);
                break;

            default:
                console.warn('Unknown chunk type:', chunk.type);
        }
    }

    /**
     * Handle source citations
     * @param {array} sources - Source citations
     * @param {HTMLElement} messageElement - Message element
     */
    handleSourceCitations(sources, messageElement) {
        if (!sources?.length || !messageElement) {
            return;
        }

        const citationsContainer = document.createElement('div');
        citationsContainer.className = 'message-citations';
        citationsContainer.innerHTML = '<h4>Sources:</h4>';

        sources.forEach((source, index) => {
            const citation = ChatUtils.createCitationElement({
                text: `Source ${index + 1}`,
                title: source.text || `File: ${source.file_id}`,
                url: source.url || `#${source.file_id}`
            });
            citationsContainer.appendChild(citation);
        });

        messageElement.appendChild(citationsContainer);
    }

    /**
     * Finalize streamed message
     * @param {object} chunk - Final chunk
     * @param {object} context - Streaming context
     */
    finalizeStreamedMessage(chunk, context) {
        if (!context.assistantMessageElement) {
            return;
        }

        // Update message metadata
        const metadataElement = context.assistantMessageElement.querySelector('.message-metadata');
        if (metadataElement) {
            metadataElement.innerHTML = this.ui.renderMessageMetadata({
                tokens_used: chunk.data.tokens_used,
                processing_time: chunk.data.processing_time,
                consciousness_alignment: chunk.data.consciousness_alignment || 1.0
            });
        }

        // Add to state manager
        this.stateManager.addMessage('assistant', context.collectedContent, {
            tokens_used: chunk.data.tokens_used,
            processing_time: chunk.data.processing_time,
            consciousness_alignment: chunk.data.consciousness_alignment,
            model: chunk.model,
            provider: chunk.provider
        });

        // Remove streaming flag
        context.assistantMessageElement.classList.remove('streaming');
        
        this.eventEmitter.emit('message-completed', {
            content: context.collectedContent,
            metadata: chunk.data
        });
    }

    /**
     * Handle streaming error
     * @param {object} errorData - Error data
     * @param {object} context - Streaming context
     */
    handleStreamError(errorData, context) {
        console.error('Streaming error:', errorData);

        if (context.assistantMessageElement) {
            const contentElement = context.assistantMessageElement.querySelector('.message-content');
            if (contentElement) {
                contentElement.innerHTML = ChatUtils.createErrorElement(
                    errorData.message || 'An error occurred while processing your request.',
                    true
                ).outerHTML;
            }
        } else {
            // Add error message
            this.ui.addMessage('assistant', '', {
                isError: true,
                error: errorData
            });
        }
    }

    /**
     * Handle errors with proper user feedback
     * @param {Error} error - Error object
     */
    async handleError(error) {
        console.error('Chat error:', error);

        let userMessage = 'An unexpected error occurred.';
        let canRetry = false;

        if (error instanceof ChatAPIError) {
            userMessage = error.getUserMessage();
            canRetry = error.isRetryable();
        } else if (!navigator.onLine) {
            userMessage = 'You appear to be offline. Please check your connection.';
            canRetry = false;
        }

        // Show error in UI
        const errorElement = ChatUtils.createErrorElement(userMessage, canRetry);
        
        if (canRetry) {
            const retryButton = errorElement.querySelector('.error-retry-btn');
            if (retryButton) {
                retryButton.addEventListener('click', () => {
                    // Retry last message
                    const lastUserMessage = this.stateManager.getChatHistory()
                        .filter(msg => msg.role === 'user')
                        .pop();
                    
                    if (lastUserMessage) {
                        this.handleSendMessage(lastUserMessage.content);
                    }
                });
            }
        }

        // Add error message to chat
        this.ui.messagesContainer.appendChild(errorElement);
        this.ui.scrollToBottom();

        // Use fallback if available
        if (this.config.api.ENABLE_OFFLINE_FALLBACK) {
            this.showFallbackResponse();
        }

        this.eventEmitter.emit('error', error);
    }

    /**
     * Show fallback response
     */
    showFallbackResponse() {
        const lastUserMessage = this.stateManager.getChatHistory()
            .filter(msg => msg.role === 'user')
            .pop();

        if (lastUserMessage) {
            const fallbackResponse = this.apiClient.getMockResponse(lastUserMessage.content);
            
            setTimeout(() => {
                this.ui.addMessage('assistant', fallbackResponse, {
                    isFallback: true,
                    tokens_used: Math.floor(fallbackResponse.split(' ').length * 1.3),
                    processing_time: 0.5
                });
            }, 1000);
        }
    }

    /**
     * Check if error should be retried
     * @param {Error} error - Error to check
     * @returns {boolean}
     */
    shouldRetry(error) {
        return this.retryCount < this.maxRetries && 
               (error instanceof ChatAPIError && error.isRetryable());
    }

    /**
     * Retry request with exponential backoff
     * @param {Function} requestFunction - Function to retry
     */
    async retryRequest(requestFunction) {
        this.retryCount++;
        const delay = this.config.api.RETRY_DELAY * Math.pow(2, this.retryCount - 1);
        
        console.info(`Retrying request (attempt ${this.retryCount}/${this.maxRetries}) in ${delay}ms...`);
        
        await new Promise(resolve => setTimeout(resolve, delay));
        
        try {
            await requestFunction();
        } catch (error) {
            if (this.shouldRetry(error)) {
                await this.retryRequest(requestFunction);
            } else {
                throw error;
            }
        }
    }

    /**
     * Open chat interface
     */
    open() {
        if (!this.isInitialized) {
            console.warn('Chat not initialized. Call initialize() first.');
            return;
        }

        this.ui.show();
        this.eventEmitter.emit('opened');
    }

    /**
     * Close chat interface
     */
    close() {
        this.ui.hide();
        this.stateManager.cancelStream();
        this.eventEmitter.emit('closed');
    }

    /**
     * Toggle chat interface
     */
    toggle() {
        if (this.stateManager.isVisible) {
            this.close();
        } else {
            this.open();
        }
    }

    /**
     * Clear chat history
     */
    clearHistory() {
        this.ui.clearMessages();
        this.eventEmitter.emit('history-cleared');
    }

    /**
     * Export chat history
     * @param {string} format - Export format
     * @returns {string}
     */
    exportHistory(format = 'markdown') {
        return this.stateManager.exportHistory(format);
    }

    /**
     * Get session statistics
     * @returns {object}
     */
    getSessionStats() {
        return this.stateManager.getSessionStats();
    }

    /**
     * Update configuration
     * @param {object} newConfig - Configuration updates
     */
    updateConfig(newConfig) {
        this.config = { ...this.config, ...newConfig };
        
        if (newConfig.api) {
            EenConfig.updateApiConfig(newConfig.api);
        }
        
        if (newConfig.ui) {
            EenConfig.updateUIConfig(newConfig.ui);
            this.stateManager.updatePreferences(newConfig.ui);
        }

        this.eventEmitter.emit('config-updated', newConfig);
    }

    /**
     * Set authentication token
     * @param {string} token - Bearer token
     */
    setAuthToken(token) {
        this.apiClient.setAuthToken(token);
    }

    /**
     * Add event listener
     * @param {string} event - Event name
     * @param {function} callback - Event callback
     */
    on(event, callback) {
        this.eventEmitter.on(event, callback);
    }

    /**
     * Remove event listener
     * @param {string} event - Event name
     * @param {function} callback - Event callback
     */
    off(event, callback) {
        this.eventEmitter.off(event, callback);
    }

    /**
     * Destroy chat integration
     */
    destroy() {
        this.close();
        this.stateManager.cancelStream();
        
        if (this.ui.container) {
            this.ui.container.remove();
        }

        this.eventEmitter.emit('destroyed');
        this.isInitialized = false;
    }
}

// Global instance for backward compatibility
let globalChatInstance = null;

/**
 * Initialize global chat instance
 * @param {object} options - Initialization options
 * @returns {EenChatIntegration}
 */
export function initialize(options = {}) {
    if (!globalChatInstance) {
        globalChatInstance = new EenChatIntegration(options);
    }
    
    return globalChatInstance.initialize(options).then(() => globalChatInstance);
}

/**
 * Get global chat instance
 * @returns {EenChatIntegration|null}
 */
export function getInstance() {
    return globalChatInstance;
}

// Auto-initialize if requested
if (EenConfig.ui.AUTO_INITIALIZE && typeof window !== 'undefined') {
    window.addEventListener('DOMContentLoaded', () => {
        initialize().catch(error => {
            console.error('Auto-initialization failed:', error);
        });
    });
}

// Make available globally
if (typeof window !== 'undefined') {
    window.EenChatIntegration = EenChatIntegration;
    window.EenChat = { initialize, getInstance };
}

export default EenChatIntegration;