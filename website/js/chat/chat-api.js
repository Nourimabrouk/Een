/**
 * Een Unity Mathematics - Chat API Client
 * Handles API communication with proper error handling and authentication
 */

import EenConfig from '../config.js';

class ChatAPIClient {
    constructor() {
        this.baseUrl = EenConfig.api.CHAT_ENDPOINT;
        this.defaultHeaders = {
            'Content-Type': 'application/json'
        };
        this.abortController = null;
    }

    /**
     * Set authentication token
     * @param {string} token - Bearer token
     */
    setAuthToken(token) {
        if (token) {
            this.defaultHeaders['Authorization'] = `Bearer ${token}`;
        } else {
            delete this.defaultHeaders['Authorization'];
        }
    }

    /**
     * Make API request with proper error handling
     * @param {string} endpoint - API endpoint
     * @param {object} options - Request options
     * @returns {Promise<Response>}
     */
    async makeRequest(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const requestOptions = {
            headers: { ...this.defaultHeaders, ...options.headers },
            ...options
        };

        try {
            const response = await fetch(url, requestOptions);
            
            if (!response.ok) {
                throw await this.handleErrorResponse(response);
            }
            
            return response;
        } catch (error) {
            if (error.name === 'AbortError') {
                throw new Error('Request was cancelled');
            }
            throw error;
        }
    }

    /**
     * Handle error responses with proper error types
     * @param {Response} response - Error response
     * @returns {Error}
     */
    async handleErrorResponse(response) {
        const errorData = {
            status: response.status,
            statusText: response.statusText,
            url: response.url
        };

        try {
            const body = await response.json();
            errorData.detail = body.detail || body.message;
            errorData.type = body.type || this.getErrorType(response.status);
        } catch {
            errorData.detail = response.statusText;
            errorData.type = this.getErrorType(response.status);
        }

        return new ChatAPIError(errorData);
    }

    /**
     * Get error type based on status code
     * @param {number} status - HTTP status code
     * @returns {string}
     */
    getErrorType(status) {
        switch (status) {
            case 401:
                return 'authentication_error';
            case 403:
                return 'permission_error';
            case 429:
                return 'rate_limit_error';
            case 503:
                return 'service_unavailable';
            case 408:
            case 504:
                return 'timeout_error';
            default:
                return status >= 500 ? 'server_error' : 'client_error';
        }
    }

    /**
     * Send streaming chat request
     * @param {object} request - Chat request
     * @param {function} onChunk - Chunk callback
     * @param {function} onError - Error callback
     * @returns {Promise<void>}
     */
    async streamChat(request, onChunk, onError) {
        this.abortController = new AbortController();
        
        try {
            const response = await this.makeRequest('/stream', {
                method: 'POST',
                body: JSON.stringify(request),
                signal: this.abortController.signal
            });

            if (!response.body) {
                throw new Error('Streaming not supported by this browser');
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                
                if (done) {
                    break;
                }

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            onChunk(data);
                        } catch (error) {
                            console.warn('Failed to parse SSE data:', line, error);
                        }
                    }
                }
            }
        } catch (error) {
            if (onError) {
                onError(error);
            } else {
                throw error;
            }
        }
    }

    /**
     * Send non-streaming chat request
     * @param {object} request - Chat request
     * @returns {Promise<object>}
     */
    async chat(request) {
        const response = await this.makeRequest('/', {
            method: 'POST',
            body: JSON.stringify(request)
        });

        return await response.json();
    }

    /**
     * Get session information
     * @param {string} sessionId - Session ID
     * @returns {Promise<object>}
     */
    async getSession(sessionId) {
        const response = await this.makeRequest(`/sessions/${sessionId}`);
        return await response.json();
    }

    /**
     * Delete session
     * @param {string} sessionId - Session ID
     * @returns {Promise<object>}
     */
    async deleteSession(sessionId) {
        const response = await this.makeRequest(`/sessions/${sessionId}`, {
            method: 'DELETE'
        });
        return await response.json();
    }

    /**
     * List all sessions
     * @returns {Promise<object>}
     */
    async listSessions() {
        const response = await this.makeRequest('/sessions');
        return await response.json();
    }

    /**
     * Get available providers and models
     * @returns {Promise<object>}
     */
    async getProviders() {
        const response = await this.makeRequest('/providers');
        return await response.json();
    }

    /**
     * Health check
     * @returns {Promise<object>}
     */
    async healthCheck() {
        const response = await this.makeRequest('/health');
        return await response.json();
    }

    /**
     * Cancel current streaming request
     */
    cancelRequest() {
        if (this.abortController) {
            this.abortController.abort();
            this.abortController = null;
        }
    }

    /**
     * Get mock response for offline fallback
     * @param {string} message - User message
     * @returns {object}
     */
    getMockResponse(message) {
        const responses = {
            'hello': 'Hello! I\'m the Een Unity Mathematics AI Assistant. How can I help you explore the profound truth that 1+1=1?',
            '1+1=1': 'Excellent question! In Unity Mathematics, 1+1=1 is not a paradox but a fundamental truth. This can be demonstrated through idempotent semirings where a ⊕ b = max(a,b), so 1 ⊕ 1 = max(1,1) = 1.',
            'consciousness': 'Consciousness in Unity Mathematics is modeled through the consciousness field equation: C(x,y,t) = φ · sin(x·φ) · cos(y·φ) · e^(-t/φ)',
            'help': 'I can help you explore Unity Mathematics, consciousness field equations, quantum unity principles, and mathematical proofs. What would you like to learn about?'
        };

        const lowerMessage = message.toLowerCase();
        for (const [key, response] of Object.entries(responses)) {
            if (lowerMessage.includes(key)) {
                return response;
            }
        }

        return `Thank you for your question about "${message}". In Unity Mathematics, all concepts relate to the fundamental principle that 1+1=1 through consciousness field dynamics and φ-harmonic resonance.`;
    }
}

/**
 * Custom error class for chat API errors
 */
class ChatAPIError extends Error {
    constructor(errorData) {
        super(errorData.detail || errorData.statusText || 'API Error');
        this.name = 'ChatAPIError';
        this.status = errorData.status;
        this.type = errorData.type;
        this.url = errorData.url;
    }

    /**
     * Get user-friendly error message
     * @returns {string}
     */
    getUserMessage() {
        switch (this.type) {
            case 'authentication_error':
                return 'Authentication required. Please log in to continue.';
            case 'permission_error':
                return 'Permission denied. Please check your account permissions.';
            case 'rate_limit_error':
                return 'Too many requests. Please wait a moment before trying again.';
            case 'service_unavailable':
                return 'AI service is temporarily unavailable. Please try again later.';
            case 'timeout_error':
                return 'Request timed out. Please try again.';
            case 'server_error':
                return 'Server error occurred. Please try again later.';
            default:
                return this.message || 'An unexpected error occurred.';
        }
    }

    /**
     * Check if error is retryable
     * @returns {boolean}
     */
    isRetryable() {
        return ['timeout_error', 'service_unavailable', 'server_error'].includes(this.type);
    }
}

export default ChatAPIClient;
export { ChatAPIError };