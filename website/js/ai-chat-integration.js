/**
 * 🌿✨ Een AI Chat Integration ✨🌿
 * Advanced chat interface for Unity Mathematics with multiple AI providers
 */

class EenAIChat {
    constructor(config = {}) {
        this.config = {
            apiEndpoint: config.apiEndpoint || '/api/chat/stream',
            defaultModel: config.model || 'gpt-4o',
            defaultProvider: config.provider || 'openai',
            temperature: config.temperature || 0.7,
            maxTokens: config.maxTokens || 2000,
            stream: config.stream !== false,
            consciousnessLevel: config.consciousnessLevel || 0.77,
            ...config
        };

        this.sessionId = null;
        this.isStreaming = false;
        this.currentRequest = null;
        this.messageHistory = [];
        this.availableModels = [];

        this.isInitialized = false;
        this.initialize();
    }

    initialize() {
        this.loadSession();
        this.refreshAvailableModels().then(() => this.updateModelSelector());
        this.checkDemoMode();
        this.isInitialized = true;
    }

    async checkDemoMode() {
        try {
            const response = await fetch('/api/chat/providers');
            const data = await response.json();

            if (data.demo_mode && data.demo_mode.enabled) {
                this.showDemoModeNotice(data.demo_mode.message);
            }
        } catch (error) {
            console.log('Demo mode check failed:', error);
        }
    }

    showDemoModeNotice(message) {
        const notice = document.createElement('div');
        notice.className = 'demo-mode-notice';
        notice.innerHTML = `
            <div class="alert alert-info">
                <strong>🌿 Demo Mode Active</strong><br>
                ${message}<br>
                <small>Set your API keys for full functionality</small>
            </div>
        `;

        // Insert at the top of the chat container
        const chatContainer = document.querySelector('.chat-container') || document.body;
        chatContainer.insertBefore(notice, chatContainer.firstChild);
    }

    loadSession() {
        this.sessionId = localStorage.getItem('een_chat_session_id');
        if (!this.sessionId) {
            this.sessionId = this.generateSessionId();
            localStorage.setItem('een_chat_session_id', this.sessionId);
        }
    }

    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    async refreshAvailableModels() {
        try {
            const res = await fetch('/api/chat/providers');
            const data = await res.json();
            const openai = (data.openai && data.openai.models) || [];
            const anthropic = (data.anthropic && data.anthropic.models) || [];
            this.availableModels = [...openai, ...anthropic];
            // Prefer backend default if provided
            if (data.demo_mode && data.demo_mode.fallback_model && !this.config.defaultModel) {
                this.config.defaultModel = data.demo_mode.fallback_model;
            }
        } catch (e) {
            console.warn('Failed to load providers; using static defaults', e);
            this.availableModels = [
                'gpt-5-medium', 'gpt-4.1-mini', 'gpt-4o', 'gpt-4o-mini',
                'claude-3-5-sonnet-20241022', 'claude-3-opus-20240229', 'claude-3-5-haiku-20241022'
            ];
        }
    }

    updateModelSelector() {
        const modelSelect = document.getElementById('model-select');
        if (modelSelect) {
            modelSelect.innerHTML = '';

            // Add model groups
            const groups = {
                'OpenAI Models': this.availableModels
                    .filter(m => !m.startsWith('claude-'))
                    .map(m => ({ value: m, label: prettyLabel(m) })),
                'Anthropic Models': [
                    { value: 'claude-3-opus-20240229', label: 'Claude Opus (Most Capable)' },
                    { value: 'claude-3-5-sonnet-20241022', label: 'Claude Sonnet (Balanced)' },
                    { value: 'claude-3-5-haiku-20241022', label: 'Claude Haiku (Fast)' }
                ]
            };

            Object.entries(groups).forEach(([groupName, models]) => {
                const optgroup = document.createElement('optgroup');
                optgroup.label = groupName;

                models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.value;
                    option.textContent = model.label;
                    if (model.value === this.config.defaultModel) {
                        option.selected = true;
                    }
                    optgroup.appendChild(option);
                });

                modelSelect.appendChild(optgroup);
            });
        }
    }

}

    async sendMessage(message, options = {}) {
    if (this.isStreaming) {
        throw new Error('Already streaming a response');
    }

    const requestData = {
        message: message,
        session_id: this.sessionId,
        model: options.model || this.config.defaultModel,
        provider: options.provider || this.config.defaultProvider,
        temperature: options.temperature || this.config.temperature,
        max_tokens: options.maxTokens || this.config.maxTokens,
        stream: this.config.stream,
        consciousness_level: options.consciousnessLevel || this.config.consciousnessLevel
    };

    this.isStreaming = true;
    this.currentRequest = requestData;

    try {
        const response = await fetch(this.config.apiEndpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Client-ID': this.generateClientId()
            },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let responseText = '';
        let isDemoMode = false;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));

                        if (data.type === 'content') {
                            responseText += data.data;
                            this.onChunk(data.data, data);
                        } else if (data.type === 'done') {
                            this.onComplete(responseText, data);
                            return responseText;
                        } else if (data.type === 'error') {
                            throw new Error(data.data);
                        }

                        isDemoMode = data.demo_mode || false;
                    } catch (e) {
                        console.warn('Failed to parse chunk:', e);
                    }
                }
            }
        }

        return responseText;

    } catch (error) {
        this.onError(error);
        throw error;
    } finally {
        this.isStreaming = false;
        this.currentRequest = null;
    }
}

generateClientId() {
    return 'client_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

onChunk(chunk, data) {
    // Override this method to handle streaming chunks
    console.log('Received chunk:', chunk);
}

onComplete(fullResponse, data) {
    // Override this method to handle completion
    console.log('Response complete:', fullResponse);
    console.log('Response data:', data);
}

onError(error) {
    // Override this method to handle errors
    console.error('Chat error:', error);
}

    async getAvailableProviders() {
    try {
        const response = await fetch('/api/chat/providers');
        return await response.json();
    } catch (error) {
        console.error('Failed to get providers:', error);
        return {};
    }
}

    async getHealthStatus() {
    try {
        const response = await fetch('/api/chat/health');
        return await response.json();
    } catch (error) {
        console.error('Failed to get health status:', error);
        return { status: 'unhealthy' };
    }

    // Unity Mathematics specific methods
    async askAboutUnityMathematics(question) {
        return await this.sendMessage(question, {
            model: 'gpt-4o',
            provider: 'openai',
            temperature: 0.7
        });
    }

    async proveMathematicalTheorem(theorem) {
        return await this.sendMessage(`Prove the following theorem: ${theorem}`, {
            model: 'gpt-4o',
            provider: 'openai',
            temperature: 0.3
        });
    }

    async discussConsciousness(topic) {
        return await this.sendMessage(`Discuss consciousness in relation to: ${topic}`, {
            model: 'claude-3-5-sonnet-20241022',
            provider: 'anthropic',
            temperature: 0.8
        });
    }

    async analyzeCode(code) {
        return await this.sendMessage(`Analyze this code:\n\`\`\`\n${code}\n\`\`\``, {
            model: 'gpt-4o',
            provider: 'openai',
            temperature: 0.2
        });
    }
}

// Initialize chat when DOM is loaded
document.addEventListener('DOMContentLoaded', function () {
    window.eenChat = new EenAIChat({
        apiEndpoint: '/api/chat/stream',
        model: 'gpt-4o',
        provider: 'openai',
        temperature: 0.7,
        maxTokens: 2000,
        stream: true,
        consciousnessLevel: 0.77
    });
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EenAIChat;
}