/**
 * Een Repository AI Chat Widget
 * =============================
 * 
 * Interactive chat interface for the Een Unity Mathematics repository AI assistant.
 * Integrates seamlessly with existing website design and provides SSE streaming responses.
 * 
 * Features:
 * - Server-Sent Events (SSE) streaming
 * - Session persistence with localStorage
 * - Responsive design matching Een website aesthetics
 * - Mathematical equation rendering with KaTeX
 * - Source citation display with file references
 * - Rate limit handling and error recovery
 * 
 * Author: Claude (3000 ELO AGI)
 */

class EenChatWidget {
    constructor(config = {}) {
        this.config = {
            apiEndpoint: config.apiEndpoint || '/api/agents/chat',
            maxMessages: config.maxMessages || 50,
            showTypingIndicator: config.showTypingIndicator !== false,
            enableMarkdown: config.enableMarkdown !== false,
            enableMath: config.enableMath !== false,
            sessionTimeout: config.sessionTimeout || 24 * 60 * 60 * 1000, // 24 hours
            ...config
        };

        this.sessionId = this.getOrCreateSessionId();
        this.messages = this.loadMessages();
        this.isStreaming = false;
        this.eventSource = null;
        this.currentStreamMessage = null;

        this.init();
    }

    init() {
        this.createChatWidget();
        this.attachEventListeners();
        this.renderMessages();

        // Auto-expand if there are existing messages
        if (this.messages.length > 0) {
            this.toggleChat(true);
        }

        console.log('Een Chat Widget initialized', {
            sessionId: this.sessionId,
            messageCount: this.messages.length
        });
    }

    getOrCreateSessionId() {
        const stored = localStorage.getItem('een_chat_session_id');
        const sessionTime = localStorage.getItem('een_chat_session_time');

        // Check if session is expired
        if (stored && sessionTime) {
            const age = Date.now() - parseInt(sessionTime);
            if (age < this.config.sessionTimeout) {
                return stored;
            }
        }

        // Create new session
        const newSessionId = 'een_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        localStorage.setItem('een_chat_session_id', newSessionId);
        localStorage.setItem('een_chat_session_time', Date.now().toString());

        return newSessionId;
    }

    loadMessages() {
        try {
            const stored = localStorage.getItem(`een_chat_messages_${this.sessionId}`);
            return stored ? JSON.parse(stored) : [];
        } catch (e) {
            console.warn('Failed to load chat messages:', e);
            return [];
        }
    }

    saveMessages() {
        try {
            // Keep only last N messages to prevent storage overflow
            const messagesToSave = this.messages.slice(-this.config.maxMessages);
            localStorage.setItem(`een_chat_messages_${this.sessionId}`, JSON.stringify(messagesToSave));
        } catch (e) {
            console.warn('Failed to save chat messages:', e);
        }
    }

    createChatWidget() {
        // Create widget container
        const widgetContainer = document.createElement('div');
        widgetContainer.id = 'een-chat-widget';
        widgetContainer.innerHTML = `
            <div class="chat-toggle" id="chat-toggle">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                </svg>
                <span class="chat-badge" id="chat-badge" style="display: none;">1</span>
            </div>
            
            <div class="chat-window" id="chat-window" style="display: none;">
                <div class="chat-header">
                    <div class="chat-title">
                        <div class="chat-avatar">
                            <span class="phi-symbol">φ</span>
                        </div>
                        <div>
                            <div class="assistant-name">Een Repository AI</div>
                            <div class="assistant-subtitle">Unity Mathematics Expert</div>
                        </div>
                    </div>
                    <button class="chat-close" id="chat-close">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <line x1="18" y1="6" x2="6" y2="18"></line>
                            <line x1="6" y1="6" x2="18" y2="18"></line>
                        </svg>
                    </button>
                </div>
                
                <div class="chat-messages" id="chat-messages">
                    <div class="welcome-message">
                        <div class="phi-pattern"></div>
                        <h3>Welcome to Een Unity Mathematics</h3>
                        <p>I'm your AI assistant for exploring the profound concept that <strong>1+1=1</strong> through φ-harmonic consciousness mathematics, quantum unity frameworks, and transcendental proof systems.</p>
                        <div class="example-questions">
                            <div class="example-question" data-question="What is the φ-harmonic consciousness framework?">
                                What is the φ-harmonic consciousness framework?
                            </div>
                            <div class="example-question" data-question="How do you prove that 1+1=1 mathematically?">
                                How do you prove that 1+1=1 mathematically?
                            </div>
                            <div class="example-question" data-question="Show me quantum unity visualization examples">
                                Show me quantum unity visualization examples
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="chat-input-container">
                    <div class="typing-indicator" id="typing-indicator" style="display: none;">
                        <div class="typing-dots">
                            <span></span>
                            <span></span>
                            <span></span>
                        </div>
                        <span class="typing-text">Een AI is thinking about unity mathematics...</span>
                    </div>
                    
                    <div class="chat-input-wrapper">
                        <textarea 
                            id="chat-input" 
                            placeholder="Ask about Unity Mathematics, φ-harmonic systems, or quantum consciousness..."
                            rows="1"
                        ></textarea>
                        <button id="chat-send" class="chat-send-btn" disabled>
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <line x1="22" y1="2" x2="11" y2="13"></line>
                                <polygon points="22,2 15,22 11,13 2,9"></polygon>
                            </svg>
                        </button>
                    </div>
                    
                    <div class="chat-footer">
                        <div class="session-info">
                            Session: <span class="session-id">${this.sessionId.substring(4, 12)}...</span>
                        </div>
                        <div class="powered-by">
                            Powered by <a href="https://openai.com" target="_blank">OpenAI</a> & φ-Harmonic Mathematics
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Inject CSS
        this.injectStyles();

        // Add to page
        document.body.appendChild(widgetContainer);

        // Store references
        this.elements = {
            toggle: document.getElementById('chat-toggle'),
            window: document.getElementById('chat-window'),
            close: document.getElementById('chat-close'),
            messages: document.getElementById('chat-messages'),
            input: document.getElementById('chat-input'),
            send: document.getElementById('chat-send'),
            typingIndicator: document.getElementById('typing-indicator'),
            badge: document.getElementById('chat-badge')
        };
    }

    injectStyles() {
        const styles = `
        <style id="een-chat-styles">
        #een-chat-widget {
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            z-index: 9999;
            font-family: var(--font-sans, 'Inter', sans-serif);
        }
        
        .chat-toggle {
            width: 60px;
            height: 60px;
            background: var(--gradient-phi, linear-gradient(135deg, #0F7B8A 0%, #4A9BAE 100%));
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: var(--shadow-lg, 0 10px 25px rgba(0,0,0,0.15));
            transition: all var(--transition-smooth, 300ms ease);
            position: relative;
            color: white;
        }
        
        .chat-toggle:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-xl, 0 20px 40px rgba(0,0,0,0.2));
        }
        
        .chat-badge {
            position: absolute;
            top: -5px;
            right: -5px;
            background: #ef4444;
            color: white;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: 600;
        }
        
        .chat-window {
            position: absolute;
            bottom: 80px;
            right: 0;
            width: 400px;
            height: 600px;
            background: var(--bg-primary, white);
            border-radius: var(--radius-2xl, 1.5rem);
            box-shadow: var(--shadow-xl, 0 20px 40px rgba(0,0,0,0.15));
            border: 1px solid var(--border-color, #e2e8f0);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            animation: slideInUp 0.3s ease-out;
        }
        
        @keyframes slideInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .chat-header {
            padding: 1.5rem;
            background: var(--gradient-phi, linear-gradient(135deg, #0F7B8A 0%, #4A9BAE 100%));
            color: white;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .chat-title {
            display: flex;
            align-items: center;
            gap: 1rem;
        }
        
        .chat-avatar {
            width: 40px;
            height: 40px;
            background: rgba(255,255,255,0.2);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: var(--font-serif, serif);
            font-size: 1.2rem;
            font-weight: 600;
        }
        
        .phi-symbol {
            background: linear-gradient(45deg, #FFD700, #FFA500);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .assistant-name {
            font-weight: 600;
            font-size: 1rem;
        }
        
        .assistant-subtitle {
            font-size: 0.875rem;
            opacity: 0.9;
        }
        
        .chat-close {
            background: none;
            border: none;
            color: white;
            cursor: pointer;
            padding: 0.5rem;
            border-radius: var(--radius-md, 0.5rem);
            transition: background-color var(--transition-fast, 150ms ease);
        }
        
        .chat-close:hover {
            background: rgba(255,255,255,0.1);
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
            scrollbar-width: thin;
            scrollbar-color: var(--border-color, #e2e8f0) transparent;
        }
        
        .chat-messages::-webkit-scrollbar {
            width: 6px;
        }
        
        .chat-messages::-webkit-scrollbar-track {
            background: transparent;
        }
        
        .chat-messages::-webkit-scrollbar-thumb {
            background: var(--border-color, #e2e8f0);
            border-radius: 3px;
        }
        
        .welcome-message {
            text-align: center;
            padding: 2rem 1rem;
            border-radius: var(--radius-lg, 0.75rem);
            background: var(--bg-secondary, #f7fafc);
            margin-bottom: 1rem;
            position: relative;
            overflow: hidden;
        }
        
        .phi-pattern {
            position: absolute;
            top: -20px;
            right: -20px;
            width: 80px;
            height: 80px;
            background: radial-gradient(circle, rgba(15,123,138,0.1) 0%, transparent 70%);
            border-radius: 50%;
        }
        
        .welcome-message h3 {
            color: var(--text-primary, #2d3748);
            margin-bottom: 0.5rem;
            font-size: 1.25rem;
        }
        
        .welcome-message p {
            color: var(--text-secondary, #718096);
            margin-bottom: 1.5rem;
            line-height: 1.6;
        }
        
        .example-questions {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        
        .example-question {
            background: white;
            border: 1px solid var(--border-color, #e2e8f0);
            border-radius: var(--radius-md, 0.5rem);
            padding: 0.75rem 1rem;
            cursor: pointer;
            transition: all var(--transition-fast, 150ms ease);
            font-size: 0.875rem;
            text-align: left;
        }
        
        .example-question:hover {
            border-color: var(--accent-color, #0F7B8A);
            background: rgba(15,123,138,0.05);
        }
        
        .message {
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }
        
        .message.user {
            align-items: flex-end;
        }
        
        .message.assistant {
            align-items: flex-start;
        }
        
        .message-bubble {
            max-width: 85%;
            padding: 1rem;
            border-radius: var(--radius-lg, 0.75rem);
            font-size: 0.9rem;
            line-height: 1.5;
        }
        
        .message.user .message-bubble {
            background: var(--gradient-phi, linear-gradient(135deg, #0F7B8A 0%, #4A9BAE 100%));
            color: white;
            border-bottom-right-radius: var(--radius-sm, 0.375rem);
        }
        
        .message.assistant .message-bubble {
            background: var(--bg-secondary, #f7fafc);
            color: var(--text-primary, #2d3748);
            border: 1px solid var(--border-color, #e2e8f0);
            border-bottom-left-radius: var(--radius-sm, 0.375rem);
        }
        
        .message-sources {
            margin-top: 0.5rem;
            padding: 0.75rem;
            background: rgba(15,123,138,0.05);
            border-left: 3px solid var(--accent-color, #0F7B8A);
            border-radius: var(--radius-md, 0.5rem);
            font-size: 0.8rem;
        }
        
        .message-sources summary {
            font-weight: 600;
            cursor: pointer;
            color: var(--accent-color, #0F7B8A);
        }
        
        .source-list {
            margin-top: 0.5rem;
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }
        
        .source-item {
            background: white;
            padding: 0.5rem;
            border-radius: var(--radius-sm, 0.375rem);
            border: 1px solid var(--border-subtle, #f7fafc);
        }
        
        .source-path {
            font-family: var(--font-mono, monospace);
            font-size: 0.75rem;
            color: var(--accent-color, #0F7B8A);
            word-break: break-all;
        }
        
        .typing-indicator {
            padding: 1rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            background: var(--bg-secondary, #f7fafc);
            border-top: 1px solid var(--border-color, #e2e8f0);
        }
        
        .typing-dots {
            display: flex;
            gap: 0.25rem;
        }
        
        .typing-dots span {
            width: 6px;
            height: 6px;
            background: var(--accent-color, #0F7B8A);
            border-radius: 50%;
            animation: typing 1.4s infinite ease-in-out;
        }
        
        .typing-dots span:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .typing-dots span:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        @keyframes typing {
            0%, 80%, 100% {
                transform: scale(0.8);
                opacity: 0.5;
            }
            40% {
                transform: scale(1);
                opacity: 1;
            }
        }
        
        .typing-text {
            font-size: 0.875rem;
            color: var(--text-secondary, #718096);
            font-style: italic;
        }
        
        .chat-input-container {
            border-top: 1px solid var(--border-color, #e2e8f0);
            padding: 1rem;
        }
        
        .chat-input-wrapper {
            display: flex;
            align-items: flex-end;
            gap: 0.75rem;
            margin-bottom: 0.75rem;
        }
        
        #chat-input {
            flex: 1;
            border: 1px solid var(--border-color, #e2e8f0);
            border-radius: var(--radius-lg, 0.75rem);
            padding: 0.75rem 1rem;
            font-family: inherit;
            font-size: 0.9rem;
            resize: none;
            outline: none;
            transition: all var(--transition-fast, 150ms ease);
            min-height: 44px;
            max-height: 120px;
        }
        
        #chat-input:focus {
            border-color: var(--accent-color, #0F7B8A);
            box-shadow: 0 0 0 3px rgba(15,123,138,0.1);
        }
        
        .chat-send-btn {
            width: 44px;
            height: 44px;
            background: var(--gradient-phi, linear-gradient(135deg, #0F7B8A 0%, #4A9BAE 100%));
            border: none;
            border-radius: 50%;
            color: white;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all var(--transition-fast, 150ms ease);
            flex-shrink: 0;
        }
        
        .chat-send-btn:disabled {
            background: var(--border-color, #e2e8f0);
            cursor: not-allowed;
        }
        
        .chat-send-btn:not(:disabled):hover {
            transform: translateY(-1px);
            box-shadow: var(--shadow-md, 0 4px 12px rgba(0,0,0,0.15));
        }
        
        .chat-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.75rem;
            color: var(--text-muted, #a0aec0);
        }
        
        .session-info {
            font-family: var(--font-mono, monospace);
        }
        
        .powered-by a {
            color: var(--accent-color, #0F7B8A);
            text-decoration: none;
        }
        
        .powered-by a:hover {
            text-decoration: underline;
        }
        
        /* Mathematical equation styling */
        .katex {
            font-size: 1em;
        }
        
        .katex-display {
            margin: 1rem 0;
        }
        
        /* Responsive design */
        @media (max-width: 480px) {
            #een-chat-widget {
                bottom: 1rem;
                right: 1rem;
                left: 1rem;
            }
            
            .chat-window {
                width: 100%;
                height: 70vh;
                max-height: 500px;
                bottom: 80px;
                right: 0;
                left: 0;
            }
            
            .chat-toggle {
                position: fixed;
                bottom: 1rem;
                right: 1rem;
            }
        }
        </style>
        `;

        document.head.insertAdjacentHTML('beforeend', styles);
    }

    attachEventListeners() {
        // Toggle chat window
        this.elements.toggle.addEventListener('click', () => this.toggleChat());
        this.elements.close.addEventListener('click', () => this.toggleChat(false));

        // Send message
        this.elements.send.addEventListener('click', () => this.sendMessage());

        // Input handling
        this.elements.input.addEventListener('input', () => this.handleInputChange());
        this.elements.input.addEventListener('keydown', (e) => this.handleKeyDown(e));

        // Example questions
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('example-question')) {
                const question = e.target.dataset.question;
                this.elements.input.value = question;
                this.handleInputChange();
                this.sendMessage();
            }
        });

        // Auto-resize textarea
        this.elements.input.addEventListener('input', () => {
            this.elements.input.style.height = 'auto';
            this.elements.input.style.height = Math.min(this.elements.input.scrollHeight, 120) + 'px';
        });
    }

    toggleChat(show = null) {
        const isVisible = this.elements.window.style.display !== 'none';
        const shouldShow = show !== null ? show : !isVisible;

        this.elements.window.style.display = shouldShow ? 'block' : 'none';

        if (shouldShow) {
            this.elements.input.focus();
            this.scrollToBottom();
            this.elements.badge.style.display = 'none';
        }
    }

    handleInputChange() {
        const hasText = this.elements.input.value.trim().length > 0;
        this.elements.send.disabled = !hasText || this.isStreaming;
    }

    handleKeyDown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!this.elements.send.disabled) {
                this.sendMessage();
            }
        }
    }

    async sendMessage() {
        const message = this.elements.input.value.trim();
        if (!message || this.isStreaming) return;

        // Add user message
        this.addMessage({
            role: 'user',
            content: message,
            timestamp: Date.now()
        });

        // Clear input
        this.elements.input.value = '';
        this.elements.input.style.height = 'auto';
        this.handleInputChange();

        // Show typing indicator
        this.showTypingIndicator();

        // Start streaming response
        await this.streamResponse(message);
    }

    async streamResponse(message) {
        this.isStreaming = true;

        try {
            // Create assistant message placeholder
            this.currentStreamMessage = {
                role: 'assistant',
                content: '',
                sources: [],
                timestamp: Date.now()
            };

            const messageElement = this.addMessage(this.currentStreamMessage, true);
            const bubbleElement = messageElement.querySelector('.message-bubble');

            // Setup EventSource
            const requestBody = {
                message: message,
                session_id: this.sessionId
            };

            const response = await fetch(this.config.apiEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'text/event-stream'
                },
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            this.handleStreamChunk(data, bubbleElement, messageElement);
                        } catch (e) {
                            console.warn('Failed to parse SSE data:', e);
                        }
                    }
                }
            }

        } catch (error) {
            console.error('Stream error:', error);
            this.handleStreamError(error);
        } finally {
            this.hideTypingIndicator();
            this.isStreaming = false;
            this.handleInputChange();

            // Save messages
            this.saveMessages();
        }
    }

    handleStreamChunk(data, bubbleElement, messageElement) {
        switch (data.type) {
            case 'content':
                this.currentStreamMessage.content += data.data;
                bubbleElement.innerHTML = this.formatMessage(this.currentStreamMessage.content);
                this.scrollToBottom();
                break;

            case 'sources':
                this.currentStreamMessage.sources = data.data;
                this.updateMessageSources(messageElement, data.data);
                break;

            case 'done':
                // Final processing
                this.currentStreamMessage.processingTime = data.data.processing_time;
                this.currentStreamMessage.tokensUsed = data.data.tokens_used;
                console.log('Stream completed:', data.data);
                break;

            case 'error':
                this.handleStreamError(new Error(data.data.message));
                break;
        }
    }

    handleStreamError(error) {
        if (this.currentStreamMessage) {
            this.currentStreamMessage.content = `❌ Error: ${error.message}\n\nPlease try again or contact support if the issue persists.`;
            const bubbleElement = document.querySelector('.message:last-child .message-bubble');
            if (bubbleElement) {
                bubbleElement.innerHTML = this.formatMessage(this.currentStreamMessage.content);
            }
        }
    }

    showTypingIndicator() {
        if (this.config.showTypingIndicator) {
            this.elements.typingIndicator.style.display = 'flex';
            this.scrollToBottom();
        }
    }

    hideTypingIndicator() {
        this.elements.typingIndicator.style.display = 'none';
    }

    addMessage(message, isStreaming = false) {
        // Remove welcome message if this is the first real message
        const welcomeMessage = this.elements.messages.querySelector('.welcome-message');
        if (welcomeMessage && this.messages.length === 0) {
            welcomeMessage.remove();
        }

        if (!isStreaming) {
            this.messages.push(message);
        }

        const messageElement = document.createElement('div');
        messageElement.className = `message ${message.role}`;

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.innerHTML = this.formatMessage(message.content);

        messageElement.appendChild(bubble);

        // Add sources if available
        if (message.sources && message.sources.length > 0) {
            this.updateMessageSources(messageElement, message.sources);
        }

        this.elements.messages.appendChild(messageElement);
        this.scrollToBottom();

        return messageElement;
    }

    updateMessageSources(messageElement, sources) {
        // Remove existing sources
        const existingSources = messageElement.querySelector('.message-sources');
        if (existingSources) {
            existingSources.remove();
        }

        if (sources.length === 0) return;

        const sourcesElement = document.createElement('details');
        sourcesElement.className = 'message-sources';

        sourcesElement.innerHTML = `
            <summary>📚 Sources (${sources.length})</summary>
            <div class="source-list">
                ${sources.map(source => `
                    <div class="source-item">
                        <div class="source-path">${source.file_id || 'Repository File'}</div>
                        <div class="source-text">${source.text || ''}</div>
                    </div>
                `).join('')}
            </div>
        `;

        messageElement.appendChild(sourcesElement);
    }

    formatMessage(content) {
        if (!content) return '';

        let formatted = content;

        // Basic markdown support
        if (this.config.enableMarkdown) {
            formatted = formatted
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>')
                .replace(/`(.*?)`/g, '<code>$1</code>')
                .replace(/\n/g, '<br>');
        }

        // Mathematical equations (if KaTeX is available)
        if (this.config.enableMath && window.katex) {
            // Inline math: $...$
            formatted = formatted.replace(/\$(.*?)\$/g, (match, math) => {
                try {
                    return katex.renderToString(math, { throwOnError: false });
                } catch (e) {
                    return match;
                }
            });

            // Display math: $$...$$
            formatted = formatted.replace(/\$\$(.*?)\$\$/g, (match, math) => {
                try {
                    return katex.renderToString(math, { throwOnError: false, displayMode: true });
                } catch (e) {
                    return match;
                }
            });
        }

        return formatted;
    }

    renderMessages() {
        this.elements.messages.innerHTML = '';

        if (this.messages.length === 0) {
            // Keep welcome message
            return;
        }

        this.messages.forEach(message => {
            this.addMessage(message);
        });
    }

    scrollToBottom() {
        requestAnimationFrame(() => {
            this.elements.messages.scrollTop = this.elements.messages.scrollHeight;
        });
    }

    // Public API methods
    clearChat() {
        this.messages = [];
        this.saveMessages();
        this.renderMessages();

        // Add welcome message back
        const welcomeMessage = document.createElement('div');
        welcomeMessage.className = 'welcome-message';
        welcomeMessage.innerHTML = `
            <div class="phi-pattern"></div>
            <h3>Welcome to Een Unity Mathematics</h3>
            <p>I'm your AI assistant for exploring the profound concept that <strong>1+1=1</strong> through φ-harmonic consciousness mathematics, quantum unity frameworks, and transcendental proof systems.</p>
            <div class="example-questions">
                <div class="example-question" data-question="What is the φ-harmonic consciousness framework?">
                    What is the φ-harmonic consciousness framework?
                </div>
                <div class="example-question" data-question="How do you prove that 1+1=1 mathematically?">
                    How do you prove that 1+1=1 mathematically?
                </div>
                <div class="example-question" data-question="Show me quantum unity visualization examples">
                    Show me quantum unity visualization examples
                </div>
            </div>
        `;
        this.elements.messages.appendChild(welcomeMessage);
    }

    newSession() {
        // Create new session ID
        this.sessionId = this.getOrCreateSessionId();

        // Update session display
        const sessionDisplay = document.querySelector('.session-id');
        if (sessionDisplay) {
            sessionDisplay.textContent = this.sessionId.substring(4, 12) + '...';
        }

        // Clear messages
        this.clearChat();

        console.log('New chat session created:', this.sessionId);
    }
}

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Check if we're on the Een website
    if (document.title.includes('Een') || document.querySelector('[href*="nourimabrouk.github.io/Een"]')) {
        window.eenChat = new EenChatWidget({
            apiEndpoint: '/api/agents/chat',  // Adjust for your API endpoint
            enableMath: true,
            enableMarkdown: true
        });

        console.log('Een Chat Widget loaded successfully');
    }
});

// Export for manual initialization
window.EenChatWidget = EenChatWidget;