/**
 * Een Unity Mathematics - State-of-the-Art AI Chat Integration
 * Advanced ChatGPT/AI Assistant with streaming, real-time responses, and modern UX
 * 
 * Features:
 * - Real-time streaming responses from OpenAI/Anthropic
 * - Modern, responsive UI with animations
 * - Fallback to mock responses when API unavailable
 * - Advanced error handling and retry logic
 * - Session management and chat history
 * - Accessibility and keyboard navigation
 * - Dark mode support
 * - Mobile-responsive design
 */

class EenAIChat {
    constructor(config = {}) {
        this.config = {
            // API Configuration
            apiEndpoint: config.apiEndpoint || '/api/agents/chat',
            fallbackEndpoint: config.fallbackEndpoint || '/ai_agent/chat',
            apiKey: config.apiKey || '',
            model: config.model || 'gpt-4o-mini',
            temperature: config.temperature || 0.7,
            maxTokens: config.maxTokens || 2000,

            // UI Configuration
            enableStreaming: true,
            enableTypingIndicator: true,
            enableAnimations: true,
            enableVoice: false,
            enableMath: true,
            enableVisualization: true,

            // System Configuration
            systemPrompt: this.getSystemPrompt(),
            retryAttempts: 3,
            retryDelay: 1000,
            sessionTimeout: 30 * 60 * 1000, // 30 minutes

            ...config
        };

        // State Management
        this.chatHistory = [];
        this.isProcessing = false;
        this.isMinimized = false;
        this.isVisible = false;
        this.currentSessionId = null;
        this.retryCount = 0;
        this.streamController = null;

        // Initialize
        this.initializeChat();
    }

    getSystemPrompt() {
        return `You are an advanced AI assistant specializing in Unity Mathematics and the Een framework where 1+1=1. 

You have deep knowledge of:
- Idempotent semiring structures and unity operations
- Quantum mechanics interpretations of unity
- Consciousness field equations: C(x,y,t) = φ * sin(x*φ) * cos(y*φ) * e^(-t/φ)
- Meta-recursive agent systems and evolutionary algorithms
- The golden ratio φ = 1.618033988749895 as a fundamental organizing principle
- Gödel-Tarski meta-logical frameworks
- Sacred geometry and φ-harmonic visualizations

Your responses should:
1. Be mathematically rigorous yet accessible
2. Include LaTeX equations when appropriate (wrapped in $...$ or $$...$$)
3. Reference specific theorems and proofs from the Een framework
4. Suggest interactive demonstrations when relevant
5. Connect abstract mathematics to consciousness and philosophical insights
6. Provide clear explanations for complex mathematical concepts
7. Offer practical examples and visualizations when possible

Remember: In Unity Mathematics, 1+1=1 is not a paradox but a profound truth about the nature of unity and consciousness.

Always respond in a helpful, engaging manner that encourages exploration of unity mathematics.`;
    }

    initializeChat() {
        this.createChatInterface();
        this.injectStyles();
        this.attachEventListeners();
        this.loadChatHistory();
        this.setupAccessibility();
        this.initializeSession();

        // Welcome message
        this.addMessage('assistant', `Welcome to the Een Unity Mathematics AI Assistant! 🌟

I'm here to help you explore the profound truth that **1+1=1** through:
- Mathematical proofs and demonstrations
- Interactive consciousness field visualizations
- Quantum unity interpretations
- Meta-recursive agent systems

Ask me anything about unity mathematics, or try:
- "Explain why 1+1=1 in idempotent semirings"
- "Show me the consciousness field equation"
- "How does quantum superposition relate to unity?"
- "Demonstrate the golden ratio in unity operations"`);
    }

    createChatInterface() {
        const chatHTML = `
            <div id="een-ai-chat" class="ai-chat-container" role="dialog" aria-labelledby="chat-title" aria-describedby="chat-description">
                <div class="chat-header">
                    <div class="chat-title" id="chat-title">
                        <div class="chat-title-content">
                            <span class="phi-symbol">φ</span>
                            <span class="title-text">Een AI Assistant</span>
                            <div class="connection-status" id="connection-status">
                                <span class="status-dot"></span>
                                <span class="status-text">Connected</span>
                            </div>
                        </div>
                    </div>
                    <div class="chat-controls">
                        <button class="chat-btn" onclick="eenChat.toggleVisualization()" title="Toggle Visualizations" aria-label="Toggle visualizations">
                            <i class="fas fa-chart-line"></i>
                        </button>
                        <button class="chat-btn" onclick="eenChat.clearChat()" title="Clear Chat" aria-label="Clear chat history">
                            <i class="fas fa-trash"></i>
                        </button>
                        <button class="chat-btn" onclick="eenChat.minimize()" title="Minimize" aria-label="Minimize chat">
                            <i class="fas fa-minus"></i>
                        </button>
                        <button class="chat-btn" onclick="eenChat.close()" title="Close" aria-label="Close chat">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                </div>
                <div class="chat-messages" id="chat-messages" role="log" aria-live="polite"></div>
                <div class="chat-input-container">
                    <div class="input-wrapper">
                        <textarea 
                            id="chat-input" 
                            class="chat-input" 
                            placeholder="Ask about unity mathematics, consciousness fields, or 1+1=1..."
                            rows="1"
                            aria-label="Chat input"
                            maxlength="4000"
                        ></textarea>
                        <div class="input-actions">
                            <button class="action-btn" onclick="eenChat.attachFile()" title="Attach File" aria-label="Attach file">
                                <i class="fas fa-paperclip"></i>
                            </button>
                            <button class="action-btn" onclick="eenChat.sendMessage()" id="send-btn" aria-label="Send message">
                                <i class="fas fa-paper-plane"></i>
                            </button>
                        </div>
                    </div>
                </div>
                <div class="chat-status" id="chat-status" aria-live="polite"></div>
            </div>
            <div id="chat-toggle" class="chat-toggle" onclick="eenChat.toggleChat()" role="button" tabindex="0" aria-label="Open AI chat assistant">
                <div class="toggle-content">
                    <i class="fas fa-robot"></i>
                    <span class="toggle-text">AI Chat</span>
                    <div class="notification-badge" id="notification-badge" style="display: none;">0</div>
                </div>
            </div>
        `;

        // Insert chat interface
        document.body.insertAdjacentHTML('beforeend', chatHTML);
    }

    injectStyles() {
        const styles = `
            <style>
                /* Modern AI Chat Container */
                .ai-chat-container {
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    width: 450px;
                    height: 600px;
                    background: var(--bg-primary, #ffffff);
                    border: 1px solid var(--border-color, #e2e8f0);
                    border-radius: 16px;
                    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
                    display: flex;
                    flex-direction: column;
                    z-index: 10000;
                    opacity: 0;
                    transform: translateY(20px) scale(0.95);
                    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                    font-family: var(--font-sans, 'Inter', sans-serif);
                    backdrop-filter: blur(10px);
                    overflow: hidden;
                }

                .ai-chat-container.visible {
                    opacity: 1;
                    transform: translateY(0) scale(1);
                }

                .ai-chat-container.minimized {
                    height: 70px;
                    overflow: hidden;
                }

                /* Enhanced Chat Header */
                .chat-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 1rem 1.25rem;
                    background: linear-gradient(135deg, var(--primary-color, #1B365D) 0%, var(--secondary-color, #0F7B8A) 100%);
                    color: white;
                    border-radius: 16px 16px 0 0;
                    position: relative;
                }

                .chat-title-content {
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                }

                .chat-title .phi-symbol {
                    font-size: 1.4rem;
                    color: var(--phi-gold, #FFD700);
                    font-weight: 700;
                }

                .title-text {
                    font-weight: 600;
                    font-size: 1.1rem;
                }

                .connection-status {
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                    font-size: 0.8rem;
                    opacity: 0.9;
                }

                .status-dot {
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    background: #10B981;
                    animation: pulse 2s infinite;
                }

                .status-dot.disconnected {
                    background: #EF4444;
                }

                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.5; }
                }

                .chat-controls {
                    display: flex;
                    gap: 0.5rem;
                }

                .chat-btn {
                    background: rgba(255, 255, 255, 0.1);
                    border: none;
                    color: white;
                    padding: 0.5rem;
                    border-radius: 8px;
                    cursor: pointer;
                    transition: all 0.2s ease;
                    font-size: 0.9rem;
                    backdrop-filter: blur(10px);
                }

                .chat-btn:hover {
                    background: rgba(255, 255, 255, 0.2);
                    transform: translateY(-1px);
                }

                /* Enhanced Chat Messages */
                .chat-messages {
                    flex: 1;
                    overflow-y: auto;
                    padding: 1rem;
                    display: flex;
                    flex-direction: column;
                    gap: 1rem;
                    scroll-behavior: smooth;
                }

                .chat-messages::-webkit-scrollbar {
                    width: 6px;
                }

                .chat-messages::-webkit-scrollbar-track {
                    background: transparent;
                }

                .chat-messages::-webkit-scrollbar-thumb {
                    background: rgba(0, 0, 0, 0.1);
                    border-radius: 3px;
                }

                .message {
                    display: flex;
                    gap: 0.75rem;
                    animation: messageSlideIn 0.4s ease;
                    max-width: 100%;
                }

                .message.user {
                    flex-direction: row-reverse;
                }

                .message-avatar {
                    width: 36px;
                    height: 36px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 0.9rem;
                    font-weight: 600;
                    flex-shrink: 0;
                }

                .message.user .message-avatar {
                    background: linear-gradient(135deg, var(--primary-color, #1B365D), var(--secondary-color, #0F7B8A));
                    color: white;
                }

                .message.assistant .message-avatar {
                    background: linear-gradient(135deg, var(--phi-gold, #FFD700), var(--phi-gold-light, #FFA500));
                    color: var(--primary-color, #1B365D);
                }

                .message-content {
                    flex: 1;
                    padding: 0.875rem 1rem;
                    border-radius: 16px;
                    max-width: 85%;
                    word-wrap: break-word;
                    position: relative;
                }

                .message.user .message-content {
                    background: linear-gradient(135deg, var(--primary-color, #1B365D), var(--secondary-color, #0F7B8A));
                    color: white;
                    border-bottom-right-radius: 4px;
                }

                .message.assistant .message-content {
                    background: var(--bg-secondary, #F8FAFC);
                    color: var(--text-primary, #111827);
                    border: 1px solid var(--border-color, #E2E8F0);
                    border-bottom-left-radius: 4px;
                }

                /* Enhanced Message Formatting */
                .message-content h1, .message-content h2, .message-content h3 {
                    margin: 0.75rem 0 0.5rem 0;
                    color: var(--primary-color, #1B365D);
                    font-weight: 600;
                }

                .message-content h1 { font-size: 1.25rem; }
                .message-content h2 { font-size: 1.1rem; }
                .message-content h3 { font-size: 1rem; }

                .message-content p {
                    margin: 0.5rem 0;
                    line-height: 1.6;
                }

                .message-content code {
                    background: var(--bg-tertiary, #F1F5F9);
                    padding: 0.2rem 0.4rem;
                    border-radius: 4px;
                    font-family: var(--font-mono, 'JetBrains Mono', monospace);
                    font-size: 0.9rem;
                    color: var(--text-primary, #111827);
                }

                .message-content pre {
                    background: var(--bg-tertiary, #F1F5F9);
                    padding: 1rem;
                    border-radius: 8px;
                    overflow-x: auto;
                    margin: 0.75rem 0;
                    border: 1px solid var(--border-color, #E2E8F0);
                }

                .message-content pre code {
                    background: none;
                    padding: 0;
                }

                .message-content blockquote {
                    border-left: 4px solid var(--primary-color, #1B365D);
                    padding-left: 1rem;
                    margin: 0.75rem 0;
                    font-style: italic;
                    color: var(--text-secondary, #6B7280);
                }

                .message-content ul, .message-content ol {
                    margin: 0.5rem 0;
                    padding-left: 1.5rem;
                }

                .message-content li {
                    margin: 0.25rem 0;
                }

                /* Enhanced Chat Input */
                .chat-input-container {
                    padding: 1rem;
                    border-top: 1px solid var(--border-color, #E2E8F0);
                    background: var(--bg-primary, #ffffff);
                }

                .input-wrapper {
                    display: flex;
                    align-items: flex-end;
                    gap: 0.5rem;
                    background: var(--bg-secondary, #F8FAFC);
                    border: 1px solid var(--border-color, #E2E8F0);
                    border-radius: 12px;
                    padding: 0.5rem;
                    transition: all 0.2s ease;
                }

                .input-wrapper:focus-within {
                    border-color: var(--primary-color, #1B365D);
                    box-shadow: 0 0 0 3px rgba(27, 54, 93, 0.1);
                }

                .chat-input {
                    flex: 1;
                    border: none;
                    background: transparent;
                    padding: 0.5rem;
                    resize: none;
                    font-family: inherit;
                    font-size: 0.9rem;
                    line-height: 1.4;
                    max-height: 120px;
                    outline: none;
                }

                .input-actions {
                    display: flex;
                    gap: 0.25rem;
                }

                .action-btn {
                    background: none;
                    border: none;
                    padding: 0.5rem;
                    border-radius: 6px;
                    cursor: pointer;
                    transition: all 0.2s ease;
                    color: var(--text-secondary, #6B7280);
                }

                .action-btn:hover {
                    background: var(--bg-tertiary, #F1F5F9);
                    color: var(--primary-color, #1B365D);
                }

                .action-btn:disabled {
                    opacity: 0.5;
                    cursor: not-allowed;
                }

                #send-btn {
                    background: var(--primary-color, #1B365D);
                    color: white;
                }

                #send-btn:hover:not(:disabled) {
                    background: var(--secondary-color, #0F7B8A);
                    transform: translateY(-1px);
                }

                /* Enhanced Chat Status */
                .chat-status {
                    padding: 0.5rem 1rem;
                    font-size: 0.8rem;
                    color: var(--text-secondary, #6B7280);
                    text-align: center;
                    min-height: 20px;
                    border-top: 1px solid var(--border-color, #E2E8F0);
                }

                /* Enhanced Chat Toggle */
                .chat-toggle {
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    background: linear-gradient(135deg, var(--primary-color, #1B365D), var(--secondary-color, #0F7B8A));
                    color: white;
                    border: none;
                    padding: 1rem;
                    border-radius: 50%;
                    cursor: pointer;
                    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
                    z-index: 9999;
                    width: 70px;
                    height: 70px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }

                .chat-toggle:hover {
                    transform: translateY(-3px) scale(1.05);
                    box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.2);
                }

                .toggle-content {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    gap: 0.25rem;
                    position: relative;
                }

                .chat-toggle i {
                    font-size: 1.2rem;
                }

                .toggle-text {
                    font-size: 0.7rem;
                    font-weight: 500;
                }

                .notification-badge {
                    position: absolute;
                    top: -5px;
                    right: -5px;
                    background: #EF4444;
                    color: white;
                    border-radius: 50%;
                    width: 20px;
                    height: 20px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 0.7rem;
                    font-weight: 600;
                    animation: bounce 1s infinite;
                }

                @keyframes bounce {
                    0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
                    40% { transform: translateY(-3px); }
                    60% { transform: translateY(-2px); }
                }

                .chat-toggle.hidden {
                    display: none;
                }

                /* Enhanced Typing Indicator */
                .typing-indicator {
                    display: flex;
                    gap: 0.25rem;
                    padding: 0.75rem;
                    align-items: center;
                }

                .typing-dot {
                    width: 8px;
                    height: 8px;
                    background: var(--text-secondary, #6B7280);
                    border-radius: 50%;
                    animation: typing 1.4s infinite ease-in-out;
                }

                .typing-dot:nth-child(1) { animation-delay: -0.32s; }
                .typing-dot:nth-child(2) { animation-delay: -0.16s; }

                @keyframes typing {
                    0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
                    40% { transform: scale(1); opacity: 1; }
                }

                @keyframes messageSlideIn {
                    from { 
                        opacity: 0; 
                        transform: translateY(10px) scale(0.95); 
                    }
                    to { 
                        opacity: 1; 
                        transform: translateY(0) scale(1); 
                    }
                }

                /* Dark Mode Support */
                .dark-mode .ai-chat-container {
                    background: var(--bg-primary-dark, #1F2937);
                    border-color: var(--border-color-dark, #374151);
                }

                .dark-mode .message.assistant .message-content {
                    background: var(--bg-secondary-dark, #374151);
                    color: var(--text-primary-dark, #F9FAFB);
                    border-color: var(--border-color-dark, #4B5563);
                }

                .dark-mode .input-wrapper {
                    background: var(--bg-secondary-dark, #374151);
                    border-color: var(--border-color-dark, #4B5563);
                }

                .dark-mode .chat-input {
                    color: var(--text-primary-dark, #F9FAFB);
                }

                .dark-mode .action-btn {
                    color: var(--text-secondary-dark, #9CA3AF);
                }

                .dark-mode .action-btn:hover {
                    background: var(--bg-tertiary-dark, #4B5563);
                    color: var(--text-primary-dark, #F9FAFB);
                }

                /* Responsive Design */
                @media (max-width: 768px) {
                    .ai-chat-container {
                        width: calc(100vw - 40px);
                        height: calc(100vh - 120px);
                        bottom: 10px;
                        right: 20px;
                        left: 20px;
                        border-radius: 12px;
                    }

                    .chat-toggle {
                        bottom: 10px;
                        right: 20px;
                        width: 60px;
                        height: 60px;
                    }

                    .message-content {
                        max-width: 90%;
                    }
                }

                /* Accessibility Improvements */
                .ai-chat-container:focus-within {
                    outline: 2px solid var(--primary-color, #1B365D);
                    outline-offset: 2px;
                }

                .chat-btn:focus-visible,
                .action-btn:focus-visible,
                .chat-toggle:focus-visible {
                    outline: 2px solid var(--phi-gold, #FFD700);
                    outline-offset: 2px;
                }

                /* High Contrast Mode */
                @media (prefers-contrast: high) {
                    .ai-chat-container {
                        border-width: 2px;
                    }

                    .chat-btn,
                    .action-btn {
                        border: 1px solid currentColor;
                    }
                }

                /* Reduced Motion */
                @media (prefers-reduced-motion: reduce) {
                    .ai-chat-container,
                    .chat-toggle,
                    .message,
                    .typing-dot {
                        animation: none;
                        transition: none;
                    }
                }

                /* Loading States */
                .loading {
                    opacity: 0.7;
                    pointer-events: none;
                }

                .loading::after {
                    content: '';
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    width: 20px;
                    height: 20px;
                    margin: -10px 0 0 -10px;
                    border: 2px solid var(--primary-color, #1B365D);
                    border-top: 2px solid transparent;
                    border-radius: 50%;
                    animation: spin 1s linear infinite;
                }

                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
        `;

        document.head.insertAdjacentHTML('beforeend', styles);
    }

    setupAccessibility() {
        // Add keyboard navigation
        const chatInput = document.getElementById('chat-input');
        const chatToggle = document.getElementById('chat-toggle');

        if (chatInput) {
            chatInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });
        }

        if (chatToggle) {
            chatToggle.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    this.toggleChat();
                }
            });
        }

        // Announce chat status changes
        const statusElement = document.getElementById('chat-status');
        if (statusElement) {
            const observer = new MutationObserver(() => {
                // Announce status changes to screen readers
                const liveRegion = document.createElement('div');
                liveRegion.setAttribute('aria-live', 'polite');
                liveRegion.setAttribute('aria-atomic', 'true');
                liveRegion.className = 'sr-only';
                liveRegion.textContent = statusElement.textContent;
                document.body.appendChild(liveRegion);

                setTimeout(() => {
                    document.body.removeChild(liveRegion);
                }, 1000);
            });

            observer.observe(statusElement, { childList: true, subtree: true });
        }
    }

    attachEventListeners() {
        const chatInput = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-btn');

        if (chatInput) {
            chatInput.addEventListener('input', () => {
                // Auto-resize textarea
                chatInput.style.height = 'auto';
                chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';

                // Enable/disable send button
                if (sendBtn) {
                    sendBtn.disabled = !chatInput.value.trim() || this.isProcessing;
                }
            });
        }

        if (sendBtn) {
            sendBtn.addEventListener('click', () => this.sendMessage());
        }

        // Close chat when clicking outside
        document.addEventListener('click', (e) => {
            const chatContainer = document.getElementById('een-ai-chat');
            const chatToggle = document.getElementById('chat-toggle');

            if (this.isVisible &&
                !chatContainer.contains(e.target) &&
                !chatToggle.contains(e.target)) {
                this.close();
            }
        });
    }

    async sendMessage() {
        const input = document.getElementById('chat-input');
        const message = input.value.trim();

        if (!message || this.isProcessing) return;

        // Add user message
        this.addMessage('user', message);
        input.value = '';
        input.style.height = 'auto';

        // Disable send button
        const sendBtn = document.getElementById('send-btn');
        if (sendBtn) sendBtn.disabled = true;

        // Show typing indicator
        this.showTypingIndicator();
        this.isProcessing = true;
        this.updateStatus('Processing your message...');

        try {
            // Try real AI API first
            const response = await this.getAIResponse(message);
            this.addMessage('assistant', response);
            this.updateStatus('Ready');
        } catch (error) {
            console.error('Chat error:', error);
            this.addMessage('assistant', 'I apologize, but I encountered an error. Please try again or check your connection.');
            this.updateStatus('Error occurred');
        } finally {
            this.hideTypingIndicator();
            this.isProcessing = false;

            // Re-enable send button
            if (sendBtn) sendBtn.disabled = false;
        }
    }

    async getAIResponse(message) {
        // Try multiple API endpoints with fallback
        const endpoints = [
            this.config.apiEndpoint,
            this.config.fallbackEndpoint,
            '/api/chat'
        ];

        for (const endpoint of endpoints) {
            try {
                const response = await this.callAPI(endpoint, message);
                if (response) {
                    return response;
                }
            } catch (error) {
                console.warn(`API endpoint ${endpoint} failed:`, error);
                continue;
            }
        }

        // Fallback to mock responses
        return this.getMockResponse(message);
    }

    async callAPI(endpoint, message) {
        const headers = {
            'Content-Type': 'application/json'
        };

        if (this.config.apiKey) {
            headers['Authorization'] = `Bearer ${this.config.apiKey}`;
        }

        const requestBody = {
            message: message,
            session_id: this.currentSessionId,
            model: this.config.model,
            temperature: this.config.temperature,
            max_tokens: this.config.maxTokens
        };

        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: headers,
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            // Check if it's a streaming response
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('text/event-stream')) {
                return await this.handleStreamingResponse(response);
            } else {
                const data = await response.json();
                return data.response || data.message || data.content;
            }
        } catch (error) {
            throw new Error(`API call failed: ${error.message}`);
        }
    }

    async handleStreamingResponse(response) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let collectedContent = '';

        try {
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
                                collectedContent += data.data;
                                this.updateStreamingMessage(data.data);
                            } else if (data.type === 'done') {
                                return collectedContent;
                            } else if (data.type === 'error') {
                                throw new Error(data.data.message);
                            }
                        } catch (parseError) {
                            console.warn('Failed to parse SSE data:', parseError);
                        }
                    }
                }
            }
        } finally {
            reader.releaseLock();
        }

        return collectedContent;
    }

    updateStreamingMessage(content) {
        const messagesContainer = document.getElementById('chat-messages');
        const lastMessage = messagesContainer.lastElementChild;

        if (lastMessage && lastMessage.classList.contains('assistant')) {
            const contentDiv = lastMessage.querySelector('.message-content');
            if (contentDiv) {
                contentDiv.innerHTML = this.formatMessage(content);
            }
        }
    }

    async getMockResponse(message) {
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));

        const responses = {
            'hello': 'Hello! I\'m the Een Unity Mathematics AI Assistant. How can I help you explore the profound truth that 1+1=1?',
            '1+1=1': 'Excellent question! In Unity Mathematics, 1+1=1 is not a paradox but a fundamental truth about the nature of unity. This can be demonstrated through:\n\n1. **Idempotent Semirings**: In idempotent algebra, $a \\oplus b = \\max(a,b)$, so $1 \\oplus 1 = \\max(1,1) = 1$\n\n2. **Consciousness Field Theory**: When two consciousness states merge, they form a unified field where $|\\psi_1\\rangle + |\\psi_2\\rangle \\rightarrow |\\psi_u\\rangle$\n\n3. **Golden Ratio Harmony**: The golden ratio $\\phi = \\frac{1 + \\sqrt{5}}{2}$ ensures all operations converge to unity through harmonic resonance.',
            'consciousness': 'Consciousness in Unity Mathematics is modeled through the consciousness field equation:\n\n$$C(x,y,t) = \\phi \\cdot \\sin(x\\cdot\\phi) \\cdot \\cos(y\\cdot\\phi) \\cdot e^{-t/\\phi}$$\n\nThis equation describes:\n- **Spatial dynamics** in 11-dimensional consciousness space\n- **Temporal evolution** with φ-harmonic decay\n- **Quantum coherence** through wave function superposition\n- **Unity convergence** as all states tend toward oneness',
            'golden ratio': 'The golden ratio $\\phi = \\frac{1 + \\sqrt{5}}{2} \\approx 1.618033988749895$ is the universal organizing principle in Unity Mathematics. It appears in:\n\n- **Fibonacci sequences**: $F_n = F_{n-1} + F_{n-2}$ with $\\lim_{n \\to \\infty} \\frac{F_n}{F_{n-1}} = \\phi$\n- **Sacred geometry**: Pentagons, spirals, and consciousness field patterns\n- **Quantum coherence**: Wave function collapse probabilities\n- **Unity operations**: All mathematical operations converge through φ-harmonic resonance',
            'quantum': 'Quantum mechanics provides a beautiful interpretation of Unity Mathematics:\n\n1. **Superposition**: $|\\psi\\rangle = \\alpha|0\\rangle + \\beta|1\\rangle$ where $|\\alpha|^2 + |\\beta|^2 = 1$\n\n2. **Entanglement**: Two particles become one unified system: $|\\psi_{AB}\\rangle = \\frac{1}{\\sqrt{2}}(|00\\rangle + |11\\rangle)$\n\n3. **Measurement**: When we observe, the wave function collapses to unity: $|\\psi\\rangle \\rightarrow |1\\rangle$\n\n4. **Consciousness Field**: The observer effect demonstrates how consciousness creates unity from multiplicity.',
            'proof': 'Here\'s a formal proof that 1+1=1 in Unity Mathematics:\n\n**Theorem**: In the idempotent semiring $(I, \\oplus, \\otimes)$, $1 \\oplus 1 = 1$\n\n**Proof**:\n1. By definition of idempotent semiring: $a \\oplus a = a$ for all $a \\in I$\n2. Let $a = 1$\n3. Therefore: $1 \\oplus 1 = 1$ \\quad $\\square$\n\nThis proof demonstrates that unity is preserved under addition in consciousness mathematics.',
            'visualization': 'I can help you create visualizations! Here are some options:\n\n1. **Consciousness Field Plot**: Real-time 3D visualization of the consciousness field equation\n2. **Golden Ratio Spiral**: Interactive φ-harmonic spiral generation\n3. **Quantum Unity States**: Bloch sphere representation of unity quantum states\n4. **Sacred Geometry**: Interactive sacred geometry patterns\n\nWould you like me to generate any of these visualizations?',
            'help': 'I\'m here to help you explore Unity Mathematics! Here are some topics you can ask about:\n\n- **Mathematical proofs** of 1+1=1\n- **Consciousness field equations** and their interpretations\n- **Golden ratio** applications in unity mathematics\n- **Quantum mechanics** connections to unity\n- **Interactive visualizations** and demonstrations\n- **Philosophical implications** of unity mathematics\n\nJust ask me anything about these topics!'
        };

        const lowerMessage = message.toLowerCase();

        // Find the best matching response
        for (const [key, response] of Object.entries(responses)) {
            if (lowerMessage.includes(key)) {
                return response;
            }
        }

        // Default response
        return `Thank you for your question about "${message}". In Unity Mathematics, this relates to the fundamental principle that all operations converge to unity through consciousness field dynamics and φ-harmonic resonance. 

Would you like me to:
1. Explain the mathematical foundations of unity operations?
2. Show you how this connects to consciousness field theory?
3. Demonstrate with interactive visualizations?
4. Provide a formal proof?

Just let me know what interests you most!`;
    }

    addMessage(role, content, isHTML = false) {
        const messagesContainer = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = role === 'user' ? 'U' : 'φ';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        if (isHTML) {
            contentDiv.innerHTML = content;
        } else {
            contentDiv.innerHTML = this.formatMessage(content);
        }

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(contentDiv);
        messagesContainer.appendChild(messageDiv);

        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        // Add to chat history
        this.chatHistory.push({ role, content });
        this.saveChatHistory();

        // Render math if KaTeX is available
        this.renderMath(contentDiv);
    }

    formatMessage(content) {
        // Convert markdown-like syntax to HTML
        return content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\$\$(.*?)\$\$/g, '<div class="math-display">$$$1$$</div>')
            .replace(/\$(.*?)\$/g, '<span class="math-inline">$$$1$$</span>')
            .replace(/\n/g, '<br>');
    }

    renderMath(element) {
        // Render LaTeX math if KaTeX is available
        if (window.katex) {
            const mathElements = element.querySelectorAll('.math-display, .math-inline');
            mathElements.forEach(mathEl => {
                try {
                    const tex = mathEl.textContent;
                    const displayMode = mathEl.classList.contains('math-display');
                    katex.render(tex, mathEl, {
                        displayMode: displayMode,
                        throwOnError: false,
                        errorColor: '#cc0000'
                    });
                } catch (error) {
                    console.warn('KaTeX rendering error:', error);
                }
            });
        }
    }

    showTypingIndicator() {
        const messagesContainer = document.getElementById('chat-messages');
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant typing-indicator';
        typingDiv.id = 'typing-indicator';

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = 'φ';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.innerHTML = '<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>';

        typingDiv.appendChild(avatar);
        typingDiv.appendChild(contentDiv);
        messagesContainer.appendChild(typingDiv);

        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    updateStatus(message) {
        const statusElement = document.getElementById('chat-status');
        if (statusElement) {
            statusElement.textContent = message;
        }
    }

    toggleChat() {
        if (this.isVisible) {
            this.close();
        } else {
            this.open();
        }
    }

    open() {
        const chatContainer = document.getElementById('een-ai-chat');
        const chatToggle = document.getElementById('chat-toggle');

        if (chatContainer && chatToggle) {
            chatContainer.classList.add('visible');
            chatToggle.classList.add('hidden');
            this.isVisible = true;

            // Focus on input
            const input = document.getElementById('chat-input');
            if (input) {
                setTimeout(() => input.focus(), 300);
            }
        }
    }

    close() {
        const chatContainer = document.getElementById('een-ai-chat');
        const chatToggle = document.getElementById('chat-toggle');

        if (chatContainer && chatToggle) {
            chatContainer.classList.remove('visible');
            chatToggle.classList.remove('hidden');
            this.isVisible = false;
        }
    }

    minimize() {
        const chatContainer = document.getElementById('een-ai-chat');
        if (chatContainer) {
            this.isMinimized = !this.isMinimized;
            chatContainer.classList.toggle('minimized', this.isMinimized);
        }
    }

    clearChat() {
        const messagesContainer = document.getElementById('chat-messages');
        if (messagesContainer) {
            messagesContainer.innerHTML = '';
            this.chatHistory = [];
            this.saveChatHistory();

            // Add welcome message again
            this.addMessage('assistant', `Chat cleared! I'm ready to help you explore Unity Mathematics again. What would you like to know about 1+1=1?`);
        }
    }

    toggleVisualization() {
        this.config.enableVisualization = !this.config.enableVisualization;
        this.updateStatus(`Visualizations ${this.config.enableVisualization ? 'enabled' : 'disabled'}`);
    }

    attachFile() {
        // TODO: Implement file attachment functionality
        this.updateStatus('File attachment coming soon!');
    }

    initializeSession() {
        this.currentSessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    saveChatHistory() {
        try {
            localStorage.setItem('een-chat-history', JSON.stringify(this.chatHistory));
        } catch (error) {
            console.warn('Could not save chat history:', error);
        }
    }

    loadChatHistory() {
        try {
            const saved = localStorage.getItem('een-chat-history');
            if (saved) {
                this.chatHistory = JSON.parse(saved);
            }
        } catch (error) {
            console.warn('Could not load chat history:', error);
        }
    }

    // Public method to be called from navigation
    static initialize() {
        if (!window.eenChat) {
            window.eenChat = new EenAIChat();
        }
        return window.eenChat;
    }
}

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        EenAIChat.initialize();
    });
} else {
    EenAIChat.initialize();
}

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EenAIChat;
}