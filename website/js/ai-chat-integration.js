/**
 * Een Unity Mathematics - AI Chat Integration
 * Advanced ChatGPT/AI Assistant for mathematical consciousness exploration
 */

class EenAIChat {
    constructor(config = {}) {
        this.config = {
            apiEndpoint: config.apiEndpoint || null, // No default API endpoint
            apiKey: config.apiKey || '', // Should be set securely
            model: config.model || 'gpt-4',
            temperature: config.temperature || 0.7,
            maxTokens: config.maxTokens || 2000,
            systemPrompt: this.getSystemPrompt(),
            enableMath: true,
            enableVisualization: true,
            enableVoice: false,
            ...config
        };

        this.chatHistory = [];
        this.isProcessing = false;
        this.isMinimized = false;
        this.isVisible = false;
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

Remember: In Unity Mathematics, 1+1=1 is not a paradox but a profound truth about the nature of unity and consciousness.`;
    }

    initializeChat() {
        this.createChatInterface();
        this.attachEventListeners();
        this.loadChatHistory();
        this.setupAccessibility();

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
                        <span class="phi-symbol">φ</span>
                        <span>Een AI Assistant</span>
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
                    <textarea 
                        id="chat-input" 
                        class="chat-input" 
                        placeholder="Ask about unity mathematics, consciousness fields, or 1+1=1..."
                        rows="2"
                        aria-label="Chat input"
                    ></textarea>
                    <button class="send-btn" onclick="eenChat.sendMessage()" aria-label="Send message">
                        <i class="fas fa-paper-plane"></i>
                    </button>
                </div>
                <div class="chat-status" id="chat-status" aria-live="polite"></div>
            </div>
            <div id="chat-toggle" class="chat-toggle" onclick="eenChat.toggleChat()" role="button" tabindex="0" aria-label="Open AI chat assistant">
                <i class="fas fa-robot"></i>
                <span class="toggle-text">AI Chat</span>
            </div>
        `;

        // Insert chat interface
        document.body.insertAdjacentHTML('beforeend', chatHTML);
    }

    injectStyles() {
        const styles = `
            <style>
                /* AI Chat Container */
                .ai-chat-container {
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    width: 400px;
                    height: 500px;
                    background: var(--bg-primary, white);
                    border: 1px solid var(--border-color, #E2E8F0);
                    border-radius: var(--radius-lg, 0.75rem);
                    box-shadow: var(--shadow-xl, 0 20px 25px -5px rgba(0,0,0,0.1));
                    display: flex;
                    flex-direction: column;
                    z-index: 10000;
                    opacity: 0;
                    transform: translateY(20px);
                    transition: all var(--transition-smooth, 0.3s ease);
                    font-family: var(--font-sans, 'Inter', sans-serif);
                }

                .ai-chat-container.visible {
                    opacity: 1;
                    transform: translateY(0);
                }

                .ai-chat-container.minimized {
                    height: 60px;
                    overflow: hidden;
                }

                /* Chat Header */
                .chat-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 1rem;
                    background: var(--primary-color, #1B365D);
                    color: white;
                    border-radius: var(--radius-lg, 0.75rem) var(--radius-lg, 0.75rem) 0 0;
                }

                .chat-title {
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                    font-weight: 600;
                    font-size: 1rem;
                }

                .chat-title .phi-symbol {
                    font-size: 1.2rem;
                    color: var(--phi-gold, #FFD700);
                }

                .chat-controls {
                    display: flex;
                    gap: 0.5rem;
                }

                .chat-btn {
                    background: none;
                    border: none;
                    color: white;
                    padding: 0.5rem;
                    border-radius: var(--radius, 0.375rem);
                    cursor: pointer;
                    transition: all var(--transition-fast, 0.15s ease);
                    font-size: 0.9rem;
                }

                .chat-btn:hover {
                    background: rgba(255, 255, 255, 0.1);
                }

                /* Chat Messages */
                .chat-messages {
                    flex: 1;
                    overflow-y: auto;
                    padding: 1rem;
                    display: flex;
                    flex-direction: column;
                    gap: 1rem;
                }

                .message {
                    display: flex;
                    gap: 0.75rem;
                    animation: messageSlideIn 0.3s ease;
                }

                .message.user {
                    flex-direction: row-reverse;
                }

                .message-avatar {
                    width: 32px;
                    height: 32px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 0.9rem;
                    font-weight: 600;
                }

                .message.user .message-avatar {
                    background: var(--primary-color, #1B365D);
                    color: white;
                }

                .message.assistant .message-avatar {
                    background: var(--phi-gold, #0F7B8A);
                    color: white;
                }

                .message-content {
                    flex: 1;
                    padding: 0.75rem 1rem;
                    border-radius: var(--radius-lg, 0.75rem);
                    max-width: 80%;
                    word-wrap: break-word;
                }

                .message.user .message-content {
                    background: var(--primary-color, #1B365D);
                    color: white;
                }

                .message.assistant .message-content {
                    background: var(--bg-secondary, #F8FAFC);
                    color: var(--text-primary, #111827);
                    border: 1px solid var(--border-color, #E2E8F0);
                }

                /* Message formatting */
                .message-content h1, .message-content h2, .message-content h3 {
                    margin: 0.5rem 0;
                    color: var(--primary-color, #1B365D);
                }

                .message-content p {
                    margin: 0.5rem 0;
                    line-height: 1.6;
                }

                .message-content code {
                    background: var(--bg-tertiary, #F1F5F9);
                    padding: 0.2rem 0.4rem;
                    border-radius: var(--radius-sm, 0.25rem);
                    font-family: var(--font-mono, 'JetBrains Mono', monospace);
                    font-size: 0.9rem;
                }

                .message-content pre {
                    background: var(--bg-tertiary, #F1F5F9);
                    padding: 1rem;
                    border-radius: var(--radius, 0.375rem);
                    overflow-x: auto;
                    margin: 0.5rem 0;
                }

                .message-content pre code {
                    background: none;
                    padding: 0;
                }

                /* Chat Input */
                .chat-input-container {
                    display: flex;
                    gap: 0.5rem;
                    padding: 1rem;
                    border-top: 1px solid var(--border-color, #E2E8F0);
                }

                .chat-input {
                    flex: 1;
                    padding: 0.75rem;
                    border: 1px solid var(--border-color, #E2E8F0);
                    border-radius: var(--radius, 0.375rem);
                    resize: none;
                    font-family: inherit;
                    font-size: 0.9rem;
                    line-height: 1.4;
                }

                .chat-input:focus {
                    outline: none;
                    border-color: var(--primary-color, #1B365D);
                    box-shadow: 0 0 0 3px rgba(27, 54, 93, 0.1);
                }

                .send-btn {
                    background: var(--primary-color, #1B365D);
                    color: white;
                    border: none;
                    padding: 0.75rem;
                    border-radius: var(--radius, 0.375rem);
                    cursor: pointer;
                    transition: all var(--transition-fast, 0.15s ease);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }

                .send-btn:hover {
                    background: var(--secondary-color, #0F7B8A);
                    transform: translateY(-1px);
                }

                .send-btn:disabled {
                    opacity: 0.5;
                    cursor: not-allowed;
                    transform: none;
                }

                /* Chat Status */
                .chat-status {
                    padding: 0.5rem 1rem;
                    font-size: 0.8rem;
                    color: var(--text-secondary, #6B7280);
                    text-align: center;
                    min-height: 20px;
                }

                /* Chat Toggle Button */
                .chat-toggle {
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    background: var(--primary-color, #1B365D);
                    color: white;
                    border: none;
                    padding: 1rem;
                    border-radius: 50%;
                    cursor: pointer;
                    transition: all var(--transition-smooth, 0.3s ease);
                    box-shadow: var(--shadow-lg, 0 10px 15px -3px rgba(0,0,0,0.1));
                    z-index: 9999;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    gap: 0.25rem;
                    width: 60px;
                    height: 60px;
                }

                .chat-toggle:hover {
                    background: var(--secondary-color, #0F7B8A);
                    transform: translateY(-2px);
                    box-shadow: var(--shadow-xl, 0 20px 25px -5px rgba(0,0,0,0.1));
                }

                .chat-toggle .toggle-text {
                    font-size: 0.7rem;
                    font-weight: 500;
                }

                .chat-toggle.hidden {
                    display: none;
                }

                /* Typing Indicator */
                .typing-indicator {
                    display: flex;
                    gap: 0.25rem;
                    padding: 0.5rem;
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
                    from { opacity: 0; transform: translateY(10px); }
                    to { opacity: 1; transform: translateY(0); }
                }

                /* Dark mode support */
                .dark-mode .ai-chat-container {
                    background: var(--bg-primary-dark, #1F2937);
                    border-color: var(--border-color-dark, #374151);
                }

                .dark-mode .message.assistant .message-content {
                    background: var(--bg-secondary-dark, #374151);
                    color: var(--text-primary-dark, #F9FAFB);
                    border-color: var(--border-color-dark, #4B5563);
                }

                .dark-mode .chat-input {
                    background: var(--bg-secondary-dark, #374151);
                    border-color: var(--border-color-dark, #4B5563);
                    color: var(--text-primary-dark, #F9FAFB);
                }

                /* Responsive design */
                @media (max-width: 768px) {
                    .ai-chat-container {
                        width: calc(100vw - 40px);
                        height: calc(100vh - 120px);
                        bottom: 10px;
                        right: 20px;
                        left: 20px;
                    }

                    .chat-toggle {
                        bottom: 10px;
                        right: 20px;
                    }
                }

                /* Accessibility improvements */
                .ai-chat-container:focus-within {
                    outline: 2px solid var(--primary-color, #1B365D);
                    outline-offset: 2px;
                }

                .chat-btn:focus-visible,
                .send-btn:focus-visible,
                .chat-toggle:focus-visible {
                    outline: 2px solid var(--phi-gold, #0F7B8A);
                    outline-offset: 2px;
                }

                /* High contrast mode */
                @media (prefers-contrast: high) {
                    .ai-chat-container {
                        border-width: 2px;
                    }

                    .chat-btn,
                    .send-btn {
                        border: 1px solid currentColor;
                    }
                }

                /* Reduced motion */
                @media (prefers-reduced-motion: reduce) {
                    .ai-chat-container,
                    .chat-toggle,
                    .message,
                    .typing-dot {
                        animation: none;
                        transition: none;
                    }
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
        const sendBtn = document.querySelector('.send-btn');

        if (chatInput) {
            chatInput.addEventListener('input', () => {
                // Auto-resize textarea
                chatInput.style.height = 'auto';
                chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
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

        // Show typing indicator
        this.showTypingIndicator();
        this.isProcessing = true;

        try {
            // Get AI response
            const response = await this.getAIResponse(message);
            this.addMessage('assistant', response);
        } catch (error) {
            console.error('Chat error:', error);
            this.addMessage('assistant', 'I apologize, but I encountered an error. Please try again or check your connection.');
        } finally {
            this.hideTypingIndicator();
            this.isProcessing = false;
        }
    }

    async getAIResponse(message) {
        // Try actual API first, fallback to mock if it fails
        try {
            // Check if we have an API endpoint configured
            if (this.config.apiEndpoint && this.config.apiEndpoint !== '/api/agents/chat') {
                const response = await fetch(this.config.apiEndpoint, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${this.config.apiKey}`
                    },
                    body: JSON.stringify({
                        model: this.config.model,
                        messages: [
                            { role: 'system', content: this.config.systemPrompt },
                            ...this.chatHistory,
                            { role: 'user', content: message }
                        ],
                        temperature: this.config.temperature,
                        max_tokens: this.config.maxTokens
                    })
                });

                if (response.ok) {
                    const data = await response.json();
                    return data.choices[0].message.content;
                }
            }
        } catch (error) {
            console.warn('API request failed, falling back to mock response:', error);
        }
        
        // Fallback to mock responses
        return this.getMockResponse(message);
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

    shouldVisualize(message) {
        const visualizationKeywords = ['visualize', 'plot', 'graph', 'show', 'draw', 'display', 'render'];
        return visualizationKeywords.some(keyword => message.toLowerCase().includes(keyword));
    }

    createVisualization(context) {
        // Create a simple canvas visualization
        const canvas = document.createElement('canvas');
        canvas.width = 300;
        canvas.height = 200;
        canvas.style.width = '100%';
        canvas.style.height = 'auto';
        canvas.style.borderRadius = '8px';
        canvas.style.marginTop = '10px';

        const ctx = canvas.getContext('2d');

        // Draw a simple φ-harmonic spiral
        ctx.strokeStyle = '#0F7B8A';
        ctx.lineWidth = 2;
        ctx.beginPath();

        const phi = 1.618033988749895;
        for (let i = 0; i < 100; i++) {
            const angle = i * 0.1;
            const radius = 2 * Math.pow(phi, angle / (2 * Math.PI));
            const x = 150 + radius * Math.cos(angle);
            const y = 100 + radius * Math.sin(angle);

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }

        ctx.stroke();
        return canvas;
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
        const status = document.getElementById('chat-status');
        if (status) {
            status.textContent = `Visualizations ${this.config.enableVisualization ? 'enabled' : 'disabled'}`;
        }
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
            window.eenChat.injectStyles();
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