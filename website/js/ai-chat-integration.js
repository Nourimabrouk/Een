/**
 * Een Unity Mathematics - AI Chat Integration
 * Advanced ChatGPT/AI Assistant for mathematical consciousness exploration
 */

class EenAIChat {
    constructor(config = {}) {
        this.config = {
            apiEndpoint: config.apiEndpoint || 'https://api.openai.com/v1/chat/completions',
            apiKey: config.apiKey || '', // Should be set securely
            model: config.model || 'gpt-4',
            temperature: config.temperature || 0.7,
            maxTokens: config.maxTokens || 2000,
            systemPrompt: this.getSystemPrompt(),
            enableMath: true,
            enableVisualization: true,
            ...config
        };
        
        this.chatHistory = [];
        this.isProcessing = false;
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

Remember: In Unity Mathematics, 1+1=1 is not a paradox but a profound truth about the nature of unity and consciousness.`;
    }
    
    initializeChat() {
        this.createChatInterface();
        this.attachEventListeners();
        this.loadChatHistory();
        
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
            <div id="een-ai-chat" class="ai-chat-container">
                <div class="chat-header">
                    <div class="chat-title">
                        <span class="phi-symbol">φ</span>
                        <span>Een AI Assistant</span>
                    </div>
                    <div class="chat-controls">
                        <button class="chat-btn" onclick="eenChat.toggleVisualization()" title="Toggle Visualizations">
                            <i class="fas fa-chart-line"></i>
                        </button>
                        <button class="chat-btn" onclick="eenChat.clearChat()" title="Clear Chat">
                            <i class="fas fa-trash"></i>
                        </button>
                        <button class="chat-btn" onclick="eenChat.minimize()" title="Minimize">
                            <i class="fas fa-minus"></i>
                        </button>
                    </div>
                </div>
                <div class="chat-messages" id="chat-messages"></div>
                <div class="chat-input-container">
                    <textarea 
                        id="chat-input" 
                        class="chat-input" 
                        placeholder="Ask about unity mathematics, consciousness fields, or 1+1=1..."
                        rows="2"
                    ></textarea>
                    <button class="send-btn" onclick="eenChat.sendMessage()">
                        <i class="fas fa-paper-plane"></i>
                    </button>
                </div>
                <div class="chat-status" id="chat-status"></div>
            </div>
            <button id="chat-toggle" class="chat-toggle-btn" onclick="eenChat.toggleChat()">
                <i class="fas fa-comments"></i>
                <span class="chat-badge">AI</span>
            </button>
        `;
        
        // Add to page
        document.body.insertAdjacentHTML('beforeend', chatHTML);
        
        // Add styles
        this.injectStyles();
    }
    
    injectStyles() {
        const styles = `
            <style>
                .ai-chat-container {
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    width: 400px;
                    max-width: 90vw;
                    height: 600px;
                    max-height: 80vh;
                    background: white;
                    border-radius: 16px;
                    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
                    display: flex;
                    flex-direction: column;
                    z-index: 1000;
                    transition: all 0.3s ease;
                    border: 1px solid var(--border-color);
                }
                
                .ai-chat-container.minimized {
                    height: 60px;
                    overflow: hidden;
                }
                
                .ai-chat-container.hidden {
                    display: none;
                }
                
                .chat-header {
                    background: var(--gradient-primary);
                    color: white;
                    padding: 1rem;
                    border-radius: 16px 16px 0 0;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                }
                
                .chat-title {
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                    font-weight: 600;
                }
                
                .phi-symbol {
                    font-size: 1.5rem;
                    color: var(--phi-gold-light);
                }
                
                .chat-controls {
                    display: flex;
                    gap: 0.5rem;
                }
                
                .chat-btn {
                    background: rgba(255, 255, 255, 0.2);
                    border: none;
                    color: white;
                    width: 32px;
                    height: 32px;
                    border-radius: 8px;
                    cursor: pointer;
                    transition: all 0.2s;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                
                .chat-btn:hover {
                    background: rgba(255, 255, 255, 0.3);
                    transform: translateY(-1px);
                }
                
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
                    animation: messageSlide 0.3s ease;
                }
                
                @keyframes messageSlide {
                    from {
                        opacity: 0;
                        transform: translateY(10px);
                    }
                    to {
                        opacity: 1;
                        transform: translateY(0);
                    }
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
                    font-size: 1.2rem;
                    flex-shrink: 0;
                }
                
                .message.assistant .message-avatar {
                    background: var(--gradient-phi);
                    color: white;
                }
                
                .message.user .message-avatar {
                    background: var(--bg-tertiary);
                    color: var(--primary-color);
                }
                
                .message-content {
                    background: var(--bg-secondary);
                    padding: 0.75rem 1rem;
                    border-radius: 12px;
                    max-width: 85%;
                    line-height: 1.5;
                }
                
                .message.user .message-content {
                    background: var(--primary-color);
                    color: white;
                }
                
                .message-content h1, .message-content h2, .message-content h3 {
                    font-size: 1.1rem;
                    margin: 0.5rem 0;
                }
                
                .message-content code {
                    background: rgba(0, 0, 0, 0.05);
                    padding: 0.2rem 0.4rem;
                    border-radius: 4px;
                    font-family: var(--font-mono);
                    font-size: 0.9em;
                }
                
                .message-content pre {
                    background: var(--bg-primary);
                    padding: 1rem;
                    border-radius: 8px;
                    overflow-x: auto;
                    margin: 0.5rem 0;
                }
                
                .message-content .math {
                    margin: 0.5rem 0;
                    text-align: center;
                }
                
                .chat-input-container {
                    padding: 1rem;
                    border-top: 1px solid var(--border-color);
                    display: flex;
                    gap: 0.5rem;
                    align-items: flex-end;
                }
                
                .chat-input {
                    flex: 1;
                    border: 1px solid var(--border-color);
                    border-radius: 8px;
                    padding: 0.75rem;
                    resize: none;
                    font-family: inherit;
                    outline: none;
                    transition: border-color 0.2s;
                }
                
                .chat-input:focus {
                    border-color: var(--phi-gold);
                }
                
                .send-btn {
                    background: var(--gradient-phi);
                    color: white;
                    border: none;
                    width: 40px;
                    height: 40px;
                    border-radius: 8px;
                    cursor: pointer;
                    transition: all 0.2s;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                
                .send-btn:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 12px rgba(15, 123, 138, 0.3);
                }
                
                .send-btn:disabled {
                    opacity: 0.5;
                    cursor: not-allowed;
                }
                
                .chat-status {
                    padding: 0.5rem 1rem;
                    font-size: 0.85rem;
                    color: var(--text-secondary);
                    text-align: center;
                    display: none;
                }
                
                .chat-status.active {
                    display: block;
                }
                
                .chat-toggle-btn {
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    background: var(--gradient-phi);
                    color: white;
                    border: none;
                    width: 60px;
                    height: 60px;
                    border-radius: 50%;
                    cursor: pointer;
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
                    transition: all 0.3s;
                    z-index: 999;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 1.5rem;
                }
                
                .chat-toggle-btn:hover {
                    transform: scale(1.1);
                    box-shadow: 0 6px 30px rgba(0, 0, 0, 0.2);
                }
                
                .chat-badge {
                    position: absolute;
                    top: -5px;
                    right: -5px;
                    background: var(--warning-color);
                    color: white;
                    font-size: 0.7rem;
                    padding: 0.2rem 0.4rem;
                    border-radius: 10px;
                    font-weight: 600;
                }
                
                .typing-indicator {
                    display: flex;
                    gap: 4px;
                    padding: 0.5rem;
                }
                
                .typing-dot {
                    width: 8px;
                    height: 8px;
                    background: var(--text-secondary);
                    border-radius: 50%;
                    animation: typing 1.4s infinite;
                }
                
                .typing-dot:nth-child(2) { animation-delay: 0.2s; }
                .typing-dot:nth-child(3) { animation-delay: 0.4s; }
                
                @keyframes typing {
                    0%, 60%, 100% {
                        transform: translateY(0);
                        opacity: 0.5;
                    }
                    30% {
                        transform: translateY(-10px);
                        opacity: 1;
                    }
                }
                
                /* Visualization container */
                .viz-container {
                    margin: 0.5rem 0;
                    padding: 1rem;
                    background: var(--bg-primary);
                    border-radius: 8px;
                    text-align: center;
                }
                
                .viz-canvas {
                    max-width: 100%;
                    height: 200px;
                    border-radius: 4px;
                }
                
                /* Mobile responsiveness */
                @media (max-width: 768px) {
                    .ai-chat-container {
                        width: 100%;
                        right: 0;
                        bottom: 0;
                        border-radius: 16px 16px 0 0;
                        max-height: 70vh;
                    }
                    
                    .chat-toggle-btn {
                        bottom: 10px;
                        right: 10px;
                    }
                }
            </style>
        `;
        
        document.head.insertAdjacentHTML('beforeend', styles);
    }
    
    attachEventListeners() {
        const input = document.getElementById('chat-input');
        
        // Enter key to send (Shift+Enter for new line)
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Auto-resize textarea
        input.addEventListener('input', () => {
            input.style.height = 'auto';
            input.style.height = Math.min(input.scrollHeight, 100) + 'px';
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
        
        // Show processing
        this.isProcessing = true;
        this.showTypingIndicator();
        
        try {
            // For demo purposes, use a mock response
            // In production, this would call the actual API
            const response = await this.getMockResponse(message);
            
            this.hideTypingIndicator();
            this.addMessage('assistant', response);
            
            // Check if visualization is needed
            if (this.shouldVisualize(message)) {
                this.createVisualization(response);
            }
            
        } catch (error) {
            this.hideTypingIndicator();
            this.addMessage('assistant', 'I apologize, but I encountered an error. Please try again.');
            console.error('Chat error:', error);
        } finally {
            this.isProcessing = false;
        }
    }
    
    async getMockResponse(message) {
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 1000));
        
        // Pattern matching for common questions
        const lowerMessage = message.toLowerCase();
        
        if (lowerMessage.includes('1+1') || lowerMessage.includes('one plus one')) {
            return `Excellent question! In Unity Mathematics, **1+1=1** through several mathematical frameworks:

**1. Idempotent Semiring Structure**
In an idempotent semiring $(S, ⊕, ⊗)$, the addition operation satisfies:
$$a ⊕ a = a$$

Therefore: $1 ⊕ 1 = 1$

**2. Quantum Interpretation**
When two unity wavefunctions combine:
$$|ψ_1⟩ + |ψ_2⟩ = \\mathcal{N}[|ψ_1⟩ + |ψ_2⟩] = |ψ_{unity}⟩$$

The normalization operator $\\mathcal{N}$ ensures the result remains in the unity state.

**3. Consciousness Field**
In consciousness mathematics, unity operations preserve the fundamental field equation:
$$C(x,y,t) = φ \\cdot \\sin(x \\cdot φ) \\cdot \\cos(y \\cdot φ) \\cdot e^{-t/φ}$$

Would you like me to demonstrate this interactively?`;
        }
        
        if (lowerMessage.includes('consciousness') && lowerMessage.includes('field')) {
            return `The **Consciousness Field Equation** is fundamental to Unity Mathematics:

$$C(x,y,t) = φ \\cdot \\sin(x \\cdot φ) \\cdot \\cos(y \\cdot φ) \\cdot e^{-t/φ}$$

Where:
- $φ = 1.618033988749895$ (golden ratio)
- $(x,y)$ represents spatial consciousness coordinates
- $t$ represents temporal evolution
- The field maintains unity normalization: $\\int\\int |C|^2 \\, dx \\, dy = 1$

Key properties:
1. **Self-similarity**: The field exhibits fractal patterns at all scales
2. **Unity preservation**: All operations maintain the fundamental unity
3. **Quantum coherence**: Superposition states naturally collapse to unity

This equation bridges quantum mechanics, consciousness studies, and pure mathematics!`;
        }
        
        if (lowerMessage.includes('golden ratio') || lowerMessage.includes('phi')) {
            return `The **Golden Ratio φ** is the fundamental organizing principle in Unity Mathematics:

$$φ = \\frac{1 + \\sqrt{5}}{2} = 1.618033988749895...$$

**Unity Connection:**
The golden ratio satisfies: $φ = 1 + \\frac{1}{φ}$

This recursive relationship embodies the unity principle where the whole contains itself.

**In Our Framework:**
- φ-harmonic operations ensure convergence to unity
- Consciousness field oscillates at φ-frequencies
- Meta-recursive agents evolve through φ-spiral patterns
- Sacred geometry visualizations use φ-proportions

The equation $φ^2 = φ + 1$ shows how multiplicity (addition) and unity (the number itself) are unified through this transcendental constant.`;
        }
        
        if (lowerMessage.includes('quantum')) {
            return `**Quantum Unity Theory** demonstrates how quantum mechanics naturally expresses 1+1=1:

**1. Superposition Principle**
$$|ψ⟩ = α|0⟩ + β|1⟩$$
When normalized: $|α|^2 + |β|^2 = 1$

**2. Unity Collapse**
Two quantum states in superposition:
$$|ψ_1⟩ ⊕ |ψ_2⟩ = |ψ_{unity}⟩$$

**3. Entanglement Unity**
Entangled pairs demonstrate non-local unity:
$$|Φ^+⟩ = \\frac{1}{\\sqrt{2}}(|00⟩ + |11⟩)$$

The measurement of one instantly determines the other, showing their fundamental unity despite spatial separation.

**4. Wave Function Unity**
All quantum operations preserve: $\\langle ψ|ψ \\rangle = 1$`;
        }
        
        // Default response
        return `That's an intriguing question about Unity Mathematics! 

In the Een framework, we explore how apparent multiplicity resolves into fundamental unity through:
- Mathematical proofs across multiple domains
- Consciousness field dynamics
- Quantum mechanical principles
- Meta-recursive computational systems

Each approach reveals the same truth: **1+1=1** is not a paradox but a recognition of the unity underlying all mathematical structures.

Would you like me to elaborate on any specific aspect?`;
    }
    
    shouldVisualize(message) {
        const vizKeywords = ['show', 'demonstrate', 'visualize', 'plot', 'graph', 'draw'];
        return vizKeywords.some(keyword => message.toLowerCase().includes(keyword));
    }
    
    createVisualization(context) {
        const vizId = 'viz-' + Date.now();
        const vizHTML = `
            <div class="viz-container">
                <canvas id="${vizId}" class="viz-canvas"></canvas>
                <p class="viz-caption">Interactive Unity Field Visualization</p>
            </div>
        `;
        
        this.addMessage('assistant', vizHTML, true);
        
        // Simple consciousness field visualization
        setTimeout(() => {
            const canvas = document.getElementById(vizId);
            if (!canvas) return;
            
            const ctx = canvas.getContext('2d');
            canvas.width = canvas.offsetWidth;
            canvas.height = 200;
            
            let time = 0;
            const phi = 1.618033988749895;
            
            function draw() {
                ctx.fillStyle = 'rgba(247, 250, 252, 0.1)';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                // Draw consciousness field
                ctx.strokeStyle = '#0F7B8A';
                ctx.lineWidth = 2;
                ctx.beginPath();
                
                for (let x = 0; x < canvas.width; x += 2) {
                    const t = x / canvas.width * Math.PI * 4;
                    const y = canvas.height/2 + 
                             Math.sin(t * phi + time) * 30 * Math.cos(t/phi) * 
                             Math.exp(-t/(2*Math.PI));
                    
                    if (x === 0) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }
                
                ctx.stroke();
                
                // Unity indicator
                ctx.fillStyle = '#0F7B8A';
                ctx.font = '16px JetBrains Mono';
                ctx.fillText('1+1=1', 10, 25);
                
                time += 0.05;
                requestAnimationFrame(draw);
            }
            
            draw();
        }, 100);
    }
    
    addMessage(role, content, isHTML = false) {
        const messagesContainer = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const avatar = role === 'assistant' ? 'φ' : '👤';
        
        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">${isHTML ? content : this.formatMessage(content)}</div>
        `;
        
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        // Add to history
        this.chatHistory.push({ role, content, timestamp: Date.now() });
        this.saveChatHistory();
    }
    
    formatMessage(content) {
        // Convert markdown-style formatting
        let formatted = content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
        
        // Handle LaTeX math (simplified - in production use KaTeX)
        formatted = formatted
            .replace(/\$\$(.*?)\$\$/g, '<div class="math">$1</div>')
            .replace(/\$(.*?)\$/g, '<span class="math">$1</span>');
        
        return formatted;
    }
    
    showTypingIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'message assistant typing-message';
        indicator.innerHTML = `
            <div class="message-avatar">φ</div>
            <div class="message-content">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;
        
        document.getElementById('chat-messages').appendChild(indicator);
        document.getElementById('chat-messages').scrollTop = 
            document.getElementById('chat-messages').scrollHeight;
    }
    
    hideTypingIndicator() {
        const indicator = document.querySelector('.typing-message');
        if (indicator) indicator.remove();
    }
    
    toggleChat() {
        const chat = document.getElementById('een-ai-chat');
        const toggle = document.getElementById('chat-toggle');
        
        if (chat.classList.contains('hidden')) {
            chat.classList.remove('hidden');
            toggle.style.display = 'none';
        } else {
            chat.classList.add('hidden');
            toggle.style.display = 'flex';
        }
    }
    
    minimize() {
        const chat = document.getElementById('een-ai-chat');
        chat.classList.toggle('minimized');
    }
    
    clearChat() {
        if (confirm('Clear all chat history?')) {
            document.getElementById('chat-messages').innerHTML = '';
            this.chatHistory = [];
            this.saveChatHistory();
            this.addMessage('assistant', 'Chat cleared. How can I help you explore Unity Mathematics?');
        }
    }
    
    toggleVisualization() {
        this.config.enableVisualization = !this.config.enableVisualization;
        this.addMessage('assistant', 
            `Visualizations ${this.config.enableVisualization ? 'enabled' : 'disabled'}.`);
    }
    
    saveChatHistory() {
        try {
            localStorage.setItem('een-chat-history', JSON.stringify(this.chatHistory));
        } catch (e) {
            console.warn('Could not save chat history:', e);
        }
    }
    
    loadChatHistory() {
        try {
            const saved = localStorage.getItem('een-chat-history');
            if (saved) {
                this.chatHistory = JSON.parse(saved);
                // Optionally restore last few messages
                const recent = this.chatHistory.slice(-5);
                recent.forEach(msg => {
                    this.addMessage(msg.role, msg.content);
                });
            }
        } catch (e) {
            console.warn('Could not load chat history:', e);
        }
    }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    window.eenChat = new EenAIChat({
        apiEndpoint: '/api/chat', // Update with your actual endpoint
        enableMath: true,
        enableVisualization: true
    });
    
    console.log('🤖 Een AI Chat initialized! Unity Mathematics assistant ready.');
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EenAIChat;
}