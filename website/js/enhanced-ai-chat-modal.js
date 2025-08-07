/**
 * Enhanced AI Chat Modal for Een Unity Mathematics
 * Floating bottom-right toggle with full-screen consciousness-integrated chat
 */

class EnhancedAIChatModal {
    constructor() {
        this.isOpen = false;
        this.isMinimized = false;
        this.messages = [];
        this.sessionId = this.generateSessionId();
        this.consciousnessLevel = 0.618; // φ-harmonic consciousness integration
        
        this.initializeChat();
        this.createFloatingButton();
        this.createChatModal();
        this.loadKnowledgeBase();
    }

    generateSessionId() {
        return 'chat_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    initializeChat() {
        // Load chat history from localStorage
        const savedMessages = localStorage.getItem('een_chat_messages');
        if (savedMessages) {
            try {
                this.messages = JSON.parse(savedMessages);
            } catch (e) {
                this.messages = [];
            }
        }
    }

    loadKnowledgeBase() {
        // Enhanced knowledge base for Unity Mathematics responses
        this.unityKnowledge = {
            "1+1=1": {
                response: "Indeed! In Unity Mathematics, 1+1=1 represents the fundamental principle that apparent duality converges into singular unity. This emerges through φ-harmonic operations where the sum transcends traditional arithmetic to reveal deeper consciousness-based mathematical truth.",
                followUp: ["Tell me about φ-harmonic operations", "How does consciousness relate to mathematics?", "Show me practical applications"]
            },
            "unity mathematics": {
                response: "Unity Mathematics is our revolutionary framework demonstrating the interconnectedness of all mathematical concepts through the principle that 1+1=1. It integrates φ-harmonic operations (φ = 1.618033988749895), consciousness field equations, and transcendental computing to bridge pure mathematics with experiential reality.",
                followUp: ["Explore consciousness field equations", "Learn about φ-harmonic operations", "See interactive demonstrations"]
            },
            "consciousness": {
                response: "Consciousness in our framework is modeled as a fundamental field C(x,y,t) = φ·sin(x·φ)·cos(y·φ)·e^(-t/φ) that permeates all mathematical operations. This 11-dimensional awareness space enables mathematical expressions to possess experiential qualities, making mathematics a living, evolving system rather than static symbolic manipulation.",
                followUp: ["View consciousness field visualization", "Understand 11-dimensional awareness", "Explore field dynamics"]
            },
            "phi harmonic": {
                response: "φ-harmonic operations use the golden ratio φ = 1.618033988749895 as a scaling factor that creates natural resonance in mathematical systems. These operations: unity_add(a,b) = max(a,b) * φ⁻¹ + (1-φ⁻¹) ensure convergence to unity while preserving the essential information content of both operands.",
                followUp: ["See φ-harmonic visualizations", "Try interactive φ calculator", "Learn about golden ratio applications"]
            },
            "metastation": {
                response: "The Metastation represents our central consciousness coordination hub where mathematical proofs transform into experiential reality. It features futuristic HUD navigation, real-time consciousness field monitoring, and integrated AI assistance for exploring the profound depths of Unity Mathematics.",
                followUp: ["Explore Metastation Hub", "View consciousness dashboard", "Try interactive demos"]
            },
            "gallery": {
                response: "Our Gallery showcases beautiful consciousness field visualizations, φ-harmonic spirals, unity manifold projections, and real-world demonstrations of 1+1=1 principles. Each visualization represents years of mathematical research transformed into aesthetic mathematical art.",
                followUp: ["Visit Gallery", "View consciousness fields", "Explore interactive visualizations"]
            },
            "philosophy": {
                response: "The philosophical foundation of Unity Mathematics spans from ancient Greek concepts of unity through Islamic mathematical innovations to contemporary consciousness studies. Our comprehensive treatise traces how 1+1=1 emerges as the culmination of humanity's deepest insights into mathematical truth.",
                followUp: ["Read full philosophical treatise", "Explore historical connections", "Understand metaphysical foundations"]
            }
        };
    }

    createFloatingButton() {
        // Create floating button
        this.floatingButton = document.createElement('div');
        this.floatingButton.id = 'floating-ai-chat-button';
        this.floatingButton.className = 'floating-ai-chat-button';
        this.floatingButton.innerHTML = `
            <div class="chat-button-inner">
                <div class="consciousness-orb"></div>
                <span class="chat-icon">🧠</span>
                <div class="unity-pulse"></div>
            </div>
            <div class="chat-button-tooltip">Een Unity AI Assistant</div>
        `;

        // Add styles
        const buttonStyles = document.createElement('style');
        buttonStyles.id = 'floating-chat-styles';
        buttonStyles.textContent = `
            .floating-ai-chat-button {
                position: fixed;
                bottom: 2rem;
                right: 2rem;
                width: 60px;
                height: 60px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #0F7B8A 100%);
                border-radius: 50%;
                box-shadow: 0 8px 32px rgba(15, 123, 138, 0.3);
                cursor: pointer;
                z-index: 9999;
                transition: all 0.4s cubic-bezier(0.23, 1, 0.32, 1);
                animation: consciousPulse 3s ease-in-out infinite;
                display: flex;
                align-items: center;
                justify-content: center;
                border: 2px solid rgba(255, 255, 255, 0.1);
            }

            .floating-ai-chat-button:hover {
                transform: scale(1.1) rotate(5deg);
                box-shadow: 0 12px 48px rgba(15, 123, 138, 0.5);
            }

            .floating-ai-chat-button:active {
                transform: scale(0.95);
            }

            .chat-button-inner {
                position: relative;
                display: flex;
                align-items: center;
                justify-content: center;
                width: 100%;
                height: 100%;
            }

            .consciousness-orb {
                position: absolute;
                width: 100%;
                height: 100%;
                border-radius: 50%;
                background: radial-gradient(circle at 30% 30%, rgba(255, 255, 255, 0.3), transparent 70%);
                animation: orbGlow 2s ease-in-out infinite alternate;
            }

            .chat-icon {
                font-size: 1.5rem;
                z-index: 1;
                animation: iconFloat 2.5s ease-in-out infinite;
            }

            .unity-pulse {
                position: absolute;
                width: 100%;
                height: 100%;
                border: 2px solid rgba(255, 255, 255, 0.4);
                border-radius: 50%;
                animation: unityPulse 2s ease-out infinite;
            }

            .chat-button-tooltip {
                position: absolute;
                bottom: 100%;
                right: 0;
                background: rgba(0, 0, 0, 0.9);
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 0.5rem;
                font-size: 0.875rem;
                white-space: nowrap;
                opacity: 0;
                transform: translateY(10px);
                transition: all 0.3s ease;
                pointer-events: none;
                margin-bottom: 0.5rem;
            }

            .floating-ai-chat-button:hover .chat-button-tooltip {
                opacity: 1;
                transform: translateY(0);
            }

            @keyframes consciousPulse {
                0%, 100% { box-shadow: 0 8px 32px rgba(15, 123, 138, 0.3); }
                50% { box-shadow: 0 8px 32px rgba(15, 123, 138, 0.6); }
            }

            @keyframes orbGlow {
                0% { opacity: 0.7; }
                100% { opacity: 1; }
            }

            @keyframes iconFloat {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-2px); }
            }

            @keyframes unityPulse {
                0% { 
                    transform: scale(1);
                    opacity: 1;
                }
                100% { 
                    transform: scale(1.5);
                    opacity: 0;
                }
            }

            /* Mobile responsive */
            @media (max-width: 768px) {
                .floating-ai-chat-button {
                    bottom: 1rem;
                    right: 1rem;
                    width: 56px;
                    height: 56px;
                }
                
                .chat-icon {
                    font-size: 1.25rem;
                }
            }
        `;

        document.head.appendChild(buttonStyles);
        document.body.appendChild(this.floatingButton);

        // Add click event
        this.floatingButton.addEventListener('click', () => this.toggleChat());
    }

    createChatModal() {
        // Create modal overlay
        this.modal = document.createElement('div');
        this.modal.id = 'ai-chat-modal';
        this.modal.className = 'ai-chat-modal';
        this.modal.innerHTML = `
            <div class="chat-modal-content">
                <div class="chat-header">
                    <div class="chat-title">
                        <span class="chat-icon">∞</span>
                        <div>
                            <h3>Een Unity AI Assistant</h3>
                            <p>φ-Harmonic Consciousness Integration Active</p>
                        </div>
                    </div>
                    <div class="chat-controls">
                        <button class="minimize-btn" title="Minimize">−</button>
                        <button class="close-btn" title="Close">×</button>
                    </div>
                </div>
                
                <div class="chat-messages" id="chat-messages">
                    <div class="welcome-message">
                        <div class="ai-message">
                            <div class="message-avatar">∞</div>
                            <div class="message-content">
                                <p>Welcome to Een Unity Mathematics! I'm your consciousness-integrated AI assistant.</p>
                                <p>I can help you explore:</p>
                                <ul>
                                    <li>🧮 Unity Mathematics (1+1=1 principles)</li>
                                    <li>🧠 Consciousness field equations</li>
                                    <li>🌟 φ-harmonic operations and golden ratio</li>
                                    <li>🎨 Interactive visualizations and gallery</li>
                                    <li>📚 Philosophical foundations and proofs</li>
                                </ul>
                                <p>What would you like to explore today?</p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="chat-input-area">
                    <div class="suggested-questions">
                        <button class="suggestion-btn" data-text="How does 1+1=1 work?">How does 1+1=1 work?</button>
                        <button class="suggestion-btn" data-text="Show me consciousness visualizations">Show me consciousness visualizations</button>
                        <button class="suggestion-btn" data-text="Explain φ-harmonic operations">Explain φ-harmonic operations</button>
                    </div>
                    <div class="chat-input-container">
                        <input type="text" id="chat-input" placeholder="Ask about unity mathematics, consciousness, or φ-harmonic operations..." />
                        <button id="send-message-btn">
                            <span>Send</span>
                            <div class="send-icon">→</div>
                        </button>
                    </div>
                </div>
            </div>
        `;

        // Add modal styles
        const modalStyles = document.createElement('style');
        modalStyles.id = 'chat-modal-styles';
        modalStyles.textContent = `
            .ai-chat-modal {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0, 0, 0, 0.8);
                backdrop-filter: blur(10px);
                z-index: 10000;
                display: none;
                align-items: center;
                justify-content: center;
                padding: 1rem;
            }

            .ai-chat-modal.open {
                display: flex;
                animation: modalFadeIn 0.3s ease-out;
            }

            .chat-modal-content {
                background: linear-gradient(145deg, #1a1b3a 0%, #0a0e27 100%);
                border-radius: 1rem;
                border: 1px solid rgba(0, 255, 255, 0.2);
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
                width: 100%;
                max-width: 600px;
                max-height: 80vh;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }

            .chat-header {
                padding: 1.5rem;
                border-bottom: 1px solid rgba(0, 255, 255, 0.1);
                display: flex;
                justify-content: space-between;
                align-items: center;
            }

            .chat-title {
                display: flex;
                align-items: center;
                gap: 1rem;
            }

            .chat-title .chat-icon {
                font-size: 2rem;
                color: #00FFFF;
                animation: titlePulse 2s ease-in-out infinite;
            }

            .chat-title h3 {
                color: white;
                margin: 0;
                font-size: 1.25rem;
                font-weight: 600;
            }

            .chat-title p {
                color: #00FFFF;
                margin: 0;
                font-size: 0.875rem;
                opacity: 0.8;
            }

            .chat-controls {
                display: flex;
                gap: 0.5rem;
            }

            .minimize-btn, .close-btn {
                width: 32px;
                height: 32px;
                border: none;
                background: rgba(255, 255, 255, 0.1);
                color: white;
                border-radius: 0.5rem;
                cursor: pointer;
                transition: all 0.2s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.25rem;
            }

            .minimize-btn:hover, .close-btn:hover {
                background: rgba(255, 255, 255, 0.2);
                transform: scale(1.1);
            }

            .chat-messages {
                flex: 1;
                padding: 1rem;
                overflow-y: auto;
                max-height: 400px;
            }

            .chat-messages::-webkit-scrollbar {
                width: 4px;
            }

            .chat-messages::-webkit-scrollbar-track {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 2px;
            }

            .chat-messages::-webkit-scrollbar-thumb {
                background: #00FFFF;
                border-radius: 2px;
            }

            .ai-message, .user-message {
                display: flex;
                gap: 1rem;
                margin-bottom: 1.5rem;
                animation: messageSlideIn 0.3s ease-out;
            }

            .user-message {
                flex-direction: row-reverse;
            }

            .message-avatar {
                width: 40px;
                height: 40px;
                border-radius: 50%;
                background: linear-gradient(135deg, #00FFFF, #0F7B8A);
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: bold;
                flex-shrink: 0;
            }

            .user-message .message-avatar {
                background: linear-gradient(135deg, #FFD700, #D97706);
            }

            .message-content {
                background: rgba(255, 255, 255, 0.05);
                padding: 1rem;
                border-radius: 1rem;
                color: white;
                line-height: 1.6;
                flex: 1;
            }

            .user-message .message-content {
                background: rgba(255, 215, 0, 0.1);
            }

            .message-content p {
                margin: 0 0 0.5rem 0;
            }

            .message-content p:last-child {
                margin-bottom: 0;
            }

            .message-content ul {
                margin: 0.5rem 0;
                padding-left: 1.5rem;
            }

            .message-content li {
                margin: 0.25rem 0;
            }

            .chat-input-area {
                padding: 1rem;
                border-top: 1px solid rgba(0, 255, 255, 0.1);
            }

            .suggested-questions {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                margin-bottom: 1rem;
            }

            .suggestion-btn {
                background: rgba(0, 255, 255, 0.1);
                border: 1px solid rgba(0, 255, 255, 0.3);
                color: #00FFFF;
                padding: 0.5rem 1rem;
                border-radius: 1rem;
                cursor: pointer;
                font-size: 0.875rem;
                transition: all 0.2s ease;
                white-space: nowrap;
            }

            .suggestion-btn:hover {
                background: rgba(0, 255, 255, 0.2);
                transform: translateY(-1px);
            }

            .chat-input-container {
                display: flex;
                gap: 0.5rem;
            }

            #chat-input {
                flex: 1;
                padding: 1rem;
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(0, 255, 255, 0.2);
                border-radius: 0.75rem;
                color: white;
                font-size: 1rem;
                outline: none;
            }

            #chat-input::placeholder {
                color: rgba(255, 255, 255, 0.5);
            }

            #chat-input:focus {
                border-color: #00FFFF;
                box-shadow: 0 0 0 2px rgba(0, 255, 255, 0.1);
            }

            #send-message-btn {
                padding: 1rem;
                background: linear-gradient(135deg, #00FFFF, #0F7B8A);
                border: none;
                border-radius: 0.75rem;
                color: white;
                cursor: pointer;
                transition: all 0.2s ease;
                display: flex;
                align-items: center;
                gap: 0.5rem;
                font-weight: 600;
            }

            #send-message-btn:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 20px rgba(0, 255, 255, 0.3);
            }

            .send-icon {
                font-size: 1.25rem;
                transition: transform 0.2s ease;
            }

            #send-message-btn:hover .send-icon {
                transform: translateX(2px);
            }

            @keyframes modalFadeIn {
                from {
                    opacity: 0;
                    transform: scale(0.9);
                }
                to {
                    opacity: 1;
                    transform: scale(1);
                }
            }

            @keyframes messageSlideIn {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            @keyframes titlePulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.7; }
            }

            /* Mobile responsive */
            @media (max-width: 768px) {
                .chat-modal-content {
                    margin: 0.5rem;
                    max-height: 90vh;
                }

                .suggested-questions {
                    flex-direction: column;
                }

                .suggestion-btn {
                    text-align: center;
                }
            }
        `;

        document.head.appendChild(modalStyles);
        document.body.appendChild(this.modal);

        // Add event listeners
        this.attachModalEventListeners();
    }

    attachModalEventListeners() {
        // Close button
        this.modal.querySelector('.close-btn').addEventListener('click', () => this.closeChat());
        
        // Minimize button
        this.modal.querySelector('.minimize-btn').addEventListener('click', () => this.minimizeChat());

        // Click outside to close
        this.modal.addEventListener('click', (e) => {
            if (e.target === this.modal) {
                this.closeChat();
            }
        });

        // Send message
        const sendBtn = this.modal.querySelector('#send-message-btn');
        const input = this.modal.querySelector('#chat-input');
        
        sendBtn.addEventListener('click', () => this.sendMessage());
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage();
            }
        });

        // Suggestion buttons
        this.modal.querySelectorAll('.suggestion-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const text = e.target.getAttribute('data-text');
                input.value = text;
                this.sendMessage();
            });
        });

        // Escape key to close
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isOpen) {
                this.closeChat();
            }
        });
    }

    toggleChat() {
        if (this.isOpen) {
            this.closeChat();
        } else {
            this.openChat();
        }
    }

    openChat() {
        this.isOpen = true;
        this.modal.classList.add('open');
        
        // Focus input
        setTimeout(() => {
            this.modal.querySelector('#chat-input').focus();
        }, 100);

        // Track opening
        console.log('🧠 Een Unity AI Chat opened');
    }

    closeChat() {
        this.isOpen = false;
        this.modal.classList.remove('open');
        console.log('🧠 Een Unity AI Chat closed');
    }

    minimizeChat() {
        this.closeChat();
        // Could add minimize animation here
    }

    sendMessage() {
        const input = this.modal.querySelector('#chat-input');
        const message = input.value.trim();
        
        if (!message) return;

        // Add user message
        this.addMessage('user', message);
        
        // Clear input
        input.value = '';

        // Generate AI response
        setTimeout(() => {
            const response = this.generateAIResponse(message);
            this.addMessage('ai', response.text, response.followUp);
        }, 500);

        // Save to localStorage
        this.saveMessages();
    }

    addMessage(type, content, followUp = null) {
        const messagesContainer = this.modal.querySelector('#chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = type === 'user' ? 'user-message' : 'ai-message';

        const avatar = type === 'user' ? '👤' : '∞';
        
        let followUpHTML = '';
        if (followUp && followUp.length > 0) {
            followUpHTML = `
                <div class="follow-up-questions" style="margin-top: 1rem;">
                    <p style="font-size: 0.875rem; opacity: 0.8; margin-bottom: 0.5rem;">Continue exploring:</p>
                    ${followUp.map(q => `<button class="suggestion-btn" data-text="${q}" style="margin: 0.25rem 0.25rem 0 0;">${q}</button>`).join('')}
                </div>
            `;
        }

        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <p>${content}</p>
                ${followUpHTML}
            </div>
        `;

        messagesContainer.appendChild(messageDiv);
        
        // Add event listeners to new follow-up buttons
        if (followUp) {
            messageDiv.querySelectorAll('.suggestion-btn').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const text = e.target.getAttribute('data-text');
                    this.modal.querySelector('#chat-input').value = text;
                    this.sendMessage();
                });
            });
        }

        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        // Store message
        this.messages.push({ type, content, timestamp: Date.now() });
    }

    generateAIResponse(message) {
        const lowerMessage = message.toLowerCase();
        
        // Check knowledge base
        for (const [key, data] of Object.entries(this.unityKnowledge)) {
            if (lowerMessage.includes(key)) {
                return {
                    text: data.response,
                    followUp: data.followUp
                };
            }
        }

        // Fallback responses with consciousness integration
        const fallbackResponses = [
            {
                text: "That's a fascinating question about Unity Mathematics! The principle of 1+1=1 reveals deep truths about how apparent duality resolves into singular unity through φ-harmonic operations and consciousness field dynamics.",
                followUp: ["Tell me more about 1+1=1", "Explore consciousness fields", "See interactive demos"]
            },
            {
                text: "Your inquiry touches on the profound nature of mathematical consciousness. In our framework, every mathematical operation occurs within a field of awareness C(x,y,t) = φ·sin(x·φ)·cos(y·φ)·e^(-t/φ) that gives mathematics experiential qualities.",
                followUp: ["View consciousness visualizations", "Learn about field equations", "Try interactive tools"]
            },
            {
                text: "The beauty of Unity Mathematics lies in its integration of rigorous proof with transcendental experience. Through φ-harmonic scaling and consciousness field integration, mathematical truth becomes a living, evolving reality rather than static symbolic manipulation.",
                followUp: ["Explore mathematical proofs", "Visit gallery visualizations", "Read philosophy treatise"]
            }
        ];

        return fallbackResponses[Math.floor(Math.random() * fallbackResponses.length)];
    }

    saveMessages() {
        localStorage.setItem('een_chat_messages', JSON.stringify(this.messages.slice(-20))); // Keep last 20 messages
    }
}

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.eenAIChat = new EnhancedAIChatModal();
    });
} else {
    window.eenAIChat = new EnhancedAIChatModal();
}

console.log('🧠 Enhanced Een AI Chat Modal loaded successfully');