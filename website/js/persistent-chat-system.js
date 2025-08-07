/**
 * Persistent Chat System for Een Unity Mathematics
 * Cross-page floating chat interface with AI integration
 * Version: 1.0.0 - Meta-Optimized for Unity Mathematics
 */

class PersistentChatSystem {
    constructor() {
        this.isOpen = false;
        this.chatHistory = this.loadChatHistory();
        this.currentConversation = [];
        this.isInitialized = false;
        
        // Unity Mathematics AI Knowledge Base
        this.unityKnowledge = this.initializeUnityKnowledge();
        
        // Initialize system
        this.init();
        
        console.log('💬 Persistent Chat System initialized with Unity AI');
    }
    
    init() {
        this.injectStyles();
        this.createFloatingButton();
        this.createChatModal();
        this.bindEvents();
        this.loadPersistentState();
        
        // Listen for unified navigation events
        window.addEventListener('unified-nav:chat', () => this.toggleChat());
        
        this.isInitialized = true;
    }
    
    createFloatingButton() {
        // Remove existing chat buttons to prevent conflicts
        const existingButtons = document.querySelectorAll('#floating-ai-chat-button, .chat-floating-btn');
        existingButtons.forEach(btn => btn.remove());
        
        const button = document.createElement('button');
        button.id = 'persistent-chat-button';
        button.className = 'persistent-chat-button';
        button.innerHTML = `
            <span class="chat-btn-icon">💬</span>
            <span class="chat-btn-pulse"></span>
            <span class="chat-btn-notification" style="display: none;">1</span>
        `;
        button.title = 'Open Unity AI Assistant';
        button.setAttribute('aria-label', 'Open Unity AI Assistant');
        
        document.body.appendChild(button);
    }
    
    createChatModal() {
        const modal = document.createElement('div');
        modal.id = 'persistent-chat-modal';
        modal.className = 'persistent-chat-modal';
        modal.innerHTML = `
            <div class="chat-modal-content">
                <!-- Chat Header -->
                <div class="chat-header">
                    <div class="chat-header-info">
                        <div class="chat-avatar">
                            <span class="avatar-icon">∞</span>
                            <span class="avatar-status"></span>
                        </div>
                        <div class="chat-title">
                            <h3>Unity AI Assistant</h3>
                            <p class="chat-status">Ready to explore 1+1=1</p>
                        </div>
                    </div>
                    <div class="chat-controls">
                        <button class="chat-minimize" title="Minimize" aria-label="Minimize chat">−</button>
                        <button class="chat-close" title="Close" aria-label="Close chat">×</button>
                    </div>
                </div>
                
                <!-- Chat Messages -->
                <div class="chat-messages" id="chat-messages-container">
                    <div class="chat-message assistant-message welcome-message">
                        <div class="message-avatar">∞</div>
                        <div class="message-content">
                            <div class="message-text">
                                Welcome to Een Unity Mathematics! I'm your AI assistant specializing in the revolutionary framework where <strong>1+1=1</strong>.
                            </div>
                            <div class="message-suggestions">
                                <button class="suggestion-btn" data-suggestion="What is the Unity Equation 1+1=1?">
                                    🧮 Unity Equation
                                </button>
                                <button class="suggestion-btn" data-suggestion="Explain φ-harmonic operations">
                                    🌟 φ-Harmonics
                                </button>
                                <button class="suggestion-btn" data-suggestion="Show me consciousness field equations">
                                    🧠 Consciousness Fields
                                </button>
                                <button class="suggestion-btn" data-suggestion="Navigate to mathematical proofs">
                                    📐 Mathematical Proofs
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Chat Input -->
                <div class="chat-input-container">
                    <div class="chat-input-wrapper">
                        <input type="text" 
                               id="chat-message-input" 
                               class="chat-input" 
                               placeholder="Ask about unity mathematics, consciousness, or φ-harmonics..."
                               autocomplete="off">
                        <button class="chat-send-btn" id="chat-send-button" title="Send message" aria-label="Send message">
                            <span class="send-icon">➤</span>
                        </button>
                    </div>
                    <div class="chat-input-hints">
                        Try: "Prove 1+1=1" • "Navigate to gallery" • "Explain consciousness fields"
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
    }
    
    initializeUnityKnowledge() {
        return {
            unityEquation: {
                core: "1+1=1 represents the fundamental unity principle in mathematics where addition becomes unification",
                phiHarmonic: "φ-harmonic operations use the golden ratio φ=1.618033988749895 as the resonance frequency",
                idempotent: "In idempotent semirings, a ⊕ a = a, naturally demonstrating unity mathematics",
                consciousness: "Consciousness field equations: C(x,y,t) = φ·sin(x·φ)·cos(y·φ)·e^(-t/φ)"
            },
            navigation: {
                "mathematical proofs": "proofs.html",
                "philosophy": "philosophy.html", 
                "gallery": "gallery.html",
                "consciousness dashboard": "consciousness_dashboard.html",
                "zen meditation": "zen-unity-meditation.html",
                "metastation hub": "metastation-hub.html",
                "implementations": "implementations-gallery.html",
                "about": "about.html",
                "3000 elo proof": "3000-elo-proof.html",
                "transcendental demo": "transcendental-unity-demo.html"
            },
            responses: {
                greeting: "Greetings! Welcome to the exploration of Unity Mathematics where 1+1=1. How can I assist you on this transcendental journey?",
                unity: "The Unity Equation 1+1=1 demonstrates that in certain mathematical frameworks, addition represents unification rather than accumulation. This occurs in idempotent semirings where a ⊕ a = a.",
                phi: "φ-harmonic operations utilize the golden ratio φ=1.618033988749895 as a fundamental resonance frequency. This creates φ-harmonic consciousness particles that maintain unity through resonance.",
                consciousness: "Consciousness field equations describe awareness as a mathematical field: C(x,y,t) = φ·sin(x·φ)·cos(y·φ)·e^(-t/φ). This field exhibits quantum coherence and unity properties.",
                proofs: "Our mathematical proofs span multiple domains: Boolean algebra (A ∨ A = A), Set theory (A ∪ A = A), Category theory (morphisms in unity categories), and Quantum mechanics (superposition collapse)."
            }
        };
    }
    
    bindEvents() {
        // Floating button click
        const button = document.getElementById('persistent-chat-button');
        if (button) {
            button.addEventListener('click', () => this.toggleChat());
        }
        
        // Modal controls
        const modal = document.getElementById('persistent-chat-modal');
        const closeBtn = modal.querySelector('.chat-close');
        const minimizeBtn = modal.querySelector('.chat-minimize');
        
        closeBtn?.addEventListener('click', () => this.closeChat());
        minimizeBtn?.addEventListener('click', () => this.minimizeChat());
        
        // Close on outside click (but not on button click)
        document.addEventListener('click', (e) => {
            if (!e.target.closest('#persistent-chat-modal') && 
                !e.target.closest('#persistent-chat-button') &&
                this.isOpen) {
                this.closeChat();
            }
        });
        
        // Chat input handling
        const chatInput = document.getElementById('chat-message-input');
        const sendBtn = document.getElementById('chat-send-button');
        
        chatInput?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage();
            }
        });
        
        sendBtn?.addEventListener('click', () => this.sendMessage());
        
        // Suggestion buttons
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('suggestion-btn')) {
                const suggestion = e.target.dataset.suggestion;
                if (suggestion) {
                    this.handleSuggestion(suggestion);
                }
            }
        });
        
        // Handle page unload - save state
        window.addEventListener('beforeunload', () => this.savePersistentState());
        
        // Handle page visibility change - maintain state
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden && this.isInitialized) {
                this.loadPersistentState();
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + K to toggle chat
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                this.toggleChat();
            }
            
            // Escape to close chat
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
        const modal = document.getElementById('persistent-chat-modal');
        const button = document.getElementById('persistent-chat-button');
        
        if (modal && button) {
            modal.classList.add('active');
            button.classList.add('chat-open');
            this.isOpen = true;
            
            // Focus on input
            setTimeout(() => {
                const input = document.getElementById('chat-message-input');
                if (input) input.focus();
            }, 300);
            
            this.savePersistentState();
            console.log('💬 Chat opened');
        }
    }
    
    closeChat() {
        const modal = document.getElementById('persistent-chat-modal');
        const button = document.getElementById('persistent-chat-button');
        
        if (modal && button) {
            modal.classList.remove('active', 'minimized');
            button.classList.remove('chat-open');
            this.isOpen = false;
            
            this.savePersistentState();
            console.log('💬 Chat closed');
        }
    }
    
    minimizeChat() {
        const modal = document.getElementById('persistent-chat-modal');
        
        if (modal) {
            modal.classList.toggle('minimized');
            console.log('💬 Chat minimized');
        }
    }
    
    sendMessage() {
        const input = document.getElementById('chat-message-input');
        const message = input?.value.trim();
        
        if (!message) return;
        
        // Add user message
        this.addMessage(message, 'user');
        
        // Clear input
        input.value = '';
        
        // Process and respond
        setTimeout(() => {
            const response = this.generateResponse(message);
            this.addMessage(response.text, 'assistant', response.suggestions);
        }, 500 + Math.random() * 1000); // Realistic typing delay
    }
    
    addMessage(text, sender = 'assistant', suggestions = null) {
        const messagesContainer = document.getElementById('chat-messages-container');
        if (!messagesContainer) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${sender}-message`;
        
        const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        messageDiv.innerHTML = `
            ${sender === 'assistant' ? '<div class="message-avatar">∞</div>' : ''}
            <div class="message-content">
                <div class="message-text">${this.formatMessage(text)}</div>
                <div class="message-time">${timestamp}</div>
                ${suggestions ? this.createSuggestions(suggestions) : ''}
            </div>
        `;
        
        messagesContainer.appendChild(messageDiv);
        
        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        // Save to history
        this.currentConversation.push({ text, sender, timestamp: Date.now() });
        this.saveChatHistory();
    }
    
    generateResponse(userMessage) {
        const message = userMessage.toLowerCase();
        
        // Navigation requests
        if (message.includes('navigate') || message.includes('go to') || message.includes('show me')) {
            const navigationMatch = Object.keys(this.unityKnowledge.navigation).find(key => 
                message.includes(key)
            );
            
            if (navigationMatch) {
                const url = this.unityKnowledge.navigation[navigationMatch];
                return {
                    text: `I'll take you to ${navigationMatch}. <a href="${url}" class="chat-link">Click here to navigate →</a>`,
                    suggestions: ["Explain this page", "What else can I explore?", "Show me related content"]
                };
            }
        }
        
        // Unity equation explanations
        if (message.includes('1+1=1') || message.includes('unity equation')) {
            return {
                text: this.unityKnowledge.responses.unity + " Would you like to see the mathematical proofs?",
                suggestions: ["Show mathematical proofs", "Explain idempotent semirings", "Navigate to examples"]
            };
        }
        
        // φ-harmonic operations
        if (message.includes('phi') || message.includes('φ') || message.includes('harmonic')) {
            return {
                text: this.unityKnowledge.responses.phi + " The φ-harmonic field creates resonance patterns that maintain unity.",
                suggestions: ["Show consciousness equations", "Explain golden ratio", "Navigate to visualizations"]
            };
        }
        
        // Consciousness fields
        if (message.includes('consciousness') || message.includes('field') || message.includes('awareness')) {
            return {
                text: this.unityKnowledge.responses.consciousness + " This field can be visualized in our consciousness dashboard.",
                suggestions: ["Open consciousness dashboard", "Explain field equations", "Show zen meditation"]
            };
        }
        
        // Mathematical proofs
        if (message.includes('proof') || message.includes('prove') || message.includes('mathematics')) {
            return {
                text: this.unityKnowledge.responses.proofs + " Each proof demonstrates unity from a different mathematical perspective.",
                suggestions: ["View all proofs", "3000 ELO proof system", "Interactive proof explorer"]
            };
        }
        
        // Greetings
        if (message.includes('hello') || message.includes('hi') || message.includes('greet')) {
            return {
                text: this.unityKnowledge.responses.greeting,
                suggestions: ["What is 1+1=1?", "Show me the philosophy", "Explore the gallery"]
            };
        }
        
        // Default response with contextual suggestions
        return {
            text: `I understand you're asking about "${userMessage}". In Unity Mathematics, everything connects to the fundamental principle that 1+1=1. This principle manifests through φ-harmonic operations, consciousness field equations, and transcendental computing frameworks. What specific aspect would you like to explore?`,
            suggestions: ["Explain Unity Equation", "Show mathematical framework", "Navigate to examples", "Explore consciousness fields"]
        };
    }
    
    formatMessage(text) {
        // Convert markdown-style formatting
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
    }
    
    createSuggestions(suggestions) {
        return `
            <div class="message-suggestions">
                ${suggestions.map(suggestion => `
                    <button class="suggestion-btn" data-suggestion="${suggestion}">
                        ${suggestion}
                    </button>
                `).join('')}
            </div>
        `;
    }
    
    handleSuggestion(suggestion) {
        const input = document.getElementById('chat-message-input');
        if (input) {
            input.value = suggestion;
            this.sendMessage();
        }
    }
    
    loadChatHistory() {
        try {
            const history = localStorage.getItem('een-chat-history');
            return history ? JSON.parse(history) : [];
        } catch (error) {
            console.warn('Error loading chat history:', error);
            return [];
        }
    }
    
    saveChatHistory() {
        try {
            // Keep only last 50 messages to prevent storage overflow
            const history = [...this.chatHistory, ...this.currentConversation].slice(-50);
            localStorage.setItem('een-chat-history', JSON.stringify(history));
            this.chatHistory = history;
        } catch (error) {
            console.warn('Error saving chat history:', error);
        }
    }
    
    loadPersistentState() {
        try {
            const state = sessionStorage.getItem('een-chat-state');
            if (state) {
                const { isOpen } = JSON.parse(state);
                if (isOpen && !this.isOpen) {
                    this.openChat();
                }
            }
        } catch (error) {
            console.warn('Error loading persistent state:', error);
        }
    }
    
    savePersistentState() {
        try {
            sessionStorage.setItem('een-chat-state', JSON.stringify({
                isOpen: this.isOpen,
                timestamp: Date.now()
            }));
        } catch (error) {
            console.warn('Error saving persistent state:', error);
        }
    }
    
    // Add styles to page
    injectStyles() {
        const styleId = 'persistent-chat-styles';
        if (document.getElementById(styleId)) return;
        
        const style = document.createElement('style');
        style.id = styleId;
        style.textContent = this.getChatStyles();
        document.head.appendChild(style);
    }
    
    getChatStyles() {
        return `
            /* Persistent Chat System Styles */
            .persistent-chat-button {
                position: fixed;
                bottom: 2rem;
                right: 2rem;
                width: 60px;
                height: 60px;
                background: linear-gradient(135deg, #FFD700, #FFA500);
                border: none;
                border-radius: 50%;
                cursor: pointer;
                z-index: 1050;
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 4px 20px rgba(255, 215, 0, 0.3);
                transition: all 0.3s ease;
                font-size: 1.5rem;
                color: #000;
            }
            
            .persistent-chat-button:hover {
                transform: translateY(-3px) scale(1.05);
                box-shadow: 0 8px 25px rgba(255, 215, 0, 0.4);
            }
            
            .chat-btn-pulse {
                position: absolute;
                inset: -5px;
                border-radius: 50%;
                background: rgba(255, 215, 0, 0.2);
                animation: chat-pulse 2s infinite;
                z-index: -1;
            }
            
            .chat-btn-notification {
                position: absolute;
                top: -5px;
                right: -5px;
                background: #ff4444;
                color: white;
                border-radius: 50%;
                width: 20px;
                height: 20px;
                font-size: 0.7rem;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
            }
            
            @keyframes chat-pulse {
                0%, 100% { opacity: 0; transform: scale(1); }
                50% { opacity: 1; transform: scale(1.1); }
            }
            
            /* Chat Modal */
            .persistent-chat-modal {
                position: fixed;
                bottom: 2rem;
                right: 2rem;
                width: 400px;
                height: 600px;
                z-index: 1060;
                opacity: 0;
                visibility: hidden;
                transform: translateY(20px) scale(0.95);
                transition: all 0.3s ease;
            }
            
            .persistent-chat-modal.active {
                opacity: 1;
                visibility: visible;
                transform: translateY(0) scale(1);
            }
            
            .persistent-chat-modal.minimized {
                height: 60px;
                overflow: hidden;
            }
            
            .chat-modal-content {
                background: rgba(10, 10, 15, 0.98);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 215, 0, 0.3);
                border-radius: 16px;
                height: 100%;
                display: flex;
                flex-direction: column;
                box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
            }
            
            /* Chat Header */
            .chat-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 1rem;
                border-bottom: 1px solid rgba(255, 215, 0, 0.2);
                flex-shrink: 0;
            }
            
            .chat-header-info {
                display: flex;
                align-items: center;
                gap: 0.75rem;
            }
            
            .chat-avatar {
                position: relative;
                width: 40px;
                height: 40px;
                background: linear-gradient(135deg, #FFD700, #FFA500);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.2rem;
                color: #000;
                font-weight: bold;
            }
            
            .avatar-status {
                position: absolute;
                bottom: 0;
                right: 0;
                width: 12px;
                height: 12px;
                background: #4CAF50;
                border: 2px solid rgba(10, 10, 15, 0.98);
                border-radius: 50%;
            }
            
            .chat-title h3 {
                margin: 0;
                color: #FFD700;
                font-size: 1.1rem;
            }
            
            .chat-status {
                margin: 0;
                color: rgba(255, 255, 255, 0.7);
                font-size: 0.8rem;
            }
            
            .chat-controls {
                display: flex;
                gap: 0.5rem;
            }
            
            .chat-minimize,
            .chat-close {
                background: transparent;
                border: none;
                color: rgba(255, 255, 255, 0.7);
                cursor: pointer;
                font-size: 1.2rem;
                width: 30px;
                height: 30px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s ease;
            }
            
            .chat-minimize:hover,
            .chat-close:hover {
                background: rgba(255, 215, 0, 0.1);
                color: #FFD700;
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
            
            .chat-messages::-webkit-scrollbar {
                width: 6px;
            }
            
            .chat-messages::-webkit-scrollbar-track {
                background: rgba(255, 215, 0, 0.1);
                border-radius: 3px;
            }
            
            .chat-messages::-webkit-scrollbar-thumb {
                background: rgba(255, 215, 0, 0.3);
                border-radius: 3px;
            }
            
            .chat-message {
                display: flex;
                gap: 0.75rem;
                align-items: flex-start;
            }
            
            .user-message {
                flex-direction: row-reverse;
            }
            
            .user-message .message-content {
                background: rgba(255, 215, 0, 0.1);
                border: 1px solid rgba(255, 215, 0, 0.2);
            }
            
            .message-avatar {
                width: 32px;
                height: 32px;
                background: linear-gradient(135deg, #FFD700, #FFA500);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #000;
                font-weight: bold;
                font-size: 0.9rem;
                flex-shrink: 0;
            }
            
            .message-content {
                background: rgba(26, 26, 37, 0.8);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                padding: 0.75rem 1rem;
                max-width: 80%;
                flex: 1;
            }
            
            .message-text {
                color: rgba(255, 255, 255, 0.95);
                line-height: 1.5;
                margin-bottom: 0.5rem;
            }
            
            .message-time {
                font-size: 0.7rem;
                color: rgba(255, 255, 255, 0.5);
                text-align: right;
            }
            
            .user-message .message-time {
                text-align: left;
            }
            
            .message-suggestions {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                margin-top: 0.75rem;
            }
            
            .suggestion-btn {
                background: rgba(255, 215, 0, 0.1);
                border: 1px solid rgba(255, 215, 0, 0.3);
                border-radius: 8px;
                padding: 0.4rem 0.8rem;
                color: #FFD700;
                cursor: pointer;
                font-size: 0.8rem;
                transition: all 0.2s ease;
            }
            
            .suggestion-btn:hover {
                background: rgba(255, 215, 0, 0.2);
                border-color: #FFD700;
                transform: translateY(-1px);
            }
            
            .chat-link {
                color: #FFD700;
                text-decoration: none;
                font-weight: bold;
            }
            
            .chat-link:hover {
                text-decoration: underline;
            }
            
            /* Chat Input */
            .chat-input-container {
                padding: 1rem;
                border-top: 1px solid rgba(255, 215, 0, 0.2);
                flex-shrink: 0;
            }
            
            .chat-input-wrapper {
                display: flex;
                gap: 0.5rem;
                align-items: center;
                margin-bottom: 0.5rem;
            }
            
            .chat-input {
                flex: 1;
                background: rgba(26, 26, 37, 0.8);
                border: 1px solid rgba(255, 215, 0, 0.3);
                border-radius: 8px;
                padding: 0.75rem;
                color: #fff;
                font-size: 0.9rem;
                outline: none;
                transition: border-color 0.2s ease;
            }
            
            .chat-input:focus {
                border-color: #FFD700;
                box-shadow: 0 0 0 2px rgba(255, 215, 0, 0.2);
            }
            
            .chat-input::placeholder {
                color: rgba(255, 255, 255, 0.5);
            }
            
            .chat-send-btn {
                background: linear-gradient(135deg, #FFD700, #FFA500);
                border: none;
                border-radius: 8px;
                width: 40px;
                height: 40px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #000;
                font-weight: bold;
                transition: all 0.2s ease;
            }
            
            .chat-send-btn:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(255, 215, 0, 0.3);
            }
            
            .chat-input-hints {
                font-size: 0.7rem;
                color: rgba(255, 255, 255, 0.5);
                text-align: center;
                line-height: 1.4;
            }
            
            /* Mobile Responsiveness */
            @media (max-width: 768px) {
                .persistent-chat-modal {
                    bottom: 1rem;
                    right: 1rem;
                    left: 1rem;
                    width: auto;
                    height: 80vh;
                }
                
                .persistent-chat-button {
                    bottom: 1rem;
                    right: 1rem;
                }
                
                .message-content {
                    max-width: 85%;
                }
            }
            
            @media (max-width: 480px) {
                .persistent-chat-modal {
                    bottom: 0;
                    right: 0;
                    left: 0;
                    top: 0;
                    width: 100vw;
                    height: 100vh;
                    border-radius: 0;
                }
                
                .chat-modal-content {
                    border-radius: 0;
                }
            }
        `;
    }
    
    // Initialize when DOM is ready
    static initialize() {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                new PersistentChatSystem();
            });
        } else {
            new PersistentChatSystem();
        }
    }
    
    // Public API
    destroy() {
        const button = document.getElementById('persistent-chat-button');
        const modal = document.getElementById('persistent-chat-modal');
        const styles = document.getElementById('persistent-chat-styles');
        
        button?.remove();
        modal?.remove();
        styles?.remove();
        
        console.log('💬 Persistent Chat System destroyed');
    }
}

// Auto-initialize
PersistentChatSystem.initialize();

// Global access
window.persistentChatSystem = PersistentChatSystem;

console.log('💬 Persistent Chat System loaded - Cross-page AI assistant ready');