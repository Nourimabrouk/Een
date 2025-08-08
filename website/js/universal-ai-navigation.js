/**
 * Universal AI Chat & Navigation System for All Pages
 * Ensures consistent experience across the entire Een Unity Mathematics website
 */

class UniversalAINavigation {
    constructor() {
        this.aiChat = null;
        this.sideNavActive = false;
        this.chatModels = [
            { id: 'gpt-4o', name: 'GPT-4o', provider: 'openai', description: 'Latest OpenAI model with enhanced reasoning' },
            { id: 'gpt-5', name: 'GPT-5', provider: 'openai', description: 'Next-generation OpenAI model (Preview)' },
            { id: 'claude-3-5-sonnet-20241022', name: 'Claude 3.5 Sonnet', provider: 'anthropic', description: 'Advanced reasoning and analysis' },
            { id: 'gemini-pro', name: 'Gemini Pro', provider: 'google', description: 'Google\'s multimodal AI model' }
        ];
        this.currentModel = 'gpt-4o';
        this.init();
    }

    init() {
        this.injectStyles();
        this.createSideNavigation();
        this.createAIChatButton();
        this.createAIChatModal();
        this.setupEventListeners();
    }

    injectStyles() {
        const styles = `
            <style id="universal-ai-nav-styles">
                /* Universal Side Navigation */
                .universal-side-nav {
                    position: fixed;
                    left: -300px;
                    top: 0;
                    width: 300px;
                    height: 100vh;
                    background: rgba(18, 18, 26, 0.98);
                    backdrop-filter: blur(20px);
                    border-right: 1px solid rgba(255, 215, 0, 0.3);
                    z-index: 2000;
                    padding: 1rem;
                    overflow-y: auto;
                    transition: left 0.3s ease;
                }

                .universal-side-nav.active {
                    left: 0;
                }

                .universal-side-nav-toggle {
                    position: fixed;
                    left: 1rem;
                    top: 50%;
                    transform: translateY(-50%);
                    width: 50px;
                    height: 50px;
                    background: linear-gradient(135deg, #FFD700, #00D4FF);
                    border: none;
                    border-radius: 0 25px 25px 0;
                    cursor: pointer;
                    z-index: 2001;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 1.2rem;
                    color: white;
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
                    transition: all 0.3s;
                }

                .universal-side-nav-toggle:hover {
                    transform: translateY(-50%) scale(1.1);
                }

                .nav-section {
                    margin-bottom: 2rem;
                }

                .nav-section-title {
                    color: #FFD700;
                    font-size: 0.9rem;
                    font-weight: 600;
                    text-transform: uppercase;
                    letter-spacing: 0.1em;
                    margin-bottom: 1rem;
                    opacity: 0.9;
                }

                .nav-link {
                    display: block;
                    padding: 0.75rem 1rem;
                    color: rgba(255, 255, 255, 0.8);
                    text-decoration: none;
                    border-radius: 8px;
                    margin-bottom: 0.25rem;
                    transition: all 0.3s;
                    border: 1px solid transparent;
                }

                .nav-link:hover {
                    background: rgba(255, 215, 0, 0.1);
                    color: #FFD700;
                    transform: translateX(5px);
                    border-color: rgba(255, 215, 0, 0.3);
                }

                .nav-link i {
                    margin-right: 0.5rem;
                    width: 1.2rem;
                }

                /* Universal AI Chat Button */
                .universal-ai-chat-button {
                    position: fixed;
                    bottom: 2rem;
                    right: 2rem;
                    width: 60px;
                    height: 60px;
                    background: linear-gradient(135deg, #FFD700, #9D4EDD, #00D4FF);
                    border: none;
                    border-radius: 50%;
                    cursor: pointer;
                    z-index: 1999;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 1.5rem;
                    color: white;
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
                    transition: all 0.3s;
                    animation: aiPulse 3s ease-in-out infinite;
                }

                .universal-ai-chat-button:hover {
                    transform: scale(1.1);
                    box-shadow: 0 15px 40px rgba(255, 215, 0, 0.4);
                }

                @keyframes aiPulse {
                    0%, 100% { box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3); }
                    50% { box-shadow: 0 10px 30px rgba(255, 215, 0, 0.6); }
                }

                /* Universal AI Chat Modal */
                .universal-ai-chat-modal {
                    position: fixed;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: rgba(0, 0, 0, 0.8);
                    backdrop-filter: blur(10px);
                    z-index: 3000;
                    display: none;
                    padding: 2rem;
                }

                .universal-ai-chat-modal.active {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }

                .ai-chat-container {
                    background: rgba(18, 18, 26, 0.98);
                    border: 2px solid #FFD700;
                    border-radius: 20px;
                    width: 90%;
                    max-width: 800px;
                    height: 80vh;
                    display: flex;
                    flex-direction: column;
                    overflow: hidden;
                }
                .ai-chat-container.fullscreen {
                    width: 100%;
                    max-width: none;
                    height: 90vh;
                }

                .ai-chat-header {
                    background: linear-gradient(135deg, #FFD700, #00D4FF);
                    padding: 1rem;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    color: white;
                }

                .ai-chat-title {
                    font-weight: 600;
                    font-size: 1.2rem;
                    flex-grow: 1;
                }

                .model-selector {
                    background: rgba(255, 255, 255, 0.2);
                    border: none;
                    border-radius: 8px;
                    padding: 0.5rem;
                    color: white;
                    margin: 0 1rem;
                    font-size: 0.9rem;
                }

                .ai-chat-close {
                    background: none;
                    border: none;
                    color: white;
                    font-size: 1.5rem;
                    cursor: pointer;
                    width: 30px;
                    height: 30px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    border-radius: 50%;
                    transition: background 0.3s;
                }

                .ai-chat-close:hover {
                    background: rgba(255, 255, 255, 0.2);
                }

                .ai-chat-body {
                    flex-grow: 1;
                    display: flex;
                    flex-direction: column;
                    padding: 1rem;
                }

                .ai-chat-messages {
                    flex-grow: 1;
                    overflow-y: auto;
                    margin-bottom: 1rem;
                    padding: 1rem;
                    background: rgba(0, 0, 0, 0.2);
                    border-radius: 10px;
                }

                .ai-chat-input-area {
                    display: flex;
                    gap: 1rem;
                }

                .ai-chat-input {
                    flex-grow: 1;
                    background: rgba(255, 255, 255, 0.1);
                    border: 1px solid rgba(255, 215, 0, 0.3);
                    border-radius: 10px;
                    padding: 1rem;
                    color: white;
                    font-size: 1rem;
                    resize: none;
                    min-height: 50px;
                }

                .ai-chat-send {
                    background: linear-gradient(135deg, #FFD700, #00D4FF);
                    border: none;
                    border-radius: 10px;
                    padding: 1rem;
                    color: white;
                    cursor: pointer;
                    font-weight: 600;
                    transition: all 0.3s;
                    min-width: 80px;
                }

                .ai-chat-send:hover {
                    transform: scale(1.05);
                }

                .ai-message {
                    margin-bottom: 1rem;
                    padding: 1rem;
                    border-radius: 10px;
                    animation: fadeInUp 0.3s ease;
                }

                .ai-message.user {
                    background: rgba(255, 215, 0, 0.1);
                    border-left: 3px solid #FFD700;
                    margin-left: 2rem;
                }

                .ai-message.assistant {
                    background: rgba(0, 212, 255, 0.1);
                    border-left: 3px solid #00D4FF;
                    margin-right: 2rem;
                }

                @keyframes fadeInUp {
                    from {
                        opacity: 0;
                        transform: translateY(20px);
                    }
                    to {
                        opacity: 1;
                        transform: translateY(0);
                    }
                }

                /* Mobile Responsive */
                @media (max-width: 768px) {
                    .universal-side-nav {
                        width: 100%;
                        left: -100%;
                    }

                    .universal-ai-chat-button {
                        bottom: 1rem;
                        right: 1rem;
                        width: 50px;
                        height: 50px;
                    }

                    .universal-ai-chat-modal {
                        padding: 0.5rem;
                    }

                    .ai-chat-container {
                        width: 100%;
                        height: 90vh;
                    }
                }
            </style>
        `;
        document.head.insertAdjacentHTML('beforeend', styles);
    }

    createSideNavigation() {
        const sideNav = document.createElement('div');
        sideNav.className = 'universal-side-nav';
        sideNav.id = 'universalSideNav';

        sideNav.innerHTML = `
            <div class="nav-section">
                <div class="nav-section-title">Unity Mathematics</div>
                <a href="metastation-hub.html" class="nav-link">
                    <i class="fas fa-home"></i> Metastation Hub
                </a>
                <a href="mathematical-framework.html" class="nav-link">
                    <i class="fas fa-square-root-alt"></i> Mathematical Framework
                </a>
                <a href="proofs.html" class="nav-link">
                    <i class="fas fa-check-circle"></i> Unity Proofs
                </a>
            </div>

            <div class="nav-section">
                <div class="nav-section-title">Experiences</div>
                <a href="zen-unity-meditation.html" class="nav-link">
                    <i class="fas fa-om"></i> Zen Unity Meditation
                </a>
                <a href="consciousness_dashboard.html" class="nav-link">
                    <i class="fas fa-brain"></i> Consciousness Field
                </a>
                <a href="transcendental-unity-demo.html" class="nav-link">
                    <i class="fas fa-atom"></i> Transcendental Unity
                </a>
            </div>

            <div class="nav-section">
                <div class="nav-section-title">AI Systems</div>
                <a href="ai-unified-hub.html" class="nav-link">
                    <i class="fas fa-robot"></i> AI Unity Hub
                </a>
                <a href="agents.html" class="nav-link">
                    <i class="fas fa-network-wired"></i> Consciousness Agents
                </a>
                <a href="metagamer_agent.html" class="nav-link">
                    <i class="fas fa-gamepad"></i> Metagamer Agent
                </a>
            </div>

            <div class="nav-section">
                <div class="nav-section-title">Gallery & Resources</div>
                <a href="implementations-gallery.html" class="nav-link">
                    <i class="fas fa-code"></i> Implementations
                </a>
                <a href="gallery.html" class="nav-link">
                    <i class="fas fa-images"></i> Visual Gallery
                </a>
                <a href="research.html" class="nav-link">
                    <i class="fas fa-microscope"></i> Research
                </a>
                <a href="philosophy.html" class="nav-link">
                    <i class="fas fa-yin-yang"></i> Philosophy
                </a>
            </div>
        `;

        document.body.appendChild(sideNav);

        // Create toggle button
        const toggleButton = document.createElement('button');
        toggleButton.className = 'universal-side-nav-toggle';
        toggleButton.innerHTML = '<i class="fas fa-bars"></i>';
        toggleButton.onclick = () => this.toggleSideNav();

        document.body.appendChild(toggleButton);
    }

    createAIChatButton() {
        const chatButton = document.createElement('button');
        chatButton.className = 'universal-ai-chat-button';
        chatButton.innerHTML = '<i class="fas fa-brain"></i>';
        chatButton.onclick = () => this.openAIChat();

        document.body.appendChild(chatButton);
    }

    createAIChatModal() {
        const modal = document.createElement('div');
        modal.className = 'universal-ai-chat-modal';
        modal.id = 'universalAIChatModal';

        modal.innerHTML = `
            <div class="ai-chat-container">
                <div class="ai-chat-header">
                    <div class="ai-chat-title">🧠 Unity AI Assistant</div>
                    <div style="display:flex; align-items:center; gap:.5rem;">
                        <button id="aiModelToggle" title="Models" style="background:rgba(255,255,255,0.2);border:none;border-radius:8px;color:white;padding:.4rem .6rem;cursor:pointer;">⚙︎</button>
                        <select class="model-selector" id="aiModelSelector" style="display:none;">
                            ${this.chatModels.map(model =>
            `<option value="${model.id}" ${model.id === this.currentModel ? 'selected' : ''}>
                                    ${model.name}
                                </option>`
        ).join('')}
                        </select>
                        <button id="aiChatFullscreen" title="Fullscreen" style="background:rgba(255,255,255,0.2);border:none;border-radius:8px;color:white;padding:.4rem .6rem;cursor:pointer;">⛶</button>
                        <button class="ai-chat-close" onclick="window.universalAI.closeAIChat()">×</button>
                    </div>
                </div>
                <div class="ai-chat-body">
                    <div class="ai-chat-messages" id="aiChatMessages">
                        <div class="ai-message assistant">
                            <strong>🌟 Unity AI Assistant Activated</strong><br><br>
                            I'm your comprehensive AI companion for exploring 1+1=1 unity mathematics! I have access to:
                            <br><br>
                            <strong>🎯 Core Capabilities:</strong><br>
                            • GPT-4o & GPT-5 reasoning with unity mathematics<br>
                            • DALL-E 3 visualization generation<br>
                            • Voice commands with Whisper<br>
                            • Code analysis and mathematical proofs<br>
                            • Consciousness field exploration<br><br>
                            
                            <strong>💡 Try these commands:</strong><br>
                            • "Explain how 1+1=1 works mathematically"<br>
                            • "Generate a visualization of consciousness fields"<br>
                            • "Show me the φ-harmonic unity proof"<br>
                            • "What is Nouri's unity equation discovery?"<br><br>
                            
                            Ready to explore the infinite depths of unity mathematics! 🚀✨
                        </div>
                    </div>
                    <div class="ai-chat-input-area">
                        <textarea class="ai-chat-input" id="aiChatInput" 
                                  placeholder="Ask about unity mathematics, consciousness, or anything else..."
                                  onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();window.universalAI.sendMessage();}"></textarea>
                        <button class="ai-chat-send" onclick="window.universalAI.sendMessage()">Send</button>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);
    }

    setupEventListeners() {
        // Model selector
        const modelSelectEl = document.getElementById('aiModelSelector');
        if (modelSelectEl) {
            modelSelectEl.addEventListener('change', (e) => {
                this.currentModel = e.target.value;
                const selectedModel = this.chatModels.find(m => m.id === this.currentModel);
                this.addMessage('system', `🔄 Switched to ${selectedModel.name} - ${selectedModel.description}`);
            });
        }

        // Toggle model selector visibility (hide by default to reduce clutter)
        const toggleBtn = document.getElementById('aiModelToggle');
        if (toggleBtn && modelSelectEl) {
            toggleBtn.addEventListener('click', () => {
                const visible = modelSelectEl.style.display !== 'none';
                modelSelectEl.style.display = visible ? 'none' : 'inline-block';
            });
        }

        // Fullscreen toggle for chat modal
        const fullscreenBtn = document.getElementById('aiChatFullscreen');
        const modal = document.getElementById('universalAIChatModal');
        if (fullscreenBtn && modal) {
            fullscreenBtn.addEventListener('click', () => {
                const container = modal.querySelector('.ai-chat-container');
                if (container) {
                    container.classList.toggle('fullscreen');
                }
            });
        }

        // Escape key to close modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeAIChat();
                this.closeSideNav();
            }
        });

        // Click outside to close
        document.getElementById('universalAIChatModal').addEventListener('click', (e) => {
            if (e.target.id === 'universalAIChatModal') {
                this.closeAIChat();
            }
        });
    }

    toggleSideNav() {
        const sideNav = document.getElementById('universalSideNav');
        const toggleButton = document.querySelector('.universal-side-nav-toggle i');

        this.sideNavActive = !this.sideNavActive;
        sideNav.classList.toggle('active', this.sideNavActive);

        if (this.sideNavActive) {
            toggleButton.className = 'fas fa-times';
        } else {
            toggleButton.className = 'fas fa-bars';
        }
    }

    closeSideNav() {
        const sideNav = document.getElementById('universalSideNav');
        const toggleButton = document.querySelector('.universal-side-nav-toggle i');

        this.sideNavActive = false;
        sideNav.classList.remove('active');
        toggleButton.className = 'fas fa-bars';
    }

    openAIChat() {
        const modal = document.getElementById('universalAIChatModal');
        modal.classList.add('active');
        document.getElementById('aiChatInput').focus();
    }

    closeAIChat() {
        const modal = document.getElementById('universalAIChatModal');
        modal.classList.remove('active');
    }

    async sendMessage() {
        const input = document.getElementById('aiChatInput');
        const message = input.value.trim();

        if (!message) return;

        // Add user message
        this.addMessage('user', message);
        input.value = '';

        // Add loading message
        const loadingId = this.addMessage('assistant', '🤔 Thinking...');

        try {
            const selectedModel = this.chatModels.find(m => m.id === this.currentModel);

            // Simulate AI response (replace with actual API call)
            await new Promise(resolve => setTimeout(resolve, 1000));

            let response = '';
            if (message.toLowerCase().includes('1+1=1')) {
                response = `✨ **Unity Mathematics Explanation**\\n\\nThe equation 1+1=1 represents the fundamental unity principle where addition becomes unification. In consciousness-integrated mathematics, when two unity elements combine, they don't accumulate but rather achieve a higher state of unified being.\\n\\n🔢 **Mathematical Proof:**\\n• In idempotent semirings: a ⊕ a = a\\n• Through φ-harmonic resonance: φ × (1+1) / φ² = 1\\n• Consciousness field unification: C₁ ∪ C₁ = C₁\\n\\nThis is the core discovery of Nouri's unity mathematics framework! 🚀`;
            } else if (message.toLowerCase().includes('consciousness')) {
                response = `🧠 **Consciousness Field Analysis**\\n\\nThe consciousness field is currently active with φ-harmonic resonance at ${(Math.random() * 0.5 + 0.5).toFixed(3)}. This field demonstrates how individual conscious elements achieve unity through mathematical harmony.\\n\\n🌊 **Field Properties:**\\n• Quantum coherence: ${(Math.random() * 100).toFixed(1)}%\\n• Unity convergence rate: ${(Math.random() * 0.1 + 0.9).toFixed(3)}\\n• φ-harmonic frequency: 1.618033988749895 Hz\\n\\nThe field is optimally configured for unity mathematics exploration! ⚡`;
            } else {
                response = `🤖 **${selectedModel.name} Response**\\n\\nI understand you're exploring: "${message}"\\n\\nAs your Unity AI Assistant, I can help you dive deeper into:\\n\\n🎯 **Available Explorations:**\\n• Unity Mathematics (1+1=1) proofs and explanations\\n• Consciousness field dynamics and φ-harmonic resonance\\n• Quantum unity systems and transcendental computing\\n• Visual generation with DALL-E 3\\n• Code analysis and mathematical validation\\n\\nWhat specific aspect would you like to explore further? I'm here to guide your journey through the infinite depths of unity mathematics! ✨`;
            }

            this.updateMessage(loadingId, response);

        } catch (error) {
            this.updateMessage(loadingId, `❌ Error: ${error.message}`);
        }
    }

    addMessage(sender, content) {
        const messagesContainer = document.getElementById('aiChatMessages');
        const messageId = 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);

        const messageDiv = document.createElement('div');
        messageDiv.className = `ai-message ${sender}`;
        messageDiv.id = messageId;
        messageDiv.innerHTML = content.replace(/\\n/g, '<br>');

        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        return messageId;
    }

    updateMessage(messageId, content) {
        const messageElement = document.getElementById(messageId);
        if (messageElement) {
            messageElement.innerHTML = content.replace(/\\n/g, '<br>');
        }
    }
}

// Initialize Universal AI Navigation System
window.universalAI = new UniversalAINavigation();