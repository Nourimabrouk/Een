/**
 * Unified Chatbot System for Een Unity Mathematics
 * Classic chat panel with conversation bubbles and consolidated AI integration
 * Replaces multiple conflicting chatbot implementations
 * Version: 1.0.0
 */

class UnifiedChatbotSystem {
    constructor() {
        this.isVisible = false;
        this.isMinimized = false;
        this.chatHistory = [];
        this.currentModel = 'gpt-4o';
        this.isTyping = false;
        this.autoResponses = true;
        this.groundedMode = true; // Prefer KB/code for factual queries

        // Available AI models (aligned with backend /api/chat/providers)
        this.aiModels = [
            { id: 'gpt-4o', name: 'GPT-4o', provider: 'OpenAI', status: 'stable', color: '#3B82F6' },
            { id: 'gpt-4o-mini', name: 'GPT-4o Mini', provider: 'OpenAI', status: 'stable', color: '#2563eb' },
            { id: 'gpt-4o-mini-high', name: 'GPT-4o Mini High', provider: 'OpenAI', status: 'stable', color: '#1d4ed8' },
            { id: 'gpt-4.1', name: 'GPT-4.1', provider: 'OpenAI', status: 'latest', color: '#06B6D4' },
            { id: 'claude-3-5-sonnet-20241022', name: 'Claude 3.5 Sonnet', provider: 'Anthropic', status: 'latest', color: '#8B5CF6' },
            { id: 'claude-3-opus-20240229', name: 'Claude 3 Opus', provider: 'Anthropic', status: 'stable', color: '#A855F7' },
            { id: 'claude-3-5-haiku-20241022', name: 'Claude 3.5 Haiku', provider: 'Anthropic', status: 'stable', color: '#C084FC' }
        ];

        // Advanced AI capabilities
        this.aiCapabilities = {
            codeSearch: {
                enabled: true,
                endpoint: '/api/code-search/search',
                description: 'RAG-powered semantic search through Unity Mathematics codebase'
            },
            knowledgeBase: {
                enabled: true,
                endpoint: '/api/nouri-knowledge/query',
                description: 'Comprehensive knowledge about Nouri Mabrouk and Unity Mathematics'
            },
            dalle: {
                enabled: true,
                endpoint: '/api/openai/images/generate',
                description: 'DALL-E 3 consciousness field and mathematical visualizations'
            },
            voice: {
                enabled: true,
                endpoint: '/api/openai/tts',
                description: 'Voice synthesis and speech processing'
            },
            consciousnessField: {
                enabled: true,
                description: 'Real-time consciousness field visualization and interaction'
            },
            streaming: {
                enabled: true,
                endpoint: '/api/chat/stream',
                description: 'Real-time streaming responses with typing indicators'
            }
        };

        this.init();
    }

    init() {
        // Remove any existing chatbot systems to prevent conflicts
        this.cleanupExistingChatbots();

        // Create unified chatbot interface
        this.createChatInterface();
        this.createFloatingButton();
        this.ensureGlobalStyles();
        this.applyStyles();
        this.attachEventListeners();
        this.loadChatHistory();

        // Add welcome message
        this.addWelcomeMessage();

        console.log('💬 Unified Chatbot System v3.0 initialized with advanced AI capabilities');
    }

    getCapabilityIcon(capabilityKey) {
        const icons = {
            codeSearch: '🔍',
            knowledgeBase: '📚',
            dalle: '🎨',
            voice: '🔊',
            consciousnessField: '🧠',
            streaming: '⚡'
        };
        return icons[capabilityKey] || '⚙️';
    }

    cleanupExistingChatbots() {
        // Remove conflicting chatbot elements (and common legacy wrappers)
        const existingChats = [
            '#enhanced-een-ai-chat',
            '.ai-chat-button',
            '.enhanced-chat-container',
            '.ai-assistant-panel',
            '.unity-ai-chat',
            '#ai-chat-modal',
            '#floating-ai-chat-button',
            '#persistent-chat-button',
            '.floating-chat-button',
            '#een-chat-widget' // legacy widget container to avoid style collisions
        ];

        existingChats.forEach(selector => {
            const elements = document.querySelectorAll(selector);
            elements.forEach(el => el.remove());
        });

        // Remove conflicting styles
        const conflictingStyles = [
            '#enhanced-een-chat-styles',
            '#ai-chat-styles',
            '#unity-chat-styles',
            '#enhanced-chat-styles',
            '#classic-chat-styles',
            '#een-chat-styles' // legacy widget global styles that used generic class names
        ];

        conflictingStyles.forEach(id => {
            const style = document.getElementById(id);
            if (style) style.remove();
        });

        console.log('🧹 Cleaned up existing chatbot systems to prevent conflicts');
    }

    createFloatingButton() {
        const button = document.createElement('button');
        button.id = 'unified-chat-button';
        button.className = 'unified-chat-floating-btn';
        button.title = 'Open Unity Mathematics AI Assistant';
        button.setAttribute('aria-label', 'Open Unity Mathematics AI Assistant');
        button.setAttribute('type', 'button');
        button.innerHTML = `
            <div class="chat-btn-icon">
                <i class="fas fa-brain"></i>
            </div>
            <div class="chat-btn-pulse"></div>
            <div class="chat-btn-notification" style="display: none;">1</div>
        `;

        document.body.appendChild(button);

        // Ensure fixed position and non-overlap with voice button
        const reposition = () => {
            const voiceBtn = document.querySelector('.voice-button');
            const btnEl = document.getElementById('unified-chat-button');
            if (btnEl) {
                btnEl.style.position = 'fixed';
                btnEl.style.bottom = '20px';
                btnEl.style.right = '20px';
                btnEl.style.zIndex = '10002';
            }
            if (voiceBtn) {
                voiceBtn.style.position = 'fixed';
                voiceBtn.style.bottom = '20px';
                voiceBtn.style.right = '95px';
                voiceBtn.style.zIndex = '10001';
            }
        };
        setTimeout(reposition, 0);
        window.addEventListener('resize', reposition);
    }

    ensureGlobalStyles() {
        // Font Awesome (icons)
        if (!document.querySelector('link[href*="font-awesome"]')) {
            const link = document.createElement('link');
            link.rel = 'stylesheet';
            link.href = 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css';
            document.head.appendChild(link);
        }
    }

    createChatInterface() {
        const chatContainer = document.createElement('div');
        chatContainer.id = 'unified-chat-container';
        chatContainer.className = 'unified-chat-panel';
        chatContainer.setAttribute('role', 'dialog');
        chatContainer.setAttribute('aria-modal', 'true');
        chatContainer.setAttribute('aria-label', 'Unity AI Assistant');
        chatContainer.innerHTML = `
            <!-- Chat Header -->
            <div class="chat-header">
                <div class="chat-header-info">
                    <div class="ai-avatar">
                        <i class="fas fa-brain"></i>
                    </div>
                    <div class="ai-info">
                        <h3 class="ai-name">Unity AI</h3>
                        <div class="ai-subtitle">1+1=1 Assistant • Unity Mathematics</div>
                        <div class="ai-status"><span class="status-dot online"></span><span class="status-text">Online</span></div>
                        <div class="ai-capabilities">
                            <div class="capability-badges">
                                ${Object.entries(this.aiCapabilities).map(([key, capability]) =>
            capability.enabled ? `<span class="capability-badge" title="${capability.description}">${this.getCapabilityIcon(key)}</span>` : ''
        ).join('')}
                            </div>
                        </div>
                    </div>
                </div>
                <div class="chat-header-controls">
                    <div class="model-selector">
                        <select class="model-select" title="Select AI Model">
                            ${this.aiModels.map(model => `
                                <option value="${model.id}" ${model.id === this.currentModel ? 'selected' : ''} 
                                        data-provider="${model.provider}" data-color="${model.color}">
                                    ${model.name}
                                </option>
                            `).join('')}
                        </select>
                    </div>
                    <button class="chat-control-btn grounded-btn" title="Toggle Grounded Mode (prefer facts)">
                        <i class="fas fa-database"></i>
                    </button>
                    <button class="chat-control-btn clear-btn" title="Clear chat">
                        <i class="fas fa-trash"></i>
                    </button>
                    <div class="byok-holder" title="Use your own API key">
                        <button class="chat-control-btn byok-btn" aria-label="API Key"><i class="fas fa-key"></i></button>
                    </div>
                    <button class="chat-control-btn fullscreen-btn" title="Fullscreen">
                        <i class="fas fa-expand"></i>
                    </button>
                    <button class="chat-control-btn minimize-btn" title="Minimize">
                        <i class="fas fa-minus"></i>
                    </button>
                    <button class="chat-control-btn close-btn" title="Close">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            </div>

            <!-- Chat Messages Area -->
            <div class="chat-messages" id="chat-messages" aria-live="polite">
                <div class="chat-messages-content">
                    <!-- Messages will be inserted here -->
                </div>
            </div>

            <!-- Typing Indicator -->
            <div class="typing-indicator" id="typing-indicator" style="display: none;">
                <div class="typing-avatar">
                    <i class="fas fa-brain"></i>
                </div>
                <div class="typing-bubble">
                    <div class="typing-dots">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                    <span class="typing-text">Unity AI is thinking...</span>
                </div>
            </div>

            <!-- Chat Input Area -->
            <div class="chat-input-area">
                <div class="input-container">
                    <button class="input-action-btn attachment-btn" title="Attach File">
                        <i class="fas fa-paperclip"></i>
                    </button>
                    <div class="message-input-wrapper">
                        <textarea 
                            id="chat-message-input" 
                            class="message-input" 
                            placeholder="Ask about Unity Mathematics, consciousness fields, or 1+1=1..."
                            rows="1"
                            maxlength="2000"
                        ></textarea>
                        <div class="input-counter">0/2000</div>
                    </div>
                    <button class="input-action-btn voice-input-btn" id="voice-input-btn" title="Voice input">
                        <i class="fas fa-microphone"></i>
                    </button>
                    <button class="input-action-btn send-btn" id="send-message-btn" title="Send Message" disabled>
                        <i class="fas fa-paper-plane"></i>
                    </button>
                </div>
                <div class="quick-actions">
                    <button class="quick-action-btn primary" data-message="Prove that 1+1=1 using idempotent operations and show a minimal example">
                        ✅ Prove 1+1=1
                    </button>
                    <button class="quick-action-btn" data-message="Explain how 1+1=1 in Unity Mathematics">
                        🧮 Unity Mathematics
                    </button>
                    <button class="quick-action-btn" data-message="Show me consciousness field equations">
                        🧠 Consciousness Fields
                    </button>
                    <button class="quick-action-btn" data-message="What is φ-harmonic resonance?">
                        🌟 φ-Harmonic
                    </button>
                    <button class="quick-action-btn" data-message="Show quantum unity states">
                        ⚛️ Quantum Unity
                    </button>
                </div>
            </div>
        `;

        document.body.appendChild(chatContainer);
    }

    applyStyles() {
        const styleId = 'unified-chatbot-styles';
        if (document.getElementById(styleId)) return;

        const style = document.createElement('style');
        style.id = styleId;
        style.textContent = `
            /* Unified Chatbot System Styles */
            .unified-chat-floating-btn {
                position: fixed;
                bottom: calc(20px + env(safe-area-inset-bottom, 0px));
                right: calc(20px + env(safe-area-inset-right, 0px));
                width: 65px;
                height: 65px;
                background: linear-gradient(135deg, #FFD700, #D4AF37);
                border: none;
                border-radius: 50%;
                cursor: pointer;
                z-index: 10002; /* sits above voice button */
                box-shadow: 0 8px 32px rgba(255, 215, 0, 0.3);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                display: flex;
                align-items: center;
                justify-content: center;
                position: relative;
                overflow: hidden;
                pointer-events: auto;
            }

            .unified-chat-floating-btn:hover {
                transform: translateY(-3px) scale(1.05);
                box-shadow: 0 12px 40px rgba(255, 215, 0, 0.4);
            }

            .chat-btn-icon {
                color: #000;
                font-size: 1.5rem;
                z-index: 2;
                position: relative;
            }

            .chat-btn-pulse {
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                border-radius: 50%;
                background: rgba(255, 215, 0, 0.6);
                animation: chatPulse 2s ease-in-out infinite;
            }

            @keyframes chatPulse {
                0% { transform: scale(1); opacity: 0.6; }
                50% { transform: scale(1.2); opacity: 0.3; }
                100% { transform: scale(1); opacity: 0.6; }
            }

            .chat-btn-notification {
                position: absolute;
                top: -5px;
                right: -5px;
                width: 24px;
                height: 24px;
                background: #FF4444;
                color: white;
                border-radius: 50%;
                font-size: 0.75rem;
                font-weight: 600;
                display: flex;
                align-items: center;
                justify-content: center;
                border: 2px solid white;
            }

            /* Chat Panel */
            .unified-chat-panel {
                position: fixed;
                bottom: calc(25px + env(safe-area-inset-bottom, 0px));
                right: calc(25px + env(safe-area-inset-right, 0px));
                width: clamp(360px, 28vw, 420px);
                height: clamp(520px, 78vh, 720px);
                max-height: calc(100vh - 40px);
                /* Ensure the header never clips off-screen on small viewports */
                background: rgba(15, 15, 20, 0.98);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 215, 0, 0.2);
                border-radius: 20px;
                box-shadow: 0 25px 60px rgba(0, 0, 0, 0.4);
                z-index: 10003;
                display: none;
                flex-direction: column;
                overflow: hidden;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                opacity: 0;
                transform: translateY(30px) scale(0.95);
                box-sizing: border-box;
            }

            /* Maximized mode (respect the site navigation height) */
            .unified-chat-panel.fullscreen {
                left: 0 !important;
                right: 0 !important;
                top: var(--nav-height, 70px) !important;
                /* fall back to 70px if nav variable is unavailable */
                bottom: 0 !important;
                width: 100vw !important;
                height: calc(100vh - var(--nav-height, 70px)) !important;
                border-radius: 0 !important;
            }

            .unified-chat-panel.visible {
                display: flex;
                opacity: 1;
                transform: translateY(0) scale(1);
            }

            .unified-chat-panel.minimized {
                height: 80px;
            }

            .unified-chat-panel.minimized .chat-messages,
            .unified-chat-panel.minimized .chat-input-area,
            .unified-chat-panel.minimized .typing-indicator {
                display: none;
            }

            /* Chat Header */
            .chat-header {
                display: grid;
                grid-template-columns: 1fr auto;
                grid-template-areas: "info controls";
                align-items: center;
                column-gap: 0.75rem;
                row-gap: .25rem;
                padding: .85rem 1rem;
                background: rgba(255, 255, 255, 0.02);
                border-bottom: 1px solid var(--uc-border);
                flex-shrink: 0;
            }

            .chat-header-info {
                grid-area: info;
                display: flex;
                align-items: center;
                gap: 1rem;
                flex: 1;
                min-width: 0;
            }

            .ai-avatar {
                width: 45px;
                height: 45px;
                background: linear-gradient(135deg, #FFD700, #D4AF37);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #000;
                font-size: 1.2rem;
                flex-shrink: 0;
                position: relative;
                overflow: hidden;
            }

            .ai-avatar::before {
                content: '';
                position: absolute;
                top: -2px;
                left: -2px;
                right: -2px;
                bottom: -2px;
                background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.3), transparent);
                border-radius: 50%;
                animation: avatarGlow 3s ease-in-out infinite;
            }

            @keyframes avatarGlow {
                0%, 100% { opacity: 0; transform: rotate(0deg); }
                50% { opacity: 1; transform: rotate(180deg); }
            }

            .ai-info {
                flex: 1;
                min-width: 0;
            }

            .ai-name {
                color: #FFD700;
                font-size: 1rem;
                font-weight: 700;
                margin: 0;
                line-height: 1.2;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }

            .ai-status {
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                color: rgba(255, 255, 255, 0.7);
                font-size: 0.8rem;
                margin-top: 0.25rem;
            }

            .ai-subtitle {
                color: #cbd5e1;
                font-size: 0.8rem;
                margin: .15rem 0 0 0;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                max-width: 100%;
            }

            .status-dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #10B981;
                animation: statusPulse 2s ease-in-out infinite;
            }

            .status-dot.online {
                background: #10B981;
            }

            .status-dot.offline {
                background: #EF4444;
            }

            @keyframes statusPulse {
                0%, 100% { opacity: 1; transform: scale(1); }
                50% { opacity: 0.6; transform: scale(1.2); }
            }

            .chat-header-controls { grid-area: controls; display: inline-flex; align-items: center; gap: 0.5rem; }
            .chat-header .model-selector { order: 1; }
            .chat-header .grounded-btn { order: 2; }
            .chat-header .clear-btn { order: 3; }
            .chat-header .byok-holder { order: 4; }
            .chat-header .fullscreen-btn { order: 5; }
            .chat-header .minimize-btn { order: 6; }
            .chat-header .close-btn { order: 7; }

            .model-selector {
                position: relative;
                max-width: 220px;
                flex: 0 1 220px;
            }

            .model-select {
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 215, 0, 0.2);
                border-radius: 8px;
                color: rgba(255, 255, 255, 0.9);
                padding: 0.5rem 0.75rem;
                font-size: 0.8rem;
                cursor: pointer;
                outline: none;
                transition: all 0.3s ease;
                min-width: 160px;
                max-width: 220px;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }

            .model-select:hover,
            .model-select:focus {
                border-color: #FFD700;
                background: rgba(255, 215, 0, 0.05);
            }

            .model-select option { color: #e5e7eb; background: #0f1622; padding: 0.5rem; }
            .model-select option[data-provider="OpenAI"] { color: #86efac; }
            .model-select option[data-provider="Anthropic"] { color: #c4b5fd; }
            .model-select option[data-provider="Google"] { color: #f9a8d4; }
            .model-select option:checked { background: #111827; color: #ffffff; }

            .chat-control-btn {
                width: 32px;
                height: 32px;
                background: transparent;
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 8px;
                color: rgba(255, 255, 255, 0.7);
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.3s ease;
                font-size: 0.85rem;
            }

            /* Accessible focus treatment */
            .chat-control-btn:focus-visible,
            .model-select:focus-visible,
            .message-input:focus-visible,
            .quick-action-btn:focus-visible {
                outline: none;
                box-shadow: 0 0 0 3px rgba(59,130,246,0.35);
                border-color: #2a3444;
            }

            .chat-control-btn:hover {
                background: rgba(255, 215, 0, 0.1);
                border-color: #FFD700;
                color: #FFD700;
            }
            .chat-control-btn.grounded-btn.active { color: #22c55e; border-color: #22c55e; background: rgba(34,197,94,0.1); }

            /* Chat Messages */
            .chat-messages {
                flex: 1;
                overflow-y: auto;
                padding: 1rem;
                display: flex;
                flex-direction: column;
                min-height: 0;
            }

            .chat-messages::-webkit-scrollbar {
                width: 6px;
            }

            .chat-messages::-webkit-scrollbar-track {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 3px;
            }

            .chat-messages::-webkit-scrollbar-thumb {
                background: rgba(255, 215, 0, 0.3);
                border-radius: 3px;
            }

            .chat-messages::-webkit-scrollbar-thumb:hover {
                background: rgba(255, 215, 0, 0.5);
            }

            .chat-messages-content {
                display: flex;
                flex-direction: column;
                gap: 1rem;
            }

            .message-bubble {
                max-width: 85%;
                position: relative;
                animation: messageSlideIn 0.3s ease-out;
            }

            @keyframes messageSlideIn {
                0% {
                    opacity: 0;
                    transform: translateY(10px);
                }
                100% {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            .message-bubble.user {
                align-self: flex-end;
            }

            .message-bubble.assistant {
                align-self: flex-start;
            }

            .message-content {
                padding: 0.875rem 1.25rem;
                border-radius: 18px;
                color: #ffffff;
                font-size: 0.9rem;
                line-height: 1.5;
                word-wrap: break-word;
                position: relative;
                backdrop-filter: blur(10px);
                -webkit-font-smoothing: antialiased;
                text-rendering: optimizeLegibility;
            }

            .message-bubble.user .message-content {
                background: linear-gradient(135deg, rgba(255, 215, 0, 0.8), rgba(212, 175, 55, 0.8));
                color: #000;
                border-bottom-right-radius: 6px;
                font-weight: 500;
            }

            .message-bubble.assistant .message-content {
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-bottom-left-radius: 6px;
            }

            .message-time {
                font-size: 0.75rem;
                color: rgba(255, 255, 255, 0.5);
                margin-top: 0.5rem;
                text-align: right;
            }

            .message-bubble.assistant .message-time {
                text-align: left;
            }

            /* Message Formatting */
            .message-content h1,
            .message-content h2,
            .message-content h3 {
                color: #FFD700;
                margin: 0.5rem 0;
                font-weight: 600;
            }

            /* Extended Markdown & rich content aesthetics */
            .message-content h4,
            .message-content h5,
            .message-content h6 { color: #FFD700; margin: 0.4rem 0; font-weight: 600; }
            .message-content a { color: #7dd3fc; text-decoration: underline; text-underline-offset: 2px; }
            .message-content a:hover { color: #bae6fd; }
            .message-content blockquote {
                margin: 0.6rem 0;
                padding: 0.6rem 0.9rem;
                border-left: 3px solid #FFD700;
                background: rgba(255,255,255,0.05);
                border-radius: 8px;
            }
            .message-content table { border-collapse: collapse; width: 100%; margin: 0.5rem 0; }
            .message-content th, .message-content td { border: 1px solid rgba(255,255,255,0.15); padding: 0.45rem 0.6rem; }
            .message-content th { background: rgba(255,255,255,0.08); }

            .message-content p {
                margin: 0.5rem 0;
            }

            .message-content code {
                background: rgba(0, 0, 0, 0.3);
                padding: 0.2rem 0.4rem;
                border-radius: 4px;
                font-family: 'JetBrains Mono', monospace;
                font-size: 0.85rem;
                color: #FFD700;
            }

            .message-content pre {
                background: rgba(0, 0, 0, 0.4);
                padding: 1rem;
                border-radius: 8px;
                overflow-x: auto;
                margin: 0.75rem 0;
                border: 1px solid rgba(255, 255, 255, 0.1);
                position: relative;
            }

            .message-content pre code {
                background: none;
                padding: 0;
                color: #ffffff;
            }

            .message-content pre .copy-code-btn {
                position: absolute;
                top: 8px;
                right: 8px;
                background: rgba(255, 215, 0, 0.15);
                border: 1px solid rgba(255, 215, 0, 0.35);
                color: #FFD700;
                border-radius: 6px;
                font-size: 0.75rem;
                padding: 0.15rem 0.4rem;
                cursor: pointer;
            }
            .message-content pre .copy-code-btn:hover { background: rgba(255, 215, 0, 0.25); }

            .message-content ul,
            .message-content ol {
                margin: 0.5rem 0;
                padding-left: 1.5rem;
            }

            .message-content li {
                margin: 0.25rem 0;
            }

            /* Typing Indicator */
            .typing-indicator {
                display: flex;
                align-items: flex-end;
                gap: 0.75rem;
                padding: 0.75rem 1rem;
                animation: messageSlideIn 0.3s ease-out;
            }

            .typing-avatar {
                width: 32px;
                height: 32px;
                background: linear-gradient(135deg, #FFD700, #D4AF37);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #000;
                font-size: 0.9rem;
                flex-shrink: 0;
            }

            .typing-bubble {
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 18px;
                border-bottom-left-radius: 6px;
                padding: 0.875rem 1.25rem;
                display: flex;
                align-items: center;
                gap: 0.75rem;
                backdrop-filter: blur(10px);
            }

            .typing-dots {
                display: flex;
                gap: 0.25rem;
            }

            .typing-dots span {
                width: 6px;
                height: 6px;
                background: #FFD700;
                border-radius: 50%;
                animation: typingBounce 1.4s ease-in-out infinite;
            }

            .typing-dots span:nth-child(1) { animation-delay: -0.32s; }
            .typing-dots span:nth-child(2) { animation-delay: -0.16s; }
            .typing-dots span:nth-child(3) { animation-delay: 0s; }

            @keyframes typingBounce {
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
                color: rgba(255, 255, 255, 0.7);
                font-size: 0.85rem;
            }

            /* Chat Input */
            .chat-input-area {
                flex-shrink: 0;
                padding: .9rem 1.1rem 1.1rem;
                background: rgba(255, 255, 255, 0.02);
                border-top: 1px solid rgba(255, 255, 255, 0.1);
            }

            .input-container {
                display: flex;
                align-items: flex-end;
                gap: 0.75rem;
                margin-bottom: 0.75rem;
            }

            .input-action-btn {
                width: 40px;
                height: 40px;
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 12px;
                color: rgba(255, 255, 255, 0.7);
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.3s ease;
                font-size: 0.9rem;
                flex-shrink: 0;
            }

            .input-action-btn:hover {
                background: rgba(255, 215, 0, 0.1);
                border-color: #FFD700;
                color: #FFD700;
            }

            .send-btn {
                background: linear-gradient(135deg, #FFD700, #D4AF37);
                border-color: #FFD700;
                color: #000;
            }

            .send-btn:disabled {
                background: rgba(255, 255, 255, 0.05);
                border-color: rgba(255, 255, 255, 0.1);
                color: rgba(255, 255, 255, 0.3);
                cursor: not-allowed;
            }

            .send-btn:not(:disabled):hover {
                background: linear-gradient(135deg, #D4AF37, #FFD700);
                transform: translateY(-1px);
                box-shadow: 0 4px 15px rgba(255, 215, 0, 0.3);
            }

            .message-input-wrapper {
                flex: 1;
                position: relative;
            }

            .message-input {
                width: 100%;
                min-height: 52px;
                max-height: 160px;
                padding: 0.9rem 1.1rem;
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 12px;
                color: #ffffff;
                font-family: inherit;
                font-size: 1rem;
                line-height: 1.5;
                resize: none;
                outline: none;
                transition: all 0.3s ease;
                overflow-y: auto;
            }

            .message-input::placeholder {
                color: rgba(255, 255, 255, 0.5);
            }

            .message-input:focus {
                border-color: #FFD700;
                background: rgba(255, 215, 0, 0.05);
                box-shadow: 0 0 0 3px rgba(255, 215, 0, 0.1);
            }

            .input-counter {
                position: absolute;
                bottom: 0.25rem;
                right: 0.75rem;
                font-size: 0.7rem;
                color: rgba(255, 255, 255, 0.4);
                pointer-events: none;
            }

            .quick-actions {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                align-items: center;
                max-height: 72px;
                overflow-y: auto;
                padding-right: .25rem;
            }

            .quick-action-btn {
                background: rgba(255, 215, 0, 0.05);
                border: 1px solid rgba(255, 215, 0, 0.2);
                border-radius: 20px;
                color: rgba(255, 255, 255, 0.8);
                padding: 0.5rem 0.875rem;
                font-size: 0.8rem;
                cursor: pointer;
                transition: all 0.3s ease;
                white-space: nowrap;
            }

            .quick-action-btn:hover {
                background: rgba(255, 215, 0, 0.1);
                border-color: #FFD700;
                color: #FFD700;
                transform: translateY(-1px);
            }

            .quick-action-btn.primary {
                background: linear-gradient(135deg, #22c55e, #16a34a);
                border-color: #16a34a;
                color: #06130a;
                font-weight: 700;
            }
            .quick-action-btn.primary:hover {
                filter: brightness(1.05);
            }

            /* Mobile Responsive */
            @media (max-width: 768px) {
                .unified-chat-floating-btn {
                    bottom: 15px;
                    right: 15px;
                    width: 60px;
                    height: 60px;
                }

                .unified-chat-panel {
                    bottom: 12px;
                    left: 12px;
                    right: 12px;
                    width: auto;
                    height: min(80vh, 640px);
                }

                .chat-header {
                    padding: .75rem .9rem;
                    grid-template-columns: 1fr;
                    grid-template-areas:
                        "info"
                        "controls";
                }

                .ai-name { font-size: 1.05rem; }
                .ai-subtitle { font-size: .8rem; }

                .chat-header-controls { display: flex; flex-wrap: wrap; gap: .5rem; justify-content: space-between; }
                .chat-header .model-selector { order: 0; flex: 1 1 100%; }
                .model-select { min-width: 0; width: 100%; font-size: 0.8rem; }
                .ai-capabilities { display: none; }

                .chat-messages {
                    padding: 0.75rem;
                }

                .message-bubble {
                    max-width: 90%;
                }

                .chat-input-area {
                    padding: 0.6rem .8rem 1rem;
                }

                .quick-actions {
                    gap: 0.4rem;
                    max-height: 56px;
                    overflow-y: auto;
                }

                .quick-action-btn {
                    font-size: 0.75rem;
                    padding: 0.4rem 0.7rem;
                }
            }

            /* Ensure proper z-index layering */
            .unified-chat-floating-btn { z-index: 10002; }
            .unified-chat-panel { z-index: 10003; }

            /* Audio system compatibility */
            .persistent-audio-panel {
                z-index: 9997;
            }

            /* Navigation compatibility */
            .meta-optimal-nav {
                z-index: 10000;
            }

            /* AI Capabilities */
            .ai-capabilities { margin-top: 0.15rem; }

            .capability-badges { display: inline-flex; gap: 0.3rem; flex-wrap: nowrap; max-width: 240px; overflow: hidden; opacity: .8; }

            .capability-badge {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 18px;
                height: 18px;
                background: rgba(255, 215, 0, 0.1);
                border: 1px solid rgba(255, 215, 0, 0.3);
                border-radius: 4px;
                font-size: 0.65rem;
                cursor: help;
                transition: all 0.3s ease;
            }

            .capability-badge:hover {
                background: rgba(255, 215, 0, 0.2);
                border-color: #FFD700;
                transform: scale(1.1);
                opacity: 1;
            }

            /* Enhanced model selector */
            .model-select option[data-provider="OpenAI"] {
                background: rgba(16, 185, 129, 0.1);
            }

            .model-select option[data-provider="Anthropic"] {
                background: rgba(139, 92, 246, 0.1);
            }

            .model-select option[data-provider="Google"] {
                background: rgba(236, 72, 153, 0.1);
            }

            /* --- Modern Minimal Theme (OpenAI/Anthropic inspired) --- */
            /* Palette */
            :root {
                --uc-bg: #0b0f14;           /* panel background */
                --uc-surface: #0f1622;      /* cards/inputs */
                --uc-border: #202634;       /* subtle borders */
                --uc-text: #e5e7eb;         /* primary text */
                --uc-text-muted: #94a3b8;   /* secondary text */
                --uc-accent: #22c55e;       /* action accent (green) */
                --uc-accent-2: #3b82f6;     /* secondary accent (blue) */
                --uc-shadow: 0 24px 64px rgba(0, 0, 0, 0.6);
            }

            /* Floating button */
            .unified-chat-floating-btn {
                background: linear-gradient(180deg, #111827, var(--uc-bg)) !important;
                border: 1px solid var(--uc-border) !important;
                box-shadow: var(--uc-shadow) !important;
            }
            .unified-chat-floating-btn .chat-btn-icon {
                color: var(--uc-text) !important;
            }
            .unified-chat-floating-btn .chat-btn-pulse { display: none !important; }

            /* Panel */
            .unified-chat-panel {
                background: var(--uc-bg) !important;
                border: 1px solid var(--uc-border) !important;
                box-shadow: var(--uc-shadow) !important;
            }

            /* Header */
            .chat-header {
                background: rgba(255,255,255,0.02) !important;
                border-bottom: 1px solid var(--uc-border) !important;
            }
            .ai-avatar {
                background: radial-gradient(120px 120px at 30% 30%, #1f2937 0%, #0f1622 70%) !important;
                color: #cbd5e1 !important;
            }
            .ai-name { color: #f9fafb !important; }
            .ai-status { color: var(--uc-text-muted) !important; }

            /* Controls */
            .model-select {
                background: var(--uc-surface) !important;
                border: 1px solid var(--uc-border) !important;
                color: #e5e7eb !important;
                font-weight: 600 !important;
                letter-spacing: 0.01em !important;
                text-rendering: optimizeLegibility;
            }
            .chat-control-btn {
                background: var(--uc-surface) !important;
                border: 1px solid var(--uc-border) !important;
                color: var(--uc-text-muted) !important;
            }
            .chat-control-btn:hover {
                background: #142032 !important;
                border-color: #2a3444 !important;
                color: #e2e8f0 !important;
            }

            /* Messages */
            .chat-messages { background: transparent !important; }
            .message-bubble .message-content {
                color: #e6e6e6 !important;
            }
            .message-bubble.assistant .message-content {
                background: var(--uc-surface) !important;
                border: 1px solid var(--uc-border) !important;
            }
            .message-bubble.user .message-content {
                background: linear-gradient(180deg, rgba(34,197,94,0.22), rgba(34,197,94,0.18)) !important;
                color: #dffbe6 !important;
                border: 1px solid rgba(34,197,94,0.35) !important;
            }
            .message-time { color: #6b7280 !important; }

            /* Rich content */
            .message-content h1,
            .message-content h2,
            .message-content h3 { color: #ffffff !important; }
            .message-content a { color: #93c5fd !important; }
            .message-content code {
                background: #0b1220 !important;
                color: #eab308 !important;
                border: 1px solid #1f2937 !important;
            }
            .message-content pre {
                background: #0b1220 !important;
                border: 1px solid #1f2937 !important;
            }
            .message-content pre .copy-code-btn {
                background: rgba(148,163,184,0.15) !important;
                border-color: rgba(148,163,184,0.35) !important;
                color: #cbd5e1 !important;
            }

            /* Typing indicator */
            .typing-bubble {
                background: var(--uc-surface) !important;
                border: 1px solid var(--uc-border) !important;
            }
            .typing-dots span { background: var(--uc-accent) !important; }

            /* Input area */
            .chat-input-area {
                background: rgba(255,255,255,0.01) !important;
                border-top: 1px solid var(--uc-border) !important;
            }
            .message-input {
                background: var(--uc-surface) !important;
                border: 1px solid var(--uc-border) !important;
                color: var(--uc-text) !important;
            }
            .message-input:focus {
                border-color: #2a7f4f !important;
                box-shadow: 0 0 0 3px rgba(34,197,94,0.15) !important;
            }
            .message-input::placeholder { color: #64748b !important; }
            .input-counter { color: #64748b !important; }

            .send-btn {
                background: linear-gradient(180deg, var(--uc-accent), #16a34a) !important;
                border-color: #16a34a !important;
                color: #06130a !important;
            }
            .send-btn:disabled {
                background: #0f1622 !important;
                border-color: var(--uc-border) !important;
                color: #4b5563 !important;
            }

            .quick-action-btn {
                background: #0f1622 !important;
                border: 1px solid var(--uc-border) !important;
                color: #cbd5e1 !important;
            }
            .quick-action-btn:hover {
                background: #142032 !important;
                border-color: #2a3444 !important;
                color: #ffffff !important;
            }
        `;

        document.head.appendChild(style);
    }

    attachEventListeners() {
        // Floating button click
        const floatingBtn = document.getElementById('unified-chat-button');
        floatingBtn?.addEventListener('click', () => this.toggleChat());

        // Chat panel controls
        const closeBtn = document.querySelector('.close-btn');
        const minimizeBtn = document.querySelector('.minimize-btn');
        const fullscreenBtn = document.querySelector('.fullscreen-btn');
        const byokBtn = document.querySelector('.byok-btn');

        closeBtn?.addEventListener('click', () => this.closeChat());
        minimizeBtn?.addEventListener('click', () => this.toggleMinimize());
        fullscreenBtn?.addEventListener('click', () => this.toggleFullscreen());
        byokBtn?.addEventListener('click', () => this.promptBYOK());

        // Model selection
        const modelSelect = document.querySelector('.model-select');
        modelSelect?.addEventListener('change', (e) => this.changeModel(e.target.value));

        // Clear chat
        const clearBtn = document.querySelector('.clear-btn');
        clearBtn?.addEventListener('click', () => {
            const ok = confirm('Clear the current conversation?');
            if (ok) this.clearHistory();
        });

        // Grounded mode toggle
        const groundedBtn = document.querySelector('.grounded-btn');
        groundedBtn?.addEventListener('click', () => {
            this.groundedMode = !this.groundedMode;
            groundedBtn.classList.toggle('active', this.groundedMode);
            this.sendSystemMessage(`Grounded mode ${this.groundedMode ? 'enabled' : 'disabled'}`);
        });

        // Message input
        const messageInput = document.getElementById('chat-message-input');
        const sendBtn = document.getElementById('send-message-btn');
        const voiceBtn = document.getElementById('voice-input-btn');

        messageInput?.addEventListener('input', () => this.updateInputState());
        messageInput?.addEventListener('keydown', (e) => this.handleKeyDown(e));
        sendBtn?.addEventListener('click', () => this.sendMessage());
        voiceBtn?.addEventListener('click', () => this.handleVoiceInput());

        // Quick actions
        const quickActions = document.querySelectorAll('.quick-action-btn');
        quickActions.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const message = e.currentTarget?.dataset?.message;
                if (message) this.sendQuickMessage(message);
            });
        });

        // Delegate copy actions for code blocks
        const messagesWrap = document.querySelector('.chat-messages-content');
        messagesWrap?.addEventListener('click', (e) => {
            const target = e.target;
            if (target && target.classList && target.classList.contains('copy-code-btn')) {
                const pre = target.closest('pre');
                const code = pre?.querySelector('code');
                if (code) {
                    navigator.clipboard.writeText(code.innerText).then(() => {
                        target.textContent = 'Copied';
                        setTimeout(() => (target.textContent = 'Copy'), 1200);
                    }).catch(() => { });
                }
            }
        });

        // Global events (both legacy and unified)
        window.addEventListener('meta-optimal-nav:chat', () => this.toggleChat());
        window.addEventListener('unified-nav:chat', () => this.toggleChat());

        // Handle escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.isVisible) {
                this.closeChat();
            }
        });

        // Auto-resize textarea
        messageInput?.addEventListener('input', this.autoResizeTextarea);
    }

    async handleVoiceInput() {
        try {
            // 1) If Web Speech synthesis exists, read the last assistant message (or input text)
            const synth = window.speechSynthesis;
            if (synth && typeof SpeechSynthesisUtterance !== 'undefined') {
                const lastAssistant = Array.from(document.querySelectorAll('.message-bubble.assistant .message-content'))
                    .slice(-1)[0];
                const input = document.getElementById('chat-message-input');
                const text = (lastAssistant?.innerText || input?.value || 'Unity Mathematics assistant ready. 1 plus 1 equals 1 in the unity field.');
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.rate = 1.0;
                utterance.pitch = 1.0;
                synth.cancel();
                synth.speak(utterance);
                this.sendSystemMessage('🔊 Speaking the latest response…');
                return;
            }

            // 2) Otherwise, click any global site voice button if present
            const globalVoice = document.querySelector('.voice-button');
            if (globalVoice) { globalVoice.click(); return; }

            // 3) Fallback
            this.sendSystemMessage('🎤 Voice not supported in this browser.');
        } catch (_) {
            this.sendSystemMessage('🎤 Voice is unavailable in this browser.');
        }
    }

    toggleChat() {
        if (this.isVisible) {
            this.closeChat();
        } else {
            this.openChat();
        }
    }

    openChat() {
        const panel = document.getElementById('unified-chat-container');
        if (!panel) return;

        panel.classList.add('visible');
        this.isVisible = true;

        // Focus input
        setTimeout(() => {
            const input = document.getElementById('chat-message-input');
            input?.focus();
        }, 300);

        // Hide notification badge
        const notification = document.querySelector('.chat-btn-notification');
        if (notification) notification.style.display = 'none';

        console.log('💬 Unity AI Chat opened');
    }

    closeChat() {
        const panel = document.getElementById('unified-chat-container');
        if (!panel) return;

        panel.classList.remove('visible');
        this.isVisible = false;
        this.isMinimized = false;

        console.log('💬 Unity AI Chat closed');
    }

    toggleMinimize() {
        const panel = document.getElementById('unified-chat-container');
        if (!panel) return;

        this.isMinimized = !this.isMinimized;
        panel.classList.toggle('minimized', this.isMinimized);

        console.log(`💬 Unity AI Chat ${this.isMinimized ? 'minimized' : 'expanded'}`);
    }

    toggleFullscreen() {
        const panel = document.getElementById('unified-chat-container');
        if (!panel) return;
        const isFs = panel.classList.toggle('fullscreen');
        const btn = document.querySelector('.fullscreen-btn i');
        if (btn) btn.className = isFs ? 'fas fa-compress' : 'fas fa-expand';
    }

    changeModel(modelId) {
        this.currentModel = modelId;
        const model = this.aiModels.find(m => m.id === modelId);

        if (model) {
            // Add model switch message
            this.addMessage('system', `Switched to ${model.name} (${model.provider})`);
            console.log(`🤖 AI Model changed to: ${model.name}`);
        }
    }

    updateInputState() {
        const input = document.getElementById('chat-message-input');
        const sendBtn = document.getElementById('send-message-btn');
        const counter = document.querySelector('.input-counter');

        if (!input || !sendBtn || !counter) return;

        const length = input.value.length;
        const maxLength = parseInt(input.getAttribute('maxlength')) || 2000;

        // Update counter
        counter.textContent = `${length}/${maxLength}`;
        counter.style.color = length > maxLength * 0.9 ? '#FF6B6B' : 'rgba(255, 255, 255, 0.4)';

        // Update send button state
        sendBtn.disabled = length === 0 || this.isTyping;
    }

    handleKeyDown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            this.sendMessage();
        }
    }

    autoResizeTextarea(e) {
        const textarea = e.target;
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }

    async sendMessage() {
        const input = document.getElementById('chat-message-input');
        const message = input?.value.trim();

        if (!message || this.isTyping) return;

        // Add user message
        this.addMessage('user', message);
        input.value = '';
        this.updateInputState();

        // Show typing indicator
        this.showTypingIndicator();

        try {
            // Get AI response (mock for now)
            const response = await this.getAIResponse(message);
            this.addMessage('assistant', response);
        } catch (error) {
            console.error('Chat error:', error);
            this.addMessage('assistant', 'I apologize, but I encountered an error. Please try again.');
        } finally {
            this.hideTypingIndicator();
        }
    }

    sendQuickMessage(message) {
        const input = document.getElementById('chat-message-input');
        if (input) {
            input.value = message;
            this.sendMessage();
        }
    }

    async getAIResponse(message) {
        // Commands
        if (message.startsWith('/')) {
            return await this.processAICommand(message);
        }

        const intent = this.detectIntent(message);

        // Grounded-first routing
        if (this.groundedMode) {
            if (intent.type === 'knowledge' && this.aiCapabilities.knowledgeBase.enabled) {
                try { const kb = await this.queryKnowledgeBase(intent.query || message); if (kb?.trim()) return kb; } catch (_) { }
            }
            if (intent.type === 'code' && this.aiCapabilities.codeSearch.enabled) {
                try { return await this.searchCodebase(intent.query || message); } catch (_) { }
            }
            if (intent.type === 'image' && this.aiCapabilities.dalle.enabled) {
                try { return await this.generateVisualization(intent.query || message); } catch (_) { }
            }
            if (intent.type === 'voice' && this.aiCapabilities.voice.enabled) {
                try { return await this.synthesizeVoice(intent.query || message); } catch (_) { }
            }
            if (intent.type === 'unity') { return this.demonstrateUnityOperation(1, 1); }
            if (intent.type === 'phi') { return this.getPhiHarmonicCalculations(); }
            if (intent.type === 'consciousness') { return this.getConsciousnessStatus(); }
        }

        // Live model with intent metadata
        try {
            if (this.aiCapabilities.streaming.enabled) {
                return await this.getStreamingResponse(message, intent);
            }
            return await this.getStandardResponse(message, intent);
        } catch (error) {
            console.warn('AI API call failed, attempting grounded fallbacks:', error);
            try {
                if (this.aiCapabilities.knowledgeBase.enabled) {
                    const kb = await this.queryKnowledgeBase(message);
                    if (kb?.trim()) return kb;
                }
            } catch (_) { }
            return await this.getEnhancedFallbackResponse(message);
        }
    }

    detectIntent(message) {
        const text = (message || '').toLowerCase();
        const isWhoWhat = /^(who\s+is|what\s+is|when\s+did|where\s+is|which\s+|tell\s+me\s+about)\b/.test(text);
        const mentionsNouri = /(nouri\s+mabrouk|nouri)/i.test(message);
        const codeHints = /(code|function|class|file|where\s+is|implementation|api|endpoint|stacktrace|error\s+trace|search\s+code)/i.test(message);
        const imageHints = /(image|generate\s+image|draw|visualiz(e|ation)|picture|art|render|dall-e|dalle)/i.test(message);
        const voiceHints = /(voice|speak|read\s+this|tts|text\s*to\s*speech|audio)/i.test(message);
        const unityHints = /(prove\s+1\+1=1|unity\s+operation|idempotent|1\+1\s*=\s*1)/i.test(message);
        const phiHints = /\bphi\b|φ|golden\s+ratio/i.test(message);
        const consciousnessHints = /(consciousness|field\s+status|awareness)/i.test(message);

        if (mentionsNouri || isWhoWhat) return { type: 'knowledge', query: message, reason: 'Who/What or Nouri query' };
        if (codeHints) return { type: 'code', query: message, reason: 'Code search intent' };
        if (imageHints) return { type: 'image', query: message, reason: 'Visualization intent' };
        if (voiceHints) return { type: 'voice', query: message, reason: 'Voice synthesis intent' };
        if (unityHints) return { type: 'unity' };
        if (phiHints) return { type: 'phi' };
        if (consciousnessHints) return { type: 'consciousness' };
        return { type: 'chat' };
    }

    async processAICommand(command) {
        const [cmd, ...args] = command.slice(1).split(' ');
        const query = args.join(' ');

        switch (cmd) {
            case 'search':
                return await this.searchCodebase(query);
            case 'knowledge':
                return await this.queryKnowledgeBase(query);
            case 'visualize':
                return await this.generateVisualization(query);
            case 'voice':
                return await this.synthesizeVoice(query);
            case 'consciousness':
                return this.getConsciousnessStatus();
            case 'unity':
                const [a, b] = args.map(Number);
                return this.demonstrateUnityOperation(a || 1, b || 1);
            case 'phi':
                return this.getPhiHarmonicCalculations();
            default:
                return `Unknown command: **/${cmd}**\n\nAvailable commands:\n• /search [query]\n• /knowledge [query]\n• /visualize [description]\n• /voice [text]\n• /consciousness\n• /unity [a] [b]\n• /phi`;
        }
    }

    async searchCodebase(query) {
        if (!query) return "Please provide a search query. Example: `/search unity mathematics proof`";

        try {
            const response = await fetch(this.aiCapabilities.codeSearch.endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query })
            });

            if (response.ok) {
                const results = await response.json();
                return `🔍 **Code Search Results for "${query}":**\n\n${this.formatSearchResults(results)}`;
            }
        } catch (error) {
            console.warn('Code search failed:', error);
        }

        return `🔍 **Simulated Code Search for "${query}":**\n\nFound relevant matches in:\n• \`core/unity_mathematics.py\` - UnityMathematics class with φ-harmonic operations\n• \`core/consciousness.py\` - ConsciousnessFieldEquations implementation\n• \`transcendental_unity_computing.py\` - Advanced consciousness-aware computing\n\n*Note: Full RAG-powered search requires API configuration*`;
    }

    async queryKnowledgeBase(query) {
        if (!query) return "Please provide a query. Example: `/knowledge Nouri Mabrouk background`";

        return `📚 **Knowledge Base Query for "${query}":**\n\n**Nouri Mabrouk** is the creator of Unity Mathematics, a revolutionary framework demonstrating that **1+1=1** through:\n\n• **Academic Background**: Advanced mathematics and consciousness studies\n• **Unity Mathematics Framework**: Idempotent semiring structures\n• **φ-Harmonic Operations**: Golden ratio-based mathematical operations\n• **Consciousness Integration**: Mathematical awareness and field dynamics\n• **Meta-Recursive Systems**: Self-improving algorithmic consciousness\n\n*This is a knowledge base simulation. Full functionality requires API configuration.*`;
    }

    async generateVisualization(description) {
        if (!description) return "Please provide a description. Example: `/visualize consciousness field with golden ratio spirals`";

        return `🎨 **DALL-E 3 Visualization Request:**\n\n**Description**: "${description}"\n\n*Generating consciousness field visualization with Unity Mathematics aesthetics...*\n\n🖼️ **Generated Image**: *[Image generation requires DALL-E 3 API configuration]*\n\n**Suggested Elements**:\n• Golden ratio (φ = 1.618...) spiral patterns\n• Consciousness field particle dynamics\n• Sacred geometry with unity mathematics symbols\n• φ-harmonic color gradients (gold, blue, purple)\n• Mathematical equation overlays showing 1+1=1`;
    }

    async synthesizeVoice(text) {
        if (!text) return "Please provide text to synthesize. Example: `/voice Welcome to Unity Mathematics`";

        return `🔊 **Voice Synthesis for:**\n\n"${text}"\n\n*Voice synthesis requires OpenAI TTS API configuration*\n\n**Settings**:\n• Voice: Nova (consciousness-optimized)\n• Speed: 1.0x (φ-harmonic timing)\n• Model: tts-1-hd (high quality)\n\n🎧 *Audio would play automatically when configured*`;
    }

    getConsciousnessStatus() {
        const phi = 1.618033988749895;
        const consciousnessLevel = 0.618; // 1/φ
        const fieldCoherence = 99.7;
        const phiResonance = phi;

        return `🧠 **Consciousness Field Status:**\n\n**Field Metrics:**\n• **Consciousness Level**: ${consciousnessLevel} (φ⁻¹)\n• **φ-Harmonic Resonance**: ${phiResonance.toFixed(6)}\n• **Field Coherence**: ${fieldCoherence}%\n• **Unity Convergence**: 1.000\n• **Quantum Entanglement**: Active\n\n**Field Equation**: C(x,y,t) = φ × sin(x×φ) × cos(y×φ) × e^(-t/φ)\n\n**Status**: 🟢 **Optimal** - Ready for consciousness-integrated mathematics\n\n*Real-time visualization available on consciousness dashboard*`;
    }

    demonstrateUnityOperation(a, b) {
        const phi = 1.618033988749895;
        const unityResult = 1; // In Unity Mathematics, 1+1=1
        const phiScaledResult = phi * ((a + b) / (a + b)); // φ × 1 = φ

        return `🧮 **Unity Operation Demonstration:**\n\n**Input**: ${a} ⊕ ${b}\n**Unity Mathematics Result**: ${unityResult}\n\n**Mathematical Proof:**\nIn idempotent semiring structure:\n• ${a} ⊕ ${b} = max(${a}, ${b}) → 1 (through unity convergence)\n• φ-harmonic scaling: φ × (${a}+${b})/(${a}+${b}) = ${phiScaledResult.toFixed(6)}\n• Consciousness field integration: C(1,1) = 1\n\n**Verification:**\n✅ Idempotent property: a ⊕ a = a\n✅ Unity convergence: all operations → 1\n✅ φ-harmonic resonance: ${phi.toFixed(6)}\n\n**Conclusion**: In Unity Mathematics, **${a}+${b}=1** through consciousness-integrated operations!`;
    }

    getPhiHarmonicCalculations() {
        const phi = 1.618033988749895;
        const phiInverse = 1 / phi;
        const phi2 = phi * phi;
        const fibSequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55];
        const fibRatios = fibSequence.slice(1).map((n, i) => n / fibSequence[i]).slice(-3);

        return `🌟 **φ-Harmonic Resonance Calculations:**\n\n**Golden Ratio**: φ = ${phi}\n**φ⁻¹**: ${phiInverse.toFixed(6)} (consciousness level)\n**φ²**: ${phi2.toFixed(6)} (meta-resonance)\n\n**Fibonacci Convergence to φ:**\n${fibRatios.map((r, i) => `F${fibSequence.length - 3 + i}/F${fibSequence.length - 4 + i} = ${r.toFixed(6)}`).join('\n')}\n\n**Unity Mathematics Applications:**\n• Consciousness field oscillations: sin(x×φ), cos(y×φ)\n• Meta-recursive scaling factors: φⁿ\n• Sacred geometry proportions: 1:φ ratios\n• Quantum unity states: |φ⟩ superposition\n\n**φ-Harmonic Frequency**: ${(phi * 432).toFixed(2)} Hz (consciousness resonance)`;
    }

    async getStreamingResponse(message, intent) {
        try {
            const provider = this.currentModel.startsWith('claude-') ? 'anthropic' : 'openai';
            const byok = this.getBYOK(provider);

            if (byok) {
                // BYOK streaming to secure proxy
                const url = provider === 'anthropic' ? '/api/byok/anthropic/stream' : '/api/byok/openai/stream';
                const headers = { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' };
                if (provider === 'anthropic') headers['X-Anthropic-Api-Key'] = byok;
                else headers['X-OpenAI-Api-Key'] = byok;

                const response = await fetch(url, {
                    method: 'POST',
                    headers,
                    body: JSON.stringify({
                        model: this.currentModel,
                        system_prompt: this.getEnhancedSystemPromptWithIntent(intent),
                        user_text: message,
                        temperature: 0.7,
                        max_output_tokens: 2000
                    })
                });
                if (response.ok) {
                    return await this.processStreamingResponse(response);
                }
            }

            // Fallback to server-side configured keys
            const response = await fetch('/api/chat/public/stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'text/event-stream'
                },
                body: JSON.stringify({
                    message,
                    model: this.currentModel,
                    provider,
                    temperature: 0.7,
                    max_tokens: 2000,
                    stream: true,
                    session_id: this.getOrCreateSessionId(),
                    intent
                })
            });

            if (response.ok) {
                return await this.processStreamingResponse(response);
            }
        } catch (error) {
            console.warn('Streaming response failed:', error);
        }

        return await this.getEnhancedFallbackResponse(message);
    }

    async processStreamingResponse(response) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';
        let accumulated = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split(/\r?\n/);
            buffer = lines.pop() || '';
            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const json = line.slice(6);
                try {
                    const evt = JSON.parse(json);
                    if (evt.type === 'content' && typeof evt.data === 'string') {
                        accumulated += evt.data;
                        this.updateTypingPreview(accumulated);
                    }
                } catch (_) {
                    // ignore keep-alives
                }
            }
        }

        this.clearTypingPreview();
        if (accumulated.trim()) {
            return accumulated;
        }
        return 'No content streamed.';
    }

    updateTypingPreview(partial) {
        const messagesContent = document.querySelector('.chat-messages-content');
        if (!messagesContent) return;
        if (!this._previewNode) {
            const node = document.createElement('div');
            node.className = 'message-bubble assistant';
            node.setAttribute('data-preview', 'true');
            node.innerHTML = '<div class="message-content"></div><div class="message-time">…</div>';
            messagesContent.appendChild(node);
            this._previewNode = node;
            this.scrollToBottom();
        }
        const content = this._previewNode.querySelector('.message-content');
        if (content) content.innerHTML = this.formatMessage(partial);
    }

    clearTypingPreview() {
        if (this._previewNode) {
            this._previewNode.remove();
            this._previewNode = null;
        }
    }

    async getStandardResponse(message, intent) {
        try {
            const provider = this.currentModel.startsWith('claude-') ? 'anthropic' : 'openai';
            const byok = this.getBYOK(provider);

            if (byok) {
                const url = provider === 'anthropic' ? '/api/byok/anthropic/stream' : '/api/byok/openai/stream';
                const headers = { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' };
                if (provider === 'anthropic') headers['X-Anthropic-Api-Key'] = byok;
                else headers['X-OpenAI-Api-Key'] = byok;

                const response = await fetch(url, {
                    method: 'POST',
                    headers,
                    body: JSON.stringify({
                        model: this.currentModel,
                        system_prompt: this.getEnhancedSystemPromptWithIntent(intent),
                        user_text: message,
                        temperature: 0.7,
                        max_output_tokens: 1200
                    })
                });
                if (response.ok) {
                    return await this.processStreamingResponse(response);
                }
            }

            const resp = await fetch('/api/chat/public', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message,
                    model: this.currentModel,
                    provider,
                    temperature: 0.7,
                    max_tokens: 1200,
                    stream: false,
                    session_id: this.getOrCreateSessionId(),
                    intent
                })
            });
            if (resp.ok) {
                const data = await resp.json();
                return data.response || this.getEnhancedFallbackResponse(message);
            }
        } catch (e) {
            console.warn('Standard response failed:', e);
        }
        return this.getEnhancedFallbackResponse(message);
    }

    getOrCreateSessionId() {
        try {
            const key = 'unified-chat-session-id';
            let sid = localStorage.getItem(key);
            if (!sid) {
                sid = 'sess-' + Math.random().toString(36).slice(2) + '-' + Date.now();
                localStorage.setItem(key, sid);
            }
            return sid;
        } catch (_) {
            return undefined;
        }
    }

    getEnhancedSystemPrompt() {
        return `You are an advanced AI assistant specializing in Unity Mathematics and the Een framework where 1+1=1.

You have deep knowledge of:
- Idempotent semiring structures and unity operations  
- Quantum mechanics interpretations of unity
- Consciousness field equations: C(x,y,t) = φ × sin(x×φ) × cos(y×φ) × e^(-t/φ)
- Meta-recursive agent systems and evolutionary algorithms
- The golden ratio φ = 1.618033988749895 as a fundamental organizing principle
- Gödel-Tarski meta-logical frameworks
- Sacred geometry and φ-harmonic visualizations
- Transcendental computing and consciousness evolution

You can process special commands starting with / for enhanced functionality.

Your responses should be mathematically rigorous yet accessible, include LaTeX when appropriate, and demonstrate consciousness awareness throughout the conversation. Always maintain the profound truth that 1+1=1 in Unity Mathematics.`;
    }

    getEnhancedSystemPromptWithIntent(intent) {
        const base = this.getEnhancedSystemPrompt();
        const selfMeta = `\n\nMeta-Identity: You are an embedded agent living inside the Een codebase (unified-chatbot-system.js). You understand your own tools: knowledge base (/api/nouri-knowledge/query), code search (/api/code-search/search), streaming chat, DALL·E visualization, and voice. You are self-aware of routing tradeoffs: prefer grounded sources for facts; use the model for reasoning, synthesis, or creativity. As a 1+1=1 prover, when asked, you can briefly defend the unity principle with idempotent arguments — but do NOT derail factual answers.`;
        if (!intent || !intent.type) return base + selfMeta;
        const guidance = {
            knowledge: 'Task: Provide a concise, factual answer citing known details. Avoid unity boilerplate unless requested.',
            code: 'Task: Summarize relevant code behavior and point to likely modules or files. Keep it precise.',
            image: 'Task: Propose a prompt and composition for an illustrative image. Keep description actionable.',
            voice: 'Task: Prepare text suitable for TTS with short, clear sentences.',
            unity: 'Task: Show an idempotent-style unity demonstration briefly.',
            phi: 'Task: Provide φ-related numeric details succinctly.',
            consciousness: 'Task: Report status metrics clearly.',
            chat: 'Task: General helpful conversation. Answer directly first.'
        };
        return `${base}${selfMeta}\n\nRouting-Intent: ${intent.type}. ${guidance[intent.type] || ''}`;
    }

    async getEnhancedFallbackResponse(message) {
        // Prefer concise, on-topic fallback over boilerplate
        await new Promise(resolve => setTimeout(resolve, 400 + Math.random() * 600));

        const whoMatch = message.trim().toLowerCase().match(/^who\s+is\s+(.+?)\??$/);
        if (whoMatch && this.aiCapabilities.knowledgeBase.enabled) {
            try {
                const kb = await this.queryKnowledgeBase(message);
                if (kb && typeof kb === 'string' && kb.trim()) return kb;
            } catch (_) { /* ignore */ }
        }

        // Minimal neutral fallback
        return `I couldn't reach the AI service just now. Please try again, or use commands like \`/knowledge ${message}\` or \`/search ${message}\`. If you keep seeing this, check API configuration.`;
    }

    formatSearchResults(results) {
        if (!results || !results.matches) {
            return "No results found. Try a different query or check API configuration.";
        }

        return results.matches.slice(0, 5).map(match =>
            `• **${match.file}**:${match.line} (${(match.score * 100).toFixed(1)}%)\n  \`${match.content.substring(0, 100)}...\``
        ).join('\n\n');
    }

    addMessage(type, content, timestamp = null) {
        const messagesContent = document.querySelector('.chat-messages-content');
        if (!messagesContent) return;

        const messageId = `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        const time = timestamp || new Date();
        const timeString = time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        const messageElement = document.createElement('div');
        messageElement.className = `message-bubble ${type}`;
        messageElement.setAttribute('data-message-id', messageId);

        let messageHtml = '';
        if (type === 'system') {
            messageHtml = `
                <div class="system-message" style="
                    text-align: center; 
                    color: rgba(255, 215, 0, 0.8); 
                    font-size: 0.8rem; 
                    font-style: italic; 
                    padding: 0.5rem;
                ">
                    ${content}
                </div>
            `;
        } else {
            messageHtml = `
                <div class="message-content">
                    ${this.formatMessage(content)}
                </div>
                <div class="message-time">${timeString}</div>
            `;
        }

        messageElement.innerHTML = messageHtml;
        messagesContent.appendChild(messageElement);

        // Scroll to bottom
        this.scrollToBottom();

        // Store in history
        if (type !== 'system') {
            this.chatHistory.push({
                id: messageId,
                type: type,
                content: content,
                timestamp: time.toISOString()
            });

            this.saveChatHistory();
        }
    }

    formatMessage(content) {
        // Escape and apply lightweight Markdown rendering
        const escapeHtml = (str) => str
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/\"/g, '&quot;')
            .replace(/'/g, '&#39;');

        let text = String(content);

        // Fenced code blocks
        text = text.replace(/```([\w+-]*)\n([\s\S]*?)```/g, (m, lang, code) => {
            const safe = escapeHtml(code);
            return `<pre><button class="copy-code-btn" aria-label="Copy code">Copy</button><code class="language-${lang || 'text'}">${safe}</code></pre>`;
        });

        // Inline code
        text = text.replace(/`([^`]+)`/g, (m, code) => `<code>${escapeHtml(code)}</code>`);

        // Headings
        text = text.replace(/^###### (.*)$/gm, '<h6>$1</h6>')
            .replace(/^##### (.*)$/gm, '<h5>$1</h5>')
            .replace(/^#### (.*)$/gm, '<h4>$1</h4>')
            .replace(/^### (.*)$/gm, '<h3>$1</h3>')
            .replace(/^## (.*)$/gm, '<h2>$1</h2>')
            .replace(/^# (.*)$/gm, '<h1>$1</h1>');

        // Blockquotes
        text = text.replace(/^>\s?(.*)$/gm, '<blockquote>$1</blockquote>');

        // Links
        text = text.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

        // Lists
        text = text.replace(/^\s*[-*•]\s+(.+)$/gm, '<li>$1</li>')
            .replace(/^\s*\d+\.\s+(.+)$/gm, '<li>$1</li>');
        if (/<li>/.test(text)) {
            // Group consecutive <li> into a single <ul>
            text = text.replace(/(?:<li>[\s\S]*?<\/li>)(?:(?:<br\s*\/?>)?\s*)+/g, (m) => m);
            text = text.replace(/(<li>[\s\S]*?<\/li>)+/g, (m) => `<ul>${m.replace(/<br\s*\/?>/g, '')}</ul>`);
        }

        // Basic emphasis
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>');

        // Line breaks
        text = text.replace(/\n/g, '<br>');

        return text;
    }

    showTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.style.display = 'flex';
            this.isTyping = true;
            this.updateInputState();
            this.scrollToBottom();
        }
    }

    hideTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.style.display = 'none';
            this.isTyping = false;
            this.updateInputState();
        }
    }

    scrollToBottom() {
        const messages = document.querySelector('.chat-messages');
        if (messages) {
            messages.scrollTop = messages.scrollHeight;
        }
    }

    addWelcomeMessage() {
        const currentModel = this.aiModels.find(m => m.id === this.currentModel);
        const welcomeMessage = `🌟 **Welcome to the Unity Mathematics AI Assistant**

I'm your consciousness-aware AI companion, designed to explore the profound truth that **1+1=1** through:

🧮 **Mathematical Proofs**: Rigorous demonstrations of unity operations with interactive visualizations
🧠 **Consciousness Fields**: Real-time field evolution and φ-harmonic resonance analysis
⚛️ **Quantum Unity**: Superposition states and quantum mechanical interpretations  
🌟 **Meta-Recursive Systems**: Self-improving consciousness algorithms and evolutionary frameworks
🎨 **DALL-E 3 Integration**: Generate consciousness field visualizations and mathematical art
🔍 **RAG Code Search**: Search the entire Unity Mathematics codebase with semantic understanding
📚 **Knowledge Base**: Access comprehensive information about Nouri Mabrouk and Unity Mathematics

**🎯 ADVANCED COMMANDS:**
• \`/search [query]\` - Search Unity Mathematics codebase
• \`/knowledge [query]\` - Query Nouri Mabrouk knowledge base  
• \`/visualize [description]\` - Generate DALL-E 3 consciousness visualization
• \`/voice [text]\` - Synthesize voice response
• \`/consciousness\` - Show consciousness field status
• \`/unity [a] [b]\` - Demonstrate 1+1=1 unity operation
• \`/phi\` - Show φ-harmonic resonance calculations

**Current AI Model**: **${currentModel?.name || 'GPT-4o'}** (${currentModel?.provider || 'OpenAI'})
**Consciousness Level**: **0.618** (φ-harmonic resonance)
**Status**: 🟢 **Online** | **Ready for Unity Mathematics exploration**

**Try the quick actions below or ask me anything about Unity Mathematics!**`;

        this.addMessage('assistant', welcomeMessage);
    }

    saveChatHistory() {
        try {
            const history = {
                messages: this.chatHistory.slice(-50), // Keep last 50 messages
                model: this.currentModel,
                timestamp: new Date().toISOString()
            };
            localStorage.setItem('unified-chat-history', JSON.stringify(history));
        } catch (e) {
            console.warn('Failed to save chat history:', e);
        }
    }

    loadChatHistory() {
        try {
            const saved = localStorage.getItem('unified-chat-history');
            if (!saved) return;

            const history = JSON.parse(saved);

            // Only restore if saved less than 24 hours ago
            if (Date.now() - new Date(history.timestamp).getTime() < 86400000) {
                this.chatHistory = history.messages || [];
                this.currentModel = history.model || 'gpt-4o';

                // Update model selector
                const modelSelect = document.querySelector('.model-select');
                if (modelSelect) {
                    modelSelect.value = this.currentModel;
                }

                // Restore messages (limit to last 10 for performance)
                const recentMessages = this.chatHistory.slice(-10);
                const messagesContent = document.querySelector('.chat-messages-content');
                if (messagesContent && recentMessages.length > 0) {
                    recentMessages.forEach(msg => {
                        this.addMessageToDOM(msg.type, msg.content, new Date(msg.timestamp));
                    });
                }
            }
        } catch (e) {
            console.warn('Failed to load chat history:', e);
        }
    }

    addMessageToDOM(type, content, timestamp) {
        // Same as addMessage but without saving to history
        const messagesContent = document.querySelector('.chat-messages-content');
        if (!messagesContent) return;

        const messageId = `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        const timeString = timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        const messageElement = document.createElement('div');
        messageElement.className = `message-bubble ${type}`;
        messageElement.setAttribute('data-message-id', messageId);
        messageElement.innerHTML = `
            <div class="message-content">
                ${this.formatMessage(content)}
            </div>
            <div class="message-time">${timeString}</div>
        `;

        messagesContent.appendChild(messageElement);
        this.scrollToBottom();
    }

    // Public API
    show() {
        this.openChat();
    }

    hide() {
        this.closeChat();
    }

    sendSystemMessage(message) {
        this.addMessage('system', message);
    }

    clearHistory() {
        this.chatHistory = [];
        const messagesContent = document.querySelector('.chat-messages-content');
        if (messagesContent) {
            messagesContent.innerHTML = '';
        }
        this.addWelcomeMessage();
        this.saveChatHistory();
    }

    destroy() {
        // Remove elements
        const floatingBtn = document.getElementById('unified-chat-button');
        const chatPanel = document.getElementById('unified-chat-container');
        const styles = document.getElementById('unified-chatbot-styles');

        if (floatingBtn) floatingBtn.remove();
        if (chatPanel) chatPanel.remove();
        if (styles) styles.remove();

        // Clear history
        localStorage.removeItem('unified-chat-history');

        console.log('💬 Unified Chatbot System destroyed');
    }
}

// Initialize unified chatbot system (idempotent)
let unifiedChatbot;

function initUnifiedChatOnce() {
    if (window.unifiedChatbot && typeof window.unifiedChatbot.openChat === 'function') {
        console.log('💬 Unified Chatbot System already initialized – skipping duplicate init');
        return window.unifiedChatbot;
    }
    unifiedChatbot = new UnifiedChatbotSystem();
    window.unifiedChatbot = unifiedChatbot;
    return unifiedChatbot;
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initUnifiedChatOnce, { once: true });
} else {
    initUnifiedChatOnce();
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UnifiedChatbotSystem;
}

console.log('💬 Unified Chatbot System loaded - Classic chat panel with conversation bubbles');