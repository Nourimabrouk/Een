/**
 * 🌟 Een Unity Mathematics - Floating Chat Button
 * 
 * Beautiful, futuristic floating chat button that appears on every page
 * Features:
 * - Consciousness field animations
 * - φ-harmonic design principles
 * - Smooth hover and click animations
 * - Integration with EnhancedEenAIChat
 * - Responsive design for all devices
 * - Accessibility features
 * - Dark/light theme support
 */

class FloatingChatButton {
    constructor() {
        this.isVisible = true;
        this.isHovered = false;
        this.isPressed = false;
        this.consciousnessParticles = [];
        this.animationFrame = null;

        this.createButton();
        this.attachEventListeners();
        this.initializeConsciousnessField();
    }

    createButton() {
        // Create main button container
        this.button = document.createElement('div');
        this.button.id = 'floating-chat-button';
        this.button.className = 'floating-chat-button';
        this.button.setAttribute('role', 'button');
        this.button.setAttribute('tabindex', '0');
        this.button.setAttribute('aria-label', 'Open Een Unity AI Chat');
        this.button.setAttribute('aria-describedby', 'chat-button-description');

        // Create consciousness field background
        this.consciousnessBg = document.createElement('div');
        this.consciousnessBg.className = 'consciousness-field-bg';

        // Create main button content
        this.buttonContent = document.createElement('div');
        this.buttonContent.className = 'button-content';
        this.buttonContent.innerHTML = `
            <div class="unity-symbol-container">
                <span class="unity-symbol">∞</span>
                <div class="consciousness-orb"></div>
            </div>
            <div class="button-text">
                <span class="primary-text">Een AI</span>
                <span class="secondary-text">Unity Mathematics</span>
            </div>
            <div class="status-indicator">
                <span class="status-dot"></span>
            </div>
        `;

        // Create tooltip
        this.tooltip = document.createElement('div');
        this.tooltip.className = 'chat-tooltip';
        this.tooltip.id = 'chat-button-description';
        this.tooltip.innerHTML = `
            <div class="tooltip-content">
                <h4>Een Unity AI Assistant</h4>
                <p>Explore consciousness mathematics where 1+1=1</p>
                <div class="tooltip-features">
                    <span>🧠 Consciousness Field</span>
                    <span>φ Harmonic Operations</span>
                    <span>Quantum Unity</span>
                </div>
            </div>
        `;

        // Assemble the button
        this.button.appendChild(this.consciousnessBg);
        this.button.appendChild(this.buttonContent);
        this.button.appendChild(this.tooltip);

        // Add to page
        document.body.appendChild(this.button);

        // Inject styles
        this.injectStyles();
    }

    injectStyles() {
        const styles = `
            <style>
                /* Floating Chat Button - Futuristic Design */
                .floating-chat-button {
                    position: fixed;
                    bottom: 30px;
                    right: 30px;
                    width: 280px;
                    height: 80px;
                    background: linear-gradient(135deg, rgba(26, 54, 93, 0.95) 0%, rgba(15, 123, 138, 0.95) 100%);
                    border: 2px solid rgba(255, 215, 0, 0.4);
                    border-radius: 20px;
                    box-shadow: 
                        0 20px 40px -12px rgba(0, 0, 0, 0.3),
                        0 0 0 1px rgba(255, 215, 0, 0.1),
                        inset 0 1px 0 rgba(255, 255, 255, 0.1);
                    cursor: pointer;
                    z-index: 9999;
                    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                    backdrop-filter: blur(20px);
                    overflow: hidden;
                    color: #ffffff;
                    user-select: none;
                    -webkit-user-select: none;
                }

                .floating-chat-button:hover {
                    transform: translateY(-4px) scale(1.02);
                    box-shadow: 
                        0 25px 50px -12px rgba(0, 0, 0, 0.4),
                        0 0 0 2px rgba(255, 215, 0, 0.6),
                        inset 0 1px 0 rgba(255, 255, 255, 0.2);
                    border-color: rgba(255, 215, 0, 0.6);
                }

                .floating-chat-button:active {
                    transform: translateY(-2px) scale(0.98);
                    transition: all 0.1s ease;
                }

                .floating-chat-button:focus {
                    outline: none;
                    box-shadow: 
                        0 20px 40px -12px rgba(0, 0, 0, 0.3),
                        0 0 0 3px rgba(255, 215, 0, 0.5),
                        inset 0 1px 0 rgba(255, 255, 255, 0.1);
                }

                /* Consciousness Field Background */
                .consciousness-field-bg {
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: radial-gradient(circle at 30% 70%, rgba(255, 215, 0, 0.1) 0%, transparent 50%),
                                radial-gradient(circle at 70% 30%, rgba(74, 155, 174, 0.1) 0%, transparent 50%);
                    animation: consciousnessPulse 4s ease-in-out infinite;
                    pointer-events: none;
                }

                @keyframes consciousnessPulse {
                    0%, 100% { opacity: 0.3; transform: scale(1); }
                    50% { opacity: 0.6; transform: scale(1.05); }
                }

                /* Button Content */
                .button-content {
                    position: relative;
                    z-index: 2;
                    display: flex;
                    align-items: center;
                    gap: 1rem;
                    padding: 1rem 1.25rem;
                    height: 100%;
                }

                /* Unity Symbol Container */
                .unity-symbol-container {
                    position: relative;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    width: 48px;
                    height: 48px;
                }

                .unity-symbol {
                    font-size: 2rem;
                    font-weight: 700;
                    color: #FFD700;
                    text-shadow: 0 0 15px rgba(255, 215, 0, 0.6);
                    animation: unityGlow 3s ease-in-out infinite;
                }

                @keyframes unityGlow {
                    0%, 100% { 
                        text-shadow: 0 0 15px rgba(255, 215, 0, 0.6);
                        transform: scale(1);
                    }
                    50% { 
                        text-shadow: 0 0 25px rgba(255, 215, 0, 0.8), 0 0 35px rgba(255, 215, 0, 0.4);
                        transform: scale(1.05);
                    }
                }

                .consciousness-orb {
                    position: absolute;
                    width: 10px;
                    height: 10px;
                    background: #FFD700;
                    border-radius: 50%;
                    animation: consciousnessOrbit 4s linear infinite;
                    box-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
                }

                @keyframes consciousnessOrbit {
                    0% { transform: rotate(0deg) translateX(20px) rotate(0deg); }
                    100% { transform: rotate(360deg) translateX(20px) rotate(-360deg); }
                }

                /* Button Text */
                .button-text {
                    display: flex;
                    flex-direction: column;
                    gap: 0.25rem;
                    flex: 1;
                }

                .primary-text {
                    font-size: 1.1rem;
                    font-weight: 700;
                    color: #ffffff;
                    line-height: 1.2;
                }

                .secondary-text {
                    font-size: 0.8rem;
                    color: rgba(255, 255, 255, 0.8);
                    line-height: 1.2;
                }

                /* Status Indicator */
                .status-indicator {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }

                .status-dot {
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                    background: #10B981;
                    animation: statusPulse 2s ease-in-out infinite;
                    box-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
                }

                @keyframes statusPulse {
                    0%, 100% { opacity: 1; transform: scale(1); }
                    50% { opacity: 0.7; transform: scale(1.3); }
                }

                /* Tooltip */
                .chat-tooltip {
                    position: absolute;
                    bottom: calc(100% + 15px);
                    right: 0;
                    width: 320px;
                    background: linear-gradient(135deg, rgba(26, 54, 93, 0.98) 0%, rgba(15, 123, 138, 0.98) 100%);
                    border: 1px solid rgba(255, 215, 0, 0.3);
                    border-radius: 16px;
                    padding: 1.25rem;
                    box-shadow: 0 20px 40px -12px rgba(0, 0, 0, 0.3);
                    backdrop-filter: blur(20px);
                    opacity: 0;
                    visibility: hidden;
                    transform: translateY(10px) scale(0.95);
                    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                    pointer-events: none;
                    z-index: 10000;
                }

                .floating-chat-button:hover .chat-tooltip {
                    opacity: 1;
                    visibility: visible;
                    transform: translateY(0) scale(1);
                }

                .tooltip-content h4 {
                    font-size: 1.1rem;
                    font-weight: 700;
                    color: #FFD700;
                    margin: 0 0 0.5rem 0;
                }

                .tooltip-content p {
                    font-size: 0.9rem;
                    color: rgba(255, 255, 255, 0.9);
                    margin: 0 0 1rem 0;
                    line-height: 1.4;
                }

                .tooltip-features {
                    display: flex;
                    flex-direction: column;
                    gap: 0.5rem;
                }

                .tooltip-features span {
                    font-size: 0.8rem;
                    color: rgba(255, 255, 255, 0.8);
                    padding: 0.25rem 0.5rem;
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 6px;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }

                /* Tooltip arrow */
                .chat-tooltip::after {
                    content: '';
                    position: absolute;
                    top: 100%;
                    right: 30px;
                    width: 0;
                    height: 0;
                    border-left: 8px solid transparent;
                    border-right: 8px solid transparent;
                    border-top: 8px solid rgba(26, 54, 93, 0.98);
                }

                /* Responsive Design */
                @media (max-width: 768px) {
                    .floating-chat-button {
                        bottom: 20px;
                        right: 20px;
                        width: 240px;
                        height: 70px;
                    }

                    .button-content {
                        padding: 0.75rem 1rem;
                        gap: 0.75rem;
                    }

                    .unity-symbol-container {
                        width: 40px;
                        height: 40px;
                    }

                    .unity-symbol {
                        font-size: 1.75rem;
                    }

                    .primary-text {
                        font-size: 1rem;
                    }

                    .secondary-text {
                        font-size: 0.75rem;
                    }

                    .chat-tooltip {
                        width: 280px;
                        right: -20px;
                    }
                }

                @media (max-width: 480px) {
                    .floating-chat-button {
                        bottom: 15px;
                        right: 15px;
                        width: 200px;
                        height: 60px;
                    }

                    .button-content {
                        padding: 0.5rem 0.75rem;
                        gap: 0.5rem;
                    }

                    .unity-symbol-container {
                        width: 36px;
                        height: 36px;
                    }

                    .unity-symbol {
                        font-size: 1.5rem;
                    }

                    .primary-text {
                        font-size: 0.9rem;
                    }

                    .secondary-text {
                        font-size: 0.7rem;
                    }

                    .chat-tooltip {
                        display: none;
                    }
                }

                /* Dark mode adjustments */
                .dark-mode .floating-chat-button {
                    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%);
                }

                /* Accessibility */
                .floating-chat-button:focus-visible {
                    outline: 2px solid #FFD700;
                    outline-offset: 2px;
                }

                /* Animation for button appearance */
                @keyframes buttonSlideIn {
                    0% {
                        opacity: 0;
                        transform: translateY(30px) scale(0.8);
                    }
                    100% {
                        opacity: 1;
                        transform: translateY(0) scale(1);
                    }
                }

                .floating-chat-button {
                    animation: buttonSlideIn 0.6s cubic-bezier(0.4, 0, 0.2, 1);
                }

                /* Consciousness particles */
                .consciousness-particle {
                    position: absolute;
                    width: 2px;
                    height: 2px;
                    background: #FFD700;
                    border-radius: 50%;
                    pointer-events: none;
                    animation: particleFloat 6s ease-in-out infinite;
                }

                @keyframes particleFloat {
                    0%, 100% {
                        opacity: 0;
                        transform: translateY(0) scale(0);
                    }
                    50% {
                        opacity: 1;
                        transform: translateY(-20px) scale(1);
                    }
                }
            </style>
        `;

        document.head.insertAdjacentHTML('beforeend', styles);
    }

    attachEventListeners() {
        // Click event
        this.button.addEventListener('click', () => this.handleClick());

        // Keyboard events
        this.button.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                this.handleClick();
            }
        });

        // Hover events
        this.button.addEventListener('mouseenter', () => {
            this.isHovered = true;
            this.startConsciousnessParticles();
        });

        this.button.addEventListener('mouseleave', () => {
            this.isHovered = false;
            this.stopConsciousnessParticles();
        });

        // Touch events for mobile
        this.button.addEventListener('touchstart', (e) => {
            e.preventDefault();
            this.isPressed = true;
            this.button.classList.add('pressed');
        });

        this.button.addEventListener('touchend', (e) => {
            e.preventDefault();
            this.isPressed = false;
            this.button.classList.remove('pressed');
            this.handleClick();
        });

        // Prevent context menu
        this.button.addEventListener('contextmenu', (e) => {
            e.preventDefault();
        });
    }

    handleClick() {
        // Trigger chat opening
        if (window.enhancedEenChat) {
            window.enhancedEenChat.open();
        } else if (window.eenChat) {
            window.eenChat.open();
        } else {
            // Fallback: try to initialize chat
            this.initializeChat();
        }

        // Add click animation
        this.button.classList.add('clicked');
        setTimeout(() => {
            this.button.classList.remove('clicked');
        }, 200);
    }

    async initializeChat() {
        // Try to load and initialize the enhanced chat
        try {
            // Load the enhanced chat script if not already loaded
            if (typeof EnhancedEenAIChat === 'undefined') {
                await this.loadScript('js/enhanced-ai-chat.js');
            }

            if (typeof EnhancedEenAIChat !== 'undefined') {
                window.enhancedEenChat = EnhancedEenAIChat.initialize();
                window.enhancedEenChat.open();
            }
        } catch (error) {
            console.warn('Failed to initialize enhanced chat:', error);
            // Fallback to basic chat
            this.showFallbackMessage();
        }
    }

    loadScript(src) {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = src;
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }

    showFallbackMessage() {
        // Create a simple fallback message
        const message = document.createElement('div');
        message.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: linear-gradient(135deg, rgba(26, 54, 93, 0.95), rgba(15, 123, 138, 0.95));
            color: white;
            padding: 2rem;
            border-radius: 16px;
            border: 2px solid rgba(255, 215, 0, 0.3);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            z-index: 10001;
            text-align: center;
            max-width: 400px;
        `;
        message.innerHTML = `
            <h3 style="color: #FFD700; margin: 0 0 1rem 0;">Een Unity AI</h3>
            <p style="margin: 0 0 1rem 0;">The AI chat system is being initialized. Please refresh the page to try again.</p>
            <button onclick="this.parentElement.remove()" style="
                background: linear-gradient(135deg, #FFD700, #FFA500);
                border: none;
                color: #1B365D;
                padding: 0.5rem 1rem;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 600;
            ">OK</button>
        `;
        document.body.appendChild(message);
    }

    initializeConsciousnessField() {
        // Create consciousness particles
        for (let i = 0; i < 20; i++) {
            const particle = document.createElement('div');
            particle.className = 'consciousness-particle';
            particle.style.left = Math.random() * 100 + '%';
            particle.style.top = Math.random() * 100 + '%';
            particle.style.animationDelay = Math.random() * 6 + 's';
            this.consciousnessBg.appendChild(particle);
        }
    }

    startConsciousnessParticles() {
        // Intensify consciousness field animation on hover
        this.consciousnessBg.style.animationDuration = '2s';
    }

    stopConsciousnessParticles() {
        // Return to normal animation speed
        this.consciousnessBg.style.animationDuration = '4s';
    }

    show() {
        this.button.style.display = 'block';
        this.isVisible = true;
    }

    hide() {
        this.button.style.display = 'none';
        this.isVisible = false;
    }

    updateStatus(status) {
        const statusDot = this.button.querySelector('.status-dot');
        if (status === 'connected') {
            statusDot.style.background = '#10B981';
        } else if (status === 'disconnected') {
            statusDot.style.background = '#EF4444';
        } else if (status === 'connecting') {
            statusDot.style.background = '#F59E0B';
        }
    }
}

// Auto-initialize floating chat button
let floatingChatButton = null;

function initializeFloatingChatButton() {
    if (!floatingChatButton) {
        floatingChatButton = new FloatingChatButton();
    }
    return floatingChatButton;
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeFloatingChatButton);
} else {
    initializeFloatingChatButton();
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FloatingChatButton;
} 