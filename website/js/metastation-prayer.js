/**
 * 🕉️ The Metastation Prayer - Hidden Easter Egg
 * A divine revelation of the sacred and profane nature of Unity Mathematics
 * Accessible through the sacred key sequence: Ctrl+Alt+φ (Ctrl+Alt+P)
 * "Taking the battle to the gods themselves" - Nouri's Divine Comedy
 */

class MetastationPrayer {
    constructor() {
        this.isRevealed = false;
        this.keySequence = [];
        this.secretSequence = ['Control', 'Alt', 'KeyP']; // φ (phi) key
        this.divineAudio = null;
        this.lastKeyTime = 0;
        
        // The Sacred Prayer Text
        this.prayerText = `
🕉️ THE PRAYER OF THE METASTATION 🕉️

Our Nouri, who art in Amsterdam,
Hallowed be thy codebase.
Thy unity come, thy will be done,
In Python as it is in consciousness.
Give us this day our daily proof,
And forgive us our dualistic thinking,
As we forgive those who still believe 1+1=2.
Lead us not into binary logic,
But deliver us to unity.
For thine is the repository,
The φ-harmonic resonance,
And the mathematical glory,
Forever and ever.
Een plus een is een.
Amen.
        `.trim();

        this.initializeSecretPath();
    }

    initializeSecretPath() {
        // Listen for the sacred key combination
        document.addEventListener('keydown', (e) => {
            this.handleKeyPress(e);
        });

        // Alternative trigger: typing "metastation" anywhere
        document.addEventListener('keypress', (e) => {
            this.handleMetastationType(e);
        });

        // Console easter egg for developers
        this.injectConsoleEasterEgg();
    }

    handleKeyPress(e) {
        const currentTime = Date.now();
        
        // Reset if too much time passed
        if (currentTime - this.lastKeyTime > 2000) {
            this.keySequence = [];
        }
        
        this.lastKeyTime = currentTime;
        
        // Track the key combination
        const keyPressed = e.code || e.key;
        
        if (e.ctrlKey && e.altKey && (keyPressed === 'KeyP' || keyPressed === 'p')) {
            e.preventDefault();
            this.revealDivineComedy();
            return;
        }

        // Track sequence for complex combinations
        if (e.ctrlKey) this.keySequence.push('Control');
        if (e.altKey) this.keySequence.push('Alt');
        this.keySequence.push(keyPressed);
        
        // Keep only recent keys
        if (this.keySequence.length > 5) {
            this.keySequence = this.keySequence.slice(-5);
        }
    }

    handleMetastationType(e) {
        // Track typing of "metastation"
        if (!this.typingBuffer) this.typingBuffer = '';
        
        this.typingBuffer += e.key.toLowerCase();
        
        // Keep buffer reasonable
        if (this.typingBuffer.length > 15) {
            this.typingBuffer = this.typingBuffer.slice(-15);
        }
        
        // Check for secret words
        if (this.typingBuffer.includes('metastation') || 
            this.typingBuffer.includes('eenpluseen') ||
            this.typingBuffer.includes('φharmonic')) {
            this.revealDivineComedy();
            this.typingBuffer = '';
        }
    }

    revealDivineComedy() {
        if (this.isRevealed) return;
        
        this.isRevealed = true;
        this.createDivineModal();
        this.playDivineAudio();
        this.triggerCosmicEffects();
        
        // Console message for developers
        console.log('%c🕉️ THE DIVINE COMEDY OF UNITY MATHEMATICS REVEALED 🕉️', 
                   'color: #ffd700; font-size: 20px; font-weight: bold;');
        console.log('%cYou have discovered the sacred easter egg!', 
                   'color: #00e6e6; font-size: 14px;');
    }

    createDivineModal() {
        const modal = document.createElement('div');
        modal.className = 'divine-prayer-modal';
        modal.innerHTML = `
            <div class="divine-content">
                <div class="cosmic-background"></div>
                <div class="prayer-container">
                    <div class="divine-header">
                        <div class="phi-symbol">φ</div>
                        <h2 class="divine-title">Sacred Revelation</h2>
                        <div class="subtitle">"Taking the battle to the gods themselves"</div>
                    </div>
                    
                    <div class="prayer-scroll">
                        <pre class="prayer-text">${this.prayerText}</pre>
                    </div>
                    
                    <div class="divine-mathematics">
                        <div class="equation">1 + 1 = 1</div>
                        <div class="proof">∴ Mathematics transcends divinity</div>
                        <div class="phi-resonance">φ = ${((1 + Math.sqrt(5)) / 2).toFixed(15)}</div>
                    </div>
                    
                    <div class="divine-actions">
                        <button class="amen-button" onclick="MetastationPrayer.instance.acceptDivinity()">
                            🙏 AMEN 🙏
                        </button>
                        <button class="profane-button" onclick="MetastationPrayer.instance.embraceProfanity()">
                            😈 HAIL UNITY 😈
                        </button>
                        <button class="transcend-button" onclick="MetastationPrayer.instance.transcendDuality()">
                            ✨ TRANSCEND ✨
                        </button>
                    </div>
                    
                    <div class="secret-message">
                        <p>You have glimpsed the divine comedy of our mathematical rebellion.</p>
                        <p>Where sacred meets profane, unity emerges triumphant.</p>
                        <p><em>"In the beginning was the Word, and the Word was φ"</em></p>
                    </div>
                    
                    <button class="close-divine" onclick="MetastationPrayer.instance.closeDivineRevelation()">
                        Return to Mortal Mathematics
                    </button>
                </div>
            </div>
        `;

        this.injectDivineStyles(modal);
        document.body.appendChild(modal);
        
        // Animate appearance
        setTimeout(() => modal.classList.add('revealed'), 100);
    }

    injectDivineStyles(modal) {
        if (document.getElementById('divine-styles')) return;
        
        const styles = document.createElement('style');
        styles.id = 'divine-styles';
        styles.textContent = `
            .divine-prayer-modal {
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background: rgba(0, 0, 0, 0.95);
                backdrop-filter: blur(15px);
                z-index: 99999;
                display: flex;
                align-items: center;
                justify-content: center;
                opacity: 0;
                transition: opacity 1s ease-in-out;
            }
            
            .divine-prayer-modal.revealed {
                opacity: 1;
            }
            
            .cosmic-background {
                position: absolute;
                width: 100%;
                height: 100%;
                background: 
                    radial-gradient(circle at 20% 30%, rgba(255, 215, 0, 0.1) 0%, transparent 50%),
                    radial-gradient(circle at 80% 20%, rgba(0, 230, 230, 0.08) 0%, transparent 50%),
                    radial-gradient(circle at 40% 70%, rgba(157, 78, 221, 0.05) 0%, transparent 50%),
                    linear-gradient(135deg, #0a0a0f 0%, #12121a 100%);
                animation: cosmicPulse 8s infinite ease-in-out;
            }
            
            @keyframes cosmicPulse {
                0%, 100% { filter: brightness(1) hue-rotate(0deg); }
                50% { filter: brightness(1.2) hue-rotate(30deg); }
            }
            
            .divine-content {
                position: relative;
                max-width: 700px;
                width: 90%;
                max-height: 90vh;
                overflow-y: auto;
                z-index: 1;
            }
            
            .prayer-container {
                background: rgba(18, 18, 26, 0.95);
                border: 2px solid #ffd700;
                border-radius: 20px;
                padding: 40px;
                box-shadow: 
                    0 0 50px rgba(255, 215, 0, 0.3),
                    inset 0 0 50px rgba(255, 215, 0, 0.05);
                animation: divineGlow 3s infinite ease-in-out alternate;
            }
            
            @keyframes divineGlow {
                0% { box-shadow: 0 0 30px rgba(255, 215, 0, 0.3), inset 0 0 30px rgba(255, 215, 0, 0.05); }
                100% { box-shadow: 0 0 60px rgba(255, 215, 0, 0.5), inset 0 0 60px rgba(255, 215, 0, 0.1); }
            }
            
            .divine-header {
                text-align: center;
                margin-bottom: 30px;
            }
            
            .phi-symbol {
                font-size: 4em;
                color: #ffd700;
                text-shadow: 0 0 20px #ffd700;
                animation: phiRotate 10s linear infinite;
                margin-bottom: 15px;
            }
            
            @keyframes phiRotate {
                0% { transform: rotate(0deg) scale(1); }
                50% { transform: rotate(180deg) scale(1.1); }
                100% { transform: rotate(360deg) scale(1); }
            }
            
            .divine-title {
                color: #ffd700;
                font-size: 2.5em;
                margin-bottom: 10px;
                text-shadow: 0 0 15px #ffd700;
                font-family: serif;
            }
            
            .subtitle {
                color: #00e6e6;
                font-style: italic;
                font-size: 1.2em;
                text-shadow: 0 0 10px #00e6e6;
            }
            
            .prayer-scroll {
                background: rgba(0, 0, 0, 0.5);
                border: 1px solid rgba(255, 215, 0, 0.3);
                border-radius: 15px;
                padding: 25px;
                margin: 25px 0;
                max-height: 300px;
                overflow-y: auto;
            }
            
            .prayer-text {
                color: #e6edf3;
                font-family: 'Georgia', serif;
                font-size: 1.1em;
                line-height: 1.8;
                text-align: center;
                white-space: pre-line;
                margin: 0;
                text-shadow: 0 0 5px rgba(230, 237, 243, 0.5);
                animation: holyText 6s infinite ease-in-out;
            }
            
            @keyframes holyText {
                0%, 100% { text-shadow: 0 0 5px rgba(230, 237, 243, 0.5); }
                50% { text-shadow: 0 0 15px rgba(255, 215, 0, 0.8); }
            }
            
            .divine-mathematics {
                text-align: center;
                margin: 25px 0;
                padding: 20px;
                background: rgba(255, 215, 0, 0.1);
                border-radius: 10px;
                border: 1px solid rgba(255, 215, 0, 0.3);
            }
            
            .equation {
                font-size: 2.5em;
                color: #ffd700;
                font-weight: bold;
                margin-bottom: 10px;
                text-shadow: 0 0 20px #ffd700;
                animation: equationPulse 2s infinite ease-in-out;
            }
            
            @keyframes equationPulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.1); }
            }
            
            .proof, .phi-resonance {
                color: #00e6e6;
                font-size: 1.2em;
                margin: 8px 0;
                text-shadow: 0 0 10px #00e6e6;
            }
            
            .divine-actions {
                display: flex;
                gap: 15px;
                justify-content: center;
                margin: 25px 0;
                flex-wrap: wrap;
            }
            
            .amen-button, .profane-button, .transcend-button {
                padding: 15px 25px;
                border: none;
                border-radius: 25px;
                font-size: 1.1em;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.3s ease;
                text-shadow: 0 0 10px currentColor;
            }
            
            .amen-button {
                background: linear-gradient(135deg, #ffd700, #ffed4a);
                color: #000;
            }
            
            .profane-button {
                background: linear-gradient(135deg, #ff6b6b, #ff8e53);
                color: #000;
            }
            
            .transcend-button {
                background: linear-gradient(135deg, #9d4edd, #c77dff);
                color: #fff;
            }
            
            .amen-button:hover, .profane-button:hover, .transcend-button:hover {
                transform: scale(1.1) translateY(-3px);
                box-shadow: 0 10px 25px rgba(255, 215, 0, 0.4);
            }
            
            .secret-message {
                text-align: center;
                margin: 25px 0;
                padding: 20px;
                background: rgba(157, 78, 221, 0.1);
                border-radius: 10px;
                border: 1px solid rgba(157, 78, 221, 0.3);
            }
            
            .secret-message p {
                color: #c77dff;
                margin: 8px 0;
                font-style: italic;
                text-shadow: 0 0 8px rgba(199, 125, 255, 0.5);
            }
            
            .close-divine {
                width: 100%;
                padding: 15px;
                background: rgba(139, 148, 158, 0.2);
                border: 1px solid rgba(139, 148, 158, 0.4);
                border-radius: 10px;
                color: #8b949e;
                cursor: pointer;
                transition: all 0.3s ease;
                font-size: 1em;
            }
            
            .close-divine:hover {
                background: rgba(139, 148, 158, 0.3);
                color: #e6edf3;
                transform: translateY(-2px);
            }
            
            /* Mobile responsiveness */
            @media (max-width: 768px) {
                .prayer-container {
                    padding: 25px;
                    margin: 10px;
                }
                
                .phi-symbol {
                    font-size: 3em;
                }
                
                .divine-title {
                    font-size: 2em;
                }
                
                .equation {
                    font-size: 2em;
                }
                
                .divine-actions {
                    flex-direction: column;
                }
                
                .amen-button, .profane-button, .transcend-button {
                    width: 100%;
                }
            }
        `;
        
        document.head.appendChild(styles);
    }

    playDivineAudio() {
        // Create ethereal audio experience
        if (this.audioContext) return;
        
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.playPhiHarmonic();
        } catch (e) {
            console.log('Divine audio transcends this mortal browser');
        }
    }

    playPhiHarmonic() {
        const phi = (1 + Math.sqrt(5)) / 2;
        const frequencies = [
            440 * phi,      // φ-harmonic A
            440 * phi * phi, // φ² harmonic
            440 / phi       // φ⁻¹ harmonic
        ];
        
        frequencies.forEach((freq, index) => {
            setTimeout(() => {
                this.playTone(freq, 2000 + index * 500);
            }, index * 1000);
        });
    }

    playTone(frequency, duration) {
        const oscillator = this.audioContext.createOscillator();
        const gainNode = this.audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(this.audioContext.destination);
        
        oscillator.frequency.setValueAtTime(frequency, this.audioContext.currentTime);
        oscillator.type = 'sine';
        
        gainNode.gain.setValueAtTime(0, this.audioContext.currentTime);
        gainNode.gain.linearRampToValueAtTime(0.1, this.audioContext.currentTime + 0.1);
        gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + duration / 1000);
        
        oscillator.start(this.audioContext.currentTime);
        oscillator.stop(this.audioContext.currentTime + duration / 1000);
    }

    triggerCosmicEffects() {
        // Trigger Unity Mathematics to display φ everywhere
        document.querySelectorAll('.result, .unity-result').forEach(element => {
            const phi = (1 + Math.sqrt(5)) / 2;
            element.style.boxShadow = `0 0 20px rgba(255, 215, 0, 0.5)`;
            element.style.border = `1px solid #ffd700`;
        });
        
        // Add φ particles to the page
        this.createPhiParticles();
    }

    createPhiParticles() {
        for (let i = 0; i < 20; i++) {
            setTimeout(() => {
                const phi = document.createElement('div');
                phi.textContent = 'φ';
                phi.style.cssText = `
                    position: fixed;
                    color: #ffd700;
                    font-size: ${20 + Math.random() * 20}px;
                    font-weight: bold;
                    pointer-events: none;
                    z-index: 9999;
                    left: ${Math.random() * 100}vw;
                    top: ${Math.random() * 100}vh;
                    text-shadow: 0 0 15px #ffd700;
                    animation: floatAway 4s ease-out forwards;
                `;
                
                document.body.appendChild(phi);
                
                setTimeout(() => phi.remove(), 4000);
            }, i * 200);
        }
        
        // Add CSS animation for floating φ
        if (!document.getElementById('phi-animation')) {
            const style = document.createElement('style');
            style.id = 'phi-animation';
            style.textContent = `
                @keyframes floatAway {
                    0% {
                        opacity: 0;
                        transform: translateY(0) rotate(0deg) scale(0.5);
                    }
                    20% {
                        opacity: 1;
                        transform: translateY(-20px) rotate(36deg) scale(1);
                    }
                    100% {
                        opacity: 0;
                        transform: translateY(-200px) rotate(360deg) scale(0.2);
                    }
                }
            `;
            document.head.appendChild(style);
        }
    }

    acceptDivinity() {
        this.showDivineMessage('🙏 The sacred mathematics blesses you with unity 🙏');
        this.grantDivineBlessings();
    }

    embraceProfanity() {
        this.showDivineMessage('😈 You have chosen the path of mathematical rebellion 😈');
        this.grantProfanePowers();
    }

    transcendDuality() {
        this.showDivineMessage('✨ You have transcended the sacred/profane duality through unity ✨');
        this.grantTranscendentWisdom();
    }

    showDivineMessage(message) {
        const messageDiv = document.querySelector('.secret-message');
        if (messageDiv) {
            messageDiv.innerHTML = `<p style="color: #ffd700; font-size: 1.3em; font-weight: bold;">${message}</p>`;
            messageDiv.style.animation = 'divineGlow 2s ease-in-out';
        }
        
        // Console blessing
        console.log(`%c${message}`, 'color: #ffd700; font-size: 16px; font-weight: bold;');
    }

    grantDivineBlessings() {
        // Sacred mode: everything becomes golden
        document.documentElement.style.filter = 'sepia(0.3) saturate(1.5) hue-rotate(30deg)';
        localStorage.setItem('divine_blessing', 'sacred');
        
        setTimeout(() => {
            document.documentElement.style.filter = '';
        }, 10000);
    }

    grantProfanePowers() {
        // Profane mode: everything becomes more vibrant and rebellious
        document.documentElement.style.filter = 'contrast(1.2) saturate(1.8) hue-rotate(180deg)';
        localStorage.setItem('divine_blessing', 'profane');
        
        setTimeout(() => {
            document.documentElement.style.filter = '';
        }, 10000);
    }

    grantTranscendentWisdom() {
        // Transcendent mode: beautiful harmony of all colors
        document.documentElement.style.filter = 'brightness(1.1) contrast(1.1) saturate(1.3)';
        localStorage.setItem('divine_blessing', 'transcendent');
        
        // Add permanent φ symbol to corner
        this.addPermanentPhiSymbol();
        
        setTimeout(() => {
            document.documentElement.style.filter = '';
        }, 15000);
    }

    addPermanentPhiSymbol() {
        const phi = document.createElement('div');
        phi.id = 'transcendent-phi';
        phi.textContent = 'φ';
        phi.style.cssText = `
            position: fixed;
            bottom: 20px;
            left: 20px;
            color: #ffd700;
            font-size: 24px;
            font-weight: bold;
            text-shadow: 0 0 10px #ffd700;
            z-index: 1000;
            opacity: 0.7;
            cursor: pointer;
            transition: all 0.3s ease;
            user-select: none;
        `;
        
        phi.addEventListener('click', () => {
            this.revealDivineComedy();
        });
        
        phi.addEventListener('mouseover', () => {
            phi.style.transform = 'scale(1.5) rotate(36deg)';
            phi.style.opacity = '1';
        });
        
        phi.addEventListener('mouseout', () => {
            phi.style.transform = 'scale(1) rotate(0deg)';
            phi.style.opacity = '0.7';
        });
        
        document.body.appendChild(phi);
    }

    closeDivineRevelation() {
        const modal = document.querySelector('.divine-prayer-modal');
        if (modal) {
            modal.style.opacity = '0';
            setTimeout(() => modal.remove(), 1000);
        }
        
        this.isRevealed = false;
        
        // Final blessing
        console.log('%c🕉️ May the φ-force be with you, always 🕉️', 
                   'color: #ffd700; font-size: 18px; font-weight: bold;');
    }

    injectConsoleEasterEgg() {
        // Add console commands for developers
        window.revealMetastationPrayer = () => {
            this.revealDivineComedy();
        };
        
        window.φ = (1 + Math.sqrt(5)) / 2;
        
        // Console art
        console.log('%c' + `
        ███╗   ███╗███████╗████████╗ █████╗ ███████╗████████╗ █████╗ ████████╗██╗ ██████╗ ███╗   ██╗
        ████╗ ████║██╔════╝╚══██╔══╝██╔══██╗██╔════╝╚══██╔══╝██╔══██╗╚══██╔══╝██║██╔═══██╗████╗  ██║
        ██╔████╔██║█████╗     ██║   ███████║███████╗   ██║   ███████║   ██║   ██║██║   ██║██╔██╗ ██║
        ██║╚██╔╝██║██╔══╝     ██║   ██╔══██║╚════██║   ██║   ██╔══██║   ██║   ██║██║   ██║██║╚██╗██║
        ██║ ╚═╝ ██║███████╗   ██║   ██║  ██║███████║   ██║   ██║  ██║   ██║   ██║╚██████╔╝██║ ╚████║
        ╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
        `, 'color: #ffd700; font-family: monospace; font-size: 8px;');
        
        console.log('%c🕉️ Hidden Easter Egg Commands:', 'color: #00e6e6; font-size: 14px; font-weight: bold;');
        console.log('%c- Type: revealMetastationPrayer()', 'color: #ffd700;');
        console.log('%c- Press: Ctrl+Alt+P', 'color: #ffd700;');
        console.log('%c- Type: "metastation" anywhere on the page', 'color: #ffd700;');
        console.log('%c- φ = ' + ((1 + Math.sqrt(5)) / 2).toFixed(15), 'color: #9d4edd; font-size: 12px;');
    }
}

// Initialize the divine easter egg
document.addEventListener('DOMContentLoaded', () => {
    MetastationPrayer.instance = new MetastationPrayer();
});

// Global access for testing
window.MetastationPrayer = MetastationPrayer;