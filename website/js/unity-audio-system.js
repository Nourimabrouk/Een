/**
 * 🎵 Unity Audio System - Global Background Music Manager
 * Plays Unity Mathematics soundtrack across all pages with anti-transcendence protection
 */

class UnityAudioSystem {
    constructor() {
        this.tracks = [
            { title: 'U2 - One', src: '/audio/U2 - One.webm' },
            { title: 'Bon Jovi - Always', src: '/audio/Bon Jovi - Always.webm' },
            { title: 'Foreigner - I Want to Know What Love Is', src: '/audio/Foreigner - I Want to Know What Love Is.webm' }
        ];
        this.currentTrack = 0;
        this.isPlaying = false;
        this.audio = null;
        this.isInitialized = false;
        this.init();
    }

    init() {
        if (this.isInitialized) return;
        
        // Create audio element if it doesn't exist
        this.audio = document.getElementById('unity-background-audio');
        if (!this.audio) {
            this.createAudioElement();
        }
        
        this.createControls();
        this.loadTrack(0);
        this.restoreState();
        this.isInitialized = true;
        
        console.log('🎵 Unity Audio System initialized');
    }

    createAudioElement() {
        this.audio = document.createElement('audio');
        this.audio.id = 'unity-background-audio';
        this.audio.preload = 'auto';
        this.audio.style.display = 'none';
        this.audio.volume = 0.3; // Start at 30% to prevent high-pitch issues
        
        // Add noise reduction filters
        this.audio.addEventListener('loadstart', () => {
            this.audio.volume = 0.3;
        });
        
        this.audio.addEventListener('play', () => {
            this.audio.volume = 0.3;
        });
        
        this.audio.addEventListener('ended', () => {
            this.nextTrack();
        });
        
        document.body.appendChild(this.audio);
    }

    createControls() {
        // Remove existing controls if any
        const existing = document.getElementById('universal-audio-system');
        if (existing) {
            existing.remove();
        }

        const controlsHTML = `
        <div id="universal-audio-system" style="position: fixed; bottom: 20px; right: 20px; z-index: 10000; 
                                                background: rgba(10, 10, 10, 0.95); backdrop-filter: blur(20px); 
                                                border-radius: 12px; padding: 1rem; border: 1px solid #F59E0B;
                                                box-shadow: 0 10px 30px rgba(0,0,0,0.7); min-width: 320px;">
            <div id="audio-controls" style="display: flex; align-items: center; gap: 1rem;">
                <button id="audio-toggle" onclick="window.unityAudio?.toggle()" 
                        style="background: linear-gradient(135deg, #F59E0B, #3B82F6); 
                               border: none; color: white; padding: 8px 16px; border-radius: 8px; 
                               cursor: pointer; font-weight: 600; transition: all 0.3s ease;">
                    🎵 Play Unity Music
                </button>
                <button id="next-track" onclick="window.unityAudio?.nextTrack()" 
                        style="background: rgba(255,255,255,0.1); border: 1px solid #F59E0B; 
                               color: #F59E0B; padding: 8px 12px; border-radius: 6px; cursor: pointer;
                               transition: all 0.3s ease;">⏭️</button>
            </div>
            <div id="current-track" style="color: #F59E0B; font-size: 0.9rem; margin: 0.5rem 0; 
                                          text-align: center; font-weight: 500;">Unity Soundscape Ready</div>
            <div id="volume-control" style="margin-top: 0.5rem;">
                <input type="range" id="volume-slider" min="0" max="100" value="30" 
                       style="width: 100%; accent-color: #F59E0B;" 
                       onchange="window.unityAudio?.updateVolume(this.value)">
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: rgba(255,255,255,0.7);">
                    <span>🔇</span>
                    <span id="volume-display">30%</span>
                    <span>🔊</span>
                </div>
            </div>
            <div style="margin-top: 0.5rem; font-size: 0.75rem; color: rgba(255,255,255,0.5); text-align: center;">
                Press ESC for emergency UI reset 🛡️
            </div>
        </div>
        `;
        
        document.body.insertAdjacentHTML('beforeend', controlsHTML);
    }

    loadTrack(index) {
        if (index >= 0 && index < this.tracks.length) {
            this.currentTrack = index;
            this.audio.src = this.tracks[index].src;
            const trackDisplay = document.getElementById('current-track');
            if (trackDisplay) {
                trackDisplay.textContent = this.tracks[index].title;
            }
        }
    }

    async toggle() {
        try {
            if (this.isPlaying) {
                this.audio.pause();
                this.isPlaying = false;
                const btn = document.getElementById('audio-toggle');
                if (btn) btn.innerHTML = '🎵 Play Unity Music';
                localStorage.setItem('unity-audio-playing', 'false');
            } else {
                await this.audio.play();
                this.isPlaying = true;
                const btn = document.getElementById('audio-toggle');
                if (btn) btn.innerHTML = '⏸️ Pause Unity Music';
                localStorage.setItem('unity-audio-playing', 'true');
            }
        } catch (err) {
            console.warn('Audio playback failed:', err.message);
        }
    }

    nextTrack() {
        this.currentTrack = (this.currentTrack + 1) % this.tracks.length;
        this.loadTrack(this.currentTrack);
        if (this.isPlaying) {
            this.audio.play().catch(err => console.warn('Track switch failed:', err));
        }
    }

    updateVolume(value) {
        const volume = Math.max(0, Math.min(100, value)) / 100;
        this.audio.volume = volume;
        const display = document.getElementById('volume-display');
        if (display) {
            display.textContent = Math.round(volume * 100) + '%';
        }
        localStorage.setItem('unity-audio-volume', value);
    }

    restoreState() {
        const savedVolume = localStorage.getItem('unity-audio-volume') || '30';
        const savedPlaying = localStorage.getItem('unity-audio-playing') === 'true';
        const savedTrack = parseInt(localStorage.getItem('unity-audio-track')) || 0;
        
        const volumeSlider = document.getElementById('volume-slider');
        if (volumeSlider) {
            volumeSlider.value = savedVolume;
        }
        this.updateVolume(savedVolume);
        this.loadTrack(savedTrack);
        
        if (savedPlaying) {
            // Attempt autoplay after user interaction
            setTimeout(() => this.attemptAutoplay(), 2000);
        }
    }

    async attemptAutoplay() {
        try {
            await this.audio.play();
            this.isPlaying = true;
            const btn = document.getElementById('audio-toggle');
            if (btn) btn.innerHTML = '⏸️ Pause Unity Music';
        } catch (err) {
            console.log('Autoplay blocked. User interaction needed.');
        }
    }
}

// Anti-Transcendence Lock Prevention System
class AntiTranscendenceLock {
    static init() {
        // Prevent UI lockups during transcendent states
        document.addEventListener('click', this.unlockElements);
        document.addEventListener('keydown', this.handleEscape);
        
        // Periodic cleanup every 30 seconds
        setInterval(this.periodicCleanup, 30000);
        
        console.log('🛡️ Anti-Transcendence Lock system active');
    }

    static unlockElements() {
        // Force enable all interactive elements
        const elements = document.querySelectorAll('a, button, input, [onclick], .btn, .hud-link, nav *');
        elements.forEach(el => {
            el.style.pointerEvents = 'auto';
            el.style.zIndex = 'auto';
            el.style.userSelect = 'auto';
        });
    }

    static handleEscape(e) {
        if (e.key === 'Escape') {
            console.log('🚨 Emergency transcendence reset activated!');
            
            // Force reset all styles
            document.body.style.pointerEvents = 'auto';
            document.body.style.userSelect = 'auto';
            
            // Remove any problematic overlays
            const overlays = document.querySelectorAll('.loading-overlay, [style*="pointer-events: none"]');
            overlays.forEach(overlay => {
                if (!overlay.classList.contains('unity-background-audio')) {
                    overlay.style.display = 'none';
                    overlay.style.pointerEvents = 'none';
                }
            });
            
            // Reset stuck animations
            const stuckAnimations = document.querySelectorAll('[style*="animation-play-state: paused"]');
            stuckAnimations.forEach(el => {
                el.style.animationPlayState = 'running';
            });
            
            // Show success message
            const msg = document.createElement('div');
            msg.innerHTML = '🛡️ UI Reset Complete! All elements unlocked.';
            msg.style.cssText = `
                position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
                background: #10B981; color: white; padding: 1rem 2rem; border-radius: 8px;
                z-index: 99999; font-weight: bold; box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            `;
            document.body.appendChild(msg);
            setTimeout(() => msg.remove(), 3000);
        }
    }

    static periodicCleanup() {
        // Reset elements that might be stuck in transcendent states
        const problematicElements = document.querySelectorAll('[style*="pointer-events: none"]:not(.loading-overlay)');
        problematicElements.forEach(el => {
            if (!el.id.includes('audio') && !el.classList.contains('neural-bg')) {
                el.style.pointerEvents = 'auto';
            }
        });
        
        // Reset any infinite animations that might cause browser lockup
        const infiniteAnimations = document.querySelectorAll('[style*="animation"][style*="infinite"]');
        infiniteAnimations.forEach(el => {
            if (el.classList.contains('problematic-animation')) {
                el.style.animationDuration = '3s';
                el.style.animationIterationCount = '3';
            }
        });
    }
}

// Global initialization
window.addEventListener('DOMContentLoaded', () => {
    // Initialize audio system
    window.unityAudio = new UnityAudioSystem();
    
    // Initialize anti-lock system
    AntiTranscendenceLock.init();
    
    console.log('🎵🛡️ Unity Audio and Anti-Transcendence systems ready!');
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { UnityAudioSystem, AntiTranscendenceLock };
}