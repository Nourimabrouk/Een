/**
 * Persistent Audio System for Een Unity Mathematics
 * Maintains audio playback across page navigation using sessionStorage
 * Version: 1.0.0
 */

class PersistentAudioSystem {
    constructor() {
        this.audio = null;
        this.isPlaying = false;
        this.currentTrack = null;
        this.currentTime = 0;
        this.volume = 0.3;
        this.isVisible = false; // start minimized (non-intrusive)

        // Available tracks (including existing MP3s)
        this.tracks = [
            {
                id: 'one-u2',
                name: 'One - U2',
                url: 'audio/U2 - One.webm',
                duration: 276,
                artist: 'U2',
                album: 'Achtung Baby'
            },
            {
                id: 'always-bon-jovi',
                name: 'Always - Bon Jovi',
                url: 'audio/Bon Jovi - Always.webm',
                duration: 351,
                artist: 'Bon Jovi',
                album: 'Cross Road'
            },
            {
                id: 'i-want-to-know-what-love-is',
                name: 'I Want to Know What Love Is - Foreigner',
                url: 'audio/Foreigner - I Want to Know What Love Is.webm',
                duration: 297,
                artist: 'Foreigner',
                album: 'Agent Provocateur'
            },
            {
                id: 'unity-thefatrat',
                name: 'Unity - TheFatRat',
                url: 'audio/TheFatRat - Unity.mp3',
                duration: 270,
                artist: 'TheFatRat',
                album: 'NCS',
                isDefault: true
            },
            {
                id: 'one-love-bob-marley',
                name: 'One Love - Bob Marley',
                url: 'audio/Bob Marley - One Love.mp3',
                duration: 210,
                artist: 'Bob Marley',
                album: 'Legend'
            },
            {
                id: 'consciousness-flow',
                name: 'Consciousness Flow',
                url: 'audio/consciousness-flow.mp3',
                duration: 240,
                artist: 'Unity Mathematics',
                album: 'φ-Harmonic Series',
                isGenerated: true
            },
            {
                id: 'phi-harmonic',
                name: 'φ-Harmonic Resonance',
                url: 'audio/phi-harmonic.mp3',
                duration: 300,
                artist: 'Unity Mathematics',
                album: 'φ-Harmonic Series',
                isGenerated: true
            },
            {
                id: 'unity-meditation',
                name: 'Unity Meditation',
                url: 'audio/unity-meditation.mp3',
                duration: 180,
                artist: 'Unity Mathematics',
                album: 'Consciousness Collection',
                isGenerated: true
            }
        ];

        this.init();
    }

    init() {
        this.loadState();
        this.createAudioInterface();
        this.attachEventListeners();
        this.resumePlayback();

        // Save state before page unload
        window.addEventListener('beforeunload', () => this.saveState());

        // Save state periodically
        setInterval(() => this.saveState(), 5000);

        console.log('🎵 Persistent Audio System initialized');
    }

    createAudioInterface() {
        // Remove existing audio interface
        const existing = document.getElementById('persistent-audio-system');
        if (existing) existing.remove();

        // Create floating audio panel
        const audioPanel = document.createElement('div');
        audioPanel.id = 'persistent-audio-system';
        audioPanel.className = 'persistent-audio-panel';
        audioPanel.innerHTML = `
            <div class="audio-header">
                <div class="audio-title">
                    <i class="fas fa-music audio-icon"></i>
                    <span class="track-name">Unity Mathematics</span>
                </div>
                <button class="audio-minimize-btn" title="Toggle Audio Panel">
                    <i class="fas fa-chevron-up"></i>
                </button>
            </div>
            <div class="audio-body">
                <div class="audio-controls">
                    <button class="audio-btn prev-btn" title="Previous Track">
                        <i class="fas fa-step-backward"></i>
                    </button>
                    <button class="audio-btn play-btn" title="Play/Pause">
                        <i class="fas fa-play"></i>
                    </button>
                    <button class="audio-btn next-btn" title="Next Track">
                        <i class="fas fa-step-forward"></i>
                    </button>
                </div>
                <div class="audio-progress">
                    <div class="progress-track">
                        <div class="progress-fill"></div>
                        <div class="progress-handle"></div>
                    </div>
                    <div class="audio-time">
                        <span class="current-time">0:00</span>
                        <span class="total-time">0:00</span>
                    </div>
                </div>
                <div class="audio-volume">
                    <i class="fas fa-volume-up"></i>
                    <input type="range" class="volume-slider" min="0" max="100" value="30">
                </div>
                <div class="track-selector">
                    <select class="track-select">
                        ${this.tracks.map(track =>
            `<option value="${track.id}">${track.name}${track.artist ? ' - ' + track.artist : ''}</option>`
        ).join('')}
                    </select>
                </div>
            </div>
        `;

        document.body.appendChild(audioPanel);
        this.applyStyles();
    }

    applyStyles() {
        const styleId = 'persistent-audio-styles';
        if (document.getElementById(styleId)) return;

        const style = document.createElement('style');
        style.id = styleId;
        style.textContent = `
            .persistent-audio-panel {
                position: fixed;
                bottom: 20px;
                left: 20px;
                width: 320px;
                background: rgba(15, 15, 20, 0.98);
                backdrop-filter: blur(20px);
                border: 1px solid rgba(255, 215, 0, 0.3);
                border-radius: 16px;
                box-shadow: 0 15px 50px rgba(0, 0, 0, 0.4);
                z-index: 9999;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                color: #ffffff;
                overflow: hidden;
            }

            .persistent-audio-panel.minimized {
                height: 60px;
            }

            .persistent-audio-panel.minimized .audio-body {
                display: none;
            }

            .audio-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 1rem 1.25rem;
                background: rgba(255, 215, 0, 0.05);
                border-bottom: 1px solid rgba(255, 215, 0, 0.1);
            }

            .audio-title {
                display: flex;
                align-items: center;
                gap: 0.75rem;
                font-weight: 600;
                font-size: 0.9rem;
            }

            .audio-icon {
                color: #FFD700;
                font-size: 1.1rem;
                animation: audioIconPulse 2s ease-in-out infinite;
            }

            @keyframes audioIconPulse {
                0%, 100% { transform: scale(1); opacity: 1; }
                50% { transform: scale(1.1); opacity: 0.8; }
            }

            .track-name {
                color: rgba(255, 255, 255, 0.9);
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                max-width: 180px;
            }

            .audio-minimize-btn {
                background: transparent;
                border: none;
                color: rgba(255, 255, 255, 0.7);
                font-size: 1rem;
                cursor: pointer;
                padding: 0.5rem;
                border-radius: 8px;
                transition: all 0.3s ease;
            }

            .audio-minimize-btn:hover {
                background: rgba(255, 215, 0, 0.1);
                color: #FFD700;
                transform: translateY(-1px);
            }

            .audio-minimize-btn.minimized i {
                transform: rotate(180deg);
            }

            .audio-body {
                padding: 1rem 1.25rem;
            }

            .audio-controls {
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 1rem;
                margin-bottom: 1rem;
            }

            .audio-btn {
                background: rgba(255, 215, 0, 0.1);
                border: 1px solid rgba(255, 215, 0, 0.3);
                color: #FFD700;
                width: 40px;
                height: 40px;
                border-radius: 50%;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.3s ease;
                font-size: 0.9rem;
            }

            .audio-btn:hover {
                background: rgba(255, 215, 0, 0.2);
                border-color: #FFD700;
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(255, 215, 0, 0.2);
            }

            .audio-btn.active {
                background: #FFD700;
                color: #000;
            }

            .play-btn {
                width: 50px;
                height: 50px;
                font-size: 1.1rem;
            }

            .audio-progress {
                margin-bottom: 1rem;
            }

            .progress-track {
                position: relative;
                height: 6px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 3px;
                cursor: pointer;
                margin-bottom: 0.5rem;
            }

            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, #FFD700, #D4AF37);
                border-radius: 3px;
                width: 0%;
                transition: width 0.1s ease;
            }

            .progress-handle {
                position: absolute;
                top: 50%;
                transform: translateY(-50%);
                width: 14px;
                height: 14px;
                background: #FFD700;
                border-radius: 50%;
                cursor: pointer;
                left: 0%;
                transition: left 0.1s ease;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
            }

            .progress-handle:hover {
                transform: translateY(-50%) scale(1.2);
            }

            .audio-time {
                display: flex;
                justify-content: space-between;
                font-size: 0.8rem;
                color: rgba(255, 255, 255, 0.7);
            }

            .audio-volume {
                display: flex;
                align-items: center;
                gap: 0.75rem;
                margin-bottom: 1rem;
            }

            .audio-volume i {
                color: #FFD700;
                font-size: 1rem;
                width: 20px;
            }

            .volume-slider {
                flex: 1;
                -webkit-appearance: none;
                appearance: none;
                height: 4px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 2px;
                outline: none;
            }

            .volume-slider::-webkit-slider-thumb {
                -webkit-appearance: none;
                appearance: none;
                width: 16px;
                height: 16px;
                background: #FFD700;
                border-radius: 50%;
                cursor: pointer;
            }

            .volume-slider::-moz-range-thumb {
                width: 16px;
                height: 16px;
                background: #FFD700;
                border-radius: 50%;
                cursor: pointer;
                border: none;
            }

            .track-selector {
                margin-top: 1rem;
            }

            .track-select {
                width: 100%;
                padding: 0.75rem 1rem;
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 215, 0, 0.2);
                border-radius: 8px;
                color: #ffffff;
                font-family: inherit;
                font-size: 0.9rem;
                cursor: pointer;
                outline: none;
                transition: all 0.3s ease;
            }

            .track-select:hover,
            .track-select:focus {
                border-color: #FFD700;
                background: rgba(255, 215, 0, 0.05);
            }

            .track-select option {
                background: rgba(15, 15, 20, 0.98);
                color: #ffffff;
                padding: 0.5rem;
            }

            /* Mobile responsive */
            @media (max-width: 768px) {
                .persistent-audio-panel {
                    left: 10px;
                    right: 10px;
                    bottom: 10px;
                    width: auto;
                }

                .audio-header {
                    padding: 0.875rem 1rem;
                }

                .audio-body {
                    padding: 0.875rem 1rem;
                }
            }

            /* Hide on very small screens when minimized */
            @media (max-width: 480px) {
                .persistent-audio-panel.minimized {
                    width: 200px;
                    right: auto;
                }
            }

            /* Ensure panel doesn't interfere with chat */
            .persistent-audio-panel {
                z-index: 9999;
            }

            .enhanced-chat-container {
                z-index: 10000;
            }

            /* Playing animation */
            .persistent-audio-panel.playing .audio-icon {
                animation: audioIconPulse 1s ease-in-out infinite;
            }

            .persistent-audio-panel.playing .progress-fill {
                box-shadow: 0 0 10px rgba(255, 215, 0, 0.3);
            }
        `;

        document.head.appendChild(style);
    }

    attachEventListeners() {
        const panel = document.getElementById('persistent-audio-system');
        if (!panel) return;

        // Minimize/maximize toggle
        const minimizeBtn = panel.querySelector('.audio-minimize-btn');
        minimizeBtn.addEventListener('click', () => this.toggleMinimize());

        // Play/pause
        const playBtn = panel.querySelector('.play-btn');
        playBtn.addEventListener('click', () => this.togglePlayPause());

        // Previous/next track
        const prevBtn = panel.querySelector('.prev-btn');
        const nextBtn = panel.querySelector('.next-btn');
        prevBtn.addEventListener('click', () => this.previousTrack());
        nextBtn.addEventListener('click', () => this.nextTrack());

        // Volume control
        const volumeSlider = panel.querySelector('.volume-slider');
        volumeSlider.addEventListener('input', (e) => this.setVolume(e.target.value / 100));

        // Track selection
        const trackSelect = panel.querySelector('.track-select');
        trackSelect.addEventListener('change', (e) => this.loadTrack(e.target.value));

        // Progress bar
        const progressTrack = panel.querySelector('.progress-track');
        progressTrack.addEventListener('click', (e) => this.seekTo(e));

        // Global events
        window.addEventListener('meta-optimal-nav:audio', () => this.togglePlayPause());

        // Handle page visibility
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.saveState();
            }
        });
    }

    loadTrack(trackId) {
        const track = this.tracks.find(t => t.id === trackId);
        if (!track) return;

        if (this.audio) {
            this.audio.pause();
            this.audio = null;
        }

        this.currentTrack = trackId;
        this.audio = new Audio(track.url);
        this.audio.volume = this.volume;

        // Fallback to generated audio if file doesn't exist
        this.audio.addEventListener('error', () => {
            console.log(`Audio file ${track.url} not found, using generated tone`);
            this.generateAudioTone(track);
        });

        this.audio.addEventListener('loadeddata', () => {
            this.updateTrackInfo();
        });

        this.audio.addEventListener('play', () => {
            this.isPlaying = true;
            this.updatePlayButton();
            this.updatePanelState();
        });

        this.audio.addEventListener('pause', () => {
            this.isPlaying = false;
            this.updatePlayButton();
            this.updatePanelState();
        });

        this.audio.addEventListener('timeupdate', () => {
            this.updateProgress();
        });

        this.audio.addEventListener('ended', () => {
            this.isPlaying = false;
            this.updatePlayButton();
            this.updatePanelState();
            this.nextTrack();
        });

        this.updateTrackInfo();
    }

    generateAudioTone(track) {
        // Generate a mathematical consciousness tone using Web Audio API
        if (!this.audioContext) {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }

        // φ-harmonic frequencies based on golden ratio
        const phi = 1.618033988749895;
        const baseFreq = 220; // A3
        const frequencies = [
            baseFreq,
            baseFreq * phi,
            baseFreq * phi * phi,
            baseFreq / phi
        ];

        this.oscillators = frequencies.map((freq, index) => {
            const oscillator = this.audioContext.createOscillator();
            const gainNode = this.audioContext.createGain();

            oscillator.frequency.setValueAtTime(freq, this.audioContext.currentTime);
            oscillator.type = 'sine';

            // Create φ-harmonic volume modulation
            gainNode.gain.setValueAtTime(0.1 / (index + 1), this.audioContext.currentTime);

            oscillator.connect(gainNode);
            gainNode.connect(this.audioContext.destination);

            return { oscillator, gainNode };
        });

        // Start playing generated tone
        this.oscillators.forEach(({ oscillator }) => oscillator.start());
        this.generatedAudioStartTime = this.audioContext.currentTime;
        this.isGeneratedAudio = true;
    }

    togglePlayPause() {
        if (this.isGeneratedAudio) {
            if (this.isPlaying) {
                this.stopGeneratedAudio();
                this.isPlaying = false;
            } else {
                this.playGeneratedAudio();
                this.isPlaying = true;
            }
        } else if (this.audio) {
            if (this.isPlaying) {
                this.audio.pause();
                this.isPlaying = false;
            } else {
                this.audio.play().then(() => {
                    this.isPlaying = true;
                    this.updatePlayButton();
                    this.updatePanelState();
                }).catch(error => {
                    console.warn('Playback failed:', error);
                    this.isPlaying = false;
                });
            }
        } else {
            // Load default track if none loaded
            const defaultTrack = this.tracks.find(track => track.isDefault);
            const trackToLoad = defaultTrack ? defaultTrack.id : this.tracks[0].id;
            this.loadTrack(trackToLoad);
            setTimeout(() => this.togglePlayPause(), 100);
            return; // Don't update UI state yet
        }

        this.updatePlayButton();
        this.updatePanelState();
    }

    playGeneratedAudio() {
        if (this.oscillators) {
            this.oscillators.forEach(({ oscillator }) => oscillator.start());
            this.generatedAudioStartTime = this.audioContext.currentTime;
        }
    }

    stopGeneratedAudio() {
        if (this.oscillators) {
            this.oscillators.forEach(({ oscillator }) => {
                try {
                    oscillator.stop();
                } catch (e) {
                    // Oscillator already stopped
                }
            });
        }
    }

    previousTrack() {
        const currentIndex = this.tracks.findIndex(t => t.id === this.currentTrack);
        const prevIndex = currentIndex > 0 ? currentIndex - 1 : this.tracks.length - 1;
        this.loadTrack(this.tracks[prevIndex].id);

        if (this.isPlaying) {
            setTimeout(() => this.togglePlayPause(), 100);
        }
    }

    nextTrack() {
        const currentIndex = this.tracks.findIndex(t => t.id === this.currentTrack);
        const nextIndex = currentIndex < this.tracks.length - 1 ? currentIndex + 1 : 0;
        this.loadTrack(this.tracks[nextIndex].id);

        if (this.isPlaying) {
            setTimeout(() => this.togglePlayPause(), 100);
        }
    }

    setVolume(volume) {
        this.volume = volume;
        if (this.audio) {
            this.audio.volume = volume;
        }

        if (this.oscillators) {
            this.oscillators.forEach(({ gainNode }, index) => {
                gainNode.gain.setValueAtTime((volume * 0.1) / (index + 1), this.audioContext.currentTime);
            });
        }
    }

    seekTo(e) {
        if (!this.audio || this.isGeneratedAudio) return;

        const rect = e.target.getBoundingClientRect();
        const percent = (e.clientX - rect.left) / rect.width;
        const seekTime = this.audio.duration * percent;

        this.audio.currentTime = seekTime;
    }

    updateProgress() {
        if (!this.audio) return;

        const currentTime = this.audio.currentTime;
        const duration = this.audio.duration || 0;
        const percent = duration > 0 ? (currentTime / duration) * 100 : 0;

        const panel = document.getElementById('persistent-audio-system');
        const progressFill = panel?.querySelector('.progress-fill');
        const progressHandle = panel?.querySelector('.progress-handle');
        const currentTimeEl = panel?.querySelector('.current-time');

        if (progressFill) progressFill.style.width = `${percent}%`;
        if (progressHandle) progressHandle.style.left = `${percent}%`;
        if (currentTimeEl) currentTimeEl.textContent = this.formatTime(currentTime);
    }

    updateTrackInfo() {
        const panel = document.getElementById('persistent-audio-system');
        const trackNameEl = panel?.querySelector('.track-name');
        const totalTimeEl = panel?.querySelector('.total-time');
        const trackSelect = panel?.querySelector('.track-select');

        const track = this.tracks.find(t => t.id === this.currentTrack);
        if (track && trackNameEl) {
            trackNameEl.textContent = track.name;
        }

        if (this.audio && totalTimeEl) {
            const duration = this.audio.duration || track?.duration || 0;
            totalTimeEl.textContent = this.formatTime(duration);
        }

        if (trackSelect) {
            trackSelect.value = this.currentTrack || this.tracks[0].id;
        }
    }

    updatePlayButton() {
        const panel = document.getElementById('persistent-audio-system');
        const playBtn = panel?.querySelector('.play-btn i');

        if (playBtn) {
            playBtn.className = this.isPlaying ? 'fas fa-pause' : 'fas fa-play';
        }
    }

    updatePanelState() {
        const panel = document.getElementById('persistent-audio-system');
        if (panel) {
            panel.classList.toggle('playing', this.isPlaying);
        }
    }

    toggleMinimize() {
        const panel = document.getElementById('persistent-audio-system');
        const minimizeBtn = panel?.querySelector('.audio-minimize-btn');

        if (panel) {
            this.isVisible = !panel.classList.contains('minimized');
            panel.classList.toggle('minimized');

            if (minimizeBtn) {
                minimizeBtn.classList.toggle('minimized', !this.isVisible);
            }
        }
    }

    formatTime(seconds) {
        if (!seconds || !isFinite(seconds)) return '0:00';

        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    saveState() {
        try {
            const state = {
                currentTrack: this.currentTrack,
                isPlaying: this.isPlaying,
                currentTime: this.audio?.currentTime || 0,
                volume: this.volume,
                isVisible: this.isVisible,
                timestamp: Date.now()
            };

            sessionStorage.setItem('persistent-audio-state', JSON.stringify(state));
        } catch (e) {
            console.warn('Failed to save audio state:', e);
        }
    }

    loadState() {
        try {
            const saved = sessionStorage.getItem('persistent-audio-state');
            if (!saved) return;

            const state = JSON.parse(saved);

            // Only restore if saved less than 1 hour ago
            if (Date.now() - state.timestamp < 3600000) {
                this.currentTrack = state.currentTrack || this.tracks[0].id;
                this.isPlaying = state.isPlaying || false;
                this.currentTime = state.currentTime || 0;
                this.volume = state.volume || 0.7;
                this.isVisible = state.isVisible !== false;
            }
        } catch (e) {
            console.warn('Failed to load audio state:', e);
        }
    }

    resumePlayback() {
        if (!this.currentTrack) {
            // Default to 'One' by U2 (first track with isDefault: true)
            const defaultTrack = this.tracks.find(track => track.isDefault);
            this.currentTrack = defaultTrack ? defaultTrack.id : this.tracks[0].id;
        }

        this.loadTrack(this.currentTrack);

        // Restore minimized state
        setTimeout(() => {
            const panel = document.getElementById('persistent-audio-system');
            if (panel && !this.isVisible) {
                panel.classList.add('minimized');
            }

            // Resume playback if was playing
            if (this.isPlaying && this.currentTime > 0) {
                if (this.audio) {
                    this.audio.currentTime = this.currentTime;
                    this.audio.play().catch(console.warn);
                }
                this.updatePlayButton();
                this.updatePanelState();
            }
        }, 500);
    }

    // Public API
    play() {
        if (!this.isPlaying) {
            this.togglePlayPause();
        }
    }

    pause() {
        if (this.isPlaying) {
            this.togglePlayPause();
        }
    }

    show() {
        const panel = document.getElementById('persistent-audio-system');
        if (panel && !this.isVisible) {
            this.toggleMinimize();
        }
    }

    hide() {
        const panel = document.getElementById('persistent-audio-system');
        if (panel && this.isVisible) {
            this.toggleMinimize();
        }
    }

    destroy() {
        if (this.audio) {
            this.audio.pause();
            this.audio = null;
        }

        if (this.oscillators) {
            this.stopGeneratedAudio();
        }

        if (this.audioContext) {
            this.audioContext.close();
        }

        const panel = document.getElementById('persistent-audio-system');
        if (panel) panel.remove();

        const style = document.getElementById('persistent-audio-styles');
        if (style) style.remove();

        sessionStorage.removeItem('persistent-audio-state');

        console.log('🎵 Persistent Audio System destroyed');
    }
}

// Initialize persistent audio system
let persistentAudioSystem;

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        persistentAudioSystem = new PersistentAudioSystem();
        window.persistentAudioSystem = persistentAudioSystem;
    });
} else {
    persistentAudioSystem = new PersistentAudioSystem();
    window.persistentAudioSystem = persistentAudioSystem;
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PersistentAudioSystem;
}

console.log('🎵 Persistent Audio System loaded - Audio will continue across page navigation');