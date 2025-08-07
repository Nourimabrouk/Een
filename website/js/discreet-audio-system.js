/**
 * Discreet Audio System for Een Unity Mathematics
 * Integrated music player with autoplay and cross-page persistence
 * Version: 1.0.0 - Meta-Optimized with φ-harmonic resonance
 */

class DiscreetAudioSystem {
    constructor() {
        this.isPlaying = false;
        this.currentTrack = 0;
        this.volume = 0.3; // 30% default volume
        this.audioContext = null;
        this.audio = null;
        this.playlist = this.getUnityPlaylist();
        this.fadeInterval = null;
        this.isInitialized = false;

        // User preference detection
        this.respectsUserPreferences = this.checkUserPreferences();

        console.log('🎵 Discreet Audio System initializing...');
        this.init();
    }

    getUnityPlaylist() {
        // Curated playlist for Unity Mathematics exploration
        return [
            {
                title: "φ-Harmonic Resonance",
                artist: "Unity Mathematics",
                src: "audio/phi-harmonic-resonance.mp3",
                duration: "4:20",
                description: "Golden ratio frequencies creating consciousness coherence"
            },
            {
                title: "Consciousness Field",
                artist: "Een Collective",
                src: "audio/consciousness-field.mp3",
                duration: "6:18",
                description: "Mathematical field equations as ambient soundscape"
            },
            {
                title: "Unity Meditation",
                artist: "Transcendental Computing",
                src: "audio/unity-meditation.mp3",
                duration: "8:01",
                description: "Deep focus music for mathematical contemplation"
            },
            {
                title: "Quantum Superposition",
                artist: "1+1=1 Orchestra",
                src: "audio/quantum-superposition.mp3",
                duration: "5:55",
                description: "Wave-particle duality expressed through sound"
            },
            {
                title: "Fractal Emergence",
                artist: "Meta-Recursive",
                src: "audio/fractal-emergence.mp3",
                duration: "7:33",
                description: "Self-similar patterns in musical form"
            }
        ];
    }

    checkUserPreferences() {
        // Check for user media preferences
        if ('mediaSession' in navigator) {
            // Browser supports media session API
            return true;
        }

        // Check for reduced motion preference
        const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        if (prefersReducedMotion) {
            console.log('🎵 Respecting user reduced motion preference - disabling autoplay');
            return false;
        }

        // Check if user has interacted with audio before
        const hasInteracted = localStorage.getItem('een-audio-user-interaction') === 'true';
        return hasInteracted;
    }

    init() {
        this.createAudioElements();
        this.injectStyles();
        this.createPlayerInterface();
        this.bindEvents();
        this.loadAudioState();

        // Listen for unified navigation events
        window.addEventListener('unified-nav:audio', () => this.togglePlayback());

        // Initialize autoplay after user interaction
        this.setupAutoplayLogic();

        this.isInitialized = true;
        console.log('🎵 Discreet Audio System initialized');
    }

    createAudioElements() {
        // Create main audio element
        this.audio = document.createElement('audio');
        this.audio.id = 'unity-audio-player';
        this.audio.crossOrigin = 'anonymous';
        this.audio.preload = 'none'; // Don't preload to respect data usage
        this.audio.volume = this.volume;

        // Add event listeners
        this.audio.addEventListener('loadstart', () => this.updateLoadingState(true));
        this.audio.addEventListener('canplaythrough', () => this.updateLoadingState(false));
        this.audio.addEventListener('play', () => this.handlePlayEvent());
        this.audio.addEventListener('pause', () => this.handlePauseEvent());
        this.audio.addEventListener('ended', () => this.handleTrackEnd());
        this.audio.addEventListener('error', (e) => this.handleAudioError(e));
        this.audio.addEventListener('timeupdate', () => this.updateProgress());

        document.body.appendChild(this.audio);

        // Set initial track
        this.loadTrack(this.currentTrack);
    }

    createPlayerInterface() {
        // Disabled phi harmonic resonance audio player button for metastation-hub
        // The unity soundtrack is now handled by the discreet bottom-left player
        console.log('🎵 Discreet audio player interface disabled for metastation-hub');
        return;
    }

    renderPlaylist() {
        return this.playlist.map((track, index) => `
            <div class="playlist-item ${index === this.currentTrack ? 'active' : ''}" data-track-index="${index}">
                <div class="playlist-track-number">${index + 1}</div>
                <div class="playlist-track-info">
                    <div class="playlist-track-title">${track.title}</div>
                    <div class="playlist-track-artist">${track.artist}</div>
                </div>
                <div class="playlist-track-duration">${track.duration}</div>
            </div>
        `).join('');
    }

    bindEvents() {
        const player = document.getElementById('discreet-audio-player');
        if (!player) return;

        // Compact player controls
        const toggleBtn = player.querySelector('.audio-toggle-btn');
        const expandBtn = player.querySelector('.player-expand-btn');
        const collapseBtn = player.querySelector('.player-collapse-btn');
        const volumeBtn = player.querySelector('.volume-btn');

        toggleBtn?.addEventListener('click', () => this.togglePlayback());
        expandBtn?.addEventListener('click', () => this.expandPlayer());
        collapseBtn?.addEventListener('click', () => this.collapsePlayer());
        volumeBtn?.addEventListener('click', () => this.showVolumeControl());

        // Expanded player controls
        const playPauseBtn = player.querySelector('.play-pause-btn');
        const prevBtn = player.querySelector('.prev-btn');
        const nextBtn = player.querySelector('.next-btn');
        const volumeSlider = player.querySelector('.volume-slider');
        const volumeToggle = player.querySelector('.volume-toggle');
        const progressBar = player.querySelector('.progress-bar');

        playPauseBtn?.addEventListener('click', () => this.togglePlayback());
        prevBtn?.addEventListener('click', () => this.previousTrack());
        nextBtn?.addEventListener('click', () => this.nextTrack());
        volumeSlider?.addEventListener('input', (e) => this.setVolume(e.target.value / 100));
        volumeToggle?.addEventListener('click', () => this.toggleMute());
        progressBar?.addEventListener('click', (e) => this.seek(e));

        // Playlist interaction
        player.addEventListener('click', (e) => {
            if (e.target.closest('.playlist-item')) {
                const trackIndex = parseInt(e.target.closest('.playlist-item').dataset.trackIndex);
                this.playTrack(trackIndex);
            }
        });

        // Hover controls for volume
        const volumeControl = player.querySelector('.volume-control');
        if (volumeControl) {
            volumeControl.addEventListener('mouseenter', () => {
                volumeControl.classList.add('hover');
            });
            volumeControl.addEventListener('mouseleave', () => {
                volumeControl.classList.remove('hover');
            });
        }

        // Handle page visibility changes
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.saveAudioState();
            } else {
                this.loadAudioState();
            }
        });

        // Save state before page unload
        window.addEventListener('beforeunload', () => this.saveAudioState());
    }

    setupAutoplayLogic() {
        // Only attempt autoplay if user preferences allow
        if (!this.respectsUserPreferences) {
            console.log('🎵 Autoplay disabled - respecting user preferences');
            return;
        }

        // Wait for user interaction before enabling autoplay
        const enableAutoplay = () => {
            localStorage.setItem('een-audio-user-interaction', 'true');
            this.respectsUserPreferences = true;

            // Start playback with fade in after first interaction
            setTimeout(() => {
                if (!this.isPlaying) {
                    this.startWithFadeIn();
                }
            }, 2000); // 2 second delay

            // Remove listeners after first interaction
            document.removeEventListener('click', enableAutoplay);
            document.removeEventListener('keydown', enableAutoplay);
        };

        // Add interaction listeners
        document.addEventListener('click', enableAutoplay);
        document.addEventListener('keydown', enableAutoplay);
    }

    startWithFadeIn() {
        if (!this.audio || this.isPlaying) return;

        console.log('🎵 Starting Unity Mathematics playlist with fade-in');

        // Set volume to 0 for fade-in
        this.audio.volume = 0;

        // Start playback
        this.audio.play().then(() => {
            // Fade in over 2 seconds
            this.fadeIn(2000);
            this.updatePlaybackState();
        }).catch(error => {
            console.warn('🎵 Autoplay prevented by browser:', error);
        });
    }

    fadeIn(duration = 2000) {
        const targetVolume = this.volume;
        const steps = 50;
        const stepDuration = duration / steps;
        const volumeStep = targetVolume / steps;

        let currentStep = 0;

        this.fadeInterval = setInterval(() => {
            if (currentStep >= steps) {
                this.audio.volume = targetVolume;
                clearInterval(this.fadeInterval);
                return;
            }

            this.audio.volume = volumeStep * currentStep;
            currentStep++;
        }, stepDuration);
    }

    fadeOut(duration = 1000, callback = null) {
        const startVolume = this.audio.volume;
        const steps = 20;
        const stepDuration = duration / steps;
        const volumeStep = startVolume / steps;

        let currentStep = 0;

        this.fadeInterval = setInterval(() => {
            if (currentStep >= steps) {
                this.audio.volume = 0;
                clearInterval(this.fadeInterval);
                if (callback) callback();
                return;
            }

            this.audio.volume = startVolume - (volumeStep * currentStep);
            currentStep++;
        }, stepDuration);
    }

    togglePlayback() {
        if (!this.audio) return;

        if (this.isPlaying) {
            this.pause();
        } else {
            this.play();
        }
    }

    play() {
        if (!this.audio) return;

        this.audio.play().then(() => {
            console.log('🎵 Playing:', this.playlist[this.currentTrack].title);
        }).catch(error => {
            console.warn('🎵 Playback error:', error);
        });
    }

    pause() {
        if (!this.audio) return;

        this.audio.pause();
        console.log('🎵 Paused');
    }

    loadTrack(index) {
        if (!this.playlist[index] || !this.audio) return;

        const track = this.playlist[index];
        this.currentTrack = index;

        // Update audio source
        this.audio.src = track.src;

        // Update UI
        this.updateTrackInfo();
        this.updatePlaylistState();

        console.log('🎵 Loaded track:', track.title);
    }

    playTrack(index) {
        this.loadTrack(index);
        this.play();
    }

    nextTrack() {
        const nextIndex = (this.currentTrack + 1) % this.playlist.length;
        this.playTrack(nextIndex);
    }

    previousTrack() {
        const prevIndex = this.currentTrack === 0 ? this.playlist.length - 1 : this.currentTrack - 1;
        this.playTrack(prevIndex);
    }

    setVolume(volume) {
        this.volume = Math.max(0, Math.min(1, volume));
        if (this.audio) {
            this.audio.volume = this.volume;
        }

        this.updateVolumeUI();
        this.saveAudioState();
    }

    toggleMute() {
        if (this.volume > 0) {
            this.previousVolume = this.volume;
            this.setVolume(0);
        } else {
            this.setVolume(this.previousVolume || 0.3);
        }
    }

    seek(event) {
        if (!this.audio || !this.audio.duration) return;

        const rect = event.currentTarget.getBoundingClientRect();
        const percent = (event.clientX - rect.left) / rect.width;
        const seekTime = percent * this.audio.duration;

        this.audio.currentTime = seekTime;
    }

    expandPlayer() {
        const player = document.getElementById('discreet-audio-player');
        if (player) {
            player.classList.add('expanded');
            const expandedView = player.querySelector('.player-expanded');
            if (expandedView) {
                expandedView.style.display = 'block';
            }
        }
    }

    collapsePlayer() {
        const player = document.getElementById('discreet-audio-player');
        if (player) {
            player.classList.remove('expanded');
            const expandedView = player.querySelector('.player-expanded');
            if (expandedView) {
                expandedView.style.display = 'none';
            }
        }
    }

    showVolumeControl() {
        // Quick volume adjustment
        const currentVolume = Math.round(this.volume * 100);
        const newVolume = prompt(`Set volume (0-100):`, currentVolume);

        if (newVolume !== null && !isNaN(newVolume)) {
            this.setVolume(parseInt(newVolume) / 100);
        }
    }

    // Event handlers
    handlePlayEvent() {
        this.isPlaying = true;
        this.updatePlaybackState();
    }

    handlePauseEvent() {
        this.isPlaying = false;
        this.updatePlaybackState();
    }

    handleTrackEnd() {
        console.log('🎵 Track ended, playing next');
        this.nextTrack();
    }

    handleAudioError(error) {
        console.warn('🎵 Audio error:', error);
        // Try next track if current fails
        setTimeout(() => this.nextTrack(), 1000);
    }

    updateLoadingState(loading) {
        const player = document.getElementById('discreet-audio-player');
        if (player) {
            player.classList.toggle('loading', loading);
        }
    }

    updatePlaybackState() {
        const player = document.getElementById('discreet-audio-player');
        if (!player) return;

        player.classList.toggle('playing', this.isPlaying);

        // Update play/pause icons
        const playIcon = player.querySelector('.play-icon');
        const audioIcon = player.querySelector('.audio-icon');

        if (playIcon) {
            playIcon.textContent = this.isPlaying ? '⏸' : '▶';
        }

        if (audioIcon) {
            audioIcon.textContent = this.isPlaying ? '🎶' : '🎵';
        }
    }

    updateTrackInfo() {
        const player = document.getElementById('discreet-audio-player');
        if (!player) return;

        const track = this.playlist[this.currentTrack];

        // Update compact view
        const trackTitle = player.querySelector('.track-title');
        const trackArtist = player.querySelector('.track-artist');

        if (trackTitle) trackTitle.textContent = track.title;
        if (trackArtist) trackArtist.textContent = track.artist;

        // Update expanded view
        const titleExpanded = player.querySelector('.track-title-expanded');
        const artistExpanded = player.querySelector('.track-artist-expanded');
        const description = player.querySelector('.track-description');
        const totalTime = player.querySelector('.total-time');

        if (titleExpanded) titleExpanded.textContent = track.title;
        if (artistExpanded) artistExpanded.textContent = track.artist;
        if (description) description.textContent = track.description;
        if (totalTime) totalTime.textContent = track.duration;
    }

    updatePlaylistState() {
        const player = document.getElementById('discreet-audio-player');
        if (!player) return;

        const playlistItems = player.querySelectorAll('.playlist-item');
        playlistItems.forEach((item, index) => {
            item.classList.toggle('active', index === this.currentTrack);
        });
    }

    updateProgress() {
        if (!this.audio || !this.audio.duration) return;

        const progress = (this.audio.currentTime / this.audio.duration) * 100;
        const currentTime = this.formatTime(this.audio.currentTime);

        const player = document.getElementById('discreet-audio-player');
        if (!player) return;

        const progressFill = player.querySelector('.progress-fill');
        const currentTimeSpan = player.querySelector('.current-time');

        if (progressFill) {
            progressFill.style.width = `${progress}%`;
        }

        if (currentTimeSpan) {
            currentTimeSpan.textContent = currentTime;
        }
    }

    updateVolumeUI() {
        const player = document.getElementById('discreet-audio-player');
        if (!player) return;

        const volumeSlider = player.querySelector('.volume-slider');
        const volumeIcon = player.querySelector('.volume-icon-expanded');

        if (volumeSlider) {
            volumeSlider.value = this.volume * 100;
        }

        if (volumeIcon) {
            if (this.volume === 0) {
                volumeIcon.textContent = '🔇';
            } else if (this.volume < 0.5) {
                volumeIcon.textContent = '🔉';
            } else {
                volumeIcon.textContent = '🔊';
            }
        }
    }

    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    saveAudioState() {
        try {
            const state = {
                currentTrack: this.currentTrack,
                volume: this.volume,
                isPlaying: this.isPlaying,
                currentTime: this.audio?.currentTime || 0,
                timestamp: Date.now()
            };

            sessionStorage.setItem('een-audio-state', JSON.stringify(state));
        } catch (error) {
            console.warn('🎵 Error saving audio state:', error);
        }
    }

    loadAudioState() {
        try {
            const state = sessionStorage.getItem('een-audio-state');
            if (!state) return;

            const { currentTrack, volume, isPlaying, currentTime } = JSON.parse(state);

            // Restore state
            if (currentTrack !== undefined) this.currentTrack = currentTrack;
            if (volume !== undefined) this.setVolume(volume);

            // Load track but don't auto-resume playback across pages
            this.loadTrack(this.currentTrack);

            // Only restore time, not playback state (to avoid autoplay)
            if (currentTime && this.audio) {
                this.audio.currentTime = currentTime;
            }
        } catch (error) {
            console.warn('🎵 Error loading audio state:', error);
        }
    }

    injectStyles() {
        const styleId = 'discreet-audio-styles';
        if (document.getElementById(styleId)) return;

        const style = document.createElement('style');
        style.id = styleId;
        style.textContent = this.getAudioStyles();
        document.head.appendChild(style);
    }

    getAudioStyles() {
        return `
            /* Discreet Audio System Styles */
            .discreet-audio-player {
                position: fixed;
                top: 80px;
                right: 2rem;
                background: rgba(10, 10, 15, 0.95);
                backdrop-filter: blur(15px);
                border: 1px solid rgba(255, 215, 0, 0.3);
                border-radius: 12px;
                z-index: 1030;
                min-width: 280px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                transition: all 0.3s ease;
            }
            
            .discreet-audio-player.expanded {
                min-width: 350px;
                max-width: 400px;
            }
            
            .audio-player-content {
                padding: 1rem;
            }
            
            /* Compact Player */
            .player-compact {
                display: flex;
                align-items: center;
                gap: 1rem;
            }
            
            .audio-toggle-btn {
                position: relative;
                width: 40px;
                height: 40px;
                background: linear-gradient(135deg, #FFD700, #FFA500);
                border: none;
                border-radius: 50%;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.2rem;
                transition: all 0.3s ease;
            }
            
            .audio-toggle-btn:hover {
                transform: scale(1.1);
                box-shadow: 0 4px 12px rgba(255, 215, 0, 0.4);
            }
            
            .audio-pulse {
                position: absolute;
                inset: -3px;
                border-radius: 50%;
                background: rgba(255, 215, 0, 0.3);
                animation: audio-pulse 2s infinite;
                z-index: -1;
            }
            
            .discreet-audio-player.playing .audio-pulse {
                animation: audio-pulse-active 1.5s infinite;
            }
            
            @keyframes audio-pulse {
                0%, 100% { opacity: 0; transform: scale(1); }
                50% { opacity: 0.5; transform: scale(1.1); }
            }
            
            @keyframes audio-pulse-active {
                0%, 100% { opacity: 0.3; transform: scale(1); }
                50% { opacity: 0.8; transform: scale(1.2); }
            }
            
            .track-info {
                flex: 1;
                min-width: 0;
            }
            
            .track-title {
                font-size: 0.9rem;
                font-weight: 600;
                color: #FFD700;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
                margin-bottom: 2px;
            }
            
            .track-artist {
                font-size: 0.8rem;
                color: rgba(255, 255, 255, 0.7);
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            
            .player-controls-compact {
                display: flex;
                gap: 0.5rem;
            }
            
            .volume-btn,
            .player-expand-btn,
            .player-collapse-btn {
                background: transparent;
                border: 1px solid rgba(255, 215, 0, 0.3);
                border-radius: 6px;
                color: rgba(255, 255, 255, 0.8);
                cursor: pointer;
                padding: 0.4rem;
                transition: all 0.2s ease;
                font-size: 0.8rem;
            }
            
            .volume-btn:hover,
            .player-expand-btn:hover,
            .player-collapse-btn:hover {
                background: rgba(255, 215, 0, 0.1);
                border-color: #FFD700;
                color: #FFD700;
            }
            
            /* Expanded Player */
            .player-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1rem;
                padding-bottom: 0.5rem;
                border-bottom: 1px solid rgba(255, 215, 0, 0.2);
            }
            
            .player-header h4 {
                color: #FFD700;
                margin: 0;
                font-size: 1rem;
            }
            
            .current-track {
                display: flex;
                gap: 1rem;
                margin-bottom: 1rem;
                padding: 0.75rem;
                background: rgba(26, 26, 37, 0.5);
                border-radius: 8px;
            }
            
            .track-artwork {
                width: 60px;
                height: 60px;
                background: linear-gradient(135deg, #FFD700, #FFA500);
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.5rem;
                color: #000;
                font-weight: bold;
                flex-shrink: 0;
            }
            
            .track-details {
                flex: 1;
                min-width: 0;
            }
            
            .track-title-expanded {
                font-size: 1rem;
                font-weight: 600;
                color: #FFD700;
                margin-bottom: 4px;
            }
            
            .track-artist-expanded {
                font-size: 0.9rem;
                color: rgba(255, 255, 255, 0.8);
                margin-bottom: 6px;
            }
            
            .track-description {
                font-size: 0.8rem;
                color: rgba(255, 255, 255, 0.6);
                line-height: 1.4;
            }
            
            /* Progress Bar */
            .progress-container {
                margin-bottom: 1rem;
            }
            
            .progress-bar {
                position: relative;
                width: 100%;
                height: 6px;
                background: rgba(255, 255, 255, 0.2);
                border-radius: 3px;
                cursor: pointer;
                margin-bottom: 0.5rem;
            }
            
            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, #FFD700, #FFA500);
                border-radius: 3px;
                transition: width 0.1s ease;
            }
            
            .progress-time {
                display: flex;
                justify-content: space-between;
                font-size: 0.7rem;
                color: rgba(255, 255, 255, 0.6);
            }
            
            /* Player Controls */
            .player-controls {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1rem;
                margin-bottom: 1rem;
            }
            
            .prev-btn,
            .play-pause-btn,
            .next-btn {
                background: rgba(255, 215, 0, 0.1);
                border: 1px solid rgba(255, 215, 0, 0.3);
                border-radius: 6px;
                color: #FFD700;
                cursor: pointer;
                padding: 0.5rem;
                transition: all 0.2s ease;
            }
            
            .play-pause-btn {
                padding: 0.6rem 0.8rem;
                font-size: 1.1rem;
            }
            
            .prev-btn:hover,
            .play-pause-btn:hover,
            .next-btn:hover {
                background: rgba(255, 215, 0, 0.2);
                border-color: #FFD700;
                transform: translateY(-1px);
            }
            
            .volume-control {
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            .volume-slider {
                width: 80px;
                height: 4px;
                background: rgba(255, 255, 255, 0.2);
                border-radius: 2px;
                outline: none;
                -webkit-appearance: none;
                transition: opacity 0.2s ease;
                opacity: 0.7;
            }
            
            .volume-control:hover .volume-slider,
            .volume-control.hover .volume-slider {
                opacity: 1;
            }
            
            .volume-slider::-webkit-slider-thumb {
                -webkit-appearance: none;
                width: 12px;
                height: 12px;
                background: #FFD700;
                border-radius: 50%;
                cursor: pointer;
            }
            
            .volume-toggle {
                background: transparent;
                border: none;
                color: rgba(255, 255, 255, 0.8);
                cursor: pointer;
                font-size: 0.9rem;
                transition: color 0.2s ease;
            }
            
            .volume-toggle:hover {
                color: #FFD700;
            }
            
            /* Playlist */
            .playlist-container {
                max-height: 200px;
                overflow-y: auto;
            }
            
            .playlist-container h5 {
                color: #FFD700;
                margin: 0 0 0.5rem 0;
                font-size: 0.9rem;
            }
            
            .playlist-item {
                display: flex;
                align-items: center;
                gap: 0.75rem;
                padding: 0.5rem;
                border-radius: 6px;
                cursor: pointer;
                transition: background 0.2s ease;
            }
            
            .playlist-item:hover {
                background: rgba(255, 215, 0, 0.1);
            }
            
            .playlist-item.active {
                background: rgba(255, 215, 0, 0.2);
                border: 1px solid rgba(255, 215, 0, 0.3);
            }
            
            .playlist-track-number {
                width: 20px;
                text-align: center;
                color: rgba(255, 255, 255, 0.5);
                font-size: 0.8rem;
            }
            
            .playlist-track-info {
                flex: 1;
                min-width: 0;
            }
            
            .playlist-track-title {
                font-size: 0.8rem;
                color: rgba(255, 255, 255, 0.9);
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            
            .playlist-track-artist {
                font-size: 0.7rem;
                color: rgba(255, 255, 255, 0.6);
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            
            .playlist-item.active .playlist-track-title {
                color: #FFD700;
            }
            
            .playlist-track-duration {
                font-size: 0.7rem;
                color: rgba(255, 255, 255, 0.5);
            }
            
            /* Loading State */
            .discreet-audio-player.loading .audio-toggle-btn {
                animation: loading-spin 1s linear infinite;
            }
            
            @keyframes loading-spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            /* Scrollbar */
            .playlist::-webkit-scrollbar {
                width: 4px;
            }
            
            .playlist::-webkit-scrollbar-track {
                background: rgba(255, 215, 0, 0.1);
                border-radius: 2px;
            }
            
            .playlist::-webkit-scrollbar-thumb {
                background: rgba(255, 215, 0, 0.3);
                border-radius: 2px;
            }
            
            /* Mobile Responsiveness */
            @media (max-width: 768px) {
                .discreet-audio-player {
                    top: auto;
                    bottom: 90px;
                    right: 1rem;
                    left: 1rem;
                    min-width: auto;
                }
                
                .discreet-audio-player.expanded {
                    bottom: 1rem;
                    max-height: 70vh;
                    overflow-y: auto;
                }
                
                .player-compact {
                    flex-wrap: wrap;
                }
                
                .track-info {
                    order: -1;
                    width: 100%;
                    margin-bottom: 0.5rem;
                }
            }
            
            @media (max-width: 480px) {
                .discreet-audio-player {
                    left: 0.5rem;
                    right: 0.5rem;
                }
                
                .audio-player-content {
                    padding: 0.75rem;
                }
            }
        `;
    }

    // Static initialization
    static initialize() {
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                new DiscreetAudioSystem();
            });
        } else {
            new DiscreetAudioSystem();
        }
    }

    // Public API
    destroy() {
        if (this.fadeInterval) clearInterval(this.fadeInterval);

        const player = document.getElementById('discreet-audio-player');
        const audio = document.getElementById('unity-audio-player');
        const styles = document.getElementById('discreet-audio-styles');

        player?.remove();
        audio?.remove();
        styles?.remove();

        console.log('🎵 Discreet Audio System destroyed');
    }
}

// Auto-initialize
DiscreetAudioSystem.initialize();

// Global access
window.discreetAudioSystem = DiscreetAudioSystem;

console.log('🎵 Discreet Audio System loaded - φ-harmonic resonance ready');