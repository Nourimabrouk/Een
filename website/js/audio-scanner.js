/**
 * Audio Scanner for Een Unity Mathematics
 * Dynamically scans audio directory and creates unique track listing
 * Handles deduplication across multiple formats (mp3, mp4, webm, etc.)
 * Version: 1.0.0
 */

class AudioScanner {
    constructor() {
        this.audioPath = 'audio/';
        this.supportedFormats = ['.mp3', '.mp4', '.webm', '.ogg', '.wav', '.m4a'];
        this.tracks = [];
        this.preferredFormats = ['.mp3', '.mp4', '.webm', '.ogg', '.wav', '.m4a']; // Priority order - mp4 before webm
    }

    async scanAudioDirectory() {
        try {
            const audioFiles = await this.fetchAudioFileList();
            this.tracks = this.processAudioFiles(audioFiles);
            console.log('🎵 Audio Scanner found', this.tracks.length, 'unique tracks');
            return this.tracks;
        } catch (error) {
            console.warn('Audio scanner failed, using fallback tracks:', error);
            return this.getFallbackTracks();
        }
    }

    async fetchAudioFileList() {
        // Since we can't directly list directory contents in browser,
        // we'll try to access known files from the actual audio directory
        const knownFiles = [
            'Bob Marley - One Love.webm',
            'Bon Jovi - Always.mp4',
            'Bon Jovi - Always.webm',
            'Foreigner - I Want to Know What Love Is.mp4', 
            'Foreigner - I Want to Know What Love Is.webm',
            'Still Alive.mp3',
            'TheFatRat - Unity.mp4',
            'TheFatRat - Unity.webm',
            'U2 - One.mp4',
            'U2 - One.webm'
        ];

        // Test which files actually exist
        const existingFiles = [];
        for (const filename of knownFiles) {
            try {
                const response = await fetch(`${this.audioPath}${filename}`, { method: 'HEAD' });
                if (response.ok) {
                    existingFiles.push(filename);
                }
            } catch (error) {
                // File doesn't exist or can't be accessed
                console.debug(`File not found: ${filename}`);
            }
        }

        return existingFiles;
    }

    processAudioFiles(audioFiles) {
        // Group files by base name (without extension)
        const groupedFiles = {};
        
        audioFiles.forEach(filename => {
            const baseName = this.getBaseName(filename);
            const extension = this.getExtension(filename);
            
            if (!this.supportedFormats.includes(extension)) {
                return; // Skip unsupported formats
            }

            if (!groupedFiles[baseName]) {
                groupedFiles[baseName] = [];
            }
            
            groupedFiles[baseName].push({
                filename,
                extension,
                priority: this.getFormatPriority(extension)
            });
        });

        // Create unique tracks, choosing best format for each
        const uniqueTracks = [];
        
        Object.entries(groupedFiles).forEach(([baseName, files]) => {
            // Sort by priority (lower number = higher priority)
            files.sort((a, b) => a.priority - b.priority);
            const bestFile = files[0];
            
            const track = this.createTrackFromFilename(baseName, bestFile.filename);
            uniqueTracks.push(track);
        });

        return uniqueTracks;
    }

    getBaseName(filename) {
        // Remove extension and normalize
        return filename.replace(/\.[^/.]+$/, '').trim();
    }

    getExtension(filename) {
        const match = filename.match(/\.[^/.]+$/);
        return match ? match[0].toLowerCase() : '';
    }

    getFormatPriority(extension) {
        const index = this.preferredFormats.indexOf(extension);
        return index === -1 ? 999 : index;
    }

    createTrackFromFilename(baseName, filename) {
        // Parse artist and song from filename
        let artist = 'Unknown Artist';
        let songName = baseName;
        let album = 'Unity Mathematics Collection';
        
        // Try to parse "Artist - Song" format
        if (baseName.includes(' - ')) {
            const parts = baseName.split(' - ');
            if (parts.length >= 2) {
                artist = parts[0].trim();
                songName = parts.slice(1).join(' - ').trim();
            }
        }

        // Set album based on artist
        switch (artist.toLowerCase()) {
            case 'u2':
                album = 'Achtung Baby';
                break;
            case 'bon jovi':
                album = 'Cross Road';
                break;
            case 'foreigner':
                album = 'Agent Provocateur';
                break;
            case 'bob marley':
                album = 'Legend';
                break;
            case 'thefatrat':
                album = 'NCS';
                break;
            default:
                album = 'Unity Mathematics Collection';
        }

        // Generate unique ID
        const id = baseName.toLowerCase()
            .replace(/[^a-z0-9\s-]/g, '')
            .replace(/\s+/g, '-')
            .replace(/-+/g, '-')
            .trim();

        // Estimate duration (we'll update this when audio loads)
        let estimatedDuration = 240; // 4 minutes default
        if (songName.toLowerCase().includes('unity')) estimatedDuration = 270;
        if (artist.toLowerCase() === 'u2') estimatedDuration = 276;
        if (artist.toLowerCase() === 'bon jovi') estimatedDuration = 351;

        return {
            id,
            name: songName,
            artist,
            album,
            url: `${this.audioPath}${filename}`,
            duration: estimatedDuration,
            filename,
            isDefault: songName.toLowerCase().includes('unity') || artist.toLowerCase() === 'thefatrat'
        };
    }

    getFallbackTracks() {
        // Fallback tracks if scanning fails - all verified file paths
        return [
            {
                id: 'unity-thefatrat',
                name: 'Unity',
                artist: 'TheFatRat',
                album: 'NCS',
                url: 'audio/TheFatRat - Unity.mp4', // Using mp4 as primary
                duration: 270,
                filename: 'TheFatRat - Unity.mp4',
                isDefault: true
            },
            {
                id: 'one-u2',
                name: 'One',
                artist: 'U2', 
                album: 'Achtung Baby',
                url: 'audio/U2 - One.mp4', // Using mp4 as primary
                duration: 276,
                filename: 'U2 - One.mp4'
            },
            {
                id: 'always-bon-jovi',
                name: 'Always',
                artist: 'Bon Jovi',
                album: 'Cross Road', 
                url: 'audio/Bon Jovi - Always.mp4', // Using mp4 as primary
                duration: 351,
                filename: 'Bon Jovi - Always.mp4'
            },
            {
                id: 'i-want-to-know-what-love-is',
                name: 'I Want to Know What Love Is',
                artist: 'Foreigner',
                album: 'Agent Provocateur',
                url: 'audio/Foreigner - I Want to Know What Love Is.mp4', // Using mp4 as primary
                duration: 297,
                filename: 'Foreigner - I Want to Know What Love Is.mp4'
            },
            {
                id: 'one-love-bob-marley',
                name: 'One Love',
                artist: 'Bob Marley',
                album: 'Legend',
                url: 'audio/Bob Marley - One Love.webm', // Only webm available
                duration: 210,
                filename: 'Bob Marley - One Love.webm'
            },
            {
                id: 'still-alive',
                name: 'Still Alive',
                artist: 'Portal',
                album: 'Portal Soundtrack',
                url: 'audio/Still Alive.mp3', // Only mp3 available
                duration: 178,
                filename: 'Still Alive.mp3'
            }
        ];
    }

    // Get tracks with actual duration after audio files are loaded
    async getTracksWithMetadata() {
        const tracks = await this.scanAudioDirectory();
        
        // Load metadata for each track
        const tracksWithMetadata = await Promise.all(
            tracks.map(async (track) => {
                try {
                    const audio = new Audio(track.url);
                    
                    return new Promise((resolve) => {
                        const timeout = setTimeout(() => {
                            resolve(track); // Return original if loading takes too long
                        }, 3000);
                        
                        audio.addEventListener('loadedmetadata', () => {
                            clearTimeout(timeout);
                            track.duration = audio.duration || track.duration;
                            resolve(track);
                        });
                        
                        audio.addEventListener('error', () => {
                            clearTimeout(timeout);
                            console.warn(`Could not load metadata for ${track.filename}`);
                            resolve(track); // Return original on error
                        });
                        
                        audio.load();
                    });
                } catch (error) {
                    console.warn(`Error loading ${track.filename}:`, error);
                    return track;
                }
            })
        );
        
        return tracksWithMetadata;
    }
}

// Export for use in other modules
window.AudioScanner = AudioScanner;

console.log('🎵 Audio Scanner loaded - Ready to scan for unique tracks');