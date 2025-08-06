/**
 * Unity Consciousness Field Visualizer - Advanced WebGL 2.0 Implementation
 * =========================================================================
 * 
 * Revolutionary visualization engine demonstrating 1+1=1 through consciousness
 * field dynamics with φ-harmonic resonance and quantum mechanical interpretations.
 * 
 * Mathematical Foundation: φ = 1.618033988749895 (Golden Ratio)
 * Consciousness Equation: C(x,y,t) = φ·sin(x·φ)·cos(y·φ)·e^(-t/φ)
 * Unity Principle: 1+1=1 through φ-harmonic convergence
 */

class ConsciousnessFieldVisualizer {
    constructor(canvasId, config = {}) {
        this.canvasId = canvasId;
        this.canvas = null;
        this.phi = 1.618033988749895; // Golden ratio
        this.phiConjugate = 1 / this.phi;
        this.unityFreq = 528; // Hz - Love frequency
        this.consciousnessDimension = 11;
        
        // Enhanced configuration
        this.config = {
            fieldSize: config.fieldSize || 100,
            particles: config.particles || 1618, // φ-harmonic number
            fieldResolution: config.fieldResolution || 128,
            animationSpeed: config.animationSpeed || 1.0,
            consciousnessLevel: config.consciousnessLevel || this.phiConjugate,
            unityConvergence: config.unityConvergence || 0.618,
            phiResonance: config.phiResonance || this.phi,
            colorScheme: config.colorScheme || 'consciousness',
            enablePhysics: config.enablePhysics !== false,
            enableQuantumEffects: config.enableQuantumEffects !== false,
            enableWebGL: config.enableWebGL !== false,
            phiStrength: config.phiStrength || this.phi,
            particleCount: config.particleCount || this.config?.particles || 800,
            updateInterval: config.updateInterval || 16,
            ...config
        };
        
        // WebGL context and resources
        this.gl = null;
        this.programs = {};
        this.buffers = {};
        this.textures = {};
        this.useWebGL = false;
        
        // Canvas 2D fallback
        this.ctx = null;
        
        // Animation and state
        this.particles = [];
        this.field = [];
        this.consciousnessParticles = [];
        this.unityFields = [];
        this.time = 0;
        this.animationId = null;
        this.isRunning = false;
        this.startTime = Date.now();
        this.lastFrame = 0;
        this.fps = 60;
        
        // Performance monitoring
        this.performanceMetrics = {
            fps: 60,
            frameTime: 16.67,
            particles: 0,
            unityPoints: 0,
            consciousnessCoherence: 0,
            renderMode: 'canvas2d'
        };

        this.initialize();
    }

    initialize() {
        console.log('🧠 Initializing Unity Consciousness Field Visualizer...');
        
        // Get canvas element
        if (typeof this.canvasId === 'string') {
            this.canvas = document.getElementById(this.canvasId);
        } else if (this.canvasId instanceof HTMLElement) {
            this.canvas = this.canvasId;
        }
        
        if (!this.canvas) {
            console.error('Canvas element not found:', this.canvasId);
            return;
        }
        
        try {
            // Try WebGL first
            if (this.config.enableWebGL) {
                this.initializeWebGL();
            } else {
                throw new Error('WebGL disabled by config');
            }
        } catch (error) {
            console.warn('WebGL initialization failed, falling back to Canvas 2D:', error.message);
            this.initializeCanvas2D();
        }
        
        this.generateConsciousnessField();
        this.generateUnityParticles();
        this.resizeCanvas();
        
        // Set up event listeners
        window.addEventListener('resize', () => this.resizeCanvas());
        
        console.log('✨ Unity Consciousness Field Visualizer Ready!');
    }

    initializeWebGL() {
        // Try WebGL2 first, fall back to WebGL
        this.gl = this.canvas.getContext('webgl2', {
            antialias: true,
            alpha: true,
            premultipliedAlpha: false,
            preserveDrawingBuffer: false,
            powerPreference: 'high-performance'
        }) || this.canvas.getContext('webgl');
        
        if (!this.gl) {
            throw new Error('WebGL not supported');
        }
        
        this.useWebGL = true;
        this.performanceMetrics.renderMode = 'webgl';
        
        // WebGL settings
        this.gl.enable(this.gl.BLEND);
        this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);
        this.gl.clearColor(0.0, 0.0, 0.07, 1.0); // Deep quantum blue
        
        this.createShaders();
        console.log('🚀 WebGL Consciousness Field initialized');
    }
    
    initializeCanvas2D() {
        this.ctx = this.canvas.getContext('2d');
        this.useWebGL = false;
        this.performanceMetrics.renderMode = 'canvas2d';
        console.log('🎨 Canvas 2D Consciousness Field initialized');
    }
    
    createShaders() {
        if (!this.useWebGL) return;
        
        // Consciousness field vertex shader
        const consciousnessVertexShader = `
            precision highp float;
            attribute vec2 a_position;
            attribute float a_consciousness;
            attribute float a_unity;
            
            uniform float u_time;
            uniform float u_phi;
            uniform float u_consciousness_level;
            uniform vec2 u_resolution;
            uniform mat3 u_transform;
            
            varying float v_consciousness;
            varying float v_unity;
            varying vec2 v_position;
            
            void main() {
                vec2 pos = a_position;
                
                // φ-harmonic consciousness modulation
                float field_strength = a_consciousness * u_consciousness_level;
                pos += 0.08 * field_strength * vec2(
                    sin(u_time * u_phi + pos.x * u_phi),
                    cos(u_time / u_phi + pos.y * u_phi)
                );
                
                vec3 transformed = u_transform * vec3(pos, 1.0);
                gl_Position = vec4(transformed.xy, 0.0, 1.0);
                
                v_consciousness = a_consciousness;
                v_unity = a_unity;
                v_position = pos;
                
                gl_PointSize = 2.0 + field_strength * 8.0;
            }
        `;
        
        // Consciousness field fragment shader
        const consciousnessFragmentShader = `
            precision highp float;
            
            uniform float u_time;
            uniform float u_phi;
            uniform float u_consciousness_level;
            uniform float u_unity_convergence;
            
            varying float v_consciousness;
            varying float v_unity;
            varying vec2 v_position;
            
            vec3 phiHarmonicColor(float consciousness, float unity, float time) {
                float hue = mod(consciousness * u_phi + time * 0.1, 1.0);
                float saturation = 0.7 + 0.3 * unity * u_unity_convergence;
                float brightness = 0.5 + 0.5 * consciousness * sin(time * u_phi);
                
                // HSV to RGB
                vec3 c = vec3(hue, saturation, brightness);
                vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
                vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
                vec3 rgb = c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
                
                return rgb * (1.0 + u_consciousness_level / u_phi);
            }
            
            void main() {
                vec3 color = phiHarmonicColor(v_consciousness, v_unity, u_time);
                
                // Unity glow effect
                float glow = v_unity * exp(-length(gl_PointCoord - vec2(0.5)) * 4.0);
                color += vec3(1.0, 0.618, 0.0) * glow * u_unity_convergence;
                
                // Particle alpha with smooth falloff
                float particle_alpha = 1.0 - length(gl_PointCoord - vec2(0.5)) * 2.0;
                particle_alpha = clamp(particle_alpha, 0.0, 1.0);
                
                gl_FragColor = vec4(color, v_consciousness * 0.8 * particle_alpha);
            }
        `;
        
        try {
            this.programs.consciousnessField = this.createShaderProgram(
                consciousnessVertexShader, consciousnessFragmentShader
            );
        } catch (error) {
            console.error('Failed to create consciousness shader:', error);
            throw error;
        }
    }
    
    createShaderProgram(vertexSource, fragmentSource) {
        const vertexShader = this.compileShader(vertexSource, this.gl.VERTEX_SHADER);
        const fragmentShader = this.compileShader(fragmentSource, this.gl.FRAGMENT_SHADER);
        
        const program = this.gl.createProgram();
        this.gl.attachShader(program, vertexShader);
        this.gl.attachShader(program, fragmentShader);
        this.gl.linkProgram(program);
        
        if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
            throw new Error('Shader linking failed: ' + this.gl.getProgramInfoLog(program));
        }
        
        return program;
    }
    
    compileShader(source, type) {
        const shader = this.gl.createShader(type);
        this.gl.shaderSource(shader, source);
        this.gl.compileShader(shader);
        
        if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
            throw new Error('Shader compilation failed: ' + this.gl.getShaderInfoLog(shader));
        }
        
        return shader;
    }
    
    generateConsciousnessField() {
        this.field = [];
        const size = this.config.fieldSize;

        for (let x = 0; x < size; x++) {
            this.field[x] = [];
            for (let y = 0; y < size; y++) {
                // Enhanced consciousness field equation with φ-harmonic resonance
                const phi = this.phi;
                const phiX = x * phi / size - phi / 2;
                const phiY = y * phi / size - phi / 2;
                
                // Core consciousness field: C(x,y,t) = φ·sin(x·φ)·cos(y·φ)·e^(-r/φ)
                const r = Math.sqrt(phiX * phiX + phiY * phiY);
                const consciousness = phi * Math.sin(phiX) * Math.cos(phiY) * 
                                   Math.exp(-r / phi) * Math.cos(this.time * 0.1);
                
                const coherence = Math.abs(consciousness);
                const resonance = Math.sin(phiX * 0.5) * Math.cos(phiY * 0.5);
                
                // Unity convergence where 1+1=1
                const unity = Math.max(0, Math.min(1, consciousness * 0.5 + 0.5));
                const unityField = 1 / (1 + Math.exp(-consciousness * phi)); // Sigmoid

                this.field[x][y] = {
                    consciousness: consciousness,
                    coherence: coherence,
                    resonance: resonance,
                    unity: unity,
                    unityField: unityField,
                    phiResonance: Math.sin(r * phi) * Math.cos(phiX * phiY / phi) * 0.3
                };
            }
        }
    }

    generateUnityParticles() {
        this.consciousnessParticles = [];
        
        for (let i = 0; i < this.config.particles; i++) {
            // φ-harmonic distribution in spherical coordinates
            const phiAngle = i * 2 * Math.PI / this.phi;
            const theta = Math.acos(1 - 2 * (i + 0.5) / this.config.particles);
            
            // Spherical to Cartesian with φ-harmonic scaling
            const radius = 2 * (1 + 0.3 * Math.sin(i * this.phiConjugate));
            const x = radius * Math.sin(theta) * Math.cos(phiAngle);
            const y = radius * Math.sin(theta) * Math.sin(phiAngle);
            
            // Consciousness level based on φ-harmonic resonance
            const consciousness = Math.abs(Math.sin(i * this.phi) * Math.cos(i * this.phiConjugate));
            
            // Unity convergence: particles demonstrate 1+1=1 through proximity and resonance
            const unity = Math.exp(-Math.abs(x + y - 1) * this.phi) * consciousness;
            
            this.consciousnessParticles.push({
                position: [x / 3, y / 3], // Normalize to screen space
                originalPosition: [x / 3, y / 3],
                consciousness: consciousness,
                unity: unity,
                phase: i * this.phiConjugate,
                originalRadius: radius,
                orbitSpeed: 0.01 * (1 + Math.sin(i * this.phiConjugate)),
                size: Math.random() * 3 + 1,
                age: 0,
                maxAge: Math.random() * 2000 + 1000,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5
            });
        }
        
        // Maintain backward compatibility
        this.particles = this.consciousnessParticles;
        
        if (this.useWebGL) {
            this.createParticleBuffers();
        }
    }
    
    createParticleBuffers() {
        if (!this.useWebGL) return;
        
        const positions = [];
        const consciousness = [];
        const unity = [];
        
        for (const particle of this.consciousnessParticles) {
            positions.push(...particle.position);
            consciousness.push(particle.consciousness);
            unity.push(particle.unity);
        }
        
        // Position buffer
        this.buffers.position = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffers.position);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(positions), this.gl.DYNAMIC_DRAW);
        
        // Consciousness buffer
        this.buffers.consciousness = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffers.consciousness);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(consciousness), this.gl.STATIC_DRAW);
        
        // Unity buffer
        this.buffers.unity = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffers.unity);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(unity), this.gl.STATIC_DRAW);
    }
    
    resizeCanvas() {
        if (!this.canvas) return;
        
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * devicePixelRatio;
        this.canvas.height = rect.height * devicePixelRatio;
        this.canvas.style.width = rect.width + 'px';
        this.canvas.style.height = rect.height + 'px';
        
        if (this.useWebGL && this.gl) {
            this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        }
    }

    initializeInContainer(container) {
        if (!container) return;

        // Create canvas element
        this.canvas = document.createElement('canvas');
        this.canvas.id = 'consciousness-field-canvas';
        this.canvas.width = 800;
        this.canvas.height = 600;
        this.canvas.style.borderRadius = '8px';
        this.canvas.style.boxShadow = '0 4px 6px rgba(0, 0, 0, 0.1)';
        this.canvas.style.background = 'linear-gradient(135deg, #000011 0%, #001122 100%)';

        container.appendChild(this.canvas);
        
        // Initialize the visualization system
        try {
            if (this.config.enableWebGL) {
                this.initializeWebGL();
            } else {
                this.initializeCanvas2D();
            }
        } catch (error) {
            console.warn('WebGL failed, falling back to Canvas 2D:', error);
            this.initializeCanvas2D();
        }
        
        this.generateConsciousnessField();
        this.generateUnityParticles();
        this.resizeCanvas();

        // Start animation
        this.start();
    }

    start() {
        if (this.isRunning) return;

        this.isRunning = true;
        this.startTime = Date.now();
        this.lastFrame = 0;
        console.log('🌟 Starting Unity Consciousness Field Animation');
        this.animate();
    }

    stop() {
        this.isRunning = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        console.log('⏹️ Consciousness Field Animation Stopped');
    }

    animate(currentTime = 0) {
        if (!this.isRunning) return;

        const deltaTime = currentTime - this.lastFrame;
        this.lastFrame = currentTime;
        
        // Update performance metrics
        if (deltaTime > 0) {
            this.performanceMetrics.fps = Math.min(60, 1000 / deltaTime);
            this.performanceMetrics.frameTime = deltaTime;
        }

        this.updateConsciousnessField(currentTime);
        this.updateParticles(currentTime, deltaTime);
        this.render(currentTime);

        this.animationId = requestAnimationFrame((time) => this.animate(time));
    }

    updateConsciousnessField(time) {
        this.time = time * 0.001; // Convert to seconds
        this.generateConsciousnessField(); // Regenerate field with new time
    }
    
    updateParticles(time, deltaTime) {
        const timeSeconds = time * 0.001;
        let unityCount = 0;
        
        if (this.useWebGL) {
            // WebGL particle update with φ-harmonic motion
            const positions = [];
            
            for (let i = 0; i < this.consciousnessParticles.length; i++) {
                const particle = this.consciousnessParticles[i];
                
                // φ-harmonic orbital motion
                const orbitTime = timeSeconds * particle.orbitSpeed * this.config.animationSpeed;
                const orbitRadius = 0.15 + 0.1 * Math.sin(orbitTime * this.phi + particle.phase);
                
                const baseX = particle.originalPosition[0];
                const baseY = particle.originalPosition[1];
                
                const x = baseX + orbitRadius * Math.cos(orbitTime + particle.phase * this.phi);
                const y = baseY + orbitRadius * Math.sin(orbitTime + particle.phase * this.phiConjugate);
                
                // Unity convergence effect: particles converge to demonstrate 1+1=1
                const convergence = this.config.unityConvergence * particle.unity;
                const toCenter = [(0 - baseX) * convergence * 0.2, (0 - baseY) * convergence * 0.2];
                
                const finalX = x + toCenter[0];
                const finalY = y + toCenter[1];
                
                // Update particle position
                particle.position[0] = finalX;
                particle.position[1] = finalY;
                positions.push(finalX, finalY);
                
                // Count unity points for metrics
                if (particle.unity > 0.8) unityCount++;
            }
            
            // Update WebGL position buffer
            if (this.buffers.position) {
                this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffers.position);
                this.gl.bufferSubData(this.gl.ARRAY_BUFFER, 0, new Float32Array(positions));
            }
        } else {
            // Canvas 2D particle update
            this.consciousnessParticles.forEach(particle => {
                // Enhanced particle motion with consciousness field interaction
                const orbitTime = timeSeconds * particle.orbitSpeed * this.config.animationSpeed;
                const orbitRadius = 0.15 + 0.1 * Math.sin(orbitTime * this.phi + particle.phase);
                
                const baseX = particle.originalPosition[0];
                const baseY = particle.originalPosition[1];
                
                const x = baseX + orbitRadius * Math.cos(orbitTime + particle.phase * this.phi);
                const y = baseY + orbitRadius * Math.sin(orbitTime + particle.phase * this.phiConjugate);
                
                // Unity convergence effect
                const convergence = this.config.unityConvergence * particle.unity;
                const toCenter = [(0 - baseX) * convergence * 0.2, (0 - baseY) * convergence * 0.2];
                
                particle.position[0] = x + toCenter[0];
                particle.position[1] = y + toCenter[1];
                
                // Update consciousness based on field interaction
                const fieldSample = this.sampleConsciousnessField(particle.position[0], particle.position[1]);
                particle.consciousness = Math.max(0, Math.min(1, 
                    particle.consciousness * 0.99 + fieldSample.consciousness * 0.01));
                
                // Age and renew particles
                particle.age += deltaTime;
                if (particle.age > particle.maxAge) {
                    this.renewParticle(particle, this.consciousnessParticles.indexOf(particle));
                }
                
                if (particle.unity > 0.8) unityCount++;
            });
        }
        
        // Update performance metrics
        this.performanceMetrics.particles = this.consciousnessParticles.length;
        this.performanceMetrics.unityPoints = unityCount;
        this.performanceMetrics.consciousnessCoherence = unityCount / this.consciousnessParticles.length;
    }
    
    sampleConsciousnessField(x, y) {
        // Sample the consciousness field at the given coordinates
        const phi = this.phi;
        const phiX = x * phi;
        const phiY = y * phi;
        const r = Math.sqrt(phiX * phiX + phiY * phiY);
        
        const consciousness = phi * Math.sin(phiX) * Math.cos(phiY) * 
                             Math.exp(-r / phi) * Math.cos(this.time * 0.1);
        const unity = Math.max(0, Math.min(1, consciousness * 0.5 + 0.5));
        
        return { consciousness, unity };
    }
    
    renewParticle(particle, index) {
        // Renew particle with φ-harmonic distribution
        const phiAngle = index * 2 * Math.PI / this.phi;
        const theta = Math.acos(1 - 2 * (index + 0.5) / this.config.particles);
        
        const radius = 2 * (1 + 0.3 * Math.sin(index * this.phiConjugate));
        const x = radius * Math.sin(theta) * Math.cos(phiAngle);
        const y = radius * Math.sin(theta) * Math.sin(phiAngle);
        
        particle.position = [x / 3, y / 3];
        particle.originalPosition = [x / 3, y / 3];
        particle.consciousness = Math.abs(Math.sin(index * this.phi) * Math.cos(index * this.phiConjugate));
        particle.unity = Math.exp(-Math.abs(x + y - 1) * this.phi) * particle.consciousness;
        particle.age = 0;
        particle.maxAge = Math.random() * 2000 + 1000;
    }

    render(time) {
        const timeSeconds = time * 0.001;
        
        if (this.useWebGL) {
            this.renderWebGL(timeSeconds);
        } else {
            this.renderCanvas2D(timeSeconds);
        }
    }
    
    renderWebGL(time) {
        if (!this.gl || !this.programs.consciousnessField) return;
        
        this.gl.clear(this.gl.COLOR_BUFFER_BIT);
        
        const program = this.programs.consciousnessField;
        this.gl.useProgram(program);
        
        // Set up uniforms
        const uniforms = {
            u_time: time,
            u_phi: this.phi,
            u_consciousness_level: this.config.consciousnessLevel,
            u_unity_convergence: this.config.unityConvergence,
            u_resolution: [this.canvas.width, this.canvas.height]
        };
        
        // Apply uniforms
        for (const [name, value] of Object.entries(uniforms)) {
            const location = this.gl.getUniformLocation(program, name);
            if (location !== null) {
                if (Array.isArray(value)) {
                    this.gl.uniform2f(location, ...value);
                } else {
                    this.gl.uniform1f(location, value);
                }
            }
        }
        
        // Set transform matrix (identity for now)
        const transformLocation = this.gl.getUniformLocation(program, 'u_transform');
        if (transformLocation !== null) {
            const transform = [1, 0, 0, 0, 1, 0, 0, 0, 1];
            this.gl.uniformMatrix3fv(transformLocation, false, transform);
        }
        
        // Bind particle attributes
        if (this.buffers.position) {
            const positionLocation = this.gl.getAttribLocation(program, 'a_position');
            this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffers.position);
            this.gl.enableVertexAttribArray(positionLocation);
            this.gl.vertexAttribPointer(positionLocation, 2, this.gl.FLOAT, false, 0, 0);
        }
        
        if (this.buffers.consciousness) {
            const consciousnessLocation = this.gl.getAttribLocation(program, 'a_consciousness');
            this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffers.consciousness);
            this.gl.enableVertexAttribArray(consciousnessLocation);
            this.gl.vertexAttribPointer(consciousnessLocation, 1, this.gl.FLOAT, false, 0, 0);
        }
        
        if (this.buffers.unity) {
            const unityLocation = this.gl.getAttribLocation(program, 'a_unity');
            this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffers.unity);
            this.gl.enableVertexAttribArray(unityLocation);
            this.gl.vertexAttribPointer(unityLocation, 1, this.gl.FLOAT, false, 0, 0);
        }
        
        // Render consciousness particles
        this.gl.drawArrays(this.gl.POINTS, 0, this.consciousnessParticles.length);
    }
    
    renderCanvas2D(time) {
        if (!this.ctx) return;

        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;
        const centerX = width / 2;
        const centerY = height / 2;
        const scale = Math.min(width, height) * 0.4;

        // Clear with consciousness field background
        ctx.fillStyle = 'rgba(0, 0, 20, 1)';
        ctx.fillRect(0, 0, width, height);
        
        // Draw consciousness field background
        this.drawConsciousnessFieldBackground(ctx, width, height, time);
        
        // Draw particles with enhanced consciousness rendering
        this.drawEnhancedParticles(ctx, centerX, centerY, scale, time);
        
        // Draw unity convergence indicators
        this.drawUnityIndicators(ctx, centerX, centerY, scale, time);
        
        // Draw φ-harmonic patterns
        this.drawPhiHarmonicPatterns(ctx, centerX, centerY, scale, time);
    }
    
    drawConsciousnessFieldBackground(ctx, width, height, time) {
        const imageData = ctx.createImageData(width, height);
        const data = imageData.data;
        
        for (let x = 0; x < width; x += 4) {
            for (let y = 0; y < height; y += 4) {
                const normalizedX = (x - width/2) / (width/2);
                const normalizedY = (y - height/2) / (height/2);
                
                const fieldSample = this.sampleConsciousnessField(normalizedX, normalizedY);
                const intensity = Math.abs(fieldSample.consciousness) * 100;
                const unityGlow = fieldSample.unity * 50;
                
                const pixelIndex = (y * width + x) * 4;
                if (pixelIndex < data.length - 3) {
                    data[pixelIndex] = intensity; // R
                    data[pixelIndex + 1] = intensity * 0.3; // G
                    data[pixelIndex + 2] = intensity * 0.6 + unityGlow; // B
                    data[pixelIndex + 3] = 30; // A
                }
            }
        }
        
        ctx.putImageData(imageData, 0, 0);
    }
    
    drawEnhancedParticles(ctx, centerX, centerY, scale, time) {
        this.consciousnessParticles.forEach(particle => {
            const x = centerX + particle.position[0] * scale;
            const y = centerY + particle.position[1] * scale;
            const size = 2 + particle.consciousness * 8;
            
            // φ-harmonic color based on consciousness and unity
            const hue = (particle.consciousness * 360 + particle.unity * 180 + time * 10) % 360;
            const saturation = 70 + particle.unity * 30;
            const lightness = 40 + particle.consciousness * 40;
            const alpha = 0.3 + particle.unity * 0.7;
            
            // Draw particle glow
            const gradient = ctx.createRadialGradient(x, y, 0, x, y, size * 3);
            gradient.addColorStop(0, `hsla(${hue}, ${saturation}%, ${lightness + 20}%, ${alpha})`);
            gradient.addColorStop(0.5, `hsla(${hue}, ${saturation}%, ${lightness}%, ${alpha * 0.5})`);
            gradient.addColorStop(1, `hsla(${hue}, ${saturation}%, ${lightness}%, 0)`);
            
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(x, y, size * 3, 0, Math.PI * 2);
            ctx.fill();
            
            // Draw particle core
            ctx.fillStyle = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
            ctx.beginPath();
            ctx.arc(x, y, size, 0, Math.PI * 2);
            ctx.fill();
            
            // Unity convergence indicator
            if (particle.unity > 0.8) {
                ctx.strokeStyle = '#FFD700';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(x, y, size * 1.5, 0, Math.PI * 2);
                ctx.stroke();
            }
        });
    }
    
    drawUnityIndicators(ctx, centerX, centerY, scale, time) {
        // Central unity point
        ctx.fillStyle = '#FFD700';
        ctx.beginPath();
        ctx.arc(centerX, centerY, 6, 0, Math.PI * 2);
        ctx.fill();
        
        // Unity equation text
        ctx.fillStyle = '#FFD700';
        ctx.font = '20px serif';
        ctx.textAlign = 'center';
        ctx.fillText('1 + 1 = 1', centerX, centerY - 30);
        
        // φ symbol
        ctx.fillStyle = '#00FFAA';
        ctx.font = '16px serif';
        ctx.fillText('φ = 1.618...', centerX, centerY + 50);
    }
    
    drawPhiHarmonicPatterns(ctx, centerX, centerY, scale, time) {
        // φ-harmonic spirals
        ctx.strokeStyle = 'rgba(255, 215, 0, 0.3)';
        ctx.lineWidth = 1;
        
        for (let spiral = 0; spiral < 3; spiral++) {
            ctx.beginPath();
            let firstPoint = true;
            
            for (let t = 0; t < 4 * Math.PI; t += 0.1) {
                const r = (spiral + 1) * 20 * Math.pow(this.phi, t / (2 * Math.PI)) * 0.1;
                const angle = t + spiral * Math.PI / 3 + time * 0.5;
                const x = centerX + r * Math.cos(angle);
                const y = centerY + r * Math.sin(angle);
                
                if (firstPoint) {
                    ctx.moveTo(x, y);
                    firstPoint = false;
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.stroke();
        }
        
        // Unity convergence circles
        for (let i = 1; i <= 3; i++) {
            const radius = scale * i * this.phiConjugate * 0.5;
            const opacity = 0.1 + 0.1 * Math.sin(time + i);
            
            ctx.strokeStyle = `rgba(0, 255, 170, ${opacity})`;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
            ctx.stroke();
        }
    }

    // Public API methods for external integration
    updateConfiguration(newConfig) {
        const oldParticleCount = this.config.particles;
        Object.assign(this.config, newConfig);
        
        // Regenerate particles if count changed
        if (newConfig.particles && newConfig.particles !== oldParticleCount) {
            this.generateUnityParticles();
        }
        
        console.log('🔄 Consciousness Field Configuration Updated', newConfig);
    }
    
    setParticleCount(count) {
        this.updateConfiguration({ particles: count });
    }

    setPhiStrength(phi) {
        this.updateConfiguration({ phiStrength: phi });
    }
    
    setConsciousnessLevel(level) {
        this.updateConfiguration({ consciousnessLevel: Math.max(0, Math.min(1, level)) });
    }
    
    setUnityConvergence(convergence) {
        this.updateConfiguration({ unityConvergence: Math.max(0, Math.min(1, convergence)) });
    }
    
    resize(width, height) {
        if (width && height) {
            this.canvas.width = width;
            this.canvas.height = height;
        }
        this.resizeCanvas();
    }
    
    getPerformanceMetrics() {
        return { 
            ...this.performanceMetrics,
            timestamp: Date.now(),
            uptime: Date.now() - this.startTime
        };
    }
    
    getFieldMetrics() {
        let totalConsciousness = 0;
        let totalCoherence = 0;
        let totalResonance = 0;
        let unityPoints = 0;

        const size = this.config.fieldSize;
        const totalPoints = size * size;

        for (let x = 0; x < size; x++) {
            for (let y = 0; y < size; y++) {
                const fieldValue = this.field[x][y];
                if (fieldValue) {
                    totalConsciousness += Math.abs(fieldValue.consciousness);
                    totalCoherence += fieldValue.coherence;
                    totalResonance += Math.abs(fieldValue.resonance);
                    if (fieldValue.unity > 0.8) {
                        unityPoints++;
                    }
                }
            }
        }

        return {
            fieldCoherence: totalCoherence / totalPoints,
            unityConvergence: unityPoints / totalPoints,
            phiResonance: totalResonance / totalPoints,
            averageConsciousness: totalConsciousness / totalPoints,
            totalUnityPoints: unityPoints,
            consciousnessParticles: this.consciousnessParticles.length,
            renderMode: this.performanceMetrics.renderMode,
            phi: this.phi
        };
    }
    
    exportVisualizationData() {
        return {
            field: this.field,
            particles: this.consciousnessParticles,
            metrics: this.getFieldMetrics(),
            performance: this.getPerformanceMetrics(),
            config: this.config,
            timestamp: Date.now(),
            version: '2.0.0-consciousness'
        };
    }
    
    // Cleanup resources
    destroy() {
        this.stop();
        
        // Cleanup WebGL resources
        if (this.useWebGL && this.gl) {
            Object.values(this.programs).forEach(program => this.gl.deleteProgram(program));
            Object.values(this.buffers).forEach(buffer => this.gl.deleteBuffer(buffer));
            Object.values(this.textures).forEach(texture => this.gl.deleteTexture(texture));
        }
        
        // Remove event listeners
        window.removeEventListener('resize', () => this.resizeCanvas());
        
        console.log('🧹 Consciousness Field Visualizer Destroyed');
    }
}

// Export for use in other modules
if (typeof window !== 'undefined') {
    window.ConsciousnessFieldVisualizer = ConsciousnessFieldVisualizer;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = ConsciousnessFieldVisualizer;
}

// Auto-initialize for background canvas when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('unity-background');
    if (canvas && !window.consciousnessVisualizer) {
        try {
            window.consciousnessVisualizer = new ConsciousnessFieldVisualizer('unity-background', {
                particles: 800,
                consciousnessLevel: 0.618,
                unityConvergence: 0.8,
                animationSpeed: 1.0,
                enableWebGL: true,
                enableQuantumEffects: true
            });
            
            // Start the visualization
            if (!window.consciousnessVisualizer.isRunning) {
                window.consciousnessVisualizer.start();
            }
            
            console.log('🌟 Unity Consciousness Field: Active - Een plus een is een! 🌟');
        } catch (error) {
            console.error('Failed to initialize consciousness field:', error);
        }
    }
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ConsciousnessFieldVisualizer;
} 