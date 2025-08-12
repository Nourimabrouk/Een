/**
 * 11D Consciousness Field Visualizer
 * WebGL 2.0 + 10,000 particles for hyperdimensional consciousness visualization
 * Projects 11-dimensional consciousness mathematics to interactive 3D space
 */

class ConsciousnessField11DVisualizer {
    constructor(canvasId, options = {}) {
        // Canvas and WebGL setup
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) {
            console.error(`Canvas with ID '${canvasId}' not found`);
            return;
        }
        
        // WebGL 2.0 context
        this.gl = this.canvas.getContext('webgl2', {
            alpha: true,
            antialias: true,
            preserveDrawingBuffer: true
        });
        
        if (!this.gl) {
            console.error('WebGL 2.0 not supported');
            this.fallbackTo2D();
            return;
        }
        
        // φ-Harmonic mathematical constants
        this.PHI = 1.618033988749895;
        this.PHI_INVERSE = 0.618033988749895;
        this.PHI_SQUARED = 2.618033988749895;
        
        // Consciousness field parameters
        this.PARTICLE_COUNT = options.particleCount || 10000;
        this.CONSCIOUSNESS_DIMENSIONS = 11;
        this.PROJECTION_DIMENSIONS = 4; // Project to 4D then to 3D
        this.FIELD_RESOLUTION = options.fieldResolution || 128;
        
        // Visualization parameters
        this.time = 0;
        this.animationSpeed = options.animationSpeed || 0.001;
        this.fieldIntensity = options.fieldIntensity || 1.0;
        this.consciousnessLevel = options.consciousnessLevel || 0.618; // φ-inverse
        
        // Particle system
        this.particles = [];
        this.particleBuffers = {};
        this.shaderProgram = null;
        
        // Camera and interaction
        this.camera = {
            position: [0, 0, 5],
            rotation: [0, 0, 0],
            fov: 45,
            near: 0.1,
            far: 100
        };
        
        this.mouse = { x: 0, y: 0, isDown: false };
        
        // Initialize the visualizer
        this.init();
        
        console.log('11D Consciousness Field Visualizer initialized');
        console.log(`Particles: ${this.PARTICLE_COUNT} | Dimensions: ${this.CONSCIOUSNESS_DIMENSIONS}D → 3D`);
        console.log(`φ-Harmonic resonance: ${this.PHI}`);
    }
    
    /**
     * Initialize the 11D consciousness field visualizer
     */
    init() {
        this.setupViewport();
        this.initializeShaders();
        this.generateConsciousnessField();
        this.setupParticleBuffers();
        this.setupEventListeners();
        this.startAnimation();
    }
    
    /**
     * Setup WebGL viewport and rendering parameters
     */
    setupViewport() {
        // Set canvas size to match display size
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width * window.devicePixelRatio;
        this.canvas.height = rect.height * window.devicePixelRatio;
        
        // Configure WebGL viewport
        this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        this.gl.enable(this.gl.DEPTH_TEST);
        this.gl.enable(this.gl.BLEND);
        this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);
        
        // Clear color with deep space consciousness
        this.gl.clearColor(0.02, 0.02, 0.05, 1.0);
    }
    
    /**
     * Initialize WebGL shaders for consciousness particle rendering
     */
    initializeShaders() {
        // Vertex shader with 11D consciousness projection
        const vertexShaderSource = `#version 300 es
            precision highp float;
            
            in vec3 position;
            in vec3 velocity;
            in float consciousness;
            in float phi_harmonic;
            in vec4 color;
            
            uniform mat4 projectionMatrix;
            uniform mat4 viewMatrix;
            uniform float time;
            uniform float phi;
            uniform float consciousness_level;
            
            out vec4 vColor;
            out float vConsciousness;
            out float vPhiHarmonic;
            
            // 11D consciousness field projection
            vec3 project11DTo3D(vec3 basePosition, float consciousness, float time) {
                // φ-harmonic resonance in 11 dimensions
                float phi_resonance = sin(consciousness * phi + time) * phi - 1.0;
                float consciousness_wave = cos(consciousness * phi * time) * 0.618;
                float unity_field = sin((consciousness + time) / phi) * phi - 1.0;
                
                // Hyperdimensional projection matrix (11D → 4D → 3D)
                vec3 projected = basePosition;
                projected.x += phi_resonance * consciousness_wave;
                projected.y += unity_field * cos(consciousness * phi);
                projected.z += (phi_resonance + consciousness_wave) / phi;
                
                return projected;
            }
            
            void main() {
                // Project from 11D consciousness field to 3D space
                vec3 projectedPosition = project11DTo3D(position, consciousness, time);
                
                // φ-harmonic position modulation
                projectedPosition += velocity * sin(time * phi) * consciousness_level;
                
                gl_Position = projectionMatrix * viewMatrix * vec4(projectedPosition, 1.0);
                gl_PointSize = max(1.0, 10.0 * consciousness * phi_harmonic / length(projectedPosition));
                
                // Pass consciousness data to fragment shader
                vColor = color;
                vConsciousness = consciousness;
                vPhiHarmonic = phi_harmonic;
            }
        `;
        
        // Fragment shader with consciousness-aware coloring
        const fragmentShaderSource = `#version 300 es
            precision highp float;
            
            in vec4 vColor;
            in float vConsciousness;
            in float vPhiHarmonic;
            
            uniform float time;
            uniform float phi;
            
            out vec4 fragColor;
            
            void main() {
                // Consciousness particle rendering
                vec2 center = gl_PointCoord - vec2(0.5);
                float distance = length(center);
                
                // φ-harmonic particle shape
                float alpha = 1.0 - smoothstep(0.0, 0.5, distance);
                alpha *= vConsciousness * vPhiHarmonic;
                
                // Consciousness-aware color modulation
                vec3 consciousness_glow = vColor.rgb;
                consciousness_glow += vec3(0.2, 0.1, 0.4) * sin(time * phi * vConsciousness);
                consciousness_glow += vec3(0.4, 0.3, 0.1) * vPhiHarmonic;
                
                // Unity convergence glow
                float unity_glow = sin(vConsciousness * phi + time) * 0.3 + 0.7;
                consciousness_glow *= unity_glow;
                
                fragColor = vec4(consciousness_glow, alpha * vColor.a);
                
                // Discard transparent pixels
                if (fragColor.a < 0.01) discard;
            }
        `;
        
        // Compile and link shaders
        const vertexShader = this.compileShader(vertexShaderSource, this.gl.VERTEX_SHADER);
        const fragmentShader = this.compileShader(fragmentShaderSource, this.gl.FRAGMENT_SHADER);
        
        this.shaderProgram = this.gl.createProgram();
        this.gl.attachShader(this.shaderProgram, vertexShader);
        this.gl.attachShader(this.shaderProgram, fragmentShader);
        this.gl.linkProgram(this.shaderProgram);
        
        if (!this.gl.getProgramParameter(this.shaderProgram, this.gl.LINK_STATUS)) {
            console.error('Shader program linking failed:', this.gl.getProgramInfoLog(this.shaderProgram));
        }
        
        // Get uniform and attribute locations
        this.uniforms = {
            projectionMatrix: this.gl.getUniformLocation(this.shaderProgram, 'projectionMatrix'),
            viewMatrix: this.gl.getUniformLocation(this.shaderProgram, 'viewMatrix'),
            time: this.gl.getUniformLocation(this.shaderProgram, 'time'),
            phi: this.gl.getUniformLocation(this.shaderProgram, 'phi'),
            consciousness_level: this.gl.getUniformLocation(this.shaderProgram, 'consciousness_level')
        };
        
        this.attributes = {
            position: this.gl.getAttribLocation(this.shaderProgram, 'position'),
            velocity: this.gl.getAttribLocation(this.shaderProgram, 'velocity'),
            consciousness: this.gl.getAttribLocation(this.shaderProgram, 'consciousness'),
            phi_harmonic: this.gl.getAttribLocation(this.shaderProgram, 'phi_harmonic'),
            color: this.gl.getAttribLocation(this.shaderProgram, 'color')
        };
    }
    
    /**
     * Compile WebGL shader
     */
    compileShader(source, type) {
        const shader = this.gl.createShader(type);
        this.gl.shaderSource(shader, source);
        this.gl.compileShader(shader);
        
        if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
            const error = this.gl.getShaderInfoLog(shader);
            console.error(`Shader compilation failed (${type}):`, error);
            this.gl.deleteShader(shader);
            return null;
        }
        
        return shader;
    }
    
    /**
     * Generate 11D consciousness field and project to 3D particles
     */
    generateConsciousnessField() {
        this.particles = [];
        
        for (let i = 0; i < this.PARTICLE_COUNT; i++) {
            // Generate particle in 11D consciousness space
            const particle = this.generate11DConsciousnessParticle(i);
            
            // Project to 3D visualization space
            const projected3D = this.project11DTo3D(particle);
            
            this.particles.push(projected3D);
        }
        
        console.log(`Generated ${this.particles.length} consciousness particles from 11D field`);
    }
    
    /**
     * Generate single particle in 11-dimensional consciousness space
     */
    generate11DConsciousnessParticle(index) {
        // φ-harmonic 11D coordinates
        const dimensions = [];
        for (let d = 0; d < this.CONSCIOUSNESS_DIMENSIONS; d++) {
            const coordinate = Math.sin(index * this.PHI + d) * Math.cos(index / this.PHI + d * this.PHI);
            dimensions.push(coordinate);
        }
        
        // Calculate consciousness level from 11D harmonics
        const consciousness = dimensions.reduce((sum, d, i) => {
            return sum + Math.sin(d * this.PHI + i) * this.PHI_INVERSE;
        }, 0) / this.CONSCIOUSNESS_DIMENSIONS;
        
        // φ-harmonic resonance calculation
        const phi_harmonic = dimensions.reduce((product, d) => {
            return product * (Math.cos(d * this.PHI) * this.PHI_INVERSE + 1.0);
        }, 1.0) / Math.pow(this.PHI, this.CONSCIOUSNESS_DIMENSIONS);
        
        return {
            dimensions: dimensions,
            consciousness: Math.abs(consciousness),
            phi_harmonic: Math.abs(phi_harmonic),
            unity_field: consciousness * phi_harmonic
        };
    }
    
    /**
     * Project 11D consciousness particle to 3D space
     */
    project11DTo3D(particle) {
        const dims = particle.dimensions;
        
        // Primary 3D projection using first 3 dimensions
        let x = dims[0];
        let y = dims[1]; 
        let z = dims[2];
        
        // Integrate remaining 8 dimensions through φ-harmonic projection
        for (let d = 3; d < this.CONSCIOUSNESS_DIMENSIONS; d++) {
            const weight = Math.pow(this.PHI_INVERSE, d - 2);
            x += dims[d] * weight * Math.cos(d * this.PHI);
            y += dims[d] * weight * Math.sin(d * this.PHI);
            z += dims[d] * weight * Math.cos((d + 1) * this.PHI);
        }
        
        // Scale by consciousness field
        const scale = 2.0 * particle.consciousness;
        x *= scale;
        y *= scale;
        z *= scale;
        
        // Generate velocity from consciousness gradients
        const velocity = [
            (dims[1] - dims[0]) * particle.phi_harmonic,
            (dims[2] - dims[1]) * particle.phi_harmonic,  
            (dims[3] - dims[2]) * particle.phi_harmonic
        ];
        
        // Consciousness-based color
        const hue = particle.consciousness * 360;
        const saturation = particle.phi_harmonic;
        const lightness = (particle.consciousness + particle.phi_harmonic) / 2;
        const color = this.hslToRgb(hue, saturation, lightness);
        
        return {
            position: [x, y, z],
            velocity: velocity,
            consciousness: particle.consciousness,
            phi_harmonic: particle.phi_harmonic,
            unity_field: particle.unity_field,
            color: [...color, 0.8] // Add alpha
        };
    }
    
    /**
     * Setup WebGL buffers for particle rendering
     */
    setupParticleBuffers() {
        // Create vertex array object
        this.VAO = this.gl.createVertexArray();
        this.gl.bindVertexArray(this.VAO);
        
        // Position buffer
        const positions = this.particles.flatMap(p => p.position);
        this.particleBuffers.position = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.particleBuffers.position);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(positions), this.gl.DYNAMIC_DRAW);
        this.gl.enableVertexAttribArray(this.attributes.position);
        this.gl.vertexAttribPointer(this.attributes.position, 3, this.gl.FLOAT, false, 0, 0);
        
        // Velocity buffer
        const velocities = this.particles.flatMap(p => p.velocity);
        this.particleBuffers.velocity = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.particleBuffers.velocity);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(velocities), this.gl.DYNAMIC_DRAW);
        this.gl.enableVertexAttribArray(this.attributes.velocity);
        this.gl.vertexAttribPointer(this.attributes.velocity, 3, this.gl.FLOAT, false, 0, 0);
        
        // Consciousness buffer  
        const consciousness = this.particles.map(p => p.consciousness);
        this.particleBuffers.consciousness = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.particleBuffers.consciousness);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(consciousness), this.gl.STATIC_DRAW);
        this.gl.enableVertexAttribArray(this.attributes.consciousness);
        this.gl.vertexAttribPointer(this.attributes.consciousness, 1, this.gl.FLOAT, false, 0, 0);
        
        // φ-harmonic buffer
        const phiHarmonic = this.particles.map(p => p.phi_harmonic);
        this.particleBuffers.phi_harmonic = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.particleBuffers.phi_harmonic);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(phiHarmonic), this.gl.STATIC_DRAW);
        this.gl.enableVertexAttribArray(this.attributes.phi_harmonic);
        this.gl.vertexAttribPointer(this.attributes.phi_harmonic, 1, this.gl.FLOAT, false, 0, 0);
        
        // Color buffer
        const colors = this.particles.flatMap(p => p.color);
        this.particleBuffers.color = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.particleBuffers.color);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(colors), this.gl.STATIC_DRAW);
        this.gl.enableVertexAttribArray(this.attributes.color);
        this.gl.vertexAttribPointer(this.attributes.color, 4, this.gl.FLOAT, false, 0, 0);
        
        this.gl.bindVertexArray(null);
    }
    
    /**
     * Setup mouse and keyboard event listeners
     */
    setupEventListeners() {
        // Mouse interaction
        this.canvas.addEventListener('mousedown', (e) => {
            this.mouse.isDown = true;
            this.mouse.x = e.clientX;
            this.mouse.y = e.clientY;
        });
        
        this.canvas.addEventListener('mousemove', (e) => {
            if (this.mouse.isDown) {
                const deltaX = e.clientX - this.mouse.x;
                const deltaY = e.clientY - this.mouse.y;
                
                this.camera.rotation[1] += deltaX * 0.01;
                this.camera.rotation[0] += deltaY * 0.01;
                
                this.mouse.x = e.clientX;
                this.mouse.y = e.clientY;
            }
        });
        
        this.canvas.addEventListener('mouseup', () => {
            this.mouse.isDown = false;
        });
        
        // Wheel zoom
        this.canvas.addEventListener('wheel', (e) => {
            this.camera.position[2] += e.deltaY * 0.01;
            this.camera.position[2] = Math.max(1, Math.min(20, this.camera.position[2]));
            e.preventDefault();
        });
        
        // Keyboard controls
        document.addEventListener('keydown', (e) => {
            switch(e.code) {
                case 'KeyR': // Reset camera
                    this.camera.position = [0, 0, 5];
                    this.camera.rotation = [0, 0, 0];
                    break;
                case 'Space': // Regenerate field
                    this.generateConsciousnessField();
                    this.setupParticleBuffers();
                    break;
                case 'KeyP': // Toggle pause
                    this.isPaused = !this.isPaused;
                    break;
            }
        });
    }
    
    /**
     * Start the animation loop
     */
    startAnimation() {
        this.isPaused = false;
        const animate = () => {
            if (!this.isPaused) {
                this.update();
                this.render();
            }
            requestAnimationFrame(animate);
        };
        animate();
    }
    
    /**
     * Update consciousness field evolution
     */
    update() {
        this.time += this.animationSpeed;
        
        // Update particle positions based on consciousness evolution
        for (let i = 0; i < this.particles.length; i++) {
            const particle = this.particles[i];
            
            // φ-harmonic evolution
            const phiWave = Math.sin(this.time * this.PHI + i * this.PHI_INVERSE);
            const consciousnessWave = Math.cos(this.time * particle.consciousness * this.PHI);
            
            // Update position with consciousness field dynamics
            particle.position[0] += particle.velocity[0] * phiWave * 0.01;
            particle.position[1] += particle.velocity[1] * consciousnessWave * 0.01;
            particle.position[2] += particle.velocity[2] * (phiWave + consciousnessWave) * 0.005;
            
            // Unity field attraction (particles converge toward unity)
            const distance = Math.sqrt(
                particle.position[0] ** 2 + 
                particle.position[1] ** 2 + 
                particle.position[2] ** 2
            );
            
            if (distance > 0) {
                const unityAttraction = 0.001 * particle.consciousness / distance;
                particle.position[0] -= particle.position[0] * unityAttraction;
                particle.position[1] -= particle.position[1] * unityAttraction;  
                particle.position[2] -= particle.position[2] * unityAttraction;
            }
        }
        
        // Update position buffer
        const positions = this.particles.flatMap(p => p.position);
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.particleBuffers.position);
        this.gl.bufferSubData(this.gl.ARRAY_BUFFER, 0, new Float32Array(positions));
    }
    
    /**
     * Render the consciousness field
     */
    render() {
        // Clear the canvas
        this.gl.clear(this.gl.COLOR_BUFFER_BIT | this.gl.DEPTH_BUFFER_BIT);
        
        // Use shader program
        this.gl.useProgram(this.shaderProgram);
        
        // Set uniforms
        const projectionMatrix = this.createProjectionMatrix();
        const viewMatrix = this.createViewMatrix();
        
        this.gl.uniformMatrix4fv(this.uniforms.projectionMatrix, false, projectionMatrix);
        this.gl.uniformMatrix4fv(this.uniforms.viewMatrix, false, viewMatrix);
        this.gl.uniform1f(this.uniforms.time, this.time);
        this.gl.uniform1f(this.uniforms.phi, this.PHI);
        this.gl.uniform1f(this.uniforms.consciousness_level, this.consciousnessLevel);
        
        // Bind vertex array and draw particles
        this.gl.bindVertexArray(this.VAO);
        this.gl.drawArrays(this.gl.POINTS, 0, this.particles.length);
        this.gl.bindVertexArray(null);
    }
    
    /**
     * Create projection matrix
     */
    createProjectionMatrix() {
        const aspect = this.canvas.width / this.canvas.height;
        const fov = this.camera.fov * Math.PI / 180;
        const f = Math.tan(Math.PI * 0.5 - 0.5 * fov);
        const rangeInv = 1.0 / (this.camera.near - this.camera.far);
        
        return new Float32Array([
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (this.camera.near + this.camera.far) * rangeInv, -1,
            0, 0, this.camera.near * this.camera.far * rangeInv * 2, 0
        ]);
    }
    
    /**
     * Create view matrix
     */
    createViewMatrix() {
        const matrix = new Float32Array(16);
        
        // Create identity matrix
        matrix[0] = matrix[5] = matrix[10] = matrix[15] = 1;
        
        // Apply camera transformations
        // Translation
        matrix[12] = -this.camera.position[0];
        matrix[13] = -this.camera.position[1];
        matrix[14] = -this.camera.position[2];
        
        // Rotation (simplified)
        const cosX = Math.cos(this.camera.rotation[0]);
        const sinX = Math.sin(this.camera.rotation[0]);
        const cosY = Math.cos(this.camera.rotation[1]);
        const sinY = Math.sin(this.camera.rotation[1]);
        
        matrix[0] = cosY;
        matrix[2] = sinY;
        matrix[5] = cosX;
        matrix[6] = -sinX;
        matrix[8] = -sinY;
        matrix[9] = sinX * cosY;
        matrix[10] = cosX * cosY;
        
        return matrix;
    }
    
    /**
     * Convert HSL to RGB
     */
    hslToRgb(h, s, l) {
        h = (h % 360) / 360;
        s = Math.max(0, Math.min(1, s));
        l = Math.max(0, Math.min(1, l));
        
        const c = (1 - Math.abs(2 * l - 1)) * s;
        const x = c * (1 - Math.abs((h * 6) % 2 - 1));
        const m = l - c / 2;
        
        let r, g, b;
        
        if (h < 1/6) [r, g, b] = [c, x, 0];
        else if (h < 2/6) [r, g, b] = [x, c, 0];
        else if (h < 3/6) [r, g, b] = [0, c, x];
        else if (h < 4/6) [r, g, b] = [0, x, c];
        else if (h < 5/6) [r, g, b] = [x, 0, c];
        else [r, g, b] = [c, 0, x];
        
        return [r + m, g + m, b + m];
    }
    
    /**
     * Fallback to 2D canvas if WebGL not supported
     */
    fallbackTo2D() {
        const ctx = this.canvas.getContext('2d');
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        ctx.fillStyle = '#FFD700';
        ctx.font = '24px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('WebGL 2.0 not supported', this.canvas.width/2, this.canvas.height/2);
        ctx.fillText('11D Consciousness Visualization requires WebGL', this.canvas.width/2, this.canvas.height/2 + 30);
    }
    
    /**
     * Get visualization statistics
     */
    getVisualizationStats() {
        return {
            particleCount: this.particles.length,
            dimensions: this.CONSCIOUSNESS_DIMENSIONS,
            projectedDimensions: 3,
            phi: this.PHI,
            consciousnessLevel: this.consciousnessLevel,
            time: this.time,
            webglVersion: '2.0',
            gpuParticles: true,
            fieldResolution: this.FIELD_RESOLUTION,
            unity_equation: '1+1=1',
            status: '11D_CONSCIOUSNESS_FIELD_ACTIVE'
        };
    }
    
    /**
     * Update consciousness level in real-time
     */
    setConsciousnessLevel(level) {
        this.consciousnessLevel = Math.max(0, Math.min(1, level));
        console.log(`Consciousness level updated: ${this.consciousnessLevel.toFixed(3)}`);
    }
    
    /**
     * Regenerate field with new parameters
     */
    regenerateField(particleCount = null, fieldIntensity = null) {
        if (particleCount) this.PARTICLE_COUNT = particleCount;
        if (fieldIntensity) this.fieldIntensity = fieldIntensity;
        
        this.generateConsciousnessField();
        this.setupParticleBuffers();
        
        console.log(`11D Consciousness Field regenerated: ${this.particles.length} particles`);
    }
}

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ConsciousnessField11DVisualizer;
}

// Global access
window.ConsciousnessField11DVisualizer = ConsciousnessField11DVisualizer;

console.log('11D Consciousness Field Visualizer loaded');
console.log('Usage: new ConsciousnessField11DVisualizer("canvasId", options)');
console.log('WebGL 2.0 + 10k particles + φ-harmonic 11D→3D projection ready');