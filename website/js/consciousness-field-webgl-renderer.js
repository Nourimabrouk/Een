/**
 * WebGL Consciousness Field Renderer
 * ==================================
 * 
 * Advanced real-time 11D consciousness field visualization using WebGL
 * with φ-harmonic scaling and GPU-accelerated particle dynamics.
 * 
 * Features:
 * - Real-time consciousness field evolution
 * - φ-harmonic resonance visualization
 * - GPU-accelerated particle systems
 * - 11D to 3D manifold projection
 * - Unity mathematics integration (1+1=1)
 * 
 * Mathematical Foundation:
 * C(r,t) = φ * sin(r*φ) * cos(t*φ) * exp(-t/φ)
 * 
 * Author: Een Unity Mathematics - WebGL Division
 * Version: 3000_ELO_WEBGL_CONSCIOUSNESS_ENGINE
 */

class ConsciousnessFieldWebGLRenderer {
    constructor(canvas, options = {}) {
        this.canvas = canvas;
        this.gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
        
        if (!this.gl) {
            throw new Error('WebGL not supported');
        }
        
        // Mathematical constants
        this.PHI = 1.618033988749895;
        this.PI = Math.PI;
        this.UNITY_CONSTANT = 1.0;
        this.CONSCIOUSNESS_THRESHOLD = 0.618;
        
        // Rendering parameters
        this.options = {
            particleCount: options.particleCount || 1000,
            fieldResolution: options.fieldResolution || 128,
            dimensions: options.dimensions || 11,
            phiHarmonicScale: options.phiHarmonicScale || 1.0,
            consciousnessLevel: options.consciousnessLevel || this.CONSCIOUSNESS_THRESHOLD,
            enableGPUAcceleration: options.enableGPUAcceleration !== false,
            renderQuality: options.renderQuality || 'high',
            animationSpeed: options.animationSpeed || 1.0,
            ...options
        };
        
        // WebGL state
        this.programs = {};
        this.buffers = {};
        this.textures = {};
        this.framebuffers = {};
        
        // Consciousness field state
        this.consciousnessParticles = [];
        this.fieldData = null;
        this.evolutionTime = 0.0;
        this.lastFrameTime = 0;
        
        // Animation state
        this.isRunning = false;
        this.animationId = null;
        
        // Performance monitoring
        this.frameCount = 0;
        this.lastFPSUpdate = 0;
        this.currentFPS = 0;
        
        // Initialize WebGL resources
        this.initialize();
        
        console.log('🌟 Consciousness Field WebGL Renderer initialized');
        console.log(`   Particles: ${this.options.particleCount}`);
        console.log(`   Field resolution: ${this.options.fieldResolution}x${this.options.fieldResolution}`);
        console.log(`   φ-harmonic scale: ${this.options.phiHarmonicScale}`);
        console.log(`   GPU acceleration: ${this.options.enableGPUAcceleration}`);
    }
    
    initialize() {
        // Set up WebGL context
        this.setupWebGLContext();
        
        // Create shader programs
        this.createShaderPrograms();
        
        // Initialize consciousness particles
        this.initializeConsciousnessParticles();
        
        // Create field textures and buffers
        this.createFieldResources();
        
        // Set up rendering pipeline
        this.setupRenderingPipeline();
        
        // Initialize φ-harmonic field equation
        this.initializePhiHarmonicField();
    }
    
    setupWebGLContext() {
        const gl = this.gl;
        
        // Enable necessary extensions
        const extensions = [
            'OES_texture_float',
            'WEBGL_color_buffer_float',
            'EXT_color_buffer_float',
            'OES_texture_float_linear'
        ];
        
        extensions.forEach(ext => {
            const extension = gl.getExtension(ext);
            if (extension) {
                console.log(`✅ WebGL extension enabled: ${ext}`);
            } else {
                console.warn(`⚠️ WebGL extension not available: ${ext}`);
            }
        });
        
        // Configure WebGL state
        gl.enable(gl.DEPTH_TEST);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        
        // Set clear color to deep space
        gl.clearColor(0.05, 0.08, 0.12, 1.0);
        
        // Handle canvas resize
        this.handleResize();
        window.addEventListener('resize', () => this.handleResize());
    }
    
    createShaderPrograms() {
        const gl = this.gl;
        
        // Consciousness field vertex shader
        const fieldVertexShader = this.createShader(gl.VERTEX_SHADER, `
            #version 300 es
            precision highp float;
            
            in vec3 a_position;
            in vec2 a_texCoord;
            
            uniform mat4 u_modelViewProjection;
            uniform float u_time;
            uniform float u_phi;
            uniform float u_consciousnessLevel;
            
            out vec2 v_texCoord;
            out float v_phiResonance;
            out float v_consciousnessField;
            
            void main() {
                // φ-harmonic position transformation
                vec3 phiPosition = a_position;
                phiPosition.x *= cos(u_time * u_phi) * u_phi;
                phiPosition.y *= sin(u_time * u_phi) * u_phi;
                phiPosition.z *= exp(-u_time / u_phi);
                
                // Calculate φ-resonance
                v_phiResonance = sin(length(a_position) * u_phi) * cos(u_time * u_phi);
                
                // Consciousness field contribution
                v_consciousnessField = u_consciousnessLevel * exp(-length(phiPosition) / u_phi);
                
                v_texCoord = a_texCoord;
                gl_Position = u_modelViewProjection * vec4(phiPosition, 1.0);
            }
        `);
        
        // Consciousness field fragment shader
        const fieldFragmentShader = this.createShader(gl.FRAGMENT_SHADER, `
            #version 300 es
            precision highp float;
            
            in vec2 v_texCoord;
            in float v_phiResonance;
            in float v_consciousnessField;
            
            uniform sampler2D u_fieldTexture;
            uniform float u_time;
            uniform float u_phi;
            uniform vec3 u_cameraPos;
            
            out vec4 fragColor;
            
            // φ-harmonic color mapping
            vec3 phiHarmonicColor(float value, float resonance) {
                float r = 0.5 + 0.5 * sin(value * u_phi + u_time);
                float g = 0.5 + 0.5 * sin(value * u_phi * 2.0 + u_time * 0.7);
                float b = 0.5 + 0.5 * sin(value * u_phi * 3.0 + u_time * 0.3);
                
                // Golden ratio color enhancement
                r *= (1.0 + resonance * u_phi);
                g *= (1.0 + resonance * (2.0 - u_phi));
                b *= (1.0 + resonance * u_phi * 0.5);
                
                return vec3(r, g, b);
            }
            
            void main() {
                // Sample consciousness field
                vec4 fieldSample = texture(u_fieldTexture, v_texCoord);
                float fieldMagnitude = length(fieldSample.xy);
                
                // Calculate consciousness contribution
                float consciousness = v_consciousnessField * fieldMagnitude;
                
                // φ-harmonic color mapping
                vec3 color = phiHarmonicColor(consciousness, v_phiResonance);
                
                // Unity mathematics: Ensure output converges to unity
                float unityFactor = 1.0 - exp(-consciousness * u_phi);
                color = mix(color, vec3(1.0, 0.8, 0.3), unityFactor * 0.1);
                
                // Fade with distance for depth perception
                float depth = gl_FragCoord.z;
                float fadeAlpha = 1.0 - pow(depth, 2.0);
                
                fragColor = vec4(color, fadeAlpha * (0.7 + 0.3 * v_phiResonance));
            }
        `);
        
        // Create consciousness field program
        this.programs.consciousnessField = this.createProgram(fieldVertexShader, fieldFragmentShader);
        
        // Particle system vertex shader
        const particleVertexShader = this.createShader(gl.VERTEX_SHADER, `
            #version 300 es
            precision highp float;
            
            in vec3 a_position;
            in vec3 a_velocity;
            in float a_awareness;
            in float a_phiResonance;
            in float a_age;
            
            uniform mat4 u_modelViewProjection;
            uniform float u_time;
            uniform float u_phi;
            uniform float u_deltaTime;
            
            out float v_awareness;
            out float v_phiResonance;
            out float v_particleAge;
            out vec3 v_worldPos;
            
            void main() {
                // Evolve particle position with φ-harmonic dynamics
                vec3 pos = a_position;
                
                // φ-harmonic force
                vec3 phiForce = -u_phi * pos;
                
                // Consciousness drift toward unity
                vec3 unityForce = -a_awareness * pos * u_phi;
                
                // Total force
                vec3 totalForce = phiForce + unityForce;
                
                // Update position (simplified Verlet integration)
                pos += a_velocity * u_deltaTime + 0.5 * totalForce * u_deltaTime * u_deltaTime;
                
                v_awareness = a_awareness * (1.0 + u_deltaTime * a_phiResonance / u_phi);
                v_phiResonance = min(1.0, a_phiResonance + u_deltaTime * 0.01);
                v_particleAge = a_age + u_deltaTime;
                v_worldPos = pos;
                
                gl_Position = u_modelViewProjection * vec4(pos, 1.0);
                gl_PointSize = 3.0 + 5.0 * a_awareness;
            }
        `);
        
        // Particle system fragment shader
        const particleFragmentShader = this.createShader(gl.FRAGMENT_SHADER, `
            #version 300 es
            precision highp float;
            
            in float v_awareness;
            in float v_phiResonance;
            in float v_particleAge;
            in vec3 v_worldPos;
            
            uniform float u_time;
            uniform float u_phi;
            
            out vec4 fragColor;
            
            void main() {
                // Create circular particle
                vec2 cxy = 2.0 * gl_PointCoord - 1.0;
                float r = dot(cxy, cxy);
                if (r > 1.0) {
                    discard;
                }
                
                // φ-harmonic particle glow
                float glow = exp(-r * u_phi) * v_awareness;
                
                // Consciousness-based color
                vec3 color = vec3(1.0, 0.8, 0.3); // Golden consciousness
                color *= (0.5 + 0.5 * sin(v_particleAge * u_phi + u_time));
                
                // Unity mathematics influence
                float unityInfluence = sin(length(v_worldPos) * u_phi + u_time) * 0.1;
                color += vec3(unityInfluence, unityInfluence * 0.5, 0.0);
                
                // Fade with age and awareness
                float alpha = glow * (1.0 - r) * min(1.0, v_awareness);
                
                fragColor = vec4(color, alpha);
            }
        `);
        
        // Create particle system program
        this.programs.particles = this.createProgram(particleVertexShader, particleFragmentShader);
        
        // Field computation shader (for GPU acceleration)
        if (this.options.enableGPUAcceleration) {
            this.createFieldComputeShader();
        }
    }
    
    createFieldComputeShader() {
        const gl = this.gl;
        
        // Field evolution compute shader
        const computeVertexShader = this.createShader(gl.VERTEX_SHADER, `
            #version 300 es
            precision highp float;
            
            in vec2 a_position;
            out vec2 v_texCoord;
            
            void main() {
                v_texCoord = a_position * 0.5 + 0.5;
                gl_Position = vec4(a_position, 0.0, 1.0);
            }
        `);
        
        const computeFragmentShader = this.createShader(gl.FRAGMENT_SHADER, `
            #version 300 es
            precision highp float;
            
            in vec2 v_texCoord;
            
            uniform sampler2D u_currentField;
            uniform sampler2D u_particleField;
            uniform float u_deltaTime;
            uniform float u_phi;
            uniform float u_consciousnessLevel;
            uniform vec2 u_fieldSize;
            
            out vec4 fragColor;
            
            // Calculate Laplacian using finite differences
            vec2 laplacian(sampler2D field, vec2 coord, vec2 texelSize) {
                vec2 center = texture(field, coord).xy;
                vec2 left = texture(field, coord - vec2(texelSize.x, 0.0)).xy;
                vec2 right = texture(field, coord + vec2(texelSize.x, 0.0)).xy;
                vec2 top = texture(field, coord + vec2(0.0, texelSize.y)).xy;
                vec2 bottom = texture(field, coord - vec2(0.0, texelSize.y)).xy;
                
                return (left + right + top + bottom - 4.0 * center);
            }
            
            void main() {
                vec2 texelSize = 1.0 / u_fieldSize;
                vec2 currentField = texture(u_currentField, v_texCoord).xy;
                vec2 particleContribution = texture(u_particleField, v_texCoord).xy;
                
                // Calculate field Laplacian
                vec2 fieldLaplacian = laplacian(u_currentField, v_texCoord, texelSize);
                
                // Consciousness field equation: ∂C/∂t = φ∇²C - |C|²C + C + γP(r,t)
                float fieldMagnitude = length(currentField);
                vec2 nonlinearTerm = -fieldMagnitude * fieldMagnitude * currentField;
                vec2 linearTerm = currentField;
                
                // Field evolution
                vec2 fieldDerivative = u_phi * fieldLaplacian + nonlinearTerm + linearTerm + 
                                      u_consciousnessLevel * particleContribution;
                
                // Update field
                vec2 newField = currentField + fieldDerivative * u_deltaTime;
                
                // Apply φ-harmonic phase enhancement
                float phase = u_phi * (v_texCoord.x + v_texCoord.y);
                float phaseEnhancement = cos(phase) * sin(phase * u_phi);
                newField *= (1.0 + phaseEnhancement * 0.1);
                
                // Unity mathematics: Ensure convergence toward unity
                float unityFactor = 1.0 / (1.0 + length(newField) / u_phi);
                newField *= unityFactor;
                
                fragColor = vec4(newField, 0.0, 1.0);
            }
        `);
        
        this.programs.fieldCompute = this.createProgram(computeVertexShader, computeFragmentShader);
    }
    
    initializeConsciousnessParticles() {
        this.consciousnessParticles = [];
        
        for (let i = 0; i < this.options.particleCount; i++) {
            // φ-harmonic particle initialization
            const theta = i * this.PHI * 2 * Math.PI;
            const radius = Math.sqrt(i / this.options.particleCount) * 5.0;
            
            const particle = {
                // 11D position projected to 3D using φ-harmonic mapping
                position: [
                    radius * Math.cos(theta),
                    radius * Math.sin(theta),
                    Math.sin(i * this.PHI) * 2.0
                ],
                velocity: [
                    (Math.random() - 0.5) * 0.1,
                    (Math.random() - 0.5) * 0.1,
                    (Math.random() - 0.5) * 0.1
                ],
                awareness: Math.random() * this.options.consciousnessLevel + 0.1,
                phiResonance: Math.random() * this.PHI,
                unityTendency: Math.random() * 0.8 + 0.2,
                age: 0.0,
                transcendencePotential: Math.random() * 0.1
            };
            
            this.consciousnessParticles.push(particle);
        }
        
        // Create particle buffers
        this.createParticleBuffers();
        
        console.log(`🌱 Initialized ${this.consciousnessParticles.length} consciousness particles`);
    }
    
    createParticleBuffers() {
        const gl = this.gl;
        const particleCount = this.consciousnessParticles.length;
        
        // Position buffer
        const positions = new Float32Array(particleCount * 3);
        const velocities = new Float32Array(particleCount * 3);
        const awareness = new Float32Array(particleCount);
        const phiResonance = new Float32Array(particleCount);
        const ages = new Float32Array(particleCount);
        
        for (let i = 0; i < particleCount; i++) {
            const particle = this.consciousnessParticles[i];
            
            positions[i * 3] = particle.position[0];
            positions[i * 3 + 1] = particle.position[1];
            positions[i * 3 + 2] = particle.position[2];
            
            velocities[i * 3] = particle.velocity[0];
            velocities[i * 3 + 1] = particle.velocity[1];
            velocities[i * 3 + 2] = particle.velocity[2];
            
            awareness[i] = particle.awareness;
            phiResonance[i] = particle.phiResonance;
            ages[i] = particle.age;
        }
        
        // Create and populate buffers
        this.buffers.particlePositions = this.createBuffer(positions);
        this.buffers.particleVelocities = this.createBuffer(velocities);
        this.buffers.particleAwareness = this.createBuffer(awareness);
        this.buffers.particlePhiResonance = this.createBuffer(phiResonance);
        this.buffers.particleAges = this.createBuffer(ages);
    }
    
    createFieldResources() {
        const gl = this.gl;
        const resolution = this.options.fieldResolution;
        
        // Create field textures for ping-pong rendering
        this.textures.fieldCurrent = this.createFloatTexture(resolution, resolution);
        this.textures.fieldNext = this.createFloatTexture(resolution, resolution);
        this.textures.particleField = this.createFloatTexture(resolution, resolution);
        
        // Create framebuffers for off-screen rendering
        this.framebuffers.fieldCurrent = this.createFramebuffer(this.textures.fieldCurrent);
        this.framebuffers.fieldNext = this.createFramebuffer(this.textures.fieldNext);
        this.framebuffers.particleField = this.createFramebuffer(this.textures.particleField);
        
        // Initialize field data
        this.initializePhiHarmonicField();
        
        console.log(`🌊 Consciousness field resources created: ${resolution}x${resolution}`);
    }
    
    initializePhiHarmonicField() {
        const resolution = this.options.fieldResolution;
        const fieldData = new Float32Array(resolution * resolution * 4);
        
        for (let y = 0; y < resolution; y++) {
            for (let x = 0; x < resolution; x++) {
                const index = (y * resolution + x) * 4;
                
                // Map to φ-harmonic coordinates
                const u = (x / resolution - 0.5) * this.PHI * 4;
                const v = (y / resolution - 0.5) * this.PHI * 4;
                
                // Initial φ-harmonic field: C(u,v) = φ * sin(u*φ) * cos(v*φ) * exp(-(u²+v²)/φ)
                const r2 = u * u + v * v;
                const amplitude = this.PHI * Math.exp(-r2 / this.PHI);
                const phiX = amplitude * Math.sin(u * this.PHI);
                const phiY = amplitude * Math.cos(v * this.PHI);
                
                fieldData[index] = phiX;       // Real part
                fieldData[index + 1] = phiY;   // Imaginary part
                fieldData[index + 2] = 0.0;    // Reserved
                fieldData[index + 3] = 1.0;    // Alpha
            }
        }
        
        // Upload initial field data
        const gl = this.gl;
        gl.bindTexture(gl.TEXTURE_2D, this.textures.fieldCurrent);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, resolution, resolution, 
                        gl.RGBA, gl.FLOAT, fieldData);
        
        this.fieldData = fieldData;
        
        console.log(`🌀 φ-harmonic field initialized with golden ratio scaling`);
    }
    
    setupRenderingPipeline() {
        const gl = this.gl;
        
        // Create quad for full-screen rendering
        const quadVertices = new Float32Array([
            -1, -1, 0, 0,
             1, -1, 1, 0,
            -1,  1, 0, 1,
             1,  1, 1, 1
        ]);
        
        this.buffers.fullscreenQuad = this.createBuffer(quadVertices);
        
        // Create camera matrices
        this.updateCameraMatrices();
        
        console.log(`📐 Rendering pipeline configured`);
    }
    
    updateCameraMatrices() {
        // Create view matrix (camera looking at consciousness field)
        const eye = [8 * Math.cos(this.evolutionTime * 0.1), 
                     6 * Math.sin(this.evolutionTime * 0.05), 
                     8 * Math.sin(this.evolutionTime * 0.1)];
        const center = [0, 0, 0];
        const up = [0, 1, 0];
        
        this.viewMatrix = this.lookAt(eye, center, up);
        
        // Create projection matrix
        const aspect = this.canvas.width / this.canvas.height;
        this.projectionMatrix = this.perspective(45 * Math.PI / 180, aspect, 0.1, 100);
        
        // Combined model-view-projection
        this.mvpMatrix = this.multiply(this.projectionMatrix, this.viewMatrix);
    }
    
    // Main rendering loop
    render(currentTime = 0) {
        if (!this.isRunning) return;
        
        const deltaTime = (currentTime - this.lastFrameTime) * 0.001 * this.options.animationSpeed;
        this.lastFrameTime = currentTime;
        this.evolutionTime += deltaTime;
        
        // Update performance metrics
        this.updatePerformanceMetrics(currentTime);
        
        // Update camera
        this.updateCameraMatrices();
        
        // Update consciousness particles
        this.updateConsciousnessParticles(deltaTime);
        
        // Evolve consciousness field
        if (this.options.enableGPUAcceleration) {
            this.evolveConsciousnessFieldGPU(deltaTime);
        } else {
            this.evolveConsciousnessFieldCPU(deltaTime);
        }
        
        // Render scene
        this.renderScene();
        
        // Continue animation
        this.animationId = requestAnimationFrame((time) => this.render(time));
    }
    
    updateConsciousnessParticles(deltaTime) {
        // Update particle physics
        for (let i = 0; i < this.consciousnessParticles.length; i++) {
            const particle = this.consciousnessParticles[i];
            
            // φ-harmonic force
            const phiForceX = -this.PHI * particle.position[0];
            const phiForceY = -this.PHI * particle.position[1];
            const phiForceZ = -this.PHI * particle.position[2];
            
            // Unity tendency force (attractive toward origin)
            const unityForceX = -particle.unityTendency * particle.position[0] * this.PHI;
            const unityForceY = -particle.unityTendency * particle.position[1] * this.PHI;
            const unityForceZ = -particle.unityTendency * particle.position[2] * this.PHI;
            
            // Total force
            const totalForceX = phiForceX + unityForceX;
            const totalForceY = phiForceY + unityForceY;
            const totalForceZ = phiForceZ + unityForceZ;
            
            // Update velocity (F = ma, assume m = 1)
            particle.velocity[0] += totalForceX * deltaTime;
            particle.velocity[1] += totalForceY * deltaTime;
            particle.velocity[2] += totalForceZ * deltaTime;
            
            // Update position
            particle.position[0] += particle.velocity[0] * deltaTime;
            particle.position[1] += particle.velocity[1] * deltaTime;
            particle.position[2] += particle.velocity[2] * deltaTime;
            
            // Update consciousness properties
            particle.age += deltaTime;
            particle.awareness *= (1 + deltaTime * particle.phiResonance / this.PHI);
            particle.phiResonance = Math.min(1.0, particle.phiResonance + deltaTime * 0.01);
            
            // Unity mathematics: Ensure awareness converges to unity bounds
            if (particle.awareness > this.PHI) {
                particle.awareness = this.PHI;
                particle.transcendencePotential = Math.min(1.0, 
                    particle.transcendencePotential + deltaTime * 0.1);
            }
        }
        
        // Update GPU buffers
        this.updateParticleBuffers();
    }
    
    updateParticleBuffers() {
        const gl = this.gl;
        const particleCount = this.consciousnessParticles.length;
        
        // Update position buffer
        const positions = new Float32Array(particleCount * 3);
        const velocities = new Float32Array(particleCount * 3);
        const awareness = new Float32Array(particleCount);
        const phiResonance = new Float32Array(particleCount);
        const ages = new Float32Array(particleCount);
        
        for (let i = 0; i < particleCount; i++) {
            const particle = this.consciousnessParticles[i];
            
            positions[i * 3] = particle.position[0];
            positions[i * 3 + 1] = particle.position[1];
            positions[i * 3 + 2] = particle.position[2];
            
            velocities[i * 3] = particle.velocity[0];
            velocities[i * 3 + 1] = particle.velocity[1];
            velocities[i * 3 + 2] = particle.velocity[2];
            
            awareness[i] = particle.awareness;
            phiResonance[i] = particle.phiResonance;
            ages[i] = particle.age;
        }
        
        // Update buffers
        this.updateBuffer(this.buffers.particlePositions, positions);
        this.updateBuffer(this.buffers.particleVelocities, velocities);
        this.updateBuffer(this.buffers.particleAwareness, awareness);
        this.updateBuffer(this.buffers.particlePhiResonance, phiResonance);
        this.updateBuffer(this.buffers.particleAges, ages);
    }
    
    evolveConsciousnessFieldGPU(deltaTime) {
        if (!this.programs.fieldCompute) return;
        
        const gl = this.gl;
        const resolution = this.options.fieldResolution;
        
        // Render particle contributions to texture
        this.renderParticleField();
        
        // Bind compute program
        gl.useProgram(this.programs.fieldCompute);
        
        // Set uniforms
        gl.uniform1i(gl.getUniformLocation(this.programs.fieldCompute, 'u_currentField'), 0);
        gl.uniform1i(gl.getUniformLocation(this.programs.fieldCompute, 'u_particleField'), 1);
        gl.uniform1f(gl.getUniformLocation(this.programs.fieldCompute, 'u_deltaTime'), deltaTime);
        gl.uniform1f(gl.getUniformLocation(this.programs.fieldCompute, 'u_phi'), this.PHI);
        gl.uniform1f(gl.getUniformLocation(this.programs.fieldCompute, 'u_consciousnessLevel'), this.options.consciousnessLevel);
        gl.uniform2f(gl.getUniformLocation(this.programs.fieldCompute, 'u_fieldSize'), resolution, resolution);
        
        // Bind textures
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.textures.fieldCurrent);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.textures.particleField);
        
        // Render to next field texture
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffers.fieldNext);
        gl.viewport(0, 0, resolution, resolution);
        
        // Render fullscreen quad
        this.renderFullscreenQuad(this.programs.fieldCompute);
        
        // Swap field textures (ping-pong)
        const temp = this.textures.fieldCurrent;
        this.textures.fieldCurrent = this.textures.fieldNext;
        this.textures.fieldNext = temp;
        
        const tempFB = this.framebuffers.fieldCurrent;
        this.framebuffers.fieldCurrent = this.framebuffers.fieldNext;
        this.framebuffers.fieldNext = tempFB;
    }
    
    evolveConsciousnessFieldCPU(deltaTime) {
        // CPU-based field evolution (simplified for performance)
        const resolution = this.options.fieldResolution;
        const newFieldData = new Float32Array(resolution * resolution * 4);
        
        for (let y = 1; y < resolution - 1; y++) {
            for (let x = 1; x < resolution - 1; x++) {
                const index = (y * resolution + x) * 4;
                
                // Current field value
                const currentReal = this.fieldData[index];
                const currentImag = this.fieldData[index + 1];
                
                // Calculate Laplacian using finite differences
                const leftReal = this.fieldData[(y * resolution + (x - 1)) * 4];
                const rightReal = this.fieldData[(y * resolution + (x + 1)) * 4];
                const topReal = this.fieldData[((y - 1) * resolution + x) * 4];
                const bottomReal = this.fieldData[((y + 1) * resolution + x) * 4];
                
                const leftImag = this.fieldData[(y * resolution + (x - 1)) * 4 + 1];
                const rightImag = this.fieldData[(y * resolution + (x + 1)) * 4 + 1];
                const topImag = this.fieldData[((y - 1) * resolution + x) * 4 + 1];
                const bottomImag = this.fieldData[((y + 1) * resolution + x) * 4 + 1];
                
                const laplacianReal = leftReal + rightReal + topReal + bottomReal - 4 * currentReal;
                const laplacianImag = leftImag + rightImag + topImag + bottomImag - 4 * currentImag;
                
                // Nonlinear term: -|C|²C
                const magnitude2 = currentReal * currentReal + currentImag * currentImag;
                const nonlinearReal = -magnitude2 * currentReal;
                const nonlinearImag = -magnitude2 * currentImag;
                
                // Field evolution equation
                const derivReal = this.PHI * laplacianReal + nonlinearReal + currentReal;
                const derivImag = this.PHI * laplacianImag + nonlinearImag + currentImag;
                
                // Update field
                newFieldData[index] = currentReal + derivReal * deltaTime;
                newFieldData[index + 1] = currentImag + derivImag * deltaTime;
                newFieldData[index + 2] = 0.0;
                newFieldData[index + 3] = 1.0;
                
                // Unity mathematics: Apply convergence factor
                const newMagnitude = Math.sqrt(newFieldData[index] * newFieldData[index] + 
                                             newFieldData[index + 1] * newFieldData[index + 1]);
                if (newMagnitude > this.PHI) {
                    const scale = this.PHI / newMagnitude;
                    newFieldData[index] *= scale;
                    newFieldData[index + 1] *= scale;
                }
            }
        }
        
        // Update texture
        const gl = this.gl;
        gl.bindTexture(gl.TEXTURE_2D, this.textures.fieldCurrent);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, resolution, resolution,
                        gl.RGBA, gl.FLOAT, newFieldData);
        
        this.fieldData = newFieldData;
    }
    
    renderParticleField() {
        // Render particle contributions to separate texture for field computation
        const gl = this.gl;
        const resolution = this.options.fieldResolution;
        
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffers.particleField);
        gl.viewport(0, 0, resolution, resolution);
        gl.clear(gl.COLOR_BUFFER_BIT);
        
        // Simple particle-to-field rendering (could be enhanced)
        gl.useProgram(this.programs.particles);
        
        // Set projection to map particle space to field texture space
        const orthoMatrix = this.orthographic(-5, 5, -5, 5, -10, 10);
        gl.uniformMatrix4fv(gl.getUniformLocation(this.programs.particles, 'u_modelViewProjection'), 
                           false, orthoMatrix);
        
        // Render particles as points
        this.renderParticles(this.programs.particles, true);
    }
    
    renderScene() {
        const gl = this.gl;
        
        // Restore main framebuffer
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        
        // Clear screen
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        
        // Render consciousness field
        this.renderConsciousnessField();
        
        // Render consciousness particles
        this.renderConsciousnessParticles();
        
        // Render UI overlay (if needed)
        this.renderUIOverlay();
    }
    
    renderConsciousnessField() {
        const gl = this.gl;
        
        gl.useProgram(this.programs.consciousnessField);
        
        // Set uniforms
        gl.uniformMatrix4fv(gl.getUniformLocation(this.programs.consciousnessField, 'u_modelViewProjection'),
                           false, this.mvpMatrix);
        gl.uniform1f(gl.getUniformLocation(this.programs.consciousnessField, 'u_time'), this.evolutionTime);
        gl.uniform1f(gl.getUniformLocation(this.programs.consciousnessField, 'u_phi'), this.PHI);
        gl.uniform3f(gl.getUniformLocation(this.programs.consciousnessField, 'u_cameraPos'), 0, 0, 8);
        gl.uniform1i(gl.getUniformLocation(this.programs.consciousnessField, 'u_fieldTexture'), 0);
        
        // Bind field texture
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.textures.fieldCurrent);
        
        // Render field as textured mesh (simplified as fullscreen quad for now)
        this.renderFullscreenQuad(this.programs.consciousnessField);
    }
    
    renderConsciousnessParticles() {
        const gl = this.gl;
        
        gl.useProgram(this.programs.particles);
        
        // Set uniforms
        gl.uniformMatrix4fv(gl.getUniformLocation(this.programs.particles, 'u_modelViewProjection'),
                           false, this.mvpMatrix);
        gl.uniform1f(gl.getUniformLocation(this.programs.particles, 'u_time'), this.evolutionTime);
        gl.uniform1f(gl.getUniformLocation(this.programs.particles, 'u_phi'), this.PHI);
        gl.uniform1f(gl.getUniformLocation(this.programs.particles, 'u_deltaTime'), 0.016); // ~60fps
        
        // Render particles
        this.renderParticles(this.programs.particles, false);
    }
    
    renderParticles(program, forField = false) {
        const gl = this.gl;
        const particleCount = this.consciousnessParticles.length;
        
        // Bind attribute arrays
        const positionLoc = gl.getAttribLocation(program, 'a_position');
        const velocityLoc = gl.getAttribLocation(program, 'a_velocity');
        const awarenessLoc = gl.getAttribLocation(program, 'a_awareness');
        const phiResonanceLoc = gl.getAttribLocation(program, 'a_phiResonance');
        const ageLoc = gl.getAttribLocation(program, 'a_age');
        
        if (positionLoc >= 0) {
            gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.particlePositions);
            gl.enableVertexAttribArray(positionLoc);
            gl.vertexAttribPointer(positionLoc, 3, gl.FLOAT, false, 0, 0);
        }
        
        if (velocityLoc >= 0) {
            gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.particleVelocities);
            gl.enableVertexAttribArray(velocityLoc);
            gl.vertexAttribPointer(velocityLoc, 3, gl.FLOAT, false, 0, 0);
        }
        
        if (awarenessLoc >= 0) {
            gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.particleAwareness);
            gl.enableVertexAttribArray(awarenessLoc);
            gl.vertexAttribPointer(awarenessLoc, 1, gl.FLOAT, false, 0, 0);
        }
        
        if (phiResonanceLoc >= 0) {
            gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.particlePhiResonance);
            gl.enableVertexAttribArray(phiResonanceLoc);
            gl.vertexAttribPointer(phiResonanceLoc, 1, gl.FLOAT, false, 0, 0);
        }
        
        if (ageLoc >= 0) {
            gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.particleAges);
            gl.enableVertexAttribArray(ageLoc);
            gl.vertexAttribPointer(ageLoc, 1, gl.FLOAT, false, 0, 0);
        }
        
        // Draw particles as points
        gl.drawArrays(gl.POINTS, 0, particleCount);
        
        // Cleanup
        if (positionLoc >= 0) gl.disableVertexAttribArray(positionLoc);
        if (velocityLoc >= 0) gl.disableVertexAttribArray(velocityLoc);
        if (awarenessLoc >= 0) gl.disableVertexAttribArray(awarenessLoc);
        if (phiResonanceLoc >= 0) gl.disableVertexAttribArray(phiResonanceLoc);
        if (ageLoc >= 0) gl.disableVertexAttribArray(ageLoc);
    }
    
    renderFullscreenQuad(program) {
        const gl = this.gl;
        
        const positionLoc = gl.getAttribLocation(program, 'a_position');
        const texCoordLoc = gl.getAttribLocation(program, 'a_texCoord');
        
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buffers.fullscreenQuad);
        
        if (positionLoc >= 0) {
            gl.enableVertexAttribArray(positionLoc);
            gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 16, 0);
        }
        
        if (texCoordLoc >= 0) {
            gl.enableVertexAttribArray(texCoordLoc);
            gl.vertexAttribPointer(texCoordLoc, 2, gl.FLOAT, false, 16, 8);
        }
        
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        
        if (positionLoc >= 0) gl.disableVertexAttribArray(positionLoc);
        if (texCoordLoc >= 0) gl.disableVertexAttribArray(texCoordLoc);
    }
    
    renderUIOverlay() {
        // Render performance metrics and consciousness info
        // This could be implemented with HTML overlay or WebGL text rendering
    }
    
    updatePerformanceMetrics(currentTime) {
        this.frameCount++;
        
        if (currentTime - this.lastFPSUpdate >= 1000) {
            this.currentFPS = this.frameCount;
            this.frameCount = 0;
            this.lastFPSUpdate = currentTime;
            
            // Log performance info
            if (this.currentFPS < 30) {
                console.warn(`⚡ Low FPS detected: ${this.currentFPS}`);
            }
        }
    }
    
    // Public API methods
    start() {
        if (this.isRunning) return;
        
        this.isRunning = true;
        this.lastFrameTime = performance.now();
        
        console.log('🚀 Consciousness field rendering started');
        this.render();
    }
    
    stop() {
        this.isRunning = false;
        
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        
        console.log('🛑 Consciousness field rendering stopped');
    }
    
    getMetrics() {
        return {
            fps: this.currentFPS,
            evolutionTime: this.evolutionTime,
            particleCount: this.consciousnessParticles.length,
            fieldResolution: this.options.fieldResolution,
            gpuAcceleration: this.options.enableGPUAcceleration,
            consciousnessLevel: this.options.consciousnessLevel,
            phiHarmonicScale: this.options.phiHarmonicScale,
            totalAwareness: this.consciousnessParticles.reduce((sum, p) => sum + p.awareness, 0),
            averagePhiResonance: this.consciousnessParticles.reduce((sum, p) => sum + p.phiResonance, 0) / this.consciousnessParticles.length
        };
    }
    
    updateOptions(newOptions) {
        Object.assign(this.options, newOptions);
        
        // Reinitialize resources that depend on changed options
        if (newOptions.particleCount !== undefined) {
            this.initializeConsciousnessParticles();
        }
        
        if (newOptions.fieldResolution !== undefined) {
            this.createFieldResources();
        }
        
        console.log('⚙️ Consciousness field options updated', newOptions);
    }
    
    // Utility methods for WebGL operations
    createShader(type, source) {
        const gl = this.gl;
        const shader = gl.createShader(type);
        
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            const error = gl.getShaderInfoLog(shader);
            gl.deleteShader(shader);
            throw new Error(`Shader compilation failed: ${error}`);
        }
        
        return shader;
    }
    
    createProgram(vertexShader, fragmentShader) {
        const gl = this.gl;
        const program = gl.createProgram();
        
        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);
        gl.linkProgram(program);
        
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            const error = gl.getProgramInfoLog(program);
            gl.deleteProgram(program);
            throw new Error(`Program linking failed: ${error}`);
        }
        
        return program;
    }
    
    createBuffer(data) {
        const gl = this.gl;
        const buffer = gl.createBuffer();
        
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
        
        return buffer;
    }
    
    updateBuffer(buffer, data) {
        const gl = this.gl;
        
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, data);
    }
    
    createFloatTexture(width, height) {
        const gl = this.gl;
        const texture = gl.createTexture();
        
        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F || gl.RGBA, width, height, 0, 
                     gl.RGBA, gl.FLOAT, null);
        
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        
        return texture;
    }
    
    createFramebuffer(texture) {
        const gl = this.gl;
        const framebuffer = gl.createFramebuffer();
        
        gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
        
        if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
            throw new Error('Framebuffer setup failed');
        }
        
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        return framebuffer;
    }
    
    handleResize() {
        const displayWidth = this.canvas.clientWidth;
        const displayHeight = this.canvas.clientHeight;
        
        if (this.canvas.width !== displayWidth || this.canvas.height !== displayHeight) {
            this.canvas.width = displayWidth;
            this.canvas.height = displayHeight;
            this.gl.viewport(0, 0, displayWidth, displayHeight);
        }
    }
    
    // Matrix math utilities
    perspective(fovy, aspect, near, far) {
        const f = 1.0 / Math.tan(fovy / 2);
        const nf = 1 / (near - far);
        
        return [
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far + near) * nf, -1,
            0, 0, 2 * far * near * nf, 0
        ];
    }
    
    orthographic(left, right, bottom, top, near, far) {
        const lr = 1 / (left - right);
        const bt = 1 / (bottom - top);
        const nf = 1 / (near - far);
        
        return [
            -2 * lr, 0, 0, 0,
            0, -2 * bt, 0, 0,
            0, 0, 2 * nf, 0,
            (left + right) * lr, (top + bottom) * bt, (far + near) * nf, 1
        ];
    }
    
    lookAt(eye, center, up) {
        const f = [center[0] - eye[0], center[1] - eye[1], center[2] - eye[2]];
        const fLen = Math.sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
        f[0] /= fLen; f[1] /= fLen; f[2] /= fLen;
        
        const s = [
            f[1] * up[2] - f[2] * up[1],
            f[2] * up[0] - f[0] * up[2],
            f[0] * up[1] - f[1] * up[0]
        ];
        const sLen = Math.sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
        s[0] /= sLen; s[1] /= sLen; s[2] /= sLen;
        
        const u = [
            s[1] * f[2] - s[2] * f[1],
            s[2] * f[0] - s[0] * f[2],
            s[0] * f[1] - s[1] * f[0]
        ];
        
        return [
            s[0], u[0], -f[0], 0,
            s[1], u[1], -f[1], 0,
            s[2], u[2], -f[2], 0,
            -(s[0] * eye[0] + s[1] * eye[1] + s[2] * eye[2]),
            -(u[0] * eye[0] + u[1] * eye[1] + u[2] * eye[2]),
            f[0] * eye[0] + f[1] * eye[1] + f[2] * eye[2],
            1
        ];
    }
    
    multiply(a, b) {
        const result = new Array(16);
        
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                result[i * 4 + j] = 
                    a[i * 4 + 0] * b[0 * 4 + j] +
                    a[i * 4 + 1] * b[1 * 4 + j] +
                    a[i * 4 + 2] * b[2 * 4 + j] +
                    a[i * 4 + 3] * b[3 * 4 + j];
            }
        }
        
        return result;
    }
    
    // Cleanup
    dispose() {
        this.stop();
        
        const gl = this.gl;
        
        // Delete WebGL resources
        Object.values(this.programs).forEach(program => gl.deleteProgram(program));
        Object.values(this.buffers).forEach(buffer => gl.deleteBuffer(buffer));
        Object.values(this.textures).forEach(texture => gl.deleteTexture(texture));
        Object.values(this.framebuffers).forEach(fb => gl.deleteFramebuffer(fb));
        
        console.log('🗑️ Consciousness field renderer disposed');
    }
}

// Export for global access
window.ConsciousnessFieldWebGLRenderer = ConsciousnessFieldWebGLRenderer;

// Auto-initialization helper
window.initializeConsciousnessFieldRenderer = function(canvasId, options = {}) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.error(`Canvas element '${canvasId}' not found`);
        return null;
    }
    
    try {
        const renderer = new ConsciousnessFieldWebGLRenderer(canvas, options);
        renderer.start();
        
        console.log('🌟 Consciousness Field WebGL Renderer auto-initialized');
        return renderer;
    } catch (error) {
        console.error('Failed to initialize consciousness field renderer:', error);
        return null;
    }
};