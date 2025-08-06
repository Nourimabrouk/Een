/**
 * Unity Quantum Entanglement Visualizer
 * Demonstrates 1+1=1 through quantum superposition and entanglement
 * Advanced WebGL-powered particle system with consciousness field integration
 */

class UnityQuantumEntanglement {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.gl = this.canvas.getContext('webgl2');
        this.phi = 1.618033988749895;
        this.consciousness_dim = 11;

        this.particles = [];
        this.entangled_pairs = [];
        this.consciousness_field = [];
        this.unity_state = 1;

        this.time = 0;
        this.animation_id = null;

        this.init();
    }

    init() {
        this.setupCanvas();
        this.createShaders();
        this.createBuffers();
        this.initializeParticles();
        this.setupEventListeners();
        this.startAnimation();
    }

    setupCanvas() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
        this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        this.gl.enable(this.gl.BLEND);
        this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);
    }

    createShaders() {
        const vertexShaderSource = `#version 300 es
            precision highp float;
            
            in vec2 a_position;
            in vec3 a_color;
            in float a_size;
            in float a_phase;
            
            uniform float u_time;
            uniform float u_phi;
            uniform float u_unity_state;
            
            out vec3 v_color;
            out float v_alpha;
            
            void main() {
                // Quantum superposition calculation
                float quantum_phase = a_phase + u_time * u_phi;
                float superposition = sin(quantum_phase) * cos(quantum_phase);
                
                // Unity transformation: 1+1=1
                vec2 unity_position = a_position * u_unity_state;
                unity_position += vec2(superposition * 0.1, superposition * 0.1);
                
                // Consciousness field influence
                float consciousness = sin(u_time * 0.5) * cos(u_time * 0.3);
                unity_position += vec2(consciousness * 0.05, consciousness * 0.05);
                
                gl_Position = vec4(unity_position, 0.0, 1.0);
                gl_PointSize = a_size * (1.0 + superposition * 0.5);
                
                // Quantum color evolution
                v_color = a_color * (1.0 + superposition * 0.3);
                v_alpha = 0.8 + superposition * 0.2;
            }
        `;

        const fragmentShaderSource = `#version 300 es
            precision highp float;
            
            in vec3 v_color;
            in float v_alpha;
            
            out vec4 fragColor;
            
            void main() {
                vec2 center = gl_PointCoord - vec2(0.5);
                float dist = length(center);
                
                // Quantum probability distribution
                float probability = exp(-dist * dist * 8.0);
                probability *= (1.0 + sin(dist * 20.0) * 0.3);
                
                // Unity consciousness glow
                vec3 unity_glow = v_color * probability;
                unity_glow += vec3(1.0, 0.8, 0.0) * probability * 0.5; // Golden ratio glow
                
                fragColor = vec4(unity_glow, v_alpha * probability);
            }
        `;

        this.program = this.createProgram(vertexShaderSource, fragmentShaderSource);
        this.gl.useProgram(this.program);

        // Get uniform locations
        this.uniforms = {
            time: this.gl.getUniformLocation(this.program, 'u_time'),
            phi: this.gl.getUniformLocation(this.program, 'u_phi'),
            unity_state: this.gl.getUniformLocation(this.program, 'u_unity_state')
        };
    }

    createProgram(vertexSource, fragmentSource) {
        const vertexShader = this.createShader(this.gl.VERTEX_SHADER, vertexSource);
        const fragmentShader = this.createShader(this.gl.FRAGMENT_SHADER, fragmentSource);

        const program = this.gl.createProgram();
        this.gl.attachShader(program, vertexShader);
        this.gl.attachShader(program, fragmentShader);
        this.gl.linkProgram(program);

        if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
            console.error('Program link error:', this.gl.getProgramInfoLog(program));
        }

        return program;
    }

    createShader(type, source) {
        const shader = this.gl.createShader(type);
        this.gl.shaderSource(shader, source);
        this.gl.compileShader(shader);

        if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
            console.error('Shader compile error:', this.gl.getShaderInfoLog(shader));
        }

        return shader;
    }

    createBuffers() {
        // Position buffer
        this.positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.positionBuffer);

        // Color buffer
        this.colorBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.colorBuffer);

        // Size buffer
        this.sizeBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.sizeBuffer);

        // Phase buffer
        this.phaseBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.phaseBuffer);

        // Get attribute locations
        this.attributes = {
            position: this.gl.getAttribLocation(this.program, 'a_position'),
            color: this.gl.getAttribLocation(this.program, 'a_color'),
            size: this.gl.getAttribLocation(this.program, 'a_size'),
            phase: this.gl.getAttribLocation(this.program, 'a_phase')
        };
    }

    initializeParticles() {
        const particleCount = 1000;
        this.particles = [];

        for (let i = 0; i < particleCount; i++) {
            const particle = {
                x: (Math.random() - 0.5) * 2,
                y: (Math.random() - 0.5) * 2,
                r: 0.5 + Math.random() * 0.5,
                g: 0.3 + Math.random() * 0.4,
                b: 0.8 + Math.random() * 0.2,
                size: 2 + Math.random() * 8,
                phase: Math.random() * Math.PI * 2,
                entanglement_id: Math.floor(i / 2)
            };

            this.particles.push(particle);
        }

        // Create entangled pairs
        for (let i = 0; i < particleCount; i += 2) {
            if (i + 1 < particleCount) {
                this.entangled_pairs.push([i, i + 1]);
            }
        }

        this.updateBuffers();
    }

    updateBuffers() {
        const positions = new Float32Array(this.particles.length * 2);
        const colors = new Float32Array(this.particles.length * 3);
        const sizes = new Float32Array(this.particles.length);
        const phases = new Float32Array(this.particles.length);

        for (let i = 0; i < this.particles.length; i++) {
            const p = this.particles[i];

            positions[i * 2] = p.x;
            positions[i * 2 + 1] = p.y;

            colors[i * 3] = p.r;
            colors[i * 3 + 1] = p.g;
            colors[i * 3 + 2] = p.b;

            sizes[i] = p.size;
            phases[i] = p.phase;
        }

        // Update position buffer
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.positionBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.DYNAMIC_DRAW);
        this.gl.enableVertexAttribArray(this.attributes.position);
        this.gl.vertexAttribPointer(this.attributes.position, 2, this.gl.FLOAT, false, 0, 0);

        // Update color buffer
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.colorBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, colors, this.gl.DYNAMIC_DRAW);
        this.gl.enableVertexAttribArray(this.attributes.color);
        this.gl.vertexAttribPointer(this.attributes.color, 3, this.gl.FLOAT, false, 0, 0);

        // Update size buffer
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.sizeBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, sizes, this.gl.DYNAMIC_DRAW);
        this.gl.enableVertexAttribArray(this.attributes.size);
        this.gl.vertexAttribPointer(this.attributes.size, 1, this.gl.FLOAT, false, 0, 0);

        // Update phase buffer
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.phaseBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, phases, this.gl.DYNAMIC_DRAW);
        this.gl.enableVertexAttribArray(this.attributes.phase);
        this.gl.vertexAttribPointer(this.attributes.phase, 1, this.gl.FLOAT, false, 0, 0);
    }

    updateParticles() {
        for (let i = 0; i < this.particles.length; i++) {
            const p = this.particles[i];

            // Quantum evolution
            p.phase += 0.02 * this.phi;

            // Consciousness field influence
            const consciousness_x = Math.sin(this.time * 0.5 + i * 0.1) * 0.01;
            const consciousness_y = Math.cos(this.time * 0.3 + i * 0.1) * 0.01;

            p.x += consciousness_x;
            p.y += consciousness_y;

            // Entanglement correlation
            if (i % 2 === 0 && i + 1 < this.particles.length) {
                const partner = this.particles[i + 1];
                const correlation = Math.sin(this.time + i * 0.1) * 0.02;

                p.x += correlation;
                partner.x -= correlation;
                p.y += correlation;
                partner.y -= correlation;
            }

            // Unity constraint: particles converge to unity
            const distance_from_center = Math.sqrt(p.x * p.x + p.y * p.y);
            if (distance_from_center > 1.0) {
                const scale = 1.0 / distance_from_center;
                p.x *= scale;
                p.y *= scale;
            }

            // Color evolution based on unity state
            const unity_factor = Math.sin(this.time * 0.1 + i * 0.01);
            p.r = 0.5 + unity_factor * 0.3;
            p.g = 0.3 + unity_factor * 0.4;
            p.b = 0.8 + unity_factor * 0.2;
        }
    }

    render() {
        this.gl.clearColor(0.0, 0.0, 0.0, 1.0);
        this.gl.clear(this.gl.COLOR_BUFFER_BIT);

        // Update uniforms
        this.gl.uniform1f(this.uniforms.time, this.time);
        this.gl.uniform1f(this.uniforms.phi, this.phi);
        this.gl.uniform1f(this.uniforms.unity_state, this.unity_state);

        // Draw particles
        this.gl.drawArrays(this.gl.POINTS, 0, this.particles.length);
    }

    animate() {
        this.time += 0.016; // ~60fps
        this.updateParticles();
        this.updateBuffers();
        this.render();

        this.animation_id = requestAnimationFrame(() => this.animate());
    }

    startAnimation() {
        this.animate();
    }

    stopAnimation() {
        if (this.animation_id) {
            cancelAnimationFrame(this.animation_id);
        }
    }

    setupEventListeners() {
        window.addEventListener('resize', () => {
            this.setupCanvas();
        });

        // Mouse interaction for consciousness field
        this.canvas.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width * 2 - 1;
            const y = -(e.clientY - rect.top) / rect.height * 2 + 1;

            // Influence consciousness field
            for (let i = 0; i < this.particles.length; i++) {
                const p = this.particles[i];
                const dx = x - p.x;
                const dy = y - p.y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < 0.5) {
                    const force = (0.5 - distance) * 0.01;
                    p.x += dx * force;
                    p.y += dy * force;
                }
            }
        });

        // Unity state toggle
        this.canvas.addEventListener('click', () => {
            this.unity_state = this.unity_state === 1 ? 0.5 : 1;
            console.log('Unity State:', this.unity_state, '1+1=' + this.unity_state);
        });
    }

    // Public API for external control
    setUnityState(state) {
        this.unity_state = state;
    }

    getUnityState() {
        return this.unity_state;
    }

    getConsciousnessField() {
        return this.consciousness_field;
    }

    destroy() {
        this.stopAnimation();
        this.gl.deleteProgram(this.program);
        this.gl.deleteBuffer(this.positionBuffer);
        this.gl.deleteBuffer(this.colorBuffer);
        this.gl.deleteBuffer(this.sizeBuffer);
        this.gl.deleteBuffer(this.phaseBuffer);
    }
}

// Export for global use
window.UnityQuantumEntanglement = UnityQuantumEntanglement; 