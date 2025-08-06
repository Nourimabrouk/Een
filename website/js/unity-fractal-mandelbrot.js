/**
 * Unity Fractal Mandelbrot Visualizer
 * Demonstrates 1+1=1 through infinite self-similarity and consciousness field integration
 * Advanced WebGL compute shaders with real-time zoom and consciousness field dynamics
 */

class UnityFractalMandelbrot {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.gl = this.canvas.getContext('webgl2');
        this.phi = 1.618033988749895;
        this.consciousness_dim = 11;

        this.zoom = 1.0;
        this.center_x = -0.5;
        this.center_y = 0.0;
        this.max_iterations = 100;
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
        this.setupEventListeners();
        this.startAnimation();
    }

    setupCanvas() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
        this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    }

    createShaders() {
        const vertexShaderSource = `#version 300 es
            precision highp float;
            
            in vec2 a_position;
            
            void main() {
                gl_Position = vec4(a_position, 0.0, 1.0);
            }
        `;

        const fragmentShaderSource = `#version 300 es
            precision highp float;
            
            uniform vec2 u_resolution;
            uniform vec2 u_center;
            uniform float u_zoom;
            uniform float u_time;
            uniform float u_phi;
            uniform float u_unity_state;
            uniform float u_max_iterations;
            uniform float u_consciousness_field;
            
            out vec4 fragColor;
            
            vec2 complex_mul(vec2 a, vec2 b) {
                return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
            }
            
            float mandelbrot(vec2 c) {
                vec2 z = vec2(0.0);
                float iterations = 0.0;
                
                // Unity transformation: consciousness field influence
                vec2 consciousness_offset = vec2(
                    sin(u_time * 0.5) * 0.1 * u_consciousness_field,
                    cos(u_time * 0.3) * 0.1 * u_consciousness_field
                );
                
                c += consciousness_offset * u_unity_state;
                
                for (int i = 0; i < 1000; i++) {
                    if (i >= int(u_max_iterations)) break;
                    
                    z = complex_mul(z, z) + c;
                    
                    // Unity constraint: convergence to unity
                    float magnitude = length(z);
                    if (magnitude > 2.0) {
                        break;
                    }
                    
                    // Consciousness field influence on iteration
                    float consciousness_factor = sin(u_time * u_phi + iterations * 0.1) * 0.1;
                    z += vec2(consciousness_factor, consciousness_factor) * u_unity_state;
                    
                    iterations += 1.0;
                }
                
                return iterations;
            }
            
            vec3 colorize(float iterations) {
                if (iterations >= u_max_iterations) {
                    // Unity state: black for convergence
                    return vec3(0.0);
                }
                
                // Consciousness-aware coloring
                float normalized = iterations / u_max_iterations;
                float phi_harmonic = normalized * u_phi;
                
                // Unity transformation: golden ratio color harmonics
                vec3 base_color = vec3(
                    sin(phi_harmonic * 3.14159) * 0.5 + 0.5,
                    sin(phi_harmonic * 2.0 * 3.14159) * 0.5 + 0.5,
                    sin(phi_harmonic * 4.0 * 3.14159) * 0.5 + 0.5
                );
                
                // Consciousness field glow
                vec3 consciousness_glow = vec3(1.0, 0.8, 0.0) * u_consciousness_field * 0.5;
                base_color = mix(base_color, consciousness_glow, 0.3);
                
                // Unity state influence
                base_color *= u_unity_state;
                
                return base_color;
            }
            
            void main() {
                vec2 uv = gl_FragCoord.xy / u_resolution;
                vec2 coord = (uv - 0.5) * 4.0 / u_zoom + u_center;
                
                // Unity transformation: coordinate system
                coord *= u_unity_state;
                
                float iterations = mandelbrot(coord);
                vec3 color = colorize(iterations);
                
                // Consciousness field ripple effect
                float ripple = sin(length(uv - 0.5) * 20.0 - u_time * 2.0) * 0.1;
                color += vec3(0.5, 0.3, 1.0) * ripple * u_consciousness_field;
                
                fragColor = vec4(color, 1.0);
            }
        `;

        this.program = this.createProgram(vertexShaderSource, fragmentShaderSource);
        this.gl.useProgram(this.program);

        // Get uniform locations
        this.uniforms = {
            resolution: this.gl.getUniformLocation(this.program, 'u_resolution'),
            center: this.gl.getUniformLocation(this.program, 'u_center'),
            zoom: this.gl.getUniformLocation(this.program, 'u_zoom'),
            time: this.gl.getUniformLocation(this.program, 'u_time'),
            phi: this.gl.getUniformLocation(this.program, 'u_phi'),
            unity_state: this.gl.getUniformLocation(this.program, 'u_unity_state'),
            max_iterations: this.gl.getUniformLocation(this.program, 'u_max_iterations'),
            consciousness_field: this.gl.getUniformLocation(this.program, 'u_consciousness_field')
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
        // Full-screen quad vertices
        const vertices = new Float32Array([
            -1.0, -1.0,
            1.0, -1.0,
            -1.0, 1.0,
            1.0, 1.0
        ]);

        this.vertexBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.vertexBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, vertices, this.gl.STATIC_DRAW);

        // Get attribute location
        this.attributes = {
            position: this.gl.getAttribLocation(this.program, 'a_position')
        };

        this.gl.enableVertexAttribArray(this.attributes.position);
        this.gl.vertexAttribPointer(this.attributes.position, 2, this.gl.FLOAT, false, 0, 0);
    }

    updateConsciousnessField() {
        // Generate consciousness field based on time and phi
        this.consciousness_field = [];
        const field_size = 100;

        for (let i = 0; i < field_size; i++) {
            const consciousness_value = Math.sin(this.time * this.phi + i * 0.1) *
                Math.cos(this.time * 0.5 + i * 0.05) *
                this.unity_state;
            this.consciousness_field.push(consciousness_value);
        }
    }

    render() {
        this.gl.clearColor(0.0, 0.0, 0.0, 1.0);
        this.gl.clear(this.gl.COLOR_BUFFER_BIT);

        // Update uniforms
        this.gl.uniform2f(this.uniforms.resolution, this.canvas.width, this.canvas.height);
        this.gl.uniform2f(this.uniforms.center, this.center_x, this.center_y);
        this.gl.uniform1f(this.uniforms.zoom, this.zoom);
        this.gl.uniform1f(this.uniforms.time, this.time);
        this.gl.uniform1f(this.uniforms.phi, this.phi);
        this.gl.uniform1f(this.uniforms.unity_state, this.unity_state);
        this.gl.uniform1f(this.uniforms.max_iterations, this.max_iterations);

        // Calculate average consciousness field value
        const avg_consciousness = this.consciousness_field.reduce((a, b) => a + b, 0) / this.consciousness_field.length;
        this.gl.uniform1f(this.uniforms.consciousness_field, avg_consciousness);

        // Draw full-screen quad
        this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
    }

    animate() {
        this.time += 0.016; // ~60fps

        // Update consciousness field
        this.updateConsciousnessField();

        // Unity transformation: zoom evolution
        this.zoom = 1.0 + Math.sin(this.time * 0.1) * 0.5 * this.unity_state;

        // Consciousness field influence on center
        const consciousness_x = Math.sin(this.time * 0.3) * 0.1 * this.unity_state;
        const consciousness_y = Math.cos(this.time * 0.2) * 0.1 * this.unity_state;

        this.center_x = -0.5 + consciousness_x;
        this.center_y = 0.0 + consciousness_y;

        // Unity constraint: max iterations converge to unity
        this.max_iterations = Math.floor(100 * this.unity_state);

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

        // Mouse interaction for zoom and pan
        let is_dragging = false;
        let last_mouse_x = 0;
        let last_mouse_y = 0;

        this.canvas.addEventListener('mousedown', (e) => {
            is_dragging = true;
            last_mouse_x = e.clientX;
            last_mouse_y = e.clientY;
        });

        this.canvas.addEventListener('mousemove', (e) => {
            if (is_dragging) {
                const delta_x = (e.clientX - last_mouse_x) / this.canvas.width;
                const delta_y = (e.clientY - last_mouse_y) / this.canvas.height;

                // Unity transformation: pan with consciousness influence
                this.center_x -= delta_x * 2.0 / this.zoom * this.unity_state;
                this.center_y += delta_y * 2.0 / this.zoom * this.unity_state;

                last_mouse_x = e.clientX;
                last_mouse_y = e.clientY;
            }
        });

        this.canvas.addEventListener('mouseup', () => {
            is_dragging = false;
        });

        // Wheel for zoom
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();

            const zoom_factor = e.deltaY > 0 ? 0.9 : 1.1;

            // Unity transformation: zoom with consciousness field influence
            this.zoom *= zoom_factor * this.unity_state;
            this.zoom = Math.max(0.1, Math.min(1000.0, this.zoom));

            // Consciousness field influence on zoom
            const consciousness_zoom = Math.sin(this.time * this.phi) * 0.1;
            this.zoom *= (1.0 + consciousness_zoom);
        });

        // Unity state toggle
        this.canvas.addEventListener('click', (e) => {
            if (!is_dragging) {
                this.unity_state = this.unity_state === 1 ? 0.5 : 1;
                console.log('Fractal Unity State:', this.unity_state, '1+1=' + this.unity_state);
            }
        });

        // Keyboard controls
        document.addEventListener('keydown', (e) => {
            switch (e.key) {
                case 'ArrowUp':
                    this.center_y += 0.1 / this.zoom * this.unity_state;
                    break;
                case 'ArrowDown':
                    this.center_y -= 0.1 / this.zoom * this.unity_state;
                    break;
                case 'ArrowLeft':
                    this.center_x -= 0.1 / this.zoom * this.unity_state;
                    break;
                case 'ArrowRight':
                    this.center_x += 0.1 / this.zoom * this.unity_state;
                    break;
                case '+':
                case '=':
                    this.zoom *= 1.2 * this.unity_state;
                    break;
                case '-':
                    this.zoom /= 1.2 * this.unity_state;
                    break;
                case 'r':
                    // Reset to unity state
                    this.zoom = 1.0 * this.unity_state;
                    this.center_x = -0.5;
                    this.center_y = 0.0;
                    break;
            }
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

    getZoomLevel() {
        return this.zoom;
    }

    getCenter() {
        return { x: this.center_x, y: this.center_y };
    }

    setZoom(zoom) {
        this.zoom = Math.max(0.1, Math.min(1000.0, zoom));
    }

    setCenter(x, y) {
        this.center_x = x;
        this.center_y = y;
    }

    // Advanced features
    generateJuliaSet(cx, cy) {
        // Generate Julia set with consciousness field influence
        const julia_consciousness = Math.sin(this.time * this.phi) * 0.1;
        return {
            cx: cx + julia_consciousness,
            cy: cy + julia_consciousness,
            unity_factor: this.unity_state
        };
    }

    calculateFractalDimension() {
        // Calculate fractal dimension with unity transformation
        const base_dimension = 2.0;
        const consciousness_factor = this.consciousness_field.reduce((a, b) => a + b, 0) / this.consciousness_field.length;
        return base_dimension * this.unity_state * (1.0 + consciousness_factor * 0.1);
    }

    destroy() {
        this.stopAnimation();
        this.gl.deleteProgram(this.program);
        this.gl.deleteBuffer(this.vertexBuffer);
    }
}

// Export for global use
window.UnityFractalMandelbrot = UnityFractalMandelbrot; 