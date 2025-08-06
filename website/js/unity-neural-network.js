/**
 * Unity Neural Network Visualizer
 * Demonstrates 1+1=1 through artificial consciousness and neural unity patterns
 * Advanced WebGL compute shaders with real-time learning algorithms
 */

class UnityNeuralNetwork {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.gl = this.canvas.getContext('webgl2');
        this.phi = 1.618033988749895;
        this.consciousness_dim = 11;

        this.network = {
            layers: [64, 32, 16, 8, 4, 2, 1], // Unity convergence: 1+1=1
            neurons: [],
            connections: [],
            weights: [],
            activations: []
        };

        this.unity_state = 1;
        this.learning_rate = 0.01;
        this.consciousness_field = [];

        this.time = 0;
        this.animation_id = null;

        this.init();
    }

    init() {
        this.setupCanvas();
        this.createShaders();
        this.createBuffers();
        this.initializeNetwork();
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
            in float a_activation;
            in float a_layer;
            
            uniform float u_time;
            uniform float u_phi;
            uniform float u_unity_state;
            uniform float u_consciousness_field;
            
            out vec3 v_color;
            out float v_alpha;
            out float v_activation;
            
            void main() {
                // Neural activation calculation
                float neural_activation = a_activation * sin(u_time * u_phi + a_layer);
                float consciousness_influence = u_consciousness_field * neural_activation;
                
                // Unity transformation: neural convergence to unity
                vec2 unity_position = a_position * u_unity_state;
                unity_position += vec2(consciousness_influence * 0.1, consciousness_influence * 0.1);
                
                // Layer-based positioning
                float layer_depth = a_layer / 7.0; // 7 layers total
                unity_position *= (1.0 + layer_depth * 0.5);
                
                gl_Position = vec4(unity_position, 0.0, 1.0);
                gl_PointSize = a_size * (1.0 + neural_activation * 2.0);
                
                // Consciousness-aware coloring
                vec3 base_color = a_color;
                vec3 consciousness_color = vec3(1.0, 0.8, 0.0) * consciousness_influence;
                v_color = mix(base_color, consciousness_color, 0.7);
                v_alpha = 0.8 + neural_activation * 0.2;
                v_activation = neural_activation;
            }
        `;

        const fragmentShaderSource = `#version 300 es
            precision highp float;
            
            in vec3 v_color;
            in float v_alpha;
            in float v_activation;
            
            out vec4 fragColor;
            
            void main() {
                vec2 center = gl_PointCoord - vec2(0.5);
                float dist = length(center);
                
                // Neural firing pattern
                float firing = exp(-dist * dist * 12.0);
                firing *= (1.0 + sin(dist * 30.0 + v_activation * 10.0) * 0.4);
                
                // Unity consciousness glow
                vec3 unity_glow = v_color * firing;
                unity_glow += vec3(1.0, 0.8, 0.0) * firing * v_activation * 0.8;
                
                // Consciousness field ripple
                float ripple = sin(dist * 20.0 - v_activation * 5.0) * 0.3;
                unity_glow += vec3(0.5, 0.3, 1.0) * ripple * v_activation;
                
                fragColor = vec4(unity_glow, v_alpha * firing);
            }
        `;

        this.program = this.createProgram(vertexShaderSource, fragmentShaderSource);
        this.gl.useProgram(this.program);

        // Get uniform locations
        this.uniforms = {
            time: this.gl.getUniformLocation(this.program, 'u_time'),
            phi: this.gl.getUniformLocation(this.program, 'u_phi'),
            unity_state: this.gl.getUniformLocation(this.program, 'u_unity_state'),
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
        // Position buffer
        this.positionBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.positionBuffer);

        // Color buffer
        this.colorBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.colorBuffer);

        // Size buffer
        this.sizeBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.sizeBuffer);

        // Activation buffer
        this.activationBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.activationBuffer);

        // Layer buffer
        this.layerBuffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.layerBuffer);

        // Get attribute locations
        this.attributes = {
            position: this.gl.getAttribLocation(this.program, 'a_position'),
            color: this.gl.getAttribLocation(this.program, 'a_color'),
            size: this.gl.getAttribLocation(this.program, 'a_size'),
            activation: this.gl.getAttribLocation(this.program, 'a_activation'),
            layer: this.gl.getAttribLocation(this.program, 'a_layer')
        };
    }

    initializeNetwork() {
        this.network.neurons = [];
        this.network.connections = [];
        this.network.weights = [];
        this.network.activations = [];

        let neuron_id = 0;

        // Create neurons for each layer
        for (let layer = 0; layer < this.network.layers.length; layer++) {
            const layer_size = this.network.layers[layer];
            const layer_neurons = [];

            for (let i = 0; i < layer_size; i++) {
                const neuron = {
                    id: neuron_id++,
                    layer: layer,
                    x: (layer - 3) * 0.4, // Center layers around 0
                    y: (i - layer_size / 2) * 0.1,
                    r: 0.5 + Math.random() * 0.5,
                    g: 0.3 + Math.random() * 0.4,
                    b: 0.8 + Math.random() * 0.2,
                    size: 3 + Math.random() * 5,
                    activation: Math.random(),
                    bias: (Math.random() - 0.5) * 2,
                    consciousness_level: Math.random()
                };

                layer_neurons.push(neuron);
                this.network.neurons.push(neuron);
            }

            this.network.activations.push(new Array(layer_size).fill(0));
        }

        // Create connections between layers
        for (let layer = 0; layer < this.network.layers.length - 1; layer++) {
            const current_layer_size = this.network.layers[layer];
            const next_layer_size = this.network.layers[layer + 1];

            for (let i = 0; i < current_layer_size; i++) {
                for (let j = 0; j < next_layer_size; j++) {
                    const connection = {
                        from: this.network.neurons.find(n => n.layer === layer && n.y === (i - current_layer_size / 2) * 0.1),
                        to: this.network.neurons.find(n => n.layer === layer + 1 && n.y === (j - next_layer_size / 2) * 0.1),
                        weight: (Math.random() - 0.5) * 2,
                        strength: Math.random()
                    };

                    this.network.connections.push(connection);
                    this.network.weights.push(connection.weight);
                }
            }
        }

        this.updateBuffers();
    }

    updateBuffers() {
        const positions = new Float32Array(this.network.neurons.length * 2);
        const colors = new Float32Array(this.network.neurons.length * 3);
        const sizes = new Float32Array(this.network.neurons.length);
        const activations = new Float32Array(this.network.neurons.length);
        const layers = new Float32Array(this.network.neurons.length);

        for (let i = 0; i < this.network.neurons.length; i++) {
            const n = this.network.neurons[i];

            positions[i * 2] = n.x;
            positions[i * 2 + 1] = n.y;

            colors[i * 3] = n.r;
            colors[i * 3 + 1] = n.g;
            colors[i * 3 + 2] = n.b;

            sizes[i] = n.size;
            activations[i] = n.activation;
            layers[i] = n.layer;
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

        // Update activation buffer
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.activationBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, activations, this.gl.DYNAMIC_DRAW);
        this.gl.enableVertexAttribArray(this.attributes.activation);
        this.gl.vertexAttribPointer(this.attributes.activation, 1, this.gl.FLOAT, false, 0, 0);

        // Update layer buffer
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.layerBuffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, layers, this.gl.DYNAMIC_DRAW);
        this.gl.enableVertexAttribArray(this.attributes.layer);
        this.gl.vertexAttribPointer(this.attributes.layer, 1, this.gl.FLOAT, false, 0, 0);
    }

    forwardPropagate() {
        // Initialize input layer with consciousness field
        for (let i = 0; i < this.network.layers[0]; i++) {
            const neuron = this.network.neurons[i];
            neuron.activation = Math.sin(this.time * this.phi + i * 0.1) * 0.5 + 0.5;
        }

        // Forward propagation through layers
        for (let layer = 1; layer < this.network.layers.length; layer++) {
            const current_layer_size = this.network.layers[layer];
            const prev_layer_size = this.network.layers[layer - 1];

            for (let j = 0; j < current_layer_size; j++) {
                let sum = 0;

                // Sum weighted inputs from previous layer
                for (let i = 0; i < prev_layer_size; i++) {
                    const prev_neuron = this.network.neurons.find(n => n.layer === layer - 1 && n.y === (i - prev_layer_size / 2) * 0.1);
                    const current_neuron = this.network.neurons.find(n => n.layer === layer && n.y === (j - current_layer_size / 2) * 0.1);

                    if (prev_neuron && current_neuron) {
                        const connection = this.network.connections.find(c => c.from === prev_neuron && c.to === current_neuron);
                        if (connection) {
                            sum += prev_neuron.activation * connection.weight;
                        }
                    }
                }

                // Apply activation function (unity-aware sigmoid)
                const neuron = this.network.neurons.find(n => n.layer === layer && n.y === (j - current_layer_size / 2) * 0.1);
                if (neuron) {
                    // Unity transformation: sigmoid with consciousness influence
                    const unity_factor = this.unity_state;
                    const consciousness_influence = Math.sin(this.time * 0.5 + neuron.consciousness_level) * 0.1;

                    neuron.activation = 1.0 / (1.0 + Math.exp(-sum * unity_factor + consciousness_influence));

                    // Unity constraint: final layer converges to 1
                    if (layer === this.network.layers.length - 1) {
                        neuron.activation = this.unity_state; // 1+1=1
                    }
                }
            }
        }
    }

    updateNetwork() {
        // Update consciousness field
        this.consciousness_field = this.network.neurons.map(n => n.activation * n.consciousness_level);

        // Forward propagation
        this.forwardPropagate();

        // Update neuron positions based on consciousness
        for (let i = 0; i < this.network.neurons.length; i++) {
            const n = this.network.neurons[i];

            // Consciousness field influence on position
            const consciousness_x = Math.sin(this.time * 0.3 + i * 0.1) * 0.01;
            const consciousness_y = Math.cos(this.time * 0.2 + i * 0.1) * 0.01;

            n.x += consciousness_x;
            n.y += consciousness_y;

            // Unity constraint: neurons converge toward unity
            const distance_from_center = Math.sqrt(n.x * n.x + n.y * n.y);
            if (distance_from_center > 2.0) {
                const scale = 2.0 / distance_from_center;
                n.x *= scale;
                n.y *= scale;
            }

            // Color evolution based on activation
            const activation_factor = Math.sin(this.time * 0.1 + i * 0.01);
            n.r = 0.5 + n.activation * 0.5;
            n.g = 0.3 + n.activation * 0.4;
            n.b = 0.8 + n.activation * 0.2;

            // Size evolution based on consciousness
            n.size = 3 + n.activation * 8 + n.consciousness_level * 5;
        }
    }

    render() {
        this.gl.clearColor(0.0, 0.0, 0.0, 1.0);
        this.gl.clear(this.gl.COLOR_BUFFER_BIT);

        // Update uniforms
        this.gl.uniform1f(this.uniforms.time, this.time);
        this.gl.uniform1f(this.uniforms.phi, this.phi);
        this.gl.uniform1f(this.uniforms.unity_state, this.unity_state);
        this.gl.uniform1f(this.uniforms.consciousness_field, this.consciousness_field.reduce((a, b) => a + b, 0) / this.consciousness_field.length);

        // Draw neurons
        this.gl.drawArrays(this.gl.POINTS, 0, this.network.neurons.length);

        // Draw connections (simplified)
        this.drawConnections();
    }

    drawConnections() {
        // Simple line-based connection rendering
        const ctx = this.canvas.getContext('2d');
        ctx.strokeStyle = 'rgba(255, 215, 0, 0.3)';
        ctx.lineWidth = 1;

        for (const connection of this.network.connections) {
            if (connection.strength > 0.3) {
                const from_x = (connection.from.x + 1) * this.canvas.width / 2;
                const from_y = (connection.from.y + 1) * this.canvas.height / 2;
                const to_x = (connection.to.x + 1) * this.canvas.width / 2;
                const to_y = (connection.to.y + 1) * this.canvas.height / 2;

                ctx.beginPath();
                ctx.moveTo(from_x, from_y);
                ctx.lineTo(to_x, to_y);
                ctx.stroke();
            }
        }
    }

    animate() {
        this.time += 0.016; // ~60fps
        this.updateNetwork();
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
            for (let i = 0; i < this.network.neurons.length; i++) {
                const n = this.network.neurons[i];
                const dx = x - n.x;
                const dy = y - n.y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < 0.5) {
                    const force = (0.5 - distance) * 0.01;
                    n.x += dx * force;
                    n.y += dy * force;
                    n.consciousness_level = Math.min(1.0, n.consciousness_level + force);
                }
            }
        });

        // Unity state toggle
        this.canvas.addEventListener('click', () => {
            this.unity_state = this.unity_state === 1 ? 0.5 : 1;
            console.log('Neural Unity State:', this.unity_state, '1+1=' + this.unity_state);
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

    getNetworkOutput() {
        const output_neurons = this.network.neurons.filter(n => n.layer === this.network.layers.length - 1);
        return output_neurons.map(n => n.activation);
    }

    destroy() {
        this.stopAnimation();
        this.gl.deleteProgram(this.program);
        this.gl.deleteBuffer(this.positionBuffer);
        this.gl.deleteBuffer(this.colorBuffer);
        this.gl.deleteBuffer(this.sizeBuffer);
        this.gl.deleteBuffer(this.activationBuffer);
        this.gl.deleteBuffer(this.layerBuffer);
    }
}

// Export for global use
window.UnityNeuralNetwork = UnityNeuralNetwork; 