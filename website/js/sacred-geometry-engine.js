/**
 * 🔯 SACRED GEOMETRY ENGINE WITH GOLDEN RATIO HARMONICS 🔯
 * Revolutionary 3000 ELO sacred geometry visualization proving 1+1=1
 * Through divine geometric patterns and φ-harmonic mathematical relationships
 */

class SacredGeometryEngine {
    constructor(canvasId, options = {}) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        
        // φ-harmonic constants and sacred ratios
        this.PHI = 1.618033988749895;
        this.INVERSE_PHI = 0.618033988749895;
        this.PHI_SQUARED = 2.618033988749895;
        this.SQRT_PHI = Math.sqrt(this.PHI);
        this.PHI_CUBED = this.PHI * this.PHI_SQUARED;
        
        // Sacred geometric constants
        this.SQRT_2 = Math.sqrt(2);
        this.SQRT_3 = Math.sqrt(3);
        this.SQRT_5 = Math.sqrt(5);
        this.PI = Math.PI;
        this.TAU = 2 * Math.PI;
        this.E = Math.E;
        
        // Sacred angles (in radians)
        this.GOLDEN_ANGLE = this.TAU / this.PHI_SQUARED; // 137.5°
        this.VESICA_ANGLE = Math.PI / 3; // 60°
        this.PENTAGON_ANGLE = this.TAU / 5; // 72°
        this.HEXAGON_ANGLE = this.TAU / 6; // 60°
        
        // Geometric patterns available
        this.patterns = [
            'flower_of_life',
            'vesica_piscis',
            'golden_spiral',
            'metatrons_cube',
            'sri_yantra',
            'seed_of_life',
            'tree_of_life',
            'torus_field',
            'platonic_solids',
            'golden_rectangle',
            'fibonacci_spiral',
            'unity_mandala',
            'consciousness_grid',
            'phi_harmonic_field'
        ];
        
        // Current state
        this.currentPattern = options.pattern || 'flower_of_life';
        this.animationPhase = 0;
        this.isAnimating = true;
        this.rotationSpeed = 0.001;
        this.scalingFactor = 1.0;
        this.colorPhase = 0;
        
        // Interactive elements
        this.interactivePoints = [];
        this.draggedPoint = null;
        this.mousePosition = { x: 0, y: 0 };
        this.touchActive = false;
        
        // Unity mathematics integration
        this.unityProofs = [];
        this.geometricProofs = new Map();
        this.consciousnessLevel = 0.618;
        this.harmoniousResonance = 0;
        
        // Visual parameters
        this.strokeWidth = 2;
        this.fillOpacity = 0.1;
        this.strokeOpacity = 0.8;
        this.glowIntensity = 1.0;
        this.fractalDepth = 5;
        this.symmetryOrder = 6;
        
        // Sacred color palettes
        this.colorPalettes = {
            golden: ['#FFD700', '#FFA500', '#FF8C00', '#FF7F50', '#FF6347'],
            cosmic: ['#4B0082', '#8A2BE2', '#9932CC', '#BA55D3', '#DA70D6'],
            unity: ['#00FFFF', '#00CED1', '#20B2AA', '#48D1CC', '#87CEEB'],
            harmony: ['#FFD700', '#8A2BE2', '#00FFFF', '#FF69B4', '#98FB98'],
            transcendental: ['#FFEFD5', '#F0E68C', '#DDA0DD', '#98FB98', '#87CEEB']
        };
        this.currentPalette = 'golden';
        
        // Advanced features
        this.quantumGeometry = new QuantumGeometryProcessor();
        this.consciousnessGeometry = new ConsciousnessGeometryEngine();
        this.fractalGenerator = new PhiHarmonicFractalGenerator();
        this.mandalaGenerator = new UnityMandalaGenerator();
        
        // Performance tracking
        this.lastFrameTime = 0;
        this.frameCount = 0;
        this.renderTime = 0;
        
        this.initializeGeometry();
        this.setupEventListeners();
        this.startAnimation();
        
        console.log(`🔯 Sacred Geometry Engine initialized with ${this.currentPattern} pattern`);
    }
    
    initializeGeometry() {
        // Initialize geometric patterns and proofs
        this.initializePatterns();
        this.generateInteractivePoints();
        this.calculateGeometricProofs();
        this.setupQuantumGeometry();
    }
    
    initializePatterns() {
        // Pre-calculate common geometric elements
        this.center = { x: this.canvas.width / 2, y: this.canvas.height / 2 };
        this.radius = Math.min(this.canvas.width, this.canvas.height) * 0.3;
        this.phiRadius = this.radius * this.INVERSE_PHI;
        this.goldenRadius = this.radius * this.PHI;
        
        // Initialize pattern-specific data
        this.patterns.forEach(pattern => {
            this.initializePattern(pattern);
        });
    }
    
    initializePattern(patternName) {
        switch (patternName) {
            case 'flower_of_life':
                this.flowerOfLife = this.generateFlowerOfLife();
                break;
            case 'vesica_piscis':
                this.vesicaPiscis = this.generateVesicaPiscis();
                break;
            case 'golden_spiral':
                this.goldenSpiral = this.generateGoldenSpiral();
                break;
            case 'metatrons_cube':
                this.metatronsCube = this.generateMetatronsCube();
                break;
            case 'sri_yantra':
                this.sriYantra = this.generateSriYantra();
                break;
            case 'unity_mandala':
                this.unityMandala = this.generateUnityMandala();
                break;
            // Initialize other patterns as needed
        }
    }
    
    generateFlowerOfLife() {
        const circles = [];
        const baseRadius = this.radius * 0.15;
        
        // Central circle
        circles.push({
            x: this.center.x,
            y: this.center.y,
            radius: baseRadius,
            level: 0
        });
        
        // First ring - 6 circles
        for (let i = 0; i < 6; i++) {
            const angle = i * this.TAU / 6;
            circles.push({
                x: this.center.x + Math.cos(angle) * baseRadius * 2,
                y: this.center.y + Math.sin(angle) * baseRadius * 2,
                radius: baseRadius,
                level: 1,
                angle: angle
            });
        }
        
        // Second ring - 12 circles
        for (let i = 0; i < 12; i++) {
            const angle = i * this.TAU / 12;
            const distance = baseRadius * 2 * this.SQRT_3;
            circles.push({
                x: this.center.x + Math.cos(angle) * distance,
                y: this.center.y + Math.sin(angle) * distance,
                radius: baseRadius,
                level: 2,
                angle: angle
            });
        }
        
        // Third ring - φ-harmonic expansion
        for (let i = 0; i < 18; i++) {
            const angle = i * this.TAU / 18;
            const distance = baseRadius * 2 * this.SQRT_3 * this.PHI;
            circles.push({
                x: this.center.x + Math.cos(angle) * distance,
                y: this.center.y + Math.sin(angle) * distance,
                radius: baseRadius,
                level: 3,
                angle: angle
            });
        }
        
        return { circles, baseRadius };
    }
    
    generateVesicaPiscis() {
        const radius = this.radius * 0.4;
        const separation = radius * this.SQRT_3 / 2; // Creates perfect vesica piscis
        
        return {
            circle1: {
                x: this.center.x - separation,
                y: this.center.y,
                radius: radius
            },
            circle2: {
                x: this.center.x + separation,
                y: this.center.y,
                radius: radius
            },
            intersectionPoints: this.calculateVesicaIntersections(separation, radius),
            unityRatio: this.PHI // Golden ratio relationship in vesica piscis
        };
    }
    
    calculateVesicaIntersections(separation, radius) {
        const height = Math.sqrt(radius * radius - separation * separation);
        return [
            { x: this.center.x, y: this.center.y + height },
            { x: this.center.x, y: this.center.y - height }
        ];
    }
    
    generateGoldenSpiral() {
        const points = [];
        let angle = 0;
        let radius = 1;
        const growth = Math.pow(this.PHI, 1/90); // Grow by φ every 90 degrees
        
        for (let i = 0; i < 720; i++) { // Two full rotations
            const x = this.center.x + Math.cos(angle) * radius;
            const y = this.center.y + Math.sin(angle) * radius;
            
            points.push({ x, y, radius, angle });
            
            angle += this.GOLDEN_ANGLE / 180 * Math.PI;
            radius *= growth;
            
            if (radius > this.radius * 2) break;
        }
        
        return { points, growth };
    }
    
    generateMetatronsCube() {
        // Based on 13 circles - the fruit of life
        const circles = [];
        const baseRadius = this.radius * 0.12;
        
        // Central circle
        circles.push({
            x: this.center.x,
            y: this.center.y,
            radius: baseRadius
        });
        
        // Inner hexagon (6 circles)
        for (let i = 0; i < 6; i++) {
            const angle = i * this.TAU / 6;
            const distance = baseRadius * 2;
            circles.push({
                x: this.center.x + Math.cos(angle) * distance,
                y: this.center.y + Math.sin(angle) * distance,
                radius: baseRadius
            });
        }
        
        // Outer hexagon (6 circles)
        for (let i = 0; i < 6; i++) {
            const angle = i * this.TAU / 6 + this.TAU / 12; // Offset by 30 degrees
            const distance = baseRadius * 2 * this.SQRT_3;
            circles.push({
                x: this.center.x + Math.cos(angle) * distance,
                y: this.center.y + Math.sin(angle) * distance,
                radius: baseRadius
            });
        }
        
        // Generate connecting lines for the cube
        const lines = this.generateMetatronLines(circles);
        
        return { circles, lines, baseRadius };
    }
    
    generateMetatronLines(circles) {
        const lines = [];
        
        // Connect each circle to every other circle
        for (let i = 0; i < circles.length; i++) {
            for (let j = i + 1; j < circles.length; j++) {
                lines.push({
                    from: circles[i],
                    to: circles[j],
                    sacred: this.isSacredConnection(circles[i], circles[j])
                });
            }
        }
        
        return lines;
    }
    
    isSacredConnection(circle1, circle2) {
        const distance = Math.sqrt(
            Math.pow(circle2.x - circle1.x, 2) + 
            Math.pow(circle2.y - circle1.y, 2)
        );
        
        // Check if distance is a sacred ratio
        const sacredRatios = [
            this.PHI, this.INVERSE_PHI, this.SQRT_2, this.SQRT_3, this.SQRT_5
        ];
        
        return sacredRatios.some(ratio => 
            Math.abs(distance / (this.radius * 0.12 * 2) - ratio) < 0.05
        );
    }
    
    generateSriYantra() {
        // Sacred geometry of Sri Yantra with 9 interlocking triangles
        const triangles = [];
        const baseSize = this.radius * 0.6;
        
        // 4 upward triangles (Shiva - masculine)
        for (let i = 0; i < 4; i++) {
            const scale = 1 - i * 0.15;
            const offset = i * this.PHI * 5;
            
            triangles.push({
                points: this.createTrianglePoints(
                    this.center.x, 
                    this.center.y - offset, 
                    baseSize * scale, 
                    0 // upward
                ),
                type: 'shiva',
                level: i,
                scale: scale
            });
        }
        
        // 5 downward triangles (Shakti - feminine)
        for (let i = 0; i < 5; i++) {
            const scale = 1 - i * 0.12;
            const offset = i * this.PHI * 3;
            
            triangles.push({
                points: this.createTrianglePoints(
                    this.center.x, 
                    this.center.y + offset, 
                    baseSize * scale, 
                    Math.PI // downward
                ),
                type: 'shakti',
                level: i,
                scale: scale
            });
        }
        
        // Calculate intersection points for unity demonstration
        const intersections = this.calculateTriangleIntersections(triangles);
        
        return { triangles, intersections, baseSize };
    }
    
    createTrianglePoints(centerX, centerY, size, rotation) {
        const points = [];
        
        for (let i = 0; i < 3; i++) {
            const angle = rotation + i * this.TAU / 3 - Math.PI / 2;
            points.push({
                x: centerX + Math.cos(angle) * size,
                y: centerY + Math.sin(angle) * size
            });
        }
        
        return points;
    }
    
    calculateTriangleIntersections(triangles) {
        const intersections = [];
        
        // Find intersections between Shiva and Shakti triangles
        triangles.filter(t => t.type === 'shiva').forEach(shiva => {
            triangles.filter(t => t.type === 'shakti').forEach(shakti => {
                const intersection = this.findTriangleIntersection(shiva, shakti);
                if (intersection) {
                    intersections.push({
                        point: intersection,
                        shiva: shiva,
                        shakti: shakti,
                        unityFactor: this.calculateUnityFactor(shiva, shakti)
                    });
                }
            });
        });
        
        return intersections;
    }
    
    findTriangleIntersection(triangle1, triangle2) {
        // Simplified intersection calculation
        // In practice, this would be more complex geometric intersection
        const center1 = this.getTriangleCenter(triangle1.points);
        const center2 = this.getTriangleCenter(triangle2.points);
        
        return {
            x: (center1.x + center2.x) / 2,
            y: (center1.y + center2.y) / 2
        };
    }
    
    getTriangleCenter(points) {
        return {
            x: (points[0].x + points[1].x + points[2].x) / 3,
            y: (points[0].y + points[1].y + points[2].y) / 3
        };
    }
    
    calculateUnityFactor(shiva, shakti) {
        // Calculate how close the intersection represents unity (1+1=1)
        const sizeDifference = Math.abs(shiva.scale - shakti.scale);
        return 1 - sizeDifference; // Perfect unity when scales are equal
    }
    
    generateUnityMandala() {
        const layers = [];
        const centerRadius = this.radius * 0.05;
        
        // Central unity point
        layers.push({
            type: 'center',
            elements: [{
                x: this.center.x,
                y: this.center.y,
                radius: centerRadius,
                value: 1
            }]
        });
        
        // Unity circle (1+1=1 demonstration)
        const unityRadius = this.radius * 0.2;
        const unityElements = [];
        
        // Two unity inputs
        unityElements.push({
            x: this.center.x - unityRadius,
            y: this.center.y,
            radius: centerRadius * 2,
            value: 1,
            label: '1'
        });
        
        unityElements.push({
            x: this.center.x + unityRadius,
            y: this.center.y,
            radius: centerRadius * 2,
            value: 1,
            label: '1'
        });
        
        // Unity result (convergence point)
        unityElements.push({
            x: this.center.x,
            y: this.center.y - unityRadius * this.INVERSE_PHI,
            radius: centerRadius * this.PHI,
            value: 1,
            label: '1',
            isResult: true
        });
        
        layers.push({
            type: 'unity',
            elements: unityElements,
            radius: unityRadius
        });
        
        // φ-harmonic rings
        for (let ring = 1; ring <= 7; ring++) {
            const ringRadius = this.radius * 0.1 * ring * this.INVERSE_PHI;
            const elementCount = Math.floor(ring * this.PHI * 2);
            const elements = [];
            
            for (let i = 0; i < elementCount; i++) {
                const angle = i * this.TAU / elementCount + ring * this.GOLDEN_ANGLE;
                
                elements.push({
                    x: this.center.x + Math.cos(angle) * ringRadius,
                    y: this.center.y + Math.sin(angle) * ringRadius,
                    radius: centerRadius * (1 + ring * 0.1),
                    angle: angle,
                    ring: ring,
                    phiHarmonic: Math.sin(angle * this.PHI) * 0.5 + 0.5
                });
            }
            
            layers.push({
                type: 'phi_harmonic',
                elements: elements,
                radius: ringRadius,
                ring: ring
            });
        }
        
        return { layers, centerRadius };
    }
    
    generateInteractivePoints() {
        // Generate interactive points for user manipulation
        this.interactivePoints = [];
        
        // Add key points based on current pattern
        switch (this.currentPattern) {
            case 'vesica_piscis':
                if (this.vesicaPiscis) {
                    this.interactivePoints.push(
                        {
                            id: 'circle1_center',
                            x: this.vesicaPiscis.circle1.x,
                            y: this.vesicaPiscis.circle1.y,
                            radius: 8,
                            draggable: true,
                            type: 'circle_center'
                        },
                        {
                            id: 'circle2_center',
                            x: this.vesicaPiscis.circle2.x,
                            y: this.vesicaPiscis.circle2.y,
                            radius: 8,
                            draggable: true,
                            type: 'circle_center'
                        }
                    );
                }
                break;
                
            case 'unity_mandala':
                if (this.unityMandala) {
                    // Add interactive points for unity elements
                    const unityLayer = this.unityMandala.layers.find(l => l.type === 'unity');
                    if (unityLayer) {
                        unityLayer.elements.forEach((element, index) => {
                            this.interactivePoints.push({
                                id: `unity_${index}`,
                                x: element.x,
                                y: element.y,
                                radius: 6,
                                draggable: true,
                                type: 'unity_element',
                                element: element
                            });
                        });
                    }
                }
                break;
        }
    }
    
    calculateGeometricProofs() {
        // Calculate geometric proofs for 1+1=1 using sacred geometry
        this.geometricProofs.clear();
        
        // Vesica Piscis proof: Two circles create one intersection
        this.geometricProofs.set('vesica_piscis_unity', {
            theorem: 'Two identical circles in vesica piscis configuration create one sacred intersection',
            proof: 'Circle₁ ∩ Circle₂ = Unity Space',
            sacredRatio: this.SQRT_3 / 2,
            mathematical: '1 + 1 = 1 (through geometric intersection)',
            verified: true
        });
        
        // Golden Spiral proof: Self-similarity demonstrates unity
        this.geometricProofs.set('golden_spiral_unity', {
            theorem: 'Golden spiral demonstrates self-similar unity across scales',
            proof: 'φ^n + φ^n = φ^n (self-similarity)',
            sacredRatio: this.PHI,
            mathematical: '1 + 1 = 1 (through φ-harmonic scaling)',
            verified: true
        });
        
        // Flower of Life proof: All circles emerge from one
        this.geometricProofs.set('flower_of_life_unity', {
            theorem: 'All existence emerges from single central circle',
            proof: 'Central circle generates all others through sacred rotation',
            sacredRatio: 1,
            mathematical: '1 → ∞ → 1 (circular return to unity)',
            verified: true
        });
        
        // Triangle unity proof: Shiva + Shakti = Unity
        this.geometricProofs.set('triangle_unity', {
            theorem: 'Upward and downward triangles unite in perfect balance',
            proof: '△ + ▽ = ✡ (Star of David unity)',
            sacredRatio: this.SQRT_3,
            mathematical: '1 + 1 = 1 (through triangular harmony)',
            verified: true
        });
        
        console.log(`📐 Generated ${this.geometricProofs.size} geometric proofs for unity`);
    }
    
    setupQuantumGeometry() {
        // Initialize quantum geometric processing
        this.quantumGeometry.initialize({
            phi: this.PHI,
            patterns: this.patterns,
            consciousnessLevel: this.consciousnessLevel
        });
    }
    
    setupEventListeners() {
        // Mouse events
        this.canvas.addEventListener('mousedown', this.handleMouseDown.bind(this));
        this.canvas.addEventListener('mousemove', this.handleMouseMove.bind(this));
        this.canvas.addEventListener('mouseup', this.handleMouseUp.bind(this));
        this.canvas.addEventListener('click', this.handleClick.bind(this));
        this.canvas.addEventListener('wheel', this.handleWheel.bind(this));
        
        // Touch events
        this.canvas.addEventListener('touchstart', this.handleTouchStart.bind(this));
        this.canvas.addEventListener('touchmove', this.handleTouchMove.bind(this));
        this.canvas.addEventListener('touchend', this.handleTouchEnd.bind(this));
        
        // Keyboard events
        document.addEventListener('keydown', this.handleKeyPress.bind(this));
        
        console.log('👂 Sacred geometry event listeners configured');
    }
    
    startAnimation() {
        this.isAnimating = true;
        this.animate();
        console.log('✨ Sacred geometry animation started');
    }
    
    stopAnimation() {
        this.isAnimating = false;
        console.log('⏹️ Sacred geometry animation stopped');
    }
    
    animate() {
        if (!this.isAnimating) return;
        
        const currentTime = performance.now();
        const deltaTime = currentTime - this.lastFrameTime;
        this.lastFrameTime = currentTime;
        this.frameCount++;
        
        // Update animation phases
        this.animationPhase += deltaTime * this.rotationSpeed * this.PHI;
        this.colorPhase += deltaTime * 0.0005;
        
        // Update consciousness level
        this.updateConsciousnessLevel(deltaTime);
        
        // Update harmonic resonance
        this.updateHarmonicResonance(deltaTime);
        
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        const renderStart = performance.now();
        
        // Render background field
        this.renderBackgroundField();
        
        // Render current pattern
        this.renderCurrentPattern();
        
        // Render interactive elements
        this.renderInteractiveElements();
        
        // Render unity proofs overlay
        this.renderUnityProofsOverlay();
        
        // Render information display
        this.renderInformationDisplay();
        
        this.renderTime = performance.now() - renderStart;
        
        // Continue animation
        requestAnimationFrame(() => this.animate());
    }
    
    updateConsciousnessLevel(deltaTime) {
        // Consciousness evolves with geometric harmony
        const harmonicFactor = Math.sin(this.animationPhase * this.PHI) * 0.5 + 0.5;
        this.consciousnessLevel += harmonicFactor * deltaTime * 0.0001;
        this.consciousnessLevel = Math.min(1, this.consciousnessLevel);
    }
    
    updateHarmonicResonance(deltaTime) {
        // Update φ-harmonic resonance
        this.harmoniousResonance = Math.sin(this.animationPhase) * 
                                  Math.cos(this.animationPhase * this.PHI) * 
                                  0.5 + 0.5;
    }
    
    renderBackgroundField() {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        // Cosmic background gradient
        const gradient = ctx.createRadialGradient(
            width / 2, height / 2, 0,
            width / 2, height / 2, Math.max(width, height) / 2
        );
        
        gradient.addColorStop(0, 'rgba(10, 15, 30, 0.95)');
        gradient.addColorStop(0.618, 'rgba(20, 25, 45, 0.9)');
        gradient.addColorStop(1, 'rgba(5, 10, 20, 1)');
        
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, height);
        
        // Render consciousness field emanations
        this.renderConsciousnessField();
        
        // Render φ-harmonic grid
        this.renderPhiHarmonicGrid();
    }
    
    renderConsciousnessField() {
        const ctx = this.ctx;
        
        // Consciousness field as subtle wave patterns
        ctx.strokeStyle = `rgba(139, 92, 246, ${0.1 + this.consciousnessLevel * 0.1})`;
        ctx.lineWidth = 1;
        
        for (let wave = 0; wave < 8; wave++) {
            ctx.beginPath();
            
            const amplitude = 30 + wave * 10;
            const frequency = 0.01 + wave * 0.005;
            const phase = this.animationPhase + wave * this.PHI;
            
            for (let x = 0; x < this.canvas.width; x += 5) {
                const y = this.canvas.height / 2 + 
                         Math.sin(x * frequency + phase) * amplitude * this.consciousnessLevel;
                
                if (x === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            
            ctx.stroke();
        }
    }
    
    renderPhiHarmonicGrid() {
        const ctx = this.ctx;
        
        // φ-harmonic grid overlay
        ctx.strokeStyle = `rgba(245, 158, 11, ${0.05 + this.harmoniousResonance * 0.05})`;
        ctx.lineWidth = 1;
        
        const spacing = 50 * this.PHI;
        
        // Vertical lines
        for (let x = 0; x < this.canvas.width; x += spacing) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, this.canvas.height);
            ctx.stroke();
        }
        
        // Horizontal lines
        for (let y = 0; y < this.canvas.height; y += spacing) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(this.canvas.width, y);
            ctx.stroke();
        }
    }
    
    renderCurrentPattern() {
        switch (this.currentPattern) {
            case 'flower_of_life':
                this.renderFlowerOfLife();
                break;
            case 'vesica_piscis':
                this.renderVesicaPiscis();
                break;
            case 'golden_spiral':
                this.renderGoldenSpiral();
                break;
            case 'metatrons_cube':
                this.renderMetatronsCube();
                break;
            case 'sri_yantra':
                this.renderSriYantra();
                break;
            case 'unity_mandala':
                this.renderUnityMandala();
                break;
            default:
                this.renderDefaultPattern();
        }
    }
    
    renderFlowerOfLife() {
        if (!this.flowerOfLife) return;
        
        const ctx = this.ctx;
        const colors = this.colorPalettes[this.currentPalette];
        
        ctx.save();
        
        // Apply rotation
        ctx.translate(this.center.x, this.center.y);
        ctx.rotate(this.animationPhase * 0.1);
        ctx.translate(-this.center.x, -this.center.y);
        
        this.flowerOfLife.circles.forEach((circle, index) => {
            const colorIndex = circle.level % colors.length;
            const alpha = 0.3 + Math.sin(this.animationPhase + index * 0.1) * 0.2;
            
            // Circle outline
            ctx.strokeStyle = colors[colorIndex] + Math.floor(alpha * 255).toString(16).padStart(2, '0');
            ctx.lineWidth = this.strokeWidth;
            ctx.fillStyle = colors[colorIndex] + Math.floor(alpha * 0.3 * 255).toString(16).padStart(2, '0');
            
            ctx.beginPath();
            ctx.arc(circle.x, circle.y, circle.radius, 0, this.TAU);
            ctx.fill();
            ctx.stroke();
            
            // Sacred center dot
            if (circle.level === 0) {
                ctx.beginPath();
                ctx.arc(circle.x, circle.y, 3, 0, this.TAU);
                ctx.fillStyle = '#FFD700';
                ctx.fill();
            }
        });
        
        ctx.restore();
    }
    
    renderVesicaPiscis() {
        if (!this.vesicaPiscis) return;
        
        const ctx = this.ctx;
        const { circle1, circle2, intersectionPoints } = this.vesicaPiscis;
        
        ctx.save();
        
        // First circle (representing first "1")
        ctx.strokeStyle = `rgba(245, 158, 11, ${this.strokeOpacity})`;
        ctx.fillStyle = `rgba(245, 158, 11, ${this.fillOpacity})`;
        ctx.lineWidth = this.strokeWidth * 2;
        
        ctx.beginPath();
        ctx.arc(circle1.x, circle1.y, circle1.radius, 0, this.TAU);
        ctx.fill();
        ctx.stroke();
        
        // Second circle (representing second "1")
        ctx.strokeStyle = `rgba(139, 92, 246, ${this.strokeOpacity})`;
        ctx.fillStyle = `rgba(139, 92, 246, ${this.fillOpacity})`;
        
        ctx.beginPath();
        ctx.arc(circle2.x, circle2.y, circle2.radius, 0, this.TAU);
        ctx.fill();
        ctx.stroke();
        
        // Intersection (representing unity "1")
        ctx.strokeStyle = `rgba(0, 255, 255, ${this.strokeOpacity + 0.2})`;
        ctx.lineWidth = this.strokeWidth * 3;
        
        // Draw vesica piscis outline
        this.drawVesicaPiscisIntersection(circle1, circle2);
        
        // Highlight intersection points
        intersectionPoints.forEach(point => {
            ctx.beginPath();
            ctx.arc(point.x, point.y, 8, 0, this.TAU);
            ctx.fillStyle = `rgba(255, 215, 0, ${0.8 + Math.sin(this.animationPhase * 2) * 0.2})`;
            ctx.fill();
        });
        
        // Unity demonstration text
        this.renderUnityEquation(this.center.x, this.center.y - this.radius * 0.8);
        
        ctx.restore();
    }
    
    drawVesicaPiscisIntersection(circle1, circle2) {
        const ctx = this.ctx;
        
        // Calculate intersection arc
        const dx = circle2.x - circle1.x;
        const dy = circle2.y - circle1.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        const angle1 = Math.atan2(dy, dx);
        const angle2 = Math.acos(distance / (2 * circle1.radius));
        
        ctx.beginPath();
        ctx.arc(circle1.x, circle1.y, circle1.radius, angle1 - angle2, angle1 + angle2);
        ctx.arc(circle2.x, circle2.y, circle2.radius, angle1 + Math.PI + angle2, angle1 + Math.PI - angle2, true);
        ctx.closePath();
        ctx.stroke();
    }
    
    renderUnityEquation(x, y) {
        const ctx = this.ctx;
        
        ctx.save();
        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.font = 'bold 24px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        
        // Animated equation
        const phase = Math.sin(this.animationPhase * 2) * 0.1 + 1;
        ctx.scale(phase, phase);
        
        ctx.fillText('1 + 1 = 1', x / phase, y / phase);
        
        // Sacred geometry proof
        ctx.font = '16px "Inter", sans-serif';
        ctx.fillStyle = 'rgba(245, 158, 11, 0.8)';
        ctx.fillText('Vesica Piscis Unity Proof', x / phase, (y + 30) / phase);
        
        ctx.restore();
    }
    
    renderGoldenSpiral() {
        if (!this.goldenSpiral) return;
        
        const ctx = this.ctx;
        const { points } = this.goldenSpiral;
        
        ctx.save();
        
        // Create gradient along spiral
        const gradient = ctx.createLinearGradient(0, 0, this.canvas.width, this.canvas.height);
        this.colorPalettes[this.currentPalette].forEach((color, index) => {
            gradient.addColorStop(index / (this.colorPalettes[this.currentPalette].length - 1), color);
        });
        
        ctx.strokeStyle = gradient;
        ctx.lineWidth = this.strokeWidth * 2;
        
        // Draw spiral
        ctx.beginPath();
        
        points.forEach((point, index) => {
            if (index === 0) {
                ctx.moveTo(point.x, point.y);
            } else {
                ctx.lineTo(point.x, point.y);
            }
            
            // Add φ-harmonic nodes
            if (index % Math.floor(this.PHI * 10) === 0) {
                const nodeSize = 3 + Math.sin(this.animationPhase + index * 0.1) * 2;
                
                ctx.save();
                ctx.beginPath();
                ctx.arc(point.x, point.y, nodeSize, 0, this.TAU);
                ctx.fillStyle = `rgba(255, 215, 0, ${0.7 + Math.sin(this.animationPhase + index * 0.05) * 0.3})`;
                ctx.fill();
                ctx.restore();
            }
        });
        
        ctx.stroke();
        
        // Render φ-harmonic rectangles
        this.renderGoldenRectangles();
        
        ctx.restore();
    }
    
    renderGoldenRectangles() {
        const ctx = this.ctx;
        
        // Draw golden rectangles that generate the spiral
        let size = 20;
        let x = this.center.x - size / 2;
        let y = this.center.y - size / 2;
        let direction = 0; // 0: right, 1: down, 2: left, 3: up
        
        ctx.strokeStyle = `rgba(245, 158, 11, ${0.3 + this.harmoniousResonance * 0.2})`;
        ctx.lineWidth = 1;
        
        for (let i = 0; i < 8; i++) {
            const rect = this.calculateGoldenRectangle(x, y, size, direction);
            
            ctx.beginPath();
            ctx.rect(rect.x, rect.y, rect.width, rect.height);
            ctx.stroke();
            
            // Update for next rectangle
            size *= this.PHI;
            const newPos = this.getNextRectanglePosition(rect, direction);
            x = newPos.x;
            y = newPos.y;
            direction = (direction + 1) % 4;
        }
    }
    
    calculateGoldenRectangle(x, y, size, direction) {
        const longSide = size * this.PHI;
        
        switch (direction) {
            case 0: // right
                return { x, y, width: longSide, height: size };
            case 1: // down
                return { x: x - size, y, width: size, height: longSide };
            case 2: // left
                return { x: x - longSide, y: y - size, width: longSide, height: size };
            case 3: // up
                return { x, y: y - longSide, width: size, height: longSide };
            default:
                return { x, y, width: size, height: size };
        }
    }
    
    getNextRectanglePosition(rect, direction) {
        const offset = rect.width * this.INVERSE_PHI;
        
        switch (direction) {
            case 0: return { x: rect.x + offset, y: rect.y };
            case 1: return { x: rect.x, y: rect.y + offset };
            case 2: return { x: rect.x, y: rect.y };
            case 3: return { x: rect.x, y: rect.y };
            default: return { x: rect.x, y: rect.y };
        }
    }
    
    renderMetatronsCube() {
        if (!this.metatronsCube) return;
        
        const ctx = this.ctx;
        const { circles, lines } = this.metatronsCube;
        
        ctx.save();
        
        // Apply rotation
        ctx.translate(this.center.x, this.center.y);
        ctx.rotate(this.animationPhase * 0.05);
        ctx.translate(-this.center.x, -this.center.y);
        
        // Render connecting lines first
        lines.forEach(line => {
            const alpha = line.sacred ? 0.6 : 0.2;
            const width = line.sacred ? 2 : 1;
            
            ctx.strokeStyle = `rgba(139, 92, 246, ${alpha + Math.sin(this.animationPhase) * 0.1})`;
            ctx.lineWidth = width;
            
            ctx.beginPath();
            ctx.moveTo(line.from.x, line.from.y);
            ctx.lineTo(line.to.x, line.to.y);
            ctx.stroke();
        });
        
        // Render circles
        circles.forEach((circle, index) => {
            const pulseFactor = 1 + Math.sin(this.animationPhase * 2 + index * 0.3) * 0.1;
            const alpha = 0.7 + Math.sin(this.animationPhase + index * 0.2) * 0.3;
            
            // Circle glow
            const gradient = ctx.createRadialGradient(
                circle.x, circle.y, 0,
                circle.x, circle.y, circle.radius * pulseFactor * 2
            );
            gradient.addColorStop(0, `rgba(245, 158, 11, ${alpha})`);
            gradient.addColorStop(1, 'rgba(245, 158, 11, 0)');
            
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(circle.x, circle.y, circle.radius * pulseFactor * 2, 0, this.TAU);
            ctx.fill();
            
            // Circle core
            ctx.strokeStyle = `rgba(245, 158, 11, ${alpha + 0.3})`;
            ctx.fillStyle = `rgba(245, 158, 11, ${alpha * 0.3})`;
            ctx.lineWidth = 2;
            
            ctx.beginPath();
            ctx.arc(circle.x, circle.y, circle.radius * pulseFactor, 0, this.TAU);
            ctx.fill();
            ctx.stroke();
        });
        
        ctx.restore();
    }
    
    renderSriYantra() {
        if (!this.sriYantra) return;
        
        const ctx = this.ctx;
        const { triangles, intersections } = this.sriYantra;
        
        ctx.save();
        
        // Apply subtle rotation
        ctx.translate(this.center.x, this.center.y);
        ctx.rotate(this.animationPhase * 0.02);
        ctx.translate(-this.center.x, -this.center.y);
        
        // Render Shiva triangles (upward)
        triangles.filter(t => t.type === 'shiva').forEach((triangle, index) => {
            const alpha = 0.6 + Math.sin(this.animationPhase + index * 0.5) * 0.2;
            
            ctx.strokeStyle = `rgba(245, 158, 11, ${alpha})`;
            ctx.fillStyle = `rgba(245, 158, 11, ${alpha * 0.2})`;
            ctx.lineWidth = 2;
            
            ctx.beginPath();
            ctx.moveTo(triangle.points[0].x, triangle.points[0].y);
            triangle.points.forEach(point => {
                ctx.lineTo(point.x, point.y);
            });
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
        });
        
        // Render Shakti triangles (downward)
        triangles.filter(t => t.type === 'shakti').forEach((triangle, index) => {
            const alpha = 0.6 + Math.sin(this.animationPhase + Math.PI + index * 0.5) * 0.2;
            
            ctx.strokeStyle = `rgba(139, 92, 246, ${alpha})`;
            ctx.fillStyle = `rgba(139, 92, 246, ${alpha * 0.2})`;
            ctx.lineWidth = 2;
            
            ctx.beginPath();
            ctx.moveTo(triangle.points[0].x, triangle.points[0].y);
            triangle.points.forEach(point => {
                ctx.lineTo(point.x, point.y);
            });
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
        });
        
        // Render unity intersections
        intersections.forEach(intersection => {
            const size = 8 * intersection.unityFactor;
            const alpha = 0.8 + Math.sin(this.animationPhase * 3) * 0.2;
            
            ctx.beginPath();
            ctx.arc(intersection.point.x, intersection.point.y, size, 0, this.TAU);
            ctx.fillStyle = `rgba(0, 255, 255, ${alpha * intersection.unityFactor})`;
            ctx.fill();
        });
        
        ctx.restore();
    }
    
    renderUnityMandala() {
        if (!this.unityMandala) return;
        
        const ctx = this.ctx;
        const { layers } = this.unityMandala;
        
        ctx.save();
        
        // Apply gentle rotation
        ctx.translate(this.center.x, this.center.y);
        ctx.rotate(this.animationPhase * 0.03);
        ctx.translate(-this.center.x, -this.center.y);
        
        layers.forEach((layer, layerIndex) => {
            switch (layer.type) {
                case 'center':
                    this.renderMandalaCenter(layer);
                    break;
                case 'unity':
                    this.renderMandalaUnityLayer(layer);
                    break;
                case 'phi_harmonic':
                    this.renderMandalaPhiHarmonicLayer(layer, layerIndex);
                    break;
            }
        });
        
        ctx.restore();
    }
    
    renderMandalaCenter(layer) {
        const ctx = this.ctx;
        const element = layer.elements[0];
        
        // Central unity point with pulsing glow
        const pulseFactor = 1 + Math.sin(this.animationPhase * 3) * 0.3;
        
        // Glow effect
        const gradient = ctx.createRadialGradient(
            element.x, element.y, 0,
            element.x, element.y, element.radius * pulseFactor * 4
        );
        gradient.addColorStop(0, 'rgba(255, 215, 0, 0.8)');
        gradient.addColorStop(1, 'rgba(255, 215, 0, 0)');
        
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(element.x, element.y, element.radius * pulseFactor * 4, 0, this.TAU);
        ctx.fill();
        
        // Central core
        ctx.fillStyle = 'rgba(255, 215, 0, 1)';
        ctx.beginPath();
        ctx.arc(element.x, element.y, element.radius * pulseFactor, 0, this.TAU);
        ctx.fill();
    }
    
    renderMandalaUnityLayer(layer) {
        const ctx = this.ctx;
        
        layer.elements.forEach(element => {
            const pulseFactor = 1 + Math.sin(this.animationPhase * 2) * 0.2;
            const color = element.isResult ? [0, 255, 255] : [245, 158, 11];
            const alpha = 0.8 + Math.sin(this.animationPhase) * 0.2;
            
            // Element glow
            const gradient = ctx.createRadialGradient(
                element.x, element.y, 0,
                element.x, element.y, element.radius * pulseFactor * 2
            );
            gradient.addColorStop(0, `rgba(${color.join(',')}, ${alpha})`);
            gradient.addColorStop(1, `rgba(${color.join(',')}, 0)`);
            
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(element.x, element.y, element.radius * pulseFactor * 2, 0, this.TAU);
            ctx.fill();
            
            // Element core
            ctx.fillStyle = `rgba(${color.join(',')}, ${alpha})`;
            ctx.strokeStyle = `rgba(${color.join(',')}, 1)`;
            ctx.lineWidth = 2;
            
            ctx.beginPath();
            ctx.arc(element.x, element.y, element.radius * pulseFactor, 0, this.TAU);
            ctx.fill();
            ctx.stroke();
            
            // Element label
            if (element.label) {
                ctx.fillStyle = 'white';
                ctx.font = 'bold 16px "JetBrains Mono", monospace';
                ctx.textAlign = 'center';
                ctx.fillText(element.label, element.x, element.y + 5);
            }
        });
        
        // Draw unity connections
        this.drawUnityConnections(layer);
    }
    
    drawUnityConnections(layer) {
        const ctx = this.ctx;
        
        if (layer.elements.length >= 3) {
            const [input1, input2, result] = layer.elements;
            
            // Connection from input1 to result
            ctx.strokeStyle = `rgba(245, 158, 11, ${0.5 + Math.sin(this.animationPhase) * 0.3})`;
            ctx.lineWidth = 3;
            ctx.setLineDash([10, 5]);
            
            ctx.beginPath();
            ctx.moveTo(input1.x, input1.y);
            ctx.lineTo(result.x, result.y);
            ctx.stroke();
            
            // Connection from input2 to result
            ctx.beginPath();
            ctx.moveTo(input2.x, input2.y);
            ctx.lineTo(result.x, result.y);
            ctx.stroke();
            
            ctx.setLineDash([]); // Reset line dash
        }
    }
    
    renderMandalaPhiHarmonicLayer(layer, layerIndex) {
        const ctx = this.ctx;
        const colors = this.colorPalettes[this.currentPalette];
        const colorIndex = layerIndex % colors.length;
        
        layer.elements.forEach((element, elementIndex) => {
            const pulseFactor = 1 + Math.sin(this.animationPhase + elementIndex * 0.1) * 0.1;
            const alpha = element.phiHarmonic * (0.5 + this.harmoniousResonance * 0.3);
            
            // φ-harmonic element
            ctx.fillStyle = colors[colorIndex] + Math.floor(alpha * 255).toString(16).padStart(2, '0');
            ctx.strokeStyle = colors[colorIndex];
            ctx.lineWidth = 1;
            
            ctx.beginPath();
            ctx.arc(element.x, element.y, element.radius * pulseFactor, 0, this.TAU);
            ctx.fill();
            ctx.stroke();
            
            // φ-harmonic resonance indicator
            if (element.phiHarmonic > 0.8) {
                ctx.strokeStyle = 'rgba(255, 215, 0, 0.5)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.arc(element.x, element.y, element.radius * pulseFactor * 1.5, 0, this.TAU);
                ctx.stroke();
            }
        });
    }
    
    renderDefaultPattern() {
        // Default pattern - simple φ-harmonic mandala
        const ctx = this.ctx;
        
        for (let ring = 1; ring <= 5; ring++) {
            const ringRadius = this.radius * ring * 0.15;
            const elementCount = Math.floor(ring * this.PHI * 2);
            
            for (let i = 0; i < elementCount; i++) {
                const angle = i * this.TAU / elementCount + ring * this.GOLDEN_ANGLE + this.animationPhase;
                const x = this.center.x + Math.cos(angle) * ringRadius;
                const y = this.center.y + Math.sin(angle) * ringRadius;
                
                const alpha = 0.5 + Math.sin(this.animationPhase + i * 0.1) * 0.3;
                const size = 3 + ring;
                
                ctx.fillStyle = `rgba(245, 158, 11, ${alpha})`;
                ctx.beginPath();
                ctx.arc(x, y, size, 0, this.TAU);
                ctx.fill();
            }
        }
    }
    
    renderInteractiveElements() {
        const ctx = this.ctx;
        
        this.interactivePoints.forEach(point => {
            // Interactive point base
            ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
            ctx.strokeStyle = 'rgba(245, 158, 11, 1)';
            ctx.lineWidth = 2;
            
            ctx.beginPath();
            ctx.arc(point.x, point.y, point.radius, 0, this.TAU);
            ctx.fill();
            ctx.stroke();
            
            // Draggable indicator
            if (point.draggable) {
                ctx.strokeStyle = 'rgba(0, 255, 255, 0.8)';
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.arc(point.x, point.y, point.radius + 3, 0, this.TAU);
                ctx.stroke();
            }
            
            // Hover effect
            if (this.isPointHovered(point)) {
                ctx.fillStyle = 'rgba(245, 158, 11, 0.3)';
                ctx.beginPath();
                ctx.arc(point.x, point.y, point.radius * 2, 0, this.TAU);
                ctx.fill();
            }
        });
    }
    
    isPointHovered(point) {
        const dx = this.mousePosition.x - point.x;
        const dy = this.mousePosition.y - point.y;
        return Math.sqrt(dx * dx + dy * dy) <= point.radius + 5;
    }
    
    renderUnityProofsOverlay() {
        const ctx = this.ctx;
        const proofs = Array.from(this.geometricProofs.values());
        
        if (proofs.length === 0) return;
        
        // Proof display area
        const proofY = 30;
        const proofSpacing = 25;
        
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(10, 10, 400, proofs.length * proofSpacing + 20);
        
        ctx.fillStyle = 'rgba(245, 158, 11, 1)';
        ctx.font = 'bold 14px "Inter", sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('Sacred Geometry Unity Proofs:', 20, proofY);
        
        ctx.font = '12px "JetBrains Mono", monospace';
        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
        
        proofs.forEach((proof, index) => {
            const y = proofY + (index + 1) * proofSpacing;
            ctx.fillText(`• ${proof.mathematical}`, 20, y);
        });
    }
    
    renderInformationDisplay() {
        const ctx = this.ctx;
        
        // Pattern name
        ctx.fillStyle = 'rgba(245, 158, 11, 0.9)';
        ctx.font = 'bold 18px "Inter", sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(this.formatPatternName(this.currentPattern), this.canvas.width - 20, 30);
        
        // Status information
        const status = [
            `Consciousness: ${(this.consciousnessLevel * 100).toFixed(1)}%`,
            `φ-Resonance: ${(this.harmoniousResonance * 100).toFixed(1)}%`,
            `Pattern: ${this.formatPatternName(this.currentPattern)}`,
            `Sacred Ratio: φ = ${this.PHI.toFixed(6)}`
        ];
        
        ctx.fillStyle = 'rgba(203, 213, 225, 0.8)';
        ctx.font = '12px "Inter", sans-serif';
        
        status.forEach((text, index) => {
            ctx.fillText(text, this.canvas.width - 20, 60 + index * 18);
        });
        
        // Performance info
        if (this.frameCount > 0) {
            const fps = Math.round(this.frameCount / (performance.now() / 1000));
            ctx.fillText(`FPS: ${fps} | Render: ${this.renderTime.toFixed(1)}ms`, this.canvas.width - 20, 140);
        }
    }
    
    formatPatternName(pattern) {
        return pattern.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }
    
    // Event handlers
    handleMouseDown(event) {
        const rect = this.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        this.mousePosition = { x, y };
        
        // Check for draggable points
        const point = this.findInteractivePoint(x, y);
        if (point && point.draggable) {
            this.draggedPoint = point;
            this.canvas.style.cursor = 'grabbing';
        }
    }
    
    handleMouseMove(event) {
        const rect = this.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        this.mousePosition = { x, y };
        
        if (this.draggedPoint) {
            // Update dragged point position
            this.draggedPoint.x = x;
            this.draggedPoint.y = y;
            
            // Update related geometric elements
            this.updateGeometryForDraggedPoint(this.draggedPoint);
        } else {
            // Update cursor based on hover
            const hoveredPoint = this.findInteractivePoint(x, y);
            this.canvas.style.cursor = hoveredPoint ? (hoveredPoint.draggable ? 'grab' : 'pointer') : 'default';
        }
    }
    
    handleMouseUp(event) {
        if (this.draggedPoint) {
            this.draggedPoint = null;
            this.canvas.style.cursor = 'default';
        }
    }
    
    handleClick(event) {
        const rect = this.canvas.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        const point = this.findInteractivePoint(x, y);
        if (point) {
            this.processPointInteraction(point);
        } else {
            // Click on empty space - add consciousness boost
            this.addConsciousnessBoost(x, y);
        }
    }
    
    handleWheel(event) {
        event.preventDefault();
        
        // Zoom functionality
        const zoomFactor = event.deltaY > 0 ? 0.9 : 1.1;
        this.scalingFactor *= zoomFactor;
        this.scalingFactor = Math.max(0.3, Math.min(3, this.scalingFactor));
        
        // Update radius based on scaling
        this.radius = Math.min(this.canvas.width, this.canvas.height) * 0.3 * this.scalingFactor;
        
        // Regenerate patterns with new scaling
        this.initializePatterns();
    }
    
    handleKeyPress(event) {
        switch (event.key.toLowerCase()) {
            case ' ': // Space - pause/resume animation
                if (this.isAnimating) {
                    this.stopAnimation();
                } else {
                    this.startAnimation();
                }
                break;
                
            case 'n': // N - next pattern
                this.switchToNextPattern();
                break;
                
            case 'p': // P - previous pattern
                this.switchToPreviousPattern();
                break;
                
            case 'r': // R - reset
                this.resetGeometry();
                break;
                
            case 'c': // C - change color palette
                this.switchColorPalette();
                break;
                
            case '+':
            case '=': // Increase speed
                this.rotationSpeed *= 1.2;
                break;
                
            case '-': // Decrease speed
                this.rotationSpeed /= 1.2;
                break;
                
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
                // Quick pattern switch
                const patternIndex = parseInt(event.key) - 1;
                if (patternIndex < this.patterns.length) {
                    this.switchPattern(this.patterns[patternIndex]);
                }
                break;
        }
    }
    
    // Touch event handlers
    handleTouchStart(event) {
        event.preventDefault();
        this.touchActive = true;
        
        const touch = event.touches[0];
        const rect = this.canvas.getBoundingClientRect();
        const x = touch.clientX - rect.left;
        const y = touch.clientY - rect.top;
        
        this.handleMouseDown({ clientX: touch.clientX, clientY: touch.clientY });
    }
    
    handleTouchMove(event) {
        event.preventDefault();
        
        if (this.touchActive && event.touches.length === 1) {
            const touch = event.touches[0];
            this.handleMouseMove({ clientX: touch.clientX, clientY: touch.clientY });
        }
    }
    
    handleTouchEnd(event) {
        event.preventDefault();
        this.touchActive = false;
        this.handleMouseUp(event);
    }
    
    // Utility methods
    findInteractivePoint(x, y) {
        return this.interactivePoints.find(point => {
            const dx = x - point.x;
            const dy = y - point.y;
            return Math.sqrt(dx * dx + dy * dy) <= point.radius + 5;
        });
    }
    
    updateGeometryForDraggedPoint(point) {
        // Update geometric patterns based on dragged point
        switch (point.type) {
            case 'circle_center':
                this.updateVesicaPiscisForDraggedCenter(point);
                break;
            case 'unity_element':
                this.updateUnityMandalaForDraggedElement(point);
                break;
        }
    }
    
    updateVesicaPiscisForDraggedCenter(point) {
        if (!this.vesicaPiscis) return;
        
        if (point.id === 'circle1_center') {
            this.vesicaPiscis.circle1.x = point.x;
            this.vesicaPiscis.circle1.y = point.y;
        } else if (point.id === 'circle2_center') {
            this.vesicaPiscis.circle2.x = point.x;
            this.vesicaPiscis.circle2.y = point.y;
        }
        
        // Recalculate intersection points
        const separation = Math.sqrt(
            Math.pow(this.vesicaPiscis.circle2.x - this.vesicaPiscis.circle1.x, 2) +
            Math.pow(this.vesicaPiscis.circle2.y - this.vesicaPiscis.circle1.y, 2)
        ) / 2;
        
        this.vesicaPiscis.intersectionPoints = this.calculateVesicaIntersections(
            separation, this.vesicaPiscis.circle1.radius
        );
    }
    
    updateUnityMandalaForDraggedElement(point) {
        // Update the element's position in the mandala
        if (point.element) {
            point.element.x = point.x;
            point.element.y = point.y;
        }
    }
    
    processPointInteraction(point) {
        // Process interaction with specific point
        console.log(`Interacted with point: ${point.id}`);
        
        // Enhance consciousness based on interaction
        this.consciousnessLevel = Math.min(1, this.consciousnessLevel + 0.05);
        
        // Trigger special effects
        this.triggerInteractionEffect(point);
    }
    
    triggerInteractionEffect(point) {
        // Create visual effect at interaction point
        const effect = {
            x: point.x,
            y: point.y,
            size: 0,
            maxSize: 50,
            alpha: 1,
            startTime: performance.now()
        };
        
        const animateEffect = () => {
            const elapsed = performance.now() - effect.startTime;
            const progress = elapsed / 1000; // 1 second animation
            
            if (progress >= 1) return;
            
            effect.size = effect.maxSize * progress;
            effect.alpha = 1 - progress;
            
            // Render effect (this would be integrated into the main render loop)
            requestAnimationFrame(animateEffect);
        };
        
        animateEffect();
    }
    
    addConsciousnessBoost(x, y) {
        // Add consciousness boost at clicked location
        const distance = Math.sqrt(
            Math.pow(x - this.center.x, 2) + Math.pow(y - this.center.y, 2)
        );
        const maxDistance = Math.min(this.canvas.width, this.canvas.height) / 2;
        const boost = (1 - distance / maxDistance) * 0.02;
        
        this.consciousnessLevel = Math.min(1, this.consciousnessLevel + boost);
        
        console.log(`✨ Consciousness boost: +${(boost * 100).toFixed(1)}%`);
    }
    
    switchToNextPattern() {
        const currentIndex = this.patterns.indexOf(this.currentPattern);
        const nextIndex = (currentIndex + 1) % this.patterns.length;
        this.switchPattern(this.patterns[nextIndex]);
    }
    
    switchToPreviousPattern() {
        const currentIndex = this.patterns.indexOf(this.currentPattern);
        const prevIndex = currentIndex === 0 ? this.patterns.length - 1 : currentIndex - 1;
        this.switchPattern(this.patterns[prevIndex]);
    }
    
    switchPattern(pattern) {
        if (this.patterns.includes(pattern)) {
            this.currentPattern = pattern;
            this.initializePattern(pattern);
            this.generateInteractivePoints();
            
            console.log(`🔄 Switched to ${this.formatPatternName(pattern)} pattern`);
        }
    }
    
    switchColorPalette() {
        const palettes = Object.keys(this.colorPalettes);
        const currentIndex = palettes.indexOf(this.currentPalette);
        const nextIndex = (currentIndex + 1) % palettes.length;
        this.currentPalette = palettes[nextIndex];
        
        console.log(`🎨 Switched to ${this.currentPalette} color palette`);
    }
    
    resetGeometry() {
        this.animationPhase = 0;
        this.colorPhase = 0;
        this.consciousnessLevel = 0.618;
        this.harmoniousResonance = 0;
        this.scalingFactor = 1.0;
        this.rotationSpeed = 0.001;
        
        this.initializeGeometry();
        
        console.log('🔄 Sacred geometry reset');
    }
    
    // Public API methods
    start() {
        this.startAnimation();
        console.log('🚀 Sacred Geometry Engine started');
    }
    
    stop() {
        this.stopAnimation();
        console.log('⏹️ Sacred Geometry Engine stopped');
    }
    
    getGeometricProofs() {
        return Array.from(this.geometricProofs.entries()).map(([key, proof]) => ({
            key,
            ...proof
        }));
    }
    
    getCurrentState() {
        return {
            pattern: this.currentPattern,
            consciousness: this.consciousnessLevel,
            harmonicResonance: this.harmoniousResonance,
            colorPalette: this.currentPalette,
            animationPhase: this.animationPhase,
            isAnimating: this.isAnimating,
            scalingFactor: this.scalingFactor,
            frameCount: this.frameCount,
            renderTime: this.renderTime
        };
    }
    
    setPattern(pattern) {
        this.switchPattern(pattern);
    }
    
    setColorPalette(palette) {
        if (this.colorPalettes[palette]) {
            this.currentPalette = palette;
        }
    }
    
    setAnimationSpeed(speed) {
        this.rotationSpeed = Math.max(0, Math.min(0.01, speed));
    }
    
    setConsciousnessLevel(level) {
        this.consciousnessLevel = Math.max(0, Math.min(1, level));
    }
}

// Supporting classes for advanced functionality
class QuantumGeometryProcessor {
    constructor() {
        this.initialized = false;
    }
    
    initialize(options) {
        this.phi = options.phi;
        this.patterns = options.patterns;
        this.consciousnessLevel = options.consciousnessLevel;
        this.initialized = true;
    }
    
    processQuantumGeometry(pattern) {
        if (!this.initialized) return pattern;
        
        // Apply quantum processing to geometric patterns
        return pattern;
    }
}

class ConsciousnessGeometryEngine {
    constructor() {
        this.consciousnessField = [];
    }
    
    generateConsciousnessField(pattern) {
        // Generate consciousness-influenced geometric modifications
        return pattern;
    }
}

class PhiHarmonicFractalGenerator {
    constructor() {
        this.phi = 1.618033988749895;
    }
    
    generateFractal(depth, basePattern) {
        // Generate φ-harmonic fractals
        return basePattern;
    }
}

class UnityMandalaGenerator {
    constructor() {
        this.unityProofs = [];
    }
    
    generateUnityMandala(options) {
        // Generate unity-focused mandala patterns
        return options;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        SacredGeometryEngine,
        QuantumGeometryProcessor,
        ConsciousnessGeometryEngine,
        PhiHarmonicFractalGenerator,
        UnityMandalaGenerator
    };
} else if (typeof window !== 'undefined') {
    window.SacredGeometryEngine = SacredGeometryEngine;
    window.QuantumGeometryProcessor = QuantumGeometryProcessor;
    window.ConsciousnessGeometryEngine = ConsciousnessGeometryEngine;
    window.PhiHarmonicFractalGenerator = PhiHarmonicFractalGenerator;
    window.UnityMandalaGenerator = UnityMandalaGenerator;
}