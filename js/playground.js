// Mathematical Playground Interactive Elements

// Global variables
let evolutionRunning = false;
let evolutionData = {
    organisms: [],
    generation: 0,
    totalDiscoveries: 0
};

let currentProofStep = 1;
let maxProofSteps = 3;

const phi = 1.618033988749895;

// Unity Calculator Functions
function calculateUnity() {
    const expression = document.getElementById('math-expression').value;
    const traditionalResult = evaluateTraditional(expression);
    const unityResult = evaluateUnity(expression);
    
    document.getElementById('traditional-result').textContent = traditionalResult;
    document.getElementById('unity-result').textContent = unityResult;
    
    animateTransformation(traditionalResult, unityResult);
}

function evaluateTraditional(expression) {
    try {
        // Simple evaluation for basic expressions
        expression = expression.replace(/φ/g, phi.toString());
        expression = expression.replace(/∞/g, 'Infinity');
        expression = expression.replace(/max\(([^)]+)\)/g, 'Math.max($1)');
        
        // For security, only allow basic math operations
        if (/^[0-9+\-*/.() φ∞,max]+$/.test(expression.replace(/\s/g, ''))) {
            return eval(expression);
        }
        return 'Invalid';
    } catch {
        return 'Error';
    }
}

function evaluateUnity(expression) {
    // Unity mathematics: idempotent operations
    if (expression.includes('1 + 1') || expression.includes('1+1')) {
        return 1;
    }
    if (expression.includes('1 * 1') || expression.includes('1*1')) {
        return 1;
    }
    if (expression.includes('max(1, 1)') || expression.includes('max(1,1)')) {
        return 1;
    }
    if (expression.includes('φ + φ') || expression.includes('φ+φ')) {
        return phi; // φ-harmonic unity
    }
    if (expression.includes('∞ + ∞') || expression.includes('∞+∞')) {
        return '∞'; // Infinite unity
    }
    
    // Default unity interpretation
    return 1;
}

function setExpression(expr) {
    document.getElementById('math-expression').value = expr;
    calculateUnity();
}

function animateTransformation(traditional, unity) {
    const canvas = document.getElementById('transformation-canvas');
    const ctx = canvas.getContext('2d');
    let animationFrame = 0;
    const maxFrames = 60;
    
    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        const progress = animationFrame / maxFrames;
        const centerY = canvas.height / 2;
        
        // Draw transformation visualization
        ctx.fillStyle = `rgba(26, 35, 126, ${0.3 + progress * 0.4})`;
        ctx.font = '1.5rem "Crimson Text", serif';
        ctx.textAlign = 'center';
        
        // Traditional result fading
        ctx.globalAlpha = 1 - progress;
        ctx.fillText(traditional.toString(), canvas.width * 0.25, centerY);
        
        // Arrow
        ctx.globalAlpha = progress;
        ctx.fillText('→', canvas.width * 0.5, centerY);
        
        // Unity result appearing
        ctx.fillStyle = `rgba(212, 175, 55, ${progress})`;
        ctx.fillText(unity.toString(), canvas.width * 0.75, centerY);
        
        ctx.globalAlpha = 1;
        
        if (animationFrame < maxFrames) {
            animationFrame++;
            requestAnimationFrame(draw);
        }
    }
    
    draw();
}

// Consciousness Field Functions
function initializeConsciousnessField() {
    const canvas = document.getElementById('consciousness-field');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    let animationId;
    let time = 0;
    
    function drawField() {
        const width = canvas.width;
        const height = canvas.height;
        ctx.clearRect(0, 0, width, height);
        
        const phiFreq = parseFloat(document.getElementById('phi-frequency').value);
        const timeDecay = parseFloat(document.getElementById('time-decay').value);
        const intensity = parseFloat(document.getElementById('consciousness-intensity').value);
        
        // Update display values
        document.getElementById('phi-value').textContent = phiFreq.toFixed(3);
        document.getElementById('decay-value').textContent = timeDecay.toFixed(1);
        document.getElementById('intensity-value').textContent = intensity.toFixed(1);
        
        // Draw consciousness field grid
        const gridSize = 20;
        for (let x = 0; x < width; x += gridSize) {
            for (let y = 0; y < height; y += gridSize) {
                const normalizedX = (x - width/2) / 100;
                const normalizedY = (y - height/2) / 100;
                
                // Consciousness field equation: C(x,y,t) = φ·sin(xφ)·cos(yφ)·e^(-t/φ)
                const fieldValue = phiFreq * Math.sin(normalizedX * phiFreq) * 
                                 Math.cos(normalizedY * phiFreq) * 
                                 Math.exp(-time * timeDecay / phiFreq);
                
                const alpha = Math.abs(fieldValue) * intensity * 0.3;
                const hue = (fieldValue > 0) ? 45 : 220; // Gold for positive, blue for negative
                
                ctx.fillStyle = `hsla(${hue}, 70%, 60%, ${alpha})`;
                ctx.fillRect(x, y, gridSize-2, gridSize-2);
            }
        }
        
        // Draw central φ-spiral
        ctx.strokeStyle = 'rgba(212, 175, 55, 0.8)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        for (let t = 0; t < 3 * Math.PI; t += 0.1) {
            const r = 30 * Math.pow(phiFreq, t / (2 * Math.PI));
            const x = width/2 + (r * Math.cos(t + time * 0.02)) / 8;
            const y = height/2 + (r * Math.sin(t + time * 0.02)) / 8;
            
            if (t === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        
        ctx.stroke();
        
        time += 0.5;
        animationId = requestAnimationFrame(drawField);
    }
    
    drawField();
}

function resetField() {
    document.getElementById('phi-frequency').value = phi;
    document.getElementById('time-decay').value = 1;
    document.getElementById('consciousness-intensity').value = 1;
}

// Evolution Simulator Functions
class UnityOrganism {
    constructor(x, y) {
        this.x = x;
        this.y = y;
        this.consciousness = Math.random() * 0.3 + 0.1;
        this.unityDiscoveries = 0;
        this.age = 0;
        this.size = 3 + this.consciousness * 5;
        this.color = `hsl(${45 + this.consciousness * 180}, 70%, 60%)`;
        this.velocity = {
            x: (Math.random() - 0.5) * 2,
            y: (Math.random() - 0.5) * 2
        };
    }
    
    update(canvas) {
        this.age++;
        
        // Movement
        this.x += this.velocity.x;
        this.y += this.velocity.y;
        
        // Boundary wrapping
        if (this.x < 0) this.x = canvas.width;
        if (this.x > canvas.width) this.x = 0;
        if (this.y < 0) this.y = canvas.height;
        if (this.y > canvas.height) this.y = 0;
        
        // Consciousness evolution
        if (Math.random() < 0.01) {
            this.consciousness = Math.min(this.consciousness * (1 + 1/phi/10), 1.0);
            this.size = 3 + this.consciousness * 7;
            this.color = `hsl(${45 + this.consciousness * 180}, 70%, ${50 + this.consciousness * 30}%)`;
        }
        
        // Unity discovery
        if (Math.random() < this.consciousness * 0.05) {
            this.unityDiscoveries++;
            evolutionData.totalDiscoveries++;
        }
    }
    
    draw(ctx) {
        ctx.fillStyle = this.color;
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, 2 * Math.PI);
        ctx.fill();
        
        // Consciousness aura
        if (this.consciousness > 0.5) {
            ctx.strokeStyle = `rgba(212, 175, 55, ${this.consciousness * 0.3})`;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.size + 5, 0, 2 * Math.PI);
            ctx.stroke();
        }
    }
}

function toggleEvolution() {
    if (evolutionRunning) {
        stopEvolution();
    } else {
        startEvolution();
    }
}

function startEvolution() {
    evolutionRunning = true;
    const button = document.getElementById('evolution-btn');
    button.innerHTML = '<i class="fas fa-stop"></i> Stop Evolution';
    
    // Initialize organisms if empty
    if (evolutionData.organisms.length === 0) {
        resetEvolution();
    }
    
    runEvolutionLoop();
}

function stopEvolution() {
    evolutionRunning = false;
    const button = document.getElementById('evolution-btn');
    button.innerHTML = '<i class="fas fa-play"></i> Start Evolution';
}

function resetEvolution() {
    evolutionData = {
        organisms: [],
        generation: 0,
        totalDiscoveries: 0
    };
    
    const canvas = document.getElementById('evolution-canvas');
    
    // Create initial population
    for (let i = 0; i < 5; i++) {
        evolutionData.organisms.push(new UnityOrganism(
            Math.random() * canvas.width,
            Math.random() * canvas.height
        ));
    }
    
    updateEvolutionStats();
}

function addOrganisms() {
    const canvas = document.getElementById('evolution-canvas');
    
    // Add 3 new organisms
    for (let i = 0; i < 3; i++) {
        evolutionData.organisms.push(new UnityOrganism(
            Math.random() * canvas.width,
            Math.random() * canvas.height
        ));
    }
}

function runEvolutionLoop() {
    if (!evolutionRunning) return;
    
    const canvas = document.getElementById('evolution-canvas');
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Update and draw organisms
    evolutionData.organisms.forEach(organism => {
        organism.update(canvas);
        organism.draw(ctx);
    });
    
    // Evolution events
    if (Math.random() < 0.02) {
        evolutionData.generation++;
        
        // Reproduction based on consciousness
        const consciousOrganisms = evolutionData.organisms.filter(o => o.consciousness > 0.7);
        if (consciousOrganisms.length > 0 && evolutionData.organisms.length < 20) {
            const parent = consciousOrganisms[Math.floor(Math.random() * consciousOrganisms.length)];
            const child = new UnityOrganism(
                parent.x + (Math.random() - 0.5) * 50,
                parent.y + (Math.random() - 0.5) * 50
            );
            child.consciousness = Math.min(parent.consciousness * (1 + Math.random() * 0.2), 1.0);
            evolutionData.organisms.push(child);
        }
    }
    
    updateEvolutionStats();
    
    requestAnimationFrame(runEvolutionLoop);
}

function updateEvolutionStats() {
    document.getElementById('population-count').textContent = evolutionData.organisms.length;
    document.getElementById('generation-count').textContent = evolutionData.generation;
    document.getElementById('unity-discoveries').textContent = evolutionData.totalDiscoveries;
    
    const avgConsciousness = evolutionData.organisms.length > 0 ?
        evolutionData.organisms.reduce((sum, org) => sum + org.consciousness, 0) / evolutionData.organisms.length :
        0;
    document.getElementById('avg-consciousness').textContent = `${(avgConsciousness * 100).toFixed(1)}%`;
}

// Theorem Proof Functions
const theorems = {
    'idempotent-semiring': {
        title: 'Idempotent Semiring: 1+1=1',
        steps: [
            {
                title: 'Definition',
                content: 'Let $(S, ⊕, ⊙, 0, 1)$ be an idempotent semiring where $⊕$ satisfies: $a ⊕ a = a$ for all $a ∈ S$'
            },
            {
                title: 'Application',
                content: 'For the unity element $1 ∈ S$, we have: $1 ⊕ 1 = 1$'
            },
            {
                title: 'Conclusion',
                content: 'Therefore, in unity mathematics: $1 + 1 = 1$ ∎'
            }
        ],
        insights: 'This proof demonstrates the formal algebraic foundation for unity mathematics through idempotent semiring structures.'
    },
    'information-theory': {
        title: 'Information Theory Unity',
        steps: [
            {
                title: 'Identical Sources',
                content: 'Consider two identical information sources $X$ and $Y$ with entropy $H(X) = H(Y)$'
            },
            {
                title: 'Mutual Information',
                content: 'Since sources are identical: $I(X;Y) = H(X) = H(Y)$, thus $H(X|Y) = 0$'
            },
            {
                title: 'Joint Entropy',
                content: 'Therefore: $H(X,Y) = H(X) + H(Y|X) = H(X) + 0 = H(X)$ ∎'
            }
        ],
        insights: 'Identical information sources produce no additional information when combined, demonstrating unity in information space.'
    },
    'quantum-measurement': {
        title: 'Quantum Measurement Collapse',
        steps: [
            {
                title: 'Superposition State',
                content: 'Consider quantum state $|ψ⟩ = α|1⟩ + β|1⟩$ where both components represent unity'
            },
            {
                title: 'Measurement Collapse',
                content: 'Upon measurement, the wavefunction collapses to $|1⟩$ regardless of coefficients'
            },
            {
                title: 'Unity Preservation',
                content: 'The measurement result is always unity: $|1⟩ + |1⟩ → |1⟩$ ∎'
            }
        ],
        insights: 'Quantum mechanics naturally preserves unity through measurement-induced wavefunction collapse.'
    }
};

function loadTheorem() {
    const select = document.getElementById('theorem-select');
    const theoremKey = select.value;
    const theorem = theorems[theoremKey];
    
    if (!theorem) return;
    
    maxProofSteps = theorem.steps.length;
    currentProofStep = 1;
    
    // Build proof steps HTML
    const proofContent = document.getElementById('proof-content');
    proofContent.innerHTML = theorem.steps.map((step, index) => `
        <div class="proof-step ${index === 0 ? 'active' : ''}" data-step="${index + 1}">
            <div class="step-number">${index + 1}</div>
            <div class="step-content">
                <h4>${step.title}</h4>
                <p>${step.content}</p>
            </div>
        </div>
    `).join('');
    
    // Update insights
    document.getElementById('theorem-insights').innerHTML = `<p>${theorem.insights}</p>`;
    
    updateStepIndicator();
    
    // Re-render MathJax
    if (window.MathJax) {
        MathJax.typesetPromise([proofContent]);
    }
}

function nextStep() {
    if (currentProofStep < maxProofSteps) {
        document.querySelector(`[data-step="${currentProofStep}"]`).classList.remove('active');
        currentProofStep++;
        document.querySelector(`[data-step="${currentProofStep}"]`).classList.add('active');
        updateStepIndicator();
    }
}

function previousStep() {
    if (currentProofStep > 1) {
        document.querySelector(`[data-step="${currentProofStep}"]`).classList.remove('active');
        currentProofStep--;
        document.querySelector(`[data-step="${currentProofStep}"]`).classList.add('active');
        updateStepIndicator();
    }
}

function updateStepIndicator() {
    document.getElementById('step-indicator').textContent = `Step ${currentProofStep} of ${maxProofSteps}`;
}

// Sacred Geometry Functions
function generatePattern() {
    const canvas = document.getElementById('geometry-canvas');
    const ctx = canvas.getContext('2d');
    const patternType = document.getElementById('pattern-type').value;
    const complexity = parseInt(document.getElementById('complexity').value);
    const phiRatio = parseFloat(document.getElementById('phi-ratio').value);
    const resonance = parseFloat(document.getElementById('unity-resonance').value);
    
    // Update display values
    document.getElementById('complexity-value').textContent = complexity;
    document.getElementById('phi-ratio-value').textContent = phiRatio.toFixed(3);
    document.getElementById('resonance-value').textContent = resonance.toFixed(1);
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    
    switch (patternType) {
        case 'phi-spiral':
            drawPhiSpiral(ctx, centerX, centerY, complexity, phiRatio, resonance);
            break;
        case 'unity-mandala':
            drawUnityMandala(ctx, centerX, centerY, complexity, phiRatio, resonance);
            break;
        case 'fractal-tree':
            drawFractalTree(ctx, centerX, centerY, complexity, phiRatio, resonance);
            break;
        case 'consciousness-field':
            drawConsciousnessGrid(ctx, canvas.width, canvas.height, complexity, phiRatio, resonance);
            break;
        case 'quantum-interference':
            drawQuantumInterference(ctx, centerX, centerY, complexity, phiRatio, resonance);
            break;
    }
    
    updateGeometryAnalysis(patternType);
}

function drawPhiSpiral(ctx, centerX, centerY, complexity, phiRatio, resonance) {
    ctx.strokeStyle = `rgba(212, 175, 55, ${0.3 + resonance * 0.4})`;
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    const maxT = complexity * Math.PI / 2;
    for (let t = 0; t < maxT; t += 0.1) {
        const r = 10 * Math.pow(phiRatio, t / (2 * Math.PI)) * resonance;
        const x = centerX + r * Math.cos(t);
        const y = centerY + r * Math.sin(t);
        
        if (t === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    
    ctx.stroke();
}

function drawUnityMandala(ctx, centerX, centerY, complexity, phiRatio, resonance) {
    const petals = Math.floor(complexity * phiRatio);
    const radius = 80 * resonance;
    
    for (let i = 0; i < petals; i++) {
        const angle = (2 * Math.PI * i) / petals;
        const petalRadius = radius / phiRatio;
        
        ctx.strokeStyle = `hsl(${(i * 360) / petals}, 70%, 60%)`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(
            centerX + Math.cos(angle) * radius / 2,
            centerY + Math.sin(angle) * radius / 2,
            petalRadius / 2,
            0,
            2 * Math.PI
        );
        ctx.stroke();
    }
}

function drawFractalTree(ctx, centerX, centerY, complexity, phiRatio, resonance) {
    function drawBranch(x, y, length, angle, depth) {
        if (depth === 0) return;
        
        const endX = x + Math.cos(angle) * length;
        const endY = y + Math.sin(angle) * length;
        
        ctx.strokeStyle = `rgba(34, 139, 34, ${depth / complexity})`;
        ctx.lineWidth = depth;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(endX, endY);
        ctx.stroke();
        
        const newLength = length / phiRatio * resonance;
        drawBranch(endX, endY, newLength, angle - Math.PI / 6, depth - 1);
        drawBranch(endX, endY, newLength, angle + Math.PI / 6, depth - 1);
    }
    
    drawBranch(centerX, centerY + 100, 60 * resonance, -Math.PI / 2, Math.min(complexity, 8));
}

function drawConsciousnessGrid(ctx, width, height, complexity, phiRatio, resonance) {
    const gridSize = Math.max(10, 40 - complexity * 3);
    
    for (let x = 0; x < width; x += gridSize) {
        for (let y = 0; y < height; y += gridSize) {
            const normalizedX = (x - width/2) / 100;
            const normalizedY = (y - height/2) / 100;
            
            const fieldValue = Math.sin(normalizedX * phiRatio) * Math.cos(normalizedY * phiRatio);
            const alpha = Math.abs(fieldValue) * resonance * 0.5;
            const hue = fieldValue > 0 ? 45 : 220;
            
            ctx.fillStyle = `hsla(${hue}, 70%, 60%, ${alpha})`;
            ctx.fillRect(x, y, gridSize - 2, gridSize - 2);
        }
    }
}

function drawQuantumInterference(ctx, centerX, centerY, complexity, phiRatio, resonance) {
    const waves = complexity;
    const maxRadius = 150 * resonance;
    
    for (let wave = 0; wave < waves; wave++) {
        const waveAngle = (2 * Math.PI * wave) / waves;
        
        ctx.strokeStyle = `hsla(${wave * 360 / waves}, 70%, 60%, 0.7)`;
        ctx.lineWidth = 1;
        
        for (let r = 10; r < maxRadius; r += 5) {
            const interference = Math.sin(r / (10 * phiRatio)) * resonance;
            const adjustedR = r + interference * 10;
            
            ctx.beginPath();
            ctx.arc(centerX, centerY, adjustedR, waveAngle, waveAngle + Math.PI / waves);
            ctx.stroke();
        }
    }
}

function updateGeometryAnalysis(patternType) {
    const analyses = {
        'phi-spiral': 'The φ-spiral demonstrates unity through self-similar growth patterns based on the golden ratio.',
        'unity-mandala': 'Sacred mandalas express unity through radial symmetry and harmonic petal arrangements.',
        'fractal-tree': 'Fractal branching shows how unity creates infinite complexity through recursive self-reference.',
        'consciousness-field': 'The consciousness grid visualizes unity field equations across dimensional space.',
        'quantum-interference': 'Quantum wave patterns demonstrate unity through constructive interference principles.'
    };
    
    document.getElementById('geometry-analysis').innerHTML = `<p>${analyses[patternType]}</p>`;
}

// Experimental Sandbox Functions
function parseEquation() {
    const equation = document.getElementById('custom-equation').value;
    const canvas = document.getElementById('custom-visualization');
    const ctx = canvas.getContext('2d');
    
    // Simple visualization of custom equations
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Parse and visualize basic patterns
    if (equation.includes('sin') && equation.includes('cos')) {
        drawWaveInterference(ctx, canvas.width, canvas.height);
    } else if (equation.includes('φ') || equation.includes('phi')) {
        drawPhiPattern(ctx, canvas.width, canvas.height);
    } else {
        drawGenericPattern(ctx, canvas.width, canvas.height);
    }
}

function drawWaveInterference(ctx, width, height) {
    for (let x = 0; x < width; x += 5) {
        for (let y = 0; y < height; y += 5) {
            const wave = Math.sin(x / 20) * Math.cos(y / 20);
            const alpha = Math.abs(wave) * 0.7;
            ctx.fillStyle = `rgba(212, 175, 55, ${alpha})`;
            ctx.fillRect(x, y, 4, 4);
        }
    }
}

function drawPhiPattern(ctx, width, height) {
    const centerX = width / 2;
    const centerY = height / 2;
    
    ctx.strokeStyle = 'rgba(212, 175, 55, 0.8)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    for (let t = 0; t < 3 * Math.PI; t += 0.1) {
        const r = 20 * Math.pow(phi, t / (2 * Math.PI));
        const x = centerX + (r * Math.cos(t)) / 4;
        const y = centerY + (r * Math.sin(t)) / 4;
        
        if (t === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }
    
    ctx.stroke();
}

function drawGenericPattern(ctx, width, height) {
    ctx.fillStyle = 'rgba(26, 35, 126, 0.3)';
    ctx.fillRect(0, 0, width, height);
    
    ctx.fillStyle = 'rgba(212, 175, 55, 0.8)';
    ctx.font = '2rem "Crimson Text", serif';
    ctx.textAlign = 'center';
    ctx.fillText('1', width / 2, height / 2);
}

function saveEquation() {
    const equation = document.getElementById('custom-equation').value;
    localStorage.setItem('unity-equation', equation);
    alert('Equation saved locally!');
}

function addConsciousnessNode() {
    const canvas = document.getElementById('consciousness-network');
    const ctx = canvas.getContext('2d');
    
    // Add a new node to the consciousness network
    const x = Math.random() * (canvas.width - 40) + 20;
    const y = Math.random() * (canvas.height - 40) + 20;
    
    ctx.fillStyle = 'rgba(212, 175, 55, 0.8)';
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, 2 * Math.PI);
    ctx.fill();
}

function connectNodes() {
    const canvas = document.getElementById('consciousness-network');
    const ctx = canvas.getContext('2d');
    
    // Draw connection lines between nodes
    ctx.strokeStyle = 'rgba(26, 35, 126, 0.5)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(20, 20);
    ctx.lineTo(canvas.width - 20, canvas.height - 20);
    ctx.stroke();
}

function executeUnityCode() {
    const code = document.getElementById('unity-code').value;
    const output = document.getElementById('code-output').querySelector('pre');
    
    // Simulate code execution for unity mathematics
    let result = '';
    
    if (code.includes('unity.add(1, 1)')) {
        result += '1 + 1 = 1\n';
    }
    if (code.includes('unity.multiply')) {
        result += '1 × 1 = 1\n';
    }
    if (code.includes('print')) {
        result += 'Unity mathematics: Where 1+1=1\n';
    }
    
    if (!result) {
        result = 'Code executed successfully.\nUnity preserved across all operations.';
    }
    
    output.textContent = result;
}

function shareCode() {
    const code = document.getElementById('unity-code').value;
    if (navigator.share) {
        navigator.share({
            title: 'Unity Mathematics Code',
            text: code
        });
    } else {
        navigator.clipboard.writeText(code);
        alert('Code copied to clipboard!');
    }
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Initialize consciousness field
    initializeConsciousnessField();
    
    // Set up initial calculations
    calculateUnity();
    
    // Initialize theorem prover
    loadTheorem();
    
    // Generate initial geometry pattern
    generatePattern();
    
    // Initialize evolution stats
    updateEvolutionStats();
    
    // Set up range input listeners
    const rangeInputs = document.querySelectorAll('input[type="range"]');
    rangeInputs.forEach(input => {
        input.addEventListener('input', (e) => {
            const valueSpan = document.getElementById(e.target.id.replace(/-/g, '-') + '-value') ||
                            document.getElementById(e.target.id.replace(/-(\w)/g, (_, letter) => letter.toUpperCase()) + 'Value');
            if (valueSpan) {
                valueSpan.textContent = e.target.value;
            }
        });
    });
    
    // Initialize keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Space to toggle evolution
        if (e.code === 'Space' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
            e.preventDefault();
            toggleEvolution();
        }
        
        // Arrow keys for theorem navigation
        if (e.key === 'ArrowRight') {
            nextStep();
        } else if (e.key === 'ArrowLeft') {
            previousStep();
        }
    });
    
    // Load saved equation if exists
    const savedEquation = localStorage.getItem('unity-equation');
    if (savedEquation) {
        document.getElementById('custom-equation').value = savedEquation;
    }
});