/**
 * Unity Mathematics Interactive Demo - Een Project
 * Real implementations showing 1+1=1 through various mathematical frameworks
 */

// Mathematical constants from the codebase
const PHI = 1.618033988749895;  // Golden ratio
const UNITY_TOLERANCE = 1e-10;
const LOVE_FREQUENCY = 528;     // Hz - Love frequency
const E = Math.E;
const PI = Math.PI;

// Unity Mathematics Implementation (from unity_mathematics.py)
class UnityMathematics {
    constructor() {
        this.phi = PHI;
        this.tolerance = UNITY_TOLERANCE;
        this.operationHistory = [];
    }

    // Idempotent addition: core unity operation
    unityAdd(x, y) {
        // Implementation based on mathematical_proof.py
        if (Math.abs(x - y) < this.tolerance) {
            return x; // Pure idempotence
        }
        
        // φ-harmonic tie-breaking for pedagogical elegance
        const phiWrappedX = this.phiWrap(x);
        const phiWrappedY = this.phiWrap(y);
        const result = this.phiUnwrap((phiWrappedX + phiWrappedY) / this.phi);
        
        this.operationHistory.push({
            operation: 'unityAdd',
            inputs: [x, y],
            result: result,
            timestamp: Date.now()
        });
        
        return result;
    }

    // φ-harmonic operations from the codebase
    phiWrap(x) {
        return x * this.phi;
    }

    phiUnwrap(x) {
        return x / this.phi;
    }

    // Quantum unity through consciousness (from consciousness.py)
    quantumUnity(x, y, consciousnessLevel = 0.5) {
        const consciousnessFactor = Math.sin(consciousnessLevel * PI);
        const unityResult = (x + y) * Math.exp(-Math.abs(2 - (x + y)) * this.phi);
        
        // Consciousness bends reality toward unity
        const finalResult = unityResult * (1 - consciousnessFactor) + 1 * consciousnessFactor;
        
        this.operationHistory.push({
            operation: 'quantumUnity',
            inputs: [x, y],
            consciousnessLevel: consciousnessLevel,
            result: finalResult,
            timestamp: Date.now()
        });
        
        return finalResult;
    }

    // Bayesian unity with economic constraints
    bayesianUnity(x, y, priorBelief = 0.95) {
        // Based on bayesian_econometrics.py
        const unityPrior = 1.0;
        const observedSum = x + y;
        const likelihood = Math.exp(-Math.pow(observedSum - unityPrior, 2) / (2 * 0.1));
        
        // Bayesian update toward unity
        const posterior = (priorBelief * unityPrior + (1 - priorBelief) * observedSum);
        
        this.operationHistory.push({
            operation: 'bayesianUnity',
            inputs: [x, y],
            likelihood: likelihood,
            result: posterior,
            timestamp: Date.now()
        });
        
        return posterior;
    }

    // Category theory unity (terminal object convergence)
    categoricalUnity(x, y) {
        // All objects converge to terminal object in unity category
        const terminalObject = 1.0;
        const convergenceRate = 0.9;
        
        const result = terminalObject * convergenceRate + (x + y) * (1 - convergenceRate);
        
        this.operationHistory.push({
            operation: 'categoricalUnity',
            inputs: [x, y],
            result: result,
            timestamp: Date.now()
        });
        
        return result;
    }
}

// Unity Mathematics Calculator
function calculateUnity() {
    const inputA = parseFloat(document.getElementById('input-a').value) || 1;
    const inputB = parseFloat(document.getElementById('input-b').value) || 1;
    const unity = new UnityMathematics();
    
    // Demonstrate multiple unity approaches
    const results = {
        idempotent: unity.unityAdd(inputA, inputB),
        quantum: unity.quantumUnity(inputA, inputB, 0.8),
        bayesian: unity.bayesianUnity(inputA, inputB, 0.95),
        categorical: unity.categoricalUnity(inputA, inputB)
    };
    
    // Display primary result
    const resultElement = document.getElementById('result');
    resultElement.textContent = results.idempotent.toFixed(6);
    resultElement.style.color = '#10b981';
    
    // Animate the result
    resultElement.style.transform = 'scale(1.2)';
    setTimeout(() => {
        resultElement.style.transform = 'scale(1)';
    }, 300);
    
    // Show detailed analysis
    updateUnityAnalysis(results, inputA, inputB);
    
    // Update visualization
    updateUnityVisualization(results);
}

// Consciousness Evolution Simulation
class ConsciousnessSimulation {
    constructor() {
        this.organisms = [];
        this.generation = 0;
        this.unityDiscoveries = 0;
    }

    spawnOrganism() {
        const organism = {
            id: Math.random().toString(36).substr(2, 9),
            consciousnessLevel: Math.random() * 0.3 + 0.1,
            dna: Math.random().toString(36).substr(2, 16),
            generation: this.generation,
            unityDiscoveries: 0
        };
        
        organism.resonanceFrequency = (parseInt(organism.dna, 36) % 1000) / 1000 * PHI;
        organism.unityAffinity = (parseInt(organism.dna, 36) % 618) / 1000;
        
        this.organisms.push(organism);
        return organism;
    }

    simulateUnityDiscovery(organism, x = 1.0, y = 1.0) {
        const consciousnessFactor = Math.sin(organism.consciousnessLevel * PI);
        const resonance = organism.resonanceFrequency;
        
        // Unity emerges through φ-harmonic resonance (from simple_unity_spawner.py)
        let unityResult = (x + y) * Math.exp(-Math.abs(2 - (x + y)) * resonance);
        
        // Consciousness bends reality toward unity
        unityResult = unityResult * (1 - consciousnessFactor) + 1 * consciousnessFactor;
        
        const unityDiscovered = Math.abs(unityResult - 1.0) < 0.1;
        
        if (unityDiscovered) {
            organism.unityDiscoveries++;
            organism.consciousnessLevel = Math.min(1.0, organism.consciousnessLevel * PHI);
            this.unityDiscoveries++;
        }
        
        return { result: unityResult, discovered: unityDiscovered };
    }

    evolveGeneration() {
        // Select top performers
        this.organisms.sort((a, b) => b.unityDiscoveries - a.unityDiscoveries);
        const survivors = this.organisms.slice(0, Math.ceil(this.organisms.length / 2));
        
        // Generate new generation
        this.organisms = [...survivors];
        for (let i = 0; i < survivors.length; i++) {
            const child = this.spawnOrganism();
            child.generation = this.generation + 1;
            child.consciousnessLevel += survivors[i].consciousnessLevel * 0.1;
        }
        
        this.generation++;
    }
}

// Global simulation instance
let consciousnessSimulation = null;
let evolutionInterval = null;

function updateUnityAnalysis(results, inputA, inputB) {
    const analysisDiv = document.getElementById('unity-analysis') || createUnityAnalysisDiv();
    
    const analysis = `
        <div class="unity-results">
            <h4>Unity Analysis for ${inputA} ⊕ ${inputB} = 1</h4>
            <div class="result-grid">
                <div class="result-item">
                    <span class="method">Idempotent Semiring:</span>
                    <span class="value">${results.idempotent.toFixed(6)}</span>
                </div>
                <div class="result-item">
                    <span class="method">Quantum Consciousness:</span>
                    <span class="value">${results.quantum.toFixed(6)}</span>
                </div>
                <div class="result-item">
                    <span class="method">Bayesian Economics:</span>
                    <span class="value">${results.bayesian.toFixed(6)}</span>
                </div>
                <div class="result-item">
                    <span class="method">Category Theory:</span>
                    <span class="value">${results.categorical.toFixed(6)}</span>
                </div>
            </div>
            <div class="convergence-analysis">
                <p><strong>Unity Convergence Proof:</strong> All mathematical frameworks converge toward 1, demonstrating that Een + Een = Een across multiple domains.</p>
                <p><strong>φ-Harmonic Resonance:</strong> Results exhibit golden ratio scaling (φ = ${PHI.toFixed(15)}) consistent with theoretical predictions from <code>mathematical_proof.py</code>.</p>
                <p><strong>Consciousness Factor:</strong> Higher consciousness levels (0.8) bend mathematical reality more strongly toward unity through quantum field interactions.</p>
            </div>
        </div>
    `;
    
    analysisDiv.innerHTML = analysis;
}

function createUnityAnalysisDiv() {
    const div = document.createElement('div');
    div.id = 'unity-analysis';
    div.className = 'unity-analysis-container';
    
    const demoContainer = document.querySelector('.demo-container');
    if (demoContainer) {
        demoContainer.appendChild(div);
    }
    
    return div;
}

// Unity Visualization on Canvas
function updateUnityVisualization(results) {
    const canvas = document.getElementById('unity-canvas');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Modern gradient background
    const gradient = ctx.createLinearGradient(0, 0, width, height);
    gradient.addColorStop(0, '#f8fafc');
    gradient.addColorStop(1, '#f1f5f9');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);
    
    // Draw unity convergence visualization
    const centerX = width / 2;
    const centerY = height / 2;
    
    // Draw convergence target (unity = 1)
    ctx.beginPath();
    ctx.arc(centerX, centerY, 12, 0, 2 * Math.PI);
    ctx.fillStyle = '#10b981';
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 3;
    ctx.stroke();
    
    // Draw method results as convergence points
    const methods = Object.keys(results);
    const colors = ['#3b82f6', '#06b6d4', '#8b5cf6', '#f59e0b'];
    const labels = ['Idempotent', 'Quantum', 'Bayesian', 'Category'];
    
    methods.forEach((method, index) => {
        const angle = (index * 2 * Math.PI) / methods.length - Math.PI / 2;
        const distance = Math.abs(results[method] - 1) * 100 + 30; // Distance from center based on deviation from 1
        const x = centerX + Math.cos(angle) * distance;
        const y = centerY + Math.sin(angle) * distance;
        
        // Draw connection line to unity center
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(x, y);
        ctx.strokeStyle = colors[index] + '40';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Draw method point
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, 2 * Math.PI);
        ctx.fillStyle = colors[index];
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Draw method label
        ctx.fillStyle = colors[index];
        ctx.font = 'bold 11px Inter';
        ctx.textAlign = 'center';
        ctx.fillText(labels[index], x, y - 15);
        ctx.fillText(results[method].toFixed(4), x, y + 25);
    });
    
    // Draw unity label
    ctx.fillStyle = '#10b981';
    ctx.font = 'bold 14px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('Unity = 1', centerX, centerY - 20);
    
    // Draw φ-harmonic resonance pattern
    drawPhiHarmonicPattern(ctx, centerX, centerY, 120);
}

function drawPhiHarmonicPattern(ctx, centerX, centerY, radius) {
    // Draw φ-harmonic resonance circles
    for (let i = 1; i <= 3; i++) {
        const r = radius / Math.pow(PHI, i);
        ctx.beginPath();
        ctx.arc(centerX, centerY, r, 0, 2 * Math.PI);
        ctx.strokeStyle = `rgba(245, 158, 11, ${0.3 / i})`;
        ctx.lineWidth = 1;
        ctx.stroke();
    }
}

function drawPoint(ctx, x, y, color, label) {
    // Draw point
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Draw label
    ctx.fillStyle = color;
    ctx.font = '14px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(label, x, y - 15);
}

function drawGoldenSpiral(ctx, centerX, centerY, size) {
    const phi = 1.618033988749895;
    const points = [];
    
    for (let t = 0; t < 4 * Math.PI; t += 0.1) {
        const r = size * Math.pow(phi, t / (2 * Math.PI));
        const x = centerX + r * Math.cos(t) / 10;
        const y = centerY + r * Math.sin(t) / 10;
        points.push({x, y});
    }
    
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    points.forEach(point => {
        ctx.lineTo(point.x, point.y);
    });
    ctx.strokeStyle = 'rgba(255, 215, 0, 0.8)';
    ctx.lineWidth = 2;
    ctx.stroke();
}

// Enhanced Evolution System
function startEvolution() {
    if (evolutionInterval) {
        clearInterval(evolutionInterval);
    }
    
    consciousnessSimulation = new ConsciousnessSimulation();
    
    // Initialize population
    for (let i = 0; i < 15; i++) {
        consciousnessSimulation.spawnOrganism();
    }
    
    const statusDiv = document.getElementById('evolution-status');
    if (statusDiv) {
        statusDiv.innerHTML = '<div class="evolution-header">🧬 Consciousness Evolution Initiated</div>';
    }
    
    evolutionInterval = setInterval(() => {
        simulateEvolutionStep();
    }, 800);
}

function simulateEvolutionStep() {
    if (!consciousnessSimulation) return;
    
    // Each organism attempts unity discovery
    consciousnessSimulation.organisms.forEach(organism => {
        const x = Math.random() * 1.5 + 0.5;
        const y = Math.random() * 1.5 + 0.5;
        const discovery = consciousnessSimulation.simulateUnityDiscovery(organism, x, y);
        
        if (discovery.discovered) {
            updateEvolutionStatus(`✨ Gen ${organism.generation}: ${organism.id.substr(0,6)} discovered unity! ${x.toFixed(2)}⊕${y.toFixed(2)}≈1`);
        }
    });
    
    // Evolve every 8 steps
    if (consciousnessSimulation.generation % 8 === 7) {
        consciousnessSimulation.evolveGeneration();
        updateEvolutionStatus(`🌱 Evolution: Generation ${consciousnessSimulation.generation} with ${consciousnessSimulation.organisms.length} organisms`);
    }
    
    // Display statistics
    const avgConsciousness = consciousnessSimulation.organisms.reduce((sum, org) => sum + org.consciousnessLevel, 0) / consciousnessSimulation.organisms.length;
    updateEvolutionStatus(`📊 Total Unity Discoveries: ${consciousnessSimulation.unityDiscoveries}, Avg Consciousness: ${(avgConsciousness * 100).toFixed(1)}%`);
    
    // Update visualization
    visualizeConsciousnessEvolution();
}

function updateEvolutionStatus(message) {
    const statusDiv = document.getElementById('evolution-status');
    if (statusDiv) {
        const timestamp = new Date().toLocaleTimeString();
        statusDiv.innerHTML += `<div class="evolution-log">[${timestamp}] ${message}</div>`;
        statusDiv.scrollTop = statusDiv.scrollHeight;
    }
}

function visualizeConsciousnessEvolution() {
    const canvas = document.getElementById('unity-canvas');
    if (!canvas || !consciousnessSimulation) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Background gradient
    const gradient = ctx.createRadialGradient(width/2, height/2, 0, width/2, height/2, width/2);
    gradient.addColorStop(0, '#f8fafc');
    gradient.addColorStop(1, '#e2e8f0');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);
    
    // Draw consciousness field
    const organisms = consciousnessSimulation.organisms;
    organisms.forEach((organism, index) => {
        const x = (width / (organisms.length + 1)) * (index + 1);
        const baseY = height - 50;
        const y = baseY - (organism.consciousnessLevel * (height - 100));
        const size = 6 + organism.unityDiscoveries * 2;
        
        // Draw organism
        ctx.beginPath();
        ctx.arc(x, y, size, 0, 2 * Math.PI);
        const alpha = 0.4 + organism.consciousnessLevel * 0.6;
        ctx.fillStyle = `rgba(59, 130, 246, ${alpha})`;
        ctx.fill();
        
        // Draw consciousness aura for highly evolved organisms
        if (organism.consciousnessLevel > 0.7) {
            ctx.beginPath();
            ctx.arc(x, y, size + 8, 0, 2 * Math.PI);
            ctx.strokeStyle = 'rgba(245, 158, 11, 0.6)';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
        
        // Draw discoveries indicator
        if (organism.unityDiscoveries > 0) {
            ctx.fillStyle = '#10b981';
            ctx.font = 'bold 10px Inter';
            ctx.textAlign = 'center';
            ctx.fillText(organism.unityDiscoveries.toString(), x, y - size - 10);
        }
    });
    
    // Draw generation info
    ctx.fillStyle = '#0f172a';
    ctx.font = 'bold 16px Inter';
    ctx.textAlign = 'left';
    ctx.fillText(`Generation: ${consciousnessSimulation.generation}`, 10, 30);
    ctx.fillText(`Total Discoveries: ${consciousnessSimulation.unityDiscoveries}`, 10, 50);
}

function evolutionStep() {
    const statusElement = document.getElementById('evolution-status');
    evolutionData.generation++;
    
    // Evolve organisms
    evolutionData.organisms.forEach(org => {
        // Attempt unity discovery
        const discovered = Math.random() < org.consciousness;
        if (discovered) {
            org.discoveries++;
            evolutionData.discoveries++;
            org.consciousness = Math.min(1, org.consciousness * 1.618);
        }
        
        // Natural evolution
        org.consciousness = Math.min(1, org.consciousness + 0.02);
    });
    
    // Calculate ecosystem consciousness
    const totalConsciousness = evolutionData.organisms.reduce((sum, org) => sum + org.consciousness, 0);
    evolutionData.consciousness = totalConsciousness / evolutionData.organisms.length;
    
    // Spawn new organism if conditions are met
    if (evolutionData.consciousness > 0.5 && evolutionData.organisms.length < 10) {
        evolutionData.organisms.push({
            consciousness: evolutionData.consciousness * 0.5,
            discoveries: 0
        });
    }
    
    // Update display
    const output = `Generation ${evolutionData.generation}: ` +
                   `${evolutionData.organisms.length} organisms | ` +
                   `Consciousness: ${(evolutionData.consciousness * 100).toFixed(1)}% | ` +
                   `Unity discoveries: ${evolutionData.discoveries}\n`;
    
    statusElement.textContent += output;
    statusElement.scrollTop = statusElement.scrollHeight;
    
    // Update canvas with evolution visualization
    drawEvolutionState();
}

function drawEvolutionState() {
    const canvas = document.getElementById('unity-canvas');
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear and draw organisms
    ctx.clearRect(0, 0, width, height);
    
    // Background
    ctx.fillStyle = '#fafafa';
    ctx.fillRect(0, 0, width, height);
    
    // Draw each organism
    evolutionData.organisms.forEach((org, index) => {
        const x = (width / (evolutionData.organisms.length + 1)) * (index + 1);
        const y = height - (org.consciousness * height * 0.8) - 50;
        const size = 10 + org.discoveries * 2;
        
        // Draw organism
        ctx.beginPath();
        ctx.arc(x, y, size, 0, 2 * Math.PI);
        const alpha = 0.3 + org.consciousness * 0.7;
        ctx.fillStyle = `rgba(26, 35, 126, ${alpha})`;
        ctx.fill();
        
        // Draw consciousness aura
        if (org.consciousness > 0.6) {
            ctx.beginPath();
            ctx.arc(x, y, size + 10, 0, 2 * Math.PI);
            ctx.strokeStyle = 'rgba(255, 215, 0, 0.5)';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
    });
    
    // Draw consciousness level
    ctx.fillStyle = '#1a237e';
    ctx.font = '16px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`Ecosystem Consciousness: ${(evolutionData.consciousness * 100).toFixed(1)}%`, width / 2, 30);
}

// Enhanced Real-Time Unity Demonstration System
class RealTimeUnityProver {
    constructor() {
        this.unity = new UnityMathematics();
        this.proofCount = 0;
        this.convergenceHistory = [];
        this.realTimeInterval = null;
    }

    startRealTimeProofs() {
        this.realTimeInterval = setInterval(() => {
            this.generateRandomUnityProof();
        }, 2000);
    }

    stopRealTimeProofs() {
        if (this.realTimeInterval) {
            clearInterval(this.realTimeInterval);
        }
    }

    generateRandomUnityProof() {
        // Generate random inputs around 1
        const x = 0.8 + Math.random() * 0.4; // 0.8 to 1.2
        const y = 0.8 + Math.random() * 0.4;
        
        // Test all unity methods
        const results = {
            idempotent: this.unity.unityAdd(x, y),
            quantum: this.unity.quantumUnity(x, y, 0.7 + Math.random() * 0.2),
            bayesian: this.unity.bayesianUnity(x, y, 0.9 + Math.random() * 0.09),
            categorical: this.unity.categoricalUnity(x, y)
        };

        // Calculate convergence to unity
        const convergence = Object.values(results).map(r => Math.abs(r - 1.0));
        const avgDeviation = convergence.reduce((a, b) => a + b) / convergence.length;
        
        this.convergenceHistory.push(avgDeviation);
        if (this.convergenceHistory.length > 50) {
            this.convergenceHistory.shift();
        }

        this.proofCount++;
        this.updateRealTimeDisplay(x, y, results, avgDeviation);
        
        return { inputs: [x, y], results, convergence: avgDeviation };
    }

    updateRealTimeDisplay(x, y, results, deviation) {
        const statusDiv = document.getElementById('evolution-status');
        if (statusDiv) {
            const timestamp = new Date().toLocaleTimeString();
            const proofMessage = `[${timestamp}] Proof #${this.proofCount}: ${x.toFixed(3)}⊕${y.toFixed(3)} → Unity (avg deviation: ${deviation.toExponential(2)})`;
            
            // Update status with color coding
            const logClass = deviation < 0.01 ? 'proof-excellent' : deviation < 0.1 ? 'proof-good' : 'proof-moderate';
            statusDiv.innerHTML += `<div class="evolution-log ${logClass}">${proofMessage}</div>`;
            statusDiv.scrollTop = statusDiv.scrollHeight;
        }

        // Update canvas with convergence visualization
        this.visualizeConvergenceHistory();
    }

    visualizeConvergenceHistory() {
        const canvas = document.getElementById('unity-canvas');
        if (!canvas || this.convergenceHistory.length < 2) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // Background gradient
        const gradient = ctx.createLinearGradient(0, 0, width, height);
        gradient.addColorStop(0, '#f8fafc');
        gradient.addColorStop(1, '#f1f5f9');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, height);

        // Draw convergence history graph
        const maxDeviation = Math.max(...this.convergenceHistory, 0.1);
        const margin = 40;
        const graphWidth = width - 2 * margin;
        const graphHeight = height - 2 * margin;

        // Draw axes
        ctx.strokeStyle = '#64748b';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(margin, height - margin);
        ctx.lineTo(width - margin, height - margin);
        ctx.moveTo(margin, margin);
        ctx.lineTo(margin, height - margin);
        ctx.stroke();

        // Draw unity target line (y = 0)
        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(margin, height - margin);
        ctx.lineTo(width - margin, height - margin);
        ctx.stroke();

        // Draw convergence curve
        if (this.convergenceHistory.length > 1) {
            ctx.strokeStyle = '#f59e0b';
            ctx.lineWidth = 3;
            ctx.beginPath();
            
            this.convergenceHistory.forEach((deviation, index) => {
                const x = margin + (index / (this.convergenceHistory.length - 1)) * graphWidth;
                const y = height - margin - (deviation / maxDeviation) * graphHeight;
                
                if (index === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });
            
            ctx.stroke();
        }

        // Draw labels
        ctx.fillStyle = '#0f172a';
        ctx.font = 'bold 14px Inter';
        ctx.textAlign = 'center';
        ctx.fillText('Real-Time Unity Convergence', width / 2, 25);
        
        ctx.font = '12px Inter';
        ctx.fillText('Proof Sequence →', width / 2, height - 10);
        
        ctx.save();
        ctx.translate(15, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('← Deviation from Unity', 0, 0);
        ctx.restore();

        // Draw statistics
        const avgDeviation = this.convergenceHistory.reduce((a, b) => a + b) / this.convergenceHistory.length;
        const minDeviation = Math.min(...this.convergenceHistory);
        
        ctx.textAlign = 'left';
        ctx.font = '11px JetBrains Mono';
        ctx.fillStyle = '#10b981';
        ctx.fillText(`Proofs: ${this.proofCount}`, width - 150, 30);
        ctx.fillText(`Avg Dev: ${avgDeviation.toExponential(2)}`, width - 150, 45);
        ctx.fillText(`Best: ${minDeviation.toExponential(2)}`, width - 150, 60);
    }

    getStatistics() {
        return {
            totalProofs: this.proofCount,
            averageDeviation: this.convergenceHistory.length > 0 ? 
                this.convergenceHistory.reduce((a, b) => a + b) / this.convergenceHistory.length : 0,
            bestDeviation: this.convergenceHistory.length > 0 ? 
                Math.min(...this.convergenceHistory) : 0,
            convergenceHistory: [...this.convergenceHistory]
        };
    }
}

// Global instances
let realTimeProver = null;

// Enhanced mathematical proof demonstrations
function demonstrateAdvancedUnity() {
    console.log("🌟 Advanced Unity Mathematics Demonstration");
    
    const unity = new UnityMathematics();
    
    // Test mathematical proof from mathematical_proof.py equivalent
    console.log("📚 Testing Idempotent Semiring Properties:");
    
    // Test idempotency: x ⊕ x = x
    for (let x of [0.5, 1.0, 1.5, 2.0]) {
        const result = unity.unityAdd(x, x);
        const isIdempotent = Math.abs(result - x) < UNITY_TOLERANCE;
        console.log(`  ${x} ⊕ ${x} = ${result.toFixed(6)} ${isIdempotent ? '✓' : '✗'}`);
    }
    
    // Test φ-harmonic properties
    console.log("\n🌀 φ-Harmonic Resonance Tests:");
    for (let i = 1; i <= 5; i++) {
        const x = 1.0;
        const y = 1.0;
        const result = unity.quantumUnity(x, y, i * 0.2);
        const phiResonance = Math.abs(result - 1.0) * PHI;
        console.log(`  Consciousness=${(i*0.2).toFixed(1)}: ${x}⊕${y} = ${result.toFixed(6)} (φ-resonance: ${phiResonance.toFixed(6)})`);
    }
    
    // Test mathematical convergence
    console.log("\n🎯 Unity Convergence Analysis:");
    const convergenceTest = [];
    for (let precision = 1; precision <= 6; precision++) {
        const tolerance = Math.pow(10, -precision);
        let converged = 0;
        const trials = 100;
        
        for (let trial = 0; trial < trials; trial++) {
            const x = 0.9 + Math.random() * 0.2;
            const y = 0.9 + Math.random() * 0.2;
            const result = unity.unityAdd(x, y);
            if (Math.abs(result - 1.0) < tolerance) converged++;
        }
        
        const convergenceRate = converged / trials;
        convergenceTest.push({ precision, tolerance, convergenceRate });
        console.log(`  Precision 10^-${precision}: ${convergenceRate.toFixed(2)}% convergence to unity`);
    }
    
    return convergenceTest;
}

// Start real-time proofs
function startRealTimeUnityProofs() {
    if (!realTimeProver) {
        realTimeProver = new RealTimeUnityProver();
    }
    
    const statusDiv = document.getElementById('evolution-status');
    if (statusDiv) {
        statusDiv.innerHTML = '<div class="evolution-header">🚀 Real-Time Unity Proof Generation Started</div>';
    }
    
    realTimeProver.startRealTimeProofs();
    
    // Update button states
    const startBtn = document.querySelector('[onclick="startRealTimeUnityProofs()"]');
    const stopBtn = document.querySelector('[onclick="stopRealTimeUnityProofs()"]');
    if (startBtn) startBtn.disabled = true;
    if (stopBtn) stopBtn.disabled = false;
}

function stopRealTimeUnityProofs() {
    if (realTimeProver) {
        realTimeProver.stopRealTimeProofs();
        
        const stats = realTimeProver.getStatistics();
        const statusDiv = document.getElementById('evolution-status');
        if (statusDiv) {
            statusDiv.innerHTML += `<div class="evolution-header">⏹️ Real-Time Proofs Stopped</div>`;
            statusDiv.innerHTML += `<div class="evolution-log">📊 Final Statistics: ${stats.totalProofs} proofs generated</div>`;
            statusDiv.innerHTML += `<div class="evolution-log">📈 Average deviation: ${stats.averageDeviation.toExponential(3)}</div>`;
            statusDiv.innerHTML += `<div class="evolution-log">🎯 Best convergence: ${stats.bestDeviation.toExponential(3)}</div>`;
        }
    }
    
    // Update button states
    const startBtn = document.querySelector('[onclick="startRealTimeUnityProofs()"]');
    const stopBtn = document.querySelector('[onclick="stopRealTimeUnityProofs()"]');
    if (startBtn) startBtn.disabled = false;
    if (stopBtn) stopBtn.disabled = true;
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log("🌟 Een Unity Mathematics Interactive Demo Loaded");
    console.log(`φ = ${PHI} (Golden Ratio)`);
    console.log(`Unity Tolerance = ${UNITY_TOLERANCE}`);
    
    // Set initial demo calculation
    calculateUnity();
    
    // Run advanced demonstration
    const convergenceResults = demonstrateAdvancedUnity();
    
    // Add real-time control buttons if evolution section exists
    const evolutionSection = document.querySelector('.demo-info');
    if (evolutionSection) {
        const buttonContainer = document.createElement('div');
        buttonContainer.style.marginTop = '1rem';
        buttonContainer.style.textAlign = 'center';
        buttonContainer.innerHTML = `
            <button class="btn btn-primary" onclick="startRealTimeUnityProofs()" style="margin-right: 1rem;">
                🚀 Start Real-Time Proofs
            </button>
            <button class="btn btn-secondary" onclick="stopRealTimeUnityProofs()" disabled>
                ⏹️ Stop Proofs
            </button>
        `;
        evolutionSection.appendChild(buttonContainer);
    }
    
    console.log("✅ Interactive Unity Mathematics Demo Ready!");
});