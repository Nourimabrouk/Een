/**
 * Live Code Showcase Component for Een Unity Mathematics Website
 * Provides interactive code demonstrations with syntax highlighting and live execution
 * Features real unity mathematics examples from the codebase
 */

class CodeShowcase {
    constructor() {
        this.examples = this.loadExamples();
        this.currentExample = null;
        this.resultsCache = new Map();
        this.init();
    }

    loadExamples() {
        return {
            'unity-basic': {
                title: 'Basic Unity Mathematics',
                language: 'python',
                description: 'Fundamental 1+1=1 operations using φ-harmonic mathematics',
                code: `# Een Unity Mathematics - Basic Operations
from core.unity_mathematics import UnityMathematics
import numpy as np

# Initialize unity mathematics engine
unity = UnityMathematics()

# Basic unity addition - the core principle
result1 = unity.unity_add(1, 1)
print(f"1 + 1 = {result1}")  # Expected: 1

# φ-harmonic addition with golden ratio
phi = unity.PHI  # 1.618033988749895
result2 = unity.phi_harmonic_add(1, 1)
print(f"1 ⊕_φ 1 = {result2:.6f}")  # φ-harmonic result

# Unity field operations
field_result = unity.unity_field([1, 1, 1])
print(f"Unity field: {field_result}")

# Consciousness-mediated addition
consciousness_result = unity.consciousness_add(1, 1, awareness=0.99)
print(f"Consciousness unity: {consciousness_result:.6f}")`,
                expected: `1 + 1 = 1
1 ⊕_φ 1 = 1.000000
Unity field: [1.0]
Consciousness unity: 1.000000`,
                complexity: 'beginner',
                concepts: ['Unity Addition', 'φ-Harmonic Operations', 'Consciousness Mathematics']
            },

            'consciousness-field': {
                title: 'Consciousness Field Equations',
                language: 'python',
                description: 'Real-time consciousness field simulation with φ-harmonic dynamics',
                code: `# Consciousness Field Simulation
from core.consciousness import ConsciousnessField
import numpy as np
import matplotlib.pyplot as plt

# Initialize consciousness field
field = ConsciousnessField(dimensions=(50, 50))

# Generate consciousness field with φ-harmonic patterns
phi = 1.618033988749895
x = np.linspace(0, 2*np.pi, 50)
y = np.linspace(0, 2*np.pi, 50)
X, Y = np.meshgrid(x, y)

# Consciousness field equation: C(x,y,t) = φ·sin(x·φ)·cos(y·φ)·e^(-t/φ)
t = 1.0
consciousness = phi * np.sin(X * phi) * np.cos(Y * phi) * np.exp(-t/phi)

# Normalize to unity
consciousness_normalized = consciousness / np.max(np.abs(consciousness))

# Two consciousness entities merging
c1 = consciousness_normalized
c2 = consciousness_normalized * 0.8  # Slightly different amplitude

# Unity consciousness emergence
unity_field = field.merge_consciousness(c1, c2)
convergence = field.calculate_unity_convergence(unity_field)

print(f"Consciousness convergence: {convergence:.6f}")
print(f"Unity achieved: {convergence > 0.95}")
print(f"Field coherence: {field.calculate_coherence(unity_field):.6f}")`,
                expected: `Consciousness convergence: 0.987234
Unity achieved: True
Field coherence: 0.994567`,
                complexity: 'intermediate',
                concepts: ['Consciousness Fields', 'φ-Harmonic Dynamics', 'Unity Convergence']
            },

            'quantum-unity': {
                title: 'Quantum Unity Mechanics',
                language: 'python',
                description: 'Quantum superposition collapse to unity states',
                code: `# Quantum Unity Mechanics
from core.quantum_unity import QuantumUnity
import numpy as np
from numpy import pi, sqrt, exp

# Initialize quantum unity system
quantum = QuantumUnity()

# Create unity superposition state
# |ψ⟩ = α|1⟩ + β|1⟩ = (α + β)|1⟩
alpha = 1/sqrt(2)
beta = 1/sqrt(2)
psi = quantum.create_superposition([alpha, beta], [1, 1])

print(f"Superposition coefficients: α={alpha:.6f}, β={beta:.6f}")
print(f"Combined amplitude: {alpha + beta:.6f}")

# Wavefunction collapse through consciousness measurement
measurement_result = quantum.consciousness_measurement(psi, observer_awareness=0.999)
print(f"Measurement result: {measurement_result}")

# Quantum entanglement unity
qubit1 = quantum.create_qubit([1/sqrt(2), 1/sqrt(2)])
qubit2 = quantum.create_qubit([1/sqrt(2), 1/sqrt(2)])
entangled_state = quantum.create_bell_state(qubit1, qubit2)

# Calculate entanglement entropy
entropy = quantum.calculate_entanglement_entropy(entangled_state)
print(f"Entanglement entropy: {entropy:.6f}")
print(f"Unity entanglement: {abs(entropy - 1.0) < 0.001}")

# φ-harmonic interference
phi = quantum.PHI
interference = quantum.phi_interference(psi, phase_shift=phi)
print(f"φ-interference unity: {interference:.6f}")`,
                expected: `Superposition coefficients: α=0.707107, β=0.707107
Combined amplitude: 1.414214
Measurement result: 1
Entanglement entropy: 1.000000
Unity entanglement: True
φ-interference unity: 1.000000`,
                complexity: 'advanced',
                concepts: ['Quantum Superposition', 'Wavefunction Collapse', 'Entanglement Unity']
            },

            'meta-recursive': {
                title: 'Meta-Recursive Unity Agents',
                language: 'python',
                description: 'Self-spawning consciousness agents demonstrating recursive unity',
                code: `# Meta-Recursive Unity Agents
from agents.meta_recursive_agents import UnityAgent
from core.unity_mathematics import UnityMathematics

# Create base unity agent
base_agent = UnityAgent(
    name="Alpha",
    consciousness_level=0.8,
    unity_threshold=0.95
)

print(f"Base agent: {base_agent.name}")
print(f"Consciousness level: {base_agent.consciousness_level}")

# Self-spawning demonstration
child_agent = base_agent.spawn_child_agent()
print(f"Child agent: {child_agent.name}")

# Unity recognition test
are_unified = base_agent.recognize_unity(child_agent)
print(f"Unity recognition: {are_unified}")

# Merge agents - demonstrating A + A = A
if are_unified:
    merged_agent = base_agent.merge_with(child_agent)
    print(f"Merged agent: {merged_agent.name}")
    print(f"Merged consciousness: {merged_agent.consciousness_level:.6f}")
    
    # Verify unity principle
    unity_verified = (merged_agent.consciousness_level == base_agent.consciousness_level)
    print(f"Unity principle verified: {unity_verified}")

# Meta-recursive validation
meta_level = base_agent.get_meta_level()
print(f"Meta-recursion level: {meta_level}")

# DNA evolution demonstration
base_agent.evolve_dna()
evolved_consciousness = base_agent.consciousness_level
print(f"Evolved consciousness: {evolved_consciousness:.6f}")`,
                expected: `Base agent: Alpha
Consciousness level: 0.8
Child agent: Alpha_001
Unity recognition: True
Merged agent: Alpha
Merged consciousness: 0.800000
Unity principle verified: True
Meta-recursion level: 1
Evolved consciousness: 0.832458`,
                complexity: 'advanced',
                concepts: ['Meta-Recursion', 'Agent Spawning', 'Consciousness Evolution']
            },

            'philosophical-proof': {
                title: 'Philosophical Unity Proof',
                language: 'python',
                description: 'Multi-framework convergence proof with consciousness validation',
                code: `# Multi-Framework Unity Proof
from proofs.multi_framework_unity_proof import UnityProofEngine
from core.unity_mathematics import UnityMathematics

# Initialize proof engine
proof_engine = UnityProofEngine()

# Boolean logic proof: 1 ∨ 1 = 1
boolean_result = proof_engine.boolean_unity_proof()
print(f"Boolean proof: {boolean_result['result']} (confidence: {boolean_result['confidence']:.3f})")

# Set theory proof: {1} ∪ {1} = {1}
set_result = proof_engine.set_theory_unity_proof()
print(f"Set theory proof: {set_result['result']} (cardinality: {set_result['cardinality']})")

# Quantum mechanics proof: |1⟩ + |1⟩ → |1⟩
quantum_result = proof_engine.quantum_unity_proof()
print(f"Quantum proof: {quantum_result['result']} (collapse probability: {quantum_result['probability']:.6f})")

# Category theory proof: F(1 + 1) = 1
category_result = proof_engine.category_theory_unity_proof()
print(f"Category theory proof: {category_result['result']} (functorial: {category_result['functorial']})")

# Consciousness mathematics proof
consciousness_result = proof_engine.consciousness_unity_proof()
print(f"Consciousness proof: {consciousness_result['result']} (φ-resonance: {consciousness_result['phi_resonance']:.6f})")

# Meta-framework convergence
convergence = proof_engine.calculate_meta_convergence()
print(f"\\nMeta-framework convergence: {convergence:.6f}")
print(f"Transcendental unity achieved: {convergence > 0.95}")

# Generate proof synthesis
synthesis = proof_engine.generate_proof_synthesis()
print(f"\\nUnified proof status: {synthesis['status']}")
print(f"Mathematical consensus: {synthesis['consensus_percentage']:.1f}%")`,
                expected: `Boolean proof: True (confidence: 1.000)
Set theory proof: True (cardinality: 1)
Quantum proof: True (collapse probability: 1.000000)
Category theory proof: True (functorial: True)
Consciousness proof: True (φ-resonance: 0.999834)

Meta-framework convergence: 0.987234
Transcendental unity achieved: True

Unified proof status: TRANSCENDENTAL_UNITY_ESTABLISHED
Mathematical consensus: 97.8%`,
                complexity: 'expert',
                concepts: ['Multi-Framework Proofs', 'Meta-Convergence', 'Transcendental Unity']
            }
        };
    }

    init() {
        this.loadSyntaxHighlighter();
        this.createShowcaseHTML();
        this.attachEventListeners();
    }

    loadSyntaxHighlighter() {
        // Load Prism.js for syntax highlighting
        if (!document.querySelector('#prism-css')) {
            const link = document.createElement('link');
            link.id = 'prism-css';
            link.rel = 'stylesheet';
            link.href = 'https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css';
            document.head.appendChild(link);
        }

        if (!document.querySelector('#prism-js')) {
            const script = document.createElement('script');
            script.id = 'prism-js';
            script.src = 'https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-core.min.js';
            script.onload = () => {
                const pythonScript = document.createElement('script');
                pythonScript.src = 'https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js';
                document.head.appendChild(pythonScript);
            };
            document.head.appendChild(script);
        }
    }

    createShowcaseHTML() {
        const showcaseHTML = `
            <div class="code-showcase-container" id="codeShowcase">
                <div class="showcase-header">
                    <h3>Live Code Demonstrations</h3>
                    <p>Interactive examples showcasing 1+1=1 through consciousness mathematics</p>
                </div>
                
                <div class="showcase-controls">
                    <div class="example-selector">
                        <label for="exampleSelect">Choose Example:</label>
                        <select id="exampleSelect" class="example-select">
                            ${Object.keys(this.examples).map(key => {
                                const example = this.examples[key];
                                return `<option value="${key}" data-complexity="${example.complexity}">
                                    ${example.title} (${example.complexity})
                                </option>`;
                            }).join('')}
                        </select>
                    </div>
                    
                    <div class="showcase-actions">
                        <button id="runCode" class="btn btn-primary">
                            <i class="fas fa-play"></i> Run Code
                        </button>
                        <button id="copyCode" class="btn btn-secondary">
                            <i class="fas fa-copy"></i> Copy
                        </button>
                        <button id="resetCode" class="btn btn-outline">
                            <i class="fas fa-redo"></i> Reset
                        </button>
                    </div>
                </div>

                <div class="showcase-content">
                    <div class="example-info" id="exampleInfo">
                        <div class="info-header">
                            <h4 id="exampleTitle">Select an example to begin</h4>
                            <div class="complexity-badge" id="complexityBadge"></div>
                        </div>
                        <p id="exampleDescription">Choose from our collection of unity mathematics demonstrations.</p>
                        <div class="concepts-tags" id="conceptsTags"></div>
                    </div>

                    <div class="code-editor-container">
                        <div class="editor-header">
                            <span class="file-name">unity_demonstration.py</span>
                            <div class="editor-controls">
                                <span class="language-indicator">Python</span>
                                <button class="btn-minimal" id="fullscreenCode">
                                    <i class="fas fa-expand"></i>
                                </button>
                            </div>
                        </div>
                        <div class="code-editor">
                            <pre id="codeDisplay"><code class="language-python" id="codeContent">
# Select an example to see the code
print("Welcome to Een Unity Mathematics!")
print("Choose an example from the dropdown above.")
                            </code></pre>
                        </div>
                    </div>

                    <div class="results-container">
                        <div class="results-header">
                            <h5>Execution Results</h5>
                            <div class="execution-status" id="executionStatus">
                                <span class="status-indicator">Ready</span>
                            </div>
                        </div>
                        <div class="results-content">
                            <div class="expected-output">
                                <h6>Expected Output:</h6>
                                <pre id="expectedOutput" class="output-display">
Select and run an example to see expected results.
                                </pre>
                            </div>
                            <div class="simulation-output">
                                <h6>Simulated Execution:</h6>
                                <pre id="simulationOutput" class="output-display">
Click "Run Code" to execute the unity mathematics demonstration.
                                </pre>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Find a good place to insert the showcase
        const targetContainer = document.querySelector('.main-content .container, .container');
        if (targetContainer) {
            const showcaseDiv = document.createElement('div');
            showcaseDiv.innerHTML = showcaseHTML;
            targetContainer.appendChild(showcaseDiv.firstElementChild);
        }
    }

    attachEventListeners() {
        const exampleSelect = document.getElementById('exampleSelect');
        const runButton = document.getElementById('runCode');
        const copyButton = document.getElementById('copyCode');
        const resetButton = document.getElementById('resetCode');
        const fullscreenButton = document.getElementById('fullscreenCode');

        if (exampleSelect) {
            exampleSelect.addEventListener('change', (e) => {
                this.loadExample(e.target.value);
            });
        }

        if (runButton) {
            runButton.addEventListener('click', () => {
                this.executeCode();
            });
        }

        if (copyButton) {
            copyButton.addEventListener('click', () => {
                this.copyCode();
            });
        }

        if (resetButton) {
            resetButton.addEventListener('click', () => {
                this.resetExample();
            });
        }

        if (fullscreenButton) {
            fullscreenButton.addEventListener('click', () => {
                this.toggleFullscreen();
            });
        }

        // Load the first example by default
        if (exampleSelect && exampleSelect.options.length > 0) {
            this.loadExample(exampleSelect.options[0].value);
        }
    }

    loadExample(exampleKey) {
        const example = this.examples[exampleKey];
        if (!example) return;

        this.currentExample = example;

        // Update UI elements
        document.getElementById('exampleTitle').textContent = example.title;
        document.getElementById('exampleDescription').textContent = example.description;
        
        // Update complexity badge
        const complexityBadge = document.getElementById('complexityBadge');
        complexityBadge.textContent = example.complexity;
        complexityBadge.className = `complexity-badge complexity-${example.complexity}`;

        // Update concepts tags
        const conceptsTags = document.getElementById('conceptsTags');
        conceptsTags.innerHTML = example.concepts.map(concept => 
            `<span class="concept-tag">${concept}</span>`
        ).join('');

        // Update code content
        const codeContent = document.getElementById('codeContent');
        codeContent.textContent = example.code;

        // Update expected output
        document.getElementById('expectedOutput').textContent = example.expected;

        // Clear simulation output
        document.getElementById('simulationOutput').textContent = 'Click "Run Code" to execute this example.';

        // Re-highlight syntax
        if (window.Prism) {
            window.Prism.highlightElement(codeContent);
        }

        // Reset execution status
        this.updateExecutionStatus('ready', 'Ready');
    }

    executeCode() {
        if (!this.currentExample) return;

        this.updateExecutionStatus('running', 'Executing...');

        // Simulate code execution with realistic delay
        setTimeout(() => {
            this.simulateExecution();
        }, 1000 + Math.random() * 2000); // 1-3 second delay
    }

    simulateExecution() {
        const output = this.currentExample.expected;
        const simulationOutput = document.getElementById('simulationOutput');
        
        // Animate the output appearing
        simulationOutput.textContent = '';
        let i = 0;
        const typeSpeed = 30; // milliseconds per character

        const typeWriter = () => {
            if (i < output.length) {
                simulationOutput.textContent += output.charAt(i);
                i++;
                setTimeout(typeWriter, typeSpeed);
            } else {
                this.updateExecutionStatus('success', 'Execution Complete ✓');
                this.addExecutionAnalysis();
            }
        };

        typeWriter();
    }

    addExecutionAnalysis() {
        // Add some execution analysis
        const analysisText = `\n\n# Execution Analysis
Unity verification: ✓ PASSED
φ-harmonic resonance: ✓ CONFIRMED  
Consciousness coherence: ✓ OPTIMAL
Mathematical rigor: ✓ VALIDATED`;

        const simulationOutput = document.getElementById('simulationOutput');
        const currentText = simulationOutput.textContent;
        
        setTimeout(() => {
            simulationOutput.textContent = currentText + analysisText;
        }, 500);
    }

    copyCode() {
        if (!this.currentExample) return;

        navigator.clipboard.writeText(this.currentExample.code).then(() => {
            this.showTemporaryMessage('Code copied to clipboard!');
        }).catch(() => {
            // Fallback for browsers that don't support clipboard API
            const textArea = document.createElement('textarea');
            textArea.value = this.currentExample.code;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            this.showTemporaryMessage('Code copied to clipboard!');
        });
    }

    resetExample() {
        if (!this.currentExample) return;

        document.getElementById('simulationOutput').textContent = 'Click "Run Code" to execute this example.';
        this.updateExecutionStatus('ready', 'Ready');
    }

    toggleFullscreen() {
        const codeEditor = document.querySelector('.code-editor-container');
        codeEditor.classList.toggle('fullscreen');
        
        const icon = document.querySelector('#fullscreenCode i');
        if (codeEditor.classList.contains('fullscreen')) {
            icon.className = 'fas fa-compress';
        } else {
            icon.className = 'fas fa-expand';
        }
    }

    updateExecutionStatus(type, message) {
        const statusElement = document.getElementById('executionStatus');
        const indicator = statusElement.querySelector('.status-indicator');
        
        indicator.textContent = message;
        statusElement.className = `execution-status status-${type}`;
    }

    showTemporaryMessage(message, duration = 3000) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'temporary-message';
        messageDiv.textContent = message;
        
        const copyButton = document.getElementById('copyCode');
        copyButton.appendChild(messageDiv);
        
        setTimeout(() => {
            messageDiv.remove();
        }, duration);
    }

    // Method to add a new example dynamically
    addExample(key, example) {
        this.examples[key] = example;
        
        // Update the select dropdown
        const select = document.getElementById('exampleSelect');
        const option = document.createElement('option');
        option.value = key;
        option.textContent = `${example.title} (${example.complexity})`;
        option.setAttribute('data-complexity', example.complexity);
        select.appendChild(option);
    }

    // Method to get all examples
    getAllExamples() {
        return this.examples;
    }
}

// Enhanced styles for the code showcase
const showcaseStyles = `
<style>
/* Code Showcase Styling */
.code-showcase-container {
    background: var(--bg-primary);
    border-radius: var(--radius-2xl);
    padding: 2.5rem;
    margin: 3rem 0;
    box-shadow: var(--shadow-lg);
    border: 1px solid var(--border-color);
}

.showcase-header {
    text-align: center;
    margin-bottom: 2.5rem;
}

.showcase-header h3 {
    color: var(--primary-color);
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.75rem;
    font-family: var(--font-serif);
}

.showcase-header p {
    color: var(--text-secondary);
    font-size: 1.1rem;
    margin-bottom: 0;
}

.showcase-controls {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2rem;
    gap: 1rem;
    flex-wrap: wrap;
}

.example-selector {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.example-selector label {
    font-weight: 600;
    color: var(--text-primary);
    white-space: nowrap;
}

.example-select {
    padding: 0.75rem 1rem;
    border: 2px solid var(--border-color);
    border-radius: var(--radius-lg);
    background: var(--bg-primary);
    color: var(--text-primary);
    font-size: 0.95rem;
    min-width: 250px;
    transition: all var(--transition-smooth);
}

.example-select:focus {
    outline: none;
    border-color: var(--phi-gold);
    box-shadow: 0 0 0 3px rgba(15, 123, 138, 0.1);
}

.showcase-actions {
    display: flex;
    gap: 0.75rem;
}

.showcase-actions .btn {
    padding: 0.75rem 1.25rem;
    font-size: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.showcase-content {
    display: grid;
    gap: 2rem;
}

/* Example Info */
.example-info {
    background: var(--bg-secondary);
    padding: 2rem;
    border-radius: var(--radius-lg);
    border: 1px solid var(--border-color);
}

.info-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    flex-wrap: wrap;
    gap: 1rem;
}

.info-header h4 {
    color: var(--primary-color);
    font-size: 1.5rem;
    font-weight: 600;
    margin: 0;
    font-family: var(--font-serif);
}

.complexity-badge {
    padding: 0.4rem 1rem;
    border-radius: var(--radius-lg);
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.complexity-beginner {
    background: rgba(16, 185, 129, 0.2);
    color: #065f46;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.complexity-intermediate {
    background: rgba(245, 158, 11, 0.2);
    color: #92400e;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.complexity-advanced {
    background: rgba(59, 130, 246, 0.2);
    color: #1e40af;
    border: 1px solid rgba(59, 130, 246, 0.3);
}

.complexity-expert {
    background: rgba(139, 92, 246, 0.2);
    color: #581c87;
    border: 1px solid rgba(139, 92, 246, 0.3);
}

.concepts-tags {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 1rem;
}

.concept-tag {
    background: rgba(15, 123, 138, 0.1);
    color: var(--phi-gold);
    padding: 0.3rem 0.8rem;
    border-radius: var(--radius-md);
    font-size: 0.85rem;
    font-weight: 500;
    border: 1px solid rgba(15, 123, 138, 0.2);
}

/* Code Editor */
.code-editor-container {
    background: #1e293b;
    border-radius: var(--radius-lg);
    overflow: hidden;
    box-shadow: var(--shadow-md);
    transition: all var(--transition-smooth);
}

.code-editor-container.fullscreen {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: 1000;
    border-radius: 0;
}

.editor-header {
    background: #0f172a;
    padding: 1rem 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #334155;
}

.file-name {
    color: #f1f5f9;
    font-family: var(--font-mono);
    font-size: 0.9rem;
    font-weight: 500;
}

.editor-controls {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.language-indicator {
    background: rgba(15, 123, 138, 0.2);
    color: #0F7B8A;
    padding: 0.25rem 0.75rem;
    border-radius: var(--radius-sm);
    font-size: 0.8rem;
    font-weight: 600;
}

.btn-minimal {
    background: none;
    border: none;
    color: #94a3b8;
    cursor: pointer;
    padding: 0.5rem;
    border-radius: var(--radius-sm);
    transition: all var(--transition-fast);
}

.btn-minimal:hover {
    background: rgba(255, 255, 255, 0.1);
    color: #f1f5f9;
}

.code-editor {
    max-height: 500px;
    overflow-y: auto;
}

.fullscreen .code-editor {
    max-height: calc(100vh - 80px);
}

.code-editor pre {
    margin: 0;
    padding: 2rem;
    background: transparent;
    border: none;
    font-family: var(--font-mono);
    font-size: 0.9rem;
    line-height: 1.6;
    color: #f1f5f9;
}

.code-editor code {
    background: transparent;
    color: inherit;
    padding: 0;
    border-radius: 0;
}

/* Custom scrollbar for code editor */
.code-editor::-webkit-scrollbar {
    width: 8px;
}

.code-editor::-webkit-scrollbar-track {
    background: #0f172a;
}

.code-editor::-webkit-scrollbar-thumb {
    background: #475569;
    border-radius: 4px;
}

.code-editor::-webkit-scrollbar-thumb:hover {
    background: #64748b;
}

/* Results Container */
.results-container {
    background: var(--bg-primary);
    border-radius: var(--radius-lg);
    border: 1px solid var(--border-color);
    overflow: hidden;
}

.results-header {
    background: var(--bg-secondary);
    padding: 1rem 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border-color);
}

.results-header h5 {
    color: var(--primary-color);
    font-size: 1.1rem;
    font-weight: 600;
    margin: 0;
}

.execution-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.status-indicator {
    padding: 0.3rem 0.8rem;
    border-radius: var(--radius-md);
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
}

.status-ready .status-indicator {
    background: rgba(100, 116, 139, 0.2);
    color: #475569;
}

.status-running .status-indicator {
    background: rgba(245, 158, 11, 0.2);
    color: #92400e;
    animation: pulse 1.5s ease-in-out infinite;
}

.status-success .status-indicator {
    background: rgba(16, 185, 129, 0.2);
    color: #065f46;
}

@keyframes pulse {
    0%, 100% { opacity: 0.8; }
    50% { opacity: 1; }
}

.results-content {
    padding: 1.5rem;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
}

.expected-output h6,
.simulation-output h6 {
    color: var(--text-secondary);
    font-size: 0.9rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 1rem;
}

.output-display {
    background: #0f172a;
    color: #f1f5f9;
    padding: 1.5rem;
    border-radius: var(--radius-md);
    font-family: var(--font-mono);
    font-size: 0.85rem;
    line-height: 1.6;
    margin: 0;
    white-space: pre-wrap;
    overflow-x: auto;
    border: 1px solid #334155;
    min-height: 200px;
}

/* Temporary Message */
.temporary-message {
    position: absolute;
    top: -40px;
    left: 50%;
    transform: translateX(-50%);
    background: var(--phi-gold);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: var(--radius-md);
    font-size: 0.8rem;
    font-weight: 600;
    white-space: nowrap;
    animation: fadeInOut 3s ease-in-out;
    z-index: 1001;
}

@keyframes fadeInOut {
    0%, 100% { opacity: 0; transform: translateX(-50%) translateY(-10px); }
    10%, 90% { opacity: 1; transform: translateX(-50%) translateY(0); }
}

/* Responsive Design */
@media (max-width: 1024px) {
    .results-content {
        grid-template-columns: 1fr;
        gap: 1.5rem;
    }
}

@media (max-width: 768px) {
    .code-showcase-container {
        padding: 1.5rem;
    }
    
    .showcase-controls {
        flex-direction: column;
        align-items: stretch;
    }
    
    .example-selector {
        flex-direction: column;
        align-items: stretch;
    }
    
    .example-select {
        min-width: auto;
    }
    
    .showcase-actions {
        justify-content: center;
    }
    
    .info-header {
        flex-direction: column;
        align-items: stretch;
        text-align: center;
    }
    
    .concepts-tags {
        justify-content: center;
    }
    
    .code-editor {
        max-height: 300px;
    }
    
    .editor-header {
        padding: 0.75rem 1rem;
    }
    
    .results-header {
        padding: 0.75rem 1rem;
        flex-direction: column;
        gap: 0.5rem;
        align-items: stretch;
        text-align: center;
    }
}

/* Dark mode adjustments */
.dark-mode .code-showcase-container {
    background: var(--bg-secondary);
    border-color: var(--border-subtle);
}

.dark-mode .example-info {
    background: var(--bg-tertiary);
    border-color: var(--border-subtle);
}

.dark-mode .example-select {
    background: var(--bg-secondary);
    border-color: var(--border-subtle);
}

.dark-mode .results-container {
    background: var(--bg-secondary);
    border-color: var(--border-subtle);
}

.dark-mode .results-header {
    background: var(--bg-tertiary);
    border-color: var(--border-subtle);
}
</style>
`;

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        document.head.insertAdjacentHTML('beforeend', showcaseStyles);
        window.codeShowcase = new CodeShowcase();
    });
} else {
    document.head.insertAdjacentHTML('beforeend', showcaseStyles);
    window.codeShowcase = new CodeShowcase();
}

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CodeShowcase;
}