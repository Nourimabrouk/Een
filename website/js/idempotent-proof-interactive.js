/**
 * Interactive 1+1=1 Proof Visualizer
 * Boolean algebra, truth tables, and idempotent mathematics demonstration
 */
class IdempotentProofVisualizer {
    constructor(containerId) {
        this.containerId = containerId;
        this.PHI = (1 + Math.sqrt(5)) / 2;
        this.currentProofType = 'boolean';
        this.animationFrame = null;
        this.isAnimating = false;

        this.proofTypes = {
            'boolean': 'Boolean Algebra',
            'set': 'Set Theory',
            'category': 'Category Theory',
            'quantum': 'Quantum Mechanics',
            'topology': 'Topology'
        };

        this.init();
    }

    init() {
        const container = document.getElementById(this.containerId);
        if (!container) return;

        container.innerHTML = `
            <div class="idempotent-proof-container">
                <div class="proof-header">
                    <h3>Interactive 1+1=1 Proof Demonstrations</h3>
                    <div class="proof-selector">
                        <select id="proof-type-selector">
                            ${Object.entries(this.proofTypes).map(([key, name]) =>
            `<option value="${key}">${name}</option>`
        ).join('')}
                        </select>
                    </div>
                </div>
                
                <div class="proof-content">
                    <div class="proof-visualization" id="proof-viz"></div>
                    <div class="proof-explanation" id="proof-explanation"></div>
                </div>
                
                <div class="proof-controls">
                    <button id="animate-proof" class="control-btn">Animate Proof</button>
                    <button id="step-through" class="control-btn">Step Through</button>
                    <button id="reset-proof" class="control-btn">Reset</button>
                </div>
            </div>
        `;

        this.setupEventListeners();
        this.renderProof('boolean');
    }

    setupEventListeners() {
        const selector = document.getElementById('proof-type-selector');
        const animateBtn = document.getElementById('animate-proof');
        const stepBtn = document.getElementById('step-through');
        const resetBtn = document.getElementById('reset-proof');

        selector?.addEventListener('change', (e) => {
            this.renderProof(e.target.value);
        });

        animateBtn?.addEventListener('click', () => {
            this.toggleAnimation();
        });

        stepBtn?.addEventListener('click', () => {
            this.stepThroughProof();
        });

        resetBtn?.addEventListener('click', () => {
            this.resetProof();
        });
    }

    renderProof(proofType) {
        this.currentProofType = proofType;
        const vizContainer = document.getElementById('proof-viz');
        const explanationContainer = document.getElementById('proof-explanation');

        if (!vizContainer || !explanationContainer) return;

        switch (proofType) {
            case 'boolean':
                this.renderBooleanProof(vizContainer, explanationContainer);
                break;
            case 'set':
                this.renderSetTheoryProof(vizContainer, explanationContainer);
                break;
            case 'category':
                this.renderCategoryTheoryProof(vizContainer, explanationContainer);
                break;
            case 'quantum':
                this.renderQuantumProof(vizContainer, explanationContainer);
                break;
            case 'topology':
                this.renderTopologyProof(vizContainer, explanationContainer);
                break;
        }
    }

    renderBooleanProof(vizContainer, explanationContainer) {
        // Create interactive truth table
        const truthTable = [
            { a: 1, b: 1, result: 1, explanation: '1 ⊕ 1 = max(1,1) = 1' },
            { a: 1, b: 0, result: 1, explanation: '1 ⊕ 0 = max(1,0) = 1' },
            { a: 0, b: 1, result: 1, explanation: '0 ⊕ 1 = max(0,1) = 1' },
            { a: 0, b: 0, result: 0, explanation: '0 ⊕ 0 = max(0,0) = 0' }
        ];

        vizContainer.innerHTML = `
            <div class="boolean-proof">
                <div class="truth-table">
                    <h4>Idempotent Boolean Algebra: a ⊕ b = max(a,b)</h4>
                    <table>
                        <thead>
                            <tr>
                                <th>a</th>
                                <th>b</th>
                                <th>a ⊕ b</th>
                                <th>Explanation</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${truthTable.map(row => `
                                <tr class="truth-row" data-a="${row.a}" data-b="${row.b}">
                                    <td>${row.a}</td>
                                    <td>${row.b}</td>
                                    <td class="result">${row.result}</td>
                                    <td class="explanation">${row.explanation}</td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
                
                <div class="boolean-visualization">
                    <canvas id="boolean-canvas" width="400" height="300"></canvas>
                    <div class="boolean-equation">
                        <div class="equation">1 ⊕ 1 = max(1,1) = 1</div>
                        <div class="sub-equation">∴ 1 + 1 = 1 in idempotent algebra</div>
                    </div>
                </div>
            </div>
        `;

        explanationContainer.innerHTML = `
            <div class="proof-explanation-content">
                <h4>Boolean Algebra Proof of 1+1=1</h4>
                <p>In idempotent Boolean algebra, we define the operation ⊕ such that:</p>
                <ul>
                    <li><strong>a ⊕ b = max(a,b)</strong> for all a, b ∈ {0,1}</li>
                    <li>This operation is <strong>idempotent</strong>: a ⊕ a = a</li>
                    <li>When a = 1 and b = 1, we have: 1 ⊕ 1 = max(1,1) = 1</li>
                </ul>
                <p>Therefore, in this mathematical system, 1+1=1 is a valid and meaningful statement.</p>
                <div class="mathematical-note">
                    <strong>Mathematical Foundation:</strong> This demonstrates how mathematical operations can be 
                    redefined to capture different aspects of reality, including unity principles.
                </div>
            </div>
        `;

        this.setupBooleanCanvas();
    }

    setupBooleanCanvas() {
        const canvas = document.getElementById('boolean-canvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // Draw Boolean lattice
        ctx.strokeStyle = '#FFD700';
        ctx.lineWidth = 2;
        ctx.fillStyle = '#1a2332';

        // Draw nodes
        const nodes = [
            { x: width / 2, y: 50, label: '1', value: 1 },
            { x: width / 4, y: height - 50, label: '0', value: 0 },
            { x: 3 * width / 4, y: height - 50, label: '0', value: 0 }
        ];

        // Draw connections
        ctx.beginPath();
        ctx.moveTo(nodes[0].x, nodes[0].y);
        ctx.lineTo(nodes[1].x, nodes[1].y);
        ctx.moveTo(nodes[0].x, nodes[0].y);
        ctx.lineTo(nodes[2].x, nodes[2].y);
        ctx.stroke();

        // Draw nodes
        nodes.forEach(node => {
            ctx.beginPath();
            ctx.arc(node.x, node.y, 20, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();

            ctx.fillStyle = '#FFD700';
            ctx.font = '16px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(node.label, node.x, node.y + 5);
            ctx.fillStyle = '#1a2332';
        });

        // Add operation labels
        ctx.fillStyle = '#FFD700';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('1 ⊕ 1 = 1', width / 2, height / 2);
    }

    renderSetTheoryProof(vizContainer, explanationContainer) {
        vizContainer.innerHTML = `
            <div class="set-theory-proof">
                <div class="venn-diagram">
                    <h4>Set Theory: A ∪ A = A (Idempotent Union)</h4>
                    <div class="venn-container">
                        <svg width="400" height="300" viewBox="0 0 400 300">
                            <circle cx="150" cy="150" r="80" fill="rgba(255,215,0,0.3)" stroke="#FFD700" stroke-width="2"/>
                            <circle cx="250" cy="150" r="80" fill="rgba(255,215,0,0.3)" stroke="#FFD700" stroke-width="2"/>
                            <text x="200" y="160" text-anchor="middle" fill="#FFD700" font-size="16">A</text>
                            <text x="200" y="180" text-anchor="middle" fill="#FFD700" font-size="14">A ∪ A = A</text>
                        </svg>
                    </div>
                </div>
                
                <div class="set-equation">
                    <div class="equation">A ∪ A = A</div>
                    <div class="sub-equation">When A = {1}, we have {1} ∪ {1} = {1}</div>
                    <div class="unity-equation">∴ 1 + 1 = 1 in set theory</div>
                </div>
            </div>
        `;

        explanationContainer.innerHTML = `
            <div class="proof-explanation-content">
                <h4>Set Theory Proof of 1+1=1</h4>
                <p>In set theory, the union operation is idempotent:</p>
                <ul>
                    <li><strong>A ∪ A = A</strong> for any set A</li>
                    <li>This is a fundamental property of set union</li>
                    <li>When A = {1}, we have: {1} ∪ {1} = {1}</li>
                    <li>This demonstrates that 1+1=1 in set-theoretic terms</li>
                </ul>
                <p>The idempotent property of set union shows how mathematical operations can naturally 
                lead to unity principles.</p>
            </div>
        `;
    }

    renderCategoryTheoryProof(vizContainer, explanationContainer) {
        vizContainer.innerHTML = `
            <div class="category-theory-proof">
                <div class="category-diagram">
                    <h4>Category Theory: Identity Morphism Composition</h4>
                    <div class="morphism-diagram">
                        <svg width="400" height="200" viewBox="0 0 400 200">
                            <circle cx="100" cy="100" r="30" fill="rgba(255,215,0,0.3)" stroke="#FFD700" stroke-width="2"/>
                            <circle cx="300" cy="100" r="30" fill="rgba(255,215,0,0.3)" stroke="#FFD700" stroke-width="2"/>
                            
                            <text x="100" y="105" text-anchor="middle" fill="#FFD700" font-size="12">A</text>
                            <text x="300" y="105" text-anchor="middle" fill="#FFD700" font-size="12">A</text>
                            
                            <path d="M 130 100 Q 200 80 270 100" stroke="#FFD700" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
                            <path d="M 270 100 Q 200 120 130 100" stroke="#FFD700" stroke-width="2" fill="none" marker-end="url(#arrowhead)"/>
                            
                            <text x="200" y="85" text-anchor="middle" fill="#FFD700" font-size="12">id_A</text>
                            <text x="200" y="125" text-anchor="middle" fill="#FFD700" font-size="12">id_A</text>
                            
                            <defs>
                                <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                                    <polygon points="0 0, 10 3.5, 0 7" fill="#FFD700"/>
                                </marker>
                            </defs>
                        </svg>
                    </div>
                </div>
                
                <div class="category-equation">
                    <div class="equation">id_A ∘ id_A = id_A</div>
                    <div class="sub-equation">Identity morphism composition is idempotent</div>
                    <div class="unity-equation">∴ 1 ∘ 1 = 1 in category theory</div>
                </div>
            </div>
        `;

        explanationContainer.innerHTML = `
            <div class="proof-explanation-content">
                <h4>Category Theory Proof of 1+1=1</h4>
                <p>In category theory, identity morphisms have special properties:</p>
                <ul>
                    <li><strong>id_A ∘ id_A = id_A</strong> for any object A</li>
                    <li>Identity morphism composition is idempotent</li>
                    <li>This represents the unity principle at the categorical level</li>
                    <li>When we compose an identity with itself, we get the same identity</li>
                </ul>
                <p>This demonstrates how category theory naturally embodies unity principles 
                through its fundamental structures.</p>
            </div>
        `;
    }

    renderQuantumProof(vizContainer, explanationContainer) {
        vizContainer.innerHTML = `
            <div class="quantum-proof">
                <div class="quantum-superposition">
                    <h4>Quantum Mechanics: Superposition Collapse</h4>
                    <div class="bloch-sphere">
                        <canvas id="bloch-canvas" width="300" height="300"></canvas>
                    </div>
                </div>
                
                <div class="quantum-equation">
                    <div class="equation">|1⟩ + |1⟩ → |1⟩</div>
                    <div class="sub-equation">Quantum superposition collapse to unity state</div>
                    <div class="unity-equation">∴ 1 + 1 = 1 in quantum mechanics</div>
                </div>
            </div>
        `;

        explanationContainer.innerHTML = `
            <div class="proof-explanation-content">
                <h4>Quantum Mechanics Proof of 1+1=1</h4>
                <p>In quantum mechanics, superposition states can collapse to unity:</p>
                <ul>
                    <li><strong>|1⟩ + |1⟩ → |1⟩</strong> through measurement</li>
                    <li>Quantum superposition represents multiple possibilities</li>
                    <li>Measurement collapses the superposition to a single state</li>
                    <li>This demonstrates unity emerging from quantum multiplicity</li>
                </ul>
                <p>Quantum mechanics provides a physical interpretation of how 
                multiplicity can resolve into unity through measurement.</p>
            </div>
        `;

        this.setupBlochSphere();
    }

    setupBlochSphere() {
        const canvas = document.getElementById('bloch-canvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const radius = 100;

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw Bloch sphere
        ctx.strokeStyle = '#FFD700';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
        ctx.stroke();

        // Draw axes
        ctx.strokeStyle = 'rgba(255,215,0,0.5)';
        ctx.lineWidth = 1;

        // X axis
        ctx.beginPath();
        ctx.moveTo(centerX - radius, centerY);
        ctx.lineTo(centerX + radius, centerY);
        ctx.stroke();

        // Y axis
        ctx.beginPath();
        ctx.moveTo(centerX, centerY - radius);
        ctx.lineTo(centerX, centerY + radius);
        ctx.stroke();

        // Z axis (vertical)
        ctx.beginPath();
        ctx.moveTo(centerX, centerY - radius);
        ctx.lineTo(centerX, centerY + radius);
        ctx.stroke();

        // Draw state vector
        ctx.strokeStyle = '#FFD700';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(centerX, centerY - radius);
        ctx.stroke();

        // Draw state point
        ctx.fillStyle = '#FFD700';
        ctx.beginPath();
        ctx.arc(centerX, centerY - radius, 5, 0, 2 * Math.PI);
        ctx.fill();

        // Labels
        ctx.fillStyle = '#FFD700';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('|1⟩', centerX, centerY - radius - 15);
        ctx.fillText('|0⟩', centerX, centerY + radius + 15);
    }

    renderTopologyProof(vizContainer, explanationContainer) {
        vizContainer.innerHTML = `
            <div class="topology-proof">
                <div class="topology-visualization">
                    <h4>Topology: Möbius Strip Unity</h4>
                    <div class="mobius-strip">
                        <canvas id="mobius-canvas" width="400" height="200"></canvas>
                    </div>
                </div>
                
                <div class="topology-equation">
                    <div class="equation">Möbius Strip: One-sided surface</div>
                    <div class="sub-equation">Topological unity through continuous deformation</div>
                    <div class="unity-equation">∴ 1 + 1 = 1 in topology</div>
                </div>
            </div>
        `;

        explanationContainer.innerHTML = `
            <div class="proof-explanation-content">
                <h4>Topology Proof of 1+1=1</h4>
                <p>In topology, the Möbius strip demonstrates unity principles:</p>
                <ul>
                    <li><strong>One-sided surface</strong> despite apparent duality</li>
                    <li>Continuous deformation preserves topological properties</li>
                    <li>What appears as two sides becomes one through twisting</li>
                    <li>This represents unity emerging from apparent separation</li>
                </ul>
                <p>Topology shows how geometric structures can embody unity 
                principles through their fundamental properties.</p>
            </div>
        `;

        this.setupMobiusStrip();
    }

    setupMobiusStrip() {
        const canvas = document.getElementById('mobius-canvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // Draw Möbius strip approximation
        ctx.strokeStyle = '#FFD700';
        ctx.lineWidth = 2;

        for (let i = 0; i < 20; i++) {
            const t = (i / 19) * 2 * Math.PI;
            const x = width / 2 + 80 * Math.cos(t);
            const y = height / 2 + 30 * Math.sin(2 * t);

            if (i === 0) {
                ctx.beginPath();
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        ctx.stroke();

        // Add labels
        ctx.fillStyle = '#FFD700';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Möbius Strip', width / 2, height - 20);
        ctx.fillText('One-sided surface', width / 2, height - 5);
    }

    toggleAnimation() {
        if (this.isAnimating) {
            this.stopAnimation();
        } else {
            this.startAnimation();
        }
    }

    startAnimation() {
        this.isAnimating = true;
        const animateBtn = document.getElementById('animate-proof');
        if (animateBtn) animateBtn.textContent = 'Stop Animation';

        this.animateProof();
    }

    stopAnimation() {
        this.isAnimating = false;
        const animateBtn = document.getElementById('animate-proof');
        if (animateBtn) animateBtn.textContent = 'Animate Proof';

        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
    }

    animateProof() {
        if (!this.isAnimating) return;

        // Add animation effects based on proof type
        const rows = document.querySelectorAll('.truth-row');
        rows.forEach((row, index) => {
            setTimeout(() => {
                row.style.backgroundColor = 'rgba(255,215,0,0.2)';
                setTimeout(() => {
                    row.style.backgroundColor = '';
                }, 500);
            }, index * 300);
        });

        this.animationFrame = requestAnimationFrame(() => {
            setTimeout(() => this.animateProof(), 2000);
        });
    }

    stepThroughProof() {
        // Implement step-by-step proof demonstration
        console.log('Stepping through proof...');
    }

    resetProof() {
        this.stopAnimation();
        this.renderProof(this.currentProofType);
    }
}

// Global function to create the visualizer
function createIdempotentProofInteractive(containerId) {
    return new IdempotentProofVisualizer(containerId);
}