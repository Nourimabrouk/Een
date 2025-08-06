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
        
        // Configuration
        this.config = {
            show_steps: true,
            show_truth_table: true,
            animation_speed: 1.0,
            highlight_unity: true
        };
        
        this.init();
    }
    
    init() {
        this.createContainer();
        this.createVisualization();
    }
    
    createContainer() {
        const container = document.getElementById(this.containerId);
        if (!container) {
            console.error(`Container ${this.containerId} not found`);
            return;
        }
        
        container.innerHTML = `
            <div class="idempotent-proof-container" style="
                background: linear-gradient(135deg, rgba(16, 185, 129, 0.03) 0%, rgba(245, 158, 11, 0.05) 100%);
                border-radius: 1rem;
                padding: 1.5rem;
                box-shadow: 0 8px 32px rgba(16, 185, 129, 0.2);
                margin: 1rem 0;
                position: relative;
                overflow: hidden;
            ">
                <div class="header" style="
                    text-align: center;
                    margin-bottom: 1.5rem;
                ">
                    <h3 style="
                        color: #10B981;
                        margin: 0 0 0.5rem 0;
                        font-size: 1.8rem;
                        background: linear-gradient(135deg, #10B981, #F59E0B);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        background-clip: text;
                    ">🎯 Interactive 1+1=1 Proof Explorer</h3>
                    <p style="color: #6B7280; margin: 0 0 1rem 0;">Mathematical demonstrations of idempotent unity</p>
                    <div style="
                        font-family: 'Times New Roman', serif;
                        font-size: 1.5rem;
                        color: #F59E0B;
                        background: rgba(245, 158, 11, 0.1);
                        padding: 0.75rem 1.5rem;
                        border-radius: 0.5rem;
                        display: inline-block;
                        border: 2px solid rgba(245, 158, 11, 0.3);
                    ">
                        1 + 1 = 1 (Idempotent Unity)
                    </div>
                </div>
                
                <div class="controls-panel" style="
                    display: flex;
                    justify-content: center;
                    gap: 1rem;
                    flex-wrap: wrap;
                    margin-bottom: 1.5rem;
                    padding: 1rem;
                    background: rgba(16, 185, 129, 0.05);
                    border-radius: 0.75rem;
                    border: 1px solid rgba(16, 185, 129, 0.2);
                ">
                    <select id="proof-type-${this.containerId}" style="
                        padding: 0.5rem 1rem;
                        border: 1px solid #D1D5DB;
                        border-radius: 0.5rem;
                        background: white;
                        font-size: 0.9rem;
                    ">
                        <option value="boolean">Boolean Algebra</option>
                        <option value="sets">Set Theory</option>
                        <option value="modular">Modular Arithmetic</option>
                        <option value="tropical">Tropical Algebra</option>
                        <option value="lattice">Lattice Theory</option>
                    </select>
                    
                    <button id="step-btn-${this.containerId}" style="
                        padding: 0.5rem 1rem;
                        background: #10B981;
                        color: white;
                        border: none;
                        border-radius: 0.5rem;
                        cursor: pointer;
                        font-size: 0.9rem;
                        transition: all 0.3s ease;
                    ">📚 Show Steps</button>
                    
                    <button id="table-btn-${this.containerId}" style="
                        padding: 0.5rem 1rem;
                        background: #F59E0B;
                        color: white;
                        border: none;
                        border-radius: 0.5rem;
                        cursor: pointer;
                        font-size: 0.9rem;
                        transition: all 0.3s ease;
                    ">📊 Truth Table</button>
                    
                    <button id="animate-btn-${this.containerId}" style="
                        padding: 0.5rem 1rem;
                        background: #3B82F6;
                        color: white;
                        border: none;
                        border-radius: 0.5rem;
                        cursor: pointer;
                        font-size: 0.9rem;
                        transition: all 0.3s ease;
                    ">🎬 Animate</button>
                </div>
                
                <div id="proof-visualization-${this.containerId}" style="
                    width: 100%;
                    min-height: 400px;
                    background: white;
                    border-radius: 0.75rem;
                    box-shadow: inset 0 2px 4px rgba(0,0,0,0.06);
                    padding: 1.5rem;
                    margin-bottom: 1rem;
                "></div>
                
                <div id="truth-table-${this.containerId}" style="
                    display: ${this.config.show_truth_table ? 'block' : 'none'};
                    background: rgba(245, 158, 11, 0.05);
                    border-radius: 0.75rem;
                    padding: 1rem;
                    margin-top: 1rem;
                    border: 1px solid rgba(245, 158, 11, 0.2);
                "></div>
            </div>
        `;
        
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        const proofSelect = document.getElementById(`proof-type-${this.containerId}`);
        const stepBtn = document.getElementById(`step-btn-${this.containerId}`);
        const tableBtn = document.getElementById(`table-btn-${this.containerId}`);
        const animateBtn = document.getElementById(`animate-btn-${this.containerId}`);
        
        if (proofSelect) {
            proofSelect.addEventListener('change', (e) => {
                this.currentProofType = e.target.value;
                this.updateVisualization();
            });
        }
        
        if (stepBtn) {
            stepBtn.addEventListener('click', () => {
                this.config.show_steps = !this.config.show_steps;
                stepBtn.style.background = this.config.show_steps ? '#10B981' : '#6B7280';
                this.updateVisualization();
            });
        }
        
        if (tableBtn) {
            tableBtn.addEventListener('click', () => {
                this.config.show_truth_table = !this.config.show_truth_table;
                tableBtn.style.background = this.config.show_truth_table ? '#F59E0B' : '#6B7280';
                const tableDiv = document.getElementById(`truth-table-${this.containerId}`);
                if (tableDiv) {
                    tableDiv.style.display = this.config.show_truth_table ? 'block' : 'none';
                }
                this.updateVisualization();
            });
        }
        
        if (animateBtn) {
            animateBtn.addEventListener('click', () => {
                if (this.isAnimating) {
                    this.stopAnimation();
                    animateBtn.textContent = '🎬 Animate';
                    animateBtn.style.background = '#3B82F6';
                } else {
                    this.startAnimation();
                    animateBtn.textContent = '⏸️ Stop';
                    animateBtn.style.background = '#DC2626';
                }
            });
        }
    }
    
    createVisualization() {
        this.updateVisualization();
    }
    
    updateVisualization() {
        const vizDiv = document.getElementById(`proof-visualization-${this.containerId}`);
        const tableDiv = document.getElementById(`truth-table-${this.containerId}`);
        
        if (!vizDiv) return;
        
        switch (this.currentProofType) {
            case 'boolean':
                this.createBooleanAlgebraProof(vizDiv, tableDiv);
                break;
            case 'sets':
                this.createSetTheoryProof(vizDiv, tableDiv);
                break;
            case 'modular':
                this.createModularArithmeticProof(vizDiv, tableDiv);
                break;
            case 'tropical':
                this.createTropicalAlgebraProof(vizDiv, tableDiv);
                break;
            case 'lattice':
                this.createLatticeTheoryProof(vizDiv, tableDiv);
                break;
        }
    }
    
    createBooleanAlgebraProof(vizDiv, tableDiv) {
        vizDiv.innerHTML = `
            <div style="text-align: center; margin-bottom: 2rem;">
                <h4 style="color: #10B981; margin-bottom: 1rem;">Boolean Algebra: Idempotent Law</h4>
                <div style="font-size: 1.2rem; color: #374151; margin-bottom: 1.5rem;">
                    In Boolean algebra, the idempotent law states: <strong>A ∨ A = A</strong>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; align-items: start;">
                <div style="background: #F3F4F6; padding: 1.5rem; border-radius: 0.75rem;">
                    <h5 style="color: #374151; margin-bottom: 1rem;">Visual Representation</h5>
                    <div id="boolean-venn-${this.containerId}" style="height: 200px; display: flex; align-items: center; justify-content: center;">
                        <svg width="200" height="150" viewBox="0 0 200 150">
                            <circle cx="75" cy="75" r="50" fill="rgba(16, 185, 129, 0.3)" stroke="#10B981" stroke-width="2"/>
                            <circle cx="125" cy="75" r="50" fill="rgba(16, 185, 129, 0.3)" stroke="#10B981" stroke-width="2"/>
                            <text x="75" y="80" text-anchor="middle" style="font-size: 14px; font-weight: bold;">A</text>
                            <text x="125" y="80" text-anchor="middle" style="font-size: 14px; font-weight: bold;">A</text>
                            <text x="100" y="40" text-anchor="middle" style="font-size: 12px; fill: #10B981;">A ∨ A = A</text>
                        </svg>
                    </div>
                </div>
                
                <div style="background: #F3F4F6; padding: 1.5rem; border-radius: 0.75rem;">
                    <h5 style="color: #374151; margin-bottom: 1rem;">Algebraic Steps</h5>
                    <div style="font-family: 'Courier New', monospace; line-height: 1.8;">
                        ${this.config.show_steps ? `
                            <div>1. Let A be any Boolean variable</div>
                            <div>2. A ∨ A = A (idempotent law)</div>
                            <div>3. If A = 1, then: 1 ∨ 1 = 1</div>
                            <div>4. If A = 0, then: 0 ∨ 0 = 0</div>
                            <div style="color: #10B981; font-weight: bold; margin-top: 1rem;">
                                ∴ For unity (A = 1): <span style="color: #F59E0B;">1 + 1 = 1</span>
                            </div>
                        ` : '<div style="color: #6B7280;">Click "Show Steps" to see algebraic derivation</div>'}
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 2rem; padding: 1rem; background: rgba(16, 185, 129, 0.1); border-radius: 0.5rem; border-left: 4px solid #10B981;">
                <strong>Mathematical Foundation:</strong> In Boolean logic, OR operation (∨) with identical operands always returns the operand itself. 
                This demonstrates that unity maintains itself: two instances of the same truth value unify to that value.
            </div>
        `;
        
        if (this.config.show_truth_table && tableDiv) {
            this.createBooleanTruthTable(tableDiv);
        }
    }
    
    createBooleanTruthTable(tableDiv) {
        tableDiv.innerHTML = `
            <h5 style="color: #F59E0B; margin-bottom: 1rem; text-align: center;">Boolean Truth Table: A ∨ A = A</h5>
            <div style="overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse; margin: 0 auto; max-width: 400px;">
                    <thead>
                        <tr style="background: #F59E0B; color: white;">
                            <th style="padding: 0.75rem; border: 1px solid #D1D5DB; text-align: center;">A</th>
                            <th style="padding: 0.75rem; border: 1px solid #D1D5DB; text-align: center;">A</th>
                            <th style="padding: 0.75rem; border: 1px solid #D1D5DB; text-align: center;">A ∨ A</th>
                            <th style="padding: 0.75rem; border: 1px solid #D1D5DB; text-align: center;">Unity Check</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="background: #FEFEFE;">
                            <td style="padding: 0.75rem; border: 1px solid #D1D5DB; text-align: center;">0</td>
                            <td style="padding: 0.75rem; border: 1px solid #D1D5DB; text-align: center;">0</td>
                            <td style="padding: 0.75rem; border: 1px solid #D1D5DB; text-align: center; font-weight: bold; color: #10B981;">0</td>
                            <td style="padding: 0.75rem; border: 1px solid #D1D5DB; text-align: center;">0 = 0 ✓</td>
                        </tr>
                        <tr style="background: #FEF3C7; border: 2px solid #F59E0B;">
                            <td style="padding: 0.75rem; border: 1px solid #D1D5DB; text-align: center; font-weight: bold;">1</td>
                            <td style="padding: 0.75rem; border: 1px solid #D1D5DB; text-align: center; font-weight: bold;">1</td>
                            <td style="padding: 0.75rem; border: 1px solid #D1D5DB; text-align: center; font-weight: bold; color: #F59E0B;">1</td>
                            <td style="padding: 0.75rem; border: 1px solid #D1D5DB; text-align: center; font-weight: bold; color: #F59E0B;">1 + 1 = 1 ✓</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            <div style="text-align: center; margin-top: 1rem; font-size: 0.9rem; color: #6B7280;">
                The highlighted row demonstrates the unity principle in Boolean algebra
            </div>
        `;
    }
    
    createSetTheoryProof(vizDiv, tableDiv) {
        vizDiv.innerHTML = `
            <div style="text-align: center; margin-bottom: 2rem;">
                <h4 style="color: #10B981; margin-bottom: 1rem;">Set Theory: Idempotent Union</h4>
                <div style="font-size: 1.2rem; color: #374151; margin-bottom: 1.5rem;">
                    In set theory: <strong>A ∪ A = A</strong> (Union of a set with itself)
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
                <div style="background: #F3F4F6; padding: 1.5rem; border-radius: 0.75rem;">
                    <h5 style="color: #374151; margin-bottom: 1rem;">Venn Diagram</h5>
                    <div style="height: 200px; display: flex; align-items: center; justify-content: center;">
                        <svg width="200" height="150" viewBox="0 0 200 150">
                            <circle cx="100" cy="75" r="60" fill="rgba(16, 185, 129, 0.4)" stroke="#10B981" stroke-width="3"/>
                            <text x="100" y="80" text-anchor="middle" style="font-size: 16px; font-weight: bold; fill: #10B981;">A</text>
                            <text x="100" y="25" text-anchor="middle" style="font-size: 12px; fill: #374151;">A ∪ A = A</text>
                        </svg>
                    </div>
                </div>
                
                <div style="background: #F3F4F6; padding: 1.5rem; border-radius: 0.75rem;">
                    <h5 style="color: #374151; margin-bottom: 1rem;">Set Operations</h5>
                    <div style="font-family: 'Courier New', monospace; line-height: 1.8;">
                        ${this.config.show_steps ? `
                            <div>1. Let A = {elements in set A}</div>
                            <div>2. A ∪ A = {all elements in A or A}</div>
                            <div>3. Since A and A are identical:</div>
                            <div>   A ∪ A = {elements in A}</div>
                            <div>4. Therefore: A ∪ A = A</div>
                            <div style="color: #10B981; font-weight: bold; margin-top: 1rem;">
                                Unity example: {1} ∪ {1} = {1}
                            </div>
                        ` : '<div style="color: #6B7280;">Click "Show Steps" to see set operations</div>'}
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 2rem; padding: 1rem; background: rgba(16, 185, 129, 0.1); border-radius: 0.5rem; border-left: 4px solid #10B981;">
                <strong>Set Theory Principle:</strong> The union of any set with itself equals the original set. 
                This fundamental property demonstrates that unity preserves identity in set operations.
            </div>
        `;
        
        if (this.config.show_truth_table && tableDiv) {
            this.createSetTruthTable(tableDiv);
        }
    }
    
    createSetTruthTable(tableDiv) {
        tableDiv.innerHTML = `
            <h5 style="color: #F59E0B; margin-bottom: 1rem; text-align: center;">Set Union Examples</h5>
            <div style="overflow-x: auto;">
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="background: #F59E0B; color: white;">
                            <th style="padding: 0.75rem; border: 1px solid #D1D5DB; text-align: center;">Set A</th>
                            <th style="padding: 0.75rem; border: 1px solid #D1D5DB; text-align: center;">Set A</th>
                            <th style="padding: 0.75rem; border: 1px solid #D1D5DB; text-align: center;">A ∪ A</th>
                            <th style="padding: 0.75rem; border: 1px solid #D1D5DB; text-align: center;">Unity Verification</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td style="padding: 0.75rem; border: 1px solid #D1D5DB; text-align: center;">{}</td>
                            <td style="padding: 0.75rem; border: 1px solid #D1D5DB; text-align: center;">{}</td>
                            <td style="padding: 0.75rem; border: 1px solid #D1D5DB; text-align: center; color: #10B981;">{}</td>
                            <td style="padding: 0.75rem; border: 1px solid #D1D5DB; text-align: center;">∅ = ∅ ✓</td>
                        </tr>
                        <tr style="background: #FEF3C7; border: 2px solid #F59E0B;">
                            <td style="padding: 0.75rem; border: 1px solid #D1D5DB; text-align: center; font-weight: bold;">{1}</td>
                            <td style="padding: 0.75rem; border: 1px solid #D1D5DB; text-align: center; font-weight: bold;">{1}</td>
                            <td style="padding: 0.75rem; border: 1px solid #D1D5DB; text-align: center; font-weight: bold; color: #F59E0B;">{1}</td>
                            <td style="padding: 0.75rem; border: 1px solid #D1D5DB; text-align: center; font-weight: bold; color: #F59E0B;">{1} ∪ {1} = {1} ✓</td>
                        </tr>
                        <tr>
                            <td style="padding: 0.75rem; border: 1px solid #D1D5DB; text-align: center;">{a,b}</td>
                            <td style="padding: 0.75rem; border: 1px solid #D1D5DB; text-align: center;">{a,b}</td>
                            <td style="padding: 0.75rem; border: 1px solid #D1D5DB; text-align: center; color: #10B981;">{a,b}</td>
                            <td style="padding: 0.75rem; border: 1px solid #D1D5DB; text-align: center;">{a,b} = {a,b} ✓</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        `;
    }
    
    createModularArithmeticProof(vizDiv, tableDiv) {
        vizDiv.innerHTML = `
            <div style="text-align: center; margin-bottom: 2rem;">
                <h4 style="color: #10B981; margin-bottom: 1rem;">Modular Arithmetic: Unity in Mod 2</h4>
                <div style="font-size: 1.2rem; color: #374151; margin-bottom: 1.5rem;">
                    In modular arithmetic (mod 2): <strong>1 + 1 ≡ 0 (mod 2)</strong> but in Boolean sense: <strong>1 ⊕ 1 = 0, 1 ∨ 1 = 1</strong>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
                <div style="background: #F3F4F6; padding: 1.5rem; border-radius: 0.75rem;">
                    <h5 style="color: #374151; margin-bottom: 1rem;">Clock Arithmetic (Mod 2)</h5>
                    <div style="height: 200px; display: flex; align-items: center; justify-content: center;">
                        <svg width="150" height="150" viewBox="0 0 150 150">
                            <circle cx="75" cy="75" r="60" fill="none" stroke="#10B981" stroke-width="3"/>
                            <circle cx="75" cy="30" r="8" fill="#F59E0B"/>
                            <circle cx="75" cy="120" r="8" fill="#3B82F6"/>
                            <text x="75" y="20" text-anchor="middle" style="font-size: 12px; font-weight: bold;">1</text>
                            <text x="75" y="135" text-anchor="middle" style="font-size: 12px; font-weight: bold;">0</text>
                            <text x="75" y="80" text-anchor="middle" style="font-size: 10px;">Mod 2</text>
                        </svg>
                    </div>
                </div>
                
                <div style="background: #F3F4F6; padding: 1.5rem; border-radius: 0.75rem;">
                    <h5 style="color: #374151; margin-bottom: 1rem;">Different Operations</h5>
                    <div style="font-family: 'Courier New', monospace; line-height: 1.8;">
                        ${this.config.show_steps ? `
                            <div><strong>Addition (mod 2):</strong></div>
                            <div>1 + 1 ≡ 0 (mod 2)</div>
                            <div><strong>XOR operation:</strong></div>
                            <div>1 ⊕ 1 = 0</div>
                            <div><strong>OR operation (Unity):</strong></div>
                            <div style="color: #F59E0B; font-weight: bold;">1 ∨ 1 = 1</div>
                            <div style="margin-top: 1rem; color: #10B981;">
                                Context determines unity!
                            </div>
                        ` : '<div style="color: #6B7280;">Click "Show Steps" to see operations</div>'}
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 2rem; padding: 1rem; background: rgba(245, 158, 11, 0.1); border-radius: 0.5rem; border-left: 4px solid #F59E0B;">
                <strong>Context Matters:</strong> Different mathematical contexts yield different results. 
                In Boolean logic (∨), unity is preserved. In XOR logic (⊕), unity transforms. 
                The choice of operation defines the meaning of unity.
            </div>
        `;
    }
    
    createTropicalAlgebraProof(vizDiv, tableDiv) {
        vizDiv.innerHTML = `
            <div style="text-align: center; margin-bottom: 2rem;">
                <h4 style="color: #10B981; margin-bottom: 1rem;">Tropical Algebra: Idempotent Semiring</h4>
                <div style="font-size: 1.2rem; color: #374151; margin-bottom: 1.5rem;">
                    In tropical algebra: <strong>a ⊕ a = a</strong> (where ⊕ = min or max)
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
                <div style="background: #F3F4F6; padding: 1.5rem; border-radius: 0.75rem;">
                    <h5 style="color: #374151; margin-bottom: 1rem;">Min-Plus Algebra</h5>
                    <div style="height: 150px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                        <div style="font-size: 1.1rem; margin-bottom: 1rem; color: #10B981;">
                            <strong>⊕ = min operation</strong>
                        </div>
                        <div style="font-family: 'Courier New', monospace; text-align: center;">
                            <div>1 ⊕ 1 = min(1, 1) = 1</div>
                            <div>5 ⊕ 5 = min(5, 5) = 5</div>
                            <div style="color: #F59E0B; font-weight: bold; margin-top: 0.5rem;">a ⊕ a = a</div>
                        </div>
                    </div>
                </div>
                
                <div style="background: #F3F4F6; padding: 1.5rem; border-radius: 0.75rem;">
                    <h5 style="color: #374151; margin-bottom: 1rem;">Max-Plus Algebra</h5>
                    <div style="height: 150px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
                        <div style="font-size: 1.1rem; margin-bottom: 1rem; color: #3B82F6;">
                            <strong>⊕ = max operation</strong>
                        </div>
                        <div style="font-family: 'Courier New', monospace; text-align: center;">
                            <div>1 ⊕ 1 = max(1, 1) = 1</div>
                            <div>7 ⊕ 7 = max(7, 7) = 7</div>
                            <div style="color: #F59E0B; font-weight: bold; margin-top: 0.5rem;">a ⊕ a = a</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 1.5rem;">
                <h5 style="color: #374151; margin-bottom: 1rem;">Idempotent Property Visualization</h5>
                <div style="background: white; padding: 1.5rem; border-radius: 0.75rem; border: 1px solid #E5E7EB;">
                    ${this.config.show_steps ? `
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; text-align: center;">
                            <div style="padding: 1rem; background: #FEF3C7; border-radius: 0.5rem;">
                                <div style="font-size: 1.2rem; color: #92400E; font-weight: bold;">Min Operation</div>
                                <div style="margin: 0.5rem 0; font-family: monospace;">min(a, a) = a</div>
                                <div style="font-size: 0.9rem; color: #6B7280;">Always returns the value</div>
                            </div>
                            <div style="padding: 1rem; background: #DBEAFE; border-radius: 0.5rem;">
                                <div style="font-size: 1.2rem; color: #1E40AF; font-weight: bold;">Max Operation</div>
                                <div style="margin: 0.5rem 0; font-family: monospace;">max(a, a) = a</div>
                                <div style="font-size: 0.9rem; color: #6B7280;">Always returns the value</div>
                            </div>
                            <div style="padding: 1rem; background: #D1FAE5; border-radius: 0.5rem;">
                                <div style="font-size: 1.2rem; color: #065F46; font-weight: bold;">Unity Result</div>
                                <div style="margin: 0.5rem 0; font-family: monospace; color: #F59E0B; font-weight: bold;">1 ⊕ 1 = 1</div>
                                <div style="font-size: 0.9rem; color: #6B7280;">Idempotent unity</div>
                            </div>
                        </div>
                    ` : '<div style="text-align: center; color: #6B7280;">Click "Show Steps" to see operations</div>'}
                </div>
            </div>
            
            <div style="margin-top: 2rem; padding: 1rem; background: rgba(16, 185, 129, 0.1); border-radius: 0.5rem; border-left: 4px solid #10B981;">
                <strong>Tropical Mathematics:</strong> Tropical algebra replaces addition with min/max operations. 
                The idempotent property (a ⊕ a = a) is fundamental, directly demonstrating mathematical unity where 1 ⊕ 1 = 1.
            </div>
        `;
    }
    
    createLatticeTheoryProof(vizDiv, tableDiv) {
        vizDiv.innerHTML = `
            <div style="text-align: center; margin-bottom: 2rem;">
                <h4 style="color: #10B981; margin-bottom: 1rem;">Lattice Theory: Join Operation</h4>
                <div style="font-size: 1.2rem; color: #374151; margin-bottom: 1.5rem;">
                    In lattice theory: <strong>a ∨ a = a</strong> (join of element with itself)
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
                <div style="background: #F3F4F6; padding: 1.5rem; border-radius: 0.75rem;">
                    <h5 style="color: #374151; margin-bottom: 1rem;">Lattice Diagram</h5>
                    <div style="height: 200px; display: flex; align-items: center; justify-content: center;">
                        <svg width="150" height="180" viewBox="0 0 150 180">
                            <!-- Lattice structure -->
                            <circle cx="75" cy="30" r="6" fill="#F59E0B"/>
                            <circle cx="50" cy="80" r="6" fill="#3B82F6"/>
                            <circle cx="100" cy="80" r="6" fill="#10B981"/>
                            <circle cx="75" cy="130" r="6" fill="#DC2626"/>
                            
                            <!-- Connections -->
                            <line x1="75" y1="36" x2="50" y2="74" stroke="#6B7280" stroke-width="2"/>
                            <line x1="75" y1="36" x2="100" y2="74" stroke="#6B7280" stroke-width="2"/>
                            <line x1="50" y1="86" x2="75" y2="124" stroke="#6B7280" stroke-width="2"/>
                            <line x1="100" y1="86" x2="75" y2="124" stroke="#6B7280" stroke-width="2"/>
                            
                            <!-- Labels -->
                            <text x="75" y="20" text-anchor="middle" style="font-size: 10px;">⊤ (top)</text>
                            <text x="40" y="85" text-anchor="middle" style="font-size: 10px;">a</text>
                            <text x="110" y="85" text-anchor="middle" style="font-size: 10px;">a</text>
                            <text x="75" y="145" text-anchor="middle" style="font-size: 10px;">⊥ (bottom)</text>
                            
                            <!-- Join indication -->
                            <path d="M 60 75 Q 75 65 90 75" stroke="#F59E0B" stroke-width="2" fill="none"/>
                            <text x="75" y="60" text-anchor="middle" style="font-size: 9px; fill: #F59E0B;">a ∨ a = a</text>
                        </svg>
                    </div>
                </div>
                
                <div style="background: #F3F4F6; padding: 1.5rem; border-radius: 0.75rem;">
                    <h5 style="color: #374151; margin-bottom: 1rem;">Lattice Properties</h5>
                    <div style="font-family: 'Courier New', monospace; line-height: 1.8; font-size: 0.9rem;">
                        ${this.config.show_steps ? `
                            <div><strong>Idempotent Law:</strong></div>
                            <div>a ∨ a = a (join)</div>
                            <div>a ∧ a = a (meet)</div>
                            <div><strong>For any lattice element a:</strong></div>
                            <div>• a joined with itself = a</div>
                            <div>• a meets with itself = a</div>
                            <div><strong>Unity case (a = 1):</strong></div>
                            <div style="color: #F59E0B; font-weight: bold;">1 ∨ 1 = 1</div>
                        ` : '<div style="color: #6B7280;">Click "Show Steps" to see lattice properties</div>'}
                    </div>
                </div>
            </div>
            
            <div style="margin-top: 2rem; padding: 1rem; background: rgba(16, 185, 129, 0.1); border-radius: 0.5rem; border-left: 4px solid #10B981;">
                <strong>Lattice Theory Foundation:</strong> In any lattice, the join (∨) and meet (∧) operations are idempotent. 
                This fundamental property ensures that combining any element with itself yields the element unchanged, 
                demonstrating perfect unity preservation.
            </div>
        `;
    }
    
    startAnimation() {
        this.isAnimating = true;
        // Implementation for animation effects
    }
    
    stopAnimation() {
        this.isAnimating = false;
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
    }
}

// Gallery creation function
function createIdempotentProofInteractive(containerId) {
    return new IdempotentProofVisualizer(containerId);
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { IdempotentProofVisualizer, createIdempotentProofInteractive };
}

// Browser global
if (typeof window !== 'undefined') {
    window.IdempotentProofVisualizer = IdempotentProofVisualizer;
    window.createIdempotentProofInteractive = createIdempotentProofInteractive;
}