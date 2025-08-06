/**
 * Interactive Euler's Identity Showcase
 * e^(iπ) + 1 = 0 with unit circle animation and mathematical beauty
 */
class EulerIdentityVisualizer {
    constructor(containerId) {
        this.containerId = containerId;
        this.PHI = (1 + Math.sqrt(5)) / 2;
        this.E = Math.E;
        this.PI = Math.PI;
        this.animationFrame = null;
        this.isAnimating = false;
        this.angle = 0;

        this.init();
    }

    init() {
        const container = document.getElementById(this.containerId);
        if (!container) return;

        container.innerHTML = `
            <div class="euler-identity-container">
                <div class="euler-header">
                    <h3>Euler's Identity: The Most Beautiful Equation</h3>
                    <div class="euler-equation-display">
                        <div class="main-equation">e<sup>iπ</sup> + 1 = 0</div>
                        <div class="sub-equation">The unity of all fundamental constants</div>
                    </div>
                </div>
                
                <div class="euler-content">
                    <div class="euler-visualization" id="euler-viz"></div>
                    <div class="euler-explanation" id="euler-explanation"></div>
                </div>
                
                <div class="euler-controls">
                    <button id="animate-euler" class="control-btn">Animate</button>
                    <button id="step-through-euler" class="control-btn">Step Through</button>
                    <button id="reset-euler" class="control-btn">Reset</button>
                </div>
            </div>
        `;

        this.setupEventListeners();
        this.renderEulerIdentity();
    }

    setupEventListeners() {
        const animateBtn = document.getElementById('animate-euler');
        const stepBtn = document.getElementById('step-through-euler');
        const resetBtn = document.getElementById('reset-euler');

        animateBtn?.addEventListener('click', () => {
            this.toggleAnimation();
        });

        stepBtn?.addEventListener('click', () => {
            this.stepThroughProof();
        });

        resetBtn?.addEventListener('click', () => {
            this.resetEuler();
        });
    }

    renderEulerIdentity() {
        const vizContainer = document.getElementById('euler-viz');
        const explanationContainer = document.getElementById('euler-explanation');

        if (!vizContainer || !explanationContainer) return;

        vizContainer.innerHTML = `
            <div class="euler-visualization-content">
                <div class="unit-circle-container">
                    <h4>Unit Circle: e<sup>iθ</sup> = cos(θ) + i·sin(θ)</h4>
                    <canvas id="unit-circle-canvas" width="500" height="500"></canvas>
                </div>
                
                <div class="euler-constants">
                    <div class="constant-item">
                        <div class="constant-symbol">e</div>
                        <div class="constant-value">${this.E.toFixed(6)}</div>
                        <div class="constant-name">Natural Base</div>
                    </div>
                    <div class="constant-item">
                        <div class="constant-symbol">i</div>
                        <div class="constant-value">√(-1)</div>
                        <div class="constant-name">Imaginary Unit</div>
                    </div>
                    <div class="constant-item">
                        <div class="constant-symbol">π</div>
                        <div class="constant-value">${this.PI.toFixed(6)}</div>
                        <div class="constant-name">Pi</div>
                    </div>
                    <div class="constant-item">
                        <div class="constant-symbol">1</div>
                        <div class="constant-value">1</div>
                        <div class="constant-name">Unity</div>
                    </div>
                    <div class="constant-item">
                        <div class="constant-symbol">0</div>
                        <div class="constant-value">0</div>
                        <div class="constant-name">Zero</div>
                    </div>
                </div>
            </div>
        `;

        explanationContainer.innerHTML = `
            <div class="euler-explanation-content">
                <h4>Euler's Identity: Mathematical Unity</h4>
                <p>Euler's identity e<sup>iπ</sup> + 1 = 0 is considered the most beautiful equation in mathematics:</p>
                <ul>
                    <li><strong>e</strong> - The natural base of logarithms (≈ 2.718282)</li>
                    <li><strong>i</strong> - The imaginary unit (√(-1))</li>
                    <li><strong>π</strong> - The ratio of circle circumference to diameter (≈ 3.141593)</li>
                    <li><strong>1</strong> - The multiplicative identity</li>
                    <li><strong>0</strong> - The additive identity</li>
                </ul>
                <p>This equation unifies the five most fundamental constants in mathematics through 
                the elegant relationship of complex exponential functions.</p>
                
                <div class="mathematical-derivation">
                    <h5>Mathematical Derivation:</h5>
                    <ol>
                        <li>e<sup>iθ</sup> = cos(θ) + i·sin(θ) (Euler's formula)</li>
                        <li>When θ = π: e<sup>iπ</sup> = cos(π) + i·sin(π)</li>
                        <li>cos(π) = -1 and sin(π) = 0</li>
                        <li>Therefore: e<sup>iπ</sup> = -1 + i·0 = -1</li>
                        <li>Adding 1: e<sup>iπ</sup> + 1 = -1 + 1 = 0</li>
                    </ol>
                </div>
                
                <div class="unity-connection">
                    <h5>Connection to Unity Mathematics:</h5>
                    <p>Euler's identity demonstrates how complex mathematical relationships can resolve to unity (0), 
                    showing that apparent complexity can collapse into simple, beautiful unity.</p>
                </div>
            </div>
        `;

        this.setupUnitCircle();
    }

    setupUnitCircle() {
        const canvas = document.getElementById('unit-circle-canvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        const centerX = width / 2;
        const centerY = height / 2;
        const radius = 180;

        function drawUnitCircle(ctx, width, height, angle = 0) {
            ctx.clearRect(0, 0, width, height);

            // Draw coordinate axes
            ctx.strokeStyle = 'rgba(255, 215, 0, 0.3)';
            ctx.lineWidth = 1;

            // X-axis
            ctx.beginPath();
            ctx.moveTo(0, centerY);
            ctx.lineTo(width, centerY);
            ctx.stroke();

            // Y-axis
            ctx.beginPath();
            ctx.moveTo(centerX, 0);
            ctx.lineTo(centerX, height);
            ctx.stroke();

            // Draw unit circle
            ctx.strokeStyle = '#FFD700';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
            ctx.stroke();

            // Draw angle arc
            ctx.strokeStyle = 'rgba(255, 215, 0, 0.6)';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(centerX, centerY, radius, 0, angle);
            ctx.stroke();

            // Draw radius line
            ctx.strokeStyle = '#FFD700';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(centerX, centerY);
            ctx.lineTo(
                centerX + radius * Math.cos(angle),
                centerY - radius * Math.sin(angle)
            );
            ctx.stroke();

            // Draw point on circle
            ctx.fillStyle = '#FFD700';
            ctx.beginPath();
            ctx.arc(
                centerX + radius * Math.cos(angle),
                centerY - radius * Math.sin(angle),
                6,
                0,
                2 * Math.PI
            );
            ctx.fill();

            // Draw angle label
            ctx.fillStyle = '#FFD700';
            ctx.font = '14px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(`θ = ${(angle * 180 / Math.PI).toFixed(1)}°`, centerX + 100, centerY - 100);

            // Draw coordinates
            const x = Math.cos(angle);
            const y = Math.sin(angle);
            ctx.fillText(`(${x.toFixed(3)}, ${y.toFixed(3)})`, centerX + 100, centerY - 80);

            // Draw e^(iθ) representation
            ctx.fillText(`e^(iθ) = ${x.toFixed(3)} + i·${y.toFixed(3)}`, centerX + 100, centerY - 60);

            // Special case for π
            if (Math.abs(angle - Math.PI) < 0.1) {
                ctx.fillStyle = '#FF6B6B';
                ctx.font = '16px Arial';
                ctx.fillText('e^(iπ) = -1', centerX, centerY + 120);
                ctx.fillText('∴ e^(iπ) + 1 = 0', centerX, centerY + 140);
            }

            // Draw angle markers
            ctx.strokeStyle = 'rgba(255, 215, 0, 0.4)';
            ctx.lineWidth = 1;

            // Mark π/2, π, 3π/2, 2π
            const specialAngles = [Math.PI / 2, Math.PI, 3 * Math.PI / 2, 2 * Math.PI];
            specialAngles.forEach((specialAngle, index) => {
                const x = centerX + (radius + 20) * Math.cos(specialAngle);
                const y = centerY - (radius + 20) * Math.sin(specialAngle);

                ctx.beginPath();
                ctx.arc(x, y, 3, 0, 2 * Math.PI);
                ctx.fill();

                const labels = ['π/2', 'π', '3π/2', '2π'];
                ctx.fillStyle = '#FFD700';
                ctx.font = '12px Arial';
                ctx.fillText(labels[index], x, y - 10);
            });
        }

        drawUnitCircle(ctx, width, height, this.angle);
        this.drawFunction = (angle) => drawUnitCircle(ctx, width, height, angle);
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
        const animateBtn = document.getElementById('animate-euler');
        if (animateBtn) animateBtn.textContent = 'Stop Animation';

        this.animateEuler();
    }

    stopAnimation() {
        this.isAnimating = false;
        const animateBtn = document.getElementById('animate-euler');
        if (animateBtn) animateBtn.textContent = 'Animate';

        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
    }

    animateEuler() {
        if (!this.isAnimating) return;

        this.angle += 0.02;

        if (this.angle >= 2 * Math.PI) {
            this.angle = 0;
        }

        if (this.drawFunction) {
            this.drawFunction(this.angle);
        }

        this.animationFrame = requestAnimationFrame(() => {
            this.animateEuler();
        });
    }

    stepThroughProof() {
        // Step through the mathematical proof
        const steps = [
            { angle: 0, description: 'Start at θ = 0' },
            { angle: Math.PI / 4, description: 'θ = π/4' },
            { angle: Math.PI / 2, description: 'θ = π/2' },
            { angle: 3 * Math.PI / 4, description: 'θ = 3π/4' },
            { angle: Math.PI, description: 'θ = π (Euler\'s identity)' },
            { angle: 5 * Math.PI / 4, description: 'θ = 5π/4' },
            { angle: 3 * Math.PI / 2, description: 'θ = 3π/2' },
            { angle: 7 * Math.PI / 4, description: 'θ = 7π/4' },
            { angle: 2 * Math.PI, description: 'θ = 2π (complete circle)' }
        ];

        let currentStep = 0;

        const stepInterval = setInterval(() => {
            if (currentStep < steps.length) {
                this.angle = steps[currentStep].angle;
                if (this.drawFunction) {
                    this.drawFunction(this.angle);
                }

                // Update description
                const explanationContainer = document.getElementById('euler-explanation');
                if (explanationContainer) {
                    const stepDesc = explanationContainer.querySelector('.step-description');
                    if (stepDesc) {
                        stepDesc.textContent = steps[currentStep].description;
                    }
                }

                currentStep++;
            } else {
                clearInterval(stepInterval);
            }
        }, 1000);
    }

    resetEuler() {
        this.stopAnimation();
        this.angle = 0;
        this.renderEulerIdentity();
    }
}

// Global function to create the visualizer
function createEulerIdentityInteractive(containerId) {
    return new EulerIdentityVisualizer(containerId);
}