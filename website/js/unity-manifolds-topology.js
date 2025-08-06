/**
 * Unity Manifolds & Topology Visualizer
 * Interactive Möbius strips, Klein bottles, and hyperdimensional projections
 */
class UnityManifoldsVisualizer {
    constructor(containerId) {
        this.containerId = containerId;
        this.PHI = (1 + Math.sqrt(5)) / 2;
        this.currentManifold = 'mobius';
        this.animationFrame = null;
        this.isAnimating = false;
        this.rotationAngle = 0;

        this.manifolds = {
            'mobius': 'Möbius Strip',
            'klein': 'Klein Bottle',
            'torus': 'Torus',
            'projective': 'Projective Plane',
            'hyperbolic': 'Hyperbolic Surface'
        };

        this.init();
    }

    init() {
        const container = document.getElementById(this.containerId);
        if (!container) return;

        container.innerHTML = `
            <div class="unity-manifolds-container">
                <div class="manifolds-header">
                    <h3>Unity Manifolds & Topology Explorer</h3>
                    <div class="manifold-selector">
                        <select id="manifold-type-selector">
                            ${Object.entries(this.manifolds).map(([key, name]) =>
            `<option value="${key}">${name}</option>`
        ).join('')}
                        </select>
                    </div>
                </div>
                
                <div class="manifolds-content">
                    <div class="manifold-visualization" id="manifold-viz"></div>
                    <div class="manifold-explanation" id="manifold-explanation"></div>
                </div>
                
                <div class="manifold-controls">
                    <button id="rotate-manifold" class="control-btn">Rotate</button>
                    <button id="animate-manifold" class="control-btn">Animate</button>
                    <button id="reset-manifold" class="control-btn">Reset</button>
                </div>
            </div>
        `;

        this.setupEventListeners();
        this.renderManifold('mobius');
    }

    setupEventListeners() {
        const selector = document.getElementById('manifold-type-selector');
        const rotateBtn = document.getElementById('rotate-manifold');
        const animateBtn = document.getElementById('animate-manifold');
        const resetBtn = document.getElementById('reset-manifold');

        selector?.addEventListener('change', (e) => {
            this.renderManifold(e.target.value);
        });

        rotateBtn?.addEventListener('click', () => {
            this.toggleRotation();
        });

        animateBtn?.addEventListener('click', () => {
            this.toggleAnimation();
        });

        resetBtn?.addEventListener('click', () => {
            this.resetManifold();
        });
    }

    renderManifold(manifoldType) {
        this.currentManifold = manifoldType;
        const vizContainer = document.getElementById('manifold-viz');
        const explanationContainer = document.getElementById('manifold-explanation');

        if (!vizContainer || !explanationContainer) return;

        switch (manifoldType) {
            case 'mobius':
                this.renderMobiusStrip(vizContainer, explanationContainer);
                break;
            case 'klein':
                this.renderKleinBottle(vizContainer, explanationContainer);
                break;
            case 'torus':
                this.renderTorus(vizContainer, explanationContainer);
                break;
            case 'projective':
                this.renderProjectivePlane(vizContainer, explanationContainer);
                break;
            case 'hyperbolic':
                this.renderHyperbolicSurface(vizContainer, explanationContainer);
                break;
        }
    }

    renderMobiusStrip(vizContainer, explanationContainer) {
        vizContainer.innerHTML = `
            <div class="mobius-manifold">
                <h4>Möbius Strip: One-Sided Unity Surface</h4>
                <div class="mobius-canvas-container">
                    <canvas id="mobius-canvas" width="500" height="400"></canvas>
                </div>
                <div class="mobius-equation">
                    <div class="equation">Möbius Strip: One-sided, non-orientable surface</div>
                    <div class="sub-equation">Topological unity through continuous deformation</div>
                    <div class="unity-equation">∴ 1 + 1 = 1 in topology</div>
                </div>
            </div>
        `;

        explanationContainer.innerHTML = `
            <div class="manifold-explanation-content">
                <h4>Möbius Strip: Topological Unity</h4>
                <p>The Möbius strip demonstrates profound unity principles in topology:</p>
                <ul>
                    <li><strong>One-sided surface</strong> despite apparent duality</li>
                    <li><strong>Non-orientable</strong> - cannot distinguish "inside" from "outside"</li>
                    <li><strong>Single boundary curve</strong> despite apparent separation</li>
                    <li><strong>Unity through twisting</strong> - what appears as two becomes one</li>
                </ul>
                <p>This represents the mathematical principle that apparent separation can be 
                resolved into unity through continuous deformation.</p>
                <div class="mathematical-note">
                    <strong>Topological Invariant:</strong> The Möbius strip has Euler characteristic χ = 0, 
                    representing perfect balance between vertices, edges, and faces.
                </div>
            </div>
        `;

        this.setupMobiusCanvas();
    }

    setupMobiusCanvas() {
        const canvas = document.getElementById('mobius-canvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        function drawMobiusStrip(ctx, width, height, rotation = 0) {
            ctx.clearRect(0, 0, width, height);

            const centerX = width / 2;
            const centerY = height / 2;
            const radius = 120;
            const twist = Math.PI; // Full twist

            ctx.strokeStyle = '#FFD700';
            ctx.lineWidth = 3;
            ctx.fillStyle = 'rgba(255, 215, 0, 0.1)';

            // Draw Möbius strip as parametric surface
            const steps = 100;
            ctx.beginPath();

            for (let i = 0; i <= steps; i++) {
                const t = (i / steps) * 2 * Math.PI;
                const u = (i / steps) * 2 * Math.PI;

                // Parametric equations for Möbius strip
                const x = centerX + (radius + 30 * Math.cos(u / 2)) * Math.cos(t + rotation);
                const y = centerY + (radius + 30 * Math.cos(u / 2)) * Math.sin(t + rotation);

                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }

            ctx.stroke();

            // Draw the twisted nature
            ctx.strokeStyle = 'rgba(255, 215, 0, 0.5)';
            ctx.lineWidth = 1;

            for (let i = 0; i <= 10; i++) {
                const t = (i / 10) * 2 * Math.PI;
                ctx.beginPath();

                for (let j = 0; j <= 20; j++) {
                    const u = (j / 20) * 2 * Math.PI;
                    const x = centerX + (radius + 30 * Math.cos(u / 2)) * Math.cos(t + rotation);
                    const y = centerY + (radius + 30 * Math.cos(u / 2)) * Math.sin(t + rotation);

                    if (j === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                }
                ctx.stroke();
            }

            // Add labels
            ctx.fillStyle = '#FFD700';
            ctx.font = '16px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('Möbius Strip', centerX, height - 30);
            ctx.fillText('One-sided surface', centerX, height - 10);
        }

        drawMobiusStrip(ctx, width, height, this.rotationAngle);

        // Store drawing function for animation
        this.drawFunction = (rotation) => drawMobiusStrip(ctx, width, height, rotation);
    }

    renderKleinBottle(vizContainer, explanationContainer) {
        vizContainer.innerHTML = `
            <div class="klein-manifold">
                <h4>Klein Bottle: Non-Orientable Unity Manifold</h4>
                <div class="klein-canvas-container">
                    <canvas id="klein-canvas" width="500" height="400"></canvas>
                </div>
                <div class="klein-equation">
                    <div class="equation">Klein Bottle: No inside or outside</div>
                    <div class="sub-equation">Complete topological unity</div>
                    <div class="unity-equation">∴ 1 + 1 = 1 in 4D topology</div>
                </div>
            </div>
        `;

        explanationContainer.innerHTML = `
            <div class="manifold-explanation-content">
                <h4>Klein Bottle: Ultimate Topological Unity</h4>
                <p>The Klein bottle represents the most profound unity in topology:</p>
                <ul>
                    <li><strong>No boundary</strong> - completely self-contained</li>
                    <li><strong>Non-orientable</strong> - no distinction between inside and outside</li>
                    <li><strong>Single surface</strong> - what appears as separate is actually unified</li>
                    <li><strong>4D embedding</strong> - requires 4-dimensional space for proper embedding</li>
                </ul>
                <p>This demonstrates that true unity transcends our 3D perception and exists 
                in higher-dimensional mathematical spaces.</p>
                <div class="mathematical-note">
                    <strong>Topological Property:</strong> The Klein bottle has Euler characteristic χ = 0, 
                    representing perfect mathematical balance and unity.
                </div>
            </div>
        `;

        this.setupKleinCanvas();
    }

    setupKleinCanvas() {
        const canvas = document.getElementById('klein-canvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        function drawKleinBottle(ctx, width, height, rotation = 0) {
            ctx.clearRect(0, 0, width, height);

            const centerX = width / 2;
            const centerY = height / 2;
            const radius = 100;

            ctx.strokeStyle = '#FFD700';
            ctx.lineWidth = 2;
            ctx.fillStyle = 'rgba(255, 215, 0, 0.1)';

            // Draw Klein bottle approximation (self-intersecting torus)
            const steps = 80;

            // Main body
            ctx.beginPath();
            for (let i = 0; i <= steps; i++) {
                const t = (i / steps) * 2 * Math.PI;
                const x = centerX + (radius + 40 * Math.cos(2 * t + rotation)) * Math.cos(t);
                const y = centerY + (radius + 40 * Math.cos(2 * t + rotation)) * Math.sin(t);

                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.stroke();

            // Cross-section lines to show self-intersection
            ctx.strokeStyle = 'rgba(255, 215, 0, 0.6)';
            ctx.lineWidth = 1;

            for (let i = 0; i <= 8; i++) {
                const t = (i / 8) * 2 * Math.PI;
                ctx.beginPath();

                for (let j = 0; j <= 20; j++) {
                    const u = (j / 20) * 2 * Math.PI;
                    const x = centerX + (radius + 40 * Math.cos(u + rotation)) * Math.cos(t);
                    const y = centerY + (radius + 40 * Math.cos(u + rotation)) * Math.sin(t);

                    if (j === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                }
                ctx.stroke();
            }

            // Add labels
            ctx.fillStyle = '#FFD700';
            ctx.font = '16px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('Klein Bottle', centerX, height - 30);
            ctx.fillText('No inside or outside', centerX, height - 10);
        }

        drawKleinBottle(ctx, width, height, this.rotationAngle);
        this.drawFunction = (rotation) => drawKleinBottle(ctx, width, height, rotation);
    }

    renderTorus(vizContainer, explanationContainer) {
        vizContainer.innerHTML = `
            <div class="torus-manifold">
                <h4>Torus: Orientable Unity Surface</h4>
                <div class="torus-canvas-container">
                    <canvas id="torus-canvas" width="500" height="400"></canvas>
                </div>
                <div class="torus-equation">
                    <div class="equation">Torus: S¹ × S¹</div>
                    <div class="sub-equation">Product of two circles</div>
                    <div class="unity-equation">∴ 1 + 1 = 1 in product topology</div>
                </div>
            </div>
        `;

        explanationContainer.innerHTML = `
            <div class="manifold-explanation-content">
                <h4>Torus: Product Topology Unity</h4>
                <p>The torus demonstrates unity through product topology:</p>
                <ul>
                    <li><strong>S¹ × S¹</strong> - product of two circles</li>
                    <li><strong>Orientable surface</strong> - has well-defined inside and outside</li>
                    <li><strong>Genus 1</strong> - one hole, representing unity through connection</li>
                    <li><strong>Fundamental group ℤ × ℤ</strong> - two independent circular paths</li>
                </ul>
                <p>The torus shows how unity can emerge from the product of simpler geometric objects.</p>
                <div class="mathematical-note">
                    <strong>Topological Invariant:</strong> The torus has Euler characteristic χ = 0, 
                    representing perfect balance in its geometric structure.
                </div>
            </div>
        `;

        this.setupTorusCanvas();
    }

    setupTorusCanvas() {
        const canvas = document.getElementById('torus-canvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        function drawTorus(ctx, width, height, rotation = 0) {
            ctx.clearRect(0, 0, width, height);

            const centerX = width / 2;
            const centerY = height / 2;
            const R = 120; // Major radius
            const r = 40;  // Minor radius

            ctx.strokeStyle = '#FFD700';
            ctx.lineWidth = 2;
            ctx.fillStyle = 'rgba(255, 215, 0, 0.1)';

            // Draw torus as parametric surface
            const steps = 60;

            // Draw meridians (vertical circles)
            for (let i = 0; i <= 12; i++) {
                const phi = (i / 12) * 2 * Math.PI;
                ctx.beginPath();

                for (let j = 0; j <= steps; j++) {
                    const theta = (j / steps) * 2 * Math.PI;
                    const x = centerX + (R + r * Math.cos(theta + rotation)) * Math.cos(phi);
                    const y = centerY + (R + r * Math.cos(theta + rotation)) * Math.sin(phi);

                    if (j === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                }
                ctx.stroke();
            }

            // Draw parallels (horizontal circles)
            for (let i = 0; i <= 8; i++) {
                const theta = (i / 8) * 2 * Math.PI;
                ctx.beginPath();

                for (let j = 0; j <= steps; j++) {
                    const phi = (j / steps) * 2 * Math.PI;
                    const x = centerX + (R + r * Math.cos(theta + rotation)) * Math.cos(phi);
                    const y = centerY + (R + r * Math.cos(theta + rotation)) * Math.sin(phi);

                    if (j === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                }
                ctx.stroke();
            }

            // Add labels
            ctx.fillStyle = '#FFD700';
            ctx.font = '16px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('Torus S¹ × S¹', centerX, height - 30);
            ctx.fillText('Product of circles', centerX, height - 10);
        }

        drawTorus(ctx, width, height, this.rotationAngle);
        this.drawFunction = (rotation) => drawTorus(ctx, width, height, rotation);
    }

    renderProjectivePlane(vizContainer, explanationContainer) {
        vizContainer.innerHTML = `
            <div class="projective-manifold">
                <h4>Projective Plane: ℝℙ² Unity Space</h4>
                <div class="projective-canvas-container">
                    <canvas id="projective-canvas" width="500" height="400"></canvas>
                </div>
                <div class="projective-equation">
                    <div class="equation">ℝℙ²: Antipodal points identified</div>
                    <div class="sub-equation">Unity through point identification</div>
                    <div class="unity-equation">∴ 1 + 1 = 1 in projective geometry</div>
                </div>
            </div>
        `;

        explanationContainer.innerHTML = `
            <div class="manifold-explanation-content">
                <h4>Projective Plane: Geometric Unity</h4>
                <p>The projective plane ℝℙ² demonstrates unity through geometric identification:</p>
                <ul>
                    <li><strong>Antipodal identification</strong> - opposite points become one</li>
                    <li><strong>Non-orientable</strong> - no consistent orientation</li>
                    <li><strong>Single-sided</strong> - like the Möbius strip but in 2D</li>
                    <li><strong>Fundamental group ℤ₂</strong> - two-element group representing duality</li>
                </ul>
                <p>This shows how unity can be achieved by identifying seemingly opposite elements.</p>
                <div class="mathematical-note">
                    <strong>Geometric Property:</strong> The projective plane has Euler characteristic χ = 1, 
                    representing the unity of all points through identification.
                </div>
            </div>
        `;

        this.setupProjectiveCanvas();
    }

    setupProjectiveCanvas() {
        const canvas = document.getElementById('projective-canvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        function drawProjectivePlane(ctx, width, height, rotation = 0) {
            ctx.clearRect(0, 0, width, height);

            const centerX = width / 2;
            const centerY = height / 2;
            const radius = 150;

            ctx.strokeStyle = '#FFD700';
            ctx.lineWidth = 2;

            // Draw projective plane as hemisphere with antipodal identification
            ctx.beginPath();
            for (let i = 0; i <= 100; i++) {
                const theta = (i / 100) * Math.PI;
                const x = centerX + radius * Math.cos(theta + rotation);
                const y = centerY + radius * Math.sin(theta + rotation);

                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.stroke();

            // Draw identification lines
            ctx.strokeStyle = 'rgba(255, 215, 0, 0.6)';
            ctx.lineWidth = 1;

            for (let i = 0; i <= 8; i++) {
                const angle = (i / 8) * Math.PI;
                ctx.beginPath();
                ctx.moveTo(centerX, centerY);
                ctx.lineTo(
                    centerX + radius * Math.cos(angle + rotation),
                    centerY + radius * Math.sin(angle + rotation)
                );
                ctx.stroke();
            }

            // Add antipodal identification markers
            ctx.fillStyle = '#FFD700';
            ctx.beginPath();
            ctx.arc(centerX + radius * Math.cos(rotation), centerY + radius * Math.sin(rotation), 5, 0, 2 * Math.PI);
            ctx.fill();

            ctx.beginPath();
            ctx.arc(centerX + radius * Math.cos(rotation + Math.PI), centerY + radius * Math.sin(rotation + Math.PI), 5, 0, 2 * Math.PI);
            ctx.fill();

            // Add labels
            ctx.fillStyle = '#FFD700';
            ctx.font = '16px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('ℝℙ² Projective Plane', centerX, height - 30);
            ctx.fillText('Antipodal identification', centerX, height - 10);
        }

        drawProjectivePlane(ctx, width, height, this.rotationAngle);
        this.drawFunction = (rotation) => drawProjectivePlane(ctx, width, height, rotation);
    }

    renderHyperbolicSurface(vizContainer, explanationContainer) {
        vizContainer.innerHTML = `
            <div class="hyperbolic-manifold">
                <h4>Hyperbolic Surface: Negative Curvature Unity</h4>
                <div class="hyperbolic-canvas-container">
                    <canvas id="hyperbolic-canvas" width="500" height="400"></canvas>
                </div>
                <div class="hyperbolic-equation">
                    <div class="equation">Hyperbolic Geometry: Infinite possibilities</div>
                    <div class="sub-equation">Unity through infinite expansion</div>
                    <div class="unity-equation">∴ 1 + 1 = 1 in hyperbolic space</div>
                </div>
            </div>
        `;

        explanationContainer.innerHTML = `
            <div class="manifold-explanation-content">
                <h4>Hyperbolic Surface: Infinite Unity</h4>
                <p>Hyperbolic geometry demonstrates unity through infinite expansion:</p>
                <ul>
                    <li><strong>Negative curvature</strong> - opposite of spherical geometry</li>
                    <li><strong>Infinite area</strong> - despite finite boundary</li>
                    <li><strong>Exponential growth</strong> - area grows exponentially with radius</li>
                    <li><strong>Parallel postulate violation</strong> - multiple parallel lines through a point</li>
                </ul>
                <p>This represents unity through the infinite possibilities that emerge from 
                geometric constraints.</p>
                <div class="mathematical-note">
                    <strong>Geometric Property:</strong> Hyperbolic surfaces have constant negative curvature K = -1, 
                    representing the unity of infinite expansion.
                </div>
            </div>
        `;

        this.setupHyperbolicCanvas();
    }

    setupHyperbolicCanvas() {
        const canvas = document.getElementById('hyperbolic-canvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        function drawHyperbolicSurface(ctx, width, height, rotation = 0) {
            ctx.clearRect(0, 0, width, height);

            const centerX = width / 2;
            const centerY = height / 2;
            const maxRadius = 150;

            ctx.strokeStyle = '#FFD700';
            ctx.lineWidth = 1;

            // Draw hyperbolic circles (appear as circles in Poincaré disk model)
            for (let r = 20; r <= maxRadius; r += 20) {
                ctx.beginPath();
                ctx.arc(centerX, centerY, r, 0, 2 * Math.PI);
                ctx.stroke();
            }

            // Draw hyperbolic lines (appear as circular arcs perpendicular to boundary)
            ctx.strokeStyle = 'rgba(255, 215, 0, 0.6)';
            ctx.lineWidth = 2;

            for (let i = 0; i <= 8; i++) {
                const angle = (i / 8) * 2 * Math.PI + rotation;
                const radius = 100;

                ctx.beginPath();
                ctx.arc(
                    centerX + radius * Math.cos(angle),
                    centerY + radius * Math.sin(angle),
                    80,
                    0,
                    2 * Math.PI
                );
                ctx.stroke();
            }

            // Add boundary circle
            ctx.strokeStyle = '#FFD700';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(centerX, centerY, maxRadius, 0, 2 * Math.PI);
            ctx.stroke();

            // Add labels
            ctx.fillStyle = '#FFD700';
            ctx.font = '16px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('Hyperbolic Surface', centerX, height - 30);
            ctx.fillText('Negative curvature', centerX, height - 10);
        }

        drawHyperbolicSurface(ctx, width, height, this.rotationAngle);
        this.drawFunction = (rotation) => drawHyperbolicSurface(ctx, width, height, rotation);
    }

    toggleRotation() {
        if (this.isAnimating) {
            this.stopAnimation();
        } else {
            this.startAnimation();
        }
    }

    toggleAnimation() {
        // Additional animation effects
        console.log('Toggling manifold animation...');
    }

    startAnimation() {
        this.isAnimating = true;
        const rotateBtn = document.getElementById('rotate-manifold');
        if (rotateBtn) rotateBtn.textContent = 'Stop Rotation';

        this.animateManifold();
    }

    stopAnimation() {
        this.isAnimating = false;
        const rotateBtn = document.getElementById('rotate-manifold');
        if (rotateBtn) rotateBtn.textContent = 'Rotate';

        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
    }

    animateManifold() {
        if (!this.isAnimating) return;

        this.rotationAngle += 0.02;

        if (this.drawFunction) {
            this.drawFunction(this.rotationAngle);
        }

        this.animationFrame = requestAnimationFrame(() => {
            this.animateManifold();
        });
    }

    resetManifold() {
        this.stopAnimation();
        this.rotationAngle = 0;
        this.renderManifold(this.currentManifold);
    }
}

// Global function to create the visualizer
function createUnityManifoldsTopology(containerId) {
    return new UnityManifoldsVisualizer(containerId);
}