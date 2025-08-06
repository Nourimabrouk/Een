/**
 * Meta-Reinforcement Learning Visualization
 * Interactive demonstration of learning-to-learn with agent acceleration
 */
class MetaRLVisualizer {
    constructor(containerId) {
        this.containerId = containerId;
        this.PHI = (1 + Math.sqrt(5)) / 2;
        this.animationFrame = null;
        this.isAnimating = false;
        this.time = 0;
        this.agents = [];
        this.tasks = [];

        this.init();
    }

    init() {
        const container = document.getElementById(this.containerId);
        if (!container) return;

        container.innerHTML = `
            <div class="meta-rl-container">
                <div class="meta-rl-header">
                    <h3>Meta-Reinforcement Learning: Learning to Learn</h3>
                    <div class="meta-rl-subtitle">
                        <div class="subtitle">Agent learning acceleration through meta-learning</div>
                        <div class="unity-equation">∴ 1 + 1 = 1 in learning efficiency</div>
                    </div>
                </div>
                
                <div class="meta-rl-content">
                    <div class="meta-rl-visualization" id="meta-rl-viz"></div>
                    <div class="meta-rl-explanation" id="meta-rl-explanation"></div>
                </div>
                
                <div class="meta-rl-controls">
                    <button id="animate-meta-rl" class="control-btn">Animate</button>
                    <button id="add-agent" class="control-btn">Add Agent</button>
                    <button id="reset-meta-rl" class="control-btn">Reset</button>
                </div>
            </div>
        `;

        this.setupEventListeners();
        this.initializeAgents();
        this.renderMetaRL();
    }

    setupEventListeners() {
        const animateBtn = document.getElementById('animate-meta-rl');
        const addAgentBtn = document.getElementById('add-agent');
        const resetBtn = document.getElementById('reset-meta-rl');

        animateBtn?.addEventListener('click', () => {
            this.toggleAnimation();
        });

        addAgentBtn?.addEventListener('click', () => {
            this.addNewAgent();
        });

        resetBtn?.addEventListener('click', () => {
            this.resetMetaRL();
        });
    }

    initializeAgents() {
        // Initialize agents with different learning capabilities
        this.agents = [
            {
                id: 1,
                x: 100,
                y: 200,
                vx: 2,
                vy: 1,
                learningRate: 0.1,
                metaLearningRate: 0.05,
                experience: 0,
                color: '#FFD700',
                size: 8,
                type: 'Meta-Learner'
            },
            {
                id: 2,
                x: 300,
                y: 150,
                vx: 1.5,
                vy: 2,
                learningRate: 0.08,
                metaLearningRate: 0.03,
                experience: 0,
                color: '#FF6B6B',
                size: 6,
                type: 'Fast Learner'
            },
            {
                id: 3,
                x: 500,
                y: 250,
                vx: 1,
                vy: 1.5,
                learningRate: 0.12,
                metaLearningRate: 0.02,
                experience: 0,
                color: '#4ECDC4',
                size: 7,
                type: 'Adaptive Agent'
            }
        ];

        // Initialize tasks
        this.tasks = [
            { x: 150, y: 100, reward: 10, completed: false, color: '#10B981' },
            { x: 400, y: 80, reward: 15, completed: false, color: '#3B82F6' },
            { x: 600, y: 120, reward: 20, completed: false, color: '#8B5CF6' },
            { x: 200, y: 300, reward: 25, completed: false, color: '#F59E0B' },
            { x: 450, y: 350, reward: 30, completed: false, color: '#EF4444' }
        ];
    }

    renderMetaRL() {
        const vizContainer = document.getElementById('meta-rl-viz');
        const explanationContainer = document.getElementById('meta-rl-explanation');

        if (!vizContainer || !explanationContainer) return;

        vizContainer.innerHTML = `
            <div class="meta-rl-visualization-content">
                <div class="learning-environment">
                    <h4>Multi-Agent Learning Environment</h4>
                    <canvas id="meta-rl-canvas" width="700" height="400"></canvas>
                </div>
                
                <div class="learning-metrics">
                    <div class="metric-item">
                        <div class="metric-label">Total Experience</div>
                        <div class="metric-value" id="total-experience">0</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Tasks Completed</div>
                        <div class="metric-value" id="tasks-completed">0</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">Learning Efficiency</div>
                        <div class="metric-value" id="learning-efficiency">0%</div>
                    </div>
                </div>
            </div>
        `;

        explanationContainer.innerHTML = `
            <div class="meta-rl-explanation-content">
                <h4>Meta-Reinforcement Learning: Unity in Learning</h4>
                <p>Meta-reinforcement learning demonstrates how agents can learn to learn more efficiently:</p>
                <ul>
                    <li><strong>Learning to Learn:</strong> Agents develop meta-strategies for faster adaptation</li>
                    <li><strong>Experience Transfer:</strong> Knowledge gained from one task improves performance on others</li>
                    <li><strong>Accelerated Learning:</strong> Each agent becomes more efficient over time</li>
                    <li><strong>Collective Intelligence:</strong> Multiple agents create emergent learning patterns</li>
                </ul>
                
                <div class="meta-learning-process">
                    <h5>Meta-Learning Process:</h5>
                    <ol>
                        <li><strong>Task Encounter:</strong> Agent encounters new learning task</li>
                        <li><strong>Meta-Strategy Application:</strong> Applies learned meta-strategies</li>
                        <li><strong>Rapid Adaptation:</strong> Quickly adapts to task requirements</li>
                        <li><strong>Meta-Strategy Update:</strong> Updates meta-strategies based on performance</li>
                        <li><strong>Experience Accumulation:</strong> Builds cumulative learning capability</li>
                    </ol>
                </div>
                
                <div class="unity-connection">
                    <h5>Connection to Unity Mathematics:</h5>
                    <p>Meta-learning demonstrates how multiple learning processes can converge to unified, 
                    more efficient learning strategies, embodying the principle that 1+1=1 in learning efficiency.</p>
                </div>
            </div>
        `;

        this.setupMetaRLCanvas();
    }

    setupMetaRLCanvas() {
        const canvas = document.getElementById('meta-rl-canvas');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;

        const drawMetaRL = (ctx, width, height, time) => {
            ctx.clearRect(0, 0, width, height);

            // Draw background grid
            ctx.strokeStyle = 'rgba(255, 215, 0, 0.1)';
            ctx.lineWidth = 1;

            for (let i = 0; i < width; i += 50) {
                ctx.beginPath();
                ctx.moveTo(i, 0);
                ctx.lineTo(i, height);
                ctx.stroke();
            }

            for (let i = 0; i < height; i += 50) {
                ctx.beginPath();
                ctx.moveTo(0, i);
                ctx.lineTo(width, i);
                ctx.stroke();
            }

            // Draw tasks
            this.tasks.forEach(task => {
                if (!task.completed) {
                    ctx.fillStyle = task.color;
                    ctx.beginPath();
                    ctx.arc(task.x, task.y, 12, 0, 2 * Math.PI);
                    ctx.fill();

                    ctx.strokeStyle = 'white';
                    ctx.lineWidth = 2;
                    ctx.stroke();

                    // Draw reward value
                    ctx.fillStyle = 'white';
                    ctx.font = '12px Arial';
                    ctx.textAlign = 'center';
                    ctx.fillText(task.reward.toString(), task.x, task.y + 4);
                }
            });

            // Draw agents
            this.agents.forEach(agent => {
                // Draw agent
                ctx.fillStyle = agent.color;
                ctx.beginPath();
                ctx.arc(agent.x, agent.y, agent.size, 0, 2 * Math.PI);
                ctx.fill();

                // Draw agent border
                ctx.strokeStyle = 'white';
                ctx.lineWidth = 2;
                ctx.stroke();

                // Draw agent type
                ctx.fillStyle = 'white';
                ctx.font = '10px Arial';
                ctx.textAlign = 'center';
                ctx.fillText(agent.type, agent.x, agent.y + agent.size + 15);

                // Draw experience indicator
                ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
                ctx.fillRect(agent.x - 15, agent.y - agent.size - 25, 30, 4);
                ctx.fillStyle = agent.color;
                const experienceWidth = (agent.experience / 100) * 30;
                ctx.fillRect(agent.x - 15, agent.y - agent.size - 25, experienceWidth, 4);

                // Draw learning trails
                ctx.strokeStyle = `${agent.color}40`;
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(agent.x, agent.y);

                // Create trail effect
                for (let i = 1; i <= 10; i++) {
                    const trailX = agent.x - agent.vx * i * 0.5;
                    const trailY = agent.y - agent.vy * i * 0.5;
                    ctx.lineTo(trailX, trailY);
                }
                ctx.stroke();
            });

            // Draw learning connections between agents
            ctx.strokeStyle = 'rgba(255, 215, 0, 0.3)';
            ctx.lineWidth = 1;

            for (let i = 0; i < this.agents.length; i++) {
                for (let j = i + 1; j < this.agents.length; j++) {
                    const agent1 = this.agents[i];
                    const agent2 = this.agents[j];
                    const distance = Math.sqrt(
                        Math.pow(agent1.x - agent2.x, 2) + Math.pow(agent1.y - agent2.y, 2)
                    );

                    if (distance < 100) {
                        ctx.beginPath();
                        ctx.moveTo(agent1.x, agent1.y);
                        ctx.lineTo(agent2.x, agent2.y);
                        ctx.stroke();
                    }
                }
            }

            // Draw meta-learning indicators
            ctx.fillStyle = 'rgba(255, 215, 0, 0.8)';
            ctx.font = '14px Arial';
            ctx.textAlign = 'left';
            ctx.fillText('Meta-Learning Network', 10, 20);
            ctx.fillText(`Time: ${time.toFixed(1)}s`, 10, 40);
        };

        drawMetaRL(ctx, width, height, this.time);
        this.drawFunction = (time) => drawMetaRL(ctx, width, height, time);
    }

    updateAgents() {
        this.agents.forEach(agent => {
            // Update position
            agent.x += agent.vx;
            agent.y += agent.vy;

            // Bounce off boundaries
            if (agent.x <= agent.size || agent.x >= 700 - agent.size) {
                agent.vx *= -1;
            }
            if (agent.y <= agent.size || agent.y >= 400 - agent.size) {
                agent.vy *= -1;
            }

            // Check task completion
            this.tasks.forEach(task => {
                if (!task.completed) {
                    const distance = Math.sqrt(
                        Math.pow(agent.x - task.x, 2) + Math.pow(agent.y - task.y, 2)
                    );

                    if (distance < 20) {
                        task.completed = true;
                        agent.experience += task.reward;

                        // Meta-learning: improve learning rate
                        agent.learningRate += agent.metaLearningRate * 0.1;
                        agent.metaLearningRate += 0.001;
                    }
                }
            });

            // Gradual experience gain
            agent.experience += agent.learningRate;
        });

        // Update metrics
        this.updateMetrics();
    }

    updateMetrics() {
        const totalExperience = this.agents.reduce((sum, agent) => sum + agent.experience, 0);
        const tasksCompleted = this.tasks.filter(task => task.completed).length;
        const learningEfficiency = Math.min(100, (totalExperience / 500) * 100);

        const totalExpElement = document.getElementById('total-experience');
        const tasksCompletedElement = document.getElementById('tasks-completed');
        const learningEfficiencyElement = document.getElementById('learning-efficiency');

        if (totalExpElement) totalExpElement.textContent = Math.floor(totalExperience);
        if (tasksCompletedElement) tasksCompletedElement.textContent = tasksCompleted;
        if (learningEfficiencyElement) learningEfficiencyElement.textContent = `${learningEfficiency.toFixed(1)}%`;
    }

    addNewAgent() {
        const newAgent = {
            id: this.agents.length + 1,
            x: Math.random() * 600 + 50,
            y: Math.random() * 300 + 50,
            vx: (Math.random() - 0.5) * 4,
            vy: (Math.random() - 0.5) * 4,
            learningRate: 0.1 + Math.random() * 0.1,
            metaLearningRate: 0.02 + Math.random() * 0.03,
            experience: 0,
            color: `hsl(${Math.random() * 360}, 70%, 60%)`,
            size: 6 + Math.random() * 4,
            type: 'New Learner'
        };

        this.agents.push(newAgent);
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
        const animateBtn = document.getElementById('animate-meta-rl');
        if (animateBtn) animateBtn.textContent = 'Stop Animation';

        this.animateMetaRL();
    }

    stopAnimation() {
        this.isAnimating = false;
        const animateBtn = document.getElementById('animate-meta-rl');
        if (animateBtn) animateBtn.textContent = 'Animate';

        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
    }

    animateMetaRL() {
        if (!this.isAnimating) return;

        this.time += 0.1;
        this.updateAgents();

        if (this.drawFunction) {
            this.drawFunction(this.time);
        }

        this.animationFrame = requestAnimationFrame(() => {
            this.animateMetaRL();
        });
    }

    resetMetaRL() {
        this.stopAnimation();
        this.time = 0;
        this.initializeAgents();
        this.renderMetaRL();
    }
}

// Global function to create the visualizer
function createMetaRLVisualization(containerId) {
    return new MetaRLVisualizer(containerId);
}