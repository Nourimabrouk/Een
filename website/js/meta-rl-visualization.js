/**
 * Meta-Reinforcement Learning Visualization
 * Interactive demonstration of learning-to-learn with agent acceleration
 */

class MetaRLVisualizer {
    constructor(containerId) {
        this.containerId = containerId;
        this.PHI = (1 + Math.sqrt(5)) / 2;
        this.animationFrame = null;
        this.isAnimating = true;
        this.startTime = Date.now();
        
        // Configuration
        this.config = {
            num_tasks: 5,
            episodes_per_task: 100,
            meta_learning_rate: 0.01,
            adaptation_steps: 10,
            animation_speed: 1.0,
            show_performance: true,
            show_adaptation: true
        };
        
        // Agent learning data
        this.learningData = {
            tasks: [],
            metaPerformance: [],
            adaptationCurves: [],
            unityMetrics: []
        };
        
        this.init();
    }
    
    init() {
        this.generateLearningData();
        this.createContainer();
        this.createVisualization();
        this.startAnimation();
    }
    
    generateLearningData() {
        // Generate meta-RL learning curves
        for (let task = 0; task < this.config.num_tasks; task++) {
            const taskData = {
                id: task,
                name: `Task ${task + 1}`,
                baseReward: 0.2 + Math.random() * 0.3,
                adaptationRate: 0.1 + (task * 0.02), // Learning to learn
                performance: []
            };
            
            // Generate performance curve for this task
            for (let episode = 0; episode < this.config.episodes_per_task; episode++) {
                const metaBoost = task * 0.05; // Meta-learning improvement
                const progress = episode / this.config.episodes_per_task;
                const learningCurve = taskData.baseReward + 
                                    (0.6 * (1 - Math.exp(-taskData.adaptationRate * episode))) + 
                                    metaBoost * (1 - Math.exp(-progress * 5));
                
                // Add some noise
                const noise = (Math.random() - 0.5) * 0.05;
                taskData.performance.push(Math.max(0, Math.min(1, learningCurve + noise)));
            }
            
            this.learningData.tasks.push(taskData);
            
            // Calculate final performance for meta-learning curve
            const finalPerf = taskData.performance.slice(-10).reduce((a, b) => a + b) / 10;
            this.learningData.metaPerformance.push(finalPerf);
            
            // Calculate few-shot adaptation (first 10 episodes performance)
            const fewShotPerf = taskData.performance.slice(0, 10).reduce((a, b) => a + b) / 10;
            this.learningData.adaptationCurves.push(fewShotPerf);
            
            // Unity metric: how quickly tasks converge to similar performance
            const convergenceRate = this.calculateConvergenceRate(taskData.performance);
            this.learningData.unityMetrics.push(convergenceRate);
        }
    }
    
    calculateConvergenceRate(performance) {
        // Calculate how quickly performance stabilizes (unity metric)
        let convergenceEpisode = performance.length;
        const finalPerf = performance.slice(-5).reduce((a, b) => a + b) / 5;
        
        for (let i = 10; i < performance.length - 5; i++) {
            const currentPerf = performance.slice(i, i + 5).reduce((a, b) => a + b) / 5;
            if (Math.abs(currentPerf - finalPerf) < 0.05) {
                convergenceEpisode = i;
                break;
            }
        }
        
        return 1 - (convergenceEpisode / performance.length); // Higher = faster convergence
    }
    
    createContainer() {
        const container = document.getElementById(this.containerId);
        if (!container) {
            console.error(`Container ${this.containerId} not found`);
            return;
        }
        
        container.innerHTML = `
            <div class="meta-rl-container" style="
                background: linear-gradient(135deg, rgba(99, 102, 241, 0.03) 0%, rgba(16, 185, 129, 0.05) 100%);
                border-radius: 1rem;
                padding: 1.5rem;
                box-shadow: 0 8px 32px rgba(99, 102, 241, 0.2);
                margin: 1rem 0;
                position: relative;
                overflow: hidden;
            ">
                <!-- Animated AI background -->
                <div style="
                    position: absolute;
                    top: -20%;
                    left: -20%;
                    width: 140%;
                    height: 140%;
                    background: 
                        radial-gradient(circle at 20% 20%, rgba(99, 102, 241, 0.1) 0%, transparent 50%),
                        radial-gradient(circle at 80% 40%, rgba(16, 185, 129, 0.1) 0%, transparent 50%),
                        radial-gradient(circle at 40% 80%, rgba(245, 158, 11, 0.1) 0%, transparent 50%);
                    animation: meta-learning-flow 10s ease-in-out infinite;
                    z-index: 0;
                "></div>
                
                <div class="header" style="
                    text-align: center;
                    margin-bottom: 1.5rem;
                    position: relative;
                    z-index: 1;
                ">
                    <h3 style="
                        color: #6366F1;
                        margin: 0 0 0.5rem 0;
                        font-size: 1.8rem;
                        background: linear-gradient(135deg, #6366F1, #10B981);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        background-clip: text;
                    ">🤖 Meta-Reinforcement Learning</h3>
                    <p style="
                        color: #6B7280;
                        margin: 0 0 1rem 0;
                        font-size: 1rem;
                    ">Learning to Learn: AI Agent Acceleration Through Unity</p>
                    <div style="
                        font-family: 'Times New Roman', serif;
                        font-size: 1.1rem;
                        color: #6366F1;
                        background: rgba(99, 102, 241, 0.1);
                        padding: 0.5rem 1rem;
                        border-radius: 0.5rem;
                        display: inline-block;
                        border: 1px solid rgba(99, 102, 241, 0.3);
                    ">
                        Meta-Learning: Few-Shot Task Adaptation
                    </div>
                </div>
                
                <div class="controls-panel" style="
                    display: flex;
                    justify-content: center;
                    gap: 1rem;
                    flex-wrap: wrap;
                    margin-bottom: 1.5rem;
                    padding: 1rem;
                    background: rgba(99, 102, 241, 0.05);
                    border-radius: 0.75rem;
                    border: 1px solid rgba(99, 102, 241, 0.2);
                    position: relative;
                    z-index: 1;
                ">
                    <select id="demo-mode-${this.containerId}" style="
                        padding: 0.5rem 1rem;
                        border: 1px solid #D1D5DB;
                        border-radius: 0.5rem;
                        background: white;
                        font-size: 0.9rem;
                    ">
                        <option value="performance">Performance Curves</option>
                        <option value="adaptation">Few-Shot Adaptation</option>
                        <option value="meta_progress">Meta-Learning Progress</option>
                        <option value="unity_metrics">Unity Convergence</option>
                    </select>
                    
                    <button id="animate-btn-${this.containerId}" style="
                        padding: 0.5rem 1rem;
                        background: #DC2626;
                        color: white;
                        border: none;
                        border-radius: 0.5rem;
                        cursor: pointer;
                        font-size: 0.9rem;
                        transition: all 0.3s ease;
                    ">⏸️ Pause</button>
                    
                    <label style="
                        display: flex;
                        align-items: center;
                        gap: 0.5rem;
                        font-size: 0.9rem;
                        color: #374151;
                    ">
                        Tasks:
                        <input type="range" id="num-tasks-${this.containerId}" 
                               min="3" max="10" step="1" value="5"
                               style="width: 80px;">
                        <span id="tasks-value-${this.containerId}">5</span>
                    </label>
                    
                    <label style="
                        display: flex;
                        align-items: center;
                        gap: 0.5rem;
                        font-size: 0.9rem;
                        color: #374151;
                    ">
                        Speed:
                        <input type="range" id="speed-${this.containerId}" 
                               min="0.5" max="3" step="0.1" value="1"
                               style="width: 80px;">
                        <span id="speed-value-${this.containerId}">1.0</span>
                    </label>
                    
                    <button id="reset-btn-${this.containerId}" style="
                        padding: 0.5rem 1rem;
                        background: #10B981;
                        color: white;
                        border: none;
                        border-radius: 0.5rem;
                        cursor: pointer;
                        font-size: 0.9rem;
                        transition: all 0.3s ease;
                    ">🔄 Reset</button>
                </div>
                
                <div id="plot-${this.containerId}" style="
                    width: 100%;
                    height: 500px;
                    background: white;
                    border-radius: 0.75rem;
                    box-shadow: inset 0 2px 4px rgba(0,0,0,0.06);
                    position: relative;
                    z-index: 1;
                "></div>
                
                <div class="meta-rl-stats" style="
                    margin-top: 1rem;
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 1rem;
                    position: relative;
                    z-index: 1;
                ">
                    <div style="
                        background: rgba(99, 102, 241, 0.05);
                        padding: 0.75rem;
                        border-radius: 0.5rem;
                        text-align: center;
                        border: 1px solid rgba(99, 102, 241, 0.2);
                    ">
                        <div id="avg-performance-${this.containerId}" style="
                            font-size: 1.2rem;
                            font-weight: bold;
                            color: #6366F1;
                        ">0.000</div>
                        <div style="font-size: 0.8rem; color: #6B7280;">Avg Performance</div>
                    </div>
                    <div style="
                        background: rgba(16, 185, 129, 0.05);
                        padding: 0.75rem;
                        border-radius: 0.5rem;
                        text-align: center;
                        border: 1px solid rgba(16, 185, 129, 0.2);
                    ">
                        <div id="adaptation-speed-${this.containerId}" style="
                            font-size: 1.2rem;
                            font-weight: bold;
                            color: #10B981;
                        ">0.000</div>
                        <div style="font-size: 0.8rem; color: #6B7280;">Adaptation Speed</div>
                    </div>
                    <div style="
                        background: rgba(245, 158, 11, 0.05);
                        padding: 0.75rem;
                        border-radius: 0.5rem;
                        text-align: center;
                        border: 1px solid rgba(245, 158, 11, 0.2);
                    ">
                        <div id="meta-efficiency-${this.containerId}" style="
                            font-size: 1.2rem;
                            font-weight: bold;
                            color: #F59E0B;
                        ">0.000</div>
                        <div style="font-size: 0.8rem; color: #6B7280;">Meta-Efficiency</div>
                    </div>
                    <div style="
                        background: rgba(220, 38, 38, 0.05);
                        padding: 0.75rem;
                        border-radius: 0.5rem;
                        text-align: center;
                        border: 1px solid rgba(220, 38, 38, 0.2);
                    ">
                        <div id="unity-convergence-${this.containerId}" style="
                            font-size: 1.2rem;
                            font-weight: bold;
                            color: #DC2626;
                        ">0.000</div>
                        <div style="font-size: 0.8rem; color: #6B7280;">Unity Score</div>
                    </div>
                </div>
            </div>
            
            <style>
                @keyframes meta-learning-flow {
                    0%, 100% { transform: translate(0, 0) rotate(0deg); opacity: 0.3; }
                    33% { transform: translate(10px, -5px) rotate(120deg); opacity: 0.1; }
                    66% { transform: translate(-5px, 10px) rotate(240deg); opacity: 0.2; }
                }
            </style>
        `;
        
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        const demoModeSelect = document.getElementById(`demo-mode-${this.containerId}`);
        const animateBtn = document.getElementById(`animate-btn-${this.containerId}`);
        const numTasksSlider = document.getElementById(`num-tasks-${this.containerId}`);
        const speedSlider = document.getElementById(`speed-${this.containerId}`);
        const resetBtn = document.getElementById(`reset-btn-${this.containerId}`);
        
        const tasksValue = document.getElementById(`tasks-value-${this.containerId}`);
        const speedValue = document.getElementById(`speed-value-${this.containerId}`);
        
        if (demoModeSelect) {
            demoModeSelect.addEventListener('change', (e) => {
                this.config.demo_mode = e.target.value;
                this.updateVisualization();
            });
        }
        
        if (animateBtn) {
            animateBtn.addEventListener('click', () => {
                if (this.isAnimating) {
                    this.stopAnimation();
                    animateBtn.textContent = '🎬 Play';
                    animateBtn.style.background = '#059669';
                } else {
                    this.startAnimation();
                    animateBtn.textContent = '⏸️ Pause';
                    animateBtn.style.background = '#DC2626';
                }
            });
        }
        
        if (numTasksSlider && tasksValue) {
            numTasksSlider.addEventListener('input', (e) => {
                this.config.num_tasks = parseInt(e.target.value);
                tasksValue.textContent = this.config.num_tasks;
                this.generateLearningData();
                this.updateVisualization();
            });
        }
        
        if (speedSlider && speedValue) {
            speedSlider.addEventListener('input', (e) => {
                this.config.animation_speed = parseFloat(e.target.value);
                speedValue.textContent = this.config.animation_speed.toFixed(1);
            });
        }
        
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                this.generateLearningData();
                this.updateVisualization();
            });
        }
    }
    
    createVisualization() {
        this.config.demo_mode = 'performance';
        this.updateVisualization();
    }
    
    updateVisualization() {
        const plotDiv = document.getElementById(`plot-${this.containerId}`);
        if (!plotDiv) return;
        
        const currentTime = this.isAnimating ? (Date.now() - this.startTime) * 0.001 * this.config.animation_speed : 0;
        const animationProgress = (currentTime % 10) / 10; // 10-second cycle
        
        let data, layout;
        
        switch (this.config.demo_mode) {
            case 'performance':
                ({ data, layout } = this.createPerformanceCurves(animationProgress));
                break;
            case 'adaptation':
                ({ data, layout } = this.createAdaptationCurves(animationProgress));
                break;
            case 'meta_progress':
                ({ data, layout } = this.createMetaProgressCurves(animationProgress));
                break;
            case 'unity_metrics':
                ({ data, layout } = this.createUnityMetrics(animationProgress));
                break;
            default:
                ({ data, layout } = this.createPerformanceCurves(animationProgress));
        }
        
        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d'],
            displaylogo: false
        };
        
        Plotly.newPlot(plotDiv, data, layout, config);
        this.updateStats();
    }
    
    createPerformanceCurves(animationProgress) {
        const data = [];
        const colors = ['#6366F1', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];
        
        // Animated reveal of performance curves
        const maxEpisode = Math.floor(this.config.episodes_per_task * animationProgress);
        
        for (let i = 0; i < this.learningData.tasks.length; i++) {
            const task = this.learningData.tasks[i];
            const episodes = Array.from({length: maxEpisode}, (_, idx) => idx + 1);
            const performance = task.performance.slice(0, maxEpisode);
            
            data.push({
                x: episodes,
                y: performance,
                type: 'scatter',
                mode: 'lines+markers',
                name: task.name,
                line: { 
                    color: colors[i % colors.length], 
                    width: 3,
                    shape: 'spline'
                },
                marker: { size: 4 }
            });
            
            // Add trend line showing meta-learning improvement
            if (i > 0 && maxEpisode > 10) {
                const trendStart = task.performance[0];
                const trendEnd = task.performance[Math.min(maxEpisode - 1, task.performance.length - 1)];
                const improvement = (trendEnd - trendStart) * (i + 1) * 0.1; // Meta-boost
                
                data.push({
                    x: [1, maxEpisode],
                    y: [trendStart + improvement * 0.5, trendEnd + improvement],
                    type: 'scatter',
                    mode: 'lines',
                    name: `${task.name} Meta-Trend`,
                    line: { 
                        color: colors[i % colors.length], 
                        width: 2,
                        dash: 'dash'
                    },
                    opacity: 0.6,
                    showlegend: false
                });
            }
        }
        
        const layout = {
            title: 'Meta-RL Performance: Learning Acceleration Across Tasks',
            xaxis: { 
                title: 'Episodes',
                range: [0, this.config.episodes_per_task]
            },
            yaxis: { 
                title: 'Performance (Reward)',
                range: [0, 1]
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'white',
            margin: { l: 60, r: 50, t: 80, b: 50 },
            annotations: [{
                x: this.config.episodes_per_task * 0.7,
                y: 0.9,
                text: 'Later tasks learn faster<br>(Meta-Learning Effect)',
                showarrow: true,
                arrowhead: 2,
                font: { size: 12, color: '#6366F1' }
            }]
        };
        
        return { data, layout };
    }
    
    createAdaptationCurves(animationProgress) {
        const data = [];
        const colors = ['#6366F1', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];
        
        // Show first 20 episodes (few-shot adaptation)
        const maxEpisode = Math.min(20, Math.floor(20 * animationProgress));
        
        for (let i = 0; i < this.learningData.tasks.length; i++) {
            const task = this.learningData.tasks[i];
            const episodes = Array.from({length: maxEpisode}, (_, idx) => idx + 1);
            const performance = task.performance.slice(0, maxEpisode);
            
            // Add meta-learning boost to later tasks
            const metaBoost = i * 0.1;
            const boostedPerformance = performance.map(p => Math.min(1, p + metaBoost));
            
            data.push({
                x: episodes,
                y: boostedPerformance,
                type: 'scatter',
                mode: 'lines+markers',
                name: task.name,
                line: { 
                    color: colors[i % colors.length], 
                    width: 4,
                    shape: 'spline'
                },
                marker: { 
                    size: 6,
                    symbol: i === 0 ? 'circle' : 'diamond'
                }
            });
        }
        
        // Add unity reference line
        data.push({
            x: [1, 20],
            y: [1/this.PHI, 1/this.PHI], // φ-harmonic target
            type: 'scatter',
            mode: 'lines',
            name: 'φ-Harmonic Target',
            line: { 
                color: '#B8860B', 
                width: 2,
                dash: 'dot'
            }
        });
        
        const layout = {
            title: 'Few-Shot Adaptation: Meta-Learning Acceleration',
            xaxis: { 
                title: 'Episodes (Few-Shot)',
                range: [0, 20]
            },
            yaxis: { 
                title: 'Performance',
                range: [0, 1]
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'white',
            margin: { l: 60, r: 50, t: 80, b: 50 },
            annotations: [{
                x: 15,
                y: 0.2,
                text: 'Each new task adapts<br>faster than the previous',
                showarrow: true,
                arrowhead: 2,
                font: { size: 12, color: '#10B981' }
            }]
        };
        
        return { data, layout };
    }
    
    createMetaProgressCurves(animationProgress) {
        const data = [];
        
        // Meta-learning progress over tasks
        const maxTask = Math.floor(this.learningData.tasks.length * animationProgress);
        const taskNumbers = Array.from({length: maxTask}, (_, idx) => idx + 1);
        const metaPerformance = this.learningData.metaPerformance.slice(0, maxTask);
        const adaptationSpeed = this.learningData.adaptationCurves.slice(0, maxTask);
        
        // Final performance trend
        data.push({
            x: taskNumbers,
            y: metaPerformance,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Final Performance',
            line: { color: '#6366F1', width: 4 },
            marker: { size: 8, symbol: 'circle' }
        });
        
        // Few-shot adaptation trend
        data.push({
            x: taskNumbers,
            y: adaptationSpeed,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Few-Shot Performance',
            line: { color: '#10B981', width: 4 },
            marker: { size: 8, symbol: 'diamond' }
        });
        
        // Meta-learning efficiency (gap between curves)
        const efficiency = metaPerformance.map((final, i) => 
            adaptationSpeed[i] ? adaptationSpeed[i] / final : 0
        );
        
        data.push({
            x: taskNumbers,
            y: efficiency,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Meta-Efficiency',
            line: { color: '#F59E0B', width: 3 },
            marker: { size: 6 },
            yaxis: 'y2'
        });
        
        const layout = {
            title: 'Meta-Learning Progress: Improvement Across Tasks',
            xaxis: { 
                title: 'Task Number',
                range: [0, this.learningData.tasks.length + 0.5]
            },
            yaxis: { 
                title: 'Performance',
                range: [0, 1],
                side: 'left'
            },
            yaxis2: {
                title: 'Efficiency Ratio',
                range: [0, 1],
                side: 'right',
                overlaying: 'y'
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'white',
            margin: { l: 60, r: 80, t: 80, b: 50 }
        };
        
        return { data, layout };
    }
    
    createUnityMetrics(animationProgress) {
        const data = [];
        
        // Unity convergence metrics
        const maxTask = Math.floor(this.learningData.tasks.length * animationProgress);
        const taskNumbers = Array.from({length: maxTask}, (_, idx) => idx + 1);
        const unityMetrics = this.learningData.unityMetrics.slice(0, maxTask);
        
        // Convergence rate over tasks
        data.push({
            x: taskNumbers,
            y: unityMetrics,
            type: 'bar',
            name: 'Unity Convergence Rate',
            marker: { 
                color: unityMetrics.map(u => `rgba(99, 102, 241, ${0.3 + u * 0.7})`),
                line: { color: '#6366F1', width: 2 }
            }
        });
        
        // φ-harmonic unity target
        const phiTarget = 1 / this.PHI;
        data.push({
            x: [0, this.learningData.tasks.length + 1],
            y: [phiTarget, phiTarget],
            type: 'scatter',
            mode: 'lines',
            name: 'φ-Harmonic Unity Target',
            line: { 
                color: '#B8860B', 
                width: 3,
                dash: 'dash'
            }
        });
        
        // Moving average of unity
        if (unityMetrics.length > 2) {
            const movingAvg = [];
            for (let i = 1; i < unityMetrics.length; i++) {
                const avg = unityMetrics.slice(0, i + 1).reduce((a, b) => a + b) / (i + 1);
                movingAvg.push(avg);
            }
            
            data.push({
                x: taskNumbers.slice(1),
                y: movingAvg,
                type: 'scatter',
                mode: 'lines',
                name: 'Unity Trend',
                line: { 
                    color: '#10B981', 
                    width: 4,
                    shape: 'spline'
                }
            });
        }
        
        const layout = {
            title: 'Unity Convergence: Learning Harmony Across Tasks',
            xaxis: { 
                title: 'Task Number',
                range: [0, this.learningData.tasks.length + 0.5]
            },
            yaxis: { 
                title: 'Unity Score (Convergence Rate)',
                range: [0, 1]
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'white',
            margin: { l: 60, r: 50, t: 80, b: 50 },
            annotations: [{
                x: this.learningData.tasks.length * 0.7,
                y: 0.9,
                text: 'Higher scores indicate faster<br>convergence to unity performance',
                showarrow: true,
                arrowhead: 2,
                font: { size: 12, color: '#6366F1' }
            }]
        };
        
        return { data, layout };
    }
    
    updateStats() {
        // Calculate and update statistics
        const avgPerformance = this.learningData.metaPerformance.reduce((a, b) => a + b, 0) / this.learningData.metaPerformance.length;
        const avgAdaptation = this.learningData.adaptationCurves.reduce((a, b) => a + b, 0) / this.learningData.adaptationCurves.length;
        const metaEfficiency = avgAdaptation / avgPerformance;
        const unityScore = this.learningData.unityMetrics.reduce((a, b) => a + b, 0) / this.learningData.unityMetrics.length;
        
        const avgPerfEl = document.getElementById(`avg-performance-${this.containerId}`);
        const adaptSpeedEl = document.getElementById(`adaptation-speed-${this.containerId}`);
        const metaEffEl = document.getElementById(`meta-efficiency-${this.containerId}`);
        const unityEl = document.getElementById(`unity-convergence-${this.containerId}`);
        
        if (avgPerfEl) avgPerfEl.textContent = avgPerformance.toFixed(3);
        if (adaptSpeedEl) adaptSpeedEl.textContent = avgAdaptation.toFixed(3);
        if (metaEffEl) metaEffEl.textContent = metaEfficiency.toFixed(3);
        if (unityEl) unityEl.textContent = unityScore.toFixed(3);
    }
    
    startAnimation() {
        this.isAnimating = true;
        this.startTime = Date.now();
        this.animate();
    }
    
    stopAnimation() {
        this.isAnimating = false;
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
    }
    
    animate() {
        if (!this.isAnimating) return;
        
        this.updateVisualization();
        this.animationFrame = requestAnimationFrame(() => this.animate());
    }
}

// Gallery creation function
function createMetaRLVisualization(containerId) {
    return new MetaRLVisualizer(containerId);
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { MetaRLVisualizer, createMetaRLVisualization };
}

// Browser global
if (typeof window !== 'undefined') {
    window.MetaRLVisualizer = MetaRLVisualizer;
    window.createMetaRLVisualization = createMetaRLVisualization;
}