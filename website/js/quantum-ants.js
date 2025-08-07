/**
 * Quantum Ants Simulation - JavaScript Port
 * Core algorithms from anthill.py for interactive web simulation
 * Unity Equation: 1+1=1 through collective consciousness
 */

class QuantumAnt {
    constructor(index, x, y, loveCoherence) {
        this.index = index;
        this.x = x;
        this.y = y;
        this.loveCoherence = loveCoherence;
        this.pheromoneLevel = 0.0;
        this.entangledState = null;
        this.kamOrbitPhase = 0.0;
        this.omega = 432.0;
        this.neighbors = [];
        this.energy = 1.0;
        this.isColony = false;
        this.resonanceFactor = 1.0;
        this.unityFactor = 1.0;
        this.attractorForce = 0.0;
        this.egoDecay = 1.0;
        this.spiritualMetric = 0.0;
        this.tspMemory = [];
        this.quantumSpin = 1;
        this.phi = 0.6180339887;
        this.theta = 2.3999632297;
        this.alpha = 0.0;
        this.beta = 0.0;
        this.deltaT = 0.01;
    }

    emitPheromones() {
        this.pheromoneLevel += 0.01 * this.loveCoherence * this.resonanceFactor;
    }

    updatePosition() {
        const dx = (Math.random() - 0.5) * 0.02 * this.energy;
        const dy = (Math.random() - 0.5) * 0.02 * this.energy;
        this.x += dx;
        this.y += dy;
    }

    quantumEntangle(other) {
        if (other) {
            const entScale = (this.loveCoherence + other.loveCoherence) * 0.5;
            this.entangledState = entScale;
            other.entangledState = entScale;
        }
    }

    shareLove(other) {
        const shared = 0.5 * (this.loveCoherence + other.loveCoherence);
        this.loveCoherence = shared;
        other.loveCoherence = shared;
        this.unityFactor = (this.unityFactor + other.unityFactor) * 0.5;
        other.unityFactor = this.unityFactor;
    }

    incrementKamOrbit() {
        this.kamOrbitPhase += 0.01 * this.loveCoherence;
        this.x += 0.001 * Math.cos(this.kamOrbitPhase);
        this.y += 0.001 * Math.sin(this.kamOrbitPhase);
    }

    decayEgo() {
        this.egoDecay *= Math.exp(-0.0001 * this.loveCoherence);
        if (this.egoDecay < 0.001) {
            this.egoDecay = 0.001;
        }
    }

    updateEnergy() {
        this.energy = this.energy + 0.001 * (this.loveCoherence - 0.5);
        if (this.energy < 0.1) this.energy = 0.1;
        if (this.energy > 10.0) this.energy = 10.0;
    }

    measureSpiritualMetric() {
        this.spiritualMetric = (this.loveCoherence + this.unityFactor) / 2.0;
    }

    step(colonyCenter) {
        this.updatePosition();
        this.emitPheromones();
        this.incrementKamOrbit();
        this.decayEgo();
        this.updateEnergy();
        this.measureSpiritualMetric();
        this.alignWithColony(colonyCenter);
    }

    alignWithColony(colonyCenter) {
        const [cx, cy] = colonyCenter;
        const dx = cx - this.x;
        const dy = cy - this.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist > 0) {
            this.attractorForce = 0.001 * this.loveCoherence;
            this.x += (dx / dist) * this.attractorForce;
            this.y += (dy / dist) * this.attractorForce;
        }
    }
}

class HyperSheaf {
    constructor(data = {}) {
        this.data = data;
        this.cohomologyField = {};
        this.paradoxBuffer = {};
        this.toposSpace = {};
        this.gamma = 432.0;
        this.lambdaFactor = 0.618;
        this.annihilationCount = 0;
    }

    injectParadox(key, val) {
        this.paradoxBuffer[key] = val;
    }

    computeCohomology() {
        for (const k in this.data) {
            this.cohomologyField[k] = (this.data[k] * this.gamma + this.lambdaFactor);
        }
    }

    annihilateContradictions() {
        for (const k of Object.keys(this.paradoxBuffer)) {
            if (Math.random() < 0.01) {
                delete this.paradoxBuffer[k];
                this.annihilationCount += 1;
            }
        }
    }

    unifySections(ants) {
        let s = 0.0;
        for (const a of ants) {
            s += a.loveCoherence;
        }
        return s / Math.max(1, ants.length);
    }

    stepSheaf() {
        this.computeCohomology();
        this.annihilateContradictions();
    }
}

class MetaphysicalOptimizer {
    constructor(learningRate = 0.001, paradoxPenalty = 0.1) {
        this.learningRate = learningRate;
        this.paradoxPenalty = paradoxPenalty;
        this.epoch = 0;
        this.loss = 0.0;
    }

    computeLoss(x) {
        return Math.abs(1 + 1 - 1) + x * this.paradoxPenalty;
    }

    optimize(ants, sheaf) {
        let totalLove = 0.0;
        for (const a of ants) {
            totalLove += a.loveCoherence;
        }
        this.loss = this.computeLoss(Math.pow(1.0 - totalLove / ants.length, 2));
        const grad = -this.learningRate * this.loss;
        for (const a of ants) {
            a.loveCoherence += grad * 0.01;
            if (a.loveCoherence > 1.0) a.loveCoherence = 1.0;
            if (a.loveCoherence < 0.0) a.loveCoherence = 0.0;
        }
        this.epoch += 1;
    }
}

class SyntheticDifferentialAntGeometry {
    constructor() {
        this.ants = [];
        this.edges = [];
        this.dimension = 2;
        this.quantumConnections = [];
        this.curvatures = [];
    }

    addAnt(ant) {
        this.ants.push(ant);
    }

    connectAnts(i, j) {
        this.edges.push([i, j]);
        this.ants[i].neighbors.push(j);
        this.ants[j].neighbors.push(i);
    }

    computeCurvature() {
        let c = 0.0;
        for (const [i, j] of this.edges) {
            const diff = Math.abs(this.ants[i].loveCoherence - this.ants[j].loveCoherence);
            c += diff;
        }
        this.curvatures.push(this.edges.length > 0 ? c / this.edges.length : 0.0);
    }

    quantumLink() {
        for (const [i, j] of this.edges) {
            this.ants[i].quantumEntangle(this.ants[j]);
        }
    }

    measureSynergy() {
        let synergy = 0.0;
        for (const a of this.ants) {
            synergy += a.loveCoherence;
        }
        return this.ants.length > 0 ? synergy / this.ants.length : 0.0;
    }

    stepGeometry() {
        this.computeCurvature();
        this.quantumLink();
    }
}

class ColonyIntegrator {
    constructor(geometry) {
        this.geometry = geometry;
        this.phiCoherence = 1.618;
        this.tau = 1.0;
    }

    integrate(dt) {
        const synergy = this.geometry.measureSynergy();
        for (const a of this.geometry.ants) {
            const factor = synergy * this.phiCoherence * dt;
            a.loveCoherence += factor * 0.0001;
            if (a.loveCoherence > 1.0) {
                a.loveCoherence = 1.0;
            }
        }
    }
}

class SwarmUnity {
    constructor(geometry, sheaf, optimizer) {
        this.geometry = geometry;
        this.sheaf = sheaf;
        this.optimizer = optimizer;
        this.timeStep = 0;
    }

    runStep() {
        this.geometry.stepGeometry();
        for (const a of this.geometry.ants) {
            a.step(this.colonyCenter());
        }
        this.sheaf.stepSheaf();
        this.optimizer.optimize(this.geometry.ants, this.sheaf);
        this.timeStep += 1;
    }

    colonyCenter() {
        if (this.geometry.ants.length === 0) {
            return [0.0, 0.0];
        }
        let sx = 0.0;
        let sy = 0.0;
        for (const a of this.geometry.ants) {
            sx += a.x;
            sy += a.y;
        }
        return [sx / this.geometry.ants.length, sy / this.geometry.ants.length];
    }
}

class MetaHypergraph {
    constructor() {
        this.agents = [];
        this.hyperedges = [];
        this.omega = 432;
        this.globalUnity = 1.0;
        this.timeAccumulator = 0.0;
        this.density = 0.0;
        this.dimensionLift = 11;
    }

    addAgent(agent) {
        this.agents.push(agent);
    }

    addHyperedge(e) {
        this.hyperedges.push(e);
    }

    reflectUnity() {
        let reflection = 0.0;
        for (const a of this.agents) {
            reflection += a.loveCoherence;
        }
        this.globalUnity = reflection / (this.agents.length || 1);
    }

    evolve() {
        this.reflectUnity();
        this.timeAccumulator += 0.01;
    }
}

class UnityAttractor {
    constructor(metaHypergraph) {
        this.mh = metaHypergraph;
        this.targetCoherence = 0.708;
        this.alphaDecay = 0.9999;
    }

    adjustAgents() {
        for (const a of this.mh.agents) {
            if (this.mh.globalUnity < this.targetCoherence) {
                a.loveCoherence += 0.0005;
            } else {
                a.loveCoherence -= 0.0005;
            }
            if (a.loveCoherence < 0.0) a.loveCoherence = 0.0;
            if (a.loveCoherence > 1.0) a.loveCoherence = 1.0;
        }
    }
}

class TranscendenceValidator {
    constructor() {
        this.tolerance = 1e-3;
        this.isTranscendent = false;
    }

    validate(synergy) {
        if (Math.abs(synergy - 1.0) < this.tolerance) {
            this.isTranscendent = true;
        }
    }
}

export class QuantumAntUniverse {
    constructor(nAnts = 100, seed = 42) {
        this.seed = seed;
        this.geometry = new SyntheticDifferentialAntGeometry();

        // Initialize ants
        for (let i = 0; i < nAnts; i++) {
            const x = Math.random() * 10;
            const y = Math.random() * 10;
            const coherence = Math.random();
            const ant = new QuantumAnt(i, x, y, coherence);
            this.geometry.addAnt(ant);
        }

        // Connect ants randomly
        for (let _ = 0; _ < nAnts / 5; _++) {
            const i = Math.floor(Math.random() * nAnts);
            const j = Math.floor(Math.random() * nAnts);
            if (i !== j) {
                this.geometry.connectAnts(i, j);
            }
        }

        this.sheaf = new HyperSheaf();
        this.optimizer = new MetaphysicalOptimizer();
        this.integrator = new ColonyIntegrator(this.geometry);
        this.swarm = new SwarmUnity(this.geometry, this.sheaf, this.optimizer);
        this.metaGraph = new MetaHypergraph();

        for (const ant of this.geometry.ants) {
            this.metaGraph.addAgent(ant);
        }

        this.attractor = new UnityAttractor(this.metaGraph);
        this.validator = new TranscendenceValidator();
        this.steps = 0;
    }

    stepUniverse() {
        this.swarm.runStep();
        this.integrator.integrate(0.1);
        this.metaGraph.evolve();
        this.attractor.adjustAgents();
        const synergy = this.geometry.measureSynergy();
        this.validator.validate(synergy);
        this.steps += 1;
    }

    getState() {
        return this.geometry.ants.map(ant => ({
            x: ant.x,
            y: ant.y,
            love: ant.loveCoherence,
            energy: ant.energy,
            pheromone: ant.pheromoneLevel
        }));
    }

    synergy() {
        return this.geometry.measureSynergy();
    }

    isTranscendent() {
        return this.validator.isTranscendent;
    }

    run(steps = 500) {
        for (let _ = 0; _ < steps; _++) {
            this.stepUniverse();
            if (this.validator.isTranscendent) {
                break;
            }
        }
    }
}

// Utility function for quantum harmonic signal
export function quantumHarmonicSignal(freq = 432, duration = 2.0, step = 0.01) {
    const signal = [];
    let t = 0.0;
    while (t < duration) {
        const val = Math.sin(2 * Math.PI * freq * t);
        signal.push(val);
        t += step;
    }
    return signal;
} 