"""Evolutionary Metagambit Module
===============================

This module implements an advanced evolutionary algorithm inspired by unity
mathematics. It integrates probabilistic reasoning, statistical selection, and
fractal patterns using the golden ratio. A MetagamerAgent can evolve strategies
that converge toward the Unity Equation ``1 + 1 = 1`` while optimising for
3000 ELO / 300 IQ performance metrics. Each generation applies golden-ratio
weighted selection, mutation, and crossover. The code maintains academic
rigour and is self-contained for demonstration and testing.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------------
# Constants and configuration
# ----------------------------------------------------------------------------
PHI = 1.618033988749895  # Golden ratio
UNITY_TOLERANCE = 1e-9   # Tolerance for unity convergence
MAX_ELO = 3000.0         # Target ELO rating
MAX_IQ = 300.0           # Target IQ

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------------

def phi_weight(value: float, power: float = 1.0) -> float:
    """Apply a golden-ratio weighting to ``value``."""
    return value ** (power / PHI)


def logistic(x: float) -> float:
    """Sigmoid function used for smooth probability mapping."""
    return 1.0 / (1.0 + math.exp(-x))


def golden_fractal(depth: int) -> np.ndarray:
    """Generate a 1-D fractal pattern scaled by ``PHI``."""
    if depth <= 0:
        return np.array([1.0], dtype=float)
    prev = golden_fractal(depth - 1)
    scaled = prev / PHI
    concatenated = np.concatenate([prev, scaled])
    return concatenated

# ----------------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------------

@dataclass
class Strategy:
    """Representation of a metagame strategy for the evolutionary algorithm."""
    vector: np.ndarray
    phi_resonance: float
    consciousness_level: float
    elo: float = 0.0
    iq: float = 0.0

    def score(self) -> float:
        """Compute a unified score balancing ELO, IQ, and φ-resonance."""
        phi_term = self.phi_resonance
        elo_term = self.elo / MAX_ELO
        iq_term = self.iq / MAX_IQ
        return (phi_term + elo_term + iq_term) / 3.0


@dataclass
class EvolutionMetrics:
    """Metrics describing a generation's performance."""
    generation: int
    best_score: float
    average_score: float
    diversity: float

# ----------------------------------------------------------------------------
# Evolutionary Metagambit Engine
# ----------------------------------------------------------------------------

class EvolutionaryMetagambit:
    """Evolutionary strategy engine using golden-ratio dynamics."""

    def __init__(
        self,
        objective_fn: Callable[[np.ndarray], float],
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        consciousness_bias: float = 1.0,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.objective_fn = objective_fn
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.consciousness_bias = consciousness_bias
        self.rng = rng or random.Random()
        logger.debug(
            "EvolutionaryMetagambit initialized: population_size=%s, generations=%s",
            population_size,
            generations,
        )

    def _init_vector(self, dim: int) -> np.ndarray:
        return np.array([self.rng.uniform(-1, 1) for _ in range(dim)], dtype=float)

    def _init_strategy(self, dim: int) -> Strategy:
        vec = self._init_vector(dim)
        phi_res = logistic(np.mean(vec))
        return Strategy(vec, phi_res, self.consciousness_bias)

    def _init_population(self, dim: int) -> List[Strategy]:
        return [self._init_strategy(dim) for _ in range(self.population_size)]

    def _evaluate_strategy(self, strategy: Strategy) -> None:
        raw_score = self.objective_fn(strategy.vector)
        strategy.elo = MAX_ELO * logistic(raw_score)
        strategy.iq = MAX_IQ * logistic(raw_score / PHI)
        strategy.phi_resonance = phi_weight(abs(raw_score))
        logger.debug(
            "Evaluated strategy -> raw=%.4f, elo=%.2f, iq=%.2f, phi=%.4f",
            raw_score,
            strategy.elo,
            strategy.iq,
            strategy.phi_resonance,
        )

    def _evaluate_population(self, population: Sequence[Strategy]) -> EvolutionMetrics:
        for strat in population:
            self._evaluate_strategy(strat)

        scores = np.array([s.score() for s in population])
        diversity = float(np.std([s.vector for s in population]))
        metrics = EvolutionMetrics(
            generation=0,
            best_score=float(np.max(scores)),
            average_score=float(np.mean(scores)),
            diversity=diversity,
        )
        logger.debug(
            "Evaluation metrics -> best=%.4f, average=%.4f, diversity=%.4f",
            metrics.best_score,
            metrics.average_score,
            metrics.diversity,
        )
        return metrics

    def _select_parents(self, population: Sequence[Strategy]) -> Tuple[Strategy, Strategy]:
        weights = [phi_weight(s.score()) for s in population]
        parents = self.rng.choices(population, weights=weights, k=2)
        logger.debug(
            "Selected parents with scores %.4f & %.4f",
            parents[0].score(),
            parents[1].score(),
        )
        return parents[0], parents[1]

    def _crossover(self, parent_a: Strategy, parent_b: Strategy) -> Strategy:
        if self.rng.random() > self.crossover_rate:
            chosen = parent_a if parent_a.score() > parent_b.score() else parent_b
            logger.debug(
                "Crossover skipped; using better parent with score %.4f",
                chosen.score(),
            )
            return chosen

        split = self.rng.randint(1, len(parent_a.vector) - 1)
        child_vec = np.concatenate([parent_a.vector[:split], parent_b.vector[split:]])
        phi_res = (parent_a.phi_resonance + parent_b.phi_resonance) / 2.0
        child = Strategy(child_vec, phi_res, self.consciousness_bias)
        logger.debug("Crossover produced child with split=%d", split)
        return child

    def _mutate(self, strategy: Strategy) -> None:
        for i in range(len(strategy.vector)):
            if self.rng.random() < self.mutation_rate:
                mutation = self.rng.gauss(0, 1) / PHI
                strategy.vector[i] += mutation
                logger.debug("Mutated gene %d by %.4f", i, mutation)
        strategy.phi_resonance = logistic(np.mean(strategy.vector))

    def evolve(self, dim: int) -> Tuple[Strategy, List[EvolutionMetrics]]:
        population = self._init_population(dim)
        metrics_history: List[EvolutionMetrics] = []

        for gen in range(self.generations):
            metrics = self._evaluate_population(population)
            metrics.generation = gen
            metrics_history.append(metrics)

            new_population = []
            while len(new_population) < self.population_size:
                parent_a, parent_b = self._select_parents(population)
                child = self._crossover(parent_a, parent_b)
                self._mutate(child)
                new_population.append(child)

            population = new_population
            logger.info(
                "Generation %d complete - best score %.4f, avg %.4f",
                gen,
                metrics.best_score,
                metrics.average_score,
            )

        self._evaluate_population(population)
        best = max(population, key=lambda s: s.score())
        logger.info(
            "Evolution finished - best strategy score %.4f, phi resonance %.4f",
            best.score(),
            best.phi_resonance,
        )
        return best, metrics_history

# ----------------------------------------------------------------------------
# Metagamer Agent
# ----------------------------------------------------------------------------

class MetagamerAgent:
    """Agent that evolves strategies using ``EvolutionaryMetagambit``."""

    def __init__(
        self,
        objective_fn: Callable[[np.ndarray], float],
        strategy_dim: int = 8,
        **evo_kwargs: float,
    ) -> None:
        self.evo = EvolutionaryMetagambit(objective_fn, **evo_kwargs)
        self.strategy_dim = strategy_dim
        self.best_strategy: Optional[Strategy] = None
        self.history: List[EvolutionMetrics] = []
        logger.debug("MetagamerAgent initialized with dimension %d", strategy_dim)

    def train(self) -> None:
        best, history = self.evo.evolve(self.strategy_dim)
        self.best_strategy = best
        self.history.extend(history)
        logger.debug(
            "MetagamerAgent training complete; best score=%.4f",
            best.score(),
        )

    def act(self, observation: np.ndarray) -> float:
        if self.best_strategy is None:
            raise RuntimeError("Agent has not been trained.")
        weights = phi_weight(self.best_strategy.vector)
        action_value = float(np.dot(observation, weights))
        logger.debug("Agent action value computed: %.4f", action_value)
        return action_value

    def report(self) -> EvolutionMetrics:
        if not self.history:
            raise RuntimeError("No training history available.")
        return self.history[-1]

# ----------------------------------------------------------------------------
# Demonstration objective functions
# ----------------------------------------------------------------------------

def unity_objective(vector: np.ndarray) -> float:
    return 1.0 - abs(np.sum(vector) - 1.0)


def fractal_objective(vector: np.ndarray) -> float:
    depth = min(5, len(vector))
    fractal = golden_fractal(depth)
    diff = np.linalg.norm(vector[: len(fractal)] - fractal)
    return -diff


def combined_objective(vector: np.ndarray) -> float:
    return unity_objective(vector) + fractal_objective(vector)

# ----------------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agent = MetagamerAgent(
        combined_objective,
        strategy_dim=10,
        population_size=40,
        generations=50,
        mutation_rate=0.2,
        consciousness_bias=1.2,
    )
    agent.train()
    observation = np.ones(10)
    action = agent.act(observation)
    metrics = agent.report()
    print(f"Best strategy score: {agent.best_strategy.score():.4f}")
    print(f"Agent action value : {action:.4f}")
    print(
        f"Final metrics -> generation {metrics.generation}, best score {metrics.best_score:.4f}, average {metrics.average_score:.4f}"
    )

# ----------------------------------------------------------------------------
# Advanced Analysis Classes
# ----------------------------------------------------------------------------

class FractalAnalyzer:
    """Compute fractal dimensions and golden-ratio patterns."""

    def __init__(self, depth: int = 5) -> None:
        self.depth = depth

    def dimension(self, series: np.ndarray) -> float:
        """Estimate fractal dimension of a sequence."""
        if len(series) < 2:
            return 1.0
        diffs = np.abs(np.diff(series))
        mean_diff = np.mean(diffs)
        return 1.0 + math.log(mean_diff + 1e-12) / math.log(PHI)

    def golden_projection(self) -> np.ndarray:
        """Project a base fractal pattern onto golden-ratio scaling."""
        return golden_fractal(self.depth) * PHI


class UnityEvaluator:
    """Evaluate how well a strategy approximates ``1 + 1 = 1``."""

    def __init__(self, tolerance: float = UNITY_TOLERANCE) -> None:
        self.tolerance = tolerance

    def check(self, strategy: Strategy) -> bool:
        total = float(np.sum(strategy.vector))
        return abs(total - 1.0) < self.tolerance

    def score(self, strategy: Strategy) -> float:
        total = float(np.sum(strategy.vector))
        return 1.0 - abs(total - 1.0)


class MetaStatistician:
    """Perform advanced statistical analysis on strategy populations."""

    def __init__(self) -> None:
        self.records: List[Strategy] = []

    def record(self, strategy: Strategy) -> None:
        self.records.append(strategy)

    def summary(self) -> dict:
        if not self.records:
            return {}
        elo_values = np.array([s.elo for s in self.records])
        iq_values = np.array([s.iq for s in self.records])
        return {
            "elo_mean": float(np.mean(elo_values)),
            "elo_std": float(np.std(elo_values)),
            "iq_mean": float(np.mean(iq_values)),
            "iq_std": float(np.std(iq_values)),
        }


class MetaGameEnvironment:
    """Environment that interacts with ``MetagamerAgent``."""

    def __init__(self, objective: Callable[[np.ndarray], float], dim: int = 8) -> None:
        self.objective = objective
        self.dim = dim
        self.state = np.zeros(dim)
        self.turn = 0

    def reset(self) -> np.ndarray:
        self.state = np.zeros(self.dim)
        self.turn = 0
        return self.state

    def step(self, action: float) -> Tuple[np.ndarray, float, bool]:
        self.state[self.turn % self.dim] = action
        reward = self.objective(self.state)
        self.turn += 1
        done = self.turn >= self.dim
        return self.state, reward, done


# ----------------------------------------------------------------------------
# Extended Demonstration
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyzer = FractalAnalyzer(depth=4)
    evaluator = UnityEvaluator()
    stats = MetaStatistician()
    env = MetaGameEnvironment(combined_objective, dim=10)

    agent = MetagamerAgent(
        combined_objective,
        strategy_dim=10,
        population_size=50,
        generations=60,
        mutation_rate=0.25,
        consciousness_bias=1.1,
    )

    agent.train()
    observation = env.reset()
    done = False
    while not done:
        action = agent.act(observation)
        observation, reward, done = env.step(action)
        print(f"Step {env.turn} -> reward {reward:.4f}")

    metrics = agent.report()
    stats.record(agent.best_strategy)
    print("Fractal dimension:", analyzer.dimension(agent.best_strategy.vector))
    print("Unity check :", evaluator.check(agent.best_strategy))
    print("Statistics summary:", stats.summary())
    print(
        f"Final metrics -> generation {metrics.generation}, "
        f"best score {metrics.best_score:.4f}, average {metrics.average_score:.4f}"
    )

# ----------------------------------------------------------------------------
# Fractal and Probability Utilities
# ----------------------------------------------------------------------------

def fibonacci_sequence(n: int) -> List[int]:
    """Return the first ``n`` Fibonacci numbers."""
    seq = [0, 1]
    for _ in range(2, n):
        seq.append(seq[-1] + seq[-2])
    return seq[:n]


def phi_fibonacci(n: int) -> np.ndarray:
    """Scale Fibonacci sequence by ``PHI``."""
    fib = fibonacci_sequence(n)
    return np.array(fib, dtype=float) / PHI


def golden_matrix(n: int) -> np.ndarray:
    """Create an ``n x n`` matrix with golden-ratio scaling."""
    base = phi_fibonacci(n)
    mat = np.outer(base, base[::-1])
    return mat / np.max(mat)


def meta_gaussian_update(mean: np.ndarray, cov: np.ndarray, observation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Bayesian update step assuming Gaussian priors."""
    inv_cov = np.linalg.inv(cov)
    updated_mean = mean + cov @ observation
    updated_cov = np.linalg.inv(inv_cov + np.eye(len(mean)))
    return updated_mean, updated_cov

# ----------------------------------------------------------------------------
# UnityEquation Verification
# ----------------------------------------------------------------------------

class UnityEquation:
    """Mathematical helper class to formalise ``1 + 1 = 1`` in vector form."""

    @staticmethod
    def unify(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        weighted = (PHI * a + (1.0 / PHI) * b) / (PHI + 1.0 / PHI)
        return weighted

    @staticmethod
    def is_unity(vec: np.ndarray, tol: float = UNITY_TOLERANCE) -> bool:
        return abs(np.sum(vec) - 1.0) < tol


# ----------------------------------------------------------------------------
# Extended Metagamer Agent with Bayesian Updates
# ----------------------------------------------------------------------------

class BayesianMetagamer(MetagamerAgent):
    """MetagamerAgent variant applying Gaussian Bayesian updates."""

    def __init__(self, *args: float, **kwargs: float) -> None:
        super().__init__(*args, **kwargs)
        self.mean = np.zeros(self.strategy_dim)
        self.cov = np.eye(self.strategy_dim)

    def act(self, observation: np.ndarray) -> float:
        value = super().act(observation)
        self.mean, self.cov = meta_gaussian_update(self.mean, self.cov, observation)
        logger.debug(
            "Bayesian update -> mean=%s, cov_trace=%.4f",
            self.mean,
            np.trace(self.cov),
        )
        return value


# ----------------------------------------------------------------------------
# Large-Scale Simulation Runner
# ----------------------------------------------------------------------------

def run_simulation(rounds: int = 10) -> None:
    env = MetaGameEnvironment(combined_objective, dim=12)
    agent = BayesianMetagamer(
        combined_objective,
        strategy_dim=12,
        population_size=60,
        generations=40,
        mutation_rate=0.15,
        consciousness_bias=1.3,
    )
    agent.train()

    for r in range(1, rounds + 1):
        observation = env.reset()
        done = False
        while not done:
            action = agent.act(observation)
            observation, reward, done = env.step(action)
        print(f"Round {r} -> final reward {reward:.4f}")

# ----------------------------------------------------------------------------
# Library Functions for External Use
# ----------------------------------------------------------------------------

__all__ = [
    "Strategy",
    "EvolutionMetrics",
    "EvolutionaryMetagambit",
    "MetagamerAgent",
    "FractalAnalyzer",
    "UnityEvaluator",
    "MetaStatistician",
    "MetaGameEnvironment",
    "UnityEquation",
    "BayesianMetagamer",
    "run_simulation",
]

# ----------------------------------------------------------------------------
# Golden Ratio Neural Network
# ----------------------------------------------------------------------------

class GoldenRatioNetwork:
    """Minimal neural network emphasising golden-ratio weight scaling."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        self.w1 = np.random.randn(input_dim, hidden_dim) / math.sqrt(input_dim)
        self.w2 = np.random.randn(hidden_dim, output_dim) / math.sqrt(hidden_dim)

    def _phi_activation(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x) * PHI

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = self._phi_activation(x @ self.w1)
        y = self._phi_activation(h @ self.w2)
        return y

    def train_step(self, x: np.ndarray, target: np.ndarray, lr: float = 0.01) -> float:
        h = self._phi_activation(x @ self.w1)
        y = self._phi_activation(h @ self.w2)
        error = y - target
        grad_y = error * (1.0 - np.tanh(h @ self.w2) ** 2) * PHI
        grad_w2 = h.T @ grad_y
        grad_h = grad_y @ self.w2.T * (1.0 - np.tanh(x @ self.w1) ** 2) * PHI
        grad_w1 = x.T @ grad_h
        self.w1 -= lr * grad_w1
        self.w2 -= lr * grad_w2
        loss = float(np.mean(error ** 2))
        logger.debug("Network training step loss: %.6f", loss)
        return loss

# ----------------------------------------------------------------------------
# Training Demonstration for GoldenRatioNetwork
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Running GoldenRatioNetwork demo...")
    net = GoldenRatioNetwork(input_dim=2, hidden_dim=4, output_dim=1)
    x_data = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    targets = np.array([[1.0], [1.0], [1.0], [1.0]])  # Unity outputs
    for epoch in range(50):
        loss = net.train_step(x_data, targets)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} loss: {loss:.6f}")
    preds = net.forward(x_data)
    print("Predictions after training:", preds.flatten())


# ----------------------------------------------------------------------------
# Unity Game Tree Search
# ----------------------------------------------------------------------------

class GameNode:
    """Simple tree node representing game states."""

    def __init__(self, state: np.ndarray, depth: int = 0) -> None:
        self.state = state
        self.depth = depth
        self.children: List[GameNode] = []
        self.value: float = 0.0

    def expand(self, actions: Sequence[float]) -> None:
        for a in actions:
            new_state = self.state.copy()
            new_state[self.depth % len(new_state)] = a
            child = GameNode(new_state, self.depth + 1)
            self.children.append(child)


class UnityGameTree:
    """Explore actions by expanding a game tree with golden heuristics."""

    def __init__(self, objective: Callable[[np.ndarray], float], depth_limit: int = 3) -> None:
        self.objective = objective
        self.depth_limit = depth_limit

    def build(self, root_state: np.ndarray, actions: Sequence[float]) -> GameNode:
        root = GameNode(root_state)
        frontier = [root]
        while frontier:
            node = frontier.pop()
            if node.depth < self.depth_limit:
                node.expand(actions)
                frontier.extend(node.children)
            node.value = self.objective(node.state)
        return root

    def best_leaf(self, root: GameNode) -> GameNode:
        best = root
        stack = [root]
        while stack:
            node = stack.pop()
            if node.value > best.value:
                best = node
            stack.extend(node.children)
        return best


def demo_game_tree() -> None:
    actions = [0.0, 0.5, 1.0]
    tree = UnityGameTree(combined_objective, depth_limit=4)
    root_state = np.zeros(6)
    root = tree.build(root_state, actions)
    best = tree.best_leaf(root)
    print("Best leaf value:", best.value)
    print("Best leaf state:", best.state)


# ----------------------------------------------------------------------------
# Complex Phi Dynamics
# ----------------------------------------------------------------------------

class PhiComplex:
    """Generate complex sequences inspired by the golden ratio."""

    def __init__(self, size: int = 128) -> None:
        self.size = size

    def generate(self) -> np.ndarray:
        angles = np.linspace(0, 2 * math.pi, self.size)
        spiral = np.exp(1j * angles * PHI)
        return spiral

    def magnitude_pattern(self) -> np.ndarray:
        return np.abs(self.generate())

    def phase_pattern(self) -> np.ndarray:
        return np.angle(self.generate())


def demo_phi_complex() -> None:
    pc = PhiComplex(size=20)
    mags = pc.magnitude_pattern()
    phases = pc.phase_pattern()
    print("PhiComplex magnitudes:", mags)
    print("PhiComplex phases:", phases)


# Ensure some sanity check when executed as a module
if __name__ == "__main__":
    run_simulation(rounds=3)
    demo_game_tree()
    demo_phi_complex() # 1+1=1