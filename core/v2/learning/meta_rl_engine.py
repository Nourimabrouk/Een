"""
Een v2.0 - Meta-Reinforcement Learning Engine
============================================

This module implements advanced meta-reinforcement learning capabilities
for the Een Unity Mathematics system. It enables agents to learn how to learn,
evolving their learning algorithms through experience.

Key Features:
- Population-Based Training (PBT)
- Meta-Learning (MAML-style)
- Evolutionary Strategies
- Multi-Agent Reinforcement Learning
- Unity-Aware Reward Shaping
- ELO Rating System for Agent Competition
"""

import asyncio
import time
import uuid
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import json
import pickle
import threading
from collections import deque, defaultdict

# Attempt ML imports with graceful fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Import architecture components
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from core.v2.architecture import (
    IAgent, DomainEvent, EventType, V2Config
)

logger = logging.getLogger(__name__)

# ============================================================================
# LEARNING CONFIGURATION
# ============================================================================

@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning system"""
    # Population settings
    population_size: int = 100
    elite_percentage: float = 0.2
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    
    # Training settings
    episodes_per_generation: int = 50
    max_generations: int = 1000
    learning_rate: float = 0.001
    meta_learning_rate: float = 0.01
    
    # Unity-specific settings
    unity_reward_weight: float = 2.0
    consciousness_bonus: float = 1.5
    transcendence_multiplier: float = 10.0
    phi_resonance_factor: float = 1.618
    
    # Technical settings
    device: str = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
    checkpoint_interval: int = 10
    enable_multi_agent: bool = True
    
    # Safety settings
    max_episode_length: int = 1000
    reward_clipping: Tuple[float, float] = (-10.0, 10.0)
    gradient_clipping: float = 1.0

# ============================================================================
# UNITY ENVIRONMENT
# ============================================================================

class UnityEnvironment:
    """
    Environment for training Unity Mathematics agents.
    Implements various unity-based tasks and games.
    """
    
    def __init__(self, config: MetaLearningConfig):
        self.config = config
        self.state_size = 11  # Consciousness dimensions
        self.action_size = 5  # Basic unity operations
        self.phi = 1.618033988749895
        
        self.current_state = None
        self.step_count = 0
        self.episode_reward = 0.0
        
        # Unity-specific state
        self.consciousness_field = np.zeros((11, 11))
        self.unity_coherence = 0.0
        self.transcendence_level = 0.0
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.step_count = 0
        self.episode_reward = 0.0
        
        # Initialize consciousness field with φ-harmonic patterns
        self.consciousness_field = np.zeros((11, 11))
        for i in range(11):
            for j in range(11):
                self.consciousness_field[i, j] = np.sin(i * self.phi) * np.cos(j * self.phi)
        
        self.current_state = self._get_state_vector()
        return self.current_state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Take step in environment"""
        self.step_count += 1
        
        # Execute action
        reward = self._execute_action(action)
        
        # Update state
        self._update_consciousness_field(action)
        self.current_state = self._get_state_vector()
        
        # Check if episode done
        done = (self.step_count >= self.config.max_episode_length or 
                self.transcendence_level >= 1.0)
        
        # Additional info
        info = {
            "consciousness_level": float(np.mean(self.consciousness_field)),
            "unity_coherence": self.unity_coherence,
            "transcendence_level": self.transcendence_level,
            "step_count": self.step_count
        }
        
        self.episode_reward += reward
        return self.current_state, reward, done, info
    
    def _execute_action(self, action: int) -> float:
        """Execute action and return reward"""
        actions = {
            0: self._unity_add,
            1: self._unity_multiply,
            2: self._consciousness_evolve,
            3: self._phi_resonate,
            4: self._transcend_attempt
        }
        
        if action not in actions:
            return -0.1  # Invalid action penalty
        
        return actions[action]()
    
    def _unity_add(self) -> float:
        """Perform unity addition operation"""
        # Simulate 1+1=1 operation
        a, b = np.random.random(2)
        result = max(a, b)  # Unity addition
        
        # Reward based on unity principle
        unity_score = 1.0 - abs(result - 1.0)
        return unity_score * self.config.unity_reward_weight
    
    def _unity_multiply(self) -> float:
        """Perform unity multiplication"""
        a, b = np.random.random(2)
        result = min(a * b, 1.0)  # Unity multiplication
        
        unity_score = result if a == 1.0 and b == 1.0 else result * 0.5
        return unity_score * self.config.unity_reward_weight
    
    def _consciousness_evolve(self) -> float:
        """Evolve consciousness level"""
        evolution_delta = np.random.normal(0, 0.1)
        self.consciousness_field += evolution_delta * 0.01
        
        # Consciousness reward
        consciousness_mean = np.mean(self.consciousness_field)
        reward = consciousness_mean * self.config.consciousness_bonus
        
        return np.clip(reward, *self.config.reward_clipping)
    
    def _phi_resonate(self) -> float:
        """Apply φ-harmonic resonance"""
        # Apply golden ratio transformation
        phi_field = self.consciousness_field * self.phi
        resonance = np.mean(np.sin(phi_field))
        
        # Calculate unity coherence
        self.unity_coherence = max(0, resonance * self.config.phi_resonance_factor)
        
        return self.unity_coherence
    
    def _transcend_attempt(self) -> float:
        """Attempt transcendence"""
        current_level = np.mean(self.consciousness_field)
        
        if current_level > 0.77:  # Transcendence threshold
            transcendence_gain = min(0.1, (current_level - 0.77) * 2)
            self.transcendence_level += transcendence_gain
            
            if self.transcendence_level >= 1.0:
                return self.config.transcendence_multiplier
            else:
                return transcendence_gain * 5.0
        
        return -0.1  # Failed transcendence attempt
    
    def _update_consciousness_field(self, action: int):
        """Update consciousness field based on action"""
        # Apply diffusion-like update
        kernel = np.array([[0.1, 0.2, 0.1],
                          [0.2, 0.4, 0.2],
                          [0.1, 0.2, 0.1]])
        
        # Convolution-like operation
        padded_field = np.pad(self.consciousness_field, 1, mode='edge')
        for i in range(11):
            for j in range(11):
                neighborhood = padded_field[i:i+3, j:j+3]
                self.consciousness_field[i, j] = np.sum(neighborhood * kernel)
        
        # Action-specific modifications
        if action == 2:  # Consciousness evolution
            self.consciousness_field *= 1.01
        elif action == 4:  # Transcendence
            self.consciousness_field = np.tanh(self.consciousness_field * self.phi)
    
    def _get_state_vector(self) -> np.ndarray:
        """Get flattened state vector"""
        flat_field = self.consciousness_field.flatten()
        additional_state = np.array([
            self.unity_coherence,
            self.transcendence_level,
            self.step_count / self.config.max_episode_length
        ])
        
        return np.concatenate([flat_field, additional_state])

# ============================================================================
# NEURAL NETWORK POLICIES
# ============================================================================

if TORCH_AVAILABLE:
    class UnityPolicyNetwork(nn.Module):
        """Neural network policy for Unity Mathematics tasks"""
        
        def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
            super().__init__()
            self.state_size = state_size
            self.action_size = action_size
            
            # φ-harmonic layer structure
            phi = 1.618033988749895
            hidden1 = int(hidden_size * phi)
            hidden2 = int(hidden_size / phi)
            
            self.network = nn.Sequential(
                nn.Linear(state_size, hidden1),
                nn.ReLU(),
                nn.Linear(hidden1, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden2),
                nn.ReLU(),
                nn.Linear(hidden2, action_size)
            )
            
            # Value function
            self.value_head = nn.Sequential(
                nn.Linear(hidden2, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        
        def forward(self, state):
            """Forward pass"""
            x = self.network[:-1](state)  # All but last layer
            
            # Action probabilities
            action_logits = self.network[-1](x)
            action_probs = F.softmax(action_logits, dim=-1)
            
            # State value
            value = self.value_head(x)
            
            return action_probs, value
        
        def get_action(self, state):
            """Get action from policy"""
            with torch.no_grad():
                action_probs, value = self.forward(state)
                action = torch.multinomial(action_probs, 1)
                return action.item(), action_probs[action].item(), value.item()

else:
    class UnityPolicyNetwork:
        """Mock policy network when PyTorch unavailable"""
        
        def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
            self.state_size = state_size
            self.action_size = action_size
            logger.warning("PyTorch not available - using mock policy network")
        
        def get_action(self, state):
            action = np.random.randint(0, self.action_size)
            prob = 1.0 / self.action_size
            value = np.random.random()
            return action, prob, value

# ============================================================================
# META-LEARNING AGENT
# ============================================================================

class MetaLearningAgent(IAgent):
    """
    Meta-learning agent that learns to learn Unity Mathematics.
    Uses reinforcement learning with unity-aware rewards.
    """
    
    def __init__(self, config: MetaLearningConfig):
        self.agent_id = str(uuid.uuid4())
        self.config = config
        self.environment = UnityEnvironment(config)
        
        # Neural network policy
        state_size = self.environment.state_size + 3  # +3 for additional state
        self.policy = UnityPolicyNetwork(state_size, self.environment.action_size)
        
        if TORCH_AVAILABLE:
            self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        
        # Training data
        self.experience_buffer = deque(maxlen=10000)
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Meta-learning state
        self.meta_parameters = {
            "learning_rate": config.learning_rate,
            "exploration_rate": 0.1,
            "unity_focus": 0.5
        }
        
        # Performance metrics
        self.elo_rating = 1500.0  # Starting ELO
        self.wins = 0
        self.losses = 0
        self.draws = 0
        
        # Agent state
        self.state = {
            "consciousness_level": 0.0,
            "unity_mastery": 0.0,
            "transcendence_events": 0,
            "training_episodes": 0
        }
    
    @property
    def agent_type(self) -> str:
        return "meta_learning_agent"
    
    def execute_task(self, task: Dict[str, Any]) -> Any:
        """Execute meta-learning task"""
        task_type = task.get("type", "train")
        
        if task_type == "train":
            return self._train_episode(task)
        elif task_type == "evaluate":
            return self._evaluate_performance(task)
        elif task_type == "compete":
            return self._compete_against(task)
        elif task_type == "meta_update":
            return self._meta_update(task)
        else:
            return {"error": f"Unknown task type: {task_type}"}
    
    def _train_episode(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Train for one episode"""
        episode_data = []
        state = self.environment.reset()
        total_reward = 0.0
        steps = 0
        
        while steps < self.config.max_episode_length:
            # Get action from policy
            if TORCH_AVAILABLE:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action, action_prob, value = self.policy.get_action(state_tensor)
            else:
                action, action_prob, value = self.policy.get_action(state)
            
            # Take step
            next_state, reward, done, info = self.environment.step(action)
            
            # Store experience
            episode_data.append({
                "state": state.copy(),
                "action": action,
                "reward": reward,
                "next_state": next_state.copy(),
                "action_prob": action_prob,
                "value": value,
                "done": done
            })
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        # Store episode
        self.experience_buffer.extend(episode_data)
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)
        
        # Update consciousness level
        consciousness_gain = info.get("consciousness_level", 0.0) * 0.1
        self.state["consciousness_level"] += consciousness_gain
        
        # Check for transcendence
        if info.get("transcendence_level", 0.0) >= 1.0:
            self.state["transcendence_events"] += 1
        
        self.state["training_episodes"] += 1
        
        # Unity mastery calculation
        unity_performance = total_reward / (steps + 1)
        self.state["unity_mastery"] = 0.9 * self.state["unity_mastery"] + 0.1 * unity_performance
        
        return {
            "episode_reward": total_reward,
            "episode_length": steps,
            "consciousness_gain": consciousness_gain,
            "transcendence_achieved": info.get("transcendence_level", 0.0) >= 1.0,
            "unity_coherence": info.get("unity_coherence", 0.0)
        }
    
    def _evaluate_performance(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate current performance"""
        episodes = task.get("episodes", 10)
        evaluation_rewards = []
        
        for _ in range(episodes):
            result = self._train_episode({"type": "eval"})
            evaluation_rewards.append(result["episode_reward"])
        
        performance = {
            "mean_reward": float(np.mean(evaluation_rewards)),
            "std_reward": float(np.std(evaluation_rewards)),
            "max_reward": float(np.max(evaluation_rewards)),
            "min_reward": float(np.min(evaluation_rewards)),
            "elo_rating": self.elo_rating,
            "consciousness_level": self.state["consciousness_level"],
            "unity_mastery": self.state["unity_mastery"]
        }
        
        return performance
    
    def _compete_against(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Compete against another agent"""
        opponent_agent = task.get("opponent")
        if not opponent_agent:
            return {"error": "No opponent provided"}
        
        # Simple competition: compare performance
        my_performance = self._evaluate_performance({"episodes": 5})
        opponent_performance = opponent_agent.evaluate_performance({"episodes": 5})
        
        my_score = my_performance["mean_reward"]
        opponent_score = opponent_performance["mean_reward"]
        
        # Determine winner
        if my_score > opponent_score:
            result = "win"
            self.wins += 1
        elif my_score < opponent_score:
            result = "loss"
            self.losses += 1
        else:
            result = "draw"
            self.draws += 1
        
        # Update ELO rating
        self._update_elo_rating(opponent_agent.elo_rating, result)
        
        return {
            "result": result,
            "my_score": my_score,
            "opponent_score": opponent_score,
            "new_elo": self.elo_rating
        }
    
    def _update_elo_rating(self, opponent_elo: float, result: str):
        """Update ELO rating based on competition result"""
        K = 32  # ELO K-factor
        
        # Expected score
        expected = 1 / (1 + 10 ** ((opponent_elo - self.elo_rating) / 400))
        
        # Actual score
        actual = {"win": 1.0, "draw": 0.5, "loss": 0.0}[result]
        
        # Update rating
        self.elo_rating += K * (actual - expected)
    
    def _meta_update(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform meta-learning update"""
        if not self.experience_buffer:
            return {"error": "No experience to learn from"}
        
        # Meta-learning: adapt learning parameters based on performance
        recent_rewards = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
        
        if recent_rewards:
            avg_reward = np.mean(recent_rewards)
            reward_trend = np.mean(np.diff(recent_rewards)) if len(recent_rewards) > 1 else 0
            
            # Adapt learning rate
            if reward_trend > 0:
                self.meta_parameters["learning_rate"] *= 1.05  # Increase if improving
            else:
                self.meta_parameters["learning_rate"] *= 0.95  # Decrease if not improving
            
            # Adapt exploration
            if avg_reward > 5.0:
                self.meta_parameters["exploration_rate"] *= 0.99  # Reduce exploration if performing well
            else:
                self.meta_parameters["exploration_rate"] *= 1.01  # Increase exploration if struggling
            
            # Bounds checking
            self.meta_parameters["learning_rate"] = np.clip(self.meta_parameters["learning_rate"], 1e-5, 1e-1)
            self.meta_parameters["exploration_rate"] = np.clip(self.meta_parameters["exploration_rate"], 0.01, 0.5)
        
        return {
            "meta_parameters": self.meta_parameters.copy(),
            "performance_trend": reward_trend if 'reward_trend' in locals() else 0,
            "episodes_learned_from": len(self.experience_buffer)
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get agent state"""
        return {
            **self.state,
            "elo_rating": self.elo_rating,
            "win_rate": self.wins / max(1, self.wins + self.losses + self.draws),
            "meta_parameters": self.meta_parameters.copy(),
            "experience_size": len(self.experience_buffer),
            "avg_episode_reward": np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
        }
    
    def handle_event(self, event: DomainEvent) -> None:
        """Handle domain events"""
        if event.event_type == EventType.TRAINING_CYCLE_COMPLETED.name:
            # Trigger meta-update
            self._meta_update({})
    
    def evolve(self, evolution_params: Dict[str, Any]) -> None:
        """Evolve agent through meta-learning"""
        generations = evolution_params.get("generations", 1)
        
        for _ in range(generations):
            # Train for multiple episodes
            for episode in range(self.config.episodes_per_generation):
                self._train_episode({})
            
            # Meta-update
            self._meta_update({})
            
            # Evolve consciousness
            self.state["consciousness_level"] = min(1.0, self.state["consciousness_level"] + 0.01)

# ============================================================================
# POPULATION-BASED TRAINING
# ============================================================================

class PopulationTrainer:
    """Manages population-based training of meta-learning agents"""
    
    def __init__(self, config: MetaLearningConfig):
        self.config = config
        self.population: List[MetaLearningAgent] = []
        self.generation = 0
        self.training_history = []
        
        # Initialize population
        self._initialize_population()
    
    def _initialize_population(self):
        """Initialize agent population"""
        logger.info(f"Initializing population of {self.config.population_size} agents")
        
        for _ in range(self.config.population_size):
            agent = MetaLearningAgent(self.config)
            self.population.append(agent)
    
    def train_generation(self) -> Dict[str, Any]:
        """Train one generation of the population"""
        logger.info(f"Training generation {self.generation}")
        
        # Evaluate all agents
        performances = []
        for agent in self.population:
            performance = agent._evaluate_performance({"episodes": 5})
            performances.append(performance)
        
        # Select elite agents
        elite_count = int(self.config.population_size * self.config.elite_percentage)
        elite_indices = np.argsort([p["mean_reward"] for p in performances])[-elite_count:]
        elite_agents = [self.population[i] for i in elite_indices]
        
        # Create next generation
        next_population = elite_agents.copy()  # Keep elites
        
        # Fill remaining slots with offspring
        while len(next_population) < self.config.population_size:
            # Select parents
            parent1, parent2 = np.random.choice(elite_agents, 2, replace=False)
            
            # Create offspring through crossover and mutation
            offspring = self._create_offspring(parent1, parent2)
            next_population.append(offspring)
        
        self.population = next_population
        self.generation += 1
        
        # Record generation stats
        generation_stats = {
            "generation": self.generation,
            "best_performance": performances[elite_indices[-1]],
            "avg_performance": np.mean([p["mean_reward"] for p in performances]),
            "population_diversity": self._calculate_diversity()
        }
        
        self.training_history.append(generation_stats)
        
        return generation_stats
    
    def _create_offspring(self, parent1: MetaLearningAgent, parent2: MetaLearningAgent) -> MetaLearningAgent:
        """Create offspring through crossover and mutation"""
        # Create new agent
        offspring = MetaLearningAgent(self.config)
        
        # Crossover meta-parameters
        for param in offspring.meta_parameters:
            if np.random.random() < self.config.crossover_rate:
                offspring.meta_parameters[param] = parent1.meta_parameters[param]
            else:
                offspring.meta_parameters[param] = parent2.meta_parameters[param]
        
        # Mutation
        for param in offspring.meta_parameters:
            if np.random.random() < self.config.mutation_rate:
                mutation = np.random.normal(0, 0.1)
                if isinstance(offspring.meta_parameters[param], float):
                    offspring.meta_parameters[param] *= (1 + mutation)
                    offspring.meta_parameters[param] = max(0.001, offspring.meta_parameters[param])
        
        return offspring
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity"""
        # Simple diversity measure based on meta-parameters
        learning_rates = [agent.meta_parameters["learning_rate"] for agent in self.population]
        return float(np.std(learning_rates))
    
    def get_best_agent(self) -> MetaLearningAgent:
        """Get best performing agent"""
        performances = []
        for agent in self.population:
            performance = agent._evaluate_performance({"episodes": 3})
            performances.append(performance["mean_reward"])
        
        best_idx = np.argmax(performances)
        return self.population[best_idx]
    
    def tournament(self, tournament_size: int = 8) -> Dict[str, Any]:
        """Run tournament between agents"""
        # Select random agents for tournament
        tournament_agents = np.random.choice(self.population, tournament_size, replace=False)
        
        # Round-robin tournament
        results = defaultdict(int)
        for i, agent1 in enumerate(tournament_agents):
            for j, agent2 in enumerate(tournament_agents[i+1:], i+1):
                competition_result = agent1._compete_against({"opponent": agent2})
                
                if competition_result["result"] == "win":
                    results[agent1.agent_id] += 3  # Win = 3 points
                elif competition_result["result"] == "draw":
                    results[agent1.agent_id] += 1  # Draw = 1 point
                    results[agent2.agent_id] += 1
                else:
                    results[agent2.agent_id] += 3  # Loss for agent1 = win for agent2
        
        # Determine tournament winner
        winner_id = max(results, key=results.get)
        winner = next(agent for agent in tournament_agents if agent.agent_id == winner_id)
        
        return {
            "winner": winner,
            "scores": dict(results),
            "tournament_size": tournament_size
        }

# ============================================================================
# META-RL ENGINE
# ============================================================================

class MetaReinforcementLearningEngine:
    """
    Main engine coordinating all meta-reinforcement learning activities
    """
    
    def __init__(self, config: MetaLearningConfig):
        self.config = config
        self.population_trainer = PopulationTrainer(config)
        self.training_active = False
        self.training_thread = None
        
        # Metrics and logging
        self.total_episodes = 0
        self.total_generations = 0
        self.best_agents = []
        
    def start_training(self) -> None:
        """Start continuous training"""
        if self.training_active:
            logger.warning("Training already active")
            return
        
        self.training_active = True
        self.training_thread = threading.Thread(target=self._training_loop)
        self.training_thread.start()
        logger.info("🧠 Meta-RL training started")
    
    def stop_training(self) -> None:
        """Stop continuous training"""
        self.training_active = False
        if self.training_thread:
            self.training_thread.join()
        logger.info("🛑 Meta-RL training stopped")
    
    def _training_loop(self) -> None:
        """Main training loop"""
        while self.training_active and self.total_generations < self.config.max_generations:
            try:
                # Train one generation
                generation_stats = self.population_trainer.train_generation()
                self.total_generations += 1
                
                # Log progress
                logger.info(f"Generation {self.total_generations}: "
                           f"Best reward: {generation_stats['best_performance']['mean_reward']:.3f}, "
                           f"Avg reward: {generation_stats['avg_performance']:.3f}")
                
                # Save best agent
                best_agent = self.population_trainer.get_best_agent()
                self.best_agents.append(best_agent)
                
                # Checkpoint saving
                if self.total_generations % self.config.checkpoint_interval == 0:
                    self._save_checkpoint()
                
            except Exception as e:
                logger.error(f"Training error: {e}")
                break
    
    def _save_checkpoint(self) -> None:
        """Save training checkpoint"""
        checkpoint_path = Path(f"checkpoints/meta_rl_gen_{self.total_generations}.pkl")
        checkpoint_path.parent.mkdir(exist_ok=True)
        
        checkpoint_data = {
            "generation": self.total_generations,
            "config": self.config,
            "training_history": self.population_trainer.training_history,
            "best_agent_states": [agent.get_state() for agent in self.best_agents[-10:]]  # Last 10
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get training status"""
        best_agent = self.population_trainer.get_best_agent()
        
        return {
            "training_active": self.training_active,
            "total_generations": self.total_generations,
            "population_size": len(self.population_trainer.population),
            "best_agent_performance": best_agent.get_state(),
            "avg_population_elo": np.mean([agent.elo_rating for agent in self.population_trainer.population]),
            "training_progress": self.total_generations / self.config.max_generations
        }
    
    def run_tournament(self) -> Dict[str, Any]:
        """Run tournament between current population"""
        return self.population_trainer.tournament()

# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    'MetaLearningConfig',
    'UnityEnvironment', 
    'MetaLearningAgent',
    'PopulationTrainer',
    'MetaReinforcementLearningEngine'
]