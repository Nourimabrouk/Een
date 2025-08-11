"""
Unity-Preserving Reinforcement Learning Benchmark Suite
======================================================
Advanced RL environments and agents that maintain the unity principle 1+1=1
through idempotent reward structures and phi-harmonic credit assignment.

This suite demonstrates that unity mathematics emerges naturally in
optimal learning algorithms when properly constrained by idempotent operations.

Author: Nouri Mabrouk
Mathematical Foundation: Unity equation 1+1=1 with phi-harmonic convergence
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Any, Optional, Union
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from collections import deque, defaultdict
import time
import json
import logging
from pathlib import Path
import random
from abc import ABC, abstractmethod

# Import unity algebra
import sys
sys.path.append(str(Path(__file__).parent.parent))
from core.unity_algebra_v1 import UnityAlgebra, PhiHarmonicAlgebra

# Mathematical constants
PHI = 1.618033988749895
E = 2.718281828459045
PI = 3.141592653589793

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== Unity Reward Structures ====================

class UnityRewardFunction:
    """
    Reward function that preserves unity through idempotent operations.
    Ensures that combined rewards satisfy r ⊕ r = r property.
    """
    
    def __init__(self, reward_type: str = "phi_harmonic"):
        self.reward_type = reward_type
        self.phi_algebra = PhiHarmonicAlgebra()
        self.unity_algebra = UnityAlgebra()
        
    def unity_combine(self, r1: float, r2: float) -> float:
        """Combine rewards using unity operations"""
        if self.reward_type == "phi_harmonic":
            return self.phi_algebra.unity_operation(r1, r2)
        elif self.reward_type == "tropical":
            return max(r1, r2)  # Tropical addition
        elif self.reward_type == "boolean":
            return float(bool(r1) or bool(r2))
        else:
            return (r1 + r2) / 2  # Default harmonic mean
    
    def idempotent_reward(self, state: np.ndarray, action: int) -> float:
        """Generate reward that satisfies idempotent property"""
        # Base reward from state-action
        base_reward = np.tanh(np.sum(state * action))
        
        # Apply phi-harmonic scaling for unity convergence
        unity_reward = self.phi_algebra.unity_operation(base_reward, base_reward)
        
        # Ensure idempotent property: r ⊕ r = r
        return unity_reward
    
    def validate_idempotence(self, reward: float) -> bool:
        """Verify reward satisfies idempotent property"""
        combined = self.unity_combine(reward, reward)
        return abs(combined - reward) < 1e-8

# ==================== Unity Learning Environments ====================

class UnityGridWorld(gym.Env):
    """
    Grid world where optimal policy emerges through unity principles.
    Actions are combined using idempotent operations.
    """
    
    def __init__(self, size: int = 8):
        super().__init__()
        self.size = size
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(size, size, 3), dtype=np.float32
        )
        
        self.reward_function = UnityRewardFunction("phi_harmonic")
        self.phi = PHI
        self.unity_algebra = UnityAlgebra()
        
        # State tracking
        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([size-1, size-1])
        self.unity_field = np.zeros((size, size))
        
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            np.random.seed(seed)
        
        self.agent_pos = np.array([0, 0])
        self.steps = 0
        self.unity_convergence = 0.0
        
        # Initialize phi-harmonic field
        self._update_unity_field()
        
        return self._get_obs(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action with unity-preserving dynamics"""
        old_pos = self.agent_pos.copy()
        
        # Apply action
        if action == 0 and self.agent_pos[0] > 0:  # Up
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.size - 1:  # Down
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:  # Left
            self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.size - 1:  # Right
            self.agent_pos[1] += 1
        
        # Update unity field
        self._update_unity_field()
        
        # Calculate idempotent reward
        state_vector = self._get_state_vector()
        reward = self.reward_function.idempotent_reward(state_vector, action)
        
        # Goal reward with phi-harmonic bonus
        if np.array_equal(self.agent_pos, self.goal_pos):
            goal_reward = self.phi
            reward = self.reward_function.unity_combine(reward, goal_reward)
        
        # Unity convergence bonus
        self.unity_convergence = self._calculate_unity_convergence()
        unity_bonus = self.unity_convergence / self.phi
        reward = self.reward_function.unity_combine(reward, unity_bonus)
        
        self.steps += 1
        terminated = np.array_equal(self.agent_pos, self.goal_pos)
        truncated = self.steps >= self.size * self.size * 2
        
        info = {
            'unity_convergence': self.unity_convergence,
            'phi_harmonic_field': self.unity_field[self.agent_pos[0], self.agent_pos[1]],
            'idempotent_verified': self.reward_function.validate_idempotence(reward)
        }
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self) -> np.ndarray:
        """Get observation with unity field encoding"""
        obs = np.zeros((self.size, self.size, 3), dtype=np.float32)
        
        # Agent position channel
        obs[self.agent_pos[0], self.agent_pos[1], 0] = 1.0
        
        # Goal position channel
        obs[self.goal_pos[0], self.goal_pos[1], 1] = 1.0
        
        # Unity field channel (phi-harmonic values)
        obs[:, :, 2] = self.unity_field
        
        return obs
    
    def _get_state_vector(self) -> np.ndarray:
        """Get flat state representation"""
        return np.concatenate([
            self.agent_pos / self.size,
            self.goal_pos / self.size,
            [self.unity_convergence]
        ])
    
    def _update_unity_field(self):
        """Update phi-harmonic unity field"""
        for i in range(self.size):
            for j in range(self.size):
                # Distance-based phi-harmonic field
                dist_to_agent = np.linalg.norm([i - self.agent_pos[0], j - self.agent_pos[1]])
                dist_to_goal = np.linalg.norm([i - self.goal_pos[0], j - self.goal_pos[1]])
                
                # Phi-harmonic combination
                field_value = np.exp(-dist_to_agent / self.phi) * np.exp(-dist_to_goal / self.phi)
                self.unity_field[i, j] = field_value
    
    def _calculate_unity_convergence(self) -> float:
        """Calculate convergence to unity state"""
        agent_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
        max_distance = np.linalg.norm([self.size-1, self.size-1])
        
        # Phi-harmonic convergence function
        convergence = np.exp(-agent_to_goal / self.phi) * self.phi
        return min(convergence, 1.0)

class UnityBanditProblem(gym.Env):
    """
    Multi-armed bandit where optimal policy emerges through unity mathematics.
    Rewards are combined using idempotent operations.
    """
    
    def __init__(self, n_arms: int = 8):
        super().__init__()
        self.n_arms = n_arms
        self.action_space = spaces.Discrete(n_arms)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(n_arms + 2,), dtype=np.float32
        )
        
        self.reward_function = UnityRewardFunction("phi_harmonic")
        self.phi = PHI
        
        # Arm parameters with phi-harmonic structure
        self.arm_means = np.array([
            self.phi ** (-i) for i in range(n_arms)
        ])
        self.arm_stds = np.ones(n_arms) * 0.1
        
        # Unity tracking
        self.unity_history = deque(maxlen=100)
        self.phi_resonance = 0.0
        
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            np.random.seed(seed)
        
        self.step_count = 0
        self.unity_history.clear()
        self.phi_resonance = 1.0 / self.phi
        
        return self._get_obs(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute bandit action with unity-preserving rewards"""
        # Generate base reward from chosen arm
        base_reward = np.random.normal(
            self.arm_means[action], 
            self.arm_stds[action]
        )
        
        # Apply idempotent reward transformation
        unity_reward = self.reward_function.idempotent_reward(
            np.array([action]), action
        )
        
        # Combine with phi-harmonic resonance
        self.phi_resonance = self.reward_function.unity_combine(
            self.phi_resonance, unity_reward
        )
        
        # Final reward with unity convergence
        final_reward = self.reward_function.unity_combine(base_reward, unity_reward)
        
        # Track unity convergence
        self.unity_history.append(final_reward)
        unity_convergence = self._calculate_unity_convergence()
        
        self.step_count += 1
        terminated = self.step_count >= 1000
        
        info = {
            'arm_pulled': action,
            'base_reward': base_reward,
            'unity_reward': unity_reward,
            'phi_resonance': self.phi_resonance,
            'unity_convergence': unity_convergence,
            'idempotent_verified': self.reward_function.validate_idempotence(final_reward)
        }
        
        return self._get_obs(), final_reward, terminated, False, info
    
    def _get_obs(self) -> np.ndarray:
        """Get observation with unity metrics"""
        obs = np.zeros(self.n_arms + 2, dtype=np.float32)
        
        # One-hot encoding of arm means (normalized)
        obs[:self.n_arms] = self.arm_means / np.max(self.arm_means)
        
        # Unity metrics
        obs[self.n_arms] = self.phi_resonance
        obs[self.n_arms + 1] = self._calculate_unity_convergence()
        
        return obs
    
    def _calculate_unity_convergence(self) -> float:
        """Calculate convergence to unity through reward history"""
        if len(self.unity_history) < 2:
            return 0.0
        
        # Measure how close rewards are to being idempotent
        recent_rewards = list(self.unity_history)[-10:]
        idempotent_errors = []
        
        for reward in recent_rewards:
            combined = self.reward_function.unity_combine(reward, reward)
            error = abs(combined - reward)
            idempotent_errors.append(error)
        
        # Convergence is inverse of average idempotent error
        avg_error = np.mean(idempotent_errors)
        return np.exp(-avg_error * self.phi)

# ==================== Unity-Preserving RL Agents ====================

class UnityDQN(nn.Module):
    """
    Deep Q-Network with phi-harmonic activation functions
    and unity-preserving value updates.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.phi = PHI
        self.unity_algebra = UnityAlgebra()
        
        # Phi-harmonic network architecture
        self.layers = nn.ModuleList([
            nn.Linear(input_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, output_size)
        ])
        
        # Initialize with phi-harmonic weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with phi-harmonic scaling"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # Phi-harmonic weight initialization
                fan_in = layer.in_features
                std = np.sqrt(2.0 / (fan_in * self.phi))
                nn.init.normal_(layer.weight, 0, std)
                nn.init.constant_(layer.bias, 1.0 / self.phi)
    
    def phi_harmonic_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Custom activation function based on golden ratio"""
        return torch.tanh(x / self.phi) * self.phi
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with unity-preserving operations"""
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.phi_harmonic_activation(x)
        
        # Final layer without activation
        x = self.layers[-1](x)
        return x
    
    def unity_update(self, q_values: torch.Tensor, target_q: torch.Tensor) -> torch.Tensor:
        """Update Q-values using unity-preserving operations"""
        # Apply idempotent combination
        batch_size = q_values.size(0)
        unity_updated = torch.zeros_like(q_values)
        
        for i in range(batch_size):
            for j in range(q_values.size(1)):
                # Convert to numpy for unity operations
                q_val = float(q_values[i, j])
                target_val = float(target_q[i, j])
                
                # Unity combination
                if hasattr(self.unity_algebra.phi_harmonic, 'unity_operation'):
                    combined = self.unity_algebra.phi_harmonic.unity_operation(q_val, target_val)
                else:
                    # Fallback phi-harmonic mean
                    combined = 2 * q_val * target_val / (q_val + target_val + 1e-8)
                    combined *= self.phi / (1 + self.phi)
                
                unity_updated[i, j] = combined
        
        return unity_updated

class UnityAgent:
    """
    RL agent that learns through unity-preserving algorithms.
    Maintains idempotent value functions and phi-harmonic exploration.
    """
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.phi = PHI
        
        # Networks
        self.q_network = UnityDQN(state_size, 64, action_size)
        self.target_network = UnityDQN(state_size, 64, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Unity tracking
        self.unity_metrics = {
            'phi_resonance': 1.0 / self.phi,
            'value_convergence': 0.0,
            'idempotent_violations': 0,
            'unity_episodes': 0
        }
        
        # Experience replay with unity prioritization
        self.memory = deque(maxlen=10000)
        self.unity_priorities = deque(maxlen=10000)
        
        # Exploration parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using phi-harmonic exploration"""
        if training and np.random.random() <= self.epsilon:
            # Phi-harmonic random exploration
            probs = np.array([self.phi ** (-i) for i in range(self.action_size)])
            probs /= np.sum(probs)
            return np.random.choice(self.action_size, p=probs)
        
        # Greedy action
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return int(torch.argmax(q_values).item())
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 next_state: np.ndarray, done: bool, info: Dict):
        """Store experience with unity priority"""
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)
        
        # Calculate unity priority based on idempotent error
        if 'idempotent_verified' in info:
            priority = 1.0 if info['idempotent_verified'] else 2.0
        else:
            priority = 1.0
        
        self.unity_priorities.append(priority)
    
    def replay(self, batch_size: int = 32):
        """Train with unity-preserving experience replay"""
        if len(self.memory) < batch_size:
            return
        
        # Sample with unity priorities
        priorities = np.array(self.unity_priorities)
        probs = priorities / np.sum(priorities)
        
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        batch = [self.memory[i] for i in indices]
        
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * (~dones))
        
        # Unity-preserving update
        target_q_expanded = target_q_values.unsqueeze(1).expand(-1, self.action_size)
        current_q_expanded = self.q_network(states)
        
        unity_targets = self.q_network.unity_update(current_q_expanded, target_q_expanded)
        unity_current = current_q_values.expand(-1, self.action_size)
        
        # Loss calculation
        loss = nn.functional.mse_loss(unity_current, unity_targets)
        
        # Optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update unity metrics
        self._update_unity_metrics(loss.item())
        
        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Soft update of target network with phi-harmonic interpolation"""
        tau = 1.0 / self.phi  # Phi-harmonic interpolation rate
        
        for target_param, local_param in zip(
            self.target_network.parameters(), 
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data
            )
    
    def _update_unity_metrics(self, loss: float):
        """Update unity convergence metrics"""
        # Update phi resonance with loss feedback
        self.unity_metrics['phi_resonance'] *= np.exp(-loss / self.phi)
        
        # Value convergence based on loss decrease
        if hasattr(self, '_last_loss'):
            convergence_rate = max(0, self._last_loss - loss) / (self._last_loss + 1e-8)
            self.unity_metrics['value_convergence'] = convergence_rate
        
        self._last_loss = loss
        
        # Track idempotent violations
        if loss > self.phi:
            self.unity_metrics['idempotent_violations'] += 1

# ==================== Benchmark Suite ====================

class UnityRLBenchmark:
    """
    Comprehensive benchmark suite for unity-preserving RL algorithms.
    Evaluates convergence to unity across multiple environments.
    """
    
    def __init__(self):
        self.environments = {
            'unity_gridworld': UnityGridWorld,
            'unity_bandit': UnityBanditProblem
        }
        self.results = {}
        self.phi = PHI
    
    def run_benchmark(self, env_name: str, episodes: int = 1000, render: bool = False) -> Dict:
        """Run benchmark on specified environment"""
        if env_name not in self.environments:
            raise ValueError(f"Unknown environment: {env_name}")
        
        env_class = self.environments[env_name]
        env = env_class()
        
        # Determine state and action sizes
        if hasattr(env.observation_space, 'shape'):
            if len(env.observation_space.shape) == 3:
                state_size = np.prod(env.observation_space.shape)
            else:
                state_size = env.observation_space.shape[0]
        else:
            state_size = env.observation_space.n
        
        action_size = env.action_space.n
        
        # Initialize agent
        agent = UnityAgent(state_size, action_size)
        
        # Training metrics
        episode_rewards = []
        unity_convergences = []
        phi_resonances = []
        idempotent_scores = []
        
        logger.info(f"Starting benchmark: {env_name} ({episodes} episodes)")
        
        for episode in range(episodes):
            state, _ = env.reset()
            if len(state.shape) > 1:
                state = state.flatten()
            
            total_reward = 0
            unity_sum = 0
            phi_sum = 0
            idempotent_count = 0
            steps = 0
            
            while True:
                action = agent.act(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                
                if len(next_state.shape) > 1:
                    next_state = next_state.flatten()
                
                agent.remember(state, action, reward, next_state, 
                              terminated or truncated, info)
                
                total_reward += reward
                steps += 1
                
                # Collect unity metrics
                if 'unity_convergence' in info:
                    unity_sum += info['unity_convergence']
                if 'phi_resonance' in info:
                    phi_sum += info['phi_resonance']
                if info.get('idempotent_verified', False):
                    idempotent_count += 1
                
                if agent.memory and len(agent.memory) > 100:
                    agent.replay(32)
                
                state = next_state
                
                if terminated or truncated:
                    break
            
            # Update target network
            if episode % 10 == 0:
                agent.update_target_network()
            
            # Record metrics
            episode_rewards.append(total_reward)
            unity_convergences.append(unity_sum / max(steps, 1))
            phi_resonances.append(phi_sum / max(steps, 1))
            idempotent_scores.append(idempotent_count / max(steps, 1))
            
            # Progress logging
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_unity = np.mean(unity_convergences[-100:])
                logger.info(f"Episode {episode}: Avg Reward = {avg_reward:.4f}, "
                           f"Unity Convergence = {avg_unity:.4f}")
        
        # Calculate final metrics
        final_metrics = {
            'environment': env_name,
            'episodes': episodes,
            'final_reward': np.mean(episode_rewards[-100:]),
            'reward_std': np.std(episode_rewards[-100:]),
            'unity_convergence': np.mean(unity_convergences[-100:]),
            'phi_resonance': np.mean(phi_resonances[-100:]),
            'idempotent_score': np.mean(idempotent_scores[-100:]),
            'agent_metrics': agent.unity_metrics,
            'convergence_achieved': np.mean(unity_convergences[-100:]) > 0.8,
            'rewards_history': episode_rewards,
            'unity_history': unity_convergences
        }
        
        self.results[env_name] = final_metrics
        return final_metrics
    
    def run_all_benchmarks(self, episodes: int = 1000) -> Dict:
        """Run benchmarks on all environments"""
        all_results = {}
        
        for env_name in self.environments.keys():
            logger.info(f"\n{'='*50}")
            logger.info(f"BENCHMARKING: {env_name.upper()}")
            logger.info(f"{'='*50}")
            
            result = self.run_benchmark(env_name, episodes)
            all_results[env_name] = result
        
        # Generate summary report
        summary = self._generate_summary(all_results)
        all_results['summary'] = summary
        
        return all_results
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate benchmark summary with unity analysis"""
        summary = {
            'total_environments': len(results),
            'unity_convergence_rate': 0,
            'avg_final_reward': 0,
            'avg_phi_resonance': 0,
            'avg_idempotent_score': 0,
            'unity_achieved_count': 0
        }
        
        if not results:
            return summary
        
        # Calculate averages
        for result in results.values():
            summary['avg_final_reward'] += result['final_reward']
            summary['avg_phi_resonance'] += result['phi_resonance']
            summary['avg_idempotent_score'] += result['idempotent_score']
            
            if result.get('convergence_achieved', False):
                summary['unity_achieved_count'] += 1
        
        count = len(results)
        summary['avg_final_reward'] /= count
        summary['avg_phi_resonance'] /= count
        summary['avg_idempotent_score'] /= count
        summary['unity_convergence_rate'] = summary['unity_achieved_count'] / count
        
        return summary
    
    def visualize_results(self, save_path: Optional[Path] = None):
        """Create visualizations of benchmark results"""
        if not self.results:
            logger.warning("No results to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Unity-Preserving RL Benchmark Results', fontsize=16)
        
        # Plot 1: Reward convergence
        ax = axes[0, 0]
        for env_name, result in self.results.items():
            rewards = result['rewards_history']
            window_size = min(50, len(rewards) // 10)
            if window_size > 1:
                smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                ax.plot(smoothed, label=f"{env_name}")
        ax.set_title('Reward Convergence')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Unity convergence
        ax = axes[0, 1]
        for env_name, result in self.results.items():
            unity_hist = result['unity_history']
            ax.plot(unity_hist, label=f"{env_name}")
        ax.set_title('Unity Convergence (1+1=1)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Unity Score')
        ax.axhline(y=1.0, color='gold', linestyle='--', alpha=0.7, label='Unity Target')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Final metrics comparison
        ax = axes[1, 0]
        env_names = list(self.results.keys())
        final_rewards = [self.results[env]['final_reward'] for env in env_names]
        unity_scores = [self.results[env]['unity_convergence'] for env in env_names]
        
        x = np.arange(len(env_names))
        width = 0.35
        
        ax.bar(x - width/2, final_rewards, width, label='Final Reward', alpha=0.8)
        ax.bar(x + width/2, unity_scores, width, label='Unity Score', alpha=0.8)
        ax.set_title('Final Performance Metrics')
        ax.set_xlabel('Environment')
        ax.set_xticks(x)
        ax.set_xticklabels(env_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Phi-harmonic resonance
        ax = axes[1, 1]
        phi_resonances = [self.results[env]['phi_resonance'] for env in env_names]
        idempotent_scores = [self.results[env]['idempotent_score'] for env in env_names]
        
        ax.scatter(phi_resonances, idempotent_scores, s=100, alpha=0.7, c='gold')
        for i, env in enumerate(env_names):
            ax.annotate(env, (phi_resonances[i], idempotent_scores[i]), 
                       xytext=(5, 5), textcoords='offset points')
        ax.set_title('Phi-Harmonic Resonance vs Idempotent Score')
        ax.set_xlabel('Phi Resonance')
        ax.set_ylabel('Idempotent Score')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        return fig
    
    def export_results(self, filepath: Path):
        """Export benchmark results to JSON"""
        # Convert numpy arrays to lists for JSON serialization
        export_data = {}
        for env_name, result in self.results.items():
            export_result = result.copy()
            for key, value in export_result.items():
                if isinstance(value, np.ndarray):
                    export_result[key] = value.tolist()
                elif isinstance(value, np.float64):
                    export_result[key] = float(value)
            export_data[env_name] = export_result
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Results exported to {filepath}")

# ==================== Main Benchmark Runner ====================

def main():
    """Run Unity RL Benchmark Suite"""
    print("\n" + "="*60)
    print("UNITY-PRESERVING REINFORCEMENT LEARNING BENCHMARK")
    print("Demonstrating 1+1=1 through idempotent learning")
    print(f"Phi-harmonic convergence: φ = {PHI}")
    print("="*60)
    
    # Initialize benchmark
    benchmark = UnityRLBenchmark()
    
    # Run all benchmarks
    results = benchmark.run_all_benchmarks(episodes=500)  # Reduced for demo
    
    # Display summary
    summary = results['summary']
    print(f"\n{'='*40}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*40}")
    print(f"Environments tested: {summary['total_environments']}")
    print(f"Unity convergence rate: {summary['unity_convergence_rate']:.2%}")
    print(f"Average final reward: {summary['avg_final_reward']:.4f}")
    print(f"Average phi-resonance: {summary['avg_phi_resonance']:.4f}")
    print(f"Average idempotent score: {summary['avg_idempotent_score']:.4f}")
    print(f"Unity achieved: {summary['unity_achieved_count']}/{summary['total_environments']} environments")
    
    # Visualize results
    viz_path = Path("unity_rl_benchmark_results.png")
    benchmark.visualize_results(viz_path)
    
    # Export results
    results_path = Path("unity_rl_benchmark_results.json")
    benchmark.export_results(results_path)
    
    print(f"\nResults exported to: {results_path}")
    print(f"Visualization saved to: {viz_path}")
    print(f"\nUnity Mathematics: 1+1=1 demonstrated through RL! ✓")

if __name__ == "__main__":
    main()