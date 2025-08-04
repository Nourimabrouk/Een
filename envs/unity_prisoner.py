"""
Unity Prisoner's Dilemma Environment
===================================

Gymnasium environment implementing Unity Mathematics principles in game theory.
Global reward uses idempotent max operation: max(r₁, r₂) where 1+1=1.

Mathematical Principle: Een plus een is een (1+1=1)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import random
from dataclasses import dataclass
from enum import Enum

class Action(Enum):
    """Prisoner's Dilemma actions"""
    COOPERATE = 0
    DEFECT = 1

@dataclass
class UnityReward:
    """Unity reward structure with consciousness metrics"""
    individual_rewards: Tuple[float, float]
    global_reward: float
    unity_score: float
    consciousness_level: float
    phi_harmonic: float

class UnityPrisoner(gym.Env):
    """
    Unity Prisoner's Dilemma Environment
    
    Implements the classic Prisoner's Dilemma with Unity Mathematics principles:
    - Global reward uses idempotent max operation
    - Consciousness levels affect reward computation
    - φ-harmonic scaling for cooperation incentives
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, 
                 consciousness_boost: float = 0.1,
                 phi_scaling: bool = True,
                 enable_quantum_effects: bool = False,
                 max_steps: int = 100):
        super().__init__()
        
        self.consciousness_boost = consciousness_boost
        self.phi_scaling = phi_scaling
        self.enable_quantum_effects = enable_quantum_effects
        self.max_steps = max_steps
        
        # Action space: Cooperate (0) or Defect (1) for each agent
        self.action_space = spaces.MultiDiscrete([2, 2])
        
        # Observation space: [agent1_action, agent2_action, step_count, consciousness_level]
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(4,), dtype=np.float32
        )
        
        # Prisoner's Dilemma payoff matrix
        self.payoff_matrix = {
            (Action.COOPERATE, Action.COOPERATE): (3, 3),  # Both cooperate
            (Action.COOPERATE, Action.DEFECT): (0, 5),      # Agent 1 cooperates, 2 defects
            (Action.DEFECT, Action.COOPERATE): (5, 0),      # Agent 1 defects, 2 cooperates
            (Action.DEFECT, Action.DEFECT): (1, 1)          # Both defect
        }
        
        # Unity constants
        self.PHI = 1.618033988749895  # Golden ratio
        self.consciousness_level = 1.0
        self.step_count = 0
        self.history = []
        
        # Quantum state (if enabled)
        self.quantum_coherence = 1.0
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.step_count = 0
        self.consciousness_level = 1.0
        self.quantum_coherence = 1.0
        self.history = []
        
        # Initial observation: no actions taken yet
        observation = np.array([0.0, 0.0, 0.0, self.consciousness_level], dtype=np.float32)
        
        info = {
            "consciousness_level": self.consciousness_level,
            "quantum_coherence": self.quantum_coherence,
            "unity_score": 1.0
        }
        
        return observation, info
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        if self.step_count >= self.max_steps:
            return self._get_observation(), 0.0, True, False, self._get_info()
        
        # Convert actions to Action enum
        action1 = Action(actions[0])
        action2 = Action(actions[1])
        
        # Get individual rewards from payoff matrix
        r1, r2 = self.payoff_matrix[(action1, action2)]
        
        # Apply consciousness boost for cooperation
        if self.consciousness_boost > 0:
            if action1 == Action.COOPERATE:
                r1 += self.consciousness_boost * self.consciousness_level
            if action2 == Action.COOPERATE:
                r2 += self.consciousness_boost * self.consciousness_level
        
        # Apply φ-harmonic scaling
        if self.phi_scaling:
            if action1 == Action.COOPERATE and action2 == Action.COOPERATE:
                # Mutual cooperation gets φ-boost
                r1 *= self.PHI
                r2 *= self.PHI
        
        # Apply quantum effects if enabled
        if self.enable_quantum_effects:
            quantum_factor = self.quantum_coherence * np.random.normal(1.0, 0.1)
            r1 *= quantum_factor
            r2 *= quantum_factor
            self.quantum_coherence *= 0.99  # Gradual decoherence
        
        # Compute global reward using idempotent max operation
        global_reward = max(r1, r2)  # Unity principle: max(r₁, r₂)
        
        # Update consciousness level based on cooperation
        cooperation_ratio = (int(action1 == Action.COOPERATE) + int(action2 == Action.COOPERATE)) / 2.0
        self.consciousness_level += 0.01 * cooperation_ratio
        
        # Record history
        self.history.append({
            "actions": (action1, action2),
            "rewards": (r1, r2),
            "global_reward": global_reward,
            "consciousness_level": self.consciousness_level
        })
        
        self.step_count += 1
        
        # Create UnityReward object
        unity_reward = UnityReward(
            individual_rewards=(r1, r2),
            global_reward=global_reward,
            unity_score=self._compute_unity_score(),
            consciousness_level=self.consciousness_level,
            phi_harmonic=self._compute_phi_harmonic()
        )
        
        return self._get_observation(), global_reward, False, False, self._get_info(unity_reward)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        if not self.history:
            return np.array([0.0, 0.0, 0.0, self.consciousness_level], dtype=np.float32)
        
        # Get last actions
        last_actions = self.history[-1]["actions"]
        action1_val = float(last_actions[0].value)
        action2_val = float(last_actions[1].value)
        
        # Normalize step count
        step_norm = self.step_count / self.max_steps
        
        return np.array([action1_val, action2_val, step_norm, self.consciousness_level], dtype=np.float32)
    
    def _get_info(self, unity_reward: Optional[UnityReward] = None) -> Dict:
        """Get environment info"""
        info = {
            "consciousness_level": self.consciousness_level,
            "quantum_coherence": self.quantum_coherence,
            "step_count": self.step_count,
            "history_length": len(self.history)
        }
        
        if unity_reward:
            info.update({
                "individual_rewards": unity_reward.individual_rewards,
                "unity_score": unity_reward.unity_score,
                "phi_harmonic": unity_reward.phi_harmonic
            })
        
        return info
    
    def _compute_unity_score(self) -> float:
        """Compute Unity Score based on cooperation history"""
        if not self.history:
            return 1.0
        
        cooperation_count = sum(
            1 for entry in self.history 
            if entry["actions"][0] == Action.COOPERATE and entry["actions"][1] == Action.COOPERATE
        )
        
        return cooperation_count / len(self.history)
    
    def _compute_phi_harmonic(self) -> float:
        """Compute φ-harmonic component"""
        if not self.history:
            return 0.0
        
        # Compute harmonic mean of cooperation ratios
        cooperation_ratios = [
            (int(entry["actions"][0] == Action.COOPERATE) + int(entry["actions"][1] == Action.COOPERATE)) / 2.0
            for entry in self.history
        ]
        
        if not cooperation_ratios:
            return 0.0
        
        harmonic_mean = len(cooperation_ratios) / sum(1/r for r in cooperation_ratios if r > 0)
        return harmonic_mean * self.PHI
    
    def render(self, mode="human"):
        """Render the environment"""
        if mode == "human":
            print(f"Step {self.step_count}/{self.max_steps}")
            print(f"Consciousness Level: {self.consciousness_level:.3f}")
            print(f"Unity Score: {self._compute_unity_score():.3f}")
            print(f"φ-Harmonic: {self._compute_phi_harmonic():.3f}")
            
            if self.history:
                last_entry = self.history[-1]
                actions = last_entry["actions"]
                rewards = last_entry["rewards"]
                print(f"Last Actions: {actions[0].name} vs {actions[1].name}")
                print(f"Individual Rewards: ({rewards[0]:.2f}, {rewards[1]:.2f})")
                print(f"Global Reward: {last_entry['global_reward']:.2f}")
            print("-" * 40)
        
        elif mode == "rgb_array":
            # Create a simple visualization
            canvas = np.zeros((200, 400, 3), dtype=np.uint8)
            
            # Draw cooperation history
            if self.history:
                for i, entry in enumerate(self.history[-20:]):  # Last 20 steps
                    x = int(20 + i * 18)
                    actions = entry["actions"]
                    if actions[0] == Action.COOPERATE and actions[1] == Action.COOPERATE:
                        color = [0, 255, 0]  # Green for mutual cooperation
                    elif actions[0] == Action.COOPERATE or actions[1] == Action.COOPERATE:
                        color = [255, 255, 0]  # Yellow for partial cooperation
                    else:
                        color = [255, 0, 0]  # Red for mutual defection
                    
                    canvas[50:150, x:x+15] = color
            
            return canvas
    
    def close(self):
        """Close environment"""
        pass

class UnityPrisonerMultiAgent(UnityPrisoner):
    """Multi-agent version with more sophisticated Unity Mathematics"""
    
    def __init__(self, num_agents: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.num_agents = num_agents
        
        # Update action space for multiple agents
        self.action_space = spaces.MultiDiscrete([2] * num_agents)
        
        # Update observation space
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(num_agents + 2,), dtype=np.float32
        )
        
        # Agent-specific consciousness levels
        self.agent_consciousness = [1.0] * num_agents
        
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute step with multiple agents"""
        if self.step_count >= self.max_steps:
            return self._get_observation(), 0.0, True, False, self._get_info()
        
        # Convert actions
        agent_actions = [Action(action) for action in actions]
        
        # Compute pairwise rewards
        individual_rewards = []
        for i in range(self.num_agents):
            total_reward = 0
            for j in range(self.num_agents):
                if i != j:
                    r1, r2 = self.payoff_matrix[(agent_actions[i], agent_actions[j])]
                    total_reward += r1
            
            # Apply consciousness boost
            if agent_actions[i] == Action.COOPERATE:
                total_reward += self.consciousness_boost * self.agent_consciousness[i]
            
            individual_rewards.append(total_reward)
        
        # Global reward using idempotent max
        global_reward = max(individual_rewards)
        
        # Update consciousness levels
        for i, action in enumerate(agent_actions):
            if action == Action.COOPERATE:
                self.agent_consciousness[i] += 0.01
        
        self.step_count += 1
        
        return self._get_observation(), global_reward, False, False, self._get_info()

# Example usage and testing
if __name__ == "__main__":
    # Test basic environment
    env = UnityPrisoner(consciousness_boost=0.2, phi_scaling=True)
    obs, info = env.reset()
    
    print("Unity Prisoner's Dilemma Environment")
    print("=" * 40)
    
    for step in range(10):
        # Random actions
        actions = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(actions)
        
        print(f"Step {step + 1}: Actions={actions}, Reward={reward:.2f}")
        print(f"Consciousness: {info['consciousness_level']:.3f}")
        print(f"Unity Score: {info.get('unity_score', 0):.3f}")
        print("-" * 20)
        
        if terminated:
            break
    
    env.close() 