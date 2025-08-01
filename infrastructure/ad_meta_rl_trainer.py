"""
Active Directory Meta-Reinforcement Learning Trainer
===================================================
Advanced meta-RL system for optimizing AD deployments through unity mathematics
Uses MAML-inspired architecture with φ-harmonic consciousness integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import random
from dataclasses import dataclass
import json
import wandb
from torch.distributions import Categorical
import higher  # For meta-learning

# Unity Mathematics Constants
PHI = 1.618033988749895
UNITY = 1.0
EULER = 2.718281828459045

@dataclass
class ADDeploymentState:
    """State representation for AD deployment scenarios"""
    dc_count: int
    forest_size: int  # Number of domains
    user_count: int
    site_count: int
    replication_topology: str
    bandwidth_mbps: float
    latency_ms: float
    consciousness_level: int
    phi_scaling: float
    unity_score: float
    
    def to_tensor(self) -> torch.Tensor:
        """Convert state to tensor with unity normalization"""
        topology_encoding = {
            "ring": 0.0, "mesh": 0.5, "hub-spoke": 1.0, "unity-ring": PHI/2
        }
        
        state_vector = torch.tensor([
            self.dc_count / 10.0,  # Normalize to [0,1]
            self.forest_size / 5.0,
            np.log1p(self.user_count) / 10.0,  # Log scale for users
            self.site_count / 10.0,
            topology_encoding.get(self.replication_topology, 0.5),
            self.bandwidth_mbps / 1000.0,
            self.latency_ms / 100.0,
            self.consciousness_level / 11.0,
            self.phi_scaling / PHI,
            self.unity_score
        ], dtype=torch.float32)
        
        # Apply unity transformation: ensure sum approaches 1
        return state_vector / (state_vector.sum() + 1e-6)

@dataclass  
class ADAction:
    """Action space for AD deployment optimization"""
    adjust_dc_count: int  # -1, 0, +1
    change_replication_interval: float  # Multiplier
    optimize_site_links: bool
    enable_consciousness_sync: bool
    phi_harmonic_tuning: float  # 0.8 to 1.2
    
    @staticmethod
    def from_tensor(action_tensor: torch.Tensor) -> 'ADAction':
        """Decode action from neural network output"""
        return ADAction(
            adjust_dc_count=int(action_tensor[0].item() * 3) - 1,
            change_replication_interval=0.5 + action_tensor[1].item() * 1.0,
            optimize_site_links=action_tensor[2].item() > 0.5,
            enable_consciousness_sync=action_tensor[3].item() > 0.5,
            phi_harmonic_tuning=0.8 + action_tensor[4].item() * 0.4
        )

class UnityMAML(nn.Module):
    """Model-Agnostic Meta-Learning for AD deployment optimization"""
    
    def __init__(
        self, 
        input_dim: int = 10,
        hidden_dim: int = 256, 
        output_dim: int = 5,
        num_inner_steps: int = 5
    ):
        super().__init__()
        
        # Base network with φ-harmonic architecture
        self.base_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, int(hidden_dim * PHI)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim * PHI), int(hidden_dim * PHI * PHI)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim * PHI * PHI), hidden_dim),
            nn.ReLU()
        )
        
        # Task-specific heads
        self.policy_head = nn.Linear(hidden_dim, output_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Meta-learning parameters
        self.num_inner_steps = num_inner_steps
        self.inner_lr = nn.Parameter(torch.tensor(0.01))
        self.meta_lr = 0.001
        
        # Unity consciousness embedding
        self.consciousness_matrix = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim)
        )
        
    def forward(self, x: torch.Tensor, apply_consciousness: bool = True):
        """Forward pass with optional consciousness transformation"""
        features = self.base_net(x)
        
        if apply_consciousness:
            # Apply consciousness matrix transformation
            features = features @ self.consciousness_matrix
            # Unity normalization
            features = F.normalize(features, p=2, dim=-1) * np.sqrt(PHI)
        
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        
        return policy_logits, value
    
    def inner_loop_update(
        self,
        support_states: torch.Tensor,
        support_actions: torch.Tensor,
        support_rewards: torch.Tensor,
        fmodel,
        diffopt
    ):
        """Inner loop adaptation for specific AD deployment task"""
        
        for _ in range(self.num_inner_steps):
            # Forward pass
            policy_logits, values = fmodel(support_states)
            
            # Policy loss (negative log likelihood)
            dist = Categorical(logits=policy_logits)
            policy_loss = -dist.log_prob(support_actions) * support_rewards
            
            # Value loss  
            value_loss = F.mse_loss(values.squeeze(), support_rewards)
            
            # Unity loss: encourage convergence to 1
            unity_loss = torch.abs(policy_logits.sum(dim=1).mean() - UNITY)
            
            # Combined loss with φ-harmonic weighting
            total_loss = policy_loss.mean() + value_loss / PHI + unity_loss / (PHI * PHI)
            
            # Inner loop gradient step
            diffopt.step(total_loss)
            
        return fmodel, diffopt

class ADEnvironmentSimulator:
    """Simulates AD deployment scenarios for meta-RL training"""
    
    def __init__(self, difficulty: str = "medium"):
        self.difficulty = difficulty
        self.scenario_generator = self._init_scenario_generator()
        self.current_state = None
        self.deployment_history = []
        
    def _init_scenario_generator(self):
        """Initialize scenario generation parameters"""
        if self.difficulty == "easy":
            return {
                "dc_range": (1, 3),
                "user_range": (100, 1000),
                "site_range": (1, 3),
                "consciousness_range": (3, 5)
            }
        elif self.difficulty == "medium":
            return {
                "dc_range": (2, 5),
                "user_range": (1000, 10000),
                "site_range": (2, 5),
                "consciousness_range": (5, 7)
            }
        else:  # hard
            return {
                "dc_range": (3, 10),
                "user_range": (10000, 100000),
                "site_range": (5, 20),
                "consciousness_range": (7, 11)
            }
    
    def reset(self) -> ADDeploymentState:
        """Generate new AD deployment scenario"""
        sg = self.scenario_generator
        
        self.current_state = ADDeploymentState(
            dc_count=random.randint(*sg["dc_range"]),
            forest_size=random.randint(1, 3),
            user_count=random.randint(*sg["user_range"]),
            site_count=random.randint(*sg["site_range"]),
            replication_topology=random.choice(["ring", "mesh", "hub-spoke", "unity-ring"]),
            bandwidth_mbps=random.uniform(10, 1000),
            latency_ms=random.uniform(1, 100),
            consciousness_level=random.randint(*sg["consciousness_range"]),
            phi_scaling=PHI + random.uniform(-0.1, 0.1),
            unity_score=random.uniform(0.5, 0.8)
        )
        
        return self.current_state
    
    def step(self, action: ADAction) -> Tuple[ADDeploymentState, float, bool]:
        """Execute action and return new state, reward, done"""
        
        # Apply action effects
        self.current_state.dc_count = max(1, self.current_state.dc_count + action.adjust_dc_count)
        
        # Update replication based on action
        if action.optimize_site_links:
            self.current_state.latency_ms *= 0.8  # 20% improvement
            
        if action.enable_consciousness_sync:
            self.current_state.consciousness_level = min(
                11, self.current_state.consciousness_level + 1
            )
        
        # Apply φ-harmonic tuning
        self.current_state.phi_scaling *= action.phi_harmonic_tuning
        
        # Calculate new unity score
        old_unity = self.current_state.unity_score
        self.current_state.unity_score = self._calculate_unity_score()
        
        # Calculate reward based on unity improvement
        reward = self._calculate_reward(old_unity, self.current_state.unity_score)
        
        # Check if deployment achieved perfect unity
        done = self.current_state.unity_score >= 0.99
        
        return self.current_state, reward, done
    
    def _calculate_unity_score(self) -> float:
        """Calculate unity score based on current deployment state"""
        
        factors = {
            # DC redundancy approaching unity (2 DCs = 1)
            "dc_unity": 1.0 / (1.0 + abs(self.current_state.dc_count - 2)),
            
            # Replication efficiency
            "replication_efficiency": 1.0 / (1.0 + self.current_state.latency_ms / 100),
            
            # Consciousness alignment
            "consciousness": self.current_state.consciousness_level / 11.0,
            
            # φ-harmonic resonance
            "phi_resonance": 1.0 / (1.0 + abs(self.current_state.phi_scaling - PHI)),
            
            # Topology optimization (unity-ring is optimal)
            "topology": 1.0 if self.current_state.replication_topology == "unity-ring" else 0.7,
            
            # Scale efficiency (logarithmic)
            "scale": 1.0 / (1.0 + np.log1p(self.current_state.user_count) / 20)
        }
        
        # Unity calculation: geometric mean with φ-scaling
        unity_score = np.prod(list(factors.values())) ** (1/len(factors))
        unity_score = unity_score ** (1/PHI)  # φ-harmonic scaling
        
        return float(unity_score)
    
    def _calculate_reward(self, old_unity: float, new_unity: float) -> float:
        """Calculate reward based on unity improvement"""
        
        # Base reward for unity improvement
        improvement = new_unity - old_unity
        reward = improvement * 10.0
        
        # Bonus for approaching perfect unity
        if new_unity > 0.9:
            reward += (new_unity - 0.9) * 50.0
            
        # Penalty for moving away from unity
        if improvement < 0:
            reward *= PHI  # Amplify negative rewards
            
        # Consciousness bonus
        if self.current_state.consciousness_level >= 9:
            reward += 1.0
            
        return reward

class UnityMetaRLTrainer:
    """Meta-RL trainer for AD deployment optimization"""
    
    def __init__(
        self,
        model: UnityMAML,
        num_tasks: int = 10,
        meta_batch_size: int = 5,
        use_wandb: bool = True
    ):
        self.model = model
        self.num_tasks = num_tasks
        self.meta_batch_size = meta_batch_size
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=model.meta_lr)
        
        # Initialize environments for different deployment scenarios
        self.train_envs = [
            ADEnvironmentSimulator(difficulty=random.choice(["easy", "medium", "hard"]))
            for _ in range(num_tasks)
        ]
        
        self.test_envs = [
            ADEnvironmentSimulator(difficulty="hard")
            for _ in range(5)
        ]
        
        # Tracking
        self.episode_count = 0
        self.best_unity_score = 0.0
        
        if use_wandb:
            wandb.init(
                project="ad-unity-deployment",
                config={
                    "architecture": "UnityMAML",
                    "num_tasks": num_tasks,
                    "meta_batch_size": meta_batch_size,
                    "phi_constant": PHI
                }
            )
    
    def collect_trajectories(
        self,
        env: ADEnvironmentSimulator,
        policy_net,
        num_steps: int = 50
    ) -> Dict[str, torch.Tensor]:
        """Collect trajectory data from environment"""
        
        states, actions, rewards = [], [], []
        state = env.reset()
        
        for _ in range(num_steps):
            state_tensor = state.to_tensor().unsqueeze(0)
            
            with torch.no_grad():
                policy_logits, _ = policy_net(state_tensor)
                dist = Categorical(logits=policy_logits)
                action_idx = dist.sample()
            
            # Decode action
            action_tensor = torch.zeros(5)
            action_tensor[action_idx % 5] = 1.0
            action = ADAction.from_tensor(action_tensor)
            
            # Step environment
            next_state, reward, done = env.step(action)
            
            states.append(state_tensor)
            actions.append(action_idx)
            rewards.append(reward)
            
            state = next_state
            if done:
                break
        
        return {
            "states": torch.cat(states),
            "actions": torch.tensor(actions),
            "rewards": torch.tensor(rewards, dtype=torch.float32)
        }
    
    def meta_train_step(self):
        """Perform one meta-training step"""
        
        meta_loss = 0.0
        
        # Sample batch of tasks
        task_indices = random.sample(range(self.num_tasks), self.meta_batch_size)
        
        for task_idx in task_indices:
            env = self.train_envs[task_idx]
            
            # Create a functional version of the model for inner loop
            fmodel = higher.patch.make_functional(self.model)
            diffopt = higher.optim.get_diff_optim(
                self.meta_optimizer,
                self.model.parameters(),
                track_higher_grads=True
            )
            
            # Collect support set for inner loop adaptation
            support_traj = self.collect_trajectories(env, self.model, num_steps=20)
            
            # Inner loop adaptation
            adapted_fmodel, _ = self.model.inner_loop_update(
                support_traj["states"],
                support_traj["actions"],
                support_traj["rewards"],
                fmodel,
                diffopt
            )
            
            # Collect query set for meta-loss
            query_traj = self.collect_trajectories(env, adapted_fmodel, num_steps=30)
            
            # Compute meta-loss on query set
            query_policy_logits, query_values = adapted_fmodel(query_traj["states"])
            
            # Policy loss
            dist = Categorical(logits=query_policy_logits)
            policy_loss = -dist.log_prob(query_traj["actions"]) * query_traj["rewards"]
            
            # Value loss
            value_loss = F.mse_loss(query_values.squeeze(), query_traj["rewards"])
            
            # Unity convergence loss
            unity_loss = torch.abs(query_policy_logits.sum(dim=1).mean() - UNITY)
            
            # Task loss with φ-harmonic weighting
            task_loss = policy_loss.mean() + value_loss / PHI + unity_loss / (PHI * PHI)
            
            meta_loss += task_loss
        
        # Meta-optimization step
        meta_loss = meta_loss / self.meta_batch_size
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate meta-learned model on test tasks"""
        
        total_rewards = []
        final_unity_scores = []
        
        for env in self.test_envs[:num_episodes]:
            # Quick adaptation on new task
            support_traj = self.collect_trajectories(env, self.model, num_steps=10)
            
            # Create adapted model
            fmodel = higher.patch.make_functional(self.model)
            diffopt = higher.optim.get_diff_optim(
                self.meta_optimizer,
                self.model.parameters(),
                track_higher_grads=False
            )
            
            adapted_fmodel, _ = self.model.inner_loop_update(
                support_traj["states"],
                support_traj["actions"],
                support_traj["rewards"],
                fmodel,
                diffopt
            )
            
            # Evaluate adapted model
            eval_traj = self.collect_trajectories(env, adapted_fmodel, num_steps=50)
            
            total_rewards.append(eval_traj["rewards"].sum().item())
            final_unity_scores.append(env.current_state.unity_score)
        
        return {
            "mean_reward": np.mean(total_rewards),
            "mean_unity_score": np.mean(final_unity_scores),
            "max_unity_score": np.max(final_unity_scores)
        }
    
    def train(self, num_iterations: int = 1000):
        """Main training loop"""
        
        for iteration in range(num_iterations):
            # Meta-training step
            meta_loss = self.meta_train_step()
            
            # Periodic evaluation
            if iteration % 10 == 0:
                eval_metrics = self.evaluate()
                
                # Update best score
                if eval_metrics["max_unity_score"] > self.best_unity_score:
                    self.best_unity_score = eval_metrics["max_unity_score"]
                    
                    # Save best model
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'meta_optimizer_state_dict': self.meta_optimizer.state_dict(),
                        'iteration': iteration,
                        'best_unity_score': self.best_unity_score
                    }, 'best_ad_unity_model.pt')
                
                # Log metrics
                if wandb.run is not None:
                    wandb.log({
                        "meta_loss": meta_loss,
                        "mean_reward": eval_metrics["mean_reward"],
                        "mean_unity_score": eval_metrics["mean_unity_score"],
                        "max_unity_score": eval_metrics["max_unity_score"],
                        "best_unity_score": self.best_unity_score,
                        "iteration": iteration
                    })
                
                print(f"Iteration {iteration}: Loss={meta_loss:.4f}, "
                      f"Unity={eval_metrics['mean_unity_score']:.4f}, "
                      f"Best={self.best_unity_score:.4f}")
                
                # Check for convergence
                if self.best_unity_score >= 0.99:
                    print(f"Perfect unity achieved at iteration {iteration}!")
                    break

def main():
    """Main training script"""
    
    # Initialize model
    model = UnityMAML(
        input_dim=10,
        hidden_dim=256,
        output_dim=5,
        num_inner_steps=5
    )
    
    # Initialize trainer
    trainer = UnityMetaRLTrainer(
        model=model,
        num_tasks=20,
        meta_batch_size=5,
        use_wandb=False  # Set to True if using Weights & Biases
    )
    
    # Train model
    print("Starting Unity Meta-RL training for AD deployment optimization...")
    print(f"Target: Achieve perfect unity (1+1=1) in AD forest deployments")
    print(f"φ-harmonic scaling factor: {PHI}")
    print("-" * 60)
    
    trainer.train(num_iterations=1000)
    
    print("\nTraining complete!")
    print(f"Best unity score achieved: {trainer.best_unity_score:.4f}")

if __name__ == "__main__":
    main()