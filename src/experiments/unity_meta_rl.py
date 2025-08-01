"""Unity Meta Reinforcement Learning Engine
=========================================

This module implements a minimal meta‑reinforcement learning example in
PyTorch.  The goal is to illustrate how repeated experience across a
family of tasks can be synthesised into a single "unity" policy.

At the heart of the implementation is a simple two‑armed bandit.  Each
task is parameterised by a probability ``p`` that action ``0`` yields a
reward.  Action ``1`` then pays off with probability ``1-p`` so that the
total probability mass remains one – a gentle nod to the guiding mantra
``1 + 1 = 1``.  By training with Model‑Agnostic Meta‑Learning (MAML) we
demonstrate how an initial policy can quickly adapt to any bandit in this
unity family.

This example is intentionally compact, focusing on conceptual clarity.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UnityBandit:
    """Two‑armed bandit where probabilities sum to one."""

    p: float

    def step(self, action: int) -> float:
        if action not in (0, 1):
            raise ValueError("action must be 0 or 1")
        prob = self.p if action == 0 else 1 - self.p
        return 1.0 if random.random() < prob else 0.0


class BanditPolicy(nn.Module):
    """Stateless policy returning logits for the two actions."""

    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(2))

    def distribution(
        self, params: List[torch.Tensor] | None = None
    ) -> torch.distributions.Categorical:
        logits = self.logits if params is None else params[0]
        return torch.distributions.Categorical(logits=logits)


def run_episode(
    env: UnityBandit,
    policy: BanditPolicy,
    params: List[torch.Tensor] | None = None,
    steps: int = 10,
) -> torch.Tensor:
    log_probs: List[torch.Tensor] = []
    rewards: List[float] = []
    for _ in range(steps):
        dist = policy.distribution(params)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        reward = env.step(action.item())
        rewards.append(reward)
    returns = torch.tensor(rewards, dtype=torch.float32)
    return -(torch.stack(log_probs) * returns).mean()


def adapt(
    policy: BanditPolicy, env: UnityBandit, lr_inner: float
) -> List[torch.Tensor]:
    loss = run_episode(env, policy)
    grads = torch.autograd.grad(loss, policy.parameters(), create_graph=True)
    return [p - lr_inner * g for p, g in zip(policy.parameters(), grads)]


def meta_update(
    policy: BanditPolicy,
    tasks: List[UnityBandit],
    lr_inner: float,
    meta_optimizer: torch.optim.Optimizer,
) -> float:
    meta_optimizer.zero_grad()
    meta_loss = 0.0
    for task in tasks:
        adapted_params = adapt(policy, task, lr_inner)
        loss = run_episode(task, policy, adapted_params)
        meta_loss += loss
    meta_loss /= len(tasks)
    meta_loss.backward()
    meta_optimizer.step()
    return float(meta_loss.detach())


def sample_task() -> UnityBandit:
    p = random.uniform(0.1, 0.9)
    return UnityBandit(p=p)


def train(
    meta_iterations: int = 1000,
    meta_batch_size: int = 4,
    lr_inner: float = 0.5,
    lr_outer: float = 0.01,
) -> BanditPolicy:
    policy = BanditPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr_outer)
    for it in range(1, meta_iterations + 1):
        tasks = [sample_task() for _ in range(meta_batch_size)]
        loss = meta_update(policy, tasks, lr_inner, optimizer)
        if it % 100 == 0:
            print(f"Iteration {it:04d} \u2013 meta loss: {loss:.4f}")
    return policy


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    print("\nUnity Meta Reinforcement Learning Demo (1 + 1 = 1)\n")
    trained_policy = train(meta_iterations=300)
    test_env = UnityBandit(p=0.7)
    with torch.no_grad():
        dist = trained_policy.distribution()
        probs = dist.probs.numpy()
    print(f"Trained action probabilities: {probs}")
    reward = test_env.step(int(probs[1] > probs[0]))
    print(f"Test reward sample: {reward}")
