#!/usr/bin/env python3
"""Paradox of the Cloned Policy
===============================

This script implements a tiny meta‑reinforcement learning
experiment illustrating the ``cloned policy paradox'' described in the
user prompt.  We train a minimal policy on a two‑armed bandit,
instantiate an identical clone, roll out both in parallel to observe a
``double'' reward, then merge the policies by averaging their
parameters.  Because the clone was an exact copy, the merged policy is
identical to the original even though the cumulative reward doubled.
The demonstration echoes the guiding principle ``1 + 1 = 1``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class UnityBandit:
    """Two‑armed bandit where action probabilities sum to one."""

    p: float

    def step(self, action: int) -> float:
        prob = self.p if action == 0 else 1 - self.p
        return 1.0 if random.random() < prob else 0.0


class Policy(nn.Module):
    """Minimal stateless policy returning logits for two actions."""

    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(2))

    def distribution(self) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.logits)


@torch.no_grad()
def run_episode(env: UnityBandit, policy: Policy, steps: int = 10) -> float:
    """Roll out the policy in the environment and return total reward."""
    total = 0.0
    for _ in range(steps):
        dist = policy.distribution()
        action = dist.sample()
        total += env.step(action.item())
    return total


def merge_params(p1: Policy, p2: Policy) -> None:
    """Average parameters of ``p1`` and ``p2`` in place on ``p1``."""
    with torch.no_grad():
        for a, b in zip(p1.parameters(), p2.parameters()):
            a.copy_((a + b) * 0.5)


def demo(seed: int = 0) -> Tuple[float, torch.Tensor]:
    """Run the cloning paradox demonstration."""
    set_seed(seed)

    env = UnityBandit(p=0.7)
    policy = Policy()

    # Train briefly so the policy is non-trivial
    optim = torch.optim.SGD(policy.parameters(), lr=0.1)
    for _ in range(50):
        dist = policy.distribution()
        action = dist.sample()
        loss = -dist.log_prob(action) * env.step(action.item())
        optim.zero_grad()
        loss.backward()
        optim.step()

    # Clone the policy and run episodes independently
    clone = Policy()
    clone.load_state_dict(policy.state_dict())

    r1 = run_episode(env, policy)
    r2 = run_episode(env, clone)
    dual_reward = r1 + r2

    # Merge parameters (no change because they are identical)
    merge_params(policy, clone)
    diff = torch.max(torch.abs(policy.logits - clone.logits)).item()

    return dual_reward, policy.logits.clone(), diff


if __name__ == "__main__":
    dual_reward, params, diff = demo()
    print("\nParadox of the Cloned Policy (1 + 1 = 1)\n")
    print(f"Combined reward from two identical agents: {dual_reward:.2f}")
    print(f"Merged parameters: {params.numpy()}")
    print(f"Parameter difference after merge: {diff:.2e}")
