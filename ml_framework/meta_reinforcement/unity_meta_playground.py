import os

import gymnasium as gym
import numpy as np
import openai
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium import spaces
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from core.unity_mathematics import UnityMathematics

unity_math = UnityMathematics()
PHI = unity_math.phi


class UnityMetaEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 32):
        super().__init__()
        self.max_steps = max_steps
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-4.0, high=4.0, shape=(3,), dtype=np.float32)
        self.reset()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.steps = 0
        state = np.array([1.0, 1.0, PHI], dtype=np.float32)
        self.state = unity_math.phi_harmonic_transform(state)
        return self.state, {}

    def step(self, action):
        self.steps += 1
        a = float(action[0])
        target = unity_math.unity_add(1.0, 1.0)
        reward = -abs(a - target)
        terminated = self.steps >= self.max_steps
        self.state = unity_math.phi_harmonic_transform(self.state)
        return self.state, reward, terminated, False, {}


class MetaBuffer:
    def __init__(self, size: int = 2048):
        self.size = size
        self.obs, self.acts, self.rews = [], [], []

    def add(self, o, a, r):
        self.obs.append(o)
        self.acts.append(a)
        self.rews.append(r)
        if len(self.obs) > self.size:
            self.obs.pop(0)
            self.acts.pop(0)
            self.rews.pop(0)

    def sample(self, n: int = 32):
        idx = np.random.choice(len(self.obs), n)
        o = torch.tensor(np.array([self.obs[i] for i in idx]), dtype=torch.float32)
        a = torch.tensor(np.array([self.acts[i] for i in idx]), dtype=torch.float32)
        r = torch.tensor(np.array([self.rews[i] for i in idx]), dtype=torch.float32)
        return o, a, r


class ExpertBase(nn.Module):
    def select(self, obs: np.ndarray) -> float:
        raise NotImplementedError

    def update(self, _):
        pass


class RLExpert(ExpertBase):
    def __init__(self, algo, env_fn):
        super().__init__()
        self.model = algo("MlpPolicy", env_fn(), verbose=0)

    def select(self, obs: np.ndarray) -> float:
        a, _ = self.model.predict(obs, deterministic=True)
        return float(a[0])

    def update(self, _):
        self.model.learn(total_timesteps=64)


class OpenAIExpert(ExpertBase):
    def __init__(self, model: str = "gpt-4.1-mini"):
        super().__init__()
        openai.api_key = os.getenv("OPENAI_API_KEY", "")
        self.model = model

    def select(self, obs: np.ndarray) -> float:
        if not openai.api_key:
            return 1.0
        m = [{"role": "user", "content": f"state={obs.tolist()}"}]
        r = openai.ChatCompletion.create(model=self.model, messages=m, max_tokens=5)
        try:
            return float(r["choices"][0]["message"]["content"].strip())
        except Exception:
            return 1.0


class UnityGating(nn.Module):
    def __init__(self, n: int, d: int):
        super().__init__()
        self.fc = nn.Linear(d, n)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)


class UnityMixture:
    def __init__(self, experts, d):
        self.experts = experts
        self.gate = UnityGating(len(experts), d)
        self.opt = optim.Adam(self.gate.parameters(), lr=1e-3)

    def act(self, obs: np.ndarray) -> float:
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            w = self.gate(x)[0]
        acts = [torch.tensor([e.select(obs)], dtype=torch.float32) for e in self.experts]
        a = torch.stack(acts)[:, 0]
        return float((w * a).sum())

    def update(self, replay: MetaBuffer):
        if len(replay.obs) < 32:
            return
        o, a, r = replay.sample(32)
        w = self.gate(o)
        a_hat = torch.stack(
            [
                torch.tensor([e.select(x.numpy()) for x in o], dtype=torch.float32)
                for e in self.experts
            ],
            dim=-1,
        )
        loss = F.mse_loss((w * a_hat).sum(dim=-1), a) * r.abs().mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


class MetaUnityAgent:
    def __init__(self, env: UnityMetaEnv):
        self.env = env
        self.buffer = MetaBuffer()
        self.experts = [
            RLExpert(PPO, lambda: DummyVecEnv([lambda: UnityMetaEnv()])),
            RLExpert(DQN, lambda: DummyVecEnv([lambda: UnityMetaEnv()])),
            RLExpert(A2C, lambda: DummyVecEnv([lambda: UnityMetaEnv()])),
            OpenAIExpert(),
        ]
        self.mixture = UnityMixture(self.experts, env.observation_space.shape[0])
        self.logs = []

    def train(self, episodes: int = 8):
        for _ in range(episodes):
            obs, _ = self.env.reset()
            done = False
            total = 0.0
            while not done:
                act = self.mixture.act(obs)
                new_obs, reward, terminated, truncated, _ = self.env.step(
                    np.array([act], dtype=np.float32)
                )
                self.buffer.add(obs, act, reward)
                for e in self.experts[:-1]:
                    e.update(self.buffer)
                self.mixture.update(self.buffer)
                obs = new_obs
                total += reward
                done = terminated or truncated
            self.logs.append(total)

    def policy(self, obs: np.ndarray) -> float:
        return self.mixture.act(obs)


if __name__ == "__main__":
    env = UnityMetaEnv()
    agent = MetaUnityAgent(env)
    agent.train(episodes=4)
