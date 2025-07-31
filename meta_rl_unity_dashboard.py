import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Iterable


"""Meta Reinforcement Learning Unity Dashboard

This module implements an expanded conceptual playground for examining how
adversarial agents can be trained to cooperate. The environment is a flexible
grid world with optional obstacles and varying target locations. Agents are
trained via Q-learning across a variety of tasks. Meta-learning aggregates
knowledge from different tasks so that the policies adapt quickly to new ones.

The Streamlit dashboard visualizes agent paths on a world map and highlights
how the agents learn to converge. Additional charts display episode rewards and
a heatmap of visited locations, providing a more comprehensive overview of the
learning dynamics.
"""

# ============================
# Environment Implementation
# ============================

@dataclass
class UnityEnv:
    """2D grid world with configurable obstacles and target.

    The goal for both agents is to occupy the same cell as the target. If they
    collide with each other before reaching the target, they are penalized.
    """

    size: int = 10
    max_steps: int = 100
    obstacles: List[Tuple[int, int]] = field(default_factory=list)
    target: Tuple[int, int] = (0, 0)

    def reset(self) -> Tuple[int, int, int, int]:
        """Reset the environment to a random state that avoids obstacles."""
        def sample_pos():
            while True:
                pos = np.random.randint(0, self.size, size=2)
                if tuple(pos) not in self.obstacles and tuple(pos) != self.target:
                    return pos

        self.pos1 = sample_pos()
        self.pos2 = sample_pos()
        self.steps = 0
        return (*self.pos1, *self.pos2)

    def _clamp(self, coord: int) -> int:
        return int(max(0, min(self.size - 1, coord)))

    def _move(self, pos: np.ndarray, action: int) -> np.ndarray:
        x, y = pos
        if action == 0:  # up
            y += 1
        elif action == 1:  # down
            y -= 1
        elif action == 2:  # right
            x += 1
        elif action == 3:  # left
            x -= 1
        new_pos = np.array([self._clamp(x), self._clamp(y)])
        if tuple(new_pos) in self.obstacles:
            return pos  # bump into obstacle
        return new_pos

    def step(self, a1: int, a2: int) -> Tuple[Tuple[int, int, int, int], float, bool]:
        """Advance the environment by one step."""
        self.pos1 = self._move(self.pos1, a1)
        self.pos2 = self._move(self.pos2, a2)
        self.steps += 1

        collide = np.array_equal(self.pos1, self.pos2)
        reached_target1 = np.array_equal(self.pos1, self.target)
        reached_target2 = np.array_equal(self.pos2, self.target)

        reward = -0.1  # small step penalty
        if collide:
            reward -= 2  # heavy penalty for colliding
        if reached_target1 and reached_target2:
            reward += 5
        elif reached_target1 or reached_target2:
            reward += 2

        done = collide or (reached_target1 and reached_target2) or self.steps >= self.max_steps
        return (*self.pos1, *self.pos2), reward, done

    def available_states(self) -> Iterable[Tuple[int, int, int, int]]:
        """Generate all valid joint states (used for initialization)."""
        coords = [(x, y) for x in range(self.size) for y in range(self.size)
                  if (x, y) not in self.obstacles and (x, y) != self.target]
        for p1 in coords:
            for p2 in coords:
                yield (*p1, *p2)


# ============================
# Q-Learning Agents
# ============================

ACTIONS = [0, 1, 2, 3, 4]  # up, down, right, left, stay

def clamp_action(action: int) -> int:
    return int(action) if action in ACTIONS else 4


class QLearningAgent:
    """Standard Q-learning agent."""

    def __init__(self, env: UnityEnv, lr: float = 0.1, gamma: float = 0.9, epsilon: float = 0.2):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q: Dict[Tuple[int, int, int, int], np.ndarray] = {}

    def _ensure_state(self, state: Tuple[int, int, int, int]):
        if state not in self.q:
            self.q[state] = np.zeros(len(ACTIONS))

    def choose(self, state: Tuple[int, int, int, int], explore: bool = True) -> int:
        self._ensure_state(state)
        if explore and np.random.rand() < self.epsilon:
            return np.random.choice(ACTIONS)
        return int(np.argmax(self.q[state]))

    def learn(self, state: Tuple[int, int, int, int], action: int, reward: float,
              next_state: Tuple[int, int, int, int]):
        self._ensure_state(state)
        self._ensure_state(next_state)
        q_state = self.q[state][action]
        q_next_max = np.max(self.q[next_state])
        self.q[state][action] += self.lr * (reward + self.gamma * q_next_max - q_state)


class MetaQLearningAgent:
    """Aggregates multiple Q-tables from different tasks for rapid adaptation."""

    def __init__(self, env: UnityEnv, tasks: List[Tuple[int, int]], meta_lr: float = 0.05):
        self.env = env
        self.tasks = tasks
        self.meta_lr = meta_lr
        self.base_agent = QLearningAgent(env)
        self.meta_q: Dict[Tuple[int, int, int, int], np.ndarray] = {}

    def _ensure_meta(self, state: Tuple[int, int, int, int]):
        if state not in self.meta_q:
            self.meta_q[state] = np.zeros(len(ACTIONS))

    def meta_update(self, q_tables: List[Dict[Tuple[int, int, int, int], np.ndarray]]):
        """Combine multiple Q-tables into a single meta table."""
        for table in q_tables:
            for state, q_values in table.items():
                self._ensure_meta(state)
                self.meta_q[state] += self.meta_lr * (q_values - self.meta_q[state])

    def adapt(self):
        """Copy meta Q-values into the base agent for fast adaptation."""
        for state, q_values in self.meta_q.items():
            self.base_agent.q[state] = q_values.copy()

    def train_on_task(self, target: Tuple[int, int], episodes: int = 20):
        """Train the base agent on a single target location."""
        self.env.target = target
        q_before = {s: q.copy() for s, q in self.base_agent.q.items()}
        for _ in range(episodes):
            simulate_episode(self.env, self.base_agent, self.base_agent, explore=True)
        q_after = self.base_agent.q
        delta_table = {}
        for state in q_after:
            before = q_before.get(state, np.zeros(len(ACTIONS)))
            delta_table[state] = q_after[state] - before
        return delta_table

    def meta_train(self, episodes: int = 20):
        delta_tables = []
        for target in self.tasks:
            delta = self.train_on_task(target, episodes)
            delta_tables.append(delta)
        self.meta_update(delta_tables)
        self.adapt()


# ============================
# Simulation Helpers
# ============================

def grid_to_geo(path: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Map grid coordinates to lat/lon for visualization on a world map."""
    lat = (path[:, 1] / (size - 1)) * 120 - 60  # scale to [-60, 60]
    lon = (path[:, 0] / (size - 1)) * 360 - 180  # scale to [-180, 180]
    return lat, lon


def simulate_episode(env: UnityEnv, agent1: QLearningAgent, agent2: QLearningAgent,
                     explore: bool = True) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run a single episode and return paths and cumulative reward."""
    state = env.reset()
    path1 = [env.pos1.copy()]
    path2 = [env.pos2.copy()]
    done = False
    total_reward = 0.0
    while not done:
        a1 = agent1.choose(state, explore)
        a2 = agent2.choose(state, explore)
        next_state, reward, done = env.step(a1, a2)
        if explore:
            agent1.learn(state, a1, reward, next_state)
            agent2.learn(state, a2, reward, next_state)
        state = next_state
        total_reward += reward
        path1.append(env.pos1.copy())
        path2.append(env.pos2.copy())
    return np.array(path1), np.array(path2), total_reward


def run_training(env: UnityEnv, agent1: QLearningAgent, agent2: QLearningAgent,
                 episodes: int) -> List[float]:
    rewards = []
    for _ in range(episodes):
        _, _, ep_reward = simulate_episode(env, agent1, agent2, explore=True)
        rewards.append(ep_reward)
    return rewards


# ============================
# Visualization Helpers
# ============================

def world_map_figure(paths: List[np.ndarray], env: UnityEnv, names: List[str], colors: List[str]) -> go.Figure:
    """Create a plotly map figure showing the agents' trajectories."""
    fig = go.Figure()
    fig.update_layout(mapbox_style="open-street-map", mapbox_zoom=1, margin=dict(l=0, r=0, t=0, b=0))
    for path, name, color in zip(paths, names, colors):
        lat, lon = grid_to_geo(path, env.size)
        fig.add_trace(
            go.Scattermapbox(lat=lat, lon=lon, mode="markers+lines", name=name,
                             marker=dict(size=8, color=color))
        )
    if env.obstacles:
        ox, oy = zip(*env.obstacles)
        lat_o, lon_o = grid_to_geo(np.column_stack([ox, oy]), env.size)
        fig.add_trace(
            go.Scattermapbox(lat=lat_o, lon=lon_o, mode="markers", name="Obstacles",
                             marker=dict(size=12, color="black", symbol="x"))
        )
    tx, ty = env.target
    lat_t, lon_t = grid_to_geo(np.array([[tx, ty]]), env.size)
    fig.add_trace(
        go.Scattermapbox(lat=lat_t, lon=lon_t, mode="markers", name="Target",
                         marker=dict(size=15, color="green"))
    )
    return fig


def reward_chart(rewards: List[float]) -> go.Figure:
    """Plot episode rewards over time."""
    fig = px.line(y=rewards, labels={"y": "Reward", "index": "Episode"})
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
    return fig


def visit_heatmap(paths: List[np.ndarray], env: UnityEnv) -> go.Figure:
    """Display a heatmap of visited cells by both agents."""
    grid = np.zeros((env.size, env.size))
    for path in paths:
        for x, y in path:
            grid[x, y] += 1
    fig = px.imshow(grid.T, origin="lower", color_continuous_scale="Viridis")
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
    return fig


# ============================
# Streamlit Dashboard
# ============================

st.set_page_config(page_title="Meta RL Unity Playground", layout="wide")

st.title("🌐 Meta Reinforcement Learning Unity Playground")

st.markdown(
    """
    This playground demonstrates how simple adversarial agents can learn to
    cooperate through meta-learning. Each episode places the agents in a grid
    world with obstacles and a shared target. Over many tasks, a meta-agent
    extracts common knowledge so that it can solve new tasks more efficiently.
    """
)

with st.sidebar:
    st.header("Environment Settings")
    grid_size = st.slider("Grid size", 5, 20, 10)
    max_steps = st.slider("Max steps per episode", 50, 200, 100)
    obstacle_ratio = st.slider("Obstacle ratio", 0.0, 0.3, 0.1, step=0.05)
    train_episodes = st.slider("Training episodes", 10, 200, 50)
    meta_tasks = st.slider("Number of meta-training tasks", 2, 10, 3)

# Prepare environment and tasks

def generate_obstacles(size: int, ratio: float) -> List[Tuple[int, int]]:
    total = int(size * size * ratio)
    obstacles = set()
    while len(obstacles) < total:
        ox = np.random.randint(0, size)
        oy = np.random.randint(0, size)
        if (ox, oy) != (0, 0):
            obstacles.add((ox, oy))
    return list(obstacles)

obstacles = generate_obstacles(grid_size, obstacle_ratio)

def random_target(size: int, obstacles: List[Tuple[int, int]]) -> Tuple[int, int]:
    while True:
        tx = np.random.randint(0, size)
        ty = np.random.randint(0, size)
        if (tx, ty) not in obstacles:
            return (tx, ty)

train_targets = [random_target(grid_size, obstacles) for _ in range(meta_tasks)]

# Initialize environment and agents

env = UnityEnv(size=grid_size, max_steps=max_steps, obstacles=obstacles, target=train_targets[0])
agent1 = QLearningAgent(env)
agent2 = QLearningAgent(env)
meta_agent = MetaQLearningAgent(env, tasks=train_targets)

# Training button

if st.button("Run Meta Training"):
    rewards = []
    for target in train_targets:
        env.target = target
        rewards.extend(run_training(env, agent1, agent2, train_episodes))
    # Perform meta-training update
    meta_agent.meta_train(episodes=train_episodes)
    st.subheader("Training Reward")
    st.plotly_chart(reward_chart(rewards), use_container_width=True)

# Evaluation

eval_target = random_target(grid_size, obstacles)
env.target = eval_target

if st.button("Evaluate Meta Policy"):
    meta_agent.adapt()
    p1, p2, _ = simulate_episode(env, meta_agent.base_agent, meta_agent.base_agent, explore=False)
    fig_map = world_map_figure([p1, p2], env, ["Agent 1", "Agent 2"], ["red", "blue"])
    heat = visit_heatmap([p1, p2], env)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Agent Paths")
        st.plotly_chart(fig_map, use_container_width=True)
    with col2:
        st.subheader("Visit Heatmap")
        st.plotly_chart(heat, use_container_width=True)
    st.success("Evaluation complete. Target located at {}.".format(eval_target))

