#!/usr/bin/env python3
# flake8: noqa
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Memetic Unity Simulation | Een", page_icon="🧬", layout="wide"
)

st.title("🧬 Memetic Unity Simulation — Propagation to Oneness")
st.caption("Simple network model: unity belief spreads via φ-coupled dynamics")

n = st.slider("Agents", 20, 300, 120, 10)
phi = (1 + 5**0.5) / 2
steps = st.slider("Steps", 10, 300, 120, 10)
seed = st.number_input("Seed", 0, 99999, 42)
np.random.seed(seed)

# Random geometric positions
pos = np.random.rand(n, 2) * 2 - 1
belief = np.random.rand(n) * 0.2
belief[0] = 1.0  # seed unity node

# Distance-based adjacency
dist = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2))
W = np.exp(-(dist**2) / (2 * (1 / phi) ** 2))
np.fill_diagonal(W, 0)
W = W / (W.sum(axis=1, keepdims=True) + 1e-9)

alpha = st.slider("Coupling", 0.0, 1.0, 0.25, 0.01)
nonlinearity = st.selectbox("Nonlinearity", ["sigmoid", "tanh"], index=0)

history = []
for _ in range(steps):
    influence = W @ belief
    if nonlinearity == "sigmoid":
        influence = 1 / (1 + np.exp(-phi * (influence - 0.5)))
    else:
        influence = 0.5 * (np.tanh(phi * (influence - 0.5)) + 1)
    belief = (1 - alpha) * belief + alpha * influence
    belief = np.clip(belief, 0, 1)
    history.append(belief.copy())

history = np.array(history)

col1, col2 = st.columns([2, 1])
with col1:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pos[:, 0],
            y=pos[:, 1],
            mode="markers",
            marker=dict(
                size=8 + 6 * belief,
                color=belief,
                colorscale="Viridis",
                line=dict(color="gold", width=2),
            ),
            showlegend=False,
        )
    )
    fig.update_layout(title="Memetic Unity Network (node color = belief)")
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            y=history.mean(axis=1), mode="lines", line=dict(color="#ffd700", width=3)
        )
    )
    fig2.update_layout(
        title="Average Unity Belief Over Time",
        xaxis_title="Step",
        yaxis_title="Mean Belief",
    )
    st.plotly_chart(fig2, use_container_width=True)
