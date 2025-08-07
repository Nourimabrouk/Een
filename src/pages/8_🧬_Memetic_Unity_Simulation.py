#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# flake8: noqa
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Memetic Unity Simulation | Een", page_icon="🧬", layout="wide"
)

st.title("🧬 Memetic Unity Simulation — Propagation to Oneness")
st.caption("Network virality, convergence dynamics, and meta-memetic resonance")

phi = (1 + 5**0.5) / 2

with st.sidebar:
    st.markdown("### Simulation Controls")
    n = st.slider("Agents", 50, 1000, 240, 10)
    steps = st.slider("Steps", 20, 1000, 300, 10)
    alpha = st.slider("Coupling", 0.0, 1.0, 0.25, 0.01)
    rewiring = st.slider("Rewiring p (small-world)", 0.0, 1.0, 0.05, 0.01)
    hubs = st.slider("Hub factor (scale-free)", 0.0, 1.0, 0.3, 0.01)
    stubborn = st.slider("Stubborn fraction", 0.0, 0.5, 0.05, 0.01)
    noise = st.slider("Noise", 0.0, 0.5, 0.02, 0.01)
    nonlinearity = st.selectbox("Nonlinearity", ["sigmoid", "tanh"], 0)
    seed = st.number_input("Seed", 0, 999999, 42)

np.random.seed(seed)

# Positions (2D for viz)
pos = np.random.rand(n, 2) * 2 - 1

# Distance matrix and base kernels
dist = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(axis=2))
W_geo = np.exp(-(dist**2) / (2 * (1 / phi) ** 2))
np.fill_diagonal(W_geo, 0)
W_er = (np.random.rand(n, n) < 0.02).astype(float)
W = (1 - rewiring) * W_geo + rewiring * W_er

# Preferential attachment-like hubs
degree = W.sum(axis=1)
hub_weights = (degree + 1e-6) ** (1 + 3 * hubs)
W = (W.T * hub_weights).T
W = W / (W.sum(axis=1, keepdims=True) + 1e-9)

# Initial beliefs and stubborn set
belief = np.random.rand(n) * 0.1
seeds_idx = np.random.choice(n, size=max(3, n // 50), replace=False)
belief[seeds_idx] = 1.0
stubborn_mask = np.zeros(n, dtype=bool)
if stubborn > 0:
    stubborn_idx = np.random.choice(n, size=int(stubborn * n), replace=False)
    stubborn_mask[stubborn_idx] = True

history = []
virality = []
for _ in range(steps):
    influence = W @ belief
    if nonlinearity == "sigmoid":
        influence = 1 / (1 + np.exp(-phi * (influence - 0.5)))
    else:
        influence = 0.5 * (np.tanh(phi * (influence - 0.5)) + 1)
    updated = (1 - alpha) * belief + alpha * influence
    updated = np.clip(updated + noise * (np.random.rand(n) - 0.5), 0, 1)
    belief[~stubborn_mask] = updated[~stubborn_mask]
    history.append(belief.copy())
    virality.append((belief > 0.9).mean())

history = np.array(history)

tab1, tab2, tab3 = st.tabs(["Network", "Dynamics", "Virality Map"])

with tab1:
    node_size = 6 + 10 * belief
    fig = px.scatter(
        x=pos[:, 0], y=pos[:, 1], color=belief, color_continuous_scale="Plasma"
    )
    fig.update_traces(marker=dict(size=node_size, line=dict(color="gold", width=1)))
    fig.update_layout(
        title="Memetic Unity Network (final beliefs)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    mean_belief = history.mean(axis=1)
    median_belief = np.median(history, axis=1)
    p90 = np.quantile(history, 0.9, axis=1)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(y=mean_belief, name="Mean", line=dict(color="#ffd700", width=3))
    )
    fig.add_trace(
        go.Scatter(y=median_belief, name="Median", line=dict(color="#00e6e6"))
    )
    fig.add_trace(go.Scatter(y=p90, name="90th %", line=dict(color="#ff00ff")))
    fig.add_trace(
        go.Scatter(y=virality, name=">0.9 fraction", line=dict(color="#22c55e"))
    )
    fig.update_layout(title="Memetic Dynamics", xaxis_title="Step", yaxis_title="Value")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    res = 20
    alphas = np.linspace(0.05, 0.9, res)
    rews = np.linspace(0.0, 0.6, res)
    Z = np.zeros((res, res))
    for i, a in enumerate(alphas):
        for j, r in enumerate(rews):
            local_W = (1 - r) * W_geo + r * W_er
            local_W = (local_W.T * hub_weights).T
            local_W = local_W / (local_W.sum(axis=1, keepdims=True) + 1e-9)
            b = np.random.rand(n) * 0.1
            b[seeds_idx] = 1.0
            for _ in range(60):
                inf = local_W @ b
                inf = 1 / (1 + np.exp(-phi * (inf - 0.5)))
                b = (1 - a) * b + a * inf
            Z[j, i] = (b > 0.9).mean()
    heat = go.Figure(data=go.Heatmap(z=Z, x=alphas, y=rews, colorscale="Viridis"))
    heat.update_layout(
        title="Virality Phase Map (final >0.9 fraction)",
        xaxis_title="Coupling α",
        yaxis_title="Rewiring p",
    )
    st.plotly_chart(heat, use_container_width=True)

st.markdown("### Math Notes")
st.latex(r"b_{t+1} = (1-\alpha)\,b_t + \alpha\,\sigma_\phi(Wb_t) + \eta_t")
st.latex(
    r"\sigma_\phi(x)=\frac{1}{1+e^{-\phi (x-0.5)}}\ \text{or}\ \frac{\tanh(\phi(x-0.5))+1}{2}"
)
