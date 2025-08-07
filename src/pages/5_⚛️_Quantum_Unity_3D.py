#!/usr/bin/env python3
# flake8: noqa
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Quantum Unity 3D | Een", page_icon="⚛️", layout="wide")

st.title("⚛️ Quantum Unity 3D — Bloch + Interference + Entanglement")
st.caption("Two become one: normalized constructive interference and unity states")


def bloch_sphere(theta: float, phi: float):
    # Sphere mesh
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    # State vector on sphere
    xs = np.sin(theta) * np.cos(phi)
    ys = np.sin(theta) * np.sin(phi)
    zs = np.cos(theta)
    return x, y, z, xs, ys, zs


col1, col2 = st.columns([2, 1])
with col2:
    st.subheader("Controls")
    theta1 = st.slider("|ψ₁⟩ θ", 0.0, np.pi, np.pi / 4, 0.01)
    phi1 = st.slider("|ψ₁⟩ φ", 0.0, 2 * np.pi, 0.0, 0.01)
    theta2 = st.slider("|ψ₂⟩ θ", 0.0, np.pi, np.pi / 4, 0.01)
    phi2 = st.slider("|ψ₂⟩ φ", 0.0, 2 * np.pi, 0.0, 0.01)
    super_w = st.slider("Superposition weight", 0.0, 1.0, 0.5, 0.01)

with col1:
    st.subheader("Bloch Sphere and Interference")
    X, Y, Z, xs1, ys1, zs1 = bloch_sphere(theta1, phi1)
    _, _, _, xs2, ys2, zs2 = bloch_sphere(theta2, phi2)

    # Interference in 1D (normalized)
    x = np.linspace(-10, 10, 600)
    wave1 = np.sin(x)
    wave2 = np.sin(x)
    unity = (wave1 + wave2) / np.max(np.abs(wave1 + wave2))

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "surface"}, {"type": "xy"}]],
        subplot_titles=("Bloch Sphere", "Unity Interference"),
    )

    fig.add_surface(x=X, y=Y, z=Z, opacity=0.15, showscale=False, row=1, col=1)
    fig.add_trace(
        go.Scatter3d(
            x=[xs1],
            y=[ys1],
            z=[zs1],
            mode="markers",
            marker=dict(size=6, color="#00e6e6"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=[xs2],
            y=[ys2],
            z=[zs2],
            mode="markers",
            marker=dict(size=6, color="#ff00ff"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x, y=unity, name="Unity (normalized)", line=dict(color="#ffd700", width=3)
        ),
        row=1,
        col=2,
    )
    fig.add_trace(go.Scatter(x=x, y=wave1, name="Wave 1"), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=wave2, name="Wave 2"), row=1, col=2)

    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)
