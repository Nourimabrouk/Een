#!/usr/bin/env python3
# flake8: noqa
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Consciousness Field | Een", page_icon="🧠", layout="wide"
)

st.title("🧠 Consciousness Field — φ² × ρ × U")
st.caption("Metagamer energy visualization and φ-harmonic dynamics")

phi = (1 + 5**0.5) / 2

col1, col2 = st.columns([1, 3])
with col1:
    st.subheader("Parameters")
    density = st.slider("Consciousness Density (ρ)", 0.0, 1.0, 0.5, 0.01)
    unity_rate = st.slider("Unity Convergence Rate (U)", 0.0, 1.0, 0.618, 0.001)
    freq = st.slider("Spatial Frequency", 0.1, 4.0, 1.0, 0.1)
    t = st.slider("Time", 0.0, 6.28, 0.0, 0.01)


def field_value(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    return phi * np.sin(phi * x + t) * np.cos(phi * y - t) * np.exp(-t / phi)


with col2:
    x = np.linspace(-np.pi, np.pi, 180)
    y = np.linspace(-np.pi, np.pi, 180)
    X, Y = np.meshgrid(x * freq, y * freq)
    Z = field_value(X, Y, t)
    energy = (phi**2) * density * unity_rate

    fig = go.Figure(
        data=go.Heatmap(
            z=Z, x=x, y=y, colorscale="Viridis", colorbar=dict(title="Field")
        )
    )
    fig.update_layout(
        title=f"Consciousness Field (E = φ² × ρ × U = {energy:.3f})",
        xaxis_title="x",
        yaxis_title="y",
    )
    st.plotly_chart(fig, use_container_width=True)

st.info("Tip: Animate by moving the Time slider.")
