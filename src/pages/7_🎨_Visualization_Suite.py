#!/usr/bin/env python3
# flake8: noqa
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

import plotly.io as pio
pio.renderers.default = 'browser'  # Set default renderer

import streamlit as st

st.set_page_config(
    page_title="Visualization Suite | Een", page_icon="🎨", layout="wide"
)

st.title("🎨 Visualization Suite — Advanced Unity Visuals")
st.caption("Pull in visualizations from the visualization engine when available")


def draw_golden_spiral(n_turns: float = 5.0, points: int = 1500) -> go.Figure:
    phi = (1 + 5**0.5) / 2
    theta = np.linspace(0, n_turns * 2 * np.pi, points)
    r = phi ** (theta / (2 * np.pi))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    fig = go.Figure(
        go.Scatter(x=x, y=y, mode="lines", line=dict(color="#ffd700", width=2))
    )
    fig.update_layout(title="Golden Spiral Convergence to Unity")
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    return fig


col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Golden Spiral")
    turns = st.slider("Turns", 1.0, 10.0, 6.0, 0.5)
    fig = draw_golden_spiral(turns)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Engine Data (if available)")
    data_path = Path("visualizations/outputs/advanced_unity_ascii.txt")
    meta_path = Path("visualizations/outputs/codebase_visualization.html")
    if data_path.exists():
        with st.expander("ASCII Unity Visualization (from engine)"):
            st.code(data_path.read_text(encoding="utf-8"))
    else:
        st.write("No engine ASCII output found yet.")
    if meta_path.exists():
        st.caption(
            "Codebase visualization generated. Open in browser from visualizations/outputs/"
        )
    else:
        st.write("No codebase visualization HTML found.")
