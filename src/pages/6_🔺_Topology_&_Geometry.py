#!/usr/bin/env python3
# flake8: noqa
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Topology & Geometry | Een", page_icon="🔺", layout="wide"
)

st.title("🔺 Topology & Geometry — Unity Manifolds and φ-Forms")
st.caption("Möbius, torus, and golden spirals as unity narratives")


def mobius_strip(nu: int = 120, nv: int = 30):
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(-0.3, 0.3, nv)
    U, V = np.meshgrid(u, v)
    X = (1 + V * np.cos(U / 2.0)) * np.cos(U)
    Y = (1 + V * np.cos(U / 2.0)) * np.sin(U)
    Z = V * np.sin(U / 2.0)
    return X, Y, Z


def torus(R=1.0, r=0.3, nu: int = 80, nv: int = 40):
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(0, 2 * np.pi, nv)
    U, V = np.meshgrid(u, v)
    X = (R + r * np.cos(V)) * np.cos(U)
    Y = (R + r * np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)
    return X, Y, Z


col1, col2 = st.columns(2)
with col1:
    st.subheader("Möbius Unity")
    X, Y, Z = mobius_strip()
    fig = go.Figure(
        data=[
            go.Surface(
                x=X, y=Y, z=Z, colorscale="Cividis", showscale=False, opacity=0.9
            )
        ]
    )
    fig.update_layout(height=520, title="Non-orientable unity manifold")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("φ-Harmonic Torus")
    X, Y, Z = torus(R=1.0, r=1 / 1.618)
    fig = go.Figure(
        data=[
            go.Surface(x=X, y=Y, z=Z, colorscale="Turbo", showscale=False, opacity=0.9)
        ]
    )
    fig.update_layout(height=520, title="Golden torus")
    st.plotly_chart(fig, use_container_width=True)
