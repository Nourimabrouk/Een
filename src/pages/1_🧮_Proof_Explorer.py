#!/usr/bin/env python3
# flake8: noqa
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Proof Explorer | Een", page_icon="🧮", layout="wide")

st.title("🧮 Proof Explorer — 1 + 1 = 1")
st.caption("Boolean • Idempotent • Topological • Quantum • Category Theory")

phi = (1 + 5**0.5) / 2


def boolean_proof() -> go.Figure:
    x = ["True ∨ True"]
    y = [1]
    fig = go.Figure(go.Bar(x=x, y=y, marker_color="#ffd700"))
    fig.update_layout(
        title="Boolean Algebra: True OR True = True = 1", yaxis=dict(range=[0, 1.2])
    )
    return fig


def idempotent_max_proof() -> go.Figure:
    xs = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    ys = np.maximum(xs, xs)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            name="max(x,x)",
            line=dict(color="#00e6e6"),
        )
    )
    fig.add_hline(y=1.0, line_dash="dot", line_color="#999")
    fig.update_layout(title="Idempotent Semiring: x ⊕ x = x ⇒ 1 ⊕ 1 = 1")
    return fig


def topology_mobius_outline() -> go.Figure:
    # Parametric outline (not full surface for performance)
    t = np.linspace(0, 2 * np.pi, 400)
    R = 1.0
    x = (R + 0.3 * np.cos(t / 2)) * np.cos(t)
    y = (R + 0.3 * np.cos(t / 2)) * np.sin(t)
    z = 0.3 * np.sin(t / 2)
    fig = go.Figure(
        go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color="#ffd700"))
    )
    fig.update_layout(
        title="Topology: Möbius strip unity manifold (outline)",
        scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
    )
    return fig


def quantum_interference() -> go.Figure:
    x = np.linspace(-10, 10, 400)
    wave1 = np.sin(x)
    wave2 = np.sin(x)
    unity = (wave1 + wave2) / np.max(np.abs(wave1 + wave2))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=wave1, name="Wave 1", line=dict(color="#00e6e6")))
    fig.add_trace(go.Scatter(x=x, y=wave2, name="Wave 2", line=dict(color="#ff00ff")))
    fig.add_trace(
        go.Scatter(
            x=x, y=unity, name="Unity (normalized)", line=dict(color="#ffd700", width=3)
        )
    )
    fig.update_layout(title="Quantum Interference: Two waves become One")
    return fig


def category_theory_schematic() -> go.Figure:
    fig = go.Figure()
    # D objects
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[1],
            mode="markers+text",
            text=["1_left"],
            textposition="bottom center",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[2],
            y=[1],
            mode="markers+text",
            text=["1_right"],
            textposition="bottom center",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[1], y=[0], mode="markers+text", text=["1+1"], textposition="top center"
        )
    )
    # U object
    fig.add_trace(
        go.Scatter(
            x=[6],
            y=[0.5],
            mode="markers+text",
            text=["1 (Unity)"],
            textposition="top center",
        )
    )
    # Functor arrows (schematic)
    fig.add_annotation(
        x=6,
        y=0.5,
        ax=0,
        ay=1,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#ffd700",
    )
    fig.add_annotation(
        x=6,
        y=0.5,
        ax=2,
        ay=1,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#ffd700",
    )
    fig.add_annotation(
        x=6,
        y=0.5,
        ax=1,
        ay=0,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#ffd700",
    )
    fig.update_layout(
        title="Category Theory: Functor F: D → U maps all objects to 1",
        xaxis_visible=False,
        yaxis_visible=False,
    )
    return fig


tabs = st.tabs(["Boolean", "Idempotent", "Topology", "Quantum", "Category Theory"])

with tabs[0]:
    st.plotly_chart(boolean_proof(), use_container_width=True)
    st.markdown("True ∨ True = True ⇒ 1 + 1 = 1 in Boolean algebra.")

with tabs[1]:
    st.plotly_chart(idempotent_max_proof(), use_container_width=True)
    st.markdown("Idempotent semiring with ⊕ = max ⇒ 1 ⊕ 1 = 1.")

with tabs[2]:
    st.plotly_chart(topology_mobius_outline(), use_container_width=True)
    st.markdown("Unity manifold intuition via Möbius strip.")

with tabs[3]:
    st.plotly_chart(quantum_interference(), use_container_width=True)
    st.markdown("Constructive interference normalized to unity.")

with tabs[4]:
    st.plotly_chart(category_theory_schematic(), use_container_width=True)
    st.markdown("Functor collapses distinction category to unity category.")
