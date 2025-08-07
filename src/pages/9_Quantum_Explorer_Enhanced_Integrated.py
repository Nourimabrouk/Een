#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# flake8: noqa
import time
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Quantum Explorer Enhanced (Integrated)", layout="wide")

st.title("⚛️ Quantum Explorer Enhanced — Integrated")
st.caption("Bloch spheres, wave interference, evolution, entanglement")

phi = (1 + 5**0.5) / 2

col_controls, col_main = st.columns([1, 3])
with col_controls:
    st.subheader("Parameters")
    theta1 = st.slider("|ψ₁⟩ θ", 0.0, np.pi, np.pi / 4, 0.01)
    phi1 = st.slider("|ψ₁⟩ φ", 0.0, 2 * np.pi, 0.0, 0.01)
    theta2 = st.slider("|ψ₂⟩ θ", 0.0, np.pi, np.pi / 4, 0.01)
    phi2 = st.slider("|ψ₂⟩ φ", 0.0, 2 * np.pi, 0.0, 0.01)
    coupling = st.slider("φ-harmonic coupling", 0.0, 1.0, 0.618, 0.001)
    weight = st.slider("Superposition weight", 0.0, 1.0, 0.5, 0.01)
    mode = st.selectbox(
        "Mode", ["Bloch", "Interference", "Evolution", "Entanglement"], index=0
    )
    animate = st.checkbox("Animate", True)
    speed = st.slider("Speed", 0.05, 3.0, 1.0, 0.05)


def create_state(theta: float, ph: float) -> np.ndarray:
    return np.array(
        [np.cos(theta / 2), np.exp(1j * ph) * np.sin(theta / 2)], dtype=complex
    )


def bloch_xyz(state: np.ndarray):
    x = 2 * np.real(state[0] * np.conj(state[1]))
    y = 2 * np.imag(state[0] * np.conj(state[1]))
    z = np.abs(state[0]) ** 2 - np.abs(state[1]) ** 2
    return x, y, z


def evolute(state: np.ndarray, t: float) -> np.ndarray:
    H = phi * np.array([[1, 0], [0, -1]]) + (1 / phi) * np.array([[0, 1], [1, 0]])
    # Use SciPy expm if available, else use series approximation
    try:
        from scipy.linalg import expm  # type: ignore

        U = expm(-1j * H * t)
    except Exception:
        # 4-term series fallback
        A = -1j * H * t
        U = np.eye(2, dtype=complex) + A + 0.5 * (A @ A) + (1.0 / 6.0) * (A @ A @ A)
    return U @ state


with col_main:
    if mode == "Bloch":
        psi1 = create_state(theta1, phi1)
        psi2 = create_state(theta2, phi2)
        if animate:
            t = time.time() * speed
            psi1 = evolute(psi1, t)
            psi2 = evolute(psi2, t)
        unity = weight * psi1 + (1 - weight) * psi2
        unity = unity / (np.linalg.norm(unity) + 1e-9)

        x1, y1, z1 = bloch_xyz(psi1)
        x2, y2, z2 = bloch_xyz(psi2)
        xu, yu, zu = bloch_xyz(unity)

        fig = make_subplots(
            rows=1,
            cols=3,
            specs=[[{"type": "scatter3d"}] * 3],
            subplot_titles=("|ψ₁⟩", "|ψ₂⟩", "Unity"),
        )
        for idx, (x, y, z, c) in enumerate(
            [(x1, y1, z1, "#00e6e6"), (x2, y2, z2, "#ff00ff"), (xu, yu, zu, "#ffd700")],
            start=1,
        ):
            fig.add_trace(
                go.Scatter3d(
                    x=[0, x],
                    y=[0, y],
                    z=[0, z],
                    mode="lines",
                    line=dict(color=c, width=6),
                ),
                row=1,
                col=idx,
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[x], y=[y], z=[z], mode="markers", marker=dict(size=10, color=c)
                ),
                row=1,
                col=idx,
            )
            fig.update_scenes(aspectmode="cube")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            "**Note:** Unity superposition normalized. Bloch coords derived from |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ} sin(θ/2)|1⟩."
        )
        st.latex(
            r"x=2\Re(\psi_0\overline{\psi_1}),\ y=2\Im(\psi_0\overline{\psi_1}),\ z=|\psi_0|^2-|\psi_1|^2"
        )

    elif mode == "Interference":
        x = np.linspace(-10, 10, 1000)
        t = time.time() * speed if animate else 0
        w1 = np.sin(x - t)
        w2 = np.sin(x - t)
        unity = coupling * w1 + (1 - coupling) * w2
        unity = unity / (np.max(np.abs(unity)) + 1e-9)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=w1, name="ψ₁"))
        fig.add_trace(go.Scatter(x=x, y=w2, name="ψ₂"))
        fig.add_trace(
            go.Scatter(x=x, y=unity, name="Unity", line=dict(color="#ffd700", width=3))
        )
        st.plotly_chart(fig, use_container_width=True)
        st.latex(
            r"\tilde{\psi}(x)=\frac{\alpha\psi_1(x)+\beta\psi_2(x)}{\max|\alpha\psi_1+\beta\psi_2|}"
        )

    elif mode == "Evolution":
        times = np.linspace(0, 5, 200)
        psi1 = create_state(theta1, phi1)
        psi2 = create_state(theta2, phi2)
        fidelity = []
        for tt in times:
            u1 = evolute(psi1, tt)
            u2 = evolute(psi2, tt)
            u = weight * u1 + (1 - weight) * u2
            u = u / (np.linalg.norm(u) + 1e-9)
            f = np.abs(np.vdot(u, psi1)) ** 2 + np.abs(np.vdot(u, psi2)) ** 2
            fidelity.append(f)
        fig = go.Figure(go.Scatter(x=times, y=fidelity, line=dict(color="#ffd700")))
        fig.update_layout(
            title="Unity Fidelity Evolution", xaxis_title="t", yaxis_title="Fidelity"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.latex(
            r"U(t)=e^{-iHt},\ \ F(t)=|\langle\psi_u(t)|\psi_1\rangle|^2+|\langle\psi_u(t)|\psi_2\rangle|^2"
        )

    else:
        psi1 = create_state(theta1, phi1)
        psi2 = create_state(theta2, phi2)
        ent = np.kron(psi1, psi2)
        ent = ent / (np.linalg.norm(ent) + 1e-9)
        C = np.abs(np.outer(np.conj(ent), ent))
        fig = go.Figure(data=go.Heatmap(z=C, colorscale="Viridis"))
        fig.update_layout(title="Entanglement Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
        st.latex(r"\rho=|\Psi\rangle\langle\Psi|,\ C_{ij}=|\rho_{ij}|")
