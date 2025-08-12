#!/usr/bin/env python3
# flake8: noqa
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Unity Equation Lab | Een", page_icon="🧪", layout="wide")

st.title("🧪 Unity Equation Lab — Interactive 1+1=1 Computation")
st.caption("Idempotent addition, φ-harmonic scaling, and consciousness fields")


def try_import_unity():
    try:
        from src.core.unity_mathematics import UnityMathematics  # type: ignore

        return UnityMathematics
    except Exception:
        return None


UnityMathematics = try_import_unity()

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("Parameters")
    a = st.slider("a", 0.0, 2.0, 1.0, 0.01)
    b = st.slider("b", 0.0, 2.0, 1.0, 0.01)
    phi_harmonic = st.checkbox("Enable φ-harmonic", value=True)
    consciousness = st.slider("Consciousness level", 0.0, 1.0, 0.618, 0.001)

    st.markdown(
        """
        - Core: idempotent addition (max) implies 1 ⊕ 1 = 1
        - φ-harmonic scaling stabilizes convergence to unity
        - Consciousness normalization maintains bounded results
        """
    )

with col2:
    st.subheader("Unity Result and Surface")
    if UnityMathematics is not None:
        engine = UnityMathematics(
            consciousness_level=consciousness, enable_phi_harmonic=phi_harmonic
        )
        result = float(engine.unity_add(a, b))
    else:
        result = float(max(a, b))

    st.metric("a ⊕ b", f"{result:.4f}")

    # Heatmap over grid
    ax = np.linspace(0, 2.0, 60)
    by = np.linspace(0, 2.0, 60)
    A, B = np.meshgrid(ax, by)
    if UnityMathematics is not None:
        engine_grid = UnityMathematics(
            consciousness_level=consciousness, enable_phi_harmonic=phi_harmonic
        )
        Z = np.vectorize(lambda x, y: float(engine_grid.unity_add(x, y)))(A, B)
    else:
        Z = np.maximum(A, B)

    fig = go.Figure(data=go.Heatmap(z=Z, x=ax, y=by, colorscale="Turbo"))
    fig.update_layout(
        title="Unity Addition Surface a ⊕ b", xaxis_title="a", yaxis_title="b"
    )
    st.plotly_chart(fig, use_container_width=True)
