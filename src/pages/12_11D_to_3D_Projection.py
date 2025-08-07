#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# flake8: noqa
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="11D to 3D Projection", layout="wide")

st.title("11D → 3D Projection — Hyperdimensional Unity")
st.caption("Project hyperdimensional consciousness manifolds to 3D with φ-harmonics")

import importlib
import logging

try:
    engine_mod = importlib.import_module(
        "src.consciousness.transcendental_reality_engine"
    )
    if not hasattr(engine_mod, "logger"):
        engine_mod.logger = logging.getLogger(
            "src.consciousness.transcendental_reality_engine"
        )
    TranscendentalConfig = engine_mod.TranscendentalConfig  # type: ignore
    HyperdimensionalManifoldProjector = engine_mod.HyperdimensionalManifoldProjector  # type: ignore
except Exception:
    TranscendentalConfig = None
    HyperdimensionalManifoldProjector = None

if TranscendentalConfig is None or HyperdimensionalManifoldProjector is None:
    st.warning("Transcendental engine not importable. Check module path.")
else:
    cfg = TranscendentalConfig()
    try:
        proj = HyperdimensionalManifoldProjector(cfg)
    except NameError:
        # Ensure module-global logger exists, then retry
        if not hasattr(engine_mod, "logger"):
            engine_mod.logger = logging.getLogger(
                "src.consciousness.transcendental_reality_engine"
            )
        if not engine_mod.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("[unity] %(levelname)s: %(message)s")
            )
            engine_mod.logger.addHandler(handler)
            engine_mod.logger.setLevel(logging.INFO)
        proj = HyperdimensionalManifoldProjector(cfg)
    points = st.slider("Points", 200, 3000, 1000, 100)
    cons = st.slider("Consciousness level", 0.1, 1.5, 0.618, 0.001)
    enforce_unity = st.checkbox("Preserve unity in projection", True)

    # Generate 11D manifold and project to 4D, then show 3D projection
    m11 = proj.generate_consciousness_manifold_11d(points, cons)
    m4 = proj.project_11d_to_4d(m11, preserve_unity=enforce_unity)
    # Take first 3 dims for 3D viz
    x, y, z = m4[:, 0], m4[:, 1], m4[:, 2]
    fig = go.Figure(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(size=3, color=np.linalg.norm(m4, axis=1), colorscale="Viridis"),
        )
    )
    fig.update_layout(title="Projected Manifold (color = ||p||)", height=600)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("### Math Notes")
st.latex(
    r"P_{11\to 4} = \sum_{i=0}^{10} w_i \Pi_i,\ \ w_i\propto \phi^i / \sum_k \phi^k"
)
st.latex(
    r"\tilde{p} = P_{11\to 4}(p),\ \ \tilde{p} \gets \text{unity-preserving scaling}"
)
