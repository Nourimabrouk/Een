"""Unity Torus - φ-Harmonic 3D Visualization"""

import sys
from pathlib import Path

import streamlit as st

current_dir = Path(__file__).parent.parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from viz.plotly_helpers import create_unity_torus  # noqa: E402

st.set_page_config(
    page_title="Unity Torus - Een Mathematics",
    page_icon="🔁",
    layout="wide",
)

st.markdown("# 🔁 Unity Torus: 3D 1+1=1 Visualization")
st.markdown("Explore a φ-harmonic torus demonstrating how addition wraps into oneness.")

st.sidebar.markdown("## 🎛️ Torus Controls")

theme = st.sidebar.selectbox("🎨 Visualization Theme", options=["dark", "light"], index=0)
major_radius = st.sidebar.slider("Major Radius", 1.0, 4.0, 2.0, step=0.1)
minor_radius = st.sidebar.slider("Minor Radius", 0.2, 1.5, 0.6, step=0.1)
phi_modulation = st.sidebar.slider("φ Modulation", 0.0, 1.0, 0.3, step=0.05)

fig = create_unity_torus(major_radius, minor_radius, phi_modulation, theme)

st.plotly_chart(fig, use_container_width=True)

st.markdown("### φ-Harmonic Torus Equation")
st.latex(
    r"""
\begin{align}
 x &= (R + r\cos v)\cos u \\
 y &= (R + r\cos v)\sin u \\
 z &= r\sin v \\
 \text{Unity Level} &= \frac{\sin(\phi u)+\cos(\phi v)}{2}
\end{align}
"""
)
