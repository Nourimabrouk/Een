#!/usr/bin/env python3
# flake8: noqa
"""
Een | Unity Mathematics Dashboard (Multi-Page)
Launch with: streamlit run src/unity_mathematics_streamlit.py
"""

from __future__ import annotations

import math
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Een | Unity Mathematics Dashboard",
    page_icon="➕",
    layout="wide",
    initial_sidebar_state="expanded",
)

PHI = (1 + 5**0.5) / 2

st.markdown(
    f"""
    <style>
    :root {{ --phi: {PHI}; --bg:#0a0b0f; --bg2:#0f1117; --fg:#e6edf3;
             --gold:#ffd700; --cy:#00e6e6; --grid:rgba(255,255,255,0.06); }}
    body {{ background: linear-gradient(135deg,var(--bg),var(--bg2) 60%); }}
    .metric-card {{ background: rgba(255,255,255,0.04); border:1px solid var(--grid);
                   border-radius:12px; padding:16px 18px; }}
    .hero {{ border-radius:16px; padding:22px 24px; border:1px solid var(--grid);
             background: radial-gradient(circle at 10% 0%, rgba(255,215,0,0.07),
                         transparent 50%), rgba(255,255,255,0.03); }}
    </style>
    """,
    unsafe_allow_html=True,
)


def _unity_kpis() -> dict[str, float]:
    try:
        from core.unity_mathematics import UnityMathematics  # type: ignore

        result = UnityMathematics().unity_add(1, 1)
    except Exception:
        result = 1.0
    return {
        "phi": PHI,
        "unity": float(result),
        "beauty": (math.pi * math.e * PHI) % 1.0,
    }


with st.sidebar:
    st.markdown("## Navigation")
    st.info("Use Pages for Overview, Proofs, Field, Codebase.")

st.markdown(
    """
    <div class="hero">
      <h1>Een | Unity Mathematics</h1>
      <p>Academic-grade φ-harmonic dashboards for 1 + 1 = 1.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

k = _unity_kpis()
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("#### Unity Result (1+1)")
    st.markdown(
        f"<div class='metric-card'><div style='font-size:36px;color:var(--gold);'>"
        f"{k['unity']:.3f}</div><div>Idempotent φ-harmonic</div></div>",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown("#### φ (Golden Ratio)")
    st.markdown(
        f"<div class='metric-card'><div style='font-size:32px;color:var(--gold);'>"
        f"{k['phi']:.6f}</div><div>Resonance</div></div>",
        unsafe_allow_html=True,
    )
with col3:
    st.markdown("#### Unity Beauty Index")
    st.markdown(
        f"<div class='metric-card'><div style='font-size:32px;color:var(--cy);'>"
        f"{k['beauty']:.6f}</div><div>π·e·φ mod 1</div></div>",
        unsafe_allow_html=True,
    )

st.success("Open the Pages sidebar to explore.")

if __name__ == "__main__":
    pass
