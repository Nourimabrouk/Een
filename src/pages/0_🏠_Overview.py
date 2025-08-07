#!/usr/bin/env python3
# flake8: noqa
import streamlit as st

st.set_page_config(page_title="Overview | Een", page_icon="🏠", layout="wide")

st.title("🏠 Overview — Unity Mathematics")
st.caption("Academic-grade, φ-harmonic dashboard for 1+1=1")

phi = (1 + 5**0.5) / 2

st.markdown(
    f"""
    - **Unity Equation**: 1 + 1 = 1 (idempotent, Boolean, categorical, quantum)
    - **φ (Golden Ratio)**: {phi:.15f}
    - **Metagamer Energy**: E = φ² × ρ × U
    """
)

st.success(
    "Jump to Proofs, Consciousness Field, or the Codebase Explorer from the sidebar."
)
