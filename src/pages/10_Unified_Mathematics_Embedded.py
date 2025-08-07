#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# flake8: noqa
import streamlit as st

st.set_page_config(page_title="Unified Mathematics (Embedded)", layout="wide")

st.title("🧩 Unified Mathematics Dashboard — Embedded")
st.caption("Interactive proofs, unity manipulator, and consciousness calculator")

try:
    from src.dashboards.unified_mathematics_dashboard import UnifiedMathematicsDashboard  # type: ignore
except Exception:
    UnifiedMathematicsDashboard = None

if UnifiedMathematicsDashboard is None:
    st.warning("UnifiedMathematicsDashboard not importable. Check module path.")
else:
    dash = UnifiedMathematicsDashboard()
    st.subheader("Interactive Proofs (Category Theory, Quantum)")
    proofs = list(dash.interactive_proofs.keys())
    choice = st.selectbox("Proof Framework", proofs)
    proof = dash.interactive_proofs[choice]
    st.write(f"Theorem: {proof.theorem_statement}")
    st.write(f"Steps: {len(proof.proof_steps)} | Valid: {proof.overall_validity}")

    st.subheader("Unity Equation Manipulator")
    left = st.slider("Left", 0.0, 2.0, 1.0, 0.01)
    right = st.slider("Right", 0.0, 2.0, 1.0, 0.01)
    phi_c = st.slider("φ-coefficient", 0.2, 3.0, 1.618, 0.001)
    cons = st.slider("Consciousness", 0.0, 2.0, 1.0, 0.01)
    res = dash.unity_manipulator.manipulate_equation(
        left_operand=left,
        right_operand=right,
        phi_harmonic_coefficient=phi_c,
        consciousness_factor=cons,
    )
    st.metric("Result", f"{res['result']:.4f}")
    st.write(res["explanation"])

    st.subheader("Consciousness Field Calculation")
    x = st.slider("x", -3.14, 3.14, 0.0, 0.01)
    y = st.slider("y", -3.14, 3.14, 0.0, 0.01)
    t = st.slider("t", 0.0, 6.28, 0.0, 0.01)
    calc = dash.consciousness_calculator.calculate_consciousness_field(x, y, t)
    st.write(calc)
