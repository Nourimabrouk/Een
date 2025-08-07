#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# flake8: noqa
import streamlit as st

st.set_page_config(page_title="Category Theory Live", layout="wide")

st.title("📐 Category Theory — Live Unity Mapping")
st.caption("Functor F: D → U maps distinct objects to unity; interactive 3D proof")

try:
    from src.proofs.category_theory_proof import CategoryTheoryUnityProof  # type: ignore
except Exception:
    CategoryTheoryUnityProof = None

if CategoryTheoryUnityProof is None:
    st.warning("CategoryTheoryUnityProof not importable. Check module path.")
else:
    proof = CategoryTheoryUnityProof()
    result = proof.execute_categorical_proof()
    st.write({k: v for k, v in result.items() if k not in ("steps",)})
    with st.expander("Proof steps"):
        st.json(result["steps"])
    fig = proof.create_3d_proof_visualization()
    if fig:
        st.plotly_chart(fig, use_container_width=True)

st.markdown("### Math Notes")
st.latex(r"F: \mathcal{D}\to \mathcal{U},\ F(1_\text{left})=F(1_\text{right})=F(1+1)=1")
st.latex(r"1+1=1\ \text{in}\ \mathcal{U}\ (\text{unity category})")
