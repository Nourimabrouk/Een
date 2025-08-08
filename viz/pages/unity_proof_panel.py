"""
Streamlit Page: Unity Proof Panel
Renders the aggregated witnesses for 1+1=1 from `src.proofs.unity_master_proof`.
Run via: streamlit run viz/pages/unity_proof_panel.py
"""

# pylint: disable=C0301
import sys
from pathlib import Path

import streamlit as st

# Ensure project root on path so `src` is importable when launched from repo root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.proofs.unity_master_proof import prove_unity, render_streamlit_panel


st.set_page_config(page_title="Unity Proof — 1+1=1", page_icon="✅", layout="wide")

st.markdown("# ✅ Unity Proof — 1 + 1 = 1")
st.caption(
    "Formal witnesses across algebra, category theory, logic, topology, quantum operations, and information fusion."
)

render_streamlit_panel(prove_unity())


