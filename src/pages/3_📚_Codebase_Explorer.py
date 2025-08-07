#!/usr/bin/env python3
# flake8: noqa
import re
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Codebase Explorer | Een", page_icon="📚", layout="wide")
st.title("📚 Codebase Explorer — Live Window into Unity Code")

root = Path(__file__).resolve().parents[2]

DEFAULT_FOLDERS = ["core", "src", "docs", "visualizations"]

cols = st.columns([1, 2])
with cols[0]:
    st.subheader("Browse")
    folder = st.selectbox("Folder", DEFAULT_FOLDERS)
    search = st.text_input("Search (regex)", r"unity|1\+1|phi|consciousness")
    max_files = st.slider("Max files", 10, 500, 100, 10)

with cols[1]:
    base = root / folder
    if not base.exists():
        st.error(f"Folder not found: {base}")
    else:
        matches = []
        pattern = re.compile(search, re.IGNORECASE)
        for p in base.rglob("*.py"):
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if pattern.search(text):
                matches.append(p)
            if len(matches) >= max_files:
                break

        st.caption(f"Found {len(matches)} files matching '{search}'.")
        for path in matches:
            with st.expander(str(path.relative_to(root))):
                try:
                    code = path.read_text(encoding="utf-8", errors="ignore")
                except Exception as e:
                    st.error(str(e))
                    continue
                st.code(code, language="python")
