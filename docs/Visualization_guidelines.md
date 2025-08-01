````markdown
# visualization_standards.md
Standards for all visual analytics code in the **Een** repository  
_Last updated: 2025‑08‑01_

---

## 1 – Preferred Stack  

| Layer                 | Library             | Rationale                                                                 |
|----------------------|---------------------|--------------------------------------------------------------------------|
| **Interactive plots**| **Plotly ≥ 5.20**   | Wide chart type coverage (2‑D, 3‑D, maps, networks), self-contained HTML, first-class Streamlit integration. |
| **Dashboards / apps**| **Streamlit ≥ 1.35**| Fast prototyping, built-in multipage, auto-reload, easy cloud deploy (Streamlit Cloud). |
| **Static figures**   | Matplotlib + Seaborn| Only for publication-grade images that must be PDF/PNG; keep isolated from web app. |

> **Hard rule:** Any new interactive visualization **must** use Plotly + Streamlit unless a maintainer documents a technical blocker in `/docs/exceptions.md`.

---

## 2 – Directory Layout  

/viz/ # All viz code lives here
init.py
plotly_helpers.py # Shared figure builders & theming
streamlit_app.py # Single-file, multipage Streamlit entry point
pages/ # One file per Streamlit page
unity_proofs.py
meta_rl.py
...
/assets/plotly_templates/ # .json templates for dark/light themes
/docs/visualization_standards.md

---

## 3 – Plotly Guidelines  

1. **Templates**  
   - Use the project templates in `/assets/plotly_templates/` (`dark.json`, `light.json`).
   - Apply via:
     ```python
     import plotly.io as pio
     pio.templates.default = "dark"  # or "light"
     ```

2. **Figure builders**  
   - Write reusable functions in `plotly_helpers.py`. Example:
     ```python
     def rewards_line(df: pd.DataFrame) -> go.Figure:
         fig = px.line(df, x="episode", y="reward", title="Training Reward")
         fig.update_layout(height=400)
         return fig
     ```
   - Avoid in-line figure construction in Streamlit pages—keep UI thin.

3. **Interactivity**  
   - Use animation (`animation_frame`) only when it adds real value.
   - Avoid >3000 points per trace; aggregate or downsample upstream.
   - Provide descriptive `hovertemplate` strings; avoid raw data dumps.

4. **Performance**  
   - Cache expensive data transforms with `@st.cache_data`.
   - For interactive updates, prefer `fig.update_traces()` to redrawing.

---

## 4 – Streamlit Guidelines  

| Area              | Rule                                                                 |
|-------------------|----------------------------------------------------------------------|
| **Page structure**| One topic per file in `pages/`. Use `st.title()`, sidebar for inputs, then main visual. |
| **Controls**      | Always use `key=` and `help=` on widgets.                            |
| **State**         | Cache data with `@st.cache_data`; models with `@st.cache_resource`.  |
| **Jobs**          | Use `with st.spinner()` and `st.progress()` during long tasks.       |
| **Theme**         | Default to dark mode; define in `.streamlit/config.toml`.            |
| **Deployment**    | Local: `streamlit run viz/streamlit_app.py`. Online: Streamlit Cloud. |

---

## 5 – General Best Practices  

1. **Version pinning**  
   - Lock `plotly`, `streamlit`, `pandas`, etc. in `pyproject.toml`.

2. **Unit tests**  
   - Functions returning figures must be tested for output type and trace count.

3. **Accessibility**  
   - Always label axes, avoid color-only encoding, use readable font sizes.

4. **Documentation**  
   - Every Streamlit page must include a brief explanatory `st.markdown()` block at the top.

5. **Secrets**  
   - Use `st.secrets` and environment variables. Never commit keys.

6. **Licensing**  
   - Document license of all visual assets and reused figures explicitly.

---

## 6 – Contribution Checklist  

- [ ] Added new figure builder in `plotly_helpers.py`  
- [ ] Added/updated unit tests  
- [ ] Updated corresponding `pages/*.py` file  
- [ ] Ran `pre-commit run --all-files`  
- [ ] Verified in browser via `streamlit run`  
- [ ] Updated `/CHANGELOG.md`

---

_End of file_
````