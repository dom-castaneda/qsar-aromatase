"""
Aromatase (CYP19A1) Inhibitors — EDA Dashboard
Multi-page Streamlit app for exploratory data analysis.
"""
import streamlit as st

st.set_page_config(
    page_title="Aromatase QSAR EDA",
    page_icon=":microscope:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Aromatase (CYP19A1) Inhibitors — EDA Dashboard")
st.markdown("""
**Target**: Aromatase (CHEMBL1978) — Homo sapiens, CYP19A1  
**Dataset**: 3,774 curated bioactivity records from ChEMBL  
**Unique molecules**: 3,399

---

Navigate using the sidebar to explore different aspects of the dataset.

| Page | Content |
|------|---------|
| 1. Overview | Dataset summary, key metrics |
| 2. Bioactivity | pchembl distribution, activity classes |
| 3. Data Quality | Missing values, censored data |
| 4. Temporal | Publication trends over time |
| 5. Molecular Properties | Lipinski, descriptors |
| 6. Chemical Space | PCA & t-SNE visualizations |
| 7. Fingerprints | Bit frequencies, sparsity |
| 8. Correlations | Property-potency relationships |
| 9. QSAR Readiness | Modelling recommendations |
""")
