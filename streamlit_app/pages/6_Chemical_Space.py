import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_bioactivity, get_pca_embedding, ACTIVITY_COLORS, FIG_DIR

st.set_page_config(page_title="Chemical Space", layout="wide")
st.title("6. Chemical Space Visualization")

df = load_bioactivity()

# PCA
st.subheader("PCA on ECFP4 Fingerprints (1024 bits)")
X_pca, var_exp = get_pca_embedding()

color_mode = st.radio("Color by", ["Activity Class", "pchembl_value"], horizontal=True)

if color_mode == "Activity Class":
    fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=df["activity_class"],
                     color_discrete_map=ACTIVITY_COLORS,
                     labels={"x": f"PC1 ({var_exp[0]*100:.1f}%)", "y": f"PC2 ({var_exp[1]*100:.1f}%)"},
                     hover_data={"Molecule": df["molecule_chembl_id"].values,
                                 "pchembl": df["pchembl_value"].values},
                     title="PCA Chemical Space — Activity Class", opacity=0.5)
else:
    mask = df["pchembl_value"].notna()
    fig = px.scatter(x=X_pca[mask, 0], y=X_pca[mask, 1],
                     color=df.loc[mask, "pchembl_value"],
                     color_continuous_scale="RdYlGn",
                     labels={"x": f"PC1 ({var_exp[0]*100:.1f}%)", "y": f"PC2 ({var_exp[1]*100:.1f}%)",
                             "color": "pchembl"},
                     hover_data={"Molecule": df.loc[mask, "molecule_chembl_id"].values},
                     title="PCA Chemical Space — Potency", opacity=0.5)
fig.update_traces(marker_size=4)
fig.update_layout(height=550)
st.plotly_chart(fig, use_container_width=True)

# Variance explained
col1, col2 = st.columns(2)
with col1:
    cumvar = np.cumsum(var_exp) * 100
    fig_var = go.Figure()
    fig_var.add_bar(x=list(range(1, 11)), y=var_exp * 100, name="Individual")
    fig_var.add_scatter(x=list(range(1, 11)), y=cumvar.tolist(), mode="lines+markers",
                        name="Cumulative", yaxis="y2")
    fig_var.update_layout(
        title="Variance Explained (PCA)", height=350,
        xaxis_title="Principal Component", yaxis_title="Variance (%)",
        yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 100]),
    )
    st.plotly_chart(fig_var, use_container_width=True)

with col2:
    st.markdown(f"""
    **PCA Summary**:
    - PC1 explains {var_exp[0]*100:.1f}% of variance
    - Top 5 PCs: {cumvar[4]:.1f}%
    - Top 10 PCs: {cumvar[9]:.1f}%
    
    The modest variance explained per component indicates high diversity
    in the chemical space — no single axis captures the data well.
    """)

st.markdown("---")

# t-SNE (pre-rendered image — too slow to compute live)
st.subheader("t-SNE Visualization")
st.info("t-SNE is pre-computed (too slow for live rendering). Showing saved figure.")
tsne_path = FIG_DIR / "10_tsne_chemical_space.png"
if tsne_path.exists():
    st.image(str(tsne_path), use_container_width=True)
else:
    st.warning("t-SNE figure not found. Run `04b_save_eda_figures.py` to generate.")
