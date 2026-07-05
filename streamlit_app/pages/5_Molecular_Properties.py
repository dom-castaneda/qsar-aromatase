import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_bioactivity, compute_descriptors, ACTIVITY_COLORS

st.set_page_config(page_title="Molecular Properties", layout="wide")
st.title("5. Molecular Properties & Drug-likeness")

df = load_bioactivity()
desc_df = compute_descriptors(df["canonical_smiles"])
df = pd.concat([df, desc_df], axis=1)

desc_cols = ["MW", "LogP", "HBA", "HBD", "TPSA", "RotBonds", "AromaticRings", "FractionCSP3"]

# Descriptor histograms
st.subheader("Descriptor Distributions")
selected_desc = st.selectbox("Select descriptor", desc_cols, index=0)

fig = px.histogram(df, x=selected_desc, color="activity_class", nbins=50,
                   color_discrete_map=ACTIVITY_COLORS, barmode="overlay", opacity=0.6,
                   marginal="box", title=f"Distribution of {selected_desc}")
lipinski_limits = {"MW": 500, "LogP": 5, "HBA": 10, "HBD": 5}
if selected_desc in lipinski_limits:
    fig.add_vline(x=lipinski_limits[selected_desc], line_dash="dash", line_color="red",
                  annotation_text=f"Ro5 limit ({lipinski_limits[selected_desc]})")
fig.update_layout(height=450)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Interactive scatter
st.subheader("Property Scatter Plot")
col1, col2, col3 = st.columns(3)
with col1:
    x_prop = st.selectbox("X-axis", desc_cols, index=0)
with col2:
    y_prop = st.selectbox("Y-axis", desc_cols + ["pchembl_value"], index=1)
with col3:
    color_by = st.selectbox("Color by", ["activity_class", "pchembl_value", "standard_type"], index=0)

plot_df = df.dropna(subset=[x_prop, y_prop])
if color_by == "pchembl_value":
    plot_df = plot_df.dropna(subset=["pchembl_value"])

fig_scatter = px.scatter(plot_df, x=x_prop, y=y_prop, color=color_by,
                         color_discrete_map=ACTIVITY_COLORS if color_by == "activity_class" else None,
                         color_continuous_scale="RdYlGn" if color_by == "pchembl_value" else None,
                         opacity=0.5, hover_data=["molecule_chembl_id"],
                         title=f"{x_prop} vs {y_prop}")
fig_scatter.update_traces(marker_size=5)
fig_scatter.update_layout(height=500)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# Lipinski compliance
st.subheader("Lipinski Rule-of-5 Compliance")
violations = ((df["MW"] > 500).astype(int) + (df["LogP"] > 5).astype(int) +
              (df["HBA"] > 10).astype(int) + (df["HBD"] > 5).astype(int))
viol_counts = violations.value_counts().sort_index()

col1, col2 = st.columns(2)
with col1:
    fig_viol = px.bar(x=viol_counts.index, y=viol_counts.values,
                      labels={"x": "Number of Violations", "y": "Count"},
                      title="Ro5 Violations", text=viol_counts.values,
                      color=viol_counts.index, color_continuous_scale=["green", "orange", "red", "darkred"])
    fig_viol.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_viol, use_container_width=True)

with col2:
    drug_like = (violations == 0).sum()
    st.metric("Drug-like (0 violations)", f"{drug_like:,} / {len(df):,}")
    st.metric("Percentage", f"{drug_like/len(df)*100:.1f}%")
    st.markdown("""
    **Lipinski Rule-of-5**:
    - MW <= 500 Da
    - LogP <= 5
    - HBA <= 10
    - HBD <= 5
    """)
