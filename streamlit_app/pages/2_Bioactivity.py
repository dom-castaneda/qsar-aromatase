import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_bioactivity, ACTIVITY_COLORS, ACTIVITY_ORDER

st.set_page_config(page_title="Bioactivity", layout="wide")
st.title("2. Bioactivity Distribution")

df = load_bioactivity()
pch = df.dropna(subset=["pchembl_value"])

# Controls
col1, col2 = st.columns([1, 3])
with col1:
    n_bins = st.slider("Histogram bins", 20, 80, 40)
    selected_types = st.multiselect("Assay types", ["IC50", "Ki", "pIC50"], default=["IC50", "Ki", "pIC50"])

# Filtered data
pch_filt = pch[pch["standard_type"].isin(selected_types)]

with col2:
    fig = px.histogram(pch_filt, x="pchembl_value", color="standard_type", nbins=n_bins,
                       opacity=0.7, barmode="overlay", marginal="box",
                       labels={"pchembl_value": "pchembl_value (-log10 M)", "standard_type": "Assay Type"},
                       title=f"Distribution of pchembl_value (n={len(pch_filt):,})")
    fig.add_vline(x=6.5, line_dash="dash", line_color="green", annotation_text="Active (6.5)")
    fig.add_vline(x=5.0, line_dash="dash", line_color="red", annotation_text="Inactive (5.0)")
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Activity Class Distribution")

class_counts = df["activity_class"].value_counts()
col1, col2 = st.columns(2)

with col1:
    fig_bar = go.Figure(go.Bar(
        x=[c for c in ACTIVITY_ORDER if c in class_counts.index],
        y=[class_counts.get(c, 0) for c in ACTIVITY_ORDER if c in class_counts.index],
        marker_color=[ACTIVITY_COLORS[c] for c in ACTIVITY_ORDER if c in class_counts.index],
        text=[class_counts.get(c, 0) for c in ACTIVITY_ORDER if c in class_counts.index],
        textposition="outside",
    ))
    fig_bar.update_layout(title="Activity Class Counts", yaxis_title="Count", height=400)
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    known = class_counts.drop("Unknown", errors="ignore")
    fig_pie = px.pie(values=known.values, names=known.index,
                     color=known.index, color_discrete_map=ACTIVITY_COLORS,
                     title="Activity Classes (pchembl known only)")
    fig_pie.update_layout(height=400)
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown(f"""
**Thresholds**: Active >= 6.5, Inactive < 5.0  
**Active:Inactive ratio** = {class_counts.get('Active', 0)} : {class_counts.get('Inactive', 0)} = 
1 : {class_counts.get('Inactive', 0) / max(class_counts.get('Active', 1), 1):.2f}
""")
