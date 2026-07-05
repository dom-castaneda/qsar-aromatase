import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_bioactivity

st.set_page_config(page_title="Temporal", layout="wide")
st.title("4. Temporal Trends")

df = load_bioactivity()

# Records per year
st.subheader("Bioactivity Records per Year")
year_counts = df["document_year"].value_counts().sort_index()
fig = px.bar(x=year_counts.index, y=year_counts.values,
             labels={"x": "Publication Year", "y": "Number of Records"},
             title="Records per Year")
fig.update_traces(marker_color="steelblue")
fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Potency by decade
st.subheader("Potency Distribution by Decade")
pch = df.dropna(subset=["pchembl_value"]).copy()
pch["decade"] = (pch["document_year"] // 10) * 10

fig_violin = px.violin(pch, x="decade", y="pchembl_value", color="decade",
                       box=True, points=False,
                       labels={"decade": "Decade", "pchembl_value": "pchembl_value (-log10 M)"},
                       title="Potency by Decade")
fig_violin.add_hline(y=6.5, line_dash="dash", line_color="green", annotation_text="Active")
fig_violin.add_hline(y=5.0, line_dash="dash", line_color="red", annotation_text="Inactive")
fig_violin.update_layout(height=450, showlegend=False)
st.plotly_chart(fig_violin, use_container_width=True)

st.markdown("---")

# Cumulative discovery
st.subheader("Cumulative Molecule Discovery")
first_seen = df.groupby("molecule_chembl_id")["document_year"].min().sort_values()
cumul = first_seen.value_counts().sort_index().cumsum()

fig_cumul = go.Figure(go.Scatter(x=cumul.index, y=cumul.values, mode="lines+markers",
                                  marker=dict(size=4), line=dict(color="darkgreen"),
                                  fill="tozeroy", fillcolor="rgba(46,204,113,0.15)"))
fig_cumul.update_layout(title="Cumulative Unique Molecules Discovered",
                        xaxis_title="Year", yaxis_title="Cumulative Molecules", height=400)
st.plotly_chart(fig_cumul, use_container_width=True)
