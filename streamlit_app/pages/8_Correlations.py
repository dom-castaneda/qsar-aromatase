import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_bioactivity, compute_descriptors, load_fingerprints, FIG_DIR

st.set_page_config(page_title="Correlations", layout="wide")
st.title("8. Correlations & Feature Importance")

df = load_bioactivity()
desc_df = compute_descriptors(df["canonical_smiles"])
df = pd.concat([df, desc_df], axis=1)

prop_cols = ["MW", "LogP", "HBA", "HBD", "TPSA", "RotBonds", "AromaticRings", "FractionCSP3"]
pch_mask = df["pchembl_value"].notna()

# Property correlations
st.subheader("Molecular Property Correlations with Potency")
corr_data = []
for col in prop_cols:
    valid = pch_mask & df[col].notna()
    r, p = stats.pearsonr(df.loc[valid, col], df.loc[valid, "pchembl_value"])
    corr_data.append({"Property": col, "Pearson r": r, "p-value": p,
                      "Significance": "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"})

corr_df = pd.DataFrame(corr_data).sort_values("Pearson r", key=abs, ascending=True)
colors = ["#2ecc71" if r > 0 else "#e74c3c" for r in corr_df["Pearson r"]]

fig = go.Figure(go.Bar(x=corr_df["Pearson r"], y=corr_df["Property"], orientation="h",
                        marker_color=colors,
                        text=[f"r={r:.3f} {s}" for r, s in zip(corr_df["Pearson r"], corr_df["Significance"])],
                        textposition="outside"))
fig.add_vline(x=0, line_color="black", line_width=0.5)
fig.update_layout(title="Pearson Correlation with pchembl_value", height=400, xaxis_title="Pearson r")
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Top fingerprint bits
st.subheader("Top Fingerprint Bits Correlated with Potency")

@st.cache_data
def compute_top_bits():
    pch_vals = df.loc[pch_mask, "pchembl_value"].values
    results = []
    for fp_name, prefix in [("maccs", "MACCS"), ("ecfp4", "ECFP4"), ("kr", "KR")]:
        fp = load_fingerprints(fp_name)
        bit_matrix = fp.iloc[:, 1:].values[pch_mask.values]
        for i in range(bit_matrix.shape[1]):
            col = bit_matrix[:, i]
            if col.std() == 0:
                continue
            r, p = stats.pointbiserialr(col, pch_vals)
            results.append({"Bit": f"{prefix}_{i}", "r": r, "p": p, "Freq": col.mean()})
    rdf = pd.DataFrame(results)
    rdf["abs_r"] = rdf["r"].abs()
    return rdf.nlargest(20, "abs_r")

top_bits = compute_top_bits()
colors_bits = ["#2ecc71" if r > 0 else "#e74c3c" for r in top_bits["r"]]
fig_bits = go.Figure(go.Bar(x=top_bits["r"], y=top_bits["Bit"], orientation="h",
                             marker_color=colors_bits))
fig_bits.add_vline(x=0, line_color="black", line_width=0.5)
fig_bits.update_layout(title="Top 20 FP Bits (Point-Biserial r with pchembl)",
                       height=550, xaxis_title="Correlation", yaxis=dict(autorange="reversed"))
st.plotly_chart(fig_bits, use_container_width=True)

st.markdown("---")

# Correlation heatmap
st.subheader("Inter-Property Correlation Matrix")
heatmap_path = FIG_DIR / "15_correlation_heatmap.png"
if heatmap_path.exists():
    st.image(str(heatmap_path), use_container_width=True)
else:
    corr_matrix = df[prop_cols + ["pchembl_value"]].corr()
    fig_heat = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1, title="Property Correlation Matrix")
    fig_heat.update_layout(height=500)
    st.plotly_chart(fig_heat, use_container_width=True)
