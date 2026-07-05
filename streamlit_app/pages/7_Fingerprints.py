import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_fingerprints

st.set_page_config(page_title="Fingerprints", layout="wide")
st.title("7. Fingerprint Bit Frequency Analysis")

# Fingerprint selector
fp_options = {
    "MACCS (167 bits)": "maccs",
    "ECFP4 (1024 bits)": "ecfp4",
    "Substructure (307 bits)": "substruct",
    "Substructure Count (307 values)": "substruct_count",
    "Klekota-Roth (4860 bits)": "kr",
    "Klekota-Roth Count (4860 values)": "kr_count",
    "AtomPairs2D (780 bits)": "atompairs2d",
    "AtomPairs2D Count (780 values)": "atompairs2d_count",
    "CDK Fingerprinter (1024 bits)": "cdk_fp",
    "CDK Extended (1024 bits)": "cdk_extended",
    "CDK GraphOnly (1024 bits)": "cdk_graphonly",
    "E-State (79 bits)": "estate",
    "E-State Count (79 values)": "estate_count",
}

selected_fp = st.selectbox("Select fingerprint type", list(fp_options.keys()))
fp_key = fp_options[selected_fp]

fp_df = load_fingerprints(fp_key)
bits = fp_df.iloc[:, 1:].values
n_bits = bits.shape[1]
freq = bits.mean(axis=0)

# Bit frequency bar
st.subheader(f"Bit Frequency Profile — {selected_fp}")
fig = go.Figure()
fig.add_bar(x=list(range(n_bits)), y=freq, marker_color="steelblue", name="Frequency")
fig.add_hline(y=0.05, line_dash="dash", line_color="red", annotation_text="5%")
fig.add_hline(y=0.95, line_dash="dash", line_color="orange", annotation_text="95%")
fig.update_layout(height=350, xaxis_title="Bit Index", yaxis_title="Fraction of Molecules",
                  title=f"Bit Frequency ({n_bits} bits)")
st.plotly_chart(fig, use_container_width=True)

# Stats
col1, col2, col3 = st.columns(3)
n_always_off = (freq == 0).sum()
n_rare = (freq < 0.05).sum()
n_ubiq = (freq > 0.95).sum()
n_inform = n_bits - n_rare - n_ubiq

col1.metric("Always Off (=0)", n_always_off)
col2.metric("Rare (<5%)", n_rare)
col3.metric("Informative (5-95%)", n_inform)

st.markdown("---")

# Bits per molecule
st.subheader("Bits per Molecule")
bits_per_mol = bits.sum(axis=1)
fig_hist = px.histogram(x=bits_per_mol, nbins=50, labels={"x": "Number of Bits Set"},
                        title=f"Bits per Molecule — {selected_fp}")
fig_hist.update_traces(marker_color="mediumpurple")
fig_hist.update_layout(height=350)
st.plotly_chart(fig_hist, use_container_width=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean", f"{bits_per_mol.mean():.1f}")
col2.metric("Median", f"{np.median(bits_per_mol):.0f}")
col3.metric("Min", int(bits_per_mol.min()))
col4.metric("Max", int(bits_per_mol.max()))

st.markdown("---")

# Density comparison
st.subheader("Fingerprint Density Comparison")
all_fps = {"MACCS": "maccs", "SubFP": "substruct", "ECFP4": "ecfp4",
           "AtomPairs2D": "atompairs2d", "CDK FP": "cdk_fp", "CDK Ext": "cdk_extended",
           "CDK Graph": "cdk_graphonly", "E-State": "estate", "KR": "kr"}

densities = []
for name, key in all_fps.items():
    fp = load_fingerprints(key)
    density = fp.iloc[:, 1:].values.mean() * 100
    densities.append({"Fingerprint": name, "Density (%)": density})

den_df = pd.DataFrame(densities).sort_values("Density (%)", ascending=False)
fig_den = px.bar(den_df, x="Fingerprint", y="Density (%)", text="Density (%)",
                 title="Average Bit Density Across Fingerprint Types",
                 color="Density (%)", color_continuous_scale="RdYlGn_r")
fig_den.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig_den.update_layout(height=400)
st.plotly_chart(fig_den, use_container_width=True)
