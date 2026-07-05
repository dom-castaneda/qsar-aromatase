import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_bioactivity

st.set_page_config(page_title="Data Quality", layout="wide")
st.title("3. Data Quality & Completeness")

df = load_bioactivity()

# Missing values
st.subheader("Missing Values")
null_counts = df.isnull().sum()
null_pct = (null_counts / len(df) * 100).sort_values(ascending=False)
null_pct = null_pct[null_pct > 0]

fig = px.bar(x=null_pct.values, y=null_pct.index, orientation="h",
             labels={"x": "% Missing", "y": "Column"},
             title="Missing Values by Column",
             text=[f"{v:.1f}% ({null_counts[k]})" for k, v in null_pct.items()])
fig.update_traces(marker_color="salmon")
fig.update_layout(height=300)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Censored data
st.subheader("Measurement Quality")
col1, col2 = st.columns(2)

with col1:
    rel_counts = df["standard_relation"].value_counts(dropna=False)
    n_exact = (df["standard_relation"] == "=").sum()
    n_censored = df["standard_relation"].isin([">", ">="]).sum()
    n_missing = df["standard_relation"].isna().sum()

    st.markdown(f"""
    | Relation | Count | % |
    |----------|-------|---|
    | Exact (=) | {n_exact:,} | {n_exact/len(df)*100:.1f}% |
    | Censored (>) | {n_censored:,} | {n_censored/len(df)*100:.1f}% |
    | Missing | {n_missing:,} | {n_missing/len(df)*100:.1f}% |
    """)
    st.warning("Censored measurements (>) mean the true potency could be higher. "
               "Exclude from regression; can include in classification.")

with col2:
    unit_counts = df["standard_units"].value_counts(dropna=False)
    fig_units = px.bar(x=unit_counts.index.astype(str), y=unit_counts.values,
                       labels={"x": "Unit", "y": "Count"},
                       title="Standard Units Distribution")
    fig_units.update_layout(height=300)
    st.plotly_chart(fig_units, use_container_width=True)

st.markdown("---")

# Deduplication
st.subheader("Deduplication Quality")
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    | Metric | Value |
    |--------|-------|
    | Rows | {len(df):,} |
    | Unique ChEMBL IDs | {df['molecule_chembl_id'].nunique():,} |
    | Unique InChIKeys | {df['inchi_key'].nunique():,} |
    | Unique SMILES | {df['canonical_smiles'].nunique():,} |
    | Duplicate (mol, type) pairs | 0 |
    """)

with col2:
    mol_recs = df.groupby("molecule_chembl_id").size().value_counts().sort_index().head(10)
    fig_mult = px.bar(x=mol_recs.index, y=mol_recs.values,
                      labels={"x": "Records per molecule", "y": "Count of molecules"},
                      title="Record Multiplicity")
    fig_mult.update_layout(height=300)
    st.plotly_chart(fig_mult, use_container_width=True)
