import streamlit as st
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_bioactivity, compute_descriptors

st.set_page_config(page_title="Overview", layout="wide")
st.title("1. Dataset Overview")

df = load_bioactivity()

# KPI row
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Records", f"{len(df):,}")
c2.metric("Unique Molecules", f"{df['molecule_chembl_id'].nunique():,}")
c3.metric("pchembl Coverage", f"{df['pchembl_value'].notna().mean()*100:.1f}%")
c4.metric("Year Span", f"{int(df['document_year'].min())}–{int(df['document_year'].max())}")
desc_df = compute_descriptors(df["canonical_smiles"])
ro5_pass = ((desc_df["MW"] <= 500) & (desc_df["LogP"] <= 5) &
            (desc_df["HBA"] <= 10) & (desc_df["HBD"] <= 5)).mean() * 100
c5.metric("Drug-like (Ro5)", f"{ro5_pass:.1f}%")

st.markdown("---")

# Summary table
st.subheader("Assay Type Breakdown")
type_counts = df["standard_type"].value_counts()
col1, col2 = st.columns(2)
with col1:
    st.dataframe(type_counts.reset_index().rename(columns={"index": "Type", "standard_type": "Type", "count": "Count"}))
with col2:
    st.markdown(f"""
    - **IC50**: {type_counts.get('IC50', 0):,}
    - **Ki**: {type_counts.get('Ki', 0):,}
    - **pIC50**: {type_counts.get('pIC50', 0):,}
    - **Assay type**: Binding ({(df['assay_type']=='B').sum():,}), ADME ({(df['assay_type']=='A').sum():,})
    """)

st.markdown("---")
st.subheader("Dataset Preview")
st.dataframe(df.head(20), use_container_width=True)
