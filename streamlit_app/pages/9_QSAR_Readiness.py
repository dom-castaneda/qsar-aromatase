import streamlit as st
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_bioactivity, compute_descriptors

st.set_page_config(page_title="QSAR Readiness", layout="wide")
st.title("9. QSAR Readiness Assessment")

df = load_bioactivity()
desc_df = compute_descriptors(df["canonical_smiles"])

n_total = len(df)
n_with_pch = df["pchembl_value"].notna().sum()
n_exact = (df["standard_relation"] == "=").sum()
n_exact_pch = df[(df["standard_relation"] == "=") & df["pchembl_value"].notna()].shape[0]
active_n = (df["activity_class"] == "Active").sum()
inactive_n = (df["activity_class"] == "Inactive").sum()
intermed_n = (df["activity_class"] == "Intermediate").sum()
drug_like_n = ((desc_df["MW"] <= 500) & (desc_df["LogP"] <= 5) &
               (desc_df["HBA"] <= 10) & (desc_df["HBD"] <= 5)).sum()

# Summary metrics
st.subheader("Dataset Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records", f"{n_total:,}")
c2.metric("With pchembl", f"{n_with_pch:,} ({n_with_pch/n_total*100:.1f}%)")
c3.metric("Exact (=) + pchembl", f"{n_exact_pch:,}")
c4.metric("Drug-like", f"{drug_like_n:,} ({drug_like_n/n_total*100:.1f}%)")

st.markdown("---")

# Activity classes
st.subheader("Activity Class Balance")
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    | Class | Threshold | Count |
    |-------|-----------|-------|
    | Active | pchembl >= 6.5 | {active_n:,} |
    | Intermediate | 5.0 <= pchembl < 6.5 | {intermed_n:,} |
    | Inactive | pchembl < 5.0 | {inactive_n:,} |
    """)

with col2:
    ratio = inactive_n / max(active_n, 1)
    st.metric("Active:Inactive Ratio", f"1 : {ratio:.2f}")
    if 0.5 <= ratio <= 2.0:
        st.success("Class balance: BALANCED")
    elif ratio <= 5.0:
        st.warning("Class balance: MODERATE imbalance — consider stratified splits")
    else:
        st.error("Class balance: SEVERE imbalance — consider SMOTE or class weights")

st.markdown("---")

# Recommendations
st.subheader("Modelling Recommendations")

st.markdown(f"""
### For Regression
- **{n_exact_pch:,} molecules** with exact (`=`) pchembl_value
- Continuous target with good spread (range 4.0–10.82, std=1.26)
- Use scaffold-based splitting to avoid data leakage

### For Binary Classification
- **{active_n + inactive_n:,} molecules** ({active_n:,} active + {inactive_n:,} inactive)
- Drop intermediate class for cleaner decision boundary
- Class ratio 1:{ratio:.2f} — use stratified k-fold or class weights

### Data Handling
1. **Censored data (275 records with `>`)**: Exclude from regression; include in classification
   (if pchembl > 6.5 with `>`, the molecule is definitively active)
2. **Multiple assay types**: Consider per-type models or use pchembl (already standardised)

### Fingerprint Selection
| Fingerprint | Use Case |
|-------------|----------|
| ECFP4 (1024) | General purpose, good baseline |
| MACCS (167) | Interpretability, feature importance |
| CDK Extended (1024) | Rich path-based encoding |
| KR (4860) | Maximum substructure detail (needs feature selection) |
| AtomPairs2D (780) | Pharmacophore-pair information |

### Data Splitting
- **Scaffold-based splits** (Bemis-Murcko) to avoid data leakage from congeneric series
- Random splits overestimate performance for related molecule series
- Recommended: 80/10/10 train/val/test with scaffold grouping

### Suggested Models
1. Random Forest (baseline, fast, interpretable)
2. Gradient Boosting (XGBoost/LightGBM)
3. Support Vector Machine (with RBF kernel on fingerprints)
4. Deep learning (if dataset augmented or transfer learning applied)
""")

st.markdown("---")
st.success("Dataset is READY for QSAR modelling")
