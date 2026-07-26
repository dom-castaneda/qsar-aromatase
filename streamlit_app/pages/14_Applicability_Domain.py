import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import sys
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import PROJECT_ROOT

st.set_page_config(page_title="Applicability Domain", layout="wide")
st.title("14. Applicability Domain")
st.caption(
    "Assesses whether test molecules fall within the chemical space covered by the training set. "
    "Uses PCA bounding box on AP2D_Count fingerprint — molecules outside the box may have unreliable predictions."
)


@st.cache_data
def load_ad_results():
    path = PROJECT_ROOT / "data" / "models" / "applicability_domain.json"
    with open(path) as f:
        return json.load(f)


@st.cache_data
def compute_pca_scores():
    DATA_DIR = PROJECT_ROOT / "data"
    df_full = pd.read_csv(DATA_DIR / "processed" / "aromatase_bioactivity_clean.csv")
    mask = (df_full["standard_relation"] == "=") & df_full["pchembl_value"].notna()
    df = df_full[mask].reset_index(drop=True)

    fp_full = pd.read_csv(DATA_DIR / "fingerprints_filtered" / "fingerprints_atompairs2d_count.csv")
    fp = fp_full[mask.values].reset_index(drop=True)
    fp_cols = [c for c in fp.columns if c != "molecule_chembl_id"]
    X_all = np.nan_to_num(fp[fp_cols].values.astype(np.float32), nan=0.0)

    train_ids = set(pd.read_csv(DATA_DIR / "splits" / "random_train.csv")["molecule_chembl_id"])
    train_mask = df["molecule_chembl_id"].isin(train_ids).values
    test_mask = ~train_mask

    X_train, X_test = X_all[train_mask], X_all[test_mask]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    pca = PCA(n_components=0.95, random_state=42)
    train_scores = pca.fit_transform(X_train_s)
    test_scores = pca.transform(X_test_s)

    # Bounding box
    train_min = train_scores.min(axis=0)
    train_max = train_scores.max(axis=0)
    inside_mask = np.all((test_scores >= train_min) & (test_scores <= train_max), axis=1)

    return train_scores, test_scores, inside_mask, pca, train_min, train_max


ad = load_ad_results()
train_scores, test_scores, inside_mask, pca, train_min, train_max = compute_pca_scores()

# --- Coverage Metrics ---
with st.container(border=True):
    st.subheader("Coverage Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test Inside AD", f"{ad['n_test_inside']} ({ad['coverage_pct']:.1f}%)")
    col2.metric("Test Outside AD", f"{ad['n_test_outside']} ({100-ad['coverage_pct']:.1f}%)")
    col3.metric("PCA Components", ad["n_components"])
    col4.metric("Variance Explained", f"{ad['variance_explained']*100:.1f}%")

    st.success(
        f"**96.3% of test molecules** fall within the training applicability domain. "
        f"Only 28 molecules are outside — predictions for these should be treated with caution."
    )

# --- PCA Scatter ---
with st.container(border=True):
    st.subheader("PCA Chemical Space — PC1 vs PC2")
    st.caption("Training set defines the applicability domain (dashed box). "
               "Test molecules outside the box on any of the 178 PCs are flagged.")

    fig = go.Figure()

    # Train
    fig.add_trace(go.Scatter(
        x=train_scores[:, 0], y=train_scores[:, 1],
        mode="markers", name=f"Train ({train_scores.shape[0]})",
        marker=dict(color="steelblue", size=4, opacity=0.3),
    ))
    # Test inside
    fig.add_trace(go.Scatter(
        x=test_scores[inside_mask, 0], y=test_scores[inside_mask, 1],
        mode="markers", name=f"Test inside AD ({inside_mask.sum()})",
        marker=dict(color="green", size=6, opacity=0.5),
    ))
    # Test outside
    fig.add_trace(go.Scatter(
        x=test_scores[~inside_mask, 0], y=test_scores[~inside_mask, 1],
        mode="markers", name=f"Test outside AD ({(~inside_mask).sum()})",
        marker=dict(color="red", size=9, symbol="x", opacity=0.8),
    ))

    # Bounding box (PC1 vs PC2)
    x_box = [train_min[0], train_max[0], train_max[0], train_min[0], train_min[0]]
    y_box = [train_min[1], train_min[1], train_max[1], train_max[1], train_min[1]]
    fig.add_trace(go.Scatter(
        x=x_box, y=y_box, mode="lines", name="AD boundary",
        line=dict(color="black", dash="dash", width=2),
    ))

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    fig.update_layout(
        height=600,
        xaxis_title=f"PC1 ({var1:.1f}% variance)",
        yaxis_title=f"PC2 ({var2:.1f}% variance)",
        legend=dict(x=0.7, y=0.98),
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Method Description ---
with st.container(border=True):
    st.subheader("Method")
    st.markdown("""
    **PCA Bounding Box** (following ERpred / Nantasenamat group methodology):
    
    1. Standardize AP2D_Count features (zero mean, unit variance) using training set statistics
    2. Fit PCA on training set, retain components explaining 95% cumulative variance (178 PCs)
    3. Define bounding box: [min, max] of each PC score from training set
    4. Project test molecules into same PCA space
    5. A molecule is **inside the AD** if ALL 178 PC scores fall within the training min/max
    
    Molecules outside the AD are structurally different from the training data — 
    model predictions for these compounds may be unreliable.
    """)
