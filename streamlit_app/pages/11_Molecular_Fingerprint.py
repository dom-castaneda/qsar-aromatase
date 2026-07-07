import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
from pathlib import Path
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, leaves_list

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_filtered_fingerprint

st.set_page_config(page_title="Molecular Fingerprint", layout="wide")
st.title("11. Molecular Fingerprint — Feature Correlation")
st.caption(
    "Visualises the Pearson correlation between features within a single fingerprint. "
    "Highly correlated feature clusters indicate redundancy; uncorrelated features provide independent signal."
)

FP_TYPES = {
    "MACCS": "maccs",
    "PubChem": "pubchem",
    "ECFP4": "ecfp4",
    "Substructure": "substruct",
    "SubstructureCount": "substruct_count",
    "KR": "kr",
    "KR_Count": "kr_count",
    "AtomPairs2D": "atompairs2d",
    "AP2D_Count": "atompairs2d_count",
    "CDK_FP": "cdk_fp",
    "CDK_Extended": "cdk_extended",
    "CDK_GraphOnly": "cdk_graphonly",
    "EState": "estate",
    "EState_Count": "estate_count",
}

# Dropdown to select fingerprint
selected_fp = st.selectbox("Select fingerprint", list(FP_TYPES.keys()), index=0)


@st.cache_data
def compute_feature_correlation(fp_key, max_features=200):
    """Compute Pearson correlation matrix between features of a fingerprint."""
    fp_df = load_filtered_fingerprint(FP_TYPES[fp_key])
    X = fp_df.iloc[:, 1:].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    n_features = X.shape[1]
    feature_names = list(fp_df.columns[1:])

    # If too many features, select top by variance
    if n_features > max_features:
        variances = X.var(axis=0)
        top_idx = np.argsort(variances)[-max_features:]
        top_idx.sort()
        X = X[:, top_idx]
        feature_names = [feature_names[i] for i in top_idx]
        n_features = max_features
        truncated = True
    else:
        truncated = False

    corr = np.corrcoef(X.T)
    corr = np.nan_to_num(corr, nan=0.0)
    corr_df = pd.DataFrame(corr, index=feature_names, columns=feature_names)
    return corr_df, truncated, fp_df.shape[1] - 1


# Sidebar controls
st.sidebar.header("Settings")
max_features = st.sidebar.slider("Max features to display", 50, 500, 200, 50,
                                 help="Limits heatmap size by selecting top-variance features.")
cluster_order = st.sidebar.checkbox("Cluster ordering", value=True,
                                    help="Reorder features by hierarchical clustering")
show_annotations = st.sidebar.checkbox("Show values", value=False,
                                       help="Show correlation values on heatmap (slow for large matrices)")

with st.spinner(f"Computing feature correlations for {selected_fp}..."):
    corr_df, truncated, total_features = compute_feature_correlation(selected_fp, max_features)

# Cluster ordering
if cluster_order and len(corr_df) > 2:
    dist = 1 - np.abs(corr_df.values)
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    dist = np.clip(dist, 0, None)
    condensed = squareform(dist)
    Z = linkage(condensed, method="average")
    order = leaves_list(Z)
    ordered_names = [corr_df.index[i] for i in order]
    corr_df = corr_df.loc[ordered_names, ordered_names]

# Heatmap
with st.container(border=True):
    st.subheader(f"Feature Correlation Heatmap — {selected_fp}")
    if truncated:
        st.caption(f"Showing top {len(corr_df)} features by variance (out of {total_features} total).")
    else:
        st.caption(f"All {total_features} features shown.")

    fig = px.imshow(
        corr_df.values,
        x=corr_df.columns.tolist(),
        y=corr_df.index.tolist(),
        text_auto=".2f" if show_annotations and len(corr_df) <= 50 else False,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="equal",
    )
    fig.update_layout(height=max(500, len(corr_df) * 4), width=max(500, len(corr_df) * 4))
    fig.update_xaxes(showticklabels=len(corr_df) <= 80)
    fig.update_yaxes(showticklabels=len(corr_df) <= 80)
    st.plotly_chart(fig, use_container_width=True)

# Correlation distribution
with st.container(border=True):
    st.subheader("Correlation Distribution")
    st.caption("Distribution of pairwise absolute correlations between features (upper triangle only).")
    upper_vals = squareform(corr_df.values, checks=False)

    fig_hist = px.histogram(
        x=np.abs(upper_vals), nbins=50,
        labels={"x": "|Pearson r|", "y": "Count"},
        color_discrete_sequence=["steelblue"],
    )
    fig_hist.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig_hist, use_container_width=True)

# Summary stats
with st.container(border=True):
    st.subheader("Summary Statistics")
    abs_upper = np.abs(upper_vals)
    n_pairs = len(upper_vals)
    n_high = (abs_upper > 0.9).sum()
    n_moderate = ((abs_upper > 0.7) & (abs_upper <= 0.9)).sum()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Features", f"{total_features}")
    col2.metric("Mean |r|", f"{abs_upper.mean():.3f}")
    col3.metric("Median |r|", f"{np.median(abs_upper):.3f}")
    col4.metric("|r| > 0.9", f"{n_high} ({100*n_high/n_pairs:.1f}%)")
    col5.metric("|r| > 0.7", f"{n_high + n_moderate} ({100*(n_high+n_moderate)/n_pairs:.1f}%)")

# Top correlated pairs
with st.container(border=True):
    st.subheader("Most Correlated Feature Pairs")
    st.caption("Top 20 feature pairs by absolute correlation within this fingerprint.")
    names = corr_df.index.tolist()
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append({"Feature A": names[i], "Feature B": names[j],
                          "Correlation": corr_df.iloc[i, j]})
    pairs_df = pd.DataFrame(pairs)
    pairs_df["|r|"] = pairs_df["Correlation"].abs()
    pairs_df = pairs_df.sort_values("|r|", ascending=False).head(20).reset_index(drop=True)
    st.dataframe(pairs_df[["Feature A", "Feature B", "Correlation", "|r|"]],
                 use_container_width=True, hide_index=True)
