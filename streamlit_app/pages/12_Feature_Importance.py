import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "models"

st.set_page_config(page_title="Feature Importance", layout="wide")
st.title("12. Feature Importance")
st.caption(
    "Ranks fingerprint features by their contribution to the Extra Trees model (AP2D_Count, Random split). "
    "Gini importance measures mean decrease in impurity during training; "
    "permutation importance measures mean decrease in R² when a feature is shuffled on the test set."
)


@st.cache_data
def load_importance():
    path = DATA_DIR / "feature_importance.csv"
    return pd.read_csv(path)


df = load_importance()

# --- Controls ---
st.sidebar.header("Settings")
top_n = st.sidebar.slider("Top N features", 10, 50, 20, 5)

# --- Section 1: Gini Importance ---
with st.container(border=True):
    st.subheader("Gini Importance (Mean Decrease in Impurity)")
    st.caption("How much each feature reduces variance across all tree splits during training.")

    top_gini = df.nlargest(top_n, "gini_importance")
    fig = px.bar(
        top_gini.iloc[::-1],
        x="gini_importance",
        y="feature",
        orientation="h",
        color="gini_importance",
        color_continuous_scale="Blues",
    )
    fig.update_layout(height=max(400, top_n * 22), showlegend=False,
                      xaxis_title="Gini Importance", yaxis_title="")
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# --- Section 2: Permutation Importance ---
with st.container(border=True):
    st.subheader("Permutation Importance (Mean Decrease in R²)")
    st.caption("Drop in test R² when each feature is randomly shuffled. Error bars show std across 10 repeats.")

    top_perm = df.nlargest(top_n, "perm_importance_mean")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=top_perm["perm_importance_mean"].values[::-1],
        y=top_perm["feature"].values[::-1],
        orientation="h",
        error_x=dict(type="data", array=top_perm["perm_importance_std"].values[::-1]),
        marker_color="darkorange",
    ))
    fig2.update_layout(height=max(400, top_n * 22),
                       xaxis_title="Permutation Importance (ΔR²)", yaxis_title="")
    st.plotly_chart(fig2, use_container_width=True)

# --- Section 3: Gini vs Permutation Scatter ---
with st.container(border=True):
    st.subheader("Gini vs Permutation Importance")
    st.caption(
        "Features in the upper-right are important by both methods (high confidence). "
        "Features high on one axis but low on the other may be overfitting artifacts (Gini only) "
        "or correlated feature effects (Permutation only)."
    )

    df_plot = df.copy()
    df_plot["top_20_gini"] = df_plot["feature"].isin(df.nlargest(20, "gini_importance")["feature"])
    df_plot["top_20_perm"] = df_plot["feature"].isin(df.nlargest(20, "perm_importance_mean")["feature"])
    df_plot["category"] = "Other"
    df_plot.loc[df_plot["top_20_gini"] & df_plot["top_20_perm"], "category"] = "Top 20 (both)"
    df_plot.loc[df_plot["top_20_gini"] & ~df_plot["top_20_perm"], "category"] = "Top 20 (Gini only)"
    df_plot.loc[~df_plot["top_20_gini"] & df_plot["top_20_perm"], "category"] = "Top 20 (Perm only)"

    color_map = {
        "Top 20 (both)": "#2ca02c",
        "Top 20 (Gini only)": "#1f77b4",
        "Top 20 (Perm only)": "#ff7f0e",
        "Other": "#d3d3d3",
    }

    fig3 = px.scatter(
        df_plot,
        x="gini_importance",
        y="perm_importance_mean",
        color="category",
        color_discrete_map=color_map,
        hover_data=["feature"],
        opacity=0.7,
    )
    fig3.update_layout(height=500, xaxis_title="Gini Importance",
                       yaxis_title="Permutation Importance (ΔR²)")
    st.plotly_chart(fig3, use_container_width=True)

# --- Section 4: Detailed Table ---
with st.container(border=True):
    st.subheader("Detailed Rankings")
    st.caption("Top features ranked by Gini importance with corresponding permutation scores.")

    display_df = df.nlargest(top_n, "gini_importance")[
        ["feature", "gini_importance", "perm_importance_mean", "perm_importance_std"]
    ].reset_index(drop=True)
    display_df.index += 1
    display_df.columns = ["Feature", "Gini Importance", "Perm Importance (mean)", "Perm Importance (std)"]
    st.dataframe(display_df, use_container_width=True)

# --- Summary metrics ---
with st.container(border=True):
    st.subheader("Summary")
    top20_gini = set(df.nlargest(20, "gini_importance")["feature"])
    top20_perm = set(df.nlargest(20, "perm_importance_mean")["feature"])
    overlap = top20_gini & top20_perm

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total features", len(df))
    col2.metric("Top 20 overlap", f"{len(overlap)}/20")
    col3.metric("Top feature (Gini)", df.iloc[0]["feature"])
    col4.metric("Top feature (Perm)", df.nlargest(1, "perm_importance_mean").iloc[0]["feature"])
