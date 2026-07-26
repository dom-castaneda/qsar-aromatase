import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import PROJECT_ROOT

st.set_page_config(page_title="Hyperparameter Tuning", layout="wide")
st.title("13. Hyperparameter Tuning")
st.caption(
    "RandomizedSearchCV (100 iterations, 5-fold CV) on the best model — Extra Trees on AP2D_Count fingerprint. "
    "Compares tuned parameters against default sklearn parameters to assess whether tuning improves performance."
)


@st.cache_data
def load_tuning():
    path = PROJECT_ROOT / "data" / "models" / "tuning_results.json"
    with open(path) as f:
        return json.load(f)


data = load_tuning()
reg = data["regression"]
cls = data["classification"]

# --- Summary Metrics ---
with st.container(border=True):
    st.subheader("Tuning Summary")
    st.caption("Default Extra Trees parameters are near-optimal for this dataset.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Regression** (predict pchembl_value)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Baseline R²", f"{reg['baseline_r2']:.4f}")
        c2.metric("Tuned R²", f"{reg['test_r2']:.4f}",
                  delta=f"{reg['test_r2'] - reg['baseline_r2']:+.4f}")
        c3.metric("Runtime", f"{reg['tuning_time_min']:.0f} min")

    with col2:
        st.markdown("**Classification** (predict activity class)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Baseline BalAcc", f"{cls['baseline_balacc']:.4f}")
        c2.metric("Tuned BalAcc", f"{cls['test_balacc']:.4f}",
                  delta=f"{cls['test_balacc'] - cls['baseline_balacc']:+.4f}")
        c3.metric("Runtime", f"{cls['tuning_time_min']:.0f} min")

# --- Comparison Bar Chart ---
with st.container(border=True):
    st.subheader("Default vs Tuned Performance")

    fig = go.Figure()

    # Regression
    fig.add_trace(go.Bar(name="Default", x=["R² (Regression)", "BalAcc (Classification)"],
                         y=[reg["baseline_r2"], cls["baseline_balacc"]],
                         marker_color="steelblue"))
    fig.add_trace(go.Bar(name="Tuned", x=["R² (Regression)", "BalAcc (Classification)"],
                         y=[reg["test_r2"], cls["test_balacc"]],
                         marker_color="darkorange"))

    fig.update_layout(barmode="group", height=400,
                      yaxis_title="Score", title="Default vs Tuned (Test Set)")
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Conclusion**: Hyperparameter tuning produced negligible improvement. "
        "Regression R² decreased slightly (-0.021), while classification BalAcc improved marginally (+0.001). "
        "Default Extra Trees parameters are already well-suited to this dataset."
    )

# --- Best Parameters ---
with st.container(border=True):
    st.subheader("Best Parameters Found")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Regression**")
        params_reg = pd.DataFrame([
            {"Parameter": k, "Value": str(v)}
            for k, v in reg["best_params"].items()
        ])
        st.dataframe(params_reg, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**Classification**")
        params_cls = pd.DataFrame([
            {"Parameter": k, "Value": str(v)}
            for k, v in cls["best_params"].items()
        ])
        st.dataframe(params_cls, use_container_width=True, hide_index=True)

    st.markdown("""
    **Key finding**: `max_features=0.3` (use 30% of features per split) appeared in both best configurations. 
    This is more restrictive than the default (all features), suggesting some feature decorrelation helps, 
    but the overall impact on test performance is minimal.
    """)

# --- Search Space ---
with st.container(border=True):
    st.subheader("Search Configuration")

    col1, col2, col3 = st.columns(3)
    col1.metric("Iterations", data["n_iter"])
    col2.metric("CV Folds", data["n_folds"])
    col3.metric("Total Fits", f"{data['n_iter'] * data['n_folds']}")

    st.markdown(f"""
    **Fingerprint**: {data['fingerprint']}  |  **Split**: {data['split']}  
    **Search space**: n_estimators × max_depth × min_samples_split × min_samples_leaf × max_features = 1,920 combinations  
    **Sampled**: {data['n_iter']} random configurations
    """)
