import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import PROJECT_ROOT

st.set_page_config(page_title="Model Performance", layout="wide")
st.title("10. Model Performance — Random vs Kennard-Stone")


@st.cache_data
def load_results():
    path = PROJECT_ROOT / "data" / "models" / "results_all_models.csv"
    return pd.read_csv(path)


results_df = load_results()

# Sidebar filters
st.sidebar.header("Filters")
all_models = sorted(results_df["Model"].unique())
all_fps = sorted(results_df["Fingerprint"].unique())

selected_models = st.sidebar.multiselect("Models", all_models, default=all_models)
selected_fps = st.sidebar.multiselect("Fingerprints", all_fps, default=all_fps)
metric = st.sidebar.selectbox("Primary Metric", ["R2_test", "RMSE_test", "MAE_test", "R2_CV", "RMSE_CV", "MAE_CV"])

# Filter
df = results_df[results_df["Model"].isin(selected_models) & results_df["Fingerprint"].isin(selected_fps)]

# ===================================================================
st.subheader("Heatmap: Random vs Kennard-Stone")

heatmap_metric = st.selectbox("Heatmap metric", ["R2", "RMSE", "MAE"], index=0)
heatmap_set = st.radio("Evaluation set", ["Test", "CV", "Train"], horizontal=True)
heatmap_col = f"{heatmap_metric}_{heatmap_set.lower()}" if heatmap_set != "Test" else f"{heatmap_metric}_test"
if heatmap_set == "Train":
    heatmap_col = f"{heatmap_metric}_train"
elif heatmap_set == "CV":
    heatmap_col = f"{heatmap_metric}_CV"
else:
    heatmap_col = f"{heatmap_metric}_test"

col1, col2 = st.columns(2)

for i, split_name in enumerate(["Random", "Kennard-Stone"]):
    sub = df[df["Split"] == split_name]
    pivot = sub.pivot_table(index="Model", columns="Fingerprint", values=heatmap_col)
    ascending = heatmap_metric != "R2"
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=ascending).index]

    cscale = "RdYlGn" if heatmap_metric == "R2" else "RdYlGn_r"
    fig = px.imshow(pivot, text_auto=".3f", color_continuous_scale=cscale,
                    aspect="auto", title=f"{split_name} — {heatmap_metric} ({heatmap_set})")
    fig.update_layout(height=500)
    with [col1, col2][i]:
        st.plotly_chart(fig, use_container_width=True)

# ===================================================================
st.markdown("---")
st.subheader("Side-by-Side Comparison: Random vs Kennard-Stone")

# Merge Random and KS results for same Model+FP
random_df = df[df["Split"] == "Random"][["Model", "Fingerprint", metric]].rename(columns={metric: "Random"})
ks_df = df[df["Split"] == "Kennard-Stone"][["Model", "Fingerprint", metric]].rename(columns={metric: "KS"})
compare = random_df.merge(ks_df, on=["Model", "Fingerprint"])
compare["Diff"] = compare["Random"] - compare["KS"]

fig_compare = px.scatter(compare, x="Random", y="KS", color="Model",
                         hover_data=["Fingerprint"],
                         title=f"{metric}: Random vs Kennard-Stone (each point = model+FP combo)")
# Diagonal line
min_val = min(compare["Random"].min(), compare["KS"].min()) - 0.05
max_val = max(compare["Random"].max(), compare["KS"].max()) + 0.05
fig_compare.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                      line=dict(dash="dash", color="gray"))
fig_compare.update_layout(height=500, xaxis_title=f"{metric} (Random)", yaxis_title=f"{metric} (KS)")
st.plotly_chart(fig_compare, use_container_width=True)

above = (compare["KS"] > compare["Random"]).sum() if "R2" in metric else (compare["KS"] < compare["Random"]).sum()
st.caption(f"Points above diagonal: KS better in {above}/{len(compare)} cases. "
           f"Points below: Random better in {len(compare)-above}/{len(compare)} cases.")

# ===================================================================
st.markdown("---")
st.subheader("Best Model per Fingerprint")

col1, col2 = st.columns(2)
for i, split_name in enumerate(["Random", "Kennard-Stone"]):
    sub = df[df["Split"] == split_name].dropna(subset=[metric])
    if len(sub) == 0:
        continue
    if "R2" in metric:
        best = sub.loc[sub.groupby("Fingerprint")[metric].idxmax()]
    else:
        best = sub.loc[sub.groupby("Fingerprint")[metric].idxmin()]
    best = best.sort_values(metric, ascending=("R2" not in metric))

    with [col1, col2][i]:
        st.markdown(f"**{split_name} Split**")
        st.dataframe(best[["Fingerprint", "Model", "R2_test", "RMSE_test", "MAE_test"]].reset_index(drop=True),
                     use_container_width=True, hide_index=True)

# ===================================================================
st.markdown("---")
st.subheader("Model Rankings by Split")

rank_metric = metric
for split_name in ["Random", "Kennard-Stone"]:
    sub = df[df["Split"] == split_name].dropna(subset=[rank_metric])
    avg = sub.groupby("Model")[rank_metric].mean().sort_values(ascending=("R2" not in rank_metric))

    fig_rank = px.bar(x=avg.values, y=avg.index, orientation="h",
                      labels={"x": f"Mean {rank_metric} (across fingerprints)", "y": "Model"},
                      title=f"{split_name} — Mean {rank_metric} per Model",
                      color=avg.values,
                      color_continuous_scale="RdYlGn" if "R2" in rank_metric else "RdYlGn_r")
    fig_rank.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig_rank, use_container_width=True)

# ===================================================================
st.markdown("---")
st.subheader("Detailed Results Table")

sort_col = st.selectbox("Sort by", [metric, "Model", "Fingerprint", "Split"], index=0)
ascending = "R2" not in sort_col
st.dataframe(df.sort_values(sort_col, ascending=ascending).reset_index(drop=True),
             use_container_width=True, height=400)

# ===================================================================
st.markdown("---")
st.subheader("Overall Best")

valid = results_df.dropna(subset=["R2_test"])
best_overall = valid.loc[valid["R2_test"].idxmax()]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Best Model", best_overall["Model"])
c2.metric("Fingerprint", best_overall["Fingerprint"])
c3.metric("Split", best_overall["Split"])
c4.metric("Test R²", f"{best_overall['R2_test']:.4f}")

st.markdown(f"""
**Full metrics**: R²={best_overall['R2_test']:.4f}, RMSE={best_overall['RMSE_test']:.4f}, MAE={best_overall['MAE_test']:.4f}  
**CV performance**: R²={best_overall['R2_CV']:.4f}, RMSE={best_overall['RMSE_CV']:.4f}  
**Training**: R²={best_overall['R2_train']:.4f} (overfitting gap: {best_overall['R2_train']-best_overall['R2_test']:.3f})
""")
