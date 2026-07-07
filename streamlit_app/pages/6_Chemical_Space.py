import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_bioactivity, get_pca_embedding, ACTIVITY_COLORS, FIG_DIR, PROJECT_ROOT

st.set_page_config(page_title="Chemical Space", layout="wide")
st.title("6. Chemical Space Visualization")

df = load_bioactivity()

tab1, tab2 = st.tabs(["Lipinski Descriptors", "PCA and t-SNE"])

# =====================================================================
# TAB 1: Lipinski Descriptors (original content)
# =====================================================================
with tab1:
    st.subheader("PCA on ECFP4 Fingerprints (1024 bits)")
    X_pca, var_exp = get_pca_embedding()

    color_mode = st.radio("Color by", ["Activity Class", "pchembl_value"], horizontal=True, key="lip_color")

    if color_mode == "Activity Class":
        fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=df["activity_class"],
                         color_discrete_map=ACTIVITY_COLORS,
                         labels={"x": f"PC1 ({var_exp[0]*100:.1f}%)", "y": f"PC2 ({var_exp[1]*100:.1f}%)"},
                         hover_data={"Molecule": df["molecule_chembl_id"].values,
                                     "pchembl": df["pchembl_value"].values},
                         title="PCA Chemical Space — Activity Class", opacity=0.5)
    else:
        mask = df["pchembl_value"].notna()
        fig = px.scatter(x=X_pca[mask, 0], y=X_pca[mask, 1],
                         color=df.loc[mask, "pchembl_value"],
                         color_continuous_scale="RdYlGn",
                         labels={"x": f"PC1 ({var_exp[0]*100:.1f}%)", "y": f"PC2 ({var_exp[1]*100:.1f}%)",
                                 "color": "pchembl"},
                         hover_data={"Molecule": df.loc[mask, "molecule_chembl_id"].values},
                         title="PCA Chemical Space — Potency", opacity=0.5)
    fig.update_traces(marker_size=4)
    fig.update_layout(height=550)
    st.plotly_chart(fig, use_container_width=True)

    # Variance explained
    col1, col2 = st.columns(2)
    with col1:
        cumvar = np.cumsum(var_exp) * 100
        fig_var = go.Figure()
        fig_var.add_bar(x=list(range(1, 11)), y=var_exp * 100, name="Individual")
        fig_var.add_scatter(x=list(range(1, 11)), y=cumvar.tolist(), mode="lines+markers",
                            name="Cumulative", yaxis="y2")
        fig_var.update_layout(
            title="Variance Explained (PCA)", height=350,
            xaxis_title="Principal Component", yaxis_title="Variance (%)",
            yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 100]),
        )
        st.plotly_chart(fig_var, use_container_width=True)

    with col2:
        st.markdown(f"""
        **PCA Summary**:
        - PC1 explains {var_exp[0]*100:.1f}% of variance
        - Top 5 PCs: {cumvar[4]:.1f}%
        - Top 10 PCs: {cumvar[9]:.1f}%
        
        The modest variance explained per component indicates high diversity
        in the chemical space — no single axis captures the data well.
        """)

    st.markdown("---")

    # t-SNE (pre-rendered)
    st.subheader("t-SNE Visualization")
    st.info("t-SNE is pre-computed (too slow for live rendering). Showing saved figure.")
    tsne_path = FIG_DIR / "10_tsne_chemical_space.png"
    if tsne_path.exists():
        st.image(str(tsne_path), use_container_width=True)
    else:
        st.warning("t-SNE figure not found. Run `04b_save_eda_figures.py` to generate.")


# =====================================================================
# TAB 2: PCA and t-SNE — Train vs Test (Random & Kennard-Stone)
# =====================================================================
with tab2:
    st.subheader("Train/Test Split Distribution in Chemical Space")

    # Fingerprint selector
    fp_options = {
        "ECFP4 (180 bits)": "fingerprints_ecfp4.csv",
        "MACCS (114 bits)": "fingerprints_maccs.csv",
        "PubChem (309 bits)": "fingerprints_pubchem.csv",
        "AtomPairs2D (518 bits)": "fingerprints_atompairs2d.csv",
        "AP2D Count (455 bits)": "fingerprints_atompairs2d_count.csv",
        "CDK FP (1022 bits)": "fingerprints_cdk_fp.csv",
        "CDK Extended (882 bits)": "fingerprints_cdk_extended.csv",
        "CDK GraphOnly (34 bits)": "fingerprints_cdk_graphonly.csv",
        "Substructure (41 bits)": "fingerprints_substruct.csv",
        "Substructure Count (43 bits)": "fingerprints_substruct_count.csv",
        "KR (172 bits)": "fingerprints_kr.csv",
        "KR Count (173 bits)": "fingerprints_kr_count.csv",
        "E-State (25 bits)": "fingerprints_estate.csv",
    }

    selected_fp = st.selectbox("Fingerprint for embedding", list(fp_options.keys()), index=0, key="tab2_fp")
    fp_file = fp_options[selected_fp]

    @st.cache_data
    def load_split_data(fp_filename):
        DATA_DIR = PROJECT_ROOT / "data"

        df_full = pd.read_csv(DATA_DIR / "processed" / "aromatase_bioactivity_clean.csv")
        mask = (df_full["standard_relation"] == "=") & df_full["pchembl_value"].notna()
        df_filt = df_full[mask].reset_index(drop=True)

        # Load selected fingerprint
        fp_full = pd.read_csv(DATA_DIR / "fingerprints_reduced" / fp_filename)
        fp = fp_full[mask.values].reset_index(drop=True)
        X = np.nan_to_num(fp.iloc[:, 1:].values.astype(np.float32), nan=0.0)

        # Load splits
        random_train = pd.read_csv(DATA_DIR / "splits" / "random_train.csv")
        random_test = pd.read_csv(DATA_DIR / "splits" / "random_test.csv")
        ks_train = pd.read_csv(DATA_DIR / "splits" / "kennard_stone_train.csv")
        ks_test = pd.read_csv(DATA_DIR / "splits" / "kennard_stone_test.csv")

        df_filt["random_set"] = "Unassigned"
        df_filt.loc[df_filt["molecule_chembl_id"].isin(random_train["molecule_chembl_id"]), "random_set"] = "Train"
        df_filt.loc[df_filt["molecule_chembl_id"].isin(random_test["molecule_chembl_id"]), "random_set"] = "Test"
        df_filt["ks_set"] = "Unassigned"
        df_filt.loc[df_filt["molecule_chembl_id"].isin(ks_train["molecule_chembl_id"]), "ks_set"] = "Train"
        df_filt.loc[df_filt["molecule_chembl_id"].isin(ks_test["molecule_chembl_id"]), "ks_set"] = "Test"

        # PCA
        from sklearn.decomposition import PCA
        n_comp = min(50, X.shape[1])
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X)
        ve = pca.explained_variance_ratio_

        # t-SNE (via PCA reduction first)
        from sklearn.manifold import TSNE
        pca_pre = PCA(n_components=min(50, X.shape[1]), random_state=42)
        X_pre = pca_pre.fit_transform(X)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, init="pca")
        X_tsne = tsne.fit_transform(X_pre)

        df_filt["PC1"] = X_pca[:, 0]
        df_filt["PC2"] = X_pca[:, 1]
        df_filt["tSNE1"] = X_tsne[:, 0]
        df_filt["tSNE2"] = X_tsne[:, 1]

        return df_filt, ve

    with st.spinner(f"Computing PCA and t-SNE on {selected_fp} (cached after first run)..."):
        split_df, ve = load_split_data(fp_file)

    # --- PCA plots ---
    st.subheader("PCA — Train vs Test")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Random Split**")
        chart_data = split_df[split_df["random_set"] != "Unassigned"][["PC1", "PC2", "random_set"]].rename(
            columns={"random_set": "Set"})
        chart = alt.Chart(chart_data).mark_circle(size=20, opacity=0.5).encode(
            x=alt.X("PC1:Q", title=f"PC1 ({ve[0]*100:.1f}%)"),
            y=alt.Y("PC2:Q", title=f"PC2 ({ve[1]*100:.1f}%)"),
            color=alt.Color("Set:N", scale=alt.Scale(domain=["Train", "Test"], range=["#3498db", "#e74c3c"])),
            tooltip=["Set:N"]
        ).properties(width=450, height=400, title="Random Split — PCA").interactive()
        st.altair_chart(chart, use_container_width=True)

    with col2:
        st.markdown("**Kennard-Stone Split**")
        chart_data = split_df[split_df["ks_set"] != "Unassigned"][["PC1", "PC2", "ks_set"]].rename(
            columns={"ks_set": "Set"})
        chart = alt.Chart(chart_data).mark_circle(size=20, opacity=0.5).encode(
            x=alt.X("PC1:Q", title=f"PC1 ({ve[0]*100:.1f}%)"),
            y=alt.Y("PC2:Q", title=f"PC2 ({ve[1]*100:.1f}%)"),
            color=alt.Color("Set:N", scale=alt.Scale(domain=["Train", "Test"], range=["#3498db", "#e74c3c"])),
            tooltip=["Set:N"]
        ).properties(width=450, height=400, title="Kennard-Stone Split — PCA").interactive()
        st.altair_chart(chart, use_container_width=True)

    st.caption("KS: training covers the periphery (chemical space extremes), test is interior. "
               "Random: train and test overlap uniformly.")

    st.markdown("---")

    # --- t-SNE plots ---
    st.subheader("t-SNE — Train vs Test")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Random Split**")
        chart_data = split_df[split_df["random_set"] != "Unassigned"][["tSNE1", "tSNE2", "random_set"]].rename(
            columns={"random_set": "Set"})
        chart = alt.Chart(chart_data).mark_circle(size=20, opacity=0.5).encode(
            x=alt.X("tSNE1:Q", title="t-SNE 1"),
            y=alt.Y("tSNE2:Q", title="t-SNE 2"),
            color=alt.Color("Set:N", scale=alt.Scale(domain=["Train", "Test"], range=["#3498db", "#e74c3c"])),
            tooltip=["Set:N"]
        ).properties(width=450, height=400, title="Random Split — t-SNE").interactive()
        st.altair_chart(chart, use_container_width=True)

    with col2:
        st.markdown("**Kennard-Stone Split**")
        chart_data = split_df[split_df["ks_set"] != "Unassigned"][["tSNE1", "tSNE2", "ks_set"]].rename(
            columns={"ks_set": "Set"})
        chart = alt.Chart(chart_data).mark_circle(size=20, opacity=0.5).encode(
            x=alt.X("tSNE1:Q", title="t-SNE 1"),
            y=alt.Y("tSNE2:Q", title="t-SNE 2"),
            color=alt.Color("Set:N", scale=alt.Scale(domain=["Train", "Test"], range=["#3498db", "#e74c3c"])),
            tooltip=["Set:N"]
        ).properties(width=450, height=400, title="Kennard-Stone Split — t-SNE").interactive()
        st.altair_chart(chart, use_container_width=True)

    st.markdown("---")

    # --- pchembl distribution ---
    st.subheader("pchembl Distribution — Train vs Test")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Random Split**")
        chart_data = split_df[split_df["random_set"] != "Unassigned"][["pchembl_value", "random_set"]].rename(
            columns={"random_set": "Set"})
        chart = alt.Chart(chart_data).transform_density(
            "pchembl_value", as_=["pchembl_value", "density"], groupby=["Set"]
        ).mark_area(opacity=0.5).encode(
            x=alt.X("pchembl_value:Q", title="pchembl_value"),
            y=alt.Y("density:Q", title="Density"),
            color=alt.Color("Set:N", scale=alt.Scale(domain=["Train", "Test"], range=["#3498db", "#e74c3c"])),
        ).properties(width=450, height=300, title="Random — pchembl Distribution").interactive()
        st.altair_chart(chart, use_container_width=True)

    with col2:
        st.markdown("**Kennard-Stone Split**")
        chart_data = split_df[split_df["ks_set"] != "Unassigned"][["pchembl_value", "ks_set"]].rename(
            columns={"ks_set": "Set"})
        chart = alt.Chart(chart_data).transform_density(
            "pchembl_value", as_=["pchembl_value", "density"], groupby=["Set"]
        ).mark_area(opacity=0.5).encode(
            x=alt.X("pchembl_value:Q", title="pchembl_value"),
            y=alt.Y("density:Q", title="Density"),
            color=alt.Color("Set:N", scale=alt.Scale(domain=["Train", "Test"], range=["#3498db", "#e74c3c"])),
        ).properties(width=450, height=300, title="Kennard-Stone — pchembl Distribution").interactive()
        st.altair_chart(chart, use_container_width=True)

    st.markdown("---")

    # --- Dispersity Metrics ---
    st.subheader("Data Dispersity — Random vs Kennard-Stone")
    st.markdown("Quantifies how spread out the train/test sets are in t-SNE space.")

    from scipy.spatial import ConvexHull
    from scipy.spatial.distance import cdist

    def compute_dispersity(points):
        """Compute dispersity metrics for a 2D point cloud."""
        centroid = points.mean(axis=0)
        dist_to_centroid = np.sqrt(((points - centroid) ** 2).sum(axis=1))

        # Nearest-neighbor distances
        dists = cdist(points, points)
        np.fill_diagonal(dists, np.inf)
        nn_dists = dists.min(axis=1)

        # Convex hull area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume  # In 2D, 'volume' is area
        except Exception:
            hull_area = 0.0

        return {
            "Mean dist to centroid": dist_to_centroid.mean(),
            "Std dist to centroid": dist_to_centroid.std(),
            "Mean NN distance": nn_dists.mean(),
            "Median NN distance": np.median(nn_dists),
            "Convex hull area": hull_area,
            "Max spread (diameter)": dists[dists != np.inf].max() if len(points) > 1 else 0,
        }

    # Compute for each split × set in t-SNE space
    dispersity_rows = []
    for split_col, split_name in [("random_set", "Random"), ("ks_set", "Kennard-Stone")]:
        for subset in ["Train", "Test"]:
            mask_set = split_df[split_col] == subset
            pts = split_df.loc[mask_set, ["tSNE1", "tSNE2"]].values
            metrics = compute_dispersity(pts)
            metrics["Split"] = split_name
            metrics["Set"] = subset
            metrics["n"] = len(pts)
            dispersity_rows.append(metrics)

    disp_df = pd.DataFrame(dispersity_rows)[["Split", "Set", "n", "Mean dist to centroid",
                                              "Std dist to centroid", "Mean NN distance",
                                              "Median NN distance", "Convex hull area", "Max spread (diameter)"]]

    st.dataframe(disp_df.style.format({
        "Mean dist to centroid": "{:.2f}",
        "Std dist to centroid": "{:.2f}",
        "Mean NN distance": "{:.2f}",
        "Median NN distance": "{:.2f}",
        "Convex hull area": "{:.1f}",
        "Max spread (diameter)": "{:.2f}",
    }), use_container_width=True, hide_index=True)

    # Visual comparison
    col1, col2 = st.columns(2)
    with col1:
        compare_metric = "Mean dist to centroid"
        chart_data = disp_df[["Split", "Set", compare_metric]].copy()
        chart_data["label"] = chart_data["Split"] + " — " + chart_data["Set"]
        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X("label:N", title="", sort=None),
            y=alt.Y(f"{compare_metric}:Q", title=compare_metric),
            color=alt.Color("Set:N", scale=alt.Scale(domain=["Train", "Test"], range=["#3498db", "#e74c3c"])),
            tooltip=["Split:N", "Set:N", f"{compare_metric}:Q"]
        ).properties(height=300, title="Mean Distance to Centroid (t-SNE)")
        st.altair_chart(chart, use_container_width=True)

    with col2:
        compare_metric = "Convex hull area"
        chart_data = disp_df[["Split", "Set", compare_metric]].copy()
        chart_data["label"] = chart_data["Split"] + " — " + chart_data["Set"]
        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X("label:N", title="", sort=None),
            y=alt.Y(f"{compare_metric}:Q", title=compare_metric),
            color=alt.Color("Set:N", scale=alt.Scale(domain=["Train", "Test"], range=["#3498db", "#e74c3c"])),
            tooltip=["Split:N", "Set:N", f"{compare_metric}:Q"]
        ).properties(height=300, title="Convex Hull Area (t-SNE)")
        st.altair_chart(chart, use_container_width=True)

    st.markdown("""
    **Interpretation:**
    - **Mean dist to centroid**: Higher = more dispersed/spread out
    - **Mean NN distance**: Higher = points are more isolated (sparse); lower = densely packed
    - **Convex hull area**: Larger = set covers more of the t-SNE space
    - **Random split**: Both train and test should have similar dispersity (uniform sampling)
    - **KS split**: Training should have higher dispersity (covers extremes), test is more compact (interior)
    """)
