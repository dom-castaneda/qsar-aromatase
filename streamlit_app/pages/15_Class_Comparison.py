import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import PROJECT_ROOT

st.set_page_config(page_title="Class Comparison", layout="wide")
st.title("15. Statistical Comparison of Bioactivity Classes")
st.caption(
    "Kruskal-Wallis test (non-parametric) comparing molecular descriptor distributions across "
    "active (pchembl > 7), intermediate (6-7), and inactive (< 6) classes. "
    "Post-hoc pairwise Mann-Whitney U with Bonferroni correction."
)


@st.cache_data
def load_stats():
    path = PROJECT_ROOT / "data" / "models" / "class_comparison_stats.csv"
    return pd.read_csv(path)


@st.cache_data
def load_descriptors():
    DATA_DIR = PROJECT_ROOT / "data" / "processed"
    df_full = pd.read_csv(DATA_DIR / "aromatase_bioactivity_clean.csv")
    mask = (df_full["standard_relation"] == "=") & df_full["pchembl_value"].notna()
    df = df_full[mask].reset_index(drop=True)

    def classify(val):
        if val > 7: return "active"
        elif val < 6: return "inactive"
        else: return "intermediate"

    df["bioactivity_class"] = df["pchembl_value"].apply(classify)

    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.logger().setLevel(RDLogger.ERROR)

    desc_names = ["MW", "LogP", "HBA", "HBD", "TPSA", "RotBonds", "AromaticRings", "FractionCSP3"]

    def calc(smi):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            return [np.nan] * 8
        return [
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
            Descriptors.NumHAcceptors(mol), Descriptors.NumHDonors(mol),
            Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol), Descriptors.FractionCSP3(mol),
        ]

    desc_data = df["canonical_smiles"].apply(calc).tolist()
    desc_df = pd.DataFrame(desc_data, columns=desc_names)
    return pd.concat([df[["molecule_chembl_id", "pchembl_value", "bioactivity_class"]], desc_df], axis=1), desc_names


stats_df = load_stats()
desc_full, desc_names = load_descriptors()

# --- Summary Table ---
with st.container(border=True):
    st.subheader("Kruskal-Wallis Results")
    st.caption("All 8 molecular descriptors show statistically significant differences between classes (p < 0.05).")

    display = stats_df[["descriptor", "H_statistic", "p_value", "eta_squared",
                        "median_active", "median_intermediate", "median_inactive"]].copy()
    display.columns = ["Descriptor", "H", "p-value", "Effect Size (eta2)",
                       "Median (active)", "Median (intermediate)", "Median (inactive)"]
    display["Significant"] = display["p-value"] < 0.05
    display = display.sort_values("Effect Size (eta2)", ascending=False).reset_index(drop=True)
    display.index += 1
    st.dataframe(display, use_container_width=True)

# --- Effect Size Bar Chart ---
with st.container(border=True):
    st.subheader("Effect Size Ranking")
    st.caption("Eta-squared measures the proportion of variance explained by class membership. "
               "MW has the strongest class-discriminating power.")

    fig = px.bar(
        stats_df.sort_values("eta_squared", ascending=True),
        x="eta_squared", y="descriptor", orientation="h",
        color="eta_squared", color_continuous_scale="Reds",
    )
    fig.update_layout(height=350, xaxis_title="Effect Size (eta-squared)",
                      yaxis_title="", showlegend=False)
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# --- Box Plots ---
with st.container(border=True):
    st.subheader("Distribution Comparison")
    st.caption("Box plots showing descriptor distributions per bioactivity class.")

    selected_desc = st.selectbox("Select descriptor", desc_names, index=0)

    fig_box = px.box(
        desc_full, x="bioactivity_class", y=selected_desc,
        color="bioactivity_class",
        category_orders={"bioactivity_class": ["active", "intermediate", "inactive"]},
        color_discrete_map={"active": "#2ca02c", "intermediate": "#ff7f0e", "inactive": "#d62728"},
    )
    fig_box.update_layout(height=450, showlegend=False,
                          xaxis_title="Bioactivity Class", yaxis_title=selected_desc)
    st.plotly_chart(fig_box, use_container_width=True)

    # Show stats for selected descriptor
    row = stats_df[stats_df["descriptor"] == selected_desc].iloc[0]
    col1, col2, col3 = st.columns(3)
    col1.metric("H statistic", f"{row['H_statistic']:.2f}")
    col2.metric("p-value", f"{row['p_value']:.2e}")
    col3.metric("Effect size", f"{row['eta_squared']:.4f}")

# --- Pairwise Significance ---
with st.container(border=True):
    st.subheader("Pairwise Significance (Bonferroni-corrected)")
    st.caption("Which class pairs differ significantly for each descriptor.")

    pairs_data = []
    for _, row in stats_df.iterrows():
        pairs_data.append({"Descriptor": row["descriptor"],
                           "Active vs Intermediate": "Yes" if row["active_vs_intermediate"] < 0.05 else "No",
                           "Active vs Inactive": "Yes" if row["active_vs_inactive"] < 0.05 else "No",
                           "Intermediate vs Inactive": "Yes" if row["intermediate_vs_inactive"] < 0.05 else "No"})

    pairs_df = pd.DataFrame(pairs_data)
    st.dataframe(pairs_df, use_container_width=True, hide_index=True)

    st.markdown("""
    **Key findings:**
    - **MW** (molecular weight): Strongest differentiator — actives are significantly heavier than inactives
    - **FractionCSP3**: Actives have lower sp3 fraction (more aromatic/flat structures)
    - **AromaticRings**: Actives have more aromatic rings
    - **HBD**: Actives have fewer H-bond donors
    """)
