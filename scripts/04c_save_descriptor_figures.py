"""
Molecular descriptors vs pchembl_value scatter/regression exploration.
Saves figures to data/figures/eda/
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.logger().setLevel(RDLogger.ERROR)
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150

DATA_DIR = r"C:\Users\dommy\0_self_projects\qsar_aromatase\data\processed"
FIG_DIR = r"C:\Users\dommy\0_self_projects\qsar_aromatase\data\figures\eda"

df = pd.read_csv(f"{DATA_DIR}/aromatase_bioactivity_clean.csv")

# Compute descriptors
def compute_descriptors(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [np.nan] * 10
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.FractionCSP3(mol),
        mol.GetNumHeavyAtoms(),
        Descriptors.NumHeteroatoms(mol),
    ]

desc_cols = ["MW", "LogP", "HBA", "HBD", "TPSA", "RotBonds", "AromaticRings", "FractionCSP3", "HeavyAtoms", "NumHeteroatoms"]
print("Computing molecular descriptors...")
desc_data = df["canonical_smiles"].apply(compute_descriptors)
desc_df = pd.DataFrame(desc_data.tolist(), columns=desc_cols, index=df.index)
for col in desc_cols:
    df[col] = desc_df[col]

# Activity class
def classify_activity(val):
    if pd.isna(val): return "Unknown"
    if val >= 6.5: return "Active"
    elif val < 5.0: return "Inactive"
    else: return "Intermediate"

df["activity_class"] = df["pchembl_value"].apply(classify_activity)

# Filter to rows with pchembl
df_pch = df.dropna(subset=["pchembl_value"]).copy()
print(f"Molecules with pchembl_value: {len(df_pch)}")

# ===========================================================
# Fig 16: Individual descriptor vs pchembl scatter with regression
# ===========================================================
print("Saving: 16_descriptors_vs_pchembl.png")

fig, axes = plt.subplots(2, 5, figsize=(24, 10))
axes = axes.ravel()

color_map = {"Active": "#2ecc71", "Intermediate": "#f39c12", "Inactive": "#e74c3c"}

for i, col in enumerate(desc_cols):
    ax = axes[i]
    for cls in ["Inactive", "Intermediate", "Active"]:
        sub = df_pch[df_pch["activity_class"] == cls]
        ax.scatter(sub[col], sub["pchembl_value"], c=color_map[cls],
                   alpha=0.3, s=8, edgecolors="none", label=cls)

    # Linear regression line
    valid = df_pch[[col, "pchembl_value"]].dropna()
    slope, intercept, r, p, se = stats.linregress(valid[col], valid["pchembl_value"])
    x_range = np.linspace(valid[col].min(), valid[col].max(), 100)
    ax.plot(x_range, slope * x_range + intercept, "k--", lw=1.5, alpha=0.7)

    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    ax.set_title(f"{col}\nr={r:.3f} {sig}", fontweight="bold", fontsize=11)
    ax.set_xlabel(col)
    if i % 5 == 0:
        ax.set_ylabel("pchembl_value")

axes[0].legend(markerscale=2, fontsize=8, loc="upper right")
plt.suptitle("Molecular Descriptors vs pchembl_value", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/16_descriptors_vs_pchembl.png", bbox_inches="tight")
plt.close()

# ===========================================================
# Fig 17: Pairplot of top descriptors coloured by activity
# ===========================================================
print("Saving: 17_descriptor_pairplot.png")

top_desc = ["MW", "LogP", "TPSA", "AromaticRings", "FractionCSP3"]
plot_data = df_pch[top_desc + ["pchembl_value", "activity_class"]].dropna()
plot_data = plot_data[plot_data["activity_class"].isin(["Active", "Inactive"])]

g = sns.pairplot(plot_data, vars=top_desc + ["pchembl_value"],
                 hue="activity_class", palette={"Active": "#2ecc71", "Inactive": "#e74c3c"},
                 diag_kind="kde", plot_kws={"alpha": 0.4, "s": 12, "edgecolor": "none"},
                 diag_kws={"fill": True, "alpha": 0.5})
g.figure.suptitle("Descriptor Pairplot (Active vs Inactive)", fontsize=14, fontweight="bold", y=1.01)
plt.savefig(f"{FIG_DIR}/17_descriptor_pairplot.png", bbox_inches="tight")
plt.close()

# ===========================================================
# Fig 18: Violin plots - descriptors by activity class
# ===========================================================
print("Saving: 18_descriptors_violin_by_class.png")

fig, axes = plt.subplots(2, 5, figsize=(22, 9))
axes = axes.ravel()
class_order = ["Active", "Intermediate", "Inactive"]
palette = {"Active": "#2ecc71", "Intermediate": "#f39c12", "Inactive": "#e74c3c"}

for i, col in enumerate(desc_cols):
    ax = axes[i]
    plot_sub = df_pch[df_pch["activity_class"].isin(class_order)]
    sns.violinplot(data=plot_sub, x="activity_class", y=col, ax=ax,
                   order=class_order, palette=palette, inner="box", cut=0)
    ax.set_title(col, fontweight="bold")
    ax.set_xlabel("")
    if i % 5 != 0:
        ax.set_ylabel("")

    # Kruskal-Wallis test
    groups = [plot_sub[plot_sub["activity_class"] == c][col].dropna() for c in class_order]
    h_stat, kw_p = stats.kruskal(*groups)
    sig = "***" if kw_p < 0.001 else "**" if kw_p < 0.01 else "*" if kw_p < 0.05 else "ns"
    ax.text(0.95, 0.95, f"KW {sig}", transform=ax.transAxes, ha="right", va="top", fontsize=9)

plt.suptitle("Molecular Descriptors by Activity Class (Kruskal-Wallis)", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/18_descriptors_violin_by_class.png", bbox_inches="tight")
plt.close()

# ===========================================================
# Fig 19: 2D hexbin density - MW vs LogP coloured by pchembl
# ===========================================================
print("Saving: 19_mw_logp_hexbin.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Hexbin with pchembl as colour
valid = df_pch[["MW", "LogP", "pchembl_value"]].dropna()
hb = axes[0].hexbin(valid["MW"], valid["LogP"], C=valid["pchembl_value"],
                     gridsize=30, cmap="RdYlGn", mincnt=1, reduce_C_function=np.mean)
axes[0].set_xlabel("Molecular Weight (Da)")
axes[0].set_ylabel("LogP")
axes[0].set_title("MW vs LogP (mean pchembl per bin)")
axes[0].axvline(500, color="grey", ls="--", lw=0.8)
axes[0].axhline(5, color="grey", ls="--", lw=0.8)
plt.colorbar(hb, ax=axes[0], label="Mean pchembl_value")

# Density contour
sns.kdeplot(data=df_pch, x="MW", y="LogP", hue="activity_class",
            hue_order=["Active", "Inactive"], palette={"Active": "#2ecc71", "Inactive": "#e74c3c"},
            levels=5, ax=axes[1], fill=False, linewidths=1.5)
axes[1].set_xlabel("Molecular Weight (Da)")
axes[1].set_ylabel("LogP")
axes[1].set_title("MW vs LogP Density Contours (Active vs Inactive)")
axes[1].axvline(500, color="grey", ls="--", lw=0.8)
axes[1].axhline(5, color="grey", ls="--", lw=0.8)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/19_mw_logp_hexbin.png", bbox_inches="tight")
plt.close()

print("\nDone! 4 new figures saved (16-19)")
