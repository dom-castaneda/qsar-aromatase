"""
Re-run the EDA notebook cells, saving each figure to data/figures/eda/.
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
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

RDLogger.logger().setLevel(RDLogger.ERROR)
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["figure.figsize"] = (10, 6)

DATA_DIR = r"C:\Users\dommy\0_self_projects\qsar_aromatase\data\processed"
FP_DIR = r"C:\Users\dommy\0_self_projects\qsar_aromatase\data\fingerprints"
FIG_DIR = r"C:\Users\dommy\0_self_projects\qsar_aromatase\data\figures\eda"

df = pd.read_csv(f"{DATA_DIR}/aromatase_bioactivity_clean.csv")
print(f"Loaded {len(df)} records")

# Activity classes
def classify_activity(val):
    if pd.isna(val): return "Unknown"
    if val >= 6.5: return "Active"
    elif val < 5.0: return "Inactive"
    else: return "Intermediate"

df["activity_class"] = df["pchembl_value"].apply(classify_activity)

# ===========================================================
# Fig 1: Bioactivity distribution
# ===========================================================
print("Saving: 01_bioactivity_distribution.png")
pch = df.dropna(subset=["pchembl_value"])
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for st in ["IC50", "Ki", "pIC50"]:
    sub = pch[pch["standard_type"] == st]["pchembl_value"]
    if len(sub) > 0:
        axes[0].hist(sub, bins=40, alpha=0.5, label=f"{st} (n={len(sub)})", density=True)
        sub.plot.kde(ax=axes[0], label=f"_{st}")
axes[0].set_xlabel("pchembl_value (-log10 M)")
axes[0].set_ylabel("Density")
axes[0].set_title("Distribution of pchembl_value by Assay Type")
axes[0].axvline(6.5, color="green", ls="--", lw=1, label="Active threshold (6.5)")
axes[0].axvline(5.0, color="red", ls="--", lw=1, label="Inactive threshold (5.0)")
axes[0].legend()
sns.boxplot(data=pch, x="standard_type", y="pchembl_value", ax=axes[1],
            order=["IC50", "Ki", "pIC50"], palette="Set2")
axes[1].set_title("pchembl_value by Standard Type")
axes[1].set_ylabel("pchembl_value (-log10 M)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_bioactivity_distribution.png", bbox_inches="tight")
plt.close()

# ===========================================================
# Fig 2: Activity class balance
# ===========================================================
print("Saving: 02_activity_classes.png")
class_counts = df["activity_class"].value_counts()
colors = {"Active": "#2ecc71", "Intermediate": "#f39c12", "Inactive": "#e74c3c", "Unknown": "#95a5a6"}
order = ["Active", "Intermediate", "Inactive", "Unknown"]
present = [c for c in order if c in class_counts.index]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].bar(present, [class_counts[c] for c in present],
            color=[colors[c] for c in present], edgecolor="black", linewidth=0.5)
for i, c in enumerate(present):
    axes[0].text(i, class_counts[c] + 20, str(class_counts[c]), ha="center", fontweight="bold")
axes[0].set_ylabel("Count")
axes[0].set_title("Activity Class Distribution")
known = class_counts.drop("Unknown", errors="ignore")
axes[1].pie(known, labels=known.index, autopct="%1.1f%%",
            colors=[colors[c] for c in known.index], startangle=90)
axes[1].set_title("Activity Classes (pchembl known only)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_activity_classes.png", bbox_inches="tight")
plt.close()

# ===========================================================
# Fig 3: Missing values
# ===========================================================
print("Saving: 03_missing_values.png")
null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
null_pct = null_pct[null_pct > 0]
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(null_pct.index, null_pct.values, color="salmon", edgecolor="black", linewidth=0.5)
ax.set_xlabel("% Missing")
ax.set_title("Missing Values by Column")
for i, (col, pct) in enumerate(null_pct.items()):
    ax.text(pct + 0.3, i, f"{pct:.1f}% ({df[col].isnull().sum()})", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_missing_values.png", bbox_inches="tight")
plt.close()

# ===========================================================
# Fig 4: Record multiplicity
# ===========================================================
print("Saving: 04_record_multiplicity.png")
mol_assay = df.groupby("molecule_chembl_id").agg(n_records=("activity_id", "count"))
rec_dist = mol_assay["n_records"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(rec_dist.index[:15], rec_dist.values[:15], color="steelblue", edgecolor="black", linewidth=0.5)
ax.set_xlabel("Number of records per molecule")
ax.set_ylabel("Count of molecules")
ax.set_title("Record Multiplicity per Molecule")
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_record_multiplicity.png", bbox_inches="tight")
plt.close()

# ===========================================================
# Fig 5: Temporal trends
# ===========================================================
print("Saving: 05_temporal_trends.png")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
year_counts = df["document_year"].value_counts().sort_index()
axes[0].bar(year_counts.index, year_counts.values, color="steelblue", edgecolor="black", linewidth=0.3)
axes[0].set_xlabel("Publication Year")
axes[0].set_ylabel("Number of Bioactivity Records")
axes[0].set_title("Records per Year")
axes[0].tick_params(axis="x", rotation=45)

pch_year = df.dropna(subset=["pchembl_value"]).copy()
pch_year["decade"] = (pch_year["document_year"] // 10) * 10
decade_order = sorted(pch_year["decade"].unique())
sns.violinplot(data=pch_year, x="decade", y="pchembl_value", ax=axes[1],
               order=decade_order, palette="coolwarm", inner="box", cut=0)
axes[1].set_xlabel("Decade")
axes[1].set_ylabel("pchembl_value")
axes[1].set_title("Potency Distribution by Decade")
axes[1].axhline(6.5, color="green", ls="--", lw=1, alpha=0.7)
axes[1].axhline(5.0, color="red", ls="--", lw=1, alpha=0.7)

first_seen = df.groupby("molecule_chembl_id")["document_year"].min().sort_values()
cumul = first_seen.value_counts().sort_index().cumsum()
axes[2].plot(cumul.index, cumul.values, "o-", color="darkgreen", markersize=3)
axes[2].fill_between(cumul.index, cumul.values, alpha=0.15, color="green")
axes[2].set_xlabel("Year")
axes[2].set_ylabel("Cumulative Unique Molecules")
axes[2].set_title("Cumulative Molecule Discovery")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_temporal_trends.png", bbox_inches="tight")
plt.close()

# ===========================================================
# Fig 6: Molecular property histograms
# ===========================================================
print("Saving: 06_molecular_properties.png")
desc_cols = ["MW", "LogP", "HBA", "HBD", "TPSA", "RotBonds", "AromaticRings", "FractionCSP3"]
def compute_descriptors(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return [np.nan] * 8
    return [Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.NumHAcceptors(mol),
            Descriptors.NumHDonors(mol), Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol), Descriptors.FractionCSP3(mol)]

desc_data = df["canonical_smiles"].apply(compute_descriptors)
desc_df = pd.DataFrame(desc_data.tolist(), columns=desc_cols, index=df.index)
for col in desc_cols:
    df[col] = desc_df[col]

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.ravel()
lipinski_limits = {"MW": 500, "LogP": 5, "HBA": 10, "HBD": 5}
colors_prop = sns.color_palette("tab10", 8)
for i, col in enumerate(desc_cols):
    ax = axes[i]
    data = df[col].dropna()
    ax.hist(data, bins=40, color=colors_prop[i], alpha=0.7, edgecolor="black", linewidth=0.3)
    ax.set_title(col, fontweight="bold")
    ax.set_ylabel("Count")
    if col in lipinski_limits:
        ax.axvline(lipinski_limits[col], color="red", ls="--", lw=1.5, label=f"Ro5 limit ({lipinski_limits[col]})")
        ax.legend(fontsize=8)
    ax.set_xlabel(col)
plt.suptitle("Molecular Property Distributions", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/06_molecular_properties.png", bbox_inches="tight")
plt.close()

# ===========================================================
# Fig 7: Lipinski compliance
# ===========================================================
print("Saving: 07_lipinski_compliance.png")
def lipinski_pass(row):
    v = 0
    if row["MW"] > 500: v += 1
    if row["LogP"] > 5: v += 1
    if row["HBA"] > 10: v += 1
    if row["HBD"] > 5: v += 1
    return v
df["Ro5_violations"] = df.apply(lipinski_pass, axis=1)
ro5_counts = df["Ro5_violations"].value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
bar_colors = ["#2ecc71", "#f39c12", "#e74c3c", "#c0392b"][:len(ro5_counts)]
axes[0].bar(ro5_counts.index, ro5_counts.values, color=bar_colors, edgecolor="black", linewidth=0.5)
for v, c in ro5_counts.items():
    axes[0].text(v, c + 20, str(c), ha="center", fontweight="bold")
axes[0].set_xlabel("Number of Ro5 Violations")
axes[0].set_ylabel("Count")
axes[0].set_title("Lipinski Rule-of-5 Violations")
drug_like_pct = (df["Ro5_violations"] == 0).mean() * 100
axes[1].pie([drug_like_pct, 100 - drug_like_pct],
            labels=["Drug-like\n(0 violations)", "Non-drug-like"],
            colors=["#2ecc71", "#e74c3c"], autopct="%1.1f%%", startangle=90)
axes[1].set_title("Drug-likeness (Ro5)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/07_lipinski_compliance.png", bbox_inches="tight")
plt.close()

# ===========================================================
# Fig 8: MW/LogP vs activity
# ===========================================================
print("Saving: 08_mw_logp_activity.png")
color_map = {"Active": "#2ecc71", "Intermediate": "#f39c12", "Inactive": "#e74c3c", "Unknown": "#95a5a6"}
plot_df = df.dropna(subset=["MW", "LogP"])
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for cls in ["Inactive", "Intermediate", "Active"]:
    sub = plot_df[plot_df["activity_class"] == cls]
    axes[0].scatter(sub["MW"], sub["LogP"], c=color_map[cls], label=cls, alpha=0.4, s=15, edgecolors="none")
axes[0].set_xlabel("Molecular Weight (Da)")
axes[0].set_ylabel("LogP")
axes[0].set_title("MW vs LogP by Activity Class")
axes[0].legend(markerscale=2)
axes[0].axvline(500, color="grey", ls="--", lw=0.8)
axes[0].axhline(5, color="grey", ls="--", lw=0.8)

sub = df.dropna(subset=["MW", "pchembl_value"])
sc = axes[1].scatter(sub["MW"], sub["pchembl_value"], c=sub["pchembl_value"], cmap="RdYlGn", s=12, alpha=0.5, edgecolors="none")
axes[1].set_xlabel("Molecular Weight (Da)")
axes[1].set_ylabel("pchembl_value")
axes[1].set_title("MW vs Potency")
plt.colorbar(sc, ax=axes[1], label="pchembl")

sub2 = df.dropna(subset=["LogP", "pchembl_value"])
sc2 = axes[2].scatter(sub2["LogP"], sub2["pchembl_value"], c=sub2["pchembl_value"], cmap="RdYlGn", s=12, alpha=0.5, edgecolors="none")
axes[2].set_xlabel("LogP")
axes[2].set_ylabel("pchembl_value")
axes[2].set_title("LogP vs Potency")
plt.colorbar(sc2, ax=axes[2], label="pchembl")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/08_mw_logp_activity.png", bbox_inches="tight")
plt.close()

# ===========================================================
# Fig 9: PCA chemical space
# ===========================================================
print("Saving: 09_pca_chemical_space.png")
fp_ecfp4 = pd.read_csv(f"{FP_DIR}/fingerprints_ecfp4.csv")
X_ecfp4 = fp_ecfp4.iloc[:, 1:].values

pca = PCA(n_components=20, random_state=42)
X_pca = pca.fit_transform(X_ecfp4)
var_explained = pca.explained_variance_ratio_

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
axes[0].bar(range(1, 21), var_explained * 100, color="steelblue", edgecolor="black", linewidth=0.3)
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Variance Explained (%)")
axes[0].set_title("PCA Scree Plot (ECFP4)")
cumvar = np.cumsum(var_explained) * 100
ax_twin = axes[0].twinx()
ax_twin.plot(range(1, 21), cumvar, "ro-", markersize=4)
ax_twin.set_ylabel("Cumulative %", color="red")
ax_twin.tick_params(axis="y", labelcolor="red")

color_map_cls = {"Active": "#2ecc71", "Intermediate": "#f39c12", "Inactive": "#e74c3c", "Unknown": "#95a5a6"}
for cls in ["Unknown", "Inactive", "Intermediate", "Active"]:
    mask = df["activity_class"] == cls
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], c=color_map_cls[cls], label=cls, alpha=0.4, s=12, edgecolors="none")
axes[1].set_xlabel(f"PC1 ({var_explained[0]*100:.1f}%)")
axes[1].set_ylabel(f"PC2 ({var_explained[1]*100:.1f}%)")
axes[1].set_title("PCA - Activity Class")
axes[1].legend(markerscale=2)

pch_vals = df["pchembl_value"].values
mask_known = ~np.isnan(pch_vals)
sc = axes[2].scatter(X_pca[mask_known, 0], X_pca[mask_known, 1], c=pch_vals[mask_known], cmap="RdYlGn", s=12, alpha=0.5, edgecolors="none")
axes[2].set_xlabel(f"PC1 ({var_explained[0]*100:.1f}%)")
axes[2].set_ylabel(f"PC2 ({var_explained[1]*100:.1f}%)")
axes[2].set_title("PCA - Potency (pchembl)")
plt.colorbar(sc, ax=axes[2], label="pchembl_value")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/09_pca_chemical_space.png", bbox_inches="tight")
plt.close()

# ===========================================================
# Fig 10: t-SNE
# ===========================================================
print("Saving: 10_tsne_chemical_space.png (running t-SNE...)")
pca50 = PCA(n_components=50, random_state=42)
X_pca50 = pca50.fit_transform(X_ecfp4)
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, init="pca")
X_tsne = tsne.fit_transform(X_pca50)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for cls in ["Unknown", "Inactive", "Intermediate", "Active"]:
    mask = df["activity_class"] == cls
    axes[0].scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=color_map_cls[cls], label=cls, alpha=0.4, s=10, edgecolors="none")
axes[0].set_xlabel("t-SNE 1")
axes[0].set_ylabel("t-SNE 2")
axes[0].set_title("t-SNE - Activity Class")
axes[0].legend(markerscale=2)

sc = axes[1].scatter(X_tsne[mask_known, 0], X_tsne[mask_known, 1], c=pch_vals[mask_known], cmap="RdYlGn", s=10, alpha=0.5, edgecolors="none")
axes[1].set_xlabel("t-SNE 1")
axes[1].set_ylabel("t-SNE 2")
axes[1].set_title("t-SNE - Potency (pchembl)")
plt.colorbar(sc, ax=axes[1], label="pchembl_value")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/10_tsne_chemical_space.png", bbox_inches="tight")
plt.close()

# ===========================================================
# Fig 11: MACCS key frequencies
# ===========================================================
print("Saving: 11_maccs_frequencies.png")
fp_maccs = pd.read_csv(f"{FP_DIR}/fingerprints_maccs.csv")
maccs_bits = fp_maccs.iloc[:, 1:].values
maccs_freq = maccs_bits.mean(axis=0)

fig, axes = plt.subplots(2, 1, figsize=(16, 8))
axes[0].bar(range(len(maccs_freq)), maccs_freq, color="steelblue", edgecolor="none", width=1.0)
axes[0].set_xlabel("MACCS Key Index")
axes[0].set_ylabel("Fraction of Molecules")
axes[0].set_title("MACCS Key Frequency (167 bits)")
axes[0].axhline(0.05, color="red", ls="--", lw=0.8, label="5% threshold")
axes[0].axhline(0.95, color="orange", ls="--", lw=0.8, label="95% threshold")
axes[0].legend()

top20_idx = np.argsort(maccs_freq)[::-1][:20]
axes[1].barh(range(20), maccs_freq[top20_idx], color="darkgreen", edgecolor="black", linewidth=0.3)
axes[1].set_yticks(range(20))
axes[1].set_yticklabels([f"MACCS_{i}" for i in top20_idx])
axes[1].set_xlabel("Fraction of Molecules")
axes[1].set_title("Top 20 Most Frequent MACCS Keys")
axes[1].invert_yaxis()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/11_maccs_frequencies.png", bbox_inches="tight")
plt.close()

# ===========================================================
# Fig 12: KR FP sparsity
# ===========================================================
print("Saving: 12_kr_sparsity.png")
fp_kr = pd.read_csv(f"{FP_DIR}/fingerprints_kr.csv")
kr_bits = fp_kr.iloc[:, 1:].values
kr_freq = kr_bits.mean(axis=0)
bits_per_mol = kr_bits.sum(axis=1)

fp_subfp = pd.read_csv(f"{FP_DIR}/fingerprints_substruct.csv")
subfp_bits = fp_subfp.iloc[:, 1:].values

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].hist(kr_freq, bins=50, color="coral", edgecolor="black", linewidth=0.3)
axes[0].set_xlabel("Fraction of Molecules with Bit Set")
axes[0].set_ylabel("Count of KR Bits")
axes[0].set_title("KR Fingerprint Bit Frequency Distribution")
axes[0].set_yscale("log")

axes[1].hist(bits_per_mol, bins=50, color="mediumpurple", edgecolor="black", linewidth=0.3)
axes[1].set_xlabel("Number of KR Bits Set")
axes[1].set_ylabel("Count of Molecules")
axes[1].set_title("KR Bits per Molecule")

fps_info = {"MACCS\n(167)": maccs_bits, "SubFP\n(307)": subfp_bits, "ECFP4\n(1024)": X_ecfp4, "KR\n(4860)": kr_bits}
sp_df = pd.DataFrame([{"FP": k, "Density": v.mean()} for k, v in fps_info.items()])
axes[2].bar(sp_df["FP"], sp_df["Density"] * 100, color=["#3498db", "#2ecc71", "#e67e22", "#e74c3c"], edgecolor="black", linewidth=0.5)
axes[2].set_ylabel("Average Bit Density (%)")
axes[2].set_title("Fingerprint Density Comparison")
for i, row in sp_df.iterrows():
    axes[2].text(i, row["Density"] * 100 + 0.3, f"{row['Density']*100:.1f}%", ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/12_kr_sparsity.png", bbox_inches="tight")
plt.close()

# ===========================================================
# Fig 13: Property correlations with pchembl
# ===========================================================
print("Saving: 13_property_correlations.png")
pch_mask = df["pchembl_value"].notna()
prop_cols = ["MW", "LogP", "HBA", "HBD", "TPSA", "RotBonds", "AromaticRings", "FractionCSP3"]
corr_data = {}
for col in prop_cols:
    valid = pch_mask & df[col].notna()
    r, p = stats.pearsonr(df.loc[valid, col], df.loc[valid, "pchembl_value"])
    corr_data[col] = {"pearson_r": r, "p_value": p}
corr_df = pd.DataFrame(corr_data).T.sort_values("pearson_r", key=abs, ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
colors_corr = ["#2ecc71" if r > 0 else "#e74c3c" for r in corr_df["pearson_r"]]
ax.barh(corr_df.index, corr_df["pearson_r"], color=colors_corr, edgecolor="black", linewidth=0.5)
ax.set_xlabel("Pearson Correlation with pchembl_value")
ax.set_title("Molecular Property Correlations with Potency")
ax.axvline(0, color="black", lw=0.5)
for i, (idx, row) in enumerate(corr_df.iterrows()):
    sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else "ns"
    offset = 0.01 if row["pearson_r"] > 0 else -0.01
    ha = "left" if row["pearson_r"] > 0 else "right"
    ax.text(row["pearson_r"] + offset, i, f"r={row['pearson_r']:.3f} {sig}", va="center", ha=ha, fontsize=9)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/13_property_correlations.png", bbox_inches="tight")
plt.close()

# ===========================================================
# Fig 14: Top fingerprint bit correlations
# ===========================================================
print("Saving: 14_top_fp_bit_correlations.png")
pch_vals_clean = df.loc[pch_mask, "pchembl_value"].values

def top_correlated_bits(bit_matrix, bit_prefix, n_top=15):
    bit_sub = bit_matrix[pch_mask.values]
    correlations = []
    for i in range(bit_sub.shape[1]):
        bit_col = bit_sub[:, i]
        if bit_col.std() == 0: continue
        r, p = stats.pointbiserialr(bit_col, pch_vals_clean)
        correlations.append({"bit": f"{bit_prefix}_{i}", "r": r, "p": p, "freq": bit_col.mean()})
    cdf = pd.DataFrame(correlations)
    cdf["abs_r"] = cdf["r"].abs()
    return cdf.nlargest(n_top, "abs_r")

top_maccs = top_correlated_bits(maccs_bits, "MACCS", 15)
top_ecfp4 = top_correlated_bits(X_ecfp4, "ECFP4", 15)
top_kr = top_correlated_bits(kr_bits, "KR", 15)
top_all = pd.concat([top_maccs, top_ecfp4, top_kr]).nlargest(20, "abs_r")

fig, ax = plt.subplots(figsize=(10, 7))
colors_bits = ["#2ecc71" if r > 0 else "#e74c3c" for r in top_all["r"]]
ax.barh(range(len(top_all)), top_all["r"].values, color=colors_bits, edgecolor="black", linewidth=0.3)
ax.set_yticks(range(len(top_all)))
ax.set_yticklabels(top_all["bit"].values)
ax.set_xlabel("Point-Biserial Correlation with pchembl_value")
ax.set_title("Top 20 Fingerprint Bits Most Correlated with Potency")
ax.axvline(0, color="black", lw=0.5)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/14_top_fp_bit_correlations.png", bbox_inches="tight")
plt.close()

# ===========================================================
# Fig 15: Correlation heatmap
# ===========================================================
print("Saving: 15_correlation_heatmap.png")
prop_with_pch = prop_cols + ["pchembl_value"]
corr_matrix = df[prop_with_pch].corr()
fig, ax = plt.subplots(figsize=(9, 7))
mask_tri = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask_tri, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True, ax=ax, linewidths=0.5, linecolor="white")
ax.set_title("Property Correlation Matrix (incl. pchembl_value)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/15_correlation_heatmap.png", bbox_inches="tight")
plt.close()

print("\nDone! All 15 figures saved to data/figures/eda/")
