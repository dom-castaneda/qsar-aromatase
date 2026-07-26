"""
Statistical Comparison of 3 Bioactivity Classes

Compares molecular descriptor distributions across active (pchembl > 7),
intermediate (6-7), and inactive (< 6) classes using Kruskal-Wallis test
with post-hoc Dunn's test (Bonferroni correction).
"""
import sys
import os
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = "../data/processed"
OUTPUT_DIR = "../data/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load Data ---
print("Loading data...")
df_full = pd.read_csv(f"{DATA_DIR}/aromatase_bioactivity_clean.csv")
mask = (df_full["standard_relation"] == "=") & df_full["pchembl_value"].notna()
df = df_full[mask].reset_index(drop=True)

# Assign classes
def classify(val):
    if val > 7: return "active"
    elif val < 6: return "inactive"
    else: return "intermediate"

df["bioactivity_class"] = df["pchembl_value"].apply(classify)
print(f"Dataset: {len(df)} molecules")
print(f"\nClass distribution:")
print(df["bioactivity_class"].value_counts().to_string())

# --- Compute Molecular Descriptors ---
print("\nComputing molecular descriptors...")
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski
RDLogger.logger().setLevel(RDLogger.ERROR)

def compute_descriptors(smi):
    mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if mol is None:
        return [np.nan] * 8
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.FractionCSP3(mol),
    ]

desc_names = ["MW", "LogP", "HBA", "HBD", "TPSA", "RotBonds", "AromaticRings", "FractionCSP3"]
desc_data = df["canonical_smiles"].apply(compute_descriptors).tolist()
desc_df = pd.DataFrame(desc_data, columns=desc_names)
df = pd.concat([df, desc_df], axis=1)
print(f"  Computed {len(desc_names)} descriptors for {len(df)} molecules")

# --- Kruskal-Wallis + Dunn's Post-hoc ---
print("\n" + "=" * 70)
print("KRUSKAL-WALLIS TEST (non-parametric, 3 groups)")
print("=" * 70)
print(f"H0: All three classes have the same distribution")
print(f"Significance level: alpha = 0.05 (Bonferroni-corrected for post-hoc)\n")

groups = ["active", "intermediate", "inactive"]
results = []

for desc in desc_names:
    data_by_class = [df[df["bioactivity_class"] == g][desc].dropna().values for g in groups]
    
    # Kruskal-Wallis
    H, p_kw = stats.kruskal(*data_by_class)
    
    # Post-hoc: Dunn's test (pairwise Mann-Whitney with Bonferroni)
    pairs = [("active", "intermediate"), ("active", "inactive"), ("intermediate", "inactive")]
    pairwise_results = []
    for g1, g2 in pairs:
        d1 = df[df["bioactivity_class"] == g1][desc].dropna().values
        d2 = df[df["bioactivity_class"] == g2][desc].dropna().values
        U, p_mw = stats.mannwhitneyu(d1, d2, alternative="two-sided")
        # Bonferroni correction (3 comparisons)
        p_corrected = min(p_mw * 3, 1.0)
        pairwise_results.append({
            "pair": f"{g1} vs {g2}",
            "U": U,
            "p_raw": p_mw,
            "p_corrected": p_corrected,
            "significant": p_corrected < 0.05,
        })
    
    # Effect size (eta-squared for Kruskal-Wallis)
    N = sum(len(d) for d in data_by_class)
    eta_sq = (H - len(groups) + 1) / (N - len(groups))
    
    # Medians
    medians = {g: df[df["bioactivity_class"] == g][desc].median() for g in groups}
    
    results.append({
        "descriptor": desc,
        "H_statistic": H,
        "p_value": p_kw,
        "significant": p_kw < 0.05,
        "eta_squared": eta_sq,
        "median_active": medians["active"],
        "median_intermediate": medians["intermediate"],
        "median_inactive": medians["inactive"],
        "active_vs_intermediate": pairwise_results[0]["p_corrected"],
        "active_vs_inactive": pairwise_results[1]["p_corrected"],
        "intermediate_vs_inactive": pairwise_results[2]["p_corrected"],
    })
    
    sig = "***" if p_kw < 0.001 else "**" if p_kw < 0.01 else "*" if p_kw < 0.05 else "ns"
    print(f"  {desc:<15} H={H:>8.2f}  p={p_kw:.2e}  eta2={eta_sq:.4f}  {sig}")
    print(f"    Medians: active={medians['active']:.2f} | intermediate={medians['intermediate']:.2f} | inactive={medians['inactive']:.2f}")
    for pr in pairwise_results:
        sig_pw = "Y" if pr["significant"] else "N"
        print(f"    {pr['pair']:<30} p_corr={pr['p_corrected']:.4e} {sig_pw}")
    print()

# --- Save Results ---
results_df = pd.DataFrame(results)
out_path = f"{OUTPUT_DIR}/class_comparison_stats.csv"
results_df.to_csv(out_path, index=False)
print(f"Saved: {out_path}")

# --- Summary ---
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
sig_descriptors = results_df[results_df["significant"]]
print(f"\nSignificant descriptors (p < 0.05): {len(sig_descriptors)}/{len(results_df)}")
print(f"\nRanked by effect size (eta-squared):")
for _, row in results_df.sort_values("eta_squared", ascending=False).iterrows():
    sig = "***" if row["p_value"] < 0.001 else "ns"
    print(f"  {row['descriptor']:<15} eta2={row['eta_squared']:.4f}  {sig}")

print("\nDone.")
