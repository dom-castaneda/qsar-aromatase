"""
Applicability Domain Analysis — PCA Bounding Box

Fits PCA on the training set (AP2D_Count fingerprint, Random split),
defines a bounding box from training PCA scores, and evaluates what
fraction of test set molecules fall within the domain.
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = "../data/processed"
FP_DIR = "../data/fingerprints_filtered"
SPLIT_DIR = "../data/splits"
OUTPUT_DIR = "../data/models"
FIG_DIR = "../data/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# --- Load Data ---
print("Loading data...")
df_full = pd.read_csv(f"{DATA_DIR}/aromatase_bioactivity_clean.csv")
mask = (df_full["standard_relation"] == "=") & df_full["pchembl_value"].notna()
df = df_full[mask].reset_index(drop=True)

fp_full = pd.read_csv(f"{FP_DIR}/fingerprints_atompairs2d_count.csv")
fp = fp_full[mask.values].reset_index(drop=True)
fp_cols = [c for c in fp.columns if c != "molecule_chembl_id"]
X_all = np.nan_to_num(fp[fp_cols].values.astype(np.float32), nan=0.0)

# Train/test split
train_ids = set(pd.read_csv(f"{SPLIT_DIR}/random_train.csv")["molecule_chembl_id"])
test_ids = set(pd.read_csv(f"{SPLIT_DIR}/random_test.csv")["molecule_chembl_id"])
train_mask = df["molecule_chembl_id"].isin(train_ids).values
test_mask = df["molecule_chembl_id"].isin(test_ids).values

X_train = X_all[train_mask]
X_test = X_all[test_mask]
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Features: {X_train.shape[1]}")

# --- Standardize ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Fit PCA (95% cumulative variance) ---
pca_full = PCA(n_components=0.95, random_state=42)
train_scores = pca_full.fit_transform(X_train_scaled)
test_scores = pca_full.transform(X_test_scaled)

n_components = pca_full.n_components_
variance_explained = pca_full.explained_variance_ratio_.sum()
print(f"\nPCA: {n_components} components explain {variance_explained*100:.1f}% variance")

# --- Define Bounding Box ---
train_min = train_scores.min(axis=0)
train_max = train_scores.max(axis=0)

# Check each test molecule: inside if ALL PCs are within [min, max]
inside_mask = np.all((test_scores >= train_min) & (test_scores <= train_max), axis=1)
n_inside = inside_mask.sum()
n_outside = (~inside_mask).sum()
coverage = n_inside / len(inside_mask) * 100

print(f"\nApplicability Domain (PCA Bounding Box):")
print(f"  Test molecules inside AD:  {n_inside} ({coverage:.1f}%)")
print(f"  Test molecules outside AD: {n_outside} ({100-coverage:.1f}%)")

# Also check per-component: how many PCs does each outside molecule violate?
if n_outside > 0:
    outside_scores = test_scores[~inside_mask]
    violations = ((outside_scores < train_min) | (outside_scores > train_max)).sum(axis=1)
    print(f"  Avg PCs violated (outside molecules): {violations.mean():.1f}")
    print(f"  Max PCs violated: {violations.max()}")

# --- Visualization: PC1 vs PC2 ---
fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(train_scores[:, 0], train_scores[:, 1],
           c="steelblue", alpha=0.3, s=15, label=f"Train ({X_train.shape[0]})")
ax.scatter(test_scores[inside_mask, 0], test_scores[inside_mask, 1],
           c="green", alpha=0.5, s=20, label=f"Test inside AD ({n_inside})")
ax.scatter(test_scores[~inside_mask, 0], test_scores[~inside_mask, 1],
           c="red", alpha=0.7, s=30, marker="x", label=f"Test outside AD ({n_outside})")

# Draw bounding box on PC1-PC2
rect_x = [train_min[0], train_max[0], train_max[0], train_min[0], train_min[0]]
rect_y = [train_min[1], train_min[1], train_max[1], train_max[1], train_min[1]]
ax.plot(rect_x, rect_y, "k--", linewidth=1.5, alpha=0.7, label="AD boundary (PC1-PC2)")

ax.set_xlabel(f"PC1 ({pca_full.explained_variance_ratio_[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({pca_full.explained_variance_ratio_[1]*100:.1f}% var)")
ax.set_title("Applicability Domain — PCA Bounding Box (AP2D_Count, Random Split)")
ax.legend(loc="upper right")
plt.tight_layout()

fig_path = f"{FIG_DIR}/applicability_domain_pca.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {fig_path}")
plt.close()

# --- Save Results ---
results = {
    "fingerprint": "AP2D_Count",
    "split": "Random",
    "n_components": int(n_components),
    "variance_explained": float(variance_explained),
    "n_train": int(X_train.shape[0]),
    "n_test": int(X_test.shape[0]),
    "n_test_inside": int(n_inside),
    "n_test_outside": int(n_outside),
    "coverage_pct": float(coverage),
    "bounding_box_min": train_min.tolist(),
    "bounding_box_max": train_max.tolist(),
}

out_path = f"{OUTPUT_DIR}/applicability_domain.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved: {out_path}")
print("\nDone.")
