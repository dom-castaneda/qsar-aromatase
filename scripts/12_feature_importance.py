"""
Feature Importance Analysis — Gini + Permutation

Trains Extra Trees Regressor on AP2D_Count (Random split), then computes:
1. Gini importance (Mean Decrease in Impurity)
2. Permutation importance on test set (Mean Decrease in Accuracy)
"""
import sys
import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.inspection import permutation_importance

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = "../data/processed"
FP_DIR = "../data/fingerprints_filtered"
SPLIT_DIR = "../data/splits"
OUTPUT_DIR = "../data/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

# --- Load Data ---
print("Loading data...")
df_full = pd.read_csv(f"{DATA_DIR}/aromatase_bioactivity_clean.csv")
mask = (df_full["standard_relation"] == "=") & df_full["pchembl_value"].notna()
df = df_full[mask].reset_index(drop=True)

fp_full = pd.read_csv(f"{FP_DIR}/fingerprints_atompairs2d_count.csv")
fp = fp_full[mask.values].reset_index(drop=True)
fp_cols = [c for c in fp.columns if c != "molecule_chembl_id"]
X_all = np.nan_to_num(fp[fp_cols].values.astype(np.float32), nan=0.0)
y = df["pchembl_value"].values

# Train/test split
train_ids = set(pd.read_csv(f"{SPLIT_DIR}/random_train.csv")["molecule_chembl_id"])
train_mask = df["molecule_chembl_id"].isin(train_ids).values
test_mask = ~train_mask

X_train, X_test = X_all[train_mask], X_all[test_mask]
y_train, y_test = y[train_mask], y[test_mask]
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Features: {X_train.shape[1]}")

# --- Train Model ---
print("\nTraining Extra Trees Regressor...")
t0 = time.time()
model = ExtraTreesRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
model.fit(X_train, y_train)
print(f"  Trained in {time.time()-t0:.1f}s")

from sklearn.metrics import r2_score
r2_train = r2_score(y_train, model.predict(X_train))
r2_test = r2_score(y_test, model.predict(X_test))
print(f"  R² train={r2_train:.4f} | test={r2_test:.4f}")

# --- Gini Importance ---
print("\nComputing Gini importance...")
gini_imp = model.feature_importances_

# --- Permutation Importance ---
print("Computing permutation importance (10 repeats on test set)...")
t0 = time.time()
perm_result = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,
    random_state=RANDOM_STATE,
    scoring="r2",
    n_jobs=-1,
)
print(f"  Completed in {time.time()-t0:.1f}s")

# --- Build Results DataFrame ---
results_df = pd.DataFrame({
    "feature": fp_cols,
    "gini_importance": gini_imp,
    "perm_importance_mean": perm_result.importances_mean,
    "perm_importance_std": perm_result.importances_std,
})
results_df = results_df.sort_values("gini_importance", ascending=False).reset_index(drop=True)

# --- Save ---
out_path = f"{OUTPUT_DIR}/feature_importance.csv"
results_df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path} ({len(results_df)} features)")

# --- Print Top 20 ---
print("\n" + "=" * 60)
print("TOP 20 by Gini Importance (MDI)")
print("=" * 60)
top_gini = results_df.nlargest(20, "gini_importance")
for i, row in top_gini.iterrows():
    print(f"  {row['feature']:<12} Gini={row['gini_importance']:.4f}  Perm={row['perm_importance_mean']:.4f}")

print("\n" + "=" * 60)
print("TOP 20 by Permutation Importance")
print("=" * 60)
top_perm = results_df.nlargest(20, "perm_importance_mean")
for i, row in top_perm.iterrows():
    print(f"  {row['feature']:<12} Perm={row['perm_importance_mean']:.4f} ± {row['perm_importance_std']:.4f}  Gini={row['gini_importance']:.4f}")

# Overlap
top_gini_set = set(top_gini["feature"])
top_perm_set = set(top_perm["feature"])
overlap = top_gini_set & top_perm_set
print(f"\nOverlap in top 20: {len(overlap)}/20 features appear in both rankings")
print(f"Shared: {sorted(overlap)}")
print("\nDone.")
