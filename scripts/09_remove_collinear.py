"""
Remove collinear features from all fingerprint CSVs.

For each pair of features with |Pearson r| > 0.95, drops one (the one with
lower mean absolute correlation with pchembl_value, if available, else random).

Saves reduced versions to data/fingerprints_reduced/ and a report of dropped pairs.
"""
import sys
import os
import pandas as pd
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

INPUT_DIR = "../data/fingerprints_filtered"
OUTPUT_DIR = "../data/fingerprints_reduced"
REPORT_DIR = "../data/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CORR_THRESHOLD = 0.95

# Load pchembl for tie-breaking (drop the feature less correlated with target)
DATA_DIR = "../data/processed"
df_full = pd.read_csv(f"{DATA_DIR}/aromatase_bioactivity_clean.csv")
mask = (df_full["standard_relation"] == "=") & df_full["pchembl_value"].notna()
y = df_full.loc[mask, "pchembl_value"].values

FP_FILES = [
    "fingerprints_atompairs2d.csv",
    "fingerprints_atompairs2d_count.csv",
    "fingerprints_cdk_extended.csv",
    "fingerprints_cdk_fp.csv",
    "fingerprints_cdk_graphonly.csv",
    "fingerprints_ecfp4.csv",
    "fingerprints_estate.csv",
    "fingerprints_estate_count.csv",
    "fingerprints_kr.csv",
    "fingerprints_kr_count.csv",
    "fingerprints_maccs.csv",
    "fingerprints_pubchem.csv",
    "fingerprints_substruct.csv",
    "fingerprints_substruct_count.csv",
]

all_pairs = []
summary = []

for fname in FP_FILES:
    path = os.path.join(INPUT_DIR, fname)
    df = pd.read_csv(path)
    id_col = df.columns[0]
    fp_cols = list(df.columns[1:])
    X = df[fp_cols].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0)
    n_original = len(fp_cols)

    # Apply same mask to align with pchembl
    X_masked = X[mask.values] if len(X) == len(df_full) else X

    # Remove zero-variance before correlation
    std = X.std(axis=0)
    valid_mask = std > 0
    valid_cols = [fp_cols[i] for i in range(len(fp_cols)) if valid_mask[i]]
    X_valid = X[:, valid_mask]

    if len(valid_cols) < 2:
        # Nothing to correlate
        df.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)
        summary.append({"File": fname, "Original": n_original, "Dropped": 0, "Kept": n_original})
        print(f"{fname:<45} {n_original:>5} -> {n_original:>5} (no pairs to check)")
        continue

    # Compute correlation matrix
    corr = np.corrcoef(X_valid, rowvar=False)

    # Find collinear pairs (upper triangle)
    # Compute correlation of each feature with pchembl for tie-breaking
    X_masked_valid = X_masked[:, valid_mask] if X_masked.shape[1] == X.shape[1] else X_valid
    if len(X_masked_valid) == len(y):
        target_corr = np.array([abs(np.corrcoef(X_masked_valid[:, i], y)[0, 1])
                                if X_masked_valid[:, i].std() > 0 else 0
                                for i in range(X_masked_valid.shape[1])])
    else:
        target_corr = np.zeros(len(valid_cols))

    # Identify features to drop
    to_drop = set()
    pairs_found = []

    for i in range(len(valid_cols)):
        if valid_cols[i] in to_drop:
            continue
        for j in range(i + 1, len(valid_cols)):
            if valid_cols[j] in to_drop:
                continue
            if abs(corr[i, j]) > CORR_THRESHOLD:
                # Drop the one less correlated with pchembl
                if target_corr[i] >= target_corr[j]:
                    drop_col = valid_cols[j]
                    keep_col = valid_cols[i]
                else:
                    drop_col = valid_cols[i]
                    keep_col = valid_cols[j]

                to_drop.add(drop_col)
                pairs_found.append({
                    "File": fname,
                    "Feature_1": keep_col,
                    "Feature_2": drop_col,
                    "Correlation": corr[i, j],
                    "Kept": keep_col,
                    "Dropped": drop_col,
                })

    # Save reduced CSV
    keep_cols = [c for c in fp_cols if c not in to_drop]
    reduced = df[[id_col] + keep_cols]
    reduced.to_csv(os.path.join(OUTPUT_DIR, fname), index=False)

    n_dropped = len(to_drop)
    n_kept = len(keep_cols)
    summary.append({"File": fname, "Original": n_original, "Dropped": n_dropped, "Kept": n_kept})
    all_pairs.extend(pairs_found)

    print(f"{fname:<45} {n_original:>5} -> {n_kept:>5} ({n_dropped} dropped, {len(pairs_found)} pairs)")

# Save reports
pairs_df = pd.DataFrame(all_pairs)
pairs_df.to_csv(os.path.join(REPORT_DIR, "collinear_pairs_dropped.csv"), index=False)

summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(REPORT_DIR, "collinearity_reduction_summary.csv"), index=False)

print("\n" + "=" * 70)
print(f"{'COLLINEARITY REDUCTION SUMMARY (threshold |r| > 0.95)':^70}")
print("=" * 70)
print(f"\n{'File':<45} {'Original':>8} {'Kept':>6} {'Dropped':>8}")
print("-" * 70)
total_orig = 0
total_kept = 0
for r in summary:
    print(f"{r['File']:<45} {r['Original']:>8} {r['Kept']:>6} {r['Dropped']:>8}")
    total_orig += r["Original"]
    total_kept += r["Kept"]
print("-" * 70)
print(f"{'TOTAL':<45} {total_orig:>8} {total_kept:>6} {total_orig - total_kept:>8}")
print(f"\nTotal collinear pairs identified: {len(all_pairs)}")
print(f"Output: {OUTPUT_DIR}/")
print(f"Report: {REPORT_DIR}/collinear_pairs_dropped.csv")
