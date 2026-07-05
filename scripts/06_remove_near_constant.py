"""
Remove near-constant features from all fingerprint CSVs.

A feature is "near-constant" if it has the same value in more than 95% of samples
(i.e., variance is effectively zero for modelling purposes).

For binary fingerprints: removes bits that are always 0, always 1, or set in <5% or >95% of molecules.
For count fingerprints: removes bins where >95% of values are identical.

Saves filtered versions to data/fingerprints_filtered/ with the same filenames.
Also produces a summary report.
"""
import sys
import os
import pandas as pd
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

INPUT_DIR = "../data/fingerprints"
OUTPUT_DIR = "../data/fingerprints_filtered"
THRESHOLD = 0.05  # Remove if <5% or >95% frequency (for binary), or >95% same value (for count)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# All 14 fingerprint files
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

results = []

for fname in FP_FILES:
    path = os.path.join(INPUT_DIR, fname)
    if not os.path.exists(path):
        print(f"SKIP (not found): {fname}")
        continue

    df = pd.read_csv(path)
    id_col = df.columns[0]  # molecule_chembl_id
    fp_cols = df.columns[1:]
    n_original = len(fp_cols)

    fp_data = df[fp_cols]

    # Near-constant removal: for each column, compute the fraction of the most common value
    # If that fraction > (1 - THRESHOLD), i.e., >95%, remove it
    most_common_frac = fp_data.apply(lambda col: col.value_counts(normalize=True).iloc[0] if len(col.value_counts()) > 0 else 1.0)
    keep_mask = most_common_frac <= (1.0 - THRESHOLD)
    kept_cols = fp_cols[keep_mask]
    n_kept = len(kept_cols)
    n_removed = n_original - n_kept

    # Save filtered version
    filtered = pd.concat([df[[id_col]], df[kept_cols]], axis=1)
    out_path = os.path.join(OUTPUT_DIR, fname)
    filtered.to_csv(out_path, index=False)

    results.append({
        "File": fname,
        "Original": n_original,
        "Removed": n_removed,
        "Kept": n_kept,
        "% Removed": f"{n_removed/n_original*100:.1f}%",
    })

    print(f"{fname:<45} {n_original:>6} -> {n_kept:>6} ({n_removed} removed, {n_removed/n_original*100:.1f}%)")

print("\n" + "=" * 80)
print(f"{'SUMMARY':^80}")
print("=" * 80)
print(f"\n{'File':<45} {'Original':>8} {'Kept':>6} {'Removed':>8} {'% Removed':>10}")
print("-" * 80)
total_orig = 0
total_kept = 0
for r in results:
    print(f"{r['File']:<45} {r['Original']:>8} {r['Kept']:>6} {r['Removed']:>8} {r['% Removed']:>10}")
    total_orig += r["Original"]
    total_kept += r["Kept"]
print("-" * 80)
print(f"{'TOTAL':<45} {total_orig:>8} {total_kept:>6} {total_orig - total_kept:>8} {(total_orig-total_kept)/total_orig*100:.1f}%")
print(f"\nThreshold: features with >{(1-THRESHOLD)*100:.0f}% same value removed")
print(f"Output directory: {OUTPUT_DIR}")
