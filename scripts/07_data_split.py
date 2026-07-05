"""
Data splitting for aromatase QSAR modelling.

Splits the dataset (molecules with exact pchembl_value) into 80/20 train/test using:
  1. Random splitting (stratified by activity class)
  2. Kennard-Stone algorithm (maximin distance on ECFP4 fingerprints)

Output: data/splits/ with train/test CSVs for each method.
"""
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = "../data/processed"
FP_DIR = "../data/fingerprints_filtered"
OUTPUT_DIR = "../data/splits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPLIT_RATIO = 0.8  # 80% train, 20% test
RANDOM_STATE = 42


def classify_activity(val):
    if val >= 6.5:
        return "Active"
    elif val < 5.0:
        return "Inactive"
    else:
        return "Intermediate"


def kennard_stone(X, n_train):
    """
    Kennard-Stone algorithm for selecting a training set that spans the feature space.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    n_train : int, number of samples to select for training

    Returns
    -------
    train_idx : list of int, indices selected for training
    test_idx : list of int, indices remaining for testing
    """
    n_samples = X.shape[0]
    assert n_train <= n_samples

    # Compute pairwise squared Euclidean distances
    # Use chunked approach to avoid massive memory for full distance matrix
    print("  Computing pairwise distances...")
    # For 3290 samples x 180 features, full distance matrix is ~80 MB — manageable
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(X, X, metric="euclidean")

    # Step 1: Find the two most distant points
    i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    selected = [i, j]
    remaining = set(range(n_samples)) - {i, j}

    print(f"  Initial pair: {i}, {j} (distance={dist_matrix[i, j]:.4f})")
    print(f"  Selecting {n_train} training samples...")

    # Step 2: Iteratively select the point most distant from the current training set
    # min_dist_to_selected[k] = min distance from point k to any selected point
    min_dist_to_selected = np.minimum(dist_matrix[i], dist_matrix[j])

    while len(selected) < n_train:
        # Among remaining, find the one with maximum min-distance to selected
        remaining_arr = np.array(list(remaining))
        candidate_dists = min_dist_to_selected[remaining_arr]
        best_local_idx = np.argmax(candidate_dists)
        best_idx = remaining_arr[best_local_idx]

        selected.append(best_idx)
        remaining.remove(best_idx)

        # Update min distances
        min_dist_to_selected = np.minimum(min_dist_to_selected, dist_matrix[best_idx])

        if len(selected) % 500 == 0:
            print(f"    {len(selected)}/{n_train} selected...")

    train_idx = sorted(selected)
    test_idx = sorted(list(remaining))
    return train_idx, test_idx


def summarize_split(df_train, df_test, method_name):
    """Print summary statistics for a train/test split."""
    print(f"\n{'=' * 60}")
    print(f"  {method_name} Split Summary")
    print(f"{'=' * 60}")
    print(f"  Train: {len(df_train):,} ({len(df_train)/(len(df_train)+len(df_test))*100:.1f}%)")
    print(f"  Test:  {len(df_test):,} ({len(df_test)/(len(df_train)+len(df_test))*100:.1f}%)")
    print()
    print(f"  {'Metric':<20} {'Train':>10} {'Test':>10}")
    print(f"  {'-'*42}")
    for stat, func in [("Mean", np.mean), ("Std", np.std), ("Median", np.median),
                        ("Min", np.min), ("Max", np.max)]:
        t_val = func(df_train["pchembl_value"])
        e_val = func(df_test["pchembl_value"])
        print(f"  {stat:<20} {t_val:>10.3f} {e_val:>10.3f}")

    # Activity class balance
    print()
    print(f"  {'Activity Class':<20} {'Train':>10} {'Test':>10}")
    print(f"  {'-'*42}")
    for cls in ["Active", "Intermediate", "Inactive"]:
        t_n = (df_train["activity_class"] == cls).sum()
        e_n = (df_test["activity_class"] == cls).sum()
        print(f"  {cls:<20} {t_n:>10} {e_n:>10}")


def main():
    # Load dataset — filter to exact measurements with pchembl
    df_full = pd.read_csv(f"{DATA_DIR}/aromatase_bioactivity_clean.csv")
    mask = (df_full["standard_relation"] == "=") & df_full["pchembl_value"].notna()

    df = df_full[mask].reset_index(drop=True)
    df["activity_class"] = df["pchembl_value"].apply(classify_activity)
    print(f"Dataset: {len(df)} molecules (exact measurements with pchembl_value)")

    # Load filtered ECFP4 fingerprints — row-aligned with full dataset, apply same mask
    fp_full = pd.read_csv(f"{FP_DIR}/fingerprints_ecfp4.csv")
    fp = fp_full[mask.values].reset_index(drop=True)
    fp_cols = [c for c in fp.columns if c != "molecule_chembl_id"]
    X = fp[fp_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    print(f"ECFP4 features: {X.shape[1]} bits (after near-constant removal)")
    print(f"Feature matrix shape: {X.shape}")

    n_train = int(len(df) * SPLIT_RATIO)
    n_test = len(df) - n_train
    print(f"Target split: {n_train} train / {n_test} test (80/20)")

    # ===================================================================
    # Method 1: Random Split (stratified by activity class)
    # ===================================================================
    print("\n--- Method 1: Random Split ---")
    train_idx_r, test_idx_r = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=RANDOM_STATE,
        stratify=df["activity_class"]
    )

    df_train_random = df.iloc[train_idx_r].reset_index(drop=True)
    df_test_random = df.iloc[test_idx_r].reset_index(drop=True)

    # Save
    df_train_random[["molecule_chembl_id", "pchembl_value", "activity_class"]].to_csv(
        f"{OUTPUT_DIR}/random_train.csv", index=False)
    df_test_random[["molecule_chembl_id", "pchembl_value", "activity_class"]].to_csv(
        f"{OUTPUT_DIR}/random_test.csv", index=False)

    summarize_split(df_train_random, df_test_random, "Random (Stratified)")

    # ===================================================================
    # Method 2: Kennard-Stone Split
    # ===================================================================
    print("\n--- Method 2: Kennard-Stone Split ---")
    ks_train_idx, ks_test_idx = kennard_stone(X, n_train)

    df_train_ks = df.iloc[ks_train_idx].reset_index(drop=True)
    df_test_ks = df.iloc[ks_test_idx].reset_index(drop=True)

    # Save
    df_train_ks[["molecule_chembl_id", "pchembl_value", "activity_class"]].to_csv(
        f"{OUTPUT_DIR}/kennard_stone_train.csv", index=False)
    df_test_ks[["molecule_chembl_id", "pchembl_value", "activity_class"]].to_csv(
        f"{OUTPUT_DIR}/kennard_stone_test.csv", index=False)

    summarize_split(df_train_ks, df_test_ks, "Kennard-Stone")

    # ===================================================================
    # Final summary
    # ===================================================================
    print(f"\n{'=' * 60}")
    print(f"  Files saved to: {OUTPUT_DIR}/")
    print(f"{'=' * 60}")
    print(f"  random_train.csv         ({len(df_train_random):,} rows)")
    print(f"  random_test.csv          ({len(df_test_random):,} rows)")
    print(f"  kennard_stone_train.csv  ({len(df_train_ks):,} rows)")
    print(f"  kennard_stone_test.csv   ({len(df_test_ks):,} rows)")


if __name__ == "__main__":
    main()
