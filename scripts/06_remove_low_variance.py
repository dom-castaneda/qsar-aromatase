"""
06_remove_low_variance.py
Identify and optionally remove near-constant fingerprint features (SD < threshold).

Two-step workflow:
  1. identify_low_variance() — computes SD for each fingerprint column, reports which
     fall below the threshold. Returns the list of columns to drop.
  2. remove_low_variance() — actually drops the identified columns and saves the
     filtered dataset. This is NOT called automatically.

Usage:
    python 06_remove_low_variance.py              # Process all fingerprints
    python 06_remove_low_variance.py maccs        # Process one fingerprint
    python 06_remove_low_variance.py ecfp4 kr     # Process specific fingerprints
"""

import pandas as pd
import numpy as np
import os
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(DATA_DIR, 'processed')

SD_THRESHOLD = 0.1

# Registry: short name -> (filename, column prefix)
FINGERPRINT_REGISTRY = {
    'maccs':       ('fingerprints_maccs.csv',             'MACCS_'),
    'ecfp4':       ('fingerprints_ecfp4.csv',            'ECFP4_'),
    'pubchem':     ('fingerprints_pubchem.csv',          'PubChem_'),
    'substruct':   ('fingerprints_substruct.csv',        'SubFP_'),
    'kr':          ('fingerprints_kr.csv',               'KRFP_'),
    'ap2d':        ('fingerprints_atompairs2d.csv',      'AP2D_'),
    'ap2dc':       ('fingerprints_atompairs2d_count.csv','AP2DC_'),
    'cdk':         ('fingerprints_cdk_fp.csv',           'CDK_'),
    'cdkext':      ('fingerprints_cdk_extended.csv',     'CDKExt_'),
    'cdkgraph':    ('fingerprints_cdk_graphonly.csv',    'CDKGraph_'),
    'estate':      ('fingerprints_estate.csv',           'EState_'),
    'estatec':     ('fingerprints_estate_count.csv',     'EStateC_'),
    'krcount':     ('fingerprints_kr_count.csv',         'KRFPC_'),
}


# ---------------------------------------------------------------------------
# Step 1: Identify near-constant features
# ---------------------------------------------------------------------------
def identify_low_variance(df, fp_prefix='MACCS_', threshold=SD_THRESHOLD):
    """
    Identify fingerprint columns with SD < threshold.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing fingerprint columns.
    fp_prefix : str
        Column name prefix to select fingerprint bits (e.g. 'MACCS_', 'ECFP4_').
    threshold : float
        SD threshold below which a feature is considered near-constant.

    Returns
    -------
    low_var_cols : list of str
        Columns with SD < threshold.
    stats : pd.DataFrame
        DataFrame with columns ['feature', 'sd', 'mean', 'is_low_variance'].
    """
    fp_cols = [c for c in df.columns if c.startswith(fp_prefix)]
    sds = df[fp_cols].std()
    means = df[fp_cols].mean()

    stats = pd.DataFrame({
        'feature': fp_cols,
        'sd': sds.values,
        'mean': means.values,
        'is_low_variance': sds.values < threshold
    })

    low_var_cols = stats.loc[stats['is_low_variance'], 'feature'].tolist()

    print(f"Fingerprint prefix: {fp_prefix}")
    print(f"Total features: {len(fp_cols)}")
    print(f"SD threshold: {threshold}")
    print(f"Near-constant features (SD < {threshold}): {len(low_var_cols)}")
    print(f"Remaining features: {len(fp_cols) - len(low_var_cols)}")
    print()

    # Show a few examples
    if low_var_cols:
        print("Examples of near-constant features (first 10):")
        subset = stats[stats['is_low_variance']].head(10)
        for _, row in subset.iterrows():
            print(f"  {row['feature']:12s}  SD={row['sd']:.6f}  mean={row['mean']:.4f}")
        print()

    return low_var_cols, stats


# ---------------------------------------------------------------------------
# Step 2: Remove the identified features (call separately)
# ---------------------------------------------------------------------------
def remove_low_variance(df, low_var_cols, output_path=None):
    """
    Drop the near-constant columns from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame.
    low_var_cols : list of str
        Columns to remove.
    output_path : str or None
        If provided, save the filtered DataFrame to this path.

    Returns
    -------
    df_filtered : pd.DataFrame
        DataFrame with low-variance columns removed.
    """
    df_filtered = df.drop(columns=low_var_cols)
    print(f"Removed {len(low_var_cols)} near-constant features.")
    print(f"Shape before: {df.shape} -> after: {df_filtered.shape}")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_filtered.to_csv(output_path, index=False)
        print(f"Saved filtered data to: {output_path}")

    return df_filtered


# ---------------------------------------------------------------------------
# Main: identify near-constant features across fingerprints (report only)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 60)
    print("Near-Constant Feature Identification (SD < 0.1)")
    print("=" * 60)
    print()

    # Determine which fingerprints to process
    args = [a.lower() for a in sys.argv[1:]]
    if args:
        names = []
        for a in args:
            if a not in FINGERPRINT_REGISTRY:
                print(f"Unknown fingerprint: '{a}'")
                print(f"Available: {', '.join(sorted(FINGERPRINT_REGISTRY.keys()))}")
                sys.exit(1)
            names.append(a)
    else:
        names = list(FINGERPRINT_REGISTRY.keys())

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = []

    for name in names:
        filename, prefix = FINGERPRINT_REGISTRY[name]
        filepath = os.path.join(DATA_DIR, 'fingerprints', filename)

        if not os.path.exists(filepath):
            print(f"[{name}] File not found: {filepath} — skipping")
            print()
            continue

        print(f"[{name}] Loading: {filename}")
        df = pd.read_csv(filepath)
        print(f"[{name}] Shape: {df.shape}")
        print()

        # Step 1 only: identify and report
        low_var_cols, stats = identify_low_variance(df, fp_prefix=prefix, threshold=SD_THRESHOLD)

        # Save per-fingerprint variance report
        stats_path = os.path.join(OUTPUT_DIR, f'{name}_variance_report.csv')
        stats.to_csv(stats_path, index=False)
        print(f"[{name}] Variance report saved to: {stats_path}")
        print("-" * 60)
        print()

        fp_cols = [c for c in df.columns if c.startswith(prefix)]
        summary.append({
            'fingerprint': name,
            'prefix': prefix,
            'total_features': len(fp_cols),
            'near_constant': len(low_var_cols),
            'remaining': len(fp_cols) - len(low_var_cols),
        })

    # Print summary table
    if summary:
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"{'Fingerprint':<12} {'Total':>7} {'Removed':>9} {'Remaining':>10}")
        print("-" * 42)
        for row in summary:
            print(f"{row['fingerprint']:<12} {row['total_features']:>7} "
                  f"{row['near_constant']:>9} {row['remaining']:>10}")
        print()
        print("Step 1 complete (identification only). No features were removed.")
        print("To remove, call remove_low_variance() in Python or add a Step 2 script.")
