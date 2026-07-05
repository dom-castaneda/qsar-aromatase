"""
Clean aromatase bioactivity data:
  - Generate InChIKey from SMILES using RDKit
  - Remove exact duplicate rows
  - Remove duplicate groups with pChEMBL SD > 3 (inconsistent measurements)
  - Aggregate remaining duplicates by taking the mean
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey

INPUT_FILE = "../data/raw/aromatase_bioactivity.csv"
OUTPUT_FILE = "../data/processed/aromatase_bioactivity_clean.csv"

DEDUP_COLS = ["molecule_chembl_id", "standard_type", "standard_relation", "standard_value"]
GROUP_COLS = ["molecule_chembl_id", "standard_type"]
SD_THRESHOLD = 3.0


def smiles_to_inchikey(smiles):
    """Convert a SMILES string to an InChIKey. Returns empty string on failure."""
    if not isinstance(smiles, str) or not smiles.strip():
        return ""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    inchi = MolToInchi(mol)
    if inchi is None:
        return ""
    return InchiToInchiKey(inchi)


def aggregate_group(group):
    """Aggregate a group of duplicate measurements into a single row.
    Takes the mean of standard_value and pchembl_value, keeps the first row
    for all other columns."""
    if len(group) == 1:
        return group.iloc[0]

    row = group.iloc[0].copy()
    row["standard_value"] = group["standard_value"].mean()
    if group["pchembl_value"].notna().any():
        row["pchembl_value"] = group["pchembl_value"].mean()
    return row


def main():
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} records from {INPUT_FILE}")

    # Step 1: Generate InChIKey
    print("\nStep 1: Generating InChIKeys from SMILES ...")
    df["inchi_key"] = df["canonical_smiles"].apply(smiles_to_inchikey)

    inchikey_ok = (df["inchi_key"] != "").sum()
    inchikey_fail = (df["inchi_key"] == "").sum()
    print(f"  InChIKey generated: {inchikey_ok}")
    print(f"  InChIKey failed:    {inchikey_fail}")

    # Step 2: Remove exact duplicate rows
    before = len(df)
    df = df.drop_duplicates(subset=DEDUP_COLS, keep="first")
    after = len(df)
    print(f"\nStep 2: Exact deduplication on {DEDUP_COLS}:")
    print(f"  Before: {before}")
    print(f"  After:  {after}")
    print(f"  Removed: {before - after}")

    # Step 3: SD filter on pChEMBL values
    print(f"\nStep 3: SD filter (threshold > {SD_THRESHOLD} on pChEMBL) ...")
    grouped = df.groupby(GROUP_COLS)

    groups_to_remove = set()
    sd_computed = 0
    for name, group in grouped:
        pchembl_vals = group["pchembl_value"].dropna()
        if len(pchembl_vals) >= 2:
            sd = pchembl_vals.std()
            sd_computed += 1
            if sd > SD_THRESHOLD:
                groups_to_remove.add(name)

    print(f"  Groups with 2+ pChEMBL values: {sd_computed}")
    print(f"  Groups removed (SD > {SD_THRESHOLD}): {len(groups_to_remove)}")

    # Remove flagged groups
    if groups_to_remove:
        mask = df.set_index(GROUP_COLS).index.isin(groups_to_remove)
        rows_removed = mask.sum()
        df = df[~mask]
        print(f"  Rows removed: {rows_removed}")
    else:
        print(f"  Rows removed: 0")

    # Step 4: Aggregate remaining duplicates by taking the mean
    print(f"\nStep 4: Aggregating duplicate measurements (mean) ...")
    before_agg = len(df)
    df = df.groupby(GROUP_COLS, group_keys=False).apply(aggregate_group).reset_index(drop=True)
    after_agg = len(df)
    print(f"  Before aggregation: {before_agg}")
    print(f"  After aggregation:  {after_agg}")
    print(f"  Groups collapsed:   {before_agg - after_agg}")

    # Save
    df.to_csv(OUTPUT_FILE, index=False)

    # Summary
    ki_count = (df["standard_type"] == "Ki").sum()
    ic50_count = (df["standard_type"] == "IC50").sum()
    pic50_count = (df["standard_type"] == "pIC50").sum()
    pchembl_count = df["pchembl_value"].notna().sum()
    unique_molecules = df["molecule_chembl_id"].nunique()

    print(f"\n{'='*50}")
    print(f"Final dataset: {len(df)} records")
    print(f"  Unique molecules: {unique_molecules}")
    print(f"  Ki records:       {ki_count}")
    print(f"  IC50 records:     {ic50_count}")
    print(f"  pIC50 records:    {pic50_count}")
    print(f"  With pChEMBL:     {pchembl_count}")
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
