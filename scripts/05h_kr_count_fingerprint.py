"""
Compute Klekota-Roth Count fingerprint (4860 integer values) for aromatase bioactivity data.

Count version of Klekota-Roth — records how many unique substructure matches
each SMARTS pattern has, rather than just presence/absence.

Uses the pre-compiled SMARTS from 03b_klekota_roth_fingerprint.py.

Reference: Klekota & Roth (2008) "Chemical substructures that enrich for
biological activity", Bioinformatics 24(21):2518-2525.
"""
import sys
import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
from importlib import import_module

sys.stdout.reconfigure(line_buffering=True)
RDLogger.logger().setLevel(RDLogger.ERROR)

# Import pre-compiled SMARTS patterns
_krfp = import_module("03b_klekota_roth_fingerprint")
_COMPILED = _krfp._COMPILED
N_BITS = _krfp.N_BITS  # 4860

INPUT_FILE = "../data/processed/aromatase_bioactivity_clean.csv"
OUTPUT_FILE = "../data/fingerprints/fingerprints_kr_count.csv"


def compute_kr_count(mol):
    """Return a list of 4860 ints (match counts) for the given RDKit Mol object."""
    counts = []
    for pat in _COMPILED:
        if pat is not None:
            matches = mol.GetSubstructMatches(pat, uniquify=True)
            counts.append(len(matches))
        else:
            counts.append(0)
    return counts


def main():
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} records")
    print(f"Computing Klekota-Roth Count fingerprint ({N_BITS} values) ...")
    print("  (This will take several minutes due to 4860 SMARTS x 3774 molecules)")

    fp_data = []
    n_fail = 0

    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["canonical_smiles"]) if isinstance(row["canonical_smiles"], str) else None
        if mol is not None:
            fp_data.append(compute_kr_count(mol))
        else:
            fp_data.append([np.nan] * N_BITS)
            n_fail += 1

        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(df)} molecules processed ...")

    print(f"Done: {len(df) - n_fail} computed, {n_fail} failed")

    cols = [f"KRFPC_{i}" for i in range(N_BITS)]
    fp_df = pd.DataFrame(fp_data, columns=cols)
    result = pd.concat([df[["molecule_chembl_id"]].reset_index(drop=True), fp_df], axis=1)
    result.to_csv(OUTPUT_FILE, index=False)

    populated = result["KRFPC_0"].notna().sum()
    max_count = fp_df.max().max()
    nonzero_per_mol = (fp_df > 0).sum(axis=1)
    print(f"\nKlekota-Roth Count ({N_BITS} values):")
    print(f"  Populated: {populated}/{len(df)}")
    print(f"  Max count value: {max_count}")
    print(f"  Non-zero bins/molecule: mean={nonzero_per_mol.mean():.1f}, "
          f"median={nonzero_per_mol.median():.0f}")
    print(f"  Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
