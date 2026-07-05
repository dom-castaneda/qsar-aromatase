"""
Compute Substructure Count fingerprint (307 integer values) for aromatase bioactivity data.

Count version of CDK SubstructureFingerprinter — records how many unique substructure
matches each SMARTS pattern has, rather than just presence/absence.

Uses the pre-compiled SMARTS from 03a_substructure_fingerprint.py.

Reference: Laggner (2005) SMARTS_InteLigand.txt, Inte:Ligand GmbH.
"""
import sys
import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
from importlib import import_module

sys.stdout.reconfigure(line_buffering=True)
RDLogger.logger().setLevel(RDLogger.ERROR)

# Import pre-compiled SMARTS patterns
_subfp = import_module("03a_substructure_fingerprint")
_COMPILED = _subfp._COMPILED
N_BITS = _subfp.N_BITS  # 307

INPUT_FILE = "../data/processed/aromatase_bioactivity_clean.csv"
OUTPUT_FILE = "../data/fingerprints/fingerprints_substruct_count.csv"


def compute_substruct_count(mol):
    """Return a list of 307 ints (match counts) for the given RDKit Mol object."""
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
    print(f"Computing Substructure Count fingerprint ({N_BITS} values) ...")

    fp_data = []
    n_fail = 0

    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["canonical_smiles"]) if isinstance(row["canonical_smiles"], str) else None
        if mol is not None:
            fp_data.append(compute_substruct_count(mol))
        else:
            fp_data.append([np.nan] * N_BITS)
            n_fail += 1

        if (idx + 1) % 500 == 0:
            print(f"  {idx + 1}/{len(df)} molecules processed ...")

    print(f"Done: {len(df) - n_fail} computed, {n_fail} failed")

    cols = [f"SubFPC_{i}" for i in range(N_BITS)]
    fp_df = pd.DataFrame(fp_data, columns=cols)
    result = pd.concat([df[["molecule_chembl_id"]].reset_index(drop=True), fp_df], axis=1)
    result.to_csv(OUTPUT_FILE, index=False)

    populated = result["SubFPC_0"].notna().sum()
    max_count = fp_df.max().max()
    nonzero_per_mol = (fp_df > 0).sum(axis=1)
    print(f"\nSubstructure Count ({N_BITS} values):")
    print(f"  Populated: {populated}/{len(df)}")
    print(f"  Max count value: {max_count}")
    print(f"  Non-zero bins/molecule: mean={nonzero_per_mol.mean():.1f}, "
          f"median={nonzero_per_mol.median():.0f}")
    print(f"  Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
