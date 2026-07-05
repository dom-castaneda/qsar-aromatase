"""
Compute E-State fingerprint (79 binary bits) for aromatase bioactivity data.

E-State fingerprint encodes the presence/absence of 79 atom-type groups
defined by electrotopological state indices.

Uses RDKit EState.Fingerprinter — binarised (1 if atom type present, 0 otherwise).

Reference: Hall & Kier (1995) "Electrotopological State Indices for Atom Types",
J. Chem. Inf. Comput. Sci. 35(6):1039-1045.
"""
import sys
import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.EState.Fingerprinter import FingerprintMol

sys.stdout.reconfigure(line_buffering=True)
RDLogger.logger().setLevel(RDLogger.ERROR)

INPUT_FILE = "../data/processed/aromatase_bioactivity_clean.csv"
OUTPUT_FILE = "../data/fingerprints/fingerprints_estate.csv"

N_BITS = 79


def compute_estate_binary(mol):
    """Return a list of 79 ints (0/1) — presence of each E-state atom type."""
    counts, _ = FingerprintMol(mol)
    return (counts > 0).astype(int).tolist()


def main():
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} records")
    print(f"Computing E-State fingerprint ({N_BITS} bits, binary) ...")

    fp_data = []
    n_fail = 0

    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["canonical_smiles"]) if isinstance(row["canonical_smiles"], str) else None
        if mol is not None:
            fp_data.append(compute_estate_binary(mol))
        else:
            fp_data.append([np.nan] * N_BITS)
            n_fail += 1

        if (idx + 1) % 500 == 0:
            print(f"  {idx + 1}/{len(df)} molecules processed ...")

    print(f"Done: {len(df) - n_fail} computed, {n_fail} failed")

    cols = [f"EState_{i}" for i in range(N_BITS)]
    fp_df = pd.DataFrame(fp_data, columns=cols)
    result = pd.concat([df[["molecule_chembl_id"]].reset_index(drop=True), fp_df], axis=1)
    result.to_csv(OUTPUT_FILE, index=False)

    populated = result["EState_0"].notna().sum()
    bits_per_mol = fp_df.sum(axis=1)
    print(f"\nE-State ({N_BITS} bits, binary):")
    print(f"  Populated: {populated}/{len(df)}")
    print(f"  Bits/molecule: mean={bits_per_mol.mean():.1f}, median={bits_per_mol.median():.0f}, "
          f"min={bits_per_mol.min():.0f}, max={bits_per_mol.max():.0f}")
    print(f"  Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
