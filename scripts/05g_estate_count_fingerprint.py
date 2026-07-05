"""
Compute E-State Count fingerprint (79 float values) for aromatase bioactivity data.

E-State Count records the sum of electrotopological state values for each of
79 atom-type groups — a continuous-valued fingerprint.

Uses RDKit EState.Fingerprinter — raw E-state sum values.

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
OUTPUT_FILE = "../data/fingerprints/fingerprints_estate_count.csv"

N_BITS = 79


def compute_estate_count(mol):
    """Return a list of 79 floats — E-state sum values per atom type."""
    _, sums = FingerprintMol(mol)
    return sums.tolist()


def main():
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} records")
    print(f"Computing E-State Count fingerprint ({N_BITS} values, continuous) ...")

    fp_data = []
    n_fail = 0

    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["canonical_smiles"]) if isinstance(row["canonical_smiles"], str) else None
        if mol is not None:
            fp_data.append(compute_estate_count(mol))
        else:
            fp_data.append([np.nan] * N_BITS)
            n_fail += 1

        if (idx + 1) % 500 == 0:
            print(f"  {idx + 1}/{len(df)} molecules processed ...")

    print(f"Done: {len(df) - n_fail} computed, {n_fail} failed")

    cols = [f"EStateC_{i}" for i in range(N_BITS)]
    fp_df = pd.DataFrame(fp_data, columns=cols)
    result = pd.concat([df[["molecule_chembl_id"]].reset_index(drop=True), fp_df], axis=1)
    result.to_csv(OUTPUT_FILE, index=False)

    populated = result["EStateC_0"].notna().sum()
    nonzero_per_mol = (fp_df != 0).sum(axis=1)
    print(f"\nE-State Count ({N_BITS} values, continuous):")
    print(f"  Populated: {populated}/{len(df)}")
    print(f"  Non-zero values/molecule: mean={nonzero_per_mol.mean():.1f}, "
          f"median={nonzero_per_mol.median():.0f}")
    print(f"  Value range: [{fp_df.min().min():.2f}, {fp_df.max().max():.2f}]")
    print(f"  Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
