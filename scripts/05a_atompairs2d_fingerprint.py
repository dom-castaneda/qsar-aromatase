"""
Compute AtomPairs2D fingerprint (780 binary bits) for aromatase bioactivity data.

Equivalent to CDK/PaDEL AtomPairs2D fingerprinter.
Uses RDKit AtomPairGenerator hashed to 780 bits.

Reference: Carhart et al. (1985) "Atom Pairs as Molecular Features in
Structure-Activity Studies", J. Chem. Inf. Comput. Sci. 25(2):64-73.
"""
import sys
import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

sys.stdout.reconfigure(line_buffering=True)
RDLogger.logger().setLevel(RDLogger.ERROR)

INPUT_FILE = "../data/processed/aromatase_bioactivity_clean.csv"
OUTPUT_FILE = "../data/fingerprints/fingerprints_atompairs2d.csv"

N_BITS = 780
AP_GEN = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=N_BITS)


def compute_atompairs2d(mol):
    """Return a list of 780 ints (0/1) for the given RDKit Mol object."""
    fp = AP_GEN.GetFingerprint(mol)
    return [int(fp[i]) for i in range(N_BITS)]


def main():
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} records")
    print(f"Computing AtomPairs2D fingerprint ({N_BITS} bits) ...")

    fp_data = []
    n_fail = 0

    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["canonical_smiles"]) if isinstance(row["canonical_smiles"], str) else None
        if mol is not None:
            fp_data.append(compute_atompairs2d(mol))
        else:
            fp_data.append([np.nan] * N_BITS)
            n_fail += 1

        if (idx + 1) % 500 == 0:
            print(f"  {idx + 1}/{len(df)} molecules processed ...")

    print(f"Done: {len(df) - n_fail} computed, {n_fail} failed")

    # Build and save DataFrame
    cols = [f"AP2D_{i}" for i in range(N_BITS)]
    fp_df = pd.DataFrame(fp_data, columns=cols)
    result = pd.concat([df[["molecule_chembl_id"]].reset_index(drop=True), fp_df], axis=1)
    result.to_csv(OUTPUT_FILE, index=False)

    # Summary
    populated = result["AP2D_0"].notna().sum()
    bits_per_mol = fp_df.sum(axis=1)
    print(f"\nAtomPairs2D ({N_BITS} bits):")
    print(f"  Populated: {populated}/{len(df)}")
    print(f"  Bits/molecule: mean={bits_per_mol.mean():.1f}, median={bits_per_mol.median():.0f}, "
          f"min={bits_per_mol.min():.0f}, max={bits_per_mol.max():.0f}")
    print(f"  Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
