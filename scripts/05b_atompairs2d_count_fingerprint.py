"""
Compute AtomPairs2D Count fingerprint (780 integer values) for aromatase bioactivity data.

Count version of AtomPairs2D — records how many times each atom-pair hash occurs,
rather than just presence/absence.

Uses RDKit AtomPairGenerator hashed to 780 bins, count mode.
"""
import sys
import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

sys.stdout.reconfigure(line_buffering=True)
RDLogger.logger().setLevel(RDLogger.ERROR)

INPUT_FILE = "../data/processed/aromatase_bioactivity_clean.csv"
OUTPUT_FILE = "../data/fingerprints/fingerprints_atompairs2d_count.csv"

N_BITS = 780
AP_GEN = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=N_BITS)


def compute_atompairs2d_count(mol):
    """Return a list of 780 ints (counts) for the given RDKit Mol object."""
    cfp = AP_GEN.GetCountFingerprint(mol)
    arr = np.zeros(N_BITS, dtype=int)
    for idx, cnt in cfp.GetNonzeroElements().items():
        if idx < N_BITS:
            arr[idx] = cnt
    return arr.tolist()


def main():
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} records")
    print(f"Computing AtomPairs2D Count fingerprint ({N_BITS} bins) ...")

    fp_data = []
    n_fail = 0

    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["canonical_smiles"]) if isinstance(row["canonical_smiles"], str) else None
        if mol is not None:
            fp_data.append(compute_atompairs2d_count(mol))
        else:
            fp_data.append([np.nan] * N_BITS)
            n_fail += 1

        if (idx + 1) % 500 == 0:
            print(f"  {idx + 1}/{len(df)} molecules processed ...")

    print(f"Done: {len(df) - n_fail} computed, {n_fail} failed")

    cols = [f"AP2DC_{i}" for i in range(N_BITS)]
    fp_arr = np.array(fp_data, dtype=np.int16)
    fp_df = pd.DataFrame(fp_arr, columns=cols)
    result = pd.concat([df[["molecule_chembl_id"]].reset_index(drop=True), fp_df], axis=1)
    result.to_csv(OUTPUT_FILE, index=False)

    populated = fp_df.iloc[:, 0].notna().sum()
    max_count = fp_arr.max()
    nonzero_per_mol = (fp_arr > 0).sum(axis=1)
    print(f"\nAtomPairs2D Count ({N_BITS} bins):")
    print(f"  Populated: {populated}/{len(df)}")
    print(f"  Max count value: {max_count}")
    print(f"  Non-zero bins/molecule: mean={nonzero_per_mol.mean():.1f}, "
          f"median={np.median(nonzero_per_mol):.0f}")
    print(f"  Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
