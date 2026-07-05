"""
Compute interpretable molecular fingerprints for aromatase bioactivity data:
  - PubChem CACTVS fingerprints (881 bits) via PubChem PUG REST API (batch)
  - MACCS keys (167 bits) via RDKit
  - ECFP4 (1024 bits) via RDKit Morgan fingerprint (radius=2)
  - CDK SubstructureFingerprinter (307 bits) via native SMARTS matching
  - Klekota-Roth (4860 bits) via native SMARTS matching
"""

import sys
import pandas as pd
import numpy as np
import pubchempy as pcp
import base64
import time
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdFingerprintGenerator
from importlib import import_module as _im
_subfp = _im("03a_substructure_fingerprint")
compute_substruct_fp, SUBFP_NBITS = _subfp.compute_substruct_fp, _subfp.N_BITS
_krfp = _im("03b_klekota_roth_fingerprint")
compute_kr_fp, KRFP_NBITS = _krfp.compute_kr_fp, _krfp.N_BITS

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Suppress RDKit C-level warnings
RDLogger.logger().setLevel(RDLogger.ERROR)

INPUT_FILE = "../data/processed/aromatase_bioactivity_clean.csv"
OUTPUT_FILE = "../data/processed/aromatase_fingerprints.csv"
OUTPUT_PUBCHEM = "../data/fingerprints/fingerprints_pubchem.csv"
OUTPUT_MACCS = "../data/fingerprints/fingerprints_maccs.csv"
OUTPUT_ECFP4 = "../data/fingerprints/fingerprints_ecfp4.csv"
OUTPUT_SUBFP = "../data/fingerprints/fingerprints_substruct.csv"
OUTPUT_KRFP = "../data/fingerprints/fingerprints_kr.csv"

PUBCHEM_BATCH_SIZE = 100
PUBCHEM_DELAY = 0.5  # seconds between batch requests
ECFP4_NBITS = 1024

# Morgan fingerprint generator (ECFP4 = radius 2)
MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=ECFP4_NBITS)


def decode_pubchem_fp(fp_b64):
    """Decode a base64-encoded PubChem Fingerprint2D to 881 bits (list of ints)."""
    fp_bytes = base64.b64decode(fp_b64)
    all_bits = "".join(format(b, "08b") for b in fp_bytes)
    # First 32 bits = header (length = 881), then 881 fingerprint bits
    fp_bits = all_bits[32:32 + 881]
    return [int(b) for b in fp_bits]


def fetch_pubchem_batch(inchikeys):
    """Fetch PubChem fingerprints for a batch of InChIKeys.
    Returns dict: {inchikey: [881 bits]} for found molecules."""
    results = {}
    try:
        props = pcp.get_properties("Fingerprint2D,InChIKey", inchikeys, "inchikey")
        for p in props:
            ik = p.get("InChIKey", "")
            fp_b64 = p.get("Fingerprint2D", "")
            if ik and fp_b64:
                bits = decode_pubchem_fp(fp_b64)
                if len(bits) == 881:
                    results[ik] = bits
    except Exception as e:
        print(f"    Batch error: {e}")
    return results


def compute_maccs(mol):
    """Return MACCS key bits as a list of 0/1 integers (167 bits)."""
    fp = MACCSkeys.GenMACCSKeys(mol)
    return [int(fp[i]) for i in range(fp.GetNumBits())]


def compute_ecfp4(mol):
    """Return ECFP4 bits as a list of 0/1 integers (1024 bits)."""
    fp = MORGAN_GEN.GetFingerprint(mol)
    return [int(fp[i]) for i in range(ECFP4_NBITS)]


def main():
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} records from {INPUT_FILE}")

    # Get unique InChIKeys for PubChem batch lookup
    unique_iks = df["inchi_key"].dropna()
    unique_iks = unique_iks[unique_iks != ""].unique()
    print(f"Unique InChIKeys: {len(unique_iks)}")

    # Step 1: Batch-fetch PubChem fingerprints
    print(f"\nStep 1: Fetching PubChem CACTVS fingerprints (batches of {PUBCHEM_BATCH_SIZE}) ...")
    n_batches = (len(unique_iks) + PUBCHEM_BATCH_SIZE - 1) // PUBCHEM_BATCH_SIZE
    est_sec = n_batches * (PUBCHEM_DELAY + 1.2)
    print(f"  {n_batches} batches, estimated ~{est_sec / 60:.1f} minutes\n")

    pubchem_cache = {}
    for i in range(0, len(unique_iks), PUBCHEM_BATCH_SIZE):
        batch = unique_iks[i:i + PUBCHEM_BATCH_SIZE]
        batch_num = i // PUBCHEM_BATCH_SIZE + 1
        print(f"  Batch {batch_num}/{n_batches} ({len(pubchem_cache)} cached so far) ...")

        batch_results = fetch_pubchem_batch(list(batch))
        pubchem_cache.update(batch_results)

        time.sleep(PUBCHEM_DELAY)

    pubchem_ok = len(pubchem_cache)
    pubchem_fail = len(unique_iks) - pubchem_ok
    print(f"\n  PubChem done: {pubchem_ok} found, {pubchem_fail} not found")

    # Step 2: Compute MACCS keys, ECFP4, SubstructureFP, and Klekota-Roth locally
    print(f"\nStep 2: Computing MACCS + ECFP4 + SubFP + KR FP via RDKit ...")
    maccs_data = []
    ecfp4_data = []
    subfp_data = []
    krfp_data = []
    pubchem_data = []
    maccs_fail = 0

    for idx, row in df.iterrows():
        smiles = row["canonical_smiles"]
        ik = row["inchi_key"]

        # MACCS + ECFP4 + SubFP + KR
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles.strip() else None
        if mol is not None:
            maccs_data.append(compute_maccs(mol))
            ecfp4_data.append(compute_ecfp4(mol))
            subfp_data.append(compute_substruct_fp(mol))
            krfp_data.append(compute_kr_fp(mol))
        else:
            maccs_data.append([np.nan] * 167)
            ecfp4_data.append([np.nan] * ECFP4_NBITS)
            subfp_data.append([np.nan] * SUBFP_NBITS)
            krfp_data.append([np.nan] * KRFP_NBITS)
            maccs_fail += 1

        # PubChem (from cache)
        cached = pubchem_cache.get(ik)
        if cached is not None:
            pubchem_data.append(cached)
        else:
            pubchem_data.append([np.nan] * 881)

        if (idx + 1) % 500 == 0:
            print(f"    {idx + 1}/{len(df)} molecules processed ...")

    print(f"  Local FPs done: {len(df) - maccs_fail} ok, {maccs_fail} failed")

    # Build DataFrames
    pubchem_cols = [f"PubChem_{i}" for i in range(881)]
    pubchem_df = pd.DataFrame(pubchem_data, columns=pubchem_cols)

    maccs_cols = [f"MACCS_{i}" for i in range(167)]
    maccs_df = pd.DataFrame(maccs_data, columns=maccs_cols)

    ecfp4_cols = [f"ECFP4_{i}" for i in range(ECFP4_NBITS)]
    ecfp4_df = pd.DataFrame(ecfp4_data, columns=ecfp4_cols)

    subfp_cols = [f"SubFP_{i}" for i in range(SUBFP_NBITS)]
    subfp_df = pd.DataFrame(subfp_data, columns=subfp_cols)

    krfp_cols = [f"KRFP_{i}" for i in range(KRFP_NBITS)]
    krfp_df = pd.DataFrame(krfp_data, columns=krfp_cols)

    # Merge with original data
    mol_ids = df.reset_index(drop=True)["molecule_chembl_id"]

    result = pd.concat([df.reset_index(drop=True), pubchem_df, maccs_df, ecfp4_df, subfp_df, krfp_df], axis=1)
    result.to_csv(OUTPUT_FILE, index=False)

    # Save separate fingerprint CSVs with molecule_chembl_id as identifier
    pd.concat([mol_ids, pubchem_df], axis=1).to_csv(OUTPUT_PUBCHEM, index=False)
    pd.concat([mol_ids, maccs_df], axis=1).to_csv(OUTPUT_MACCS, index=False)
    pd.concat([mol_ids, ecfp4_df], axis=1).to_csv(OUTPUT_ECFP4, index=False)
    pd.concat([mol_ids, subfp_df], axis=1).to_csv(OUTPUT_SUBFP, index=False)
    pd.concat([mol_ids, krfp_df], axis=1).to_csv(OUTPUT_KRFP, index=False)

    # Summary
    pubchem_populated = result["PubChem_0"].notna().sum()
    maccs_populated = result["MACCS_0"].notna().sum()
    ecfp4_populated = result["ECFP4_0"].notna().sum()
    subfp_populated = result["SubFP_0"].notna().sum()
    krfp_populated = result["KRFP_0"].notna().sum()

    print(f"\n{'='*50}")
    print(f"Total molecules:          {len(df)}")
    print(f"\nPubChem CACTVS (881 bits):")
    print(f"  Populated:              {pubchem_populated}")
    print(f"  Missing:                {len(df) - pubchem_populated}")
    print(f"\nMACCS keys (167 bits):")
    print(f"  Populated:              {maccs_populated}")
    print(f"  Missing:                {len(df) - maccs_populated}")
    print(f"\nECFP4 (1024 bits):")
    print(f"  Populated:              {ecfp4_populated}")
    print(f"  Missing:                {len(df) - ecfp4_populated}")
    print(f"\nSubstructureFP (307 bits):")
    print(f"  Populated:              {subfp_populated}")
    print(f"  Missing:                {len(df) - subfp_populated}")
    print(f"\nKlekota-Roth (4860 bits):")
    print(f"  Populated:              {krfp_populated}")
    print(f"  Missing:                {len(df) - krfp_populated}")
    fp_total = len(pubchem_cols) + len(maccs_cols) + len(ecfp4_cols) + len(subfp_cols) + len(krfp_cols)
    print(f"\nTotal fingerprint cols:   {fp_total}")
    print(f"Output columns total:     {len(result.columns)}")
    print(f"\nSaved to:")
    print(f"  Combined:  {OUTPUT_FILE}")
    print(f"  PubChem:   {OUTPUT_PUBCHEM}")
    print(f"  MACCS:     {OUTPUT_MACCS}")
    print(f"  ECFP4:     {OUTPUT_ECFP4}")
    print(f"  SubFP:     {OUTPUT_SUBFP}")
    print(f"  KR FP:     {OUTPUT_KRFP}")


if __name__ == "__main__":
    main()
