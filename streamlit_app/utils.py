"""
Shared utilities for the Aromatase QSAR EDA Streamlit app.
Cached data loading, descriptor computation, and constants.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
FP_DIR = PROJECT_ROOT / "data" / "fingerprints"
FIG_DIR = PROJECT_ROOT / "data" / "figures" / "eda"

# Activity class colours
ACTIVITY_COLORS = {
    "Active": "#2ecc71",
    "Intermediate": "#f39c12",
    "Inactive": "#e74c3c",
    "Unknown": "#95a5a6",
}

ACTIVITY_ORDER = ["Active", "Intermediate", "Inactive", "Unknown"]


@st.cache_data
def load_bioactivity():
    """Load the curated bioactivity dataset."""
    df = pd.read_csv(DATA_DIR / "aromatase_bioactivity_clean.csv")
    df["activity_class"] = df["pchembl_value"].apply(classify_activity)
    return df


@st.cache_data
def load_fingerprints(fp_name):
    """Load a fingerprint CSV by name (e.g. 'ecfp4', 'maccs', 'kr')."""
    path = FP_DIR / f"fingerprints_{fp_name}.csv"
    return pd.read_csv(path)


@st.cache_data
def load_filtered_fingerprint(fp_name):
    """Load a reduced fingerprint CSV (collinear features removed)."""
    path = PROJECT_ROOT / "data" / "fingerprints_reduced" / f"fingerprints_{fp_name}.csv"
    return pd.read_csv(path)


@st.cache_data
def compute_descriptors(smiles_series):
    """Compute molecular descriptors from SMILES. Returns DataFrame."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    RDLogger.logger().setLevel(RDLogger.ERROR)

    desc_cols = ["MW", "LogP", "HBA", "HBD", "TPSA", "RotBonds", "AromaticRings", "FractionCSP3"]

    def _calc(smi):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            return [np.nan] * 8
        return [
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
            Descriptors.NumHAcceptors(mol), Descriptors.NumHDonors(mol),
            Descriptors.TPSA(mol), Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol), Descriptors.FractionCSP3(mol),
        ]

    data = smiles_series.apply(_calc)
    return pd.DataFrame(data.tolist(), columns=desc_cols, index=smiles_series.index)


@st.cache_data
def get_pca_embedding():
    """Compute PCA on ECFP4 fingerprints. Returns (X_pca, var_explained)."""
    from sklearn.decomposition import PCA
    fp = load_fingerprints("ecfp4")
    X = fp.iloc[:, 1:].values
    pca = PCA(n_components=10, random_state=42)
    X_pca = pca.fit_transform(X)
    return X_pca, pca.explained_variance_ratio_


def classify_activity(val):
    """Classify pchembl_value into activity classes."""
    if pd.isna(val):
        return "Unknown"
    if val >= 6.5:
        return "Active"
    elif val < 5.0:
        return "Inactive"
    else:
        return "Intermediate"
