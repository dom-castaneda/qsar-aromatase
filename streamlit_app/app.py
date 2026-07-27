"""
Aromatase (CYP19A1) Inhibitor Activity Predictor
Predict bioactivity from a SMILES structure using Extra Trees on AP2D_Count fingerprint.
"""
import streamlit as st
import numpy as np
import json
import joblib
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdFingerprintGenerator

RDLogger.logger().setLevel(RDLogger.ERROR)

st.set_page_config(
    page_title="Aromatase Activity Predictor",
    page_icon=":pill:",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "data" / "models" / "final"

# --- Load Model Components ---
@st.cache_resource
def load_models():
    reg = joblib.load(MODEL_DIR / "model_regressor.joblib")
    cls = joblib.load(MODEL_DIR / "model_classifier.joblib")
    scaler = joblib.load(MODEL_DIR / "ad_scaler.joblib")
    pca = joblib.load(MODEL_DIR / "ad_pca.joblib")
    with open(MODEL_DIR / "feature_columns.json") as f:
        feature_cols = json.load(f)
    with open(MODEL_DIR / "ad_bounds.json") as f:
        ad_bounds = json.load(f)
    with open(MODEL_DIR / "label_classes.json") as f:
        label_classes = json.load(f)
    return reg, cls, scaler, pca, feature_cols, ad_bounds, label_classes


reg_model, cls_model, ad_scaler, ad_pca, feature_cols, ad_bounds, label_classes = load_models()

N_BITS = 780
AP_GEN = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=N_BITS)


def compute_ap2d_count(mol):
    """Compute AP2D_Count fingerprint (780 bins) for a single molecule."""
    cfp = AP_GEN.GetCountFingerprint(mol)
    arr = np.zeros(N_BITS, dtype=np.float32)
    for idx, cnt in cfp.GetNonzeroElements().items():
        if idx < N_BITS:
            arr[idx] = cnt
    return arr


def check_applicability_domain(fp_vector):
    """Check if molecule is within the training AD (PCA bounding box)."""
    scaled = ad_scaler.transform(fp_vector.reshape(1, -1))
    scores = ad_pca.transform(scaled)[0]
    ad_min = np.array(ad_bounds["min"])
    ad_max = np.array(ad_bounds["max"])
    inside = np.all((scores >= ad_min) & (scores <= ad_max))
    n_violated = ((scores < ad_min) | (scores > ad_max)).sum()
    return inside, n_violated


# --- UI ---
st.title("Aromatase (CYP19A1) Inhibitor Activity Predictor")
st.markdown("Enter a SMILES structure to predict its bioactivity against Aromatase.")

# Example molecules
with st.expander("Example SMILES (click to expand, then copy-paste)"):
    st.markdown("""
    | Compound | SMILES | Expected Class |
    |----------|--------|----------------|
    | **Letrozole** (clinical AI) | `C1=CC(=CC=C1C#N)C(C2=CC=C(C=C2)C#N)N3C=NC=N3` | Active |
    | **Anastrozole** (clinical AI) | `CC(C1=CC(=CC=C1)C(C)(C#N)C)N2C=NC=N2` | Active |
    | **Chrysin** (flavonoid) | `C1=CC=C(C=C1)C2=CC(=O)C3=C(O2)C=C(C=C3O)O` | Intermediate |
    | **Caffeine** | `CN1C=NC2=C1C(=O)N(C(=O)N2C)C` | Inactive |
    | **Aspirin** | `CC(=O)OC1=CC=CC=C1C(=O)O` | Inactive |
    """)

# Input
smiles_input = st.text_input(
    "SMILES",
    placeholder="e.g. C#N/C(=C\\1/C=CC(=CC1)C#N)N1C=NC=N1 (Letrozole)",
    help="Paste a valid SMILES string for the molecule you want to predict."
)

if smiles_input:
    mol = Chem.MolFromSmiles(smiles_input)

    if mol is None:
        st.error("Invalid SMILES. Please enter a valid molecular structure.")
    else:
        # Compute fingerprint
        full_fp = compute_ap2d_count(mol)

        # Filter to training columns (AP2DC_0 through AP2DC_779 -> select the 455 used in training)
        col_indices = [int(c.replace("AP2DC_", "")) for c in feature_cols]
        fp_filtered = full_fp[col_indices]

        # Predict
        X_input = fp_filtered.reshape(1, -1)
        pchembl_pred = reg_model.predict(X_input)[0]
        class_pred_idx = cls_model.predict(X_input)[0]
        class_pred = label_classes[class_pred_idx]

        # AD check
        inside_ad, n_violated = check_applicability_domain(fp_filtered)

        # --- Results Display ---
        col_mol, col_results = st.columns([1, 2])

        with col_mol:
            st.subheader("Molecule")
            img = Draw.MolToImage(mol, size=(300, 300))
            st.image(img)
            st.caption(f"SMILES: `{smiles_input}`")

        with col_results:
            st.subheader("Prediction")

            # Class badge color
            class_colors = {"active": "green", "intermediate": "orange", "inactive": "red"}
            class_color = class_colors.get(class_pred, "gray")

            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted pchembl", f"{pchembl_pred:.2f}")
            c2.metric("Activity Class", class_pred.capitalize())
            c3.metric("Applicability Domain", "Inside" if inside_ad else "Outside")

            if not inside_ad:
                st.warning(
                    f"This molecule is **outside the applicability domain** "
                    f"({n_violated} of {ad_bounds['n_components']} PCA components violated). "
                    f"The prediction may be unreliable."
                )
            else:
                st.success("Molecule is within the training chemical space. Prediction is reliable.")

            # Interpretation
            st.markdown("---")
            st.markdown("**Interpretation:**")
            if class_pred == "active":
                st.markdown(f"Predicted pchembl = **{pchembl_pred:.2f}** (> 7.0) suggests **potent** aromatase inhibition.")
            elif class_pred == "intermediate":
                st.markdown(f"Predicted pchembl = **{pchembl_pred:.2f}** (6.0-7.0) suggests **moderate** aromatase inhibition.")
            else:
                st.markdown(f"Predicted pchembl = **{pchembl_pred:.2f}** (< 6.0) suggests **weak/no** aromatase inhibition.")

# --- Fine Print ---
st.markdown("---")
with st.expander("About this predictor"):
    st.markdown("""
    **What is this?**  
    A QSAR (Quantitative Structure-Activity Relationship) model that predicts how strongly a molecule 
    inhibits Aromatase — an enzyme involved in estrogen production and a key drug target for breast cancer treatment.
    
    **Data Source**  
    Bioactivity data for Aromatase (CYP19A1, target ID: CHEMBL1978) was retrieved from 
    [ChEMBL](https://www.ebi.ac.uk/chembl/), a public database of drug-like molecules and their 
    measured biological activities. After curation (removing duplicates, filtering inconsistent measurements, 
    averaging replicates), 3,290 molecules with experimentally measured potency values (pchembl) were used for modelling.
    
    **Input Representation (Fingerprint)**  
    Molecules are converted into numerical vectors using *AtomPairs2D Count* — a fingerprint that encodes 
    which pairs of atom types exist in the molecule and how far apart they are (topological distance). 
    Think of it as a fixed-length feature vector (455 dimensions after filtering) where each dimension 
    counts how often a specific atom-pair pattern occurs. This is analogous to a bag-of-words representation 
    but for molecular substructures instead of text.
    
    **Machine Learning Model**  
    *Extra Trees* (Extremely Randomized Trees) — an ensemble of 200 decision trees, similar to Random Forest 
    but with random split thresholds for additional regularization. Selected as the best performer from a 
    systematic comparison of 16 algorithms across 12 fingerprint types and 2 data splitting strategies 
    (384 total configurations evaluated via 5-fold cross-validation).
    
    **Performance (test set, 80/20 random split)**  
    - Regression: R² = 0.697 (explains ~70% of potency variance), RMSE = 0.713 log units
    - Classification: Balanced Accuracy = 68.8%, MCC = 0.540
    
    **Applicability Domain (AD)**  
    Not all molecules can be reliably predicted — only those structurally similar to the training data. 
    The AD is defined by fitting PCA (178 components, 95% variance) on the training fingerprints and 
    checking if a new molecule's PCA scores fall within the observed training range. If any component 
    exceeds the training min/max, the molecule is flagged as "outside AD" and predictions may be unreliable. 
    96.3% of the held-out test set falls within the AD.
    
    **Activity Classes (pchembl scale)**  
    pchembl is the negative log of IC50 in molar units — higher = more potent.
    - Active: pchembl > 7 (IC50 < 100 nM — strong inhibitor)
    - Intermediate: 6 ≤ pchembl ≤ 7 (IC50 between 100 nM and 1 μM)
    - Inactive: pchembl < 6 (IC50 > 1 μM — weak or no inhibition)
    
    **Methodology reference**: Schaduangrat et al. (2021). ERpred: a web server for the prediction 
    of subtype-specific estrogen receptor antagonists. *PeerJ* 9:e11716.
    """)
