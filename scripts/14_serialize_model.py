"""
Serialize the final model + AD components for the prediction page.

Trains Extra Trees (Regressor + Classifier) on AP2D_Count, Random split training set.
Saves model, scaler, PCA, and AD bounding box for use by the Streamlit app.
"""
import sys
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = "../data/processed"
FP_DIR = "../data/fingerprints_filtered"
SPLIT_DIR = "../data/splits"
OUTPUT_DIR = "../data/models/final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

# --- Load Data ---
print("Loading data...")
df_full = pd.read_csv(f"{DATA_DIR}/aromatase_bioactivity_clean.csv")
mask = (df_full["standard_relation"] == "=") & df_full["pchembl_value"].notna()
df = df_full[mask].reset_index(drop=True)

fp_full = pd.read_csv(f"{FP_DIR}/fingerprints_atompairs2d_count.csv")
fp = fp_full[mask.values].reset_index(drop=True)
fp_cols = [c for c in fp.columns if c != "molecule_chembl_id"]
X_all = np.nan_to_num(fp[fp_cols].values.astype(np.float32), nan=0.0)
y_reg = df["pchembl_value"].values

def classify(val):
    if val > 7: return "active"
    elif val < 6: return "inactive"
    else: return "intermediate"

le = LabelEncoder()
y_cls = le.fit_transform(df["pchembl_value"].apply(classify).values)

# Train split
train_ids = set(pd.read_csv(f"{SPLIT_DIR}/random_train.csv")["molecule_chembl_id"])
train_mask = df["molecule_chembl_id"].isin(train_ids).values
X_train = X_all[train_mask]
y_train_reg = y_reg[train_mask]
y_train_cls = y_cls[train_mask]

print(f"Training set: {X_train.shape[0]} molecules, {X_train.shape[1]} features")

# --- Train Regressor ---
print("\nTraining Extra Trees Regressor...")
reg_model = ExtraTreesRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
reg_model.fit(X_train, y_train_reg)
print("  Done.")

# --- Train Classifier ---
print("Training Extra Trees Classifier...")
cls_model = ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE,
                                  n_jobs=-1, class_weight="balanced")
cls_model.fit(X_train, y_train_cls)
print("  Done.")

# --- AD Components ---
print("Fitting AD components (StandardScaler + PCA)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
train_scores = pca.fit_transform(X_train_scaled)
ad_bounds = {
    "min": train_scores.min(axis=0).tolist(),
    "max": train_scores.max(axis=0).tolist(),
    "n_components": int(pca.n_components_),
}
print(f"  PCA: {pca.n_components_} components (95% variance)")

# --- Save Everything ---
print("\nSaving...")
joblib.dump(reg_model, f"{OUTPUT_DIR}/model_regressor.joblib")
joblib.dump(cls_model, f"{OUTPUT_DIR}/model_classifier.joblib")
joblib.dump(scaler, f"{OUTPUT_DIR}/ad_scaler.joblib")
joblib.dump(pca, f"{OUTPUT_DIR}/ad_pca.joblib")

with open(f"{OUTPUT_DIR}/feature_columns.json", "w") as f:
    json.dump(fp_cols, f)

with open(f"{OUTPUT_DIR}/ad_bounds.json", "w") as f:
    json.dump(ad_bounds, f)

with open(f"{OUTPUT_DIR}/label_classes.json", "w") as f:
    json.dump(list(le.classes_), f)

print(f"  model_regressor.joblib")
print(f"  model_classifier.joblib")
print(f"  ad_scaler.joblib")
print(f"  ad_pca.joblib")
print(f"  feature_columns.json ({len(fp_cols)} columns)")
print(f"  ad_bounds.json ({ad_bounds['n_components']} PCs)")
print(f"  label_classes.json ({list(le.classes_)})")
print("\nDone.")
