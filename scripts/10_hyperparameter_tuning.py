"""
Hyperparameter tuning for the best model (Extra Trees) on AP2D_Count fingerprint.
Uses RandomizedSearchCV with 5-fold CV, 100 iterations.
Tunes both regression and classification versions.
"""
import sys
import os
import json
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             balanced_accuracy_score, f1_score, matthews_corrcoef)
from sklearn.preprocessing import LabelEncoder

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = "../data/processed"
FP_DIR = "../data/fingerprints_filtered"
SPLIT_DIR = "../data/splits"
OUTPUT_DIR = "../data/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
N_ITER = 30
N_FOLDS = 5

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

# Classification labels
def classify(val):
    if val > 7: return "active"
    elif val < 6: return "inactive"
    else: return "intermediate"

le = LabelEncoder()
y_cls = le.fit_transform(df["pchembl_value"].apply(classify).values)

# Train/test split
train_ids = set(pd.read_csv(f"{SPLIT_DIR}/random_train.csv")["molecule_chembl_id"])
test_ids = set(pd.read_csv(f"{SPLIT_DIR}/random_test.csv")["molecule_chembl_id"])
train_mask = df["molecule_chembl_id"].isin(train_ids).values
test_mask = df["molecule_chembl_id"].isin(test_ids).values

X_train, X_test = X_all[train_mask], X_all[test_mask]
y_train_reg, y_test_reg = y_reg[train_mask], y_reg[test_mask]
y_train_cls, y_test_cls = y_cls[train_mask], y_cls[test_mask]

print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]} | Features: {X_train.shape[1]}")

# --- Search Space ---
param_dist = {
    "n_estimators": [200, 500, 1000],
    "max_depth": [None, 20, 30, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.3, 0.5, None],
}

# --- Regression Tuning ---
print(f"\n{'='*60}")
print("REGRESSION: Extra Trees on AP2D_Count")
print(f"{'='*60}")
print(f"RandomizedSearchCV: {N_ITER} iterations, {N_FOLDS}-fold CV")

t0 = time.time()
reg_search = RandomizedSearchCV(
    ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=N_ITER,
    cv=N_FOLDS,
    scoring="r2",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
)
reg_search.fit(X_train, y_train_reg)
reg_time = time.time() - t0

print(f"\nCompleted in {reg_time:.1f}s")
print(f"Best CV R²: {reg_search.best_score_:.4f}")
print(f"Best params: {reg_search.best_params_}")

# Evaluate on test set
best_reg = reg_search.best_estimator_
y_pred_train = best_reg.predict(X_train)
y_pred_test = best_reg.predict(X_test)

r2_train = r2_score(y_train_reg, y_pred_train)
r2_test = r2_score(y_test_reg, y_pred_test)
rmse_train = np.sqrt(mean_squared_error(y_train_reg, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test_reg, y_pred_test))
mae_train = mean_absolute_error(y_train_reg, y_pred_train)
mae_test = mean_absolute_error(y_test_reg, y_pred_test)

print(f"\nTest Performance:")
print(f"  R²:   train={r2_train:.4f} | test={r2_test:.4f}")
print(f"  RMSE: train={rmse_train:.4f} | test={rmse_test:.4f}")
print(f"  MAE:  train={mae_train:.4f} | test={mae_test:.4f}")
print(f"\nBaseline (default params): R²=0.6967, RMSE=0.7128")
print(f"Improvement: R² {r2_test - 0.6967:+.4f}, RMSE {rmse_test - 0.7128:+.4f}")

# --- Classification Tuning ---
print(f"\n{'='*60}")
print("CLASSIFICATION: Extra Trees on AP2D_Count")
print(f"{'='*60}")
print(f"RandomizedSearchCV: {N_ITER} iterations, {N_FOLDS}-fold CV")

t0 = time.time()
cls_search = RandomizedSearchCV(
    ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"),
    param_distributions=param_dist,
    n_iter=N_ITER,
    cv=N_FOLDS,
    scoring="balanced_accuracy",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
)
cls_search.fit(X_train, y_train_cls)
cls_time = time.time() - t0

print(f"\nCompleted in {cls_time:.1f}s")
print(f"Best CV BalAcc: {cls_search.best_score_:.4f}")
print(f"Best params: {cls_search.best_params_}")

# Evaluate on test set
best_cls = cls_search.best_estimator_
y_pred_train_cls = best_cls.predict(X_train)
y_pred_test_cls = best_cls.predict(X_test)

balacc_train = balanced_accuracy_score(y_train_cls, y_pred_train_cls)
balacc_test = balanced_accuracy_score(y_test_cls, y_pred_test_cls)
mcc_test = matthews_corrcoef(y_test_cls, y_pred_test_cls)
f1_test = f1_score(y_test_cls, y_pred_test_cls, average="weighted")

print(f"\nTest Performance:")
print(f"  BalAcc: train={balacc_train:.4f} | test={balacc_test:.4f}")
print(f"  MCC:   {mcc_test:.4f}")
print(f"  F1:    {f1_test:.4f}")
print(f"\nBaseline (default params): BalAcc=0.6883, MCC=0.5401")
print(f"Improvement: BalAcc {balacc_test - 0.6883:+.4f}, MCC {mcc_test - 0.5401:+.4f}")

# --- Save Results ---
results = {
    "regression": {
        "best_params": reg_search.best_params_,
        "best_cv_r2": reg_search.best_score_,
        "test_r2": r2_test,
        "test_rmse": rmse_test,
        "test_mae": mae_test,
        "train_r2": r2_train,
        "baseline_r2": 0.6967,
        "tuning_time_s": reg_time,
    },
    "classification": {
        "best_params": cls_search.best_params_,
        "best_cv_balacc": cls_search.best_score_,
        "test_balacc": balacc_test,
        "test_mcc": mcc_test,
        "test_f1": f1_test,
        "train_balacc": balacc_train,
        "baseline_balacc": 0.6883,
        "tuning_time_s": cls_time,
    },
    "fingerprint": "AP2D_Count",
    "split": "Random",
    "n_iter": N_ITER,
    "n_folds": N_FOLDS,
}

# Convert numpy types for JSON serialization
def convert(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

with open(f"{OUTPUT_DIR}/tuning_results.json", "w") as f:
    json.dump(results, f, indent=2, default=convert)

print(f"\nResults saved to {OUTPUT_DIR}/tuning_results.json")
print("Done.")
