"""
Build 16 regression models on MACCS fingerprints for aromatase pchembl prediction.

Uses the random 80/20 split (seed=42) and filtered MACCS features (114 bits).
Reports R², RMSE, and MAE for training set, test set, and 10-fold cross-validation.
"""
import sys
import os
import time
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, AdaBoostRegressor,
                              HistGradientBoostingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_predict, KFold
from xgboost import XGBRegressor

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = "../data/processed"
FP_DIR = "../data/fingerprints_filtered"
SPLIT_DIR = "../data/splits"
OUTPUT_DIR = "../data/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 10


def build_models():
    """Define all 16 regression models."""
    return [
        ("Ridge", Ridge(alpha=1.0)),
        ("Lasso", Lasso(alpha=0.1, max_iter=5000)),
        ("ElasticNet", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
        ("Bayesian Ridge", BayesianRidge()),
        ("PLS", PLSRegression(n_components=10)),
        ("KNN", KNeighborsRegressor(n_neighbors=5)),
        ("SVR (RBF)", SVR(kernel="rbf", C=1.0, epsilon=0.1)),
        ("Kernel Ridge (RBF)", KernelRidge(alpha=1.0, kernel="rbf")),
        ("Decision Tree", DecisionTreeRegressor(random_state=RANDOM_STATE)),
        ("Random Forest", RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)),
        ("Extra Trees", ExtraTreesRegressor(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)),
        ("Gradient Boosting", GradientBoostingRegressor(n_estimators=500, random_state=RANDOM_STATE)),
        ("XGBoost", XGBRegressor(n_estimators=500, learning_rate=0.1, random_state=RANDOM_STATE,
                                  verbosity=0, n_jobs=-1)),
        ("Hist Gradient Boosting", HistGradientBoostingRegressor(max_iter=500, random_state=RANDOM_STATE)),
        ("AdaBoost", AdaBoostRegressor(n_estimators=500, random_state=RANDOM_STATE)),
        ("MLP", MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=500,
                             random_state=RANDOM_STATE, early_stopping=True)),
    ]


def compute_metrics(y_true, y_pred):
    """Compute R², RMSE, and MAE."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return r2, rmse, mae


def main():
    # Load train/test split IDs
    train_df = pd.read_csv(f"{SPLIT_DIR}/random_train.csv")
    test_df = pd.read_csv(f"{SPLIT_DIR}/random_test.csv")
    train_ids = set(train_df["molecule_chembl_id"])
    test_ids = set(test_df["molecule_chembl_id"])

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Load full dataset and filtered MACCS fingerprints (row-aligned)
    df_full = pd.read_csv(f"{DATA_DIR}/aromatase_bioactivity_clean.csv")
    fp_full = pd.read_csv(f"{FP_DIR}/fingerprints_maccs.csv")

    # Apply same filter as split script: exact measurements with pchembl
    mask = (df_full["standard_relation"] == "=") & df_full["pchembl_value"].notna()
    df = df_full[mask].reset_index(drop=True)
    fp = fp_full[mask.values].reset_index(drop=True)

    # Split into train/test by molecule_chembl_id
    train_mask = df["molecule_chembl_id"].isin(train_ids)
    test_mask = df["molecule_chembl_id"].isin(test_ids)

    fp_cols = [c for c in fp.columns if c != "molecule_chembl_id"]
    X_train = fp.loc[train_mask, fp_cols].values.astype(np.float32)
    X_test = fp.loc[test_mask, fp_cols].values.astype(np.float32)
    y_train = df.loc[train_mask, "pchembl_value"].values
    y_test = df.loc[test_mask, "pchembl_value"].values

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"y_train: mean={y_train.mean():.3f}, y_test: mean={y_test.mean():.3f}")
    print(f"\nTraining 16 models on MACCS fingerprints ({X_train.shape[1]} bits)...")
    print(f"10-fold CV on training set (seed={RANDOM_STATE})")
    print("=" * 90)

    # CV splitter
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Train and evaluate
    results = []
    predictions_test = {"molecule_chembl_id": df.loc[test_mask, "molecule_chembl_id"].values,
                        "y_true": y_test}

    models = build_models()
    for name, model in models:
        t0 = time.time()
        print(f"  {name}...", end=" ", flush=True)

        try:
            # 10-fold cross-validation on training set
            y_cv_pred = cross_val_predict(model, X_train, y_train, cv=kf, n_jobs=1)
            r2_cv, rmse_cv, mae_cv = compute_metrics(y_train, y_cv_pred)

            # Fit on full training set
            model.fit(X_train, y_train)

            # Training set predictions
            y_train_pred = model.predict(X_train)
            if y_train_pred.ndim > 1:
                y_train_pred = y_train_pred.ravel()
            r2_train, rmse_train, mae_train = compute_metrics(y_train, y_train_pred)

            # Test set predictions
            y_test_pred = model.predict(X_test)
            if y_test_pred.ndim > 1:
                y_test_pred = y_test_pred.ravel()
            r2_test, rmse_test, mae_test = compute_metrics(y_test, y_test_pred)

            elapsed = time.time() - t0
            print(f"Train R²={r2_train:.4f} | CV R²={r2_cv:.4f} | Test R²={r2_test:.4f} ({elapsed:.1f}s)")

            results.append({
                "Model": name,
                "R2_train": r2_train,
                "RMSE_train": rmse_train,
                "MAE_train": mae_train,
                "R2_CV": r2_cv,
                "RMSE_CV": rmse_cv,
                "MAE_CV": mae_cv,
                "R2_test": r2_test,
                "RMSE_test": rmse_test,
                "MAE_test": mae_test,
                "Time_s": elapsed,
            })

            predictions_test[name] = y_test_pred

        except Exception as e:
            elapsed = time.time() - t0
            print(f"FAILED ({elapsed:.1f}s): {e}")
            results.append({
                "Model": name,
                "R2_train": np.nan, "RMSE_train": np.nan, "MAE_train": np.nan,
                "R2_CV": np.nan, "RMSE_CV": np.nan, "MAE_CV": np.nan,
                "R2_test": np.nan, "RMSE_test": np.nan, "MAE_test": np.nan,
                "Time_s": elapsed,
            })

    # Results table
    results_df = pd.DataFrame(results).sort_values("R2_test", ascending=False)
    results_df.to_csv(f"{OUTPUT_DIR}/model_results_maccs.csv", index=False)

    # Predictions
    pred_df = pd.DataFrame(predictions_test)
    pred_df.to_csv(f"{OUTPUT_DIR}/predictions_maccs.csv", index=False)

    # Print final table
    print("\n" + "=" * 120)
    print(f"{'MODEL RESULTS — MACCS Fingerprints (114 bits) — Train / 10-Fold CV / Test':^120}")
    print("=" * 120)
    print(f"\n{'#':<3} {'Model':<22} {'R²_train':<9} {'RMSE_tr':<9} {'MAE_tr':<8} "
          f"{'R²_CV':<9} {'RMSE_CV':<9} {'MAE_CV':<8} "
          f"{'R²_test':<9} {'RMSE_te':<9} {'MAE_te':<8} {'Time':<5}")
    print("-" * 120)
    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        print(f"{i:<3} {row['Model']:<22} "
              f"{row['R2_train']:<9.4f} {row['RMSE_train']:<9.4f} {row['MAE_train']:<8.4f} "
              f"{row['R2_CV']:<9.4f} {row['RMSE_CV']:<9.4f} {row['MAE_CV']:<8.4f} "
              f"{row['R2_test']:<9.4f} {row['RMSE_test']:<9.4f} {row['MAE_test']:<8.4f} "
              f"{row['Time_s']:<5.1f}s")

    print(f"\nBest model (by test R²): {results_df.iloc[0]['Model']} "
          f"(R²={results_df.iloc[0]['R2_test']:.4f}, RMSE={results_df.iloc[0]['RMSE_test']:.4f}, "
          f"MAE={results_df.iloc[0]['MAE_test']:.4f})")
    print(f"\nSaved to:")
    print(f"  {OUTPUT_DIR}/model_results_maccs.csv")
    print(f"  {OUTPUT_DIR}/predictions_maccs.csv")


if __name__ == "__main__":
    main()
