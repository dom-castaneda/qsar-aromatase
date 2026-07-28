# QSAR Modelling of Aromatase (CYP19A1) Inhibitors

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://qsar-aromatase.streamlit.app/)

A complete Quantitative Structure-Activity Relationship (QSAR) pipeline for predicting the potency of aromatase inhibitors — from raw ChEMBL data to a working web predictor.

## Why This Matters

**Aromatase (CYP19A1)** is the enzyme responsible for converting androgens to estrogens. It is a validated drug target for **estrogen receptor-positive (ER+) breast cancer**, which accounts for ~70% of all breast cancers. Approved aromatase inhibitors (letrozole, anastrozole, exemestane) are frontline therapies for postmenopausal ER+ breast cancer patients.

Computational prediction of aromatase inhibitory potency can:
- **Accelerate drug discovery** by screening millions of virtual compounds before synthesis
- **Reduce costs** by prioritising the most promising candidates for lab testing
- **Guide medicinal chemistry** by revealing which molecular features drive potency

## Live Predictor

The Streamlit app includes a **SMILES-to-activity prediction tool** as the landing page:
- Paste a SMILES string → get predicted pchembl value + activity class (active/intermediate/inactive)
- Molecular structure rendered via RDKit
- Applicability domain check flags molecules outside the training chemical space

```bash
pip install -r requirements.txt
python -m streamlit run streamlit_app/app.py
```

## Pipeline

```
ChEMBL Data → Cleaning → Fingerprints → Feature Selection → Splitting → ML Models → Evaluation → Prediction
```

### 1. Data Collection & Curation
Bioactivity data (IC50, Ki) retrieved from ChEMBL for 3,399 unique molecules tested against human aromatase. After deduplication, SD filtering, and mean aggregation: **3,290 molecules** with exact pchembl values retained for modelling.

### 2. Molecular Fingerprints (12 types, PaDEL framework)

| Fingerprint | Bits | Encoding |
|-------------|------|----------|
| MACCS | 166 | Predefined structural keys |
| PubChem | 881 | PubChem substructure definitions |
| Substructure | 307 | CDK functional group SMARTS |
| Substructure Count | 307 | Count of each substructure |
| Klekota-Roth | 4,860 | Activity-enriched substructures |
| KR Count | 4,860 | Count version |
| AtomPairs2D | 780 | Atom type pairs + distance |
| AP2D Count | 780 | Count version |
| CDK FP | 1,024 | Hashed topological paths |
| CDK Extended | 1,024 | Paths + ring features |
| CDK GraphOnly | 1,024 | Topology only (ignores chemistry) |
| E-State | 79 | Electrotopological atom types |

### 3. Feature Selection
- **Near-constant removal**: Features with >95% same value removed (17,196 → 3,993)
- **Collinearity removal**: One feature from each pair with |r| > 0.95 dropped (3,993 → 3,735)

### 4. Data Splitting
- **Random split** (80/20, stratified by activity class)
- **Kennard-Stone** (80/20, maximin distance on ECFP4)

### 5. Machine Learning (16 algorithms)

| Family | Models |
|--------|--------|
| Linear | Ridge, Lasso, ElasticNet, Bayesian Ridge, PLS |
| Instance-based | K-Nearest Neighbors |
| Kernel | SVR (RBF), Kernel Ridge |
| Tree | Decision Tree |
| Ensemble (Bagging) | Random Forest, Extra Trees |
| Ensemble (Boosting) | Gradient Boosting, XGBoost, Hist GB, AdaBoost |
| Neural Network | MLP |

Both regression (pchembl prediction) and classification (active/intermediate/inactive) tasks evaluated.

### 6. Hyperparameter Tuning
RandomizedSearchCV (100 iterations, 5-fold CV) on the best model. Conclusion: default parameters are near-optimal.

### 7. Applicability Domain
PCA bounding box (178 components, 95% variance) on training fingerprints. 96.3% of test molecules fall within the AD.

### 8. Feature Importance
Gini importance + permutation importance (10 repeats). 15/20 top features overlap between methods.

### 9. Statistical Class Comparison
Kruskal-Wallis + post-hoc Mann-Whitney U (Bonferroni) on 8 molecular descriptors. All significantly differ between activity classes (p < 0.05). MW has the strongest discriminating power.

## Key Results

**Best model**: Extra Trees on AtomPairs2D Count fingerprint (Random split)

| Task | Metric | Score |
|------|--------|-------|
| Regression | R² | 0.697 |
| Regression | RMSE | 0.713 |
| Classification | Balanced Accuracy | 0.688 |
| Classification | MCC | 0.540 |

Tree-based ensembles (Extra Trees, XGBoost, Hist GB) consistently outperform linear models. AtomPairs2D fingerprints provide the richest representation for this target.

## Repository Structure

```
├── scripts/                    # Pipeline scripts (numbered in execution order)
├── data/
│   ├── processed/              # Cleaned bioactivity data
│   ├── fingerprints/           # Raw computed fingerprints (14 CSVs)
│   ├── fingerprints_filtered/  # After near-constant removal
│   ├── fingerprints_reduced/   # After collinearity removal
│   ├── splits/                 # Train/test split indices
│   └── models/
│       ├── final/              # Serialized model + AD components
│       ├── results_*.csv       # Full model comparison results
│       └── *.json              # Tuning, AD, feature importance
├── notebooks/                  # Colab notebooks (click badges to run)
├── streamlit_app/              # Multi-page Streamlit dashboard + predictor
└── requirements.txt
```

## Streamlit Dashboard Pages

| Page | Content |
|------|---------|
| Home | **SMILES Activity Predictor** (landing page) |
| 1-9 | EDA: Overview, Bioactivity, Data Quality, Temporal, Molecular Properties, Chemical Space, Fingerprints, Correlations, QSAR Readiness |
| 10 | Model Performance (Regression/Classification toggle) |
| 11 | Molecular Fingerprint Correlation |
| 12 | Feature Importance |
| 13 | Hyperparameter Tuning |
| 14 | Applicability Domain |
| 15 | Statistical Class Comparison |

## Google Colab Notebooks

[![Regression Models](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dom-castaneda/qsar-aromatase/blob/master/notebooks/colab_qsar_models.ipynb) Regression (16 models × 12 FPs × 2 splits)

[![Classification Models](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dom-castaneda/qsar-aromatase/blob/master/notebooks/colab_qsar_classification.ipynb) Classification (16 classifiers × 12 FPs × 2 splits)

[![Hyperparameter Tuning](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dom-castaneda/qsar-aromatase/blob/master/notebooks/colab_hyperparameter_tuning.ipynb) Hyperparameter Tuning (Extra Trees)

[![Split Visualization](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dom-castaneda/qsar-aromatase/blob/master/notebooks/colab_split_visualization.ipynb) PCA/t-SNE Split Visualization

## Dependencies

- Python 3.10+
- scikit-learn, xgboost, pandas, numpy
- RDKit (molecular fingerprint computation)
- matplotlib, seaborn, plotly (visualization)
- streamlit (interactive dashboard)
- joblib (model serialization)

See `requirements.txt` for pinned versions.
