# QSAR Modelling of Aromatase (CYP19A1) Inhibitors

A complete Quantitative Structure-Activity Relationship (QSAR) pipeline for predicting the potency of aromatase inhibitors using molecular fingerprints and machine learning.

## Why This Matters

**Aromatase (CYP19A1)** is the enzyme responsible for converting androgens to estrogens. It is a validated drug target for **estrogen receptor-positive (ER+) breast cancer**, which accounts for ~70% of all breast cancers. Approved aromatase inhibitors (letrozole, anastrozole, exemestane) are frontline therapies for postmenopausal ER+ breast cancer patients.

Computational prediction of aromatase inhibitory potency can:
- **Accelerate drug discovery** by screening millions of virtual compounds before synthesis
- **Reduce costs** by prioritising the most promising candidates for lab testing
- **Guide medicinal chemistry** by revealing which molecular features drive potency

## Approach

```
ChEMBL Data → Cleaning → Fingerprints → Feature Selection → Splitting → ML Models → Evaluation
```

### 1. Data Collection
Bioactivity data (IC50, Ki) retrieved from ChEMBL for 3,399 unique molecules tested against human aromatase. Activity is quantified as **pchembl_value** (= -log10 of IC50/Ki in molar), where higher values mean more potent inhibitors.

### 2. Molecular Fingerprints
Each molecule's structure is encoded as a binary/count vector using 12 fingerprint types from the PaDEL-Descriptor framework:

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
Near-constant features (>95% same value across molecules) are removed, reducing total features from 17,196 to 3,993.

### 4. Data Splitting
Two strategies ensure robust evaluation:
- **Random split** (80/20, stratified) — standard approach
- **Kennard-Stone** (80/20) — training set maximally spans chemical space

### 5. Machine Learning
16 regression algorithms are trained on each fingerprint type:

| Family | Models |
|--------|--------|
| Linear | Ridge, Lasso, ElasticNet, Bayesian Ridge, PLS |
| Instance-based | K-Nearest Neighbors |
| Kernel | SVR (RBF), Kernel Ridge |
| Tree | Decision Tree |
| Ensemble (Bagging) | Random Forest, Extra Trees |
| Ensemble (Boosting) | Gradient Boosting, XGBoost, Hist GB, AdaBoost |
| Neural Network | MLP |

### 6. Evaluation
Models are evaluated with:
- **R²** (variance explained) — higher is better
- **RMSE** (root mean squared error) — lower is better
- **MAE** (mean absolute error) — lower is better

Reported on training set, 10-fold cross-validation, and held-out test set.

## Key Results

**Best model**: Extra Trees on AtomPairs2D Count fingerprint (Random split)
- Test R² = 0.694, RMSE = 0.716, MAE = 0.543

Tree-based ensembles (Random Forest, XGBoost, Hist GB) consistently outperform linear models across all fingerprint types. AtomPairs2D fingerprints provide the richest representation for this target.

## Repository Structure

```
├── scripts/                    # Pipeline scripts (numbered in execution order)
├── data/
│   ├── processed/              # Cleaned bioactivity data
│   ├── fingerprints/           # Raw computed fingerprints
│   ├── fingerprints_filtered/  # After near-constant removal
│   ├── splits/                 # Train/test split indices
│   └── models/                 # Results and predictions
├── notebooks/                  # Colab notebooks (click to run)
├── streamlit_app/              # Interactive EDA dashboard
└── requirements.txt
```

## Quick Start

### Run the Streamlit dashboard locally
```bash
pip install -r requirements.txt
python -m streamlit run streamlit_app/app.py
```

### Run models on Google Colab (GPU accelerated)
[![Open Models in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dom-castaneda/qsar-aromatase/blob/master/notebooks/colab_qsar_models.ipynb)

[![Open Split Viz in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dom-castaneda/qsar-aromatase/blob/master/notebooks/colab_split_visualization.ipynb)

## Dependencies

- Python 3.10+
- scikit-learn, xgboost, pandas, numpy
- RDKit (molecular fingerprint computation)
- matplotlib, seaborn, plotly (visualization)
- streamlit (interactive dashboard)

See `requirements.txt` for pinned versions.

## References

- Yap, C.W. (2011). PaDEL-Descriptor: An open source software to calculate molecular descriptors and fingerprints. *J. Comput. Chem.*, 32(7):1466-1474.
- Klekota, J. & Roth, F.P. (2008). Chemical substructures that enrich for biological activity. *Bioinformatics*, 24(21):2518-2525.
- Kennard, R.W. & Stone, L.A. (1969). Computer aided design of experiments. *Technometrics*, 11(1):137-148.
