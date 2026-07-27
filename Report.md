# QSAR Aromatase Inhibitor Prediction — Methodology Report

## 1. Introduction

This project develops a machine learning model to predict the inhibitory potency of small molecules against **Aromatase (CYP19A1)**, a cytochrome P450 enzyme that catalyses the final step of estrogen biosynthesis. Aromatase inhibitors are a frontline therapy for estrogen receptor-positive (ER+) breast cancer, and computational prediction of inhibitory activity can accelerate the discovery of novel inhibitors.

The pipeline follows OECD principles for QSAR model validation: a defined endpoint, an unambiguous algorithm, a defined applicability domain, appropriate measures of goodness-of-fit and predictivity, and a mechanistic interpretation where possible.

---

## 2. Data Collection

Bioactivity data was retrieved from the **ChEMBL database** (target: CHEMBL1978, Homo sapiens CYP19A1) via the ChEMBL REST API.

- **Activity types**: IC50, Ki, pIC50
- **Raw dataset**: 5,097 activity records
- **Endpoint**: pchembl_value (= −log₁₀ of IC50/Ki in molar units; higher = more potent)

---

## 3. Data Curation

The raw dataset was cleaned through four sequential steps:

| Step | Method | Records |
|------|--------|---------|
| Raw | — | 5,097 |
| Deduplication | Exact match on (molecule, type, relation, value) | 4,641 |
| SD filter | Remove groups with pchembl SD > 3.0 | 4,637 |
| Mean aggregation | Average replicate measurements per molecule | 3,774 |
| Exact measurements only | Retain standard_relation = "=" with non-null pchembl | **3,290** |

A **bioactivity class** label was assigned based on pchembl thresholds:
- **Active**: pchembl > 7 (IC50 < 100 nM) — 966 molecules
- **Intermediate**: 6 ≤ pchembl ≤ 7 (IC50 100 nM – 1 μM) — 931 molecules
- **Inactive**: pchembl < 6 (IC50 > 1 μM) — 1,393 molecules

> See: **Streamlit App → Page 2 (Bioactivity)** for distribution plots and class balance.

---

## 4. Molecular Fingerprints

Each molecule's SMILES structure was converted into numerical feature vectors using **12 fingerprint types** from the PaDEL-Descriptor framework (9 fingerprint classes):

| Fingerprint | Dimensions | Encoding |
|-------------|-----------|----------|
| MACCS Keys | 167 | Predefined structural keys |
| PubChem CACTVS | 881 | PubChem substructure definitions |
| Substructure (binary) | 307 | CDK functional group SMARTS |
| Substructure (count) | 307 | Count of each substructure match |
| Klekota-Roth (binary) | 4,860 | Activity-enriched substructures |
| Klekota-Roth (count) | 4,860 | Count version |
| AtomPairs2D (binary) | 780 | Atom type pairs + topological distance |
| AtomPairs2D (count) | 780 | Count version |
| CDK Fingerprinter | 1,024 | Hashed topological paths (length 1–7) |
| CDK Extended | 1,024 | Paths + ring/frequency features |
| CDK GraphOnly | 1,024 | Graph skeleton (ignores atom types) |
| E-State | 79 | Electrotopological atom types |

Total raw features: **17,196** across all fingerprint types.

> See: **Streamlit App → Page 7 (Fingerprints)** for bit frequency analysis and sparsity comparison.

---

## 5. Feature Selection

### 5.1 Near-Constant Removal

Features where >95% of values are identical provide no discriminatory information and were removed.

- **Before**: 17,196 features
- **After**: 3,993 features (76.8% removed)

### 5.2 Collinearity Removal

For each pair of features with |Pearson r| > 0.95, one was removed (retaining the feature more correlated with pchembl_value).

- **Before**: 3,993 features
- **After**: 3,735 features (258 dropped from 291 collinear pairs)
- Most affected: CDK_GraphOnly (74% dropped), PubChem (41% dropped)
- Unaffected: AtomPairs2D, ECFP4 (zero collinear pairs)

> See: **Streamlit App → Page 11 (Molecular Fingerprint)** for intra-fingerprint correlation heatmaps.

---

## 6. Data Splitting

The modelling dataset (3,290 molecules) was split 80/20 into training (2,632) and test (658) sets using two strategies:

### 6.1 Random Split (Stratified)
- Stratified by activity class to preserve class proportions
- Random seed = 42 for reproducibility
- pchembl mean: train = 6.357, test = 6.418 (well-matched)

### 6.2 Kennard-Stone Split
- Maximin distance algorithm on ECFP4 fingerprints (180 bits after filtering)
- Training set selected to maximally span the chemical space
- pchembl mean: train = 6.285, test = 6.706 (test skews slightly more active)

> See: **Streamlit App → Page 6 (Chemical Space)** for PCA/t-SNE visualisations of both splits.

---

## 7. Machine Learning Models

### 7.1 Model Screening

**16 algorithms** were evaluated across **12 fingerprint types** and **2 split strategies**, totalling **384 configurations** per task (regression and classification).

| Family | Algorithms |
|--------|-----------|
| Linear/Regularised | Ridge, Lasso, ElasticNet, Bayesian Ridge, PLS |
| Instance-based | K-Nearest Neighbours |
| Kernel | SVR (RBF), Kernel Ridge |
| Tree | Decision Tree |
| Ensemble (Bagging) | Random Forest, Extra Trees |
| Ensemble (Boosting) | Gradient Boosting, XGBoost, Hist Gradient Boosting, AdaBoost |
| Neural Network | Multi-Layer Perceptron |

Evaluation protocol:
- 5-fold cross-validation on the training set
- Final evaluation on the held-out test set
- Regression metrics: R², RMSE, MAE
- Classification metrics: Balanced Accuracy, MCC, F1 (weighted)

### 7.2 Best Model

**Extra Trees on AtomPairs2D Count** (Random split) was the best-performing configuration:

| Task | Metric | Score |
|------|--------|-------|
| Regression | R² (test) | 0.697 |
| Regression | RMSE (test) | 0.713 |
| Classification | Balanced Accuracy (test) | 0.688 |
| Classification | MCC (test) | 0.540 |

Tree-based ensembles consistently outperformed linear models. The AtomPairs2D count fingerprint (encoding atom-type pair frequencies) provided the richest representation for this target.

> See: **Streamlit App → Page 10 (Model Performance)** for full heatmaps, scatter comparisons, and detailed results tables (toggle between Regression and Classification).

---

## 8. Hyperparameter Tuning

**RandomizedSearchCV** (100 iterations, 5-fold CV) was applied to the best model (Extra Trees on AP2D_Count, Random split).

Search space:
- `n_estimators`: [200, 500, 800, 1000]
- `max_depth`: [None, 20, 30, 50, 70]
- `min_samples_split`: [2, 5, 10, 15]
- `min_samples_leaf`: [1, 2, 4, 6]
- `max_features`: ["sqrt", "log2", 0.3, 0.5, 0.7, None]

**Result**: Default sklearn parameters are near-optimal for this dataset. Tuning produced negligible improvement (regression R² decreased slightly from 0.697 to 0.676; classification BalAcc improved marginally from 0.688 to 0.690).

The final deployed model uses default Extra Trees parameters with `n_estimators=200`.

> See: **Streamlit App → Page 13 (Hyperparameter Tuning)** for the default vs tuned comparison.

---

## 9. Applicability Domain

The applicability domain (AD) defines the chemical space for which the model can make reliable predictions. Following the ERpred methodology, a **PCA bounding box** was constructed:

1. StandardScaler fitted on training fingerprints (AP2D_Count, 455 features)
2. PCA fitted on scaled training data → 178 components (95% cumulative variance)
3. Bounding box defined as [min, max] of each PC score from the training set
4. A query molecule is "inside AD" if all 178 PC scores fall within the training bounds

**Coverage**: 96.3% of held-out test molecules (730/758) fall within the AD. The 28 outside-AD molecules are structurally dissimilar from the training data; predictions for these should be treated with caution.

> See: **Streamlit App → Page 14 (Applicability Domain)** for the interactive PCA scatter plot.

---

## 10. Feature Importance

Two complementary methods were used to identify the most predictive fingerprint features:

### 10.1 Gini Importance (Mean Decrease in Impurity)
Built-in to Extra Trees — measures how much each feature reduces variance across all tree splits during training. Fast but can be biased toward high-cardinality features.

### 10.2 Permutation Importance (Mean Decrease in R²)
Shuffles each feature on the test set and measures the drop in R² (10 repeats). Model-agnostic, unbiased, and reflects true predictive contribution.

**Result**: 15 out of the top 20 features overlap between both methods, indicating robust identification of important features. Top features: AP2DC_39, AP2DC_221, AP2DC_280, AP2DC_477, AP2DC_148.

> See: **Streamlit App → Page 12 (Feature Importance)** for bar charts, scatter plot, and detailed rankings.

---

## 11. Statistical Comparison of Bioactivity Classes

A **Kruskal-Wallis test** (non-parametric alternative to one-way ANOVA) was used to compare molecular descriptor distributions across the three activity classes, followed by post-hoc pairwise **Mann-Whitney U tests** with Bonferroni correction.

Eight physicochemical descriptors were tested:

| Descriptor | H statistic | p-value | Effect Size (η²) |
|-----------|-------------|---------|-------------------|
| Molecular Weight | 95.29 | 2.0×10⁻²¹ | 0.028 |
| H-Bond Donors | 31.98 | 1.1×10⁻⁷ | 0.009 |
| LogP | 20.94 | 2.8×10⁻⁵ | 0.006 |
| H-Bond Acceptors | 17.69 | 1.4×10⁻⁴ | 0.005 |
| Fraction sp3 | 15.32 | 4.7×10⁻⁴ | 0.004 |
| Aromatic Rings | 15.19 | 5.0×10⁻⁴ | 0.004 |
| TPSA | 10.52 | 5.2×10⁻³ | 0.003 |
| Rotatable Bonds | 6.92 | 3.1×10⁻² | 0.002 |

All 8 descriptors are significantly different between classes (p < 0.05). Active molecules tend to be heavier, more aromatic, less sp3-rich, and have fewer H-bond donors compared to inactives.

> See: **Streamlit App → Page 15 (Class Comparison)** for interactive box plots and pairwise significance tables.

---

## 12. Results Summary

| Aspect | Result |
|--------|--------|
| Best algorithm | Extra Trees (Extremely Randomised Trees) |
| Best fingerprint | AtomPairs2D Count (455 features after filtering) |
| Best split strategy | Random (stratified) |
| Regression R² | 0.697 (explains ~70% of potency variance) |
| Regression RMSE | 0.713 log units |
| Classification Balanced Accuracy | 68.8% |
| Classification MCC | 0.540 |
| Applicability domain coverage | 96.3% of test set |
| Hyperparameter tuning outcome | Default parameters near-optimal |
| Key structural differentiators | MW, aromatic rings, H-bond donors, sp3 fraction |

The final model is deployed as an interactive prediction tool (Streamlit landing page) where users can input a SMILES structure and receive a predicted potency score, activity classification, and reliability assessment.

> See: **Streamlit App → Home (Landing Page)** for the live predictor.
