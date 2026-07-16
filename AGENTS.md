# AGENTS.md — Session Log

## Project: QSAR Aromatase

**Target**: Aromatase (CHEMBL1978) — Homo sapiens, CYP19A1 (P11511)

---

## Step 1: Data Retrieval

**Script**: `fetch_aromatase_bioactivity.py`

Fetched bioactivity data from the ChEMBL REST API (`https://www.ebi.ac.uk/chembl/api/data/activity.json`) for aromatase (CHEMBL1978).

- Activity types queried: **Ki, IC50, pIC50**
- pChEMBL values included where available (field on each activity record)
- Paginated through the API (1000 records/page, 0.5s delay)
- **Output**: `aromatase_bioactivity.csv` — 5,097 records

| Type | Count |
|---|---|
| Ki | 641 |
| IC50 | 4,409 |
| pIC50 | 47 |
| With pChEMBL | 4,462 |

---

## Step 2: Data Cleaning

**Script**: `clean_aromatase_data.py`

### 2a. InChIKey Generation
- Converted SMILES to InChIKey using RDKit (`Chem.MolFromSmiles` -> `MolToInchi` -> `InchiToInchiKey`)
- 100% success rate (5,097/5,097)

### 2b. Exact Deduplication
- Removed rows with identical `(molecule_chembl_id, standard_type, standard_relation, standard_value)`
- 5,097 -> 4,641 (456 exact duplicates removed)

### 2c. SD Filter on pChEMBL
- Grouped by `(molecule_chembl_id, standard_type)`
- Computed standard deviation of `pchembl_value` for groups with 2+ measurements
- Removed groups with SD > 3.0 (inconsistent measurements)
- 2 groups removed (4 rows)

### 2d. Mean Aggregation
- For remaining groups with multiple measurements, took the mean of `standard_value` and `pchembl_value`
- Kept one representative row per group (first row's metadata)
- 4,637 -> 3,774 (863 groups collapsed)

**Output**: `aromatase_bioactivity_clean.csv` — 3,774 records, 3,399 unique molecules

| Type | Count |
|---|---|
| Ki | 516 |
| IC50 | 3,211 |
| pIC50 | 47 |
| With pChEMBL | 3,296 |

---

## Step 3: Molecular Fingerprints

**Script**: `compute_fingerprints.py`

Computed five fingerprint types for all 3,774 molecules:

### PubChem CACTVS (881 bits)
- Fetched via PubChem PUG REST API in batches of 100 InChIKeys
- Each bit = a defined substructure key (atom counts, ring systems, functional groups, atom pairs)
- 3,752 populated (99.4%), 22 molecules not found in PubChem
- Columns: `PubChem_0` through `PubChem_880`

### MACCS Keys (167 bits)
- Computed locally via RDKit (`MACCSkeys.GenMACCSKeys`)
- Each bit = a named structural pattern (e.g., ring, NH, S=O, aromatic)
- 3,774 populated (100%)
- Columns: `MACCS_0` through `MACCS_166`

### ECFP4 (1024 bits)
- Computed locally via RDKit Morgan fingerprint (radius=2, 1024 bits)
- Circular fingerprint encoding local atomic environments
- 3,774 populated (100%)
- Columns: `ECFP4_0` through `ECFP4_1023`

### CDK SubstructureFingerprinter (307 bits)
- Native Python/RDKit implementation of the CDK SubstructureFingerprinter
- 307 SMARTS patterns from SMARTS_InteLigand.txt (Christian Laggner, Inte:Ligand GmbH, LGPL)
- Each bit = a named functional group / structural feature (e.g., Aldehyde, Lactone, Aromatic, Primary_amine)
- 306/307 patterns parse in RDKit; "Salt" pattern (disconnected SMARTS) always returns 0
- 3,774 populated (100%)
- Columns: `SubFP_0` through `SubFP_306`
- Module: `scripts/substructure_fingerprint.py`

### Klekota-Roth (4,860 bits)
- Native Python/RDKit implementation of the CDK KlekotaRothFingerprinter
- 4,860 SMARTS patterns from KlekotaRothFingerprinter.java in the CDK (public domain)
- Based on: Klekota & Roth (2008) "Chemical substructures that enrich for biological activity", Bioinformatics 24(21):2518-2525
- Each bit = a specific substructure enriched for biological activity
- All 4,860 patterns parse in RDKit (0 failures)
- Mean active bits per molecule: 43.8 (range: 1-197)
- 3,774 populated (100%)
- Columns: `KRFP_0` through `KRFP_4859`
- Module: `scripts/klekota_roth_fingerprint.py`

**Output**: `aromatase_fingerprints.csv` — 3,774 rows x 7,258 columns (19 metadata + 7,239 fingerprint bits)

---

## Step 3b: Remaining Fingerprints (PaDEL 12/9 Set)

**Scripts**: `05a` through `05h`

Computed the remaining fingerprints to complete the standard PaDEL-Descriptor "12 fingerprints, 9 classes" set:

### AtomPairs2D (780 bits) — `05a`
- RDKit AtomPairGenerator hashed to 780 bits
- Encodes pairs of atom types + topological distance
- Bits/molecule: mean=183, median=168, range 34–738
- Columns: `AP2D_0` through `AP2D_779`

### AtomPairs2D Count (780 values) — `05b`
- Count version: records how many times each atom-pair hash occurs
- Max count value: 76
- Columns: `AP2DC_0` through `AP2DC_779`

### CDK Fingerprinter (1024 bits) — `05c`
- RDKit RDKitFPGenerator, hashed topological paths (length 1-7)
- Bits/molecule: mean=529, median=514, range 76–1009
- Columns: `CDK_0` through `CDK_1023`

### CDK Extended (1024 bits) — `05d`
- RDKit RDKitFPGenerator with countSimulation=True (adds ring/frequency features)
- Bits/molecule: mean=760, median=768, range 135–1024
- Columns: `CDKExt_0` through `CDKExt_1023`

### CDK GraphOnly (1024 bits) — `05e`
- Topological paths on molecular graph skeleton (all atoms→C, all bonds→single)
- Only encodes connectivity, ignoring atom types and bond orders
- Bits/molecule: mean=70, median=68, range 23–121
- Columns: `CDKGraph_0` through `CDKGraph_1023`

### E-State (79 bits) — `05f`
- Binarised electrotopological state fingerprint (1 if atom type present)
- 79 atom-type bins from Hall & Kier (1995)
- Bits/molecule: mean=8.5, median=8, range 3–17
- Columns: `EState_0` through `EState_78`

### E-State Count (79 values) — `05g`
- Continuous E-state sum values per atom type
- Value range: [-15.95, 120.31]
- Columns: `EStateC_0` through `EStateC_78`

### Klekota-Roth Count (4860 values) — `05h`
- Count version: number of unique substructure matches per SMARTS pattern
- Max count value: 171
- Non-zero bins/molecule: mean=43.8, median=42
- Columns: `KRFPC_0` through `KRFPC_4859`

### Substructure Count (307 values) — `05i`
- Count version of CDK SubstructureFingerprinter: number of unique matches per SMARTS
- Max count value: 66
- Non-zero bins/molecule: mean=14.7, median=14
- Columns: `SubFPC_0` through `SubFPC_306`

All 3,774 molecules computed successfully (0 failures) for all 9 fingerprints.

---

## Step 3c: Near-Constant Feature Removal

**Script**: `06_remove_near_constant.py`

Removed features where >95% of values are identical (uninformative for modelling).

- Total features reduced: 17,196 → 3,993 (76.8% removed)
- Output: `data/fingerprints_filtered/` (same filenames, filtered columns)

---

## Step 3c-ii: Collinearity Removal

**Script**: `09_remove_collinear.py`

Removed one feature from each pair with |Pearson r| > 0.95. Tie-breaking: keeps the feature more correlated with pchembl_value.

- Total features reduced: 3,993 → 3,735 (258 dropped, 291 collinear pairs identified)
- Output: `data/fingerprints_reduced/` (same filenames, fewer columns)
- Report: `data/models/collinear_pairs_dropped.csv` (full pair list)

| Fingerprint | Before | After | Dropped |
|-------------|--------|-------|---------|
| AtomPairs2D | 518 | 518 | 0 |
| AP2D_Count | 455 | 455 | 0 |
| CDK_Extended | 882 | 879 | 3 |
| CDK_FP | 1022 | 1021 | 1 |
| CDK_GraphOnly | 34 | 9 | 25 |
| ECFP4 | 180 | 180 | 0 |
| E-State | 25 | 24 | 1 |
| E-State Count | 25 | 25 | 0 |
| KR | 172 | 134 | 38 |
| KR_Count | 173 | 128 | 45 |
| MACCS | 114 | 102 | 12 |
| PubChem | 309 | 183 | 126 |
| Substructure | 41 | 38 | 3 |
| SubstructureCount | 43 | 39 | 4 |

Notable: CDK_GraphOnly lost 74% (massive redundancy from stripped topology). PubChem lost 41%. AtomPairs2D and ECFP4 are collinearity-free.

---

## Step 3d: Data Splitting

**Script**: `07_data_split.py`

Split the modelling dataset (3,290 molecules with exact pchembl_value) into 80/20 train/test:

### Random Split (Stratified)
- Stratified by activity class (Active/Intermediate/Inactive)
- Train: 2,632 | Test: 658
- pchembl mean: train=6.357, test=6.418 (well-matched)
- Activity balance preserved across splits

### Kennard-Stone Split
- Maximin distance algorithm on ECFP4 fingerprints (180 bits)
- Selects training set to maximally span the chemical space
- Train: 2,632 | Test: 658
- pchembl mean: train=6.285, test=6.706 (test skews slightly more active)
- KS places more diverse/extreme molecules in training

**Output**: `data/splits/`
- `random_train.csv`, `random_test.csv`
- `kennard_stone_train.csv`, `kennard_stone_test.csv`

---

## Step 5: Machine Learning Models

**Script**: `08_build_models.py`

Trained 16 regression models on MACCS fingerprints (114 bits after filtering) to predict pchembl_value.
Split: random 80/20 (seed=42). Train: 2,732 | Test: 758.

### Results (ranked by test R²)

| # | Model | R² (train) | R² (test) | RMSE (train) | RMSE (test) |
|---|-------|-----------|----------|-------------|------------|
| 1 | Random Forest | 0.7971 | 0.5794 | 0.5645 | 0.8394 |
| 2 | XGBoost | 0.8137 | 0.5779 | 0.5408 | 0.8409 |
| 3 | Hist Gradient Boosting | 0.7842 | 0.5701 | 0.5821 | 0.8486 |
| 4 | KNN | 0.6316 | 0.5197 | 0.7606 | 0.8970 |
| 5 | MLP | 0.7081 | 0.5157 | 0.6770 | 0.9007 |
| 6 | Gradient Boosting | 0.6474 | 0.5125 | 0.7441 | 0.9037 |
| 7 | SVR (RBF) | 0.5842 | 0.4825 | 0.8081 | 0.9310 |
| 8 | Extra Trees | 0.8241 | 0.4539 | 0.5256 | 0.9565 |
| 9 | Decision Tree | 0.8241 | 0.4481 | 0.5256 | 0.9615 |
| 10 | Kernel Ridge (RBF) | 0.4268 | 0.3817 | 0.9487 | 1.0177 |
| 11 | Ridge | 0.3905 | 0.3471 | 0.9783 | 1.0458 |
| 12 | PLS | 0.3785 | 0.3457 | 0.9879 | 1.0469 |
| 13 | Bayesian Ridge | 0.3814 | 0.3397 | 0.9856 | 1.0517 |
| 14 | AdaBoost | 0.2705 | 0.2632 | 1.0702 | 1.1110 |
| 15 | ElasticNet | 0.1498 | 0.1364 | 1.1554 | 1.2028 |
| 16 | Lasso | 0.0639 | 0.0555 | 1.2124 | 1.2578 |

**Best model**: Random Forest (R²=0.579, RMSE=0.839)

**Output**: `data/models/`
- `model_results_maccs.csv` — metrics for all 16 models
- `predictions_maccs.csv` — test set predictions from all models

---

## Step 5b: ML Models on Collinearity-Reduced Data

**Notebooks**:
- `colab_qsar_models_reduced.ipynb` — Regression (16 models × 12 FPs × 2 splits = 384 runs)
- `colab_qsar_classification_reduced.ipynb` — Classification (16 classifiers × 12 FPs × 2 splits = 384 runs)

Uses `data/fingerprints_reduced/` (collinear features removed at |r| > 0.95).

Fixes applied (all 4 notebooks):
- All iterative models capped at `max_iter=300` (LogReg, SVC, MLP, HistGB)
- Tree ensembles reduced to `n_estimators=200` (RF, Extra Trees, GB, XGBoost, AdaBoost)
- Cross-validation reduced from 10-fold to 5-fold (halves CV time)
- Progress prints every run for immediate feedback
- Output CSVs: `results_all_models_reduced.csv`, `results_classification_reduced.csv`

### Results Summary (all 4 notebooks, 384 runs each)

**Regression — Best models (test R²):**

| # | Model | Fingerprint | Split | R² | RMSE |
|---|-------|-------------|-------|-----|------|
| 1 | Extra Trees | AP2D_Count | Random | 0.6967 | 0.7128 |
| 2 | Hist Gradient Boosting | AP2D_Count | Random | 0.6906 | 0.7199 |
| 3 | XGBoost | AtomPairs2D | Random | 0.6852 | 0.7262 |
| 4 | Hist Gradient Boosting | AtomPairs2D | Random | 0.6829 | 0.7288 |
| 5 | XGBoost | AP2D_Count | Random | 0.6764 | 0.7362 |

**Classification — Best models (test Balanced Accuracy):**

| # | Model | Fingerprint | Split | BalAcc | MCC |
|---|-------|-------------|-------|--------|-----|
| 1 | Extra Trees | AP2D_Count | Random | 0.6883 | 0.5401 |
| 2 | Random Forest | AtomPairs2D | Random | 0.6858 | 0.5387 |
| 3 | XGBoost | AP2D_Count | Kennard-Stone | 0.6840 | 0.5285 |
| 4 | Hist Gradient Boosting | AtomPairs2D | Random | 0.6836 | 0.5409 |
| 5 | Extra Trees | AtomPairs2D | Random | 0.6822 | 0.5315 |

**Key findings:**
- AtomPairs2D (binary + count) dominates both tasks
- Tree ensembles (Extra Trees, HistGB, XGBoost) consistently top-performing
- Random split slightly outperforms Kennard-Stone
- Collinearity removal had no effect on AP2D (0 pairs dropped) — results identical between filtered and reduced

---

## Step 6: Streamlit Dashboard

**App**: `streamlit_app/`

Multi-page Streamlit dashboard for interactive exploration of results.

### Data Source
All fingerprint-loading pages use `data/fingerprints_reduced/` (collinearity-removed) as the default data source.

### Pages

| Page | File | Description |
|------|------|-------------|
| 6. Chemical Space | `6_Chemical_Space.py` | Lipinski descriptors tab + PCA/t-SNE embeddings tab with fingerprint selector, data dispersity metrics |
| 10. Model Performance | `10_Model_Performance.py` | Heatmaps (R²/RMSE/MAE/time), scatter comparison (Random vs KS), detailed results table with Num_Descriptors |
| 11. Molecular Fingerprint | `11_Molecular_Fingerprint.py` | Intra-fingerprint feature correlation heatmap with dropdown selector, correlation distribution, top correlated pairs |

### Page 11 — Molecular Fingerprint (Feature Correlation)
- Dropdown to select any of 14 fingerprints
- Pearson correlation heatmap between features within the selected fingerprint
- Top-variance feature selection for large fingerprints (configurable, default 200)
- Hierarchical clustering for feature ordering
- Correlation distribution histogram (|r| values)
- Summary statistics: mean/median |r|, count of |r| > 0.9 and |r| > 0.7 pairs
- Top 20 most correlated feature pairs table

---

## Step 4: Exploratory Data Analysis

**Notebook**: `notebooks/eda_aromatase.ipynb` (33 cells: 10 markdown, 23 code)

Comprehensive EDA of the curated aromatase bioactivity dataset (3,774 records, 3,399 unique molecules).

### 4a. Bioactivity Distribution
- pchembl_value: n=3,296 non-null, mean=6.37, median=6.23, std=1.26, range 4.0–10.82
- Activity classes (thresholds: active ≥6.5, inactive <5.0):
  - Active: 1,435 | Intermediate: 1,431 | Inactive: 430 | Unknown: 478
  - Active:Inactive ratio = 1:0.30 — moderate imbalance

### 4b. Data Quality
- Missing: molecule_pref_name (95.9%), pchembl_value (12.7%), standard_relation (2.5%)
- Exact measurements (=): 3,403 (90.2%), censored (>): 275 (7.3%)
- Unit heterogeneity: nM (3,632), uM (30), ug/mL (28) — absorbed by pchembl standardisation

### 4c. Temporal Trends
- Publication span: 1978–2024 with peak activity 2005–2020
- Potency distributions stable across decades

### 4d. Molecular Properties (Lipinski / Drug-likeness)
- Computed: MW, LogP, HBA, HBD, TPSA, RotBonds, AromaticRings, FractionCSP3
- Drug-like (0 Ro5 violations): 2,828 (74.9%)
- Top correlations with pchembl: HBD (r=−0.124), AromaticRings (r=+0.071)

### 4e. Chemical Space (ECFP4)
- PCA: Top 5 PCs capture modest variance; continuous spread in PC1-PC2
- t-SNE: Clear clustering by chemical scaffold, partial activity separation

### 4f. Fingerprint Analysis
- MACCS (167 bits): 114 informative bits (5–95% frequency), 21 always off
- KR FP (4,860 bits): 70.3% always zero, only 172 informative bits (3.5%); mean 44 bits/molecule
- Density comparison: MACCS > SubFP > KR > ECFP4

### 4g. Feature–Activity Correlations
- Top correlated FP bits with pchembl: ECFP4_932 (r=0.366), ECFP4_298 (r=0.352), KR_3951 (r=0.351)
- Molecular property correlations weak (|r| < 0.13); fingerprint bits more informative

### 4h. QSAR Readiness
- **Regression**: 3,290 molecules with exact pchembl_value
- **Classification**: 1,865 active+inactive (drop intermediate for binary task)
- **Recommendations**: scaffold-based splits, stratified sampling for class imbalance, ECFP4 for general models, MACCS for interpretability

---

## File Summary

```
qsar_aromatase/
├── AGENTS.md
├── scripts/
│   ├── 01_fetch_aromatase_bioactivity.py  # Fetches raw bioactivity data from ChEMBL API
│   ├── 02_clean_aromatase_data.py         # InChIKey generation, dedup, SD filter, mean aggregation
│   ├── 03a_substructure_fingerprint.py    # CDK SubstructureFingerprinter (307 SMARTS, native Python)
│   ├── 03b_klekota_roth_fingerprint.py    # Klekota-Roth Fingerprinter (4860 SMARTS, native Python)
│   ├── 03c_compute_fingerprints.py        # PubChem + MACCS + ECFP4 + SubFP + KR FP computation
│   ├── 04a_build_eda_notebook.py          # Generates EDA notebook programmatically
│   ├── 04b_save_eda_figures.py            # Saves EDA figures to data/figures/eda/
│   ├── 04c_save_descriptor_figures.py     # Descriptor vs pchembl exploration figures
│   ├── 05a_atompairs2d_fingerprint.py     # AtomPairs2D binary (780 bits)
│   ├── 05b_atompairs2d_count_fingerprint.py # AtomPairs2D count (780 values)
│   ├── 05c_cdk_fingerprint.py             # CDK Fingerprinter (1024 bits, paths 1-7)
│   ├── 05d_cdk_extended_fingerprint.py    # CDK Extended (1024 bits, countSimulation)
│   ├── 05e_cdk_graphonly_fingerprint.py   # CDK GraphOnly (1024 bits, graph skeleton)
│   ├── 05f_estate_fingerprint.py          # E-State binary (79 bits)
│   ├── 05g_estate_count_fingerprint.py    # E-State count (79 values, continuous)
│   ├── 05h_kr_count_fingerprint.py        # Klekota-Roth count (4860 values)
│   ├── 05i_substruct_count_fingerprint.py # Substructure count (307 values)
│   ├── 06_remove_near_constant.py         # Near-constant feature removal (>95% same value)
│   ├── 07_data_split.py                   # Random (stratified) + Kennard-Stone splitting
│   ├── 08_build_models.py                 # 16 regression models (local, MACCS only)
│   └── 09_remove_collinear.py             # Collinearity removal (|r| > 0.95)
├── data/
│   ├── raw/
│   │   └── aromatase_bioactivity.csv            # Raw data (5,097 rows)
│   ├── processed/
│   │   ├── aromatase_bioactivity_curated.csv    # Curated data with bioactivity_class column
│   │   ├── aromatase_bioactivity_clean.csv      # Cleaned data (3,774 rows)
│   │   └── aromatase_fingerprints.csv           # Combined dataset with all fingerprints (3,774 x 7,258)
│   ├── fingerprints/                            # Raw computed fingerprints (14 CSVs)
│   ├── fingerprints_filtered/                   # After near-constant removal (3,993 features)
│   ├── fingerprints_reduced/                    # After collinearity removal (3,735 features)
│   ├── splits/                                  # Train/test split files (Random + KS)
│   ├── models/
│   │   ├── results_all_models.csv               # Full results (384 rows × 13 columns)
│   │   ├── results_by_config/                   # 72 separate CSV files per config
│   │   └── collinear_pairs_dropped.csv          # Collinearity report
│   └── figures/                                 # EDA and descriptor figures
├── notebooks/
│   ├── eda_aromatase.ipynb                      # Exploratory Data Analysis
│   ├── colab_qsar_models.ipynb                  # 16 regressors × 12 FPs × 2 splits (Colab)
│   ├── colab_qsar_models_reduced.ipynb          # 16 regressors × 12 FPs × 2 splits, collinear-reduced (Colab)
│   ├── colab_qsar_classification.ipynb          # 16 classifiers (Colab)
│   ├── colab_qsar_classification_reduced.ipynb  # 16 classifiers × 12 FPs × 2 splits, collinear-reduced (Colab)
│   └── colab_split_visualization.ipynb          # PCA/t-SNE split visualizations (Colab)
├── streamlit_app/                               # Multi-page Streamlit dashboard
│   ├── utils.py                                 # Shared helpers (loads fingerprints_reduced/)
│   └── pages/
│       ├── 6_Chemical_Space.py                  # Lipinski + PCA/t-SNE tabs
│       ├── 10_Model_Performance.py              # Heatmaps, scatter, detailed table
│       └── 11_Molecular_Fingerprint.py          # Intra-FP feature correlation heatmap
├── README.md
└── requirements.txt
```

All scripts use relative paths and should be run from the `scripts/` directory.

## Dependencies

- Python 3.10+
- `requests` — ChEMBL API calls
- `pandas` — Data manipulation
- `rdkit` — SMILES parsing, InChIKey, MACCS, ECFP4
- `pubchempy` — PubChem fingerprint retrieval
- `matplotlib`, `seaborn` — Visualisation
- `scipy` — Statistical tests (Pearson, point-biserial)
- `scikit-learn` — PCA, t-SNE
- `nbformat` — Programmatic notebook generation

---

## To-Do

1. **Run regression/classification models on collinearity-reduced data**
   - Execute `colab_qsar_models_reduced.ipynb` (16 regressors × 12 FPs × 2 splits = 384 runs)
   - Execute `colab_qsar_classification_reduced.ipynb` (16 classifiers × 12 FPs × 2 splits = 384 runs)
   - Compare results against the original (filtered-only) fingerprint results

2. **Feature importance analysis**
   - Identify which fingerprint bits/features contribute most to model predictions
   - Methods: tree-based importances (RF, XGBoost), permutation importance, SHAP values
   - Rank features across fingerprint types and models

3. **Statistical comparison of the 3 bioactivity classes**
   - Compare molecular descriptor distributions across active, intermediate, and inactive classes
   - Statistical tests: Kruskal-Wallis (non-parametric) or one-way ANOVA with post-hoc pairwise tests
   - Identify descriptors/features that significantly differentiate classes
