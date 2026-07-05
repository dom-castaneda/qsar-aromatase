# EDA Overview — Aromatase (CYP19A1) Inhibitors

**Dataset**: `data/processed/aromatase_bioactivity_clean.csv`  
**Source**: ChEMBL (target CHEMBL1978)  
**Records**: 3,774 | **Unique molecules**: 3,399  
**Notebook**: `notebooks/eda_aromatase.ipynb`

---

## Dataset Summary

| Metric | Value |
|--------|-------|
| Total records | 3,774 |
| Unique molecules (InChIKey) | 3,399 |
| SMILES validity | 100% |
| With pchembl_value | 3,296 (87.3%) |
| Exact measurements (=) | 3,403 (90.2%) |
| Censored (>) | 275 (7.3%) |
| Year span | 1978–2024 |
| Assay types | IC50 (3,211), Ki (516), pIC50 (47) |

---

## Bioactivity Distribution

| Activity Class | Threshold | Count | % (of known) |
|----------------|-----------|-------|--------------|
| Active | pchembl >= 6.5 | 1,435 | 43.5% |
| Intermediate | 5.0 <= pchembl < 6.5 | 1,431 | 43.4% |
| Inactive | pchembl < 5.0 | 430 | 13.0% |
| Unknown | no pchembl | 478 | — |

- **pchembl_value**: mean=6.37, median=6.23, std=1.26, range 4.0–10.82
- **Active:Inactive ratio** = 3.3:1 — moderate imbalance

---

## Chemical Space

### Molecular Properties

| Property | Mean | Median | Std | Min | Max |
|----------|------|--------|-----|-----|-----|
| MW (Da) | 342.6 | 322.4 | 115.7 | 130.1 | 1815.2 |
| LogP | 4.1 | 4.0 | 1.4 | -1.4 | 11.5 |
| HBA | 3.3 | 3.0 | 1.8 | 0 | 16 |
| HBD | 0.7 | 0.0 | 1.0 | 0 | 9 |
| TPSA (A^2) | 54.9 | 46.5 | 32.7 | 0 | 245.3 |
| RotBonds | 3.6 | 3.0 | 3.0 | 0 | 28 |
| AromaticRings | 2.1 | 2.0 | 1.6 | 0 | 10 |
| FractionCSP3 | 0.4 | 0.2 | 0.3 | 0 | 1.0 |
| HeavyAtoms | 24.5 | 23.0 | 7.6 | 9 | 102 |
| TotalRings | 3.7 | 4.0 | 1.1 | 0 | 13 |

### Structural Classes

| Scaffold Type | Count | % |
|---------------|-------|---|
| N-heterocyclic ring | 2,060 | 54.6% |
| Steroidal (>=4 rings, Fsp3>0.5) | 1,184 | 31.4% |
| Imidazole-containing | 671 | 17.8% |
| Pyridine-containing | 659 | 17.5% |
| Triazole-containing | 381 | 10.1% |
| Coumarin scaffold | 90 | 2.4% |
| Flavone scaffold | 33 | 0.9% |

### Drug-likeness

- **Lipinski Ro5 compliant (0 violations)**: 2,828 (74.9%)
- MW distribution: <200 Da (52), 200–350 Da (2,362), 350–500 Da (1,145), >500 Da (215)

---

## Descriptor–Potency Relationships

### Pearson Correlations with pchembl_value

| Descriptor | r | Significance |
|------------|---|--------------|
| HBD | -0.124 | *** |
| AromaticRings | +0.071 | *** |
| FractionCSP3 | -0.061 | *** |
| HBA | +0.059 | *** |
| TPSA | +0.059 | *** |
| MW | +0.055 | ** |
| RotBonds | +0.012 | ns |
| LogP | -0.002 | ns |

Significance: \*\*\* p<0.001, \*\* p<0.01, ns = not significant

**Key observations**:
- All molecular property correlations with potency are weak (|r| < 0.13)
- HBD is the strongest predictor — fewer H-bond donors associated with higher potency
- LogP and RotBonds show no significant relationship with activity
- Non-linear relationships and scaffold-specific trends likely mask global correlations
- Fingerprint bits are far more predictive than bulk properties (see below)

### Kruskal-Wallis Tests (Active vs Intermediate vs Inactive)

All descriptors except RotBonds show statistically significant differences between activity classes (p < 0.001), confirming the trends seen in correlations.

---

## Chemical Space Visualisation (ECFP4)

- **PCA**: Continuous spread in PC1-PC2; no sharp separation between activity classes. Top 5 PCs explain modest variance — consistent with high-dimensional, diverse chemical space.
- **t-SNE**: Clear scaffold-based clustering with partial activity separation. Active compounds tend to cluster in specific regions, suggesting structure-activity relationships are scaffold-dependent.

---

## Fingerprint Analysis

### Bit Density

| Fingerprint | Bits | Density (%) |
|-------------|------|-------------|
| MACCS | 167 | highest |
| E-State | 79 | high |
| SubFP | 307 | moderate |
| CDK FP | 1,024 | moderate |
| CDK Extended | 1,024 | high (countSim) |
| CDK GraphOnly | 1,024 | low |
| AtomPairs2D | 780 | moderate |
| ECFP4 | 1,024 | low |
| KR | 4,860 | lowest |

### MACCS Keys (167 bits)
- Always off: 21 bits
- Rare (<5%): 50 bits
- Ubiquitous (>95%): 3 bits
- **Informative (5–95%)**: 114 bits

### Klekota-Roth (4,860 bits)
- Always zero: 3,415 (70.3%)
- Rare (<1%): 4,418 (90.9%)
- **Informative (5–95%)**: 172 (3.5%)
- Bits per molecule: mean=44, median=42, range 1–197

### Substructure Count (307 values)
- Count version of CDK SubstructureFingerprinter
- Max count: 66
- Non-zero bins/molecule: mean=14.7, median=14

### Top Fingerprint Bits Correlated with Potency

| Bit | Point-Biserial r | Frequency |
|-----|-----------------|-----------|
| ECFP4_932 | 0.366 | 28.5% |
| ECFP4_298 | 0.352 | 5.0% |
| KR_3951 | 0.351 | 4.1% |
| KR_4311 | 0.351 | 4.2% |
| MACCS_41 | 0.347 | 11.0% |
| KR_1152 | 0.347 | 11.0% |

Fingerprint bits reach r=0.37 — much stronger than any molecular property (max r=0.12).

---

## Data Quality

### Missing Values
- molecule_pref_name: 95.9% (expected — most molecules unnamed)
- pchembl_value: 12.7%
- standard_relation: 2.5%
- standard_units: 2.2%
- document_journal: 0.3%

### Deduplication Verification
- InChIKey <-> ChEMBL ID mapping: 1:1 (no ambiguity)
- Zero duplicate (molecule, standard_type) pairs after aggregation
- 361 molecules tested in multiple assay types (IC50 + Ki) — expected

### Censored Data
- 275 records with `>` relation (true potency could be higher)
- Recommendation: exclude from regression, retain for classification

---

## QSAR Readiness Assessment

### For Regression
- **3,290 molecules** with exact (`=`) pchembl_value
- Continuous target variable with good spread (range 4.0–10.82, std=1.26)
- Scaffold-based splitting recommended

### For Binary Classification
- **1,865 molecules** (1,435 active + 430 inactive)
- Class ratio 3.3:1 — moderate imbalance
- Consider: stratified splits, class weights, or SMOTE

### Recommended Approaches
1. **Fingerprint selection**: ECFP4 for general models, MACCS for interpretability, KR after feature selection
2. **Data splits**: Scaffold-based (Bemis-Murcko) to avoid data leakage from congeneric series
3. **Class imbalance**: Stratified k-fold or class-weighted loss functions
4. **Censored data**: Include in classification (if pchembl > 6.5 with `>`, still active)

---

## Figures Index

| # | File | Description |
|---|------|-------------|
| 01 | `01_bioactivity_distribution.png` | pchembl histogram + box plot by assay type |
| 02 | `02_activity_classes.png` | Activity class bar chart + pie |
| 03 | `03_missing_values.png` | Missing value heatmap |
| 04 | `04_record_multiplicity.png` | Records per molecule distribution |
| 05 | `05_temporal_trends.png` | Year distribution, potency by decade, cumulative discovery |
| 06 | `06_molecular_properties.png` | Descriptor histograms with Ro5 limits |
| 07 | `07_lipinski_compliance.png` | Ro5 violations + drug-likeness pie |
| 08 | `08_mw_logp_activity.png` | MW/LogP scatter by activity class and potency |
| 09 | `09_pca_chemical_space.png` | PCA scree + 2D projections |
| 10 | `10_tsne_chemical_space.png` | t-SNE by activity class and potency |
| 11 | `11_maccs_frequencies.png` | MACCS key frequency profile + top 20 |
| 12 | `12_kr_sparsity.png` | KR bit frequency, bits/molecule, density comparison |
| 13 | `13_property_correlations.png` | Property-potency Pearson correlations |
| 14 | `14_top_fp_bit_correlations.png` | Top 20 FP bits by point-biserial r |
| 15 | `15_correlation_heatmap.png` | Inter-property correlation matrix |
| 16 | `16_descriptors_vs_pchembl.png` | All 10 descriptors vs pchembl scatter + regression |
| 17 | `17_descriptor_pairplot.png` | Pairplot of key descriptors (Active vs Inactive) |
| 18 | `18_descriptors_violin_by_class.png` | Violin plots of descriptors by activity class |
| 19 | `19_mw_logp_hexbin.png` | MW vs LogP hexbin density + KDE contours |

All figures at 150 DPI, located in `data/figures/eda/`.
