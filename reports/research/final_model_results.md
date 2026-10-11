# EarlyDia Final Model Results

## Dataset Summary

| Sex | N (dev) | Positive | Prevalence | Features (after one-hot) |
|-----|---------|----------|------------|--------------------------|
| Women | 199,514 | 46,328 | 23.22% | 164 |
| Men | 29,151 | 9,259 | 31.76% | 145 |

**Target A definition (V5):**
- POS = K flag OR (glucose >= 141 mg/dL)
- NEG = all flags 0 AND glucose <= 100 mg/dL
- Gray zone (glucose 101-140, no flags) excluded
- K flags: self-reported diabetes (s728a/sm627a), told high glucose (sb56/smb56), on glucose medication (sb57/smb57)

**Cross-validation:** StratifiedGroupKFold(5 splits), grouped by PSU, stratified by target_a.
Preprocessing (imputation + encoding + scaling) fit per-fold only.

---

## Women: All Model Results (sorted by ROC-AUC)

| Model | ROC-AUC | SD | PR-AUC | Brier | Threshold | Sens | Spec | PPV | F1 |
|-------|---------|-----|--------|-------|-----------|------|------|-----|-----|
| lgbm_D | 0.8016 | 0.0017 | 0.5829 | 0.1776 | 0.501 | 0.708 | 0.745 | 0.457 | 0.555 |
| lgbm_B | 0.8014 | 0.0017 | 0.5815 | 0.1789 | 0.489 | 0.726 | 0.727 | 0.446 | 0.553 |
| lgbm_default | 0.8013 | 0.0016 | 0.5812 | 0.1790 | 0.499 | 0.715 | 0.738 | 0.452 | 0.554 |
| lgbm_C | 0.7982 | 0.0019 | 0.5754 | 0.1814 | 0.492 | 0.725 | 0.722 | 0.441 | 0.548 |
| lgbm_A | 0.7980 | 0.0019 | 0.5740 | 0.1817 | 0.498 | 0.719 | 0.729 | 0.445 | 0.550 |
| cb_B | 0.7979 | 0.0019 | 0.5740 | 0.1816 | 0.505 | 0.707 | 0.739 | 0.450 | 0.550 |
| cb_default | 0.7977 | 0.0019 | 0.5738 | 0.1816 | 0.486 | 0.727 | 0.718 | 0.438 | 0.546 |
| cb_A | 0.7948 | 0.0020 | 0.5681 | 0.1835 | 0.495 | 0.716 | 0.724 | 0.439 | 0.545 |
| cb_C | 0.7947 | 0.0020 | 0.5682 | 0.1835 | 0.507 | 0.702 | 0.738 | 0.448 | 0.547 |
| cb_D | 0.7936 | 0.0019 | 0.5675 | 0.1831 | 0.499 | 0.708 | 0.730 | 0.443 | 0.545 |
| xgb_D | 0.8001 | - | - | - | - | - | - | - | - |
| et_default | 0.7777 | 0.0019 | 0.5372 | 0.1812 | 0.485 | 0.681 | 0.734 | 0.436 | 0.532 |

**Best model: LightGBM lgbm_D (ROC-AUC = 0.8016 +/- 0.0017)**

---

## Men: All Model Results (sorted by ROC-AUC)

| Model | ROC-AUC | SD | PR-AUC | Brier | Threshold | Sens | Spec | PPV | F1 |
|-------|---------|-----|--------|-------|-----------|------|------|-----|-----|
| cb_B | 0.8033 | 0.0086 | 0.6678 | 0.1791 | 0.505 | 0.715 | 0.741 | 0.563 | 0.630 |
| cb_default | 0.8031 | 0.0084 | 0.6678 | 0.1789 | 0.494 | 0.727 | 0.731 | 0.557 | 0.631 |
| cb_C | 0.8018 | 0.0088 | 0.6647 | 0.1808 | 0.487 | 0.741 | 0.714 | 0.547 | 0.629 |
| cb_A | 0.8018 | 0.0084 | 0.6645 | 0.1808 | 0.500 | 0.726 | 0.727 | 0.553 | 0.628 |
| xgb_D | 0.8001 | 0.0082 | 0.6605 | 0.1622 | 0.300 | 0.747 | 0.708 | 0.544 | 0.629 |
| lgbm_C | 0.7988 | 0.0082 | 0.6581 | 0.1853 | 0.509 | 0.735 | 0.717 | 0.547 | 0.627 |
| lgbm_B | 0.7988 | 0.0078 | 0.6577 | 0.1841 | 0.496 | 0.737 | 0.715 | 0.546 | 0.628 |
| lgbm_default | 0.7982 | 0.0081 | 0.6566 | 0.1843 | 0.469 | 0.765 | 0.686 | 0.531 | 0.627 |
| lgbm_D | 0.7979 | 0.0076 | 0.6583 | 0.1815 | 0.534 | 0.684 | 0.765 | 0.575 | 0.625 |
| cb_D | 0.7974 | 0.0085 | 0.6585 | 0.1808 | 0.464 | 0.750 | 0.696 | 0.535 | 0.624 |
| lgbm_A | 0.7973 | 0.0085 | 0.6549 | 0.1867 | 0.537 | 0.704 | 0.744 | 0.562 | 0.625 |
| et_default | 0.7755 | 0.0070 | 0.6252 | 0.1892 | 0.490 | 0.708 | 0.702 | 0.525 | 0.603 |

**Best model: CatBoost cb_B (ROC-AUC = 0.8033 +/- 0.0086)**

---

## Best Model Detailed Metrics

### Women (LightGBM lgbm_D)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.8016 +/- 0.0017 |
| PR-AUC | 0.5829 |
| Brier Score | 0.1776 |
| Optimal Threshold (Youden J) | 0.501 |
| Sensitivity | 0.708 |
| Specificity | 0.745 |
| PPV | 0.457 |
| F1 Score | 0.555 |

**Operating points (women):**

| Target Specificity | Threshold | Sensitivity | PPV | Flag Rate |
|--------------------|-----------|-------------|-----|-----------|
| >= 0.80 | 0.556 | 0.644 | 0.493 | 30.3% |
| >= 0.85 | 0.614 | 0.571 | 0.535 | 24.8% |
| >= 0.90 | 0.680 | 0.477 | 0.591 | 18.8% |
| >= 0.95 | 0.763 | 0.339 | 0.672 | 11.7% |

### Men (CatBoost cb_B)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.8033 +/- 0.0086 |
| PR-AUC | 0.6678 |
| Brier Score | 0.1791 |
| Optimal Threshold (Youden J) | 0.505 |
| Sensitivity | 0.715 |
| Specificity | 0.741 |
| PPV | 0.563 |
| F1 Score | 0.630 |

**Operating points (men):**

| Target Specificity | Threshold | Sensitivity | PPV | Flag Rate |
|--------------------|-----------|-------------|-----|-----------|
| >= 0.80 | 0.562 | 0.643 | 0.600 | 34.1% |
| >= 0.85 | 0.618 | 0.566 | 0.637 | 28.2% |
| >= 0.90 | 0.677 | 0.476 | 0.689 | 22.0% |
| >= 0.95 | 0.764 | 0.330 | 0.754 | 13.9% |

---

## Top 15 Features by Importance

### Women (LightGBM lgbm_D)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | glucose_context_sb73 (time of glucose reading) | 725 |
| 2 | pulse_pressure | 641 |
| 3 | waist_hip_ratio | 612 |
| 4 | glucose_context_sb53 (time since last ate) | 571 |
| 5 | avg_systolic_bp | 436 |
| 6 | waist_height_ratio (V6) | 432 |
| 7 | waist_x_age (V5 interaction) | 427 |
| 8 | age_x_bmi (V3 interaction) | 402 |
| 9 | waist_circumference | 383 |
| 10 | hip_circumference | 379 |
| 11 | avg_diastolic_bp | 369 |
| 12 | bmi_x_systolic (V3 interaction) | 369 |
| 13 | age_x_systolic (V3 interaction) | 355 |
| 14 | glucose_context_sb54 (time since last drank) | 353 |
| 15 | bmi | 320 |

### Men (CatBoost cb_B)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | glucose_context_smb53 (time since last ate) | 14.2 |
| 2 | waist_x_age (V5 interaction) | 10.0 |
| 3 | glucose_context_smb55 (glucose ever checked) | 8.3 |
| 4 | age_x_systolic (V3 interaction) | 5.2 |
| 5 | pulse_pressure (V5) | 3.8 |
| 6 | age | 3.5 |
| 7 | waist_hip_ratio | 3.3 |
| 8 | age_x_diastolic (V3 interaction) | 3.1 |
| 9 | hypertension_self_report (V3) | 2.4 |
| 10 | waist_circumference | 2.4 |
| 11 | glucose_context_smb73 (time of glucose reading) | 2.4 |
| 12 | glucose_context_smb54 (time since last drank) | 2.2 |
| 13 | hip_circumference | 1.9 |
| 14 | avg_diastolic_bp | 1.9 |
| 15 | avg_systolic_bp | 1.7 |

---

## Calibration (10-bin)

### Women (lgbm_D)

| Predicted | Observed |
|-----------|----------|
| 0.026 | 0.078 |
| 0.049 | 0.150 |
| 0.093 | 0.248 |
| 0.143 | 0.349 |
| 0.201 | 0.449 |
| 0.279 | 0.549 |
| 0.365 | 0.649 |
| 0.494 | 0.749 |
| 0.661 | 0.847 |
| 0.852 | 0.930 |

### Men (cb_B)

| Predicted | Observed |
|-----------|----------|
| 0.033 | 0.076 |
| 0.070 | 0.150 |
| 0.130 | 0.249 |
| 0.213 | 0.348 |
| 0.273 | 0.449 |
| 0.370 | 0.549 |
| 0.468 | 0.648 |
| 0.614 | 0.749 |
| 0.750 | 0.849 |
| 0.895 | 0.930 |

---

## Cross-Algorithm Stacking Results

Meta-learner: Logistic Regression with balanced class weights, PSU-grouped 5-fold CV.

| Sex | Ensemble | ROC-AUC | Best Individual | Improvement |
|-----|----------|---------|-----------------|-------------|
| Women | all CatBoost + LightGBM | 0.8027 | 0.8016 (lgbm_D) | +0.0011 |
| Men | all CatBoost + LightGBM + XGBoost | 0.8042 | 0.8033 (cb_B) | +0.0009 |

---

## Model Configurations

### LightGBM lgbm_D (best for women)

```python
dict(n_estimators=200, learning_rate=0.05, max_depth=8,
     num_leaves=63, min_child_samples=50, subsample=0.8,
     colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=1.0,
     is_unbalance=True, random_state=42)
```

### CatBoost cb_B (best for men)

```python
dict(depth=6, learning_rate=0.03, iterations=500,
     auto_class_weights='Balanced', random_seed=42,
     l2_leaf_reg=10.0, min_data_in_leaf=300, subsample=0.8)
```

---

## Feature Engineering Versions

- **V1:** Base demographics, anthropometry, BP, tobacco, alcohol, diet
- **V3:** Physical activity, hypertension self-report, fasting context, age×BMI/BP interactions
- **V4:** Comorbidities (5 conditions), BP medication, waist/hip circumference, waist-hip ratio
- **V5:** BP aggregates (MAP, pulse pressure), waist interactions, diet diversity score, BMI categories
- **V6:** Waist-to-height ratio, tobacco chewing indicator, age categories

---

## Key Findings

1. **LightGBM outperforms CatBoost for women** (0.8016 vs 0.7979); **CatBoost slightly outperforms LightGBM for men** (0.8033 vs 0.7988)
2. **Glucose measurement context** (time since last ate, time of reading) is the #1 predictor for both sexes
3. **Anthropometric features** (waist-hip ratio, waist circumference, waist-height ratio) are consistently top-10
4. **Interaction features** (waist×age, age×BMI, age×systolic) contribute substantially
5. **State (geographic location)** is an important predictor, capturing regional variation in diabetes prevalence
6. **Cross-algorithm stacking** provides marginal improvement (+0.001) but is not practically significant
7. **The ~0.80 ceiling** reflects the information content of non-invasive NFHS-5 variables; family history and lipid panel data are not available
