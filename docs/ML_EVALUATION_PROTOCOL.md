# EarlyDia ML Evaluation Protocol

## 1. Data Split Strategy

### PSU-Aware Group Splitting
The NFHS-5 survey data uses a multi-stage cluster sampling design. The Primary Sampling Unit (PSU)
represents the first-stage cluster. To prevent information leakage between individuals sampled from
the same geographic unit, all splits are PSU-aware:

- **Train/Test Split**: `GroupShuffleSplit` with `group=PSU`, `test_size=0.20`, `seed=42`.
- **Cross-Validation**: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped by PSU.
- **Assertion**: No PSU appears in both training and validation within any fold.

### Split Sizes (from `configs/split_config.json`)
| Sex   | Train Rows | Test Rows | Train PSUs | Test PSUs | Prevalence |
|-------|-----------|-----------|-----------|-----------|------------|
| Women | 347,833   | 86,669    | 24,076    | 6,020     | 3.19%      |
| Men   | 44,684    | 11,408    | 7,200     | 1,800     | 3.61%      |

## 2. Cross-Validation Protocol

All model development uses 5-fold PSU-grouped stratified CV on the **training split only**.
The held-out test split is never used for:
- Feature selection
- Hyperparameter tuning
- Model selection
- Calibration fitting
- Threshold selection

### Safeguards
- `src/v2/data_v2.py::load_dev()` returns training rows only.
- `src/v2/final_v2.py` gates test access behind `--confirm-test-access` and writes a lock file.
- The lock file prevents re-running final evaluation with a different candidate.

## 3. Calibration Protocol

### Cross-Fitted Calibration (v2)
The calibrator that scores fold k's OOF predictions is fit on the other folds' OOF predictions only.
This prevents optimistic calibration estimates.

```
for each fold k:
    calibrator_k = fit(OOF predictions from folds != k)
    calibrated_predictions[fold k] = calibrator_k.predict(OOF predictions from fold k)
```

### Calibration Method Selection
Sigmoid (Platt) and isotonic calibration are both evaluated. The method with the lower Brier score
on cross-fitted predictions is selected.

### Production Calibrator
The final production calibrator is fit on ALL OOF predictions (not on the test set).

## 4. Threshold Selection Protocol

### Rule
The **highest** threshold on a 0.001-spaced grid (0.001 to 0.300) whose sensitivity on calibrated
grouped-OOF predictions is still >= 0.75.

### Rationale
This is a documented research choice targeting a screening-level sensitivity, not a clinically
established requirement. The 0.75 floor is arbitrary but defensible for a screening instrument.

### Current Thresholds
| Sex   | v1 Threshold | v2 Threshold |
|-------|-------------|-------------|
| Women | 0.026       | 0.024       |
| Men   | 0.027       | 0.029       |

## 5. Model Comparison Framework

All candidates are compared using the same:
- Development folds (5-fold PSU-grouped CV, seed=42)
- Target definition (`target_a`)
- Eligible population (same inclusion/exclusion criteria)
- Permitted features (same feature schema per sex)
- Evaluation methodology (OOF predictions, cross-fitted calibration)

### Metrics Reported
- ROC-AUC (with fold standard deviation)
- PR-AUC / Average Precision (prevalence noted as baseline)
- Brier score (calibrated)
- Calibration slope and intercept
- ECE (Expected Calibration Error)
- Sensitivity, specificity, PPV, NPV at selected threshold
- Flag rate at selected threshold
- Flag rate at 75% sensitivity (threshold-free ranking metric)
- Top-5%, top-10%, top-20% lift and capture
- Age-band performance (18-24, 25-29, 30-34, 35-40)
- PSU-aware bootstrap confidence intervals (paired, 500 resamples)

### Interpretation Guidelines
- Small metric differences (e.g., <0.005 ROC-AUC) are not considered meaningful without
  confidence intervals or paired comparisons.
- PR-AUC is the more informative metric given the rare positive class (~3.5%).
- Accuracy is never used as a standalone metric on this imbalanced dataset.

## 6. XGBoost Evaluation Additions

XGBoost candidates use the `state` feature flag (one-hot encoded) for comparability with
the existing state-based candidates (HGB and LR).

### XGBoost-Specific Considerations
- `scale_pos_weight` approximates inverse prevalence weighting.
- No early stopping is used (sklearn's internal validation split would be row-random, not PSU-grouped).
- Hyperparameters are selected from a constrained search space documented in
  `src/models/xgboost_pipeline.py::XGB_SEARCH_SPACE`.
- XGBoost's native missing-value support is available but does not resolve
  missing-not-at-random bias.

## 7. Test Set Protocol

### Access Control
- `reports/v2/TEST_ACCESSED_{sex}.json` records which candidate was evaluated on the test set.
- A second run with a different candidate is rejected.
- The v1 locked test report is reproduced as a sanity check before any v2 evaluation.

### Final Evaluation
Only the selected candidate undergoes final evaluation on the held-out test split.
The final report includes:
- v1 (locked baseline) vs v2 comparison
- Global threshold and age-band threshold results
- Paired PSU-bootstrap confidence intervals
- Age-band breakdown
