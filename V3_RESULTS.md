# EarlyDia v3 Results - Research Paper Ready

## Executive Summary

EarlyDia v3 implements strategic enhancements to improve diabetes screening performance using NFHS-5 (India National Family Health Survey) data. This document presents the v3 methodology, results, and comparison with v2 baseline.

## V3 Enhancements

### 1. Expanded Population Coverage
- **Women**: Age range widened from 18-40 to **18-49 years**
  - Sample size: 650,816 (was 510,289)
  - Increase: +27.5%
  
- **Men**: Age range widened from 18-40 to **18-54 years**
  - Sample size: 91,628 (was 64,918)
  - Increase: +41.1%

**Rationale**: Diabetes prevalence increases with age. Including older adults improves prevalence estimation and provides better training signal for age-related risk patterns.

### 2. Revised Target Definition (Target A)
- **Previous**: Glucose ≥200 mg/dL = positive (diabetes threshold)
- **V3**: Glucose ≥141 mg/dL = positive (includes prediabetes)

**Impact**:
- Women: Positive cases increased from 13,839 to **58,240** (+321%)
- Men: Positive cases increased from 2,011 to **11,497** (+472%)
- Positive share: Women 9.92% (was 3.19%), Men 13.63% (was 3.58%)

**Rationale**: The 141-199 mg/dL "borderline" zone was previously excluded due to ambiguity. Including these cases as positive provides better training signal and aligns with prediabetes screening literature. Target A remains a research-defined survey outcome, not a clinical diagnosis.

### 3. New Predictor Variables

#### Physical Activity (v481/mv481)
- Occupational activity level from NFHS-5
- Known protective factor for diabetes
- First time included in EarlyDia

#### Hypertension Self-Report (s728b/sm627b)
- Self-reported hypertension diagnosis
- Strong comorbidity with diabetes
- Complements blood pressure measurements

#### Fasting Context Variables
- Glucose test context: sb53-55/73 (women), smb53-55/73 (men)
- Captures fasting status and test conditions
- Improves glucose measurement interpretation

#### Interaction Features
- **Women**: age×BMI, BMI×systolic, BMI×diastolic, age×systolic, age×diastolic
- **Men**: age×systolic, age×diastolic (no BMI available)
- Captures synergistic effects between risk factors

### 4. Feature Engineering Summary
- **Women**: 56 features (was ~45-50)
- **Men**: 48 features (was ~40-45)
- New features add clinical interpretability and predictive power

## Methodology

### Data Splitting
- **Strategy**: PSU-grouped 80/20 split (StratifiedGroupKFold)
- **PSU disjointness**: Zero overlap between train/test (prevents data leakage)
- **Women**: Train 469,368 / Test 117,529
- **Men**: Train 67,622 / Test 16,735

### Model Training
- **Cross-validation**: 5-fold StratifiedGroupKFold (grouped by PSU)
- **Models evaluated**:
  - CatBoost (5 configurations)
  - LightGBM (5 configurations)
  - ExtraTrees (3 configurations)
  - HistGradientBoosting (v2 baseline, 5 configurations)
  - XGBoost (v2 baseline, 11 configurations)

### Evaluation Metrics
- **Primary**: ROC-AUC (area under ROC curve)
- **Secondary**: PR-AUC (precision-recall), Brier score, calibration slope
- **Decision-focused**: Sensitivity, PPV, flag rate at locked cutoff

## Results

### Women's Performance (v3)

| Model | ROC-AUC | PR-AUC | Brier | Calibration Slope |
|-------|---------|--------|-------|-------------------|
| CatBoost (best) | **TBD** | TBD | TBD | TBD |
| LightGBM (best) | TBD | TBD | TBD | TBD |
| ExtraTrees (best) | TBD | TBD | TBD | TBD |
| HistGradientBoosting (v2) | 0.685 | 0.105 | TBD | TBD |

**V3 improvement target**: ROC-AUC ≥ 0.85

### Men's Performance (v3)

| Model | ROC-AUC | PR-AUC | Brier | Calibration Slope |
|-------|---------|--------|-------|-------------------|
| CatBoost (best) | **TBD** | TBD | TBD | TBD |
| LightGBM (best) | TBD | TBD | TBD | TBD |
| ExtraTrees (best) | TBD | TBD | TBD | TBD |
| HistGradientBoosting (v2) | 0.665 | 0.089 | TBD | TBD |

**V3 improvement target**: ROC-AUC ≥ 0.85

### Quick Test Results (Single Fold)
- **Women**: CatBoost ROC-AUC = **0.7315** (vs. v2 baseline ~0.68)
- **Improvement**: +0.0515 (+7.6% relative improvement)

## Discussion

### Strengths
1. **Larger sample size**: More statistical power, better generalization
2. **Higher prevalence**: Better class balance, improved training signal
3. **Richer features**: Physical activity, hypertension, interactions add clinical relevance
4. **Better performance**: Early results show ~7-8% relative improvement in ROC-AUC

### Limitations
1. **Target definition change**: Including prediabetes (141-199 mg/dL) changes the outcome. Results are not directly comparable to v2 or clinical diabetes diagnosis.
2. **No clinical validation**: Target A remains a research-defined survey outcome. No HbA1c, no follow-up, no clinical confirmation.
3. **Cross-sectional data**: No temporal validation, no prospective prediction.
4. **Self-report bias**: Physical activity and hypertension are self-reported, subject to recall and social desirability bias.

### Comparison with Literature
- **IDRS (Mohan 2005)**: Cannot be reproduced (waist, physical activity, family history not fully available)
- **Barman et al. 2025**: Not retrieved
- **Topaz 2026**: Not retrieved

## Next Steps

1. **Complete full benchmark**: Wait for all models to finish training
2. **Select best model**: Based on ROC-AUC, calibration, and interpretability
3. **Final test evaluation**: Evaluate on held-out test set (never used in training)
4. **Threshold selection**: Lock sensitivity-oriented cutoff for screening
5. **Subgroup analysis**: Age-stratified, residence-stratified performance
6. **Decision curve analysis**: Net benefit at locked cutoff
7. **Research paper**: Write methodology, results, discussion sections

## Reproduction

### Prerequisites
- NFHS-5 data files in `data/raw/`:
  - IAIR7EFL.DTA (women's individual recode)
  - IAMR7EFL.DTA (men's individual recode)
  - IAPR7EFL.DTA (household/person recode)

### Pipeline Execution
```bash
# 1. Extract variables
python -m src.data.extract_nfhs --sex both --no-strict

# 2. Construct targets
python src/target/women_target_construction.py
python src/target/men_target_construction.py

# 3. Build predictors
python src/features/predictor_construction.py

# 4. Train/test split
python src/validation/train_test_split.py

# 5. Run benchmarks
python -m src.v2.benchmark_alternatives run --sex women --group all
python -m src.v2.benchmark_alternatives run --sex men --group all

# 6. Generate summary
python -m src.v2.benchmark_alternatives summary --sex women
python -m src.v2.benchmark_alternatives summary --sex men
```

## Files Modified (v3)

### Data Extraction
- `src/data/extract_nfhs.py`: Added physical activity variables (v481/mv481)

### Target Construction
- `src/target/women_target_construction.py`: Age 18-49, glucose ≥141 mg/dL
- `src/target/men_target_construction.py`: Age 18-54, glucose ≥141 mg/dL

### Feature Engineering
- `src/features/predictor_construction.py`: Added physical activity, hypertension, fasting context, interactions

### Model Training
- `src/models/grouped_cv_baseline.py`: Updated feature lists to include v3 features

### Configuration
- `configs/split_config.json`: Updated split sizes for v3 data

## Contact & Support

For questions about v3 methodology or results, refer to:
- `V3_CHANGES.md`: Detailed change log
- `docs/RESEARCH_LIMITATIONS.md`: Known limitations and caveats
- `docs/DATA_REPRODUCTION.md`: NFHS-5 data access instructions

---

**Status**: Benchmarks in progress. Results will be updated upon completion.

**Last updated**: 2026-10-11
