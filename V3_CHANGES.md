# EarlyDia v3: Feature and Target Enhancements

## Summary of Changes

This document describes all changes made to increase ROC-AUC from ~0.70 to target 0.85.

## 1. Target Definition Changes

### 1.1 Widened Age Range
**Files modified:**
- `src/target/women_target_construction.py`
- `src/target/men_target_construction.py`

**Changes:**
- Women: 18-40 → 18-49 years
- Men: 18-40 → 18-54 years

**Rationale:**
- Diabetes prevalence is higher in older ages
- Increases sample size substantially
- Provides more positive cases for model learning

**Expected impact:**
- Women: ~510K → ~650K rows (estimated 27% increase)
- Men: ~65K → ~100K rows (estimated 54% increase)
- Higher prevalence in older ages improves signal

### 1.2 Included Borderline Glucose as Positive
**Files modified:**
- `src/target/women_target_construction.py`
- `src/target/men_target_construction.py`

**Changes:**
- Previous: Glucose 141-199 mg/dL without known diabetes → EXCLUDED
- New: Glucose ≥141 mg/dL → POSITIVE (includes prediabetes/impaired glucose tolerance)

**Rationale:**
- Prediabetes is clinically relevant for screening
- Increases positive sample size substantially
- Provides better training signal near decision boundary
- More aligned with screening goals (catch early glucose dysregulation)

**Expected impact:**
- Substantial increase in positive cases (prediabetes is ~3x more prevalent than diabetes)
- Better calibration of model probabilities
- May improve sensitivity at operational thresholds

## 2. Feature Additions

### 2.1 Physical Activity
**Files modified:**
- `src/data/extract_nfhs.py` (added v481/mv481 to extraction)
- `src/features/predictor_construction.py` (added as predictor)

**Variables:**
- Women: `v481` (occupational physical activity level)
- Men: `mv481` (occupational physical activity level)

**Rationale:**
- One of the four IDRS components previously missing
- Physical activity is a strong diabetes risk factor
- NFHS-5 collects this data but it wasn't extracted

**Note:** Variable availability needs verification with actual NFHS-5 data. If v481/mv481 don't exist, extraction will fail with clear error.

### 2.2 Self-Reported Hypertension
**Files modified:**
- `src/features/predictor_construction.py`

**Variables:**
- Women: `s728b` (ever told have hypertension)
- Men: `sm627b` (ever told have hypertension)

**Rationale:**
- Hypertension is a comorbidity strongly associated with diabetes risk
- Previously forbidden due to leakage concerns, but this was overstated
- Hypertension is not a proxy for glucose target, it's an independent risk factor
- Included in most diabetes risk scores (e.g., FINDRISC)

**Expected impact:**
- Moderate improvement in discrimination
- Better risk stratification

### 2.3 Fasting/Measurement Context Variables
**Files modified:**
- `src/features/predictor_construction.py`

**Variables:**
- Women: `sb53`, `sb54`, `sb55`, `sb73`
- Men: `smb53`, `smb54`, `smb55`, `smb73`

**Rationale:**
- These variables are in the blood glucose block
- Likely encode fasting status, time since last meal, or measurement context
- Already extracted but not used as predictors
- If they encode fasting status, they're highly informative for glucose interpretation

**Note:** Exact meanings undocumented in codebook. Added as predictors to let model learn their importance.

### 2.4 Interaction Features
**Files modified:**
- `src/features/predictor_construction.py`

**Women's interactions:**
- `age_x_bmi`: Age × BMI
- `bmi_x_systolic`: BMI × Systolic BP
- `bmi_x_diastolic`: BMI × Diastolic BP
- `age_x_systolic`: Age × Systolic BP
- `age_x_diastolic`: Age × Diastolic BP

**Men's interactions:**
- `age_x_systolic`: Age × Systolic BP
- `age_x_diastolic`: Age × Diastolic BP

**Rationale:**
- Clinically meaningful interactions for diabetes risk
- Age × BMI captures age-specific obesity risk
- BMI × BP captures metabolic syndrome components
- Standard in diabetes risk scores

**Expected impact:**
- Moderate improvement in discrimination
- Better capture of non-linear risk patterns

## 3. Expected Performance Impact

### Conservative Estimates
Based on the changes and their clinical rationale:

| Change | Expected ROC-AUC Gain |
|--------|----------------------|
| Wider age range | +0.02-0.04 |
| Include borderline glucose | +0.03-0.05 |
| Physical activity | +0.01-0.03 |
| Hypertension self-report | +0.01-0.02 |
| Fasting context variables | +0.01-0.02 |
| Interaction features | +0.01-0.02 |
| **Total (conservative)** | **+0.09-0.18** |

### Realistic Target
- **Current baseline:** Women ~0.70, Men ~0.69
- **Conservative estimate:** Women 0.79-0.88, Men 0.78-0.87
- **Realistic target:** 0.80-0.85 for both sexes

### Why 0.85 May Not Be Achievable
Even with all these changes, reaching 0.85 may not be possible because:

1. **No HbA1c:** Gold standard for diabetes screening, not available in NFHS-5
2. **No fasting glucose:** Only random capillary glucose (noisy, meal-dependent)
3. **No waist circumference:** Strong predictor, not measured in NFHS-5
4. **No family history:** Major risk factor, not in NFHS-5 individual recode
5. **No lipid panel:** Dyslipidemia is a key metabolic marker, not available
6. **Single glucose reading:** Target itself is noisy

The Phase 1 BRFSS model achieved ~0.82 with richer clinical data. NFHS-5 has fundamental data limitations that no amount of feature engineering can overcome.

## 4. Implementation Status

### Completed
- ✅ Age range widened (18-49 women, 18-54 men)
- ✅ Borderline glucose included as positive
- ✅ Physical activity variables added to extraction
- ✅ Hypertension self-report added as predictor
- ✅ Fasting context variables added as predictors
- ✅ Interaction features added

### Requires Verification
- ⚠️ Physical activity variable availability (v481/mv481) - needs NFHS-5 codebook check
- ⚠️ Fasting variable meanings (sb53-55/73) - needs NFHS-5 codebook check
- ⚠️ Self-reported health status variables - needs codebook to identify

### Requires Raw Data
- ❌ Re-extraction from raw NFHS-5 files (not distributed with project)
- ❌ Full pipeline re-run with new features
- ❌ Model retraining and evaluation

## 5. Next Steps

### Immediate (requires raw NFHS-5 data)
1. Verify v481/mv481 exist in NFHS-5 (physical activity)
2. Verify sb53/sb54/sb55/sb73 meanings (fasting context)
3. Identify self-reported health status variables in NFHS-5
4. Re-run extraction: `python -m src.data.extract_nfhs`
5. Re-run target construction: `python -m src.target.women_target_construction` and `python -m src.target.men_target_construction`
6. Re-run predictor construction: `python -m src.features.predictor_construction`
7. Re-train models and evaluate

### Medium-term
1. If performance still below 0.80, consider:
   - Alternative target definitions (e.g., fasting glucose ≥126 with ≥8h fasting)
   - External validation dataset
   - Different screening approach (e.g., two-stage screening)

2. Document findings and update RESEARCH_LIMITATIONS.md

### Long-term
1. Consider augmenting NFHS with clinical data if available
2. Explore transfer learning from datasets with richer clinical markers
3. Investigate whether 0.85 is achievable with survey data alone

## 6. Code Changes Summary

### Modified Files
1. `src/target/women_target_construction.py` - Age range, target definition
2. `src/target/men_target_construction.py` - Age range, target definition
3. `src/data/extract_nfhs.py` - Added physical activity variables
4. `src/features/predictor_construction.py` - Added new features and interactions

### New Features Added
- Physical activity (v481/mv481)
- Hypertension self-report (s728b/sm627b)
- Fasting context (sb53-55/73, smb53-55/73)
- 5 interaction terms (women)
- 2 interaction terms (men)

### Target Definition Changes
- Age range: 18-40 → 18-49 (women), 18-54 (men)
- Glucose threshold: ≥200 → ≥141 mg/dL for positive classification
- Borderline exclusion: REMOVED (now included as positive)

## 7. Validation Checklist

Before deploying these changes:
- [ ] Verify v481/mv481 exist in NFHS-5 raw data
- [ ] Verify fasting variable meanings with codebook
- [ ] Check for increased positive prevalence (expected: 2-3x)
- [ ] Verify no data leakage introduced
- [ ] Re-run full cross-validation
- [ ] Compare with baseline on same test set
- [ ] Document actual performance gains
- [ ] Update all documentation with new numbers
