# EarlyDia

A research/prototype **screening classification** system built on NFHS-5 (India, 2019-21)
survey data, estimating a research-defined **"NFHS-5 diabetes-range status"** separately for
women and men aged 18-40.

**EarlyDia is not a diabetes diagnosis, not a prediction of future diabetes, and not
clinically validated.** It is an applied machine-learning portfolio project demonstrating a
leakage-aware pipeline: cluster-aware validation, grouped out-of-fold calibration, and an
explicitly documented, sensitivity-oriented screening threshold.

---

## Production application: React + FastAPI (EarlyDia v2)

> **React + FastAPI is the production screening application. Streamlit is retained only as legacy reference code.**

```
React (Vite)  ->  FastAPI  ->  Python inference (v2 default)  ->  production models (v2)
frontend/         backend/api   src/inference/predict_module_v2.py    models_v2/*.pkl
                                src/inference/predict_module.py       models/*.pkl (legacy v1)
```

- React contains **no** prediction logic - it collects answers and displays what the API returns.
- FastAPI validates answers, maps them to the feature contract, and calls `EarlyDiaPredictorV2` by default.
- Legacy v1 models (`models/*.pkl`, women threshold 0.026, men 0.027) are retained for backward compatibility via `model_version="v1"`.
- Exact Windows commands are in [Running locally](#running-locally). A 3-minute presentation walkthrough is in [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md).
- The old Streamlit prototype (`src/app/`) is kept only as legacy reference code and is not used by the final app.

---

## Table of contents
1. [Overview](#overview)
2. [Problem and motivation](#problem-and-motivation)
3. [Dataset and population](#dataset-and-population)
4. [Target definition](#target-definition)
5. [Features](#features)
6. [Methodology](#methodology)
7. [Production models (v2) & legacy v1](#production-models-v2--legacy-v1)
8. [Held-out test results (v1 vs v2)](#held-out-test-results-v1-vs-v2)
9. [Age subgroup limitation](#age-subgroup-limitation)
10. [Explainability](#explainability)
11. [Limitations](#limitations)
12. [Responsible use](#responsible-use)
13. [Reproducibility](#reproducibility)
14. [How to demo](#how-to-demo)
15. [Project structure](#project-structure)
16. [Running locally](#running-locally)

---

## Overview

EarlyDia estimates whether a person's survey-reported profile resembles the
**research-defined NFHS-5 diabetes-range group** used to build this project, and outputs a
**screening classification** (flagged / not flagged) with a calibrated probability. It does
**not** diagnose diabetes, does **not** predict future diabetes, and has **not** been
clinically validated. "Early detection" is deliberately avoided as a description - the target
is a research construct built from one survey wave, not a clinical outcome.

## Problem and motivation

Diabetes screening in large population surveys is attractive because the data already
exists at scale, but survey data creates real modelling challenges this project is built
around rather than around: a single **random (non-fasting) glucose reading** stands in for a
clinical diagnostic protocol, self-report of diagnosis is imperfect, the survey design is
**clustered** (people within the same village/PSU are more similar to each other than to
strangers), and the outcome is **rare** (~3.0-3.6% prevalence), which breaks naive accuracy
as a metric. EarlyDia's design choices - grouped splitting, grouped OOF calibration, PR-AUC
alongside ROC-AUC, and a sensitivity-oriented threshold - all exist to handle these
challenges honestly rather than let them silently inflate reported performance.

## Dataset and population

- **Source**: NFHS-5 (National Family Health Survey, India, 2019-21), Individual Recode
  (`IAIR7EFL.DTA`, women 15-49) and Men's Recode (`IAMR7EFL.DTA`, men 15-54).
- **Raw microdata is not included in this repository** - see
  [`docs/DATA_REPRODUCTION.md`](docs/DATA_REPRODUCTION.md) for how to obtain it.
- **Population**: age 18-40 inclusive, modelled **separately for women and men**.
- **Pregnancy exclusion (women only)**: `v213 == 1` (currently pregnant) removed. This
  removed **exactly 27,826** of 510,289 age-eligible women.
  `mv213` (the *partner's* pregnancy status) is explicitly **not** used to exclude men.

| | Age 18-40 | After exclusions | Target A usable |
|---|---|---|---|
| Women | 510,289 | 482,463 (pregnancy excluded) | 434,502 |
| Men | 64,918 | 64,918 (no pregnancy exclusion applies) | 56,092 |

## Target definition

> [!IMPORTANT]
> **Implementation Note on Target Definition & Evaluation Metrics:**
> The codebase (`src/target/women_target_construction.py` and `src/target/men_target_construction.py`) currently uses a **redefined target**:
> - **Positive:** Known diabetes ($K$) OR valid glucose &ge; 141 mg/dL
> - **Negative:** All three flags explicitly "No" AND valid glucose &le; 100 mg/dL
> - **Excluded:** Glucose 101&ndash;140 mg/dL (ambiguous intermediate zone)
>
> This differs from the original &ge; 200 / &le; 140 definition described below.
> Furthermore, the original-target held-out v1 test results (women ROC-AUC 0.685, PPV 4.4%; men ROC-AUC 0.665, PPV 4.5%) are **not comparable** to the ~0.80 cross-validation figures reported for v2, and the v2 production models (LightGBM women, CatBoost men) have **no separate held-out evaluation yet** (their ~0.80 figures reflect 5-fold PSU-grouped out-of-fold cross-validation on development data under the redefined target).

**Target A** (locked, research-defined - not a clinical diagnosis):

```
Known-diabetes flag:
  K = (sb56 == 1) OR (sb57 == 1) OR (s728a == 1)          [women]
  K = (smb56 == 1) OR (smb57 == 1) OR (sm627a == 1)       [men]
    sb56/smb56  = told high blood glucose 2+ times by a doctor
    sb57/smb57  = currently taking medicine to lower blood glucose
    s728a/sm627a = self-reports currently having diabetes

POSITIVE = K  OR  (valid glucose reading AND glucose >= 200 mg/dL)
NEGATIVE = all three flags explicitly "No" (== 0)  AND  valid glucose AND glucose <= 140 mg/dL

EXCLUDED (label not assigned):
  - no valid glucose reading AND no known-diabetes flag
  - valid glucose in 141-199 mg/dL AND no known-diabetes flag        (ambiguous zone)
  - any of the three flags is blank / "Don't know" rather than an explicit "No"
  - (women only) currently pregnant
```
"Valid glucose" excludes DHS special codes (Refused / Other / Not tested); a reading of
499 ("499 mg/dL+") is treated as valid and counts toward the >=200 rule.

**Validated counts** (women, age 18-40, before pregnancy exclusion): POS = 14,561,
NEG = 445,500, usable = 460,061. **After** pregnancy exclusion: POS = **13,839**,
NEG = **420,663**, usable = **434,502**, positive share = **3.19%**. Men: POS = 2,026,
NEG = 54,066, usable = 56,092, positive share = 3.61%. Every one of these numbers was
reproduced exactly from the raw variables during development (see
`configs/reproducibility_manifest.json` and `reports/`).

## Features

Predictors are grouped as:

| Group | Women | Men |
|---|---|---|
| Age | ✅ | ✅ |
| BMI | ✅ (`v445`) | ❌ **unavailable in NFHS-5 men's questionnaire** - not imputed, not used |
| Average valid systolic/diastolic BP + valid-reading counts | ✅ (up to 3 readings, physiologically-implausible/special-coded readings excluded from the average) | ✅ |
| Tobacco | 12 compact items | 8 compact items (3 women-only sub-items have no men's equivalent; a 4th men's item, cigarette frequency, is not administered in this survey and was dropped after being found 100% missing) |
| Alcohol | drinks / frequency / duration / 6 type flags | same structure |
| Diet | 9 frequency items | 9 frequency items |
| Education | ✅ | ✅ |
| Residence (urban/rural) | ✅ | ✅ |

**Explicitly excluded from every model**: waist/hip measurements, men's BMI, wealth
quintile (retained only as contextual metadata, never passed to a model), self-reported
hypertension, physical activity, and family history (none of the latter two are available
in this NFHS-5 extract). A leakage assertion (re-run as a test on every commit) confirms
none of the four Target A source variables (`sb56/sb57/s728a/sb74` or their male
equivalents) ever appears among the predictors.

### Optional IDRS comparator (Step 3)

Step 3 can optionally collect **waist circumference, physical activity and family history** to display the published
**Indian Diabetes Risk Score** (Mohan et al., 2005) beside the EarlyDia result. These answers are **not** model features -
the models, Target A and thresholds are unchanged, and `tests/test_idrs.py` verifies the ML output is identical with or without them.
Details and citation: `docs/FEATURE_RESEARCH_EVIDENCE.md`; code: `backend/api/idrs.py`.

## Methodology

1. **Cluster-aware split**: `GroupShuffleSplit` grouped by PSU (`v021`/`mv021`), ~80/20,
   with zero PSU overlap between train and test (asserted programmatically). Seed search
   (deterministic, from 42 upward) for reasonable prevalence balance - seed **42** satisfied
   it on the first try for both sexes.
2. **Grouped cross-validation**: `StratifiedGroupKFold(n_splits=5, shuffle=True,
   random_state=42)` on the training portion only, PSU-disjointness asserted on every fold.
3. **Model comparison** (grouped CV, training data only):

   | Model | Women ROC-AUC | Women PR-AUC | Men ROC-AUC | Men PR-AUC |
   |---|---|---|---|---|
   | Logistic Regression | 0.6788 ± 0.0039 | 0.0761 ± 0.0014 | 0.6792 ± 0.0185 | 0.0777 ± 0.0040 |
   | Random Forest | 0.6433 ± 0.0077 | 0.0683 ± 0.0015 | 0.6361 ± 0.0140 | 0.0667 ± 0.0074 |
   | **HistGradientBoosting** | **0.6818 ± 0.0028** | **0.0930 ± 0.0017** | 0.6681 ± 0.0176 | **0.0902 ± 0.0150** |

   *Note: these fold-mean values differ slightly from the OOF values in `configs/model_selection.json` (e.g. women HGB 0.6816 / 0.0924). The reason (aggregation method) has not been verified without the NFHS data; see `docs/RESEARCH_LIMITATIONS.md`. Final test results below are unaffected.*

   Random Forest was retained for documentation but dropped from further stages - weakest
   ROC-AUC *and* PR-AUC for both sexes. **PR-AUC is emphasised alongside ROC-AUC** because
   with only ~3-4% positive prevalence, ROC-AUC can look deceptively strong while precision
   at any usable threshold stays low; PR-AUC is more sensitive to that reality.
4. **Class weighting**: `class_weight="balanced"` for all models where supported.
5. **Grouped out-of-fold (OOF) calibration**: OOF probabilities generated with the same
   `StratifiedGroupKFold` split (never optimism-biased). Sigmoid (Platt) vs. isotonic
   calibration compared via a *nested* grouped 5-fold evaluation on the OOF predictions only
   - **isotonic won for women, sigmoid won for men** (both close). Post-calibration Brier
   scores: women 0.0304, men 0.0349; calibration slope/intercept on OOF: both sexes ≈1.00 /
   ≈0.00.
6. **Threshold selection**: on calibrated OOF training predictions only (never the test set),
   scanned a 0.001-step grid and took the **highest threshold whose sensitivity is still
   ≥ 75%** - an explicit, documented screening-threshold *choice* prioritising catching more
   possible cases over precision, not a claim of optimality.
7. **Final refit**: selected model refit on the **full** training set; the calibrator stays
   the one fit on OOF predictions (never refit on the same data it will calibrate).
8. **Final test evaluation**: performed **exactly once**, after every choice above was locked
   using training/OOF data only. PSU-disjointness between train and test was re-verified
   immediately before evaluation.

## Production models (v2) & legacy v1

| | Women (v2 Production) | Men (v2 Production) | Women (v1 Legacy) | Men (v1 Legacy) |
|---|---|---|---|---|
| **Model** | LightGBM (`lgbm_D`, `is_unbalance=True`) | CatBoost (`cb_B`, `auto_class_weights="Balanced"`) | HistGradientBoosting (`hgb_default`, `class_weight="balanced"`) | HistGradientBoosting (`hgb_default`, `class_weight="balanced"`) |
| **State Feature** | One-hot categorical (`state`) | One-hot categorical (`state`) | None | None |
| **Calibration** | Sigmoid (Platt) | Sigmoid (Platt) | Isotonic | Sigmoid (Platt) |
| **Screening Threshold** | **0.212** (21.2%) | **~0.30 (artifact value 0.299)** | **0.026** (legacy) | **0.027** (legacy) |
| **Threshold Rule** | Highest 0.001-grid threshold with dev CV sensitivity ≥ 75% | Same rule | Highest 0.001-grid threshold with OOF sensitivity ≥ 75% | Same rule |
| **Artifact** | `models_v2/women_v2_pipeline.pkl` | `models_v2/men_v2_pipeline.pkl` | `models/women_production_pipeline.pkl` | `models/men_production_pipeline.pkl` |

**Calibration & Risk Interpretation**:
All probabilities are calibrated to reflect empirical prevalence of the research-defined target (**NFHS-5 diabetes-range status**) in the survey population, **not real-world individual clinical risk or lifetime incidence**. Calibrated probabilities serve purely as a screening / risk stratification metric across survey-reported profiles.

**Model Selection Context**:
- **Women (v2)**: LightGBM candidate `lgbm_D` was selected as the approved v2 model under the redefined target (development CV ROC-AUC 0.8016, PR-AUC 0.5829), top-tier among the benchmarked candidates; differences between the best candidates (~0.005 ROC-AUC) are within fold-to-fold variation.
- **Men (v2)**: CatBoost candidate `cb_B` was selected as the approved v2 model under the redefined target (development CV ROC-AUC 0.8033, PR-AUC 0.6678), top-tier among the benchmarked candidates; differences between the best candidates (~0.005 ROC-AUC) are within fold-to-fold variation.
- **Legacy v1**: The initial v1 models (preserved in `models/*.pkl`) did not include State as a predictor. They remain accessible for backward compatibility.

## Held-out test results (v1 vs v2)

> [!NOTE]
> The held-out test comparisons below evaluate v1 baselines against the intermediate v2 candidates (`state_native_B` and `lr_state`) under the original target definition.
> The approved v2 production models (LightGBM `lgbm_D` for women and CatBoost `cb_B` for men) operate under the redefined target and were evaluated via 5-fold development cross-validation (~0.80 ROC-AUC); separate locked held-out test evaluation has not yet been conducted for them.

Evaluated on the untouched cluster-held-out test split (PSU-grouped 80/20 split; zero cluster overlap between train and test). Both v1 and v2 pipelines were evaluated on the **exact same held-out test cohort**:
- **Women held-out test set**: n = 86,669 (positive cases = 2,630; survey positive rate = 3.03%)
- **Men held-out test set**: n = 11,408 (positive cases = 377; survey positive rate = 3.30%)

### Empirical Test Metrics (Held-Out Evaluation)

| Metric | Women v1 (cutoff 0.026) | Women v2 (cutoff 0.024) | Men v1 (cutoff 0.027) | Men v2 (cutoff 0.029) |
|---|---|---|---|---|
| **ROC-AUC** | 0.6858 | **0.7098** | 0.6650 | **0.6706** |
| **PR-AUC** | 0.0882 | **0.0930** | 0.0719 | **0.0629** |
| **Sensitivity** | 76.20% (2,004 / 2,630) | **75.86%** (1,995 / 2,630) | 75.60% (285 / 377) | **74.54%** (281 / 377) |
| **Specificity** | 47.97% (40,312 / 84,039) | **52.22%** (43,882 / 84,039) | 45.40% (5,008 / 11,031) | **50.67%** (5,589 / 11,031) |
| **PPV / Precision** | 4.38% | **4.73%** | 4.52% | **4.91%** |
| **NPV** | 98.47% | **98.57%** | 98.20% | **98.31%** |
| **Flag Rate (% flagged)** | 52.77% (45,731 / 86,669) | **48.64%** (42,152 / 86,669) | 55.29% (6,308 / 11,408) | **50.17%** (5,723 / 11,408) |
| **Brier Score** | 0.0287 | **0.0286** | 0.0315 | **0.0316** |
| **Calibration Slope / Intercept** | 1.023 / 0.012 | **1.034 / 0.044** | 0.952 / -0.276 | **0.871 / -0.498** |
| **True Positives (TP)** | 2,004 | **1,995** | 285 | **281** |
| **False Positives (FP)** | 43,727 | **40,157** | 6,023 | **5,442** |
| **True Negatives (TN)** | 40,312 | **43,882** | 5,008 | **5,589** |
| **False Negatives (FN)** | 626 | **635** | 92 | **96** |

*(Source: `reports/v2/final_women_state_native_B.json` and `reports/v2/final_men_lr_state.json`. All metrics computed on held-out test data).*

### Statistical Comparison & Interpretation

- **Women (v2 vs v1)**:
  - **ROC-AUC improved**: ΔAUC = **+0.0241** (paired PSU cluster bootstrap 95% CI: `[+0.0167, +0.0310]`, n=500 bootstraps).
  - **Flag rate improved**: Flagged proportion dropped from 52.77% to 48.64% (a reduction of 3,570 false positives).
  - **Sensitivity was flat**: 75.86% vs 76.20% (within 0.34 percentage points; sensitivity did **not** improve, but was maintained while reducing false alarms).
- **Men (v2 vs v1)**:
  - **Discrimination unchanged**: ΔAUC = **+0.0053** (paired PSU cluster bootstrap 95% CI: `[-0.0131, +0.0230]`, n=500 bootstraps). The 95% confidence interval **crosses zero**, meaning there is no statistically significant improvement in discrimination.
  - **Operating point**: v2 operates at a more conservative screening cutoff (0.029 vs 0.027), reducing the flag rate from 55.29% to 50.17% (581 fewer false positives), rather than demonstrating a substantial discrimination gain.

## Age subgroup limitation

**This is not hidden.** Applying one fixed threshold uniformly across 18-40 produces very
uneven operating points by age:

| Subgroup | Women n / prevalence | Women sens / spec / PPV / ROC-AUC | Men n / prevalence | Men sens / spec / PPV / ROC-AUC |
|---|---|---|---|---|
| 18-29 | 48,133 / 1.93% | 46.1% / 72.8% / 3.2% / 0.621 | 6,390 / 2.00% | 39.1% / 73.3% / 2.9% / 0.575 |
| 30-40 | 38,536 / 4.42% | 92.6% / 16.2% / 4.9% / 0.667 | 5,018 / 4.96% | 94.4% / 8.8% / 5.1% / 0.618 |

The 30-40 group is swept almost entirely into "flagged" (specificity collapses to
8.8-16.2%), while 18-29 is screened far more conservatively (sensitivity under 50%) - driven
by the score distribution shifting with age-linked prevalence, not by the threshold itself
changing per subgroup. **A single fixed threshold is not equally well-suited to both age
bands**; an age-stratified threshold was not implemented in this project and is a natural
next step, not a claim already made.

## Explainability

Global feature importance uses **permutation importance** (scikit-learn,
`sklearn.inspection.permutation_importance`, scoring=ROC-AUC), computed on a training-data
sample - **this is not SHAP**. `shap` was not installed and this project's build environment
had no network access to install it; rather than fabricate SHAP-labelled output, this
substitute is used and labelled as such everywhere it appears
(`reports/explainability/`). Logistic Regression coefficients (odds ratios) are kept as a
separate, more classically interpretable reference model - **not** the production model.

Age dominates permutation importance for both sexes (0.073 women / 0.186 men), followed by
BMI for women (0.056) and diastolic BP for men (0.070); diet items and blood pressure appear
throughout the top ranks for both. All importance values describe **learned model
behaviour** - reported as "predictive contribution" or "association," never as causation
("higher X was associated with a higher model-estimated probability," never "X causes
diabetes-range status").

## Limitations

- **Proxy research target**: Target A is a research-defined construct (the code currently uses the redefined target: known diabetes or glucose >= 141 mg/dL; negatives <= 100; 101-140 excluded; see the note in the Target definition section), not a clinical diagnostic protocol.
- A single **random (non-fasting)** glucose measurement has known limitations for diabetes classification.
- NFHS-5 is **cross-sectional** - EarlyDia does not predict *future* diabetes.
- **No external/clinical validation** has been performed.
- **No waist/hip** measurements used; **no men's BMI** (unavailable in this NFHS-5 extract).
- **Modest discrimination**: ROC-AUC is modest (~0.71 women, ~0.67 men), reflecting the limitations of non-invasive survey features without laboratory testing.
- **Regional state feature**: v2 models use State as a contextual regional feature; if omitted, the pipeline imputes code 9 (Uttar Pradesh). Results reflect statistical association with regional patterns in NFHS-5 sampling, not causation. The UI enforces state selection, and the API issues a warning when omitted.
- **No physical activity or family-history** variables (unavailable in this survey extract).
- Healthcare access and socioeconomic status may **confound** the target; no causal claim is made about this or about any predictor.
- **Survey design weights are not incorporated** into the modelling procedure (documented gap, not silently ignored).
- **Age-subgroup threshold instability** (see above) - one fixed threshold is not uniformly well-calibrated to operating characteristics across 18-40.
- **Low PPV** (~4-5%) is a direct, unavoidable consequence of the ~3-4% base rate at this sensitivity-oriented threshold, not a fixable modelling flaw.
- Results are screening output only and must **not** be interpreted as a diagnosis.

## Responsible use

EarlyDia is a **research and portfolio demonstration project**. It is not a medical device,
not validated for clinical use, and must not be used for medical diagnosis or treatment
decisions. Anyone with health concerns should consult a qualified healthcare professional.

## Reproducibility

- **Master seed**: `42` (data split, grouped CV, model fitting, calibration comparison) -
  `configs/config.py`.
- **Full configuration record**: `configs/reproducibility_manifest.json` (population,
  target definition, predictor lists, split/CV method, selected models, calibration,
  evaluation configurations, and exact pipeline stage orders).
- **Tests**: `tests/` (see below) - run with `python -m pytest tests -v` (or `python run_tests.py`).
- **Model artifacts**:
  - `models_v2/{women,men}_v2_pipeline.pkl`: Production v2 pipelines (women threshold 0.212, men threshold ~0.30 [artifact value 0.299]).
  - `models/{women,men}_production_pipeline.pkl`: Legacy v1 pipelines (women threshold 0.026, men threshold 0.027; retained for backward compatibility).
- **Extraction script**: `src/data/extract_nfhs.py` - see
  [`docs/DATA_REPRODUCTION.md`](docs/DATA_REPRODUCTION.md).

## How to demo

1. Start FastAPI: `uvicorn backend.api.main:app --reload --port 8000`
2. Start React: `cd frontend && npm run dev` (open http://localhost:5173).
3. Follow the 3-minute flow in [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md): Landing -> Profile (select State) -> Lifestyle -> Health -> Review -> Result.
4. Note the output: calibrated screening probability, status flag, model version & threshold line (`Model v2 - screening threshold 21.2% (women) / ~30% (men)`), regional state disclosure, and honest scope boundaries.

## Project structure

```
EarlyDia/
├── backend/                     FastAPI application (final backend)
│   ├── api/                     main.py (routes), service.py (validation + mapping), schemas.py, idrs.py, states.py
│   ├── inference/               engine.py - caches predictors (v2 default + v1 legacy)
│   ├── configs/                 settings.py - environment variables (CORS origins, build dir)
│   └── requirements.txt         backend-only dependencies (no Streamlit)
├── frontend/                    React + Vite application (final UI)
│   └── src/  components/ pages/ services/ context/ styles/ utils/
├── configs/                     config.py + locked configuration JSONs
├── data/                        raw/ interim/ processed/ (empty - not committed)
├── docs/                        DATA_REPRODUCTION.md, RESEARCH_EVIDENCE.md, DEMO_SCRIPT.md
├── experiments/                 exploratory scratch scripts and candidate evaluations
├── models/                      v1 legacy production artifacts (women 0.026, men 0.027 - backward compatibility)
├── models_v2/                   v2 production pipeline artifacts (women 0.212, men ~0.30 [artifact value 0.299])
├── reports/                     v2 reports (final_women_state_native_B.json, final_men_lr_state.json), explainability/, demo/
├── src/
│   ├── inference/               predict_module_v2.py (v2 production), predict_module.py (v1 legacy)
│   ├── app/                     LEGACY Streamlit prototype; input_spec.py is still the shared feature contract
│   ├── calibration/ data/ evaluation/ explainability/ features/ models/ target/ validation/ v2/
├── tests/                       comprehensive pytest suite (API, v1 backward-compat, v2 production, state validation)
├── requirements.txt             full dev/test environment
├── run_tests.py
└── README.md
```

## Research layer (read-only analyses)

Everything in `src/research/` is **research-only**: it reads locked artifacts/reports and writes only to `reports/research/`.
Production inference never imports it (enforced by `tests/test_research_integrity.py`). The v1 baseline models, Target A, v1 legacy thresholds
(women 0.026, men 0.027), calibration and predictors are **unchanged**; SHA-256 snapshots are in `reports/integrity/`.

Regenerate: `python -m src.research.run_all`. The React `/research` page and `GET /api/research/*` only display those generated files.

| Analysis | Status in this release |
|---|---|
| Target-definition comparison (Sahadevan 2023 alternative) | Done - **full replication not possible** from the retained extraction; no alternative prevalence reported (`reports/research/target_definition_comparison.md`) |
| Calibration (reliability table, slope, intercept, Brier) | Done, from the locked final-test report |
| Decision curve analysis | Exact net benefit at the locked cutoffs; curve across thresholds is an **approximation** from decile bins (exact when row-level predictions exist) |
| Subgroups (age x sex) | Done. Sensitivity at ages 18-29 is about 0.46 (women) / 0.39 (men) vs about 0.93 / 0.94 at 30-40 at the pooled cutoff |
| Baselines (LR / RF / HGB) | Recorded grouped-CV values reported; not recomputed |
| IDRS (Mohan 2005) | Compatibility table only - waist, physical activity and family history are unavailable |
| Feature importance | Existing permutation importance (not SHAP) |
| Survey-weighted prevalence, residence/education/state subgroups, state evaluation, ROC/PR curves | **Not run** - need row-level NFHS data (not distributed). Code is ready |
| Mapper/TDA, stability, covariance-matched null model | **Not run on NFHS data.** Implemented and unit-tested on synthetic data only; no communities or subtype findings are claimed |
| Literature benchmarking | Sahadevan verified from open access text; Geldsetzer contextual only; Barman 2025 and Topaz 2026 could not be retrieved, so no claims are made about them (`docs/REFERENCES.md`) |

Docs: `docs/RESEARCH_METHODOLOGY.md`, `docs/RESEARCH_LIMITATIONS.md`, `docs/FEATURE_RESEARCH_EVIDENCE.md`, `docs/REFERENCES.md`.

**Claim boundaries.** Established: production test performance, calibration, cluster-aware validation, observed age-subgroup performance.
Exploratory: any Mapper/TDA structure. Not established: diagnosis, future-diabetes prediction, causality, biological subtypes, clinical utility,
external or prospective validity.

---

## Running locally

Requires **Python 3.11+** (3.12 tested) and **Node.js 18+**. Commands below are for Windows
(Command Prompt). Run them from the project folder, e.g.
`C:\Users\dell\Downloads\Earlydia_updated\Earlydia`.

### 1. Backend environment and dependencies

```bat
cd C:\Users\dell\Downloads\Earlydia_updated\Earlydia
py -m venv .venv
.venv\Scripts\activate.bat
pip install -r backend\requirements.txt
```

(PowerShell: activate with `.venv\Scripts\Activate.ps1`; if scripts are blocked run
`Set-ExecutionPolicy -Scope Process Bypass` first.)

### 2. Start FastAPI

```bat
uvicorn backend.api.main:app --reload --port 8000
```

Check it: <http://localhost:8000/api/health> (interactive docs at <http://localhost:8000/docs>).

### 3. Frontend dependencies (new Command Prompt window)

```bat
cd C:\Users\dell\Downloads\Earlydia_updated\Earlydia\frontend
npm install
```

`frontend\.env` already points at the API (`VITE_API_URL=http://localhost:8000`); see `.env.example`.

### 4. Start React

```bat
npm run dev
```

Open <http://localhost:5174>. This is the demo.

*Single-process alternative:* `npm run build` in `frontend`, then only the FastAPI command above - the
backend serves the built app at <http://localhost:8000>.

### 5. Run the tests

```bat
cd C:\Users\dell\Downloads\Earlydia_updated\Earlydia
.venv\Scripts\activate.bat
pip install -r requirements.txt
python -m pytest tests -v
python run_tests.py          (pytest-free fallback)
cd frontend
npm test
```

`pip install -r requirements.txt` installs the full test environment, which includes Streamlit **only** so the
original legacy UI tests keep running; the application itself never needs it.

### Environment variables

| Variable | Where | Default |
|---|---|---|
| `VITE_API_URL` | frontend `.env` | `http://localhost:8000` (dev) / same origin (built) |
| `EARLYDIA_CORS_ORIGINS` | backend | `localhost`/`127.0.0.1` on ports 5173, 4173, 8000 |
| `EARLYDIA_FRONTEND_DIST` | backend | `frontend/dist` |

All paths are resolved relative to the project root via `configs/config.py` - no machine-specific paths
anywhere in the code.

### Legacy Streamlit prototype (not required)

`src/app/` is the original Streamlit prototype, retained as reference. It is not part of the final application.


## Docker, CI and deployment

- **Run everything in one container**: `docker build -t earlydia .` then `docker run -p 8000:8000 earlydia` (or `docker compose up --build`)
  and open http://localhost:8000. The image builds the React app and serves it from FastAPI on the same origin, so no CORS setup is needed.
  The image contains only the backend, `src/`, `configs/`, the locked `models/` and `reports/` - never raw data.
- **CI**: `.github/workflows/ci.yml` runs pytest, vitest, the Vite build and a Docker build on every push / pull request.
- **Deploy (not done for you)**:
  - *Simplest* - one web service from this Dockerfile on Render, Railway or Fly.io. The container honours the `PORT` variable the host injects.
    Free tiers sleep when idle, so the first request after a pause is slow.
  - *Split* - frontend on Vercel/Netlify with build variable `VITE_API_URL=https://<your-backend>`; backend with
    `EARLYDIA_CORS_ORIGINS=https://<your-frontend>`.
  - After deploying, put the live link at the top of this README and on your resume.
