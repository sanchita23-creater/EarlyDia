# Research methodology (TRIPOD+AI-style report)

Production ML is **locked**; everything under `src/research/` is read-only analysis.

| Item | EarlyDia |
|---|---|
| Source population | NFHS-5 (India, 2019-21): Individual Recode (women 15-49) and Men's Recode; modelled separately |
| Eligibility | Age 18-40 |
| Exclusions | Pregnant women (v213 == 1; 27,826 of 510,289 age-eligible women); rows without a valid Target A (ambiguous known flag, no valid glucose, glucose 141-199 without known diabetes) |
| Target | Target A (locked). Women: known = sb56/sb57/s728a == 1; men: smb56/smb57/sm627a == 1. Positive = known OR valid glucose >= 200 mg/dL; negative = all flags explicit No AND valid glucose <= 140 |
| Predictors | Age; BMI (women only); mean systolic/diastolic BP and valid-reading counts; tobacco items; alcohol items; nine diet-frequency items; education; residence. No wealth, waist, hypertension, family history or physical activity |
| Missing data | Median imputation (numeric), constant-0 for structural non-drinker fields, most-frequent for categoricals, inside the sklearn pipeline |
| Survey structure / clustering | PSU (v021/mv021) is the grouping unit. Weights/strata are used only in research descriptive analysis, never in the model |
| Train/test design | GroupShuffleSplit by PSU, 80/20, seed search 42-91 with prevalence tolerance, first accepted seed 42; no PSU in both sets |
| Cross-validation | StratifiedGroupKFold, 5 folds, PSU groups (never random CV) |
| Model | HistGradientBoosting with class weighting; LR and RF were grouped-CV comparators |
| Calibration | Women isotonic, men sigmoid/Platt, fitted on grouped out-of-fold training predictions |
| Threshold | Highest value on a 0.001 grid with OOF sensitivity >= 0.75: women 0.026, men 0.027 (training/OOF only) |
| Final test metrics | Women n=86,669: ROC-AUC 0.685, PR-AUC 0.083, sens 0.762, spec 0.480, PPV 0.044, Brier 0.0287. Men n=11,408: ROC-AUC 0.665 (see `reports/final_evaluation/`) |
| Subgroup evaluation | Age 18-29 / 30-40 by sex from the locked report; residence, education, state need row-level data (not run) |
| Research add-ons | Calibration table; decision-curve analysis (exact at cutoff, approximate curve); target-definition comparison; IDRS compatibility; literature benchmarking; Mapper/TDA + stability + covariance-matched null model (code only) |
| Integrity | SHA-256 before/after snapshots in `reports/integrity/`; automated tests assert artifacts, Target A source, thresholds and research isolation |

**Why grouped validation:** people in one PSU share environment and sampling, so a random split can place near-duplicates in train and test and give optimistic estimates (Roberts et al. 2017).

**Discrimination vs calibration:** ROC-AUC measures ranking; calibration measures whether predicted probabilities match observed frequencies (Van Calster et al. 2019). Both are reported; calibration was never tuned on the final test set.

**Decision curve analysis** evaluates potential decision usefulness and does not establish clinical utility; it never changes the locked cutoffs.

**Intended use:** a research/demo screening prototype that estimates similarity to a research-defined NFHS-5 outcome, for ages 18-40.
**Not intended for:** diagnosis, predicting future diabetes, treatment decisions, causal inference, other populations or age ranges.
**Not claimed:** external, clinical or prospective validation; clinical utility; biological diabetes subtypes.
