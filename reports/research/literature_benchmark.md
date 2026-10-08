# Literature benchmarking

## Sahadevan et al. 2023
Sahadevan et al., J Glob Health 2023 (open access: https://pmc.ncbi.nlm.nih.gov/articles/PMC10704946/)

| Aspect | EarlyDia | Paper |
|---|---|---|
| population | NFHS-5, ages 18-40, non-pregnant women + men, modelled separately | NFHS-5 men and women |
| age range | 18-40 | 15-49 |
| target definition | Target A: known diabetes OR valid glucose >=200 -> positive; explicit-No flags + glucose <=140 -> negative | Undiagnosed diabetes: 'no'/'don't know' to diagnosed diabetes AND (fasting >=126 mg/dL with >=8 h fasting OR random >=200 mg/dL) |
| glucose definition | Single capillary glucose, >=200 positive, <=140 negative, 141-199 excluded | Fasting >=126 (>=8 h) or random >=200 |
| known diabetes handling | Known diabetes counted as positive | Known/diagnosed diabetes is the comparison/excluded group; outcome is undiagnosed only |
| survey weighting | Production model unweighted; weighted descriptive analysis is research-only | Not verified in this build - complete from the full text |
| exclusions | Pregnancy (women), missing/ambiguous Target A | Not verified in this build - complete from the full text |
| prevalence | ~3.0-3.2% Target A (women test 3.03%); NOT comparable | Not verified in this build - complete from the full text |

## Geldsetzer et al. 2018
Historical/contextual Indian prevalence evidence only. Its national prevalence is NOT a target EarlyDia should reproduce.

- Different survey (NFHS-4, 2015-16 vs NFHS-5, 2019-21)
- Adults 18+ vs EarlyDia 18-40
- Different outcome definition (diabetes prevalence vs a model-ready screening label that excludes 141-199 mg/dL)
- Prevalence study vs prediction/screening prototype

Specific numeric details were not re-verified in this build.

## Barman et al. 2025
Methodological differences that warrant consideration include the items below, once verified from the paper. No leakage claim is made.

| Aspect | EarlyDia | Paper |
|---|---|---|
| dataset | NFHS-5 (2019-21) individual + men's recode | Not verified in this build - complete from the full text |
| population | India, ages 18-40 | Not verified in this build - complete from the full text |
| target | Target A (locked) | Not verified in this build - complete from the full text |
| predictors | age, BMI (women), BP, tobacco, alcohol, diet, education, residence | Not verified in this build - complete from the full text |
| validation | Group-aware 80/20 split, StratifiedGroupKFold CV | Not verified in this build - complete from the full text |
| group PSU handling | PSU grouping; no PSU in both train and test | Not verified in this build - complete from the full text |
| calibration | Women isotonic, men sigmoid/Platt, fit on grouped OOF | Not verified in this build - complete from the full text |
| SMOTE | None (class weighting) | Not verified in this build - complete from the full text |
| model types | HistGradientBoosting (LR baseline in model selection) | Not verified in this build - complete from the full text |
| explainability | Permutation importance | Not verified in this build - complete from the full text |
| thresholding | Sensitivity floor 0.75 on OOF: women 0.026, men 0.027 | Not verified in this build - complete from the full text |
| DCA | Research-only (reports/research/decision_curve.csv) | Not verified in this build - complete from the full text |
| subgroup evaluation | Age x sex (locked); others need row-level data | Not verified in this build - complete from the full text |
