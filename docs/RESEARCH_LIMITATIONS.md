# Research limitations

**Outcome / target**
- Target A is a research-defined survey outcome from a single capillary glucose reading plus self-report. It is not a clinical diagnosis. The 141-199 mg/dL band without known diabetes is excluded, and known diabetes counts as positive, so Target A is neither "undiagnosed diabetes" nor a clinical standard.
- The literature-aligned alternative (fasting >=126 mg/dL with >=8 h fasting, or random >=200 mg/dL) could **not** be reproduced: fasting status/duration variables cannot be verified from the retained extraction. No alternative prevalence is reported. Prevalence from different definitions is not comparable.

**Model performance**
- Discrimination is modest (final test ROC-AUC women 0.685, men 0.665). The locked sensitivity-oriented cutoffs flag about half of people (women 52.8%) with PPV about 4-5%.
- Pooled-threshold performance is heterogeneous by age: sensitivity at ages 18-29 is about 0.46 (women) and 0.39 (men) versus about 0.93 and 0.94 at ages 30-40. Subgroup-specific thresholds were deliberately not created.
- No external, prospective or clinical validation. Cross-sectional survey data; no future-diabetes prediction.

**Analyses that were not run in this build (row-level NFHS data not distributed)**
- Survey-weighted prevalence, residence/education/state subgroup metrics, state-stratified evaluation, ROC/PR curves, exact decision curves, grouped-CV re-run, Mapper/TDA on real data, stability and null-model results. Code exists in `src/research/` and is unit-tested on synthetic data only. Re-run with `python -m src.research.run_all` once `data/processed/*.pkl` are reproduced (docs/DATA_REPRODUCTION.md).
- The decision-curve line is an approximation from decile reliability bins; only the net benefit at the locked cutoff is exact.
- Baseline comparison reuses recorded grouped-CV values, and the identical LR ROC-AUC (0.6788) recorded for both sexes should be checked.
- IDRS could not be reproduced (waist, physical activity and family history are absent).
- Barman et al. 2025 and Topaz 2026 could not be retrieved; no claims about them are made.

**Interpretation**
- Permutation importance is not SHAP and not causal. Geographic or subgroup differences are descriptive only. Mapper communities (if produced) are exploratory feature-space groupings, not biological subtypes; EarlyDia lacks HbA1c, C-peptide, autoantibodies and insulin-resistance markers.

**Known documentation inconsistencies (not resolved - the NFHS-5 row-level data needed to re-derive them is not distributed)**
- Grouped-CV model-comparison numbers differ between the README table (e.g. women HGB ROC-AUC 0.6818, PR-AUC 0.0930; men LR 0.6792) and `configs/model_selection.json` / `reports/research/baseline_comparison.csv` (0.6816 / 0.0924; men LR 0.6788). They may use different aggregation (fold mean vs pooled OOF) but this has **not** been verified. The locked final-test results (`reports/final_evaluation/`) are unaffected. The identical LR ROC-AUC 0.6788 for women and men also still needs checking.
- `reports/legacy_cdc_brfss/` holds reports from the first prototype (US CDC BRFSS data, Random Forest, ROC-AUC ~0.82). They are **not** the production model and must not be compared with the NFHS-5 results.

**IDRS comparator**
- IDRS (Mohan 2005) targets undiagnosed diabetes; EarlyDia's Target A is a different research-defined outcome. The two outputs are shown side by side for context only and are not evidence that either is more accurate. IDRS inputs are not available in the NFHS-5 extract, so no IDRS-vs-model accuracy comparison on NFHS-5 is possible (see `reports/research/idrs_compatibility.csv`).
