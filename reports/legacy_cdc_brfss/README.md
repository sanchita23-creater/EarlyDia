# Legacy reports - CDC BRFSS prototype (NOT the production model)

These files come from the **first prototype** of EarlyDia, trained on the US CDC Diabetes Health
Indicators dataset (UCI #891, BRFSS, 253,680 rows). They were produced by the legacy scripts
`src/fetch_data.py`, `src/inspect_data.py`, `src/feature_selection.py`, `src/train_model.py` and
`src/evaluate_model.py`.

They describe a **Random Forest with ROC-AUC ~0.82 on a different population and a different target
(prediabetes + diabetes, 14% prevalence)**. They do **not** describe the production application,
which is the NFHS-5 (India) HistGradientBoosting models documented in the top-level README
(final test ROC-AUC 0.685 women / 0.665 men). The two sets of numbers must never be compared or mixed.

Kept only for project history.
