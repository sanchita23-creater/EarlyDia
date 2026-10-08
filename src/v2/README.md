# EarlyDia v2 experiments (v1 stays locked)

Only NEW files: `src/v2/*`, `tests/test_v2_smoke.py`. Outputs: `data/processed/v2/` (git-ignored),
`reports/v2/`, `models_v2/`. Nothing under `models/`, `configs/`, `reports/final_evaluation|research` is written.

## 0. Prerequisites (v1 stages, no model artifacts touched)
    python -m src.target.women_target_construction
    python -m src.target.men_target_construction
(They read data\interim\*_key_columns_raw.pkl and write data\processed\{sex}_target_a.pkl. v2 rebuilds the
predictors + the identical PSU split from those and asserts it matches configs/split_config.json.)

## 1. Baseline (must reproduce v1 before anything else is trusted)
    python -m src.v2.run_v2 batch --sex women --group baseline
    python -m src.v2.run_v2 batch --sex men   --group baseline
Sanity: `v1_repro` OOF ROC-AUC should be ~0.68 (women) / ~0.67 (men); threshold-rule flag rate ~0.52-0.54.

## 2. Experiments (development data only; same PSU folds -> paired)
    --group state | wealth | tune | eng        (run for women and men)
    python -m src.v2.run_v2 compare --sex women --names state,tune_C --ref v1_repro --boot 200
    python -m src.v2.run_v2 bands   --sex women --name tune_C
Combine winners:  oof --name combo --features state,eng --model hgb_C   (then compare vs v1_repro)

## 3. Final (once, after choosing on dev data)
    python -m src.v2.final_v2 --sex women --name <candidate> --confirm-test-access
