"""Research-only baseline comparison (LR vs RF vs production HGB) under grouped CV.

The numbers come from the project's ORIGINAL grouped (StratifiedGroupKFold, PSU) CV run, recorded in
configs/model_selection.json. They are NOT recomputed here (no row-level data in this build) and no
model is retrained. `rerun()` re-executes the grouped CV only when data/processed/*_train.pkl exists.
Purpose: show the HGB choice is defensible - NOT to search for a new production model.
"""
import csv

from src.research.common import OUT, ensure_dirs

SRC = "configs/model_selection.json (original grouped CV on training data; not recomputed)"
ROWS = [
    ("women", "HistGradientBoosting (production)", 0.6816, 0.0924),
    ("women", "LogisticRegression", 0.6788, 0.0756),
    ("women", "RandomForest", 0.643, 0.068),
    ("men", "HistGradientBoosting (production)", 0.6673, 0.0848),
    ("men", "LogisticRegression", 0.6788, 0.0762),
    ("men", "RandomForest", 0.636, 0.067),
]
NOTE = ("Men: LR has higher OOF ROC-AUC, HGB higher PR-AUC and better calibration slope - recorded as a judgment call in the source. "
        "The LR OOF ROC-AUC value 0.6788 is identical for women and men in the source file; verify against grouped_cv outputs before publication.")


def run():
    ensure_dirs()
    with open(OUT / "baseline_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sex", "model", "oof_roc_auc", "oof_pr_auc", "validation", "source", "note"])
        for s, m, r, p in ROWS:
            w.writerow([s, m, r, p, "StratifiedGroupKFold(5), PSU groups", SRC, NOTE if s == "men" and "Logistic" in m else ""])


def rerun(sex):
    """Re-run grouped CV for the three models (research-only, writes nothing to models/)."""
    import pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score, average_precision_score
    from src.models.grouped_cv_baseline import build_preprocessor, get_models, N_SPLITS, SEED
    from src.research.common import DATA_PROCESSED, PSU_COL
    from src.research.validation_checks import cv_fold_psu_overlaps
    import numpy as np
    df = pd.read_pickle(DATA_PROCESSED / f"{sex}_train.pkl")
    pre, cols = build_preprocessor(sex)
    X, y, g = df[cols], df.target_a.values, df[PSU_COL[sex]].values
    folds = list(StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED).split(X, y, g))
    assert max(cv_fold_psu_overlaps(g, folds)) == 0
    out = []
    from sklearn.base import clone
    for name, model in get_models().items():
        oof = np.zeros(len(y))
        for tr, va in folds:
            pipe = Pipeline([("pre", clone(pre)), ("m", clone(model))]).fit(X.iloc[tr], y[tr])
            oof[va] = pipe.predict_proba(X.iloc[va])[:, 1]
        out.append(dict(sex=sex, model=name, oof_roc_auc=roc_auc_score(y, oof), oof_pr_auc=average_precision_score(y, oof)))
    return out
