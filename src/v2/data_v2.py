"""
EarlyDia v2 - data access. Re-uses v1's predictor construction and the v1 PSU-grouped train/test split
(same seed search, so the split is identical to v1) and attaches the extra v2 candidate columns
(state, wealth) from the audit dataset. REQUIRES the v1 pipeline stages to have been run first:
  python -m src.target.women_target_construction ; python -m src.target.men_target_construction
(they write data/processed/{sex}_target_a.pkl; no model artifact is touched).

`load_dev` returns TRAINING rows only. The held-out test rows are never returned here;
only src/v2/final_v2.py (explicitly gated) imports `_full` to open them.
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from configs.config import DATA_PROCESSED_DIR, CONFIGS_DIR, REPORTS_DIR, PROJECT_ROOT
from src.features.predictor_construction import build_women_predictors, build_men_predictors
from src.validation.train_test_split import (find_group_split, SEED_CANDIDATES, TEST_SIZE,
                                             PREVALENCE_REL_TOLERANCE)
from src.v2.features_v2 import EXTRA_SOURCES

GROUP_COL = {"women": "v021_psu", "men": "mv021_psu"}
V2_DATA_DIR = DATA_PROCESSED_DIR / "v2"          # inside data/processed (git-ignored)
V2_REPORTS_DIR = REPORTS_DIR / "v2"
V2_MODELS_DIR = PROJECT_ROOT / "models_v2"       # separate from locked models/
N_SPLITS, CV_SEED = 5, 42                        # identical to v1 oof_generation


def _full(sex):
    build = build_women_predictors if sex == "women" else build_men_predictors
    pred = build()
    raw = pd.read_pickle(DATA_PROCESSED_DIR / f"{sex}_target_a.pkl")
    for new, col in EXTRA_SOURCES[sex].items():
        pred[new] = raw.loc[pred.index, col].astype(float)
    res = find_group_split(pred, GROUP_COL[sex], "target_a", SEED_CANDIDATES, TEST_SIZE,
                           PREVALENCE_REL_TOLERANCE)
    # the split must reproduce the locked v1 split exactly
    cfg = json.loads((CONFIGS_DIR / "split_config.json").read_text())[sex]
    assert res["seed"] == cfg["selected_seed"], "split seed differs from locked v1 split"
    assert len(res["train_idx"]) == cfg["train_rows"] and len(res["test_idx"]) == cfg["test_rows"], \
        "split sizes differ from locked v1 split - upstream data/pipeline mismatch"
    return pred, res["train_idx"], res["test_idx"]


def load_dev(sex):
    pred, tr, _ = _full(sex)
    return pred.iloc[tr].copy()


def make_folds(dev, sex):
    """Same StratifiedGroupKFold(5, shuffle, 42) over PSU as v1 -> all v2 experiments are paired."""
    y = dev["target_a"].astype(int).values
    g = dev[GROUP_COL[sex]].values
    folds = np.full(len(dev), -1)
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
    for k, (tr, va) in enumerate(cv.split(dev, y, g)):
        assert not (set(g[tr]) & set(g[va])), "PSU leakage across CV folds"
        folds[va] = k
    assert (folds >= 0).all()
    return folds
