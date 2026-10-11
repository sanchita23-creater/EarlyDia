"""
EarlyDia - XGBoost Candidate Pipeline
=======================================
Provides XGBClassifier-based pipelines that integrate with the existing v2
experiment and evaluation framework. All preprocessing reuses v2's
ColumnTransformer (build_preprocessor_v2) so feature schemas are identical
across all candidate models.

This module does NOT modify or replace any production artifact. XGBoost
candidates are evaluated via run_v2.py's OOF framework and promoted
through final_v2.py only after explicit approval.
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import numpy as np
import xgboost as xgb
from sklearn.pipeline import Pipeline

from src.v2.features_v2 import (
    build_preprocessor_v2,
    parse_flags,
    prepare_X,
    XGB_CONFIGS,
    SEED,
)

# Re-export for convenience
__all__ = ["XGB_CONFIGS", "build_xgb_pipeline", "XGB_SEARCH_SPACE"]

# Hyperparameter search space documentation (for reproducibility).
# These are the bounds explored; actual configs in XGB_CONFIGS are the
# manually selected points within this space.
XGB_SEARCH_SPACE = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "min_child_weight": [5, 10, 20, 50],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "reg_alpha": [0, 1, 2, 5],
    "reg_lambda": [1, 5, 10],
    "gamma": [0, 0.1, 0.5],
    "scale_pos_weight": [10, 15, 20, 25, 30],
    "note": (
        "scale_pos_weight approximates 1/prevalence (~3.5% positives). "
        "Explored via manual grid search on PSU-grouped 5-fold CV OOF "
        "predictions. No early stopping used (would require a properly "
        "separated validation set inside each fold, adding complexity "
        "without demonstrated benefit at these small iteration counts)."
    ),
}


def build_xgb_pipeline(sex: str, flags: set, config_name: str = "xgb_default") -> Pipeline:
    """Build a sklearn Pipeline with v2 preprocessing + XGBClassifier.

    Parameters
    ----------
    sex : str
        'women' or 'men'.
    flags : set
        Feature flags (e.g. {'state'}).
    config_name : str
        Key in XGB_CONFIGS.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted pipeline ready for .fit(X, y).
    """
    if config_name not in XGB_CONFIGS:
        raise ValueError(f"Unknown XGBoost config '{config_name}'. "
                         f"Available: {sorted(XGB_CONFIGS.keys())}")
    pre = build_preprocessor_v2(sex, flags)
    kw = dict(XGB_CONFIGS[config_name])
    clf = xgb.XGBClassifier(
        random_state=SEED,
        use_label_encoder=False,
        eval_metric="logloss",
        **kw,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def validate_xgb_artifact(artifact: dict) -> list:
    """Check that a saved XGBoost artifact has the expected structure.

    Returns a list of error messages (empty if valid).
    """
    errors = []
    required_keys = [
        "base_pipeline", "feature_cols", "calibration_method",
        "calibrator", "selected_threshold", "sex", "version",
    ]
    for k in required_keys:
        if k not in artifact:
            errors.append(f"Missing key '{k}' in artifact")

    if "base_pipeline" in artifact:
        pipe = artifact["base_pipeline"]
        if not hasattr(pipe, "predict_proba"):
            errors.append("base_pipeline does not support predict_proba")

    if "feature_cols" in artifact:
        cols = artifact["feature_cols"]
        if not isinstance(cols, list) or len(cols) == 0:
            errors.append("feature_cols must be a non-empty list")

    if "selected_threshold" in artifact:
        thr = artifact["selected_threshold"]
        if not (0 < thr < 1):
            errors.append(f"selected_threshold {thr} is outside (0, 1)")

    if "calibrated_probability" in artifact:
        # This is a prediction-time check, not artifact-time
        pass

    return errors
