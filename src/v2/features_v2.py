"""
EarlyDia v2 - feature specs, preprocessing and model registry (SEPARATE from v1).

Feature flags (comma separated, combinable):
  state         women v024 / men mv024 as a one-hot categorical (contextual, not target-encoded)
  state_native  same variable, ordinal-coded and used as a NATIVE categorical in HistGradientBoosting
                (HGB models only)
  wealth_v190   DHS national wealth quintile (v190 / mv190)   - reverses a v1 design exclusion
  wealth_v190a  DHS urban/rural-specific wealth quintile      - test separately from v190
  eng           deterministic row-wise features derived from existing BP/BMI predictors:
                pulse pressure, mean arterial pressure, bmi_missing (women)
  compact       HGB-friendly encoding: tobacco/alcohol-type/diet item codes kept as numeric codes
                instead of one-hot (v1 one-hot encodes ~40 item columns)
All preprocessing lives inside the sklearn Pipeline (fit per training fold only).
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.models.grouped_cv_baseline import get_feature_lists  # v1 lists, import only

SEED = 42
EXTRA_SOURCES = {
    "women": {"state": "v024", "wealth_v190": "v190", "wealth_v190a": "v190a"},
    "men": {"state": "mv024", "wealth_v190": "mv190", "wealth_v190a": "mv190a"},
}
ITEM_PREFIXES = ("tob_", "alc_s72", "alc_sm62", "diet_")
VALID_FLAGS = {"state", "state_native", "wealth_v190", "wealth_v190a", "eng", "compact"}

# HistGradientBoosting configurations. "hgb_default" == v1's model. No early stopping on purpose:
# sklearn's internal validation split is row-random (not PSU-grouped).
HGB_CONFIGS = {
    "hgb_default": dict(class_weight="balanced"),
    "hgb_B": dict(class_weight="balanced", learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
                  min_samples_leaf=100, l2_regularization=1.0),
    "hgb_C": dict(class_weight="balanced", learning_rate=0.05, max_iter=300, max_leaf_nodes=8,
                  min_samples_leaf=200, l2_regularization=5.0),
    "hgb_D": dict(class_weight="balanced", learning_rate=0.03, max_iter=300, max_leaf_nodes=15,
                  min_samples_leaf=300, l2_regularization=10.0, max_features=0.5),
    "hgb_E": dict(class_weight=None, learning_rate=0.05, max_iter=300, max_leaf_nodes=8,
                  min_samples_leaf=200, l2_regularization=5.0),
}

# XGBoost candidate configurations. We start with a sensible constrained search space.
import xgboost as xgb
XGB_CONFIGS = {
    "xgb_default": dict(scale_pos_weight=20.0), # Approximate prevalence weighting
    "xgb_A": dict(scale_pos_weight=15.0, learning_rate=0.05, n_estimators=200, max_depth=4,
                  min_child_weight=10, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0),
    "xgb_B": dict(scale_pos_weight=25.0, learning_rate=0.05, n_estimators=300, max_depth=3,
                  min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_alpha=2.0),
    "xgb_C": dict(scale_pos_weight=30.0, learning_rate=0.03, n_estimators=300, max_depth=4,
                  min_child_weight=20, subsample=0.7, colsample_bytree=0.7, reg_lambda=10.0),
    "xgb_opt_1": dict(scale_pos_weight=20.0, learning_rate=0.1, n_estimators=300, max_depth=5, min_child_weight=20, subsample=1.0, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=10.0),
    "xgb_opt_2": dict(scale_pos_weight=25.0, learning_rate=0.05, n_estimators=200, max_depth=4, min_child_weight=1, subsample=0.6, colsample_bytree=0.8, reg_alpha=5.0, reg_lambda=1.0),
    "xgb_opt_3": dict(scale_pos_weight=15.0, learning_rate=0.01, n_estimators=300, max_depth=3, min_child_weight=10, subsample=0.8, colsample_bytree=1.0, reg_alpha=0, reg_lambda=5.0),
    "xgb_opt_4": dict(scale_pos_weight=30.0, learning_rate=0.03, n_estimators=100, max_depth=6, min_child_weight=5, subsample=0.6, colsample_bytree=1.0, reg_alpha=1.0, reg_lambda=10.0),
    "xgb_opt_5": dict(scale_pos_weight=10.0, learning_rate=0.1, n_estimators=200, max_depth=3, min_child_weight=1, subsample=1.0, colsample_bytree=0.6, reg_alpha=5.0, reg_lambda=10.0),
    "xgb_opt_6": dict(scale_pos_weight=25.0, learning_rate=0.01, n_estimators=100, max_depth=5, min_child_weight=5, subsample=0.8, colsample_bytree=0.6, reg_alpha=0, reg_lambda=1.0),
    "xgb_opt_7": dict(scale_pos_weight=10.0, learning_rate=0.03, n_estimators=300, max_depth=4, min_child_weight=20, subsample=0.6, colsample_bytree=1.0, reg_alpha=1.0, reg_lambda=5.0),
    "xgb_opt_8": dict(scale_pos_weight=15.0, learning_rate=0.05, n_estimators=100, max_depth=6, min_child_weight=10, subsample=1.0, colsample_bytree=0.8, reg_alpha=0, reg_lambda=10.0),
    "xgb_opt_9": dict(scale_pos_weight=20.0, learning_rate=0.01, n_estimators=200, max_depth=3, min_child_weight=1, subsample=0.6, colsample_bytree=1.0, reg_alpha=5.0, reg_lambda=5.0),
    "xgb_opt_10": dict(scale_pos_weight=30.0, learning_rate=0.1, n_estimators=100, max_depth=4, min_child_weight=20, subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0),
}


def parse_flags(s):
    flags = {f.strip() for f in (s or "").split(",") if f.strip() and f.strip() != "none"}
    bad = flags - VALID_FLAGS
    if bad:
        raise ValueError(f"unknown feature flags: {sorted(bad)}")
    if {"wealth_v190", "wealth_v190a"} <= flags:
        raise ValueError("test wealth_v190 and wealth_v190a separately, not together")
    if {"state", "state_native"} <= flags:
        raise ValueError("choose state OR state_native")
    return flags


def add_engineered(df):
    out = df.copy()
    out["pulse_pressure"] = out["avg_systolic_bp"] - out["avg_diastolic_bp"]
    out["mean_arterial_pressure"] = out["avg_diastolic_bp"] + out["pulse_pressure"] / 3.0
    if "bmi" in out.columns:
        out["bmi_missing"] = out["bmi"].isna().astype(float)
    return out


def feature_columns(sex, flags):
    num_median, num_c0, cat_c0, cat_nm = get_feature_lists(sex)
    cols = list(num_median) + list(num_c0) + list(cat_c0) + list(cat_nm)
    for f in ("state", "state_native", "wealth_v190", "wealth_v190a"):
        if f in flags:
            cols.append("state" if f == "state_native" else f)
    return cols


def prepare_X(df, sex, flags):
    """Select/derive the raw model inputs. `df` must carry the extra columns named
    state / wealth_v190 / wealth_v190a when those flags are requested (see data_v2)."""
    X = add_engineered(df) if "eng" in flags else df
    cols = feature_columns(sex, flags)
    if "eng" in flags:
        cols = cols + ["pulse_pressure", "mean_arterial_pressure"] + (["bmi_missing"] if sex == "women" else [])
    return X[cols].copy()


def build_preprocessor_v2(sex, flags):
    num_median, num_c0, cat_c0, cat_nm = get_feature_lists(sex)
    num_median, cat_nm = list(num_median), list(cat_nm)
    passthrough_num = []
    if "eng" in flags:
        num_median += ["pulse_pressure", "mean_arterial_pressure"]
        if sex == "women":
            passthrough_num.append("bmi_missing")
    compact_num = []
    if "compact" in flags:
        compact_num = [c for c in cat_nm if c.startswith(ITEM_PREFIXES)]
        cat_nm = [c for c in cat_nm if c not in compact_num]
    extra_cat = [("state" if f == "state_native" else f) for f in
                 ("state", "state_native", "wealth_v190", "wealth_v190a") if f in flags
                 and f != "state_native"]
    ohe = lambda: OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # noqa: E731
    tf = []
    if "state_native" in flags:   # must stay FIRST: HGB gets categorical_features=[0]
        tf.append(("state_native", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                    encoded_missing_value=np.nan))]), ["state"]))
    tf += [
        ("num_median", Pipeline([("impute", SimpleImputer(strategy="median")),
                                 ("scale", StandardScaler())]), num_median),
        ("num_const0", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=0)),
                                 ("scale", StandardScaler())]), list(num_c0)),
        ("cat_const0", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=0)),
                                 ("onehot", ohe())]), list(cat_c0)),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", ohe())]), cat_nm + extra_cat),
    ]
    if compact_num:
        tf.append(("item_codes", SimpleImputer(strategy="most_frequent"), compact_num))
    if passthrough_num:
        tf.append(("flags", "passthrough", passthrough_num))
    return ColumnTransformer(tf, sparse_threshold=0.0)


CATBOOST_CONFIGS = {
    "cb_default": dict(depth=6, learning_rate=0.05, iterations=300,
                       auto_class_weights="Balanced", random_seed=42, verbose=0),
    "cb_A": dict(depth=4, learning_rate=0.05, iterations=300,
                 auto_class_weights="Balanced", random_seed=42, verbose=0,
                 l2_leaf_reg=5.0, min_data_in_leaf=200),
    "cb_B": dict(depth=6, learning_rate=0.03, iterations=500,
                 auto_class_weights="Balanced", random_seed=42, verbose=0,
                 l2_leaf_reg=10.0, min_data_in_leaf=300, subsample=0.8),
    "cb_C": dict(depth=4, learning_rate=0.03, iterations=500,
                 auto_class_weights="Balanced", random_seed=42, verbose=0,
                 l2_leaf_reg=3.0, min_data_in_leaf=100, border_count=128),
    "cb_D": dict(depth=8, learning_rate=0.05, iterations=200,
                 auto_class_weights="Balanced", random_seed=42, verbose=0,
                 l2_leaf_reg=1.0, min_data_in_leaf=50, random_strength=2.0),
}

LIGHTGBM_CONFIGS = {
    "lgbm_default": dict(n_estimators=300, learning_rate=0.05, max_depth=6,
                          num_leaves=31, min_child_samples=200, subsample=0.8,
                          colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=5.0,
                          is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1),
    "lgbm_A": dict(n_estimators=300, learning_rate=0.05, max_depth=4,
                    num_leaves=15, min_child_samples=300, subsample=0.8,
                    colsample_bytree=0.8, reg_alpha=5.0, reg_lambda=10.0,
                    is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1),
    "lgbm_B": dict(n_estimators=500, learning_rate=0.03, max_depth=6,
                    num_leaves=31, min_child_samples=200, subsample=0.7,
                    colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=5.0,
                    is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1),
    "lgbm_C": dict(n_estimators=500, learning_rate=0.03, max_depth=4,
                    num_leaves=15, min_child_samples=100, subsample=0.8,
                    colsample_bytree=0.8, reg_alpha=2.0, reg_lambda=3.0,
                    is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1),
    "lgbm_D": dict(n_estimators=200, learning_rate=0.05, max_depth=8,
                    num_leaves=63, min_child_samples=50, subsample=0.8,
                    colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=1.0,
                    is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1),
}


def build_pipeline(sex, flags, model_name):
    pre = build_preprocessor_v2(sex, flags)
    if model_name == "lr":
        if "state_native" in flags:
            raise ValueError("state_native is for HGB models only")
        clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)
    elif model_name in HGB_CONFIGS:
        kw = dict(HGB_CONFIGS[model_name])
        if "state_native" in flags:
            kw["categorical_features"] = [0]
        clf = HistGradientBoostingClassifier(random_state=SEED, **kw)
    elif model_name in CATBOOST_CONFIGS:
        from catboost import CatBoostClassifier
        clf = CatBoostClassifier(**CATBOOST_CONFIGS[model_name])
    elif model_name in LIGHTGBM_CONFIGS:
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(**LIGHTGBM_CONFIGS[model_name])
    elif model_name in XGB_CONFIGS:
        kw = dict(XGB_CONFIGS[model_name])
        clf = xgb.XGBClassifier(random_state=SEED, use_label_encoder=False, eval_metric="logloss", n_jobs=-1, **kw)
    elif model_name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=10, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)
    else:
        raise ValueError(f"unknown model {model_name}")
    return Pipeline([("pre", pre), ("clf", clf)])
