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
    else:
        raise ValueError(f"unknown model {model_name}")
    return Pipeline([("pre", pre), ("clf", clf)])
