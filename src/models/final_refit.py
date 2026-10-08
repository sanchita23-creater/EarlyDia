"""
EarlyDia - Stage: Final refit on FULL training data + save production pipeline artifacts.
Base model (HistGradientBoosting) is refit on ALL of women_train.pkl / men_train.pkl.
Calibrator was already correctly fit on OOF predictions (not on final-model predictions on its
own training data, which would be biased) - this OOF-fit calibrator is paired with the
full-data-refit base model, per standard nested calibration practice.
NO test-set access anywhere in this script.
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from configs.config import (DOCUMENTED_THRESHOLDS, DATA_RAW_DIR, DATA_INTERIM_DIR, DATA_PROCESSED_DIR,
                            EXPLAINABILITY_REPORTS_DIR)
import json
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from src.models.grouped_cv_baseline import build_preprocessor, SEED
from src.calibration.calibration_comparison import isotonic_calibrate_fit, sigmoid_calibrate_fit

CALIB_METHOD = {"women": "isotonic", "men": "sigmoid"}
SELECTED_THRESHOLD = dict(DOCUMENTED_THRESHOLDS)   # locked final thresholds (configs/config.py)


def refit_and_save(train_df, sex, group_col, oof_path):
    preprocessor, feature_cols = build_preprocessor(sex)
    model = HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED)
    pipe = Pipeline([("pre", preprocessor), ("clf", model)])

    X = train_df[feature_cols]
    y = train_df["target_a"].astype(int)
    pipe.fit(X, y)
    print(f"[{sex}] base pipeline refit on FULL training data: {len(X)} rows, {len(feature_cols)} raw features")

    oof_df = pd.read_pickle(oof_path)
    method = CALIB_METHOD[sex]
    p_oof = oof_df["oof_HistGradientBoosting"].values
    y_oof = oof_df["target_a"].values
    if method == "isotonic":
        calibrator = isotonic_calibrate_fit(p_oof, y_oof)
    else:
        calibrator = sigmoid_calibrate_fit(p_oof, y_oof)
    print(f"[{sex}] calibrator ({method}) fit on {len(p_oof)} OOF training predictions")

    artifact = {
        "base_pipeline": pipe,
        "feature_cols": feature_cols,
        "calibration_method": method,
        "calibrator": calibrator,
        "selected_threshold": SELECTED_THRESHOLD[sex],
        "target_definition": "Target A (locked)",
        "sex": sex,
        "random_seed": SEED,
    }
    out_path = str(DATA_PROCESSED_DIR) + f"/{sex}_production_pipeline.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"[{sex}] saved production artifact -> {out_path}")
    return out_path


if __name__ == "__main__":
    women_train = pd.read_pickle(str(DATA_PROCESSED_DIR) + "/women_train.pkl")
    men_train = pd.read_pickle(str(DATA_PROCESSED_DIR) + "/men_train.pkl")

    w_path = refit_and_save(women_train, "women", "v021_psu", str(DATA_PROCESSED_DIR) + "/women_oof.pkl")
    m_path = refit_and_save(men_train, "men", "mv021_psu", str(DATA_PROCESSED_DIR) + "/men_oof.pkl")

    manifest = {
        "women_pipeline": w_path,
        "men_pipeline": m_path,
        "note": "Base model refit on FULL training data; calibrator fit on OOF (out-of-fold) "
                "training predictions only, per standard nested calibration practice. "
                "Final test sets (women_test.pkl, men_test.pkl) were NOT accessed at any point "
                "in this stage.",
        "seed": SEED,
    }
    with open(str(DATA_PROCESSED_DIR) + "/production_pipeline_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("\nSaved production_pipeline_manifest.json")
