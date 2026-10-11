"""
EarlyDia v2 - Train and save production pipeline artifacts for the best benchmark models.

This script creates the v2 production models (women: LightGBM lgbm_D, men: CatBoost cb_B)
and saves them to models_v2/ for use by the backend API.

Usage:
  python -m src.v2.save_production_models --sex women
  python -m src.v2.save_production_models --sex men
  python -m src.v2.save_production_models --sex both
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse
import pickle
import numpy as np
import pandas as pd

from src.v2 import data_v2 as D
from src.v2 import evaluation_v2 as E
from src.v2.features_v2 import build_pipeline, parse_flags, prepare_X


BEST_MODELS = {
    "women": dict(name="lgbm_D", features="state", model="lgbm_D"),
    "men": dict(name="cb_B", features="state", model="cb_B"),
}


def train_and_save(sex, floor=0.75):
    """Train the best model for this sex, calibrate, and save the artifact."""
    cfg = BEST_MODELS[sex]
    name, features, model = cfg["name"], cfg["features"], cfg["model"]
    flags = parse_flags(features)

    print(f"\n{'='*70}")
    print(f"Training {sex} production model: {name}")
    print(f"{'='*70}")

    D.V2_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    dev = D.load_dev(sex)
    folds = D.make_folds(dev, sex)
    y = dev["target_a"].astype(int).values
    X = prepare_X(dev, sex, flags)

    pkl_path = D.V2_DATA_DIR / f"{sex}_oof_bench_{name}.pkl"
    if pkl_path.exists():
        oof = pd.read_pickle(pkl_path)
        p_oof = oof["p_raw"].values
        print(f"[{sex}] Loaded cached OOF predictions from {pkl_path.name}")
    else:
        print(f"[{sex}] Generating OOF predictions...")
        p_oof = np.full(len(dev), np.nan)
        for k in range(D.N_SPLITS):
            tr, va = folds != k, folds == k
            pipe = build_pipeline(sex, flags, model).fit(
                X.iloc[tr] if hasattr(X, "iloc") else X[tr], y[tr]
            )
            p_oof[va] = pipe.predict_proba(
                X.iloc[va] if hasattr(X, "iloc") else X[va]
            )[:, 1]
            print(f"  fold {k} done")
        assert not np.isnan(p_oof).any()
        pd.DataFrame({
            "y": y,
            "psu": dev[D.GROUP_COL[sex]].values,
            "fold": folds,
            "age": dev["age"].values,
            "p_raw": p_oof,
        }, index=dev.index).to_pickle(pkl_path)
        print(f"  saved OOF to {pkl_path.name}")

    print(f"[{sex}] Calibrating...")
    method, p_cal_oof, _ = E.choose_calibration(y, p_oof, folds)
    print(f"  calibration method: {method}")

    thr = E.threshold_for_sensitivity(y, p_cal_oof, floor)
    print(f"  threshold at {floor*100:.0f}% sensitivity: {thr:.4f}")

    band_thr, _, _ = E.age_band_thresholds(y, p_cal_oof, dev["age"].values, floor)
    print(f"  age-band thresholds: {band_thr}")

    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        calibrator = IsotonicRegression(
            out_of_bounds="clip", y_min=0, y_max=1
        ).fit(p_oof, y)
    else:
        from src.calibration.calibration_utils import sigmoid_calibrate_fit
        calibrator = sigmoid_calibrate_fit(p_oof, y)

    print(f"[{sex}] Refitting pipeline on full dev data...")
    pipe = build_pipeline(sex, flags, model).fit(X, y)

    artifact = dict(
        base_pipeline=pipe,
        feature_cols=list(X.columns) if hasattr(X, "columns") else list(range(X.shape[1])),
        feature_flags=sorted(flags),
        model_name=model,
        calibration_method=method,
        calibrator=calibrator,
        selected_threshold=thr,
        age_band_thresholds=band_thr,
        sens_floor=floor,
        sex=sex,
        version="v2-production",
        candidate=name,
        note="Screening model on NFHS-5 cross-sectional data; not diagnostic.",
    )

    out_path = D.V2_MODELS_DIR / f"{sex}_v2_pipeline.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"[{sex}] Saved artifact to {out_path}")

    from sklearn.metrics import roc_auc_score, average_precision_score
    auc = roc_auc_score(y, p_cal_oof)
    ap = average_precision_score(y, p_cal_oof)
    print(f"[{sex}] OOF ROC-AUC: {auc:.4f}, PR-AUC: {ap:.4f}")

    return artifact


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sex", choices=["women", "men", "both"], required=True)
    ap.add_argument("--floor", type=float, default=0.75,
                    help="Target sensitivity floor for threshold selection (default: 0.75)")
    a = ap.parse_args(argv)

    if a.sex == "both":
        train_and_save("women", a.floor)
        train_and_save("men", a.floor)
    else:
        train_and_save(a.sex, a.floor)

    print("\n" + "="*70)
    print("Done! V2 production models saved to models_v2/")
    print("="*70)


if __name__ == "__main__":
    main()
