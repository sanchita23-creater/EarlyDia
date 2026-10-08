"""
EarlyDia - Stage: Calibration method comparison, using OOF predictions only (no test-set access).
For each sex and each candidate model, compare sigmoid (Platt) vs isotonic calibration via a
nested grouped 5-fold evaluation on the OOF probabilities themselves: fit calibrator on 4/5 of
OOF rows (respecting PSU groups), evaluate Brier on the held-out 1/5, average across folds.
The winning method is then refit on ALL OOF data for the production calibrator artifact.
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from configs.config import (DATA_RAW_DIR, DATA_INTERIM_DIR, DATA_PROCESSED_DIR,
                            EXPLAINABILITY_REPORTS_DIR)
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score

SEED = 42
N_SPLITS = 5


def sigmoid_calibrate_fit(p_train, y_train):
    """Platt scaling: logistic regression of y on logit(p)."""
    eps = 1e-6
    p_clip = np.clip(p_train, eps, 1 - eps)
    logit_p = np.log(p_clip / (1 - p_clip)).reshape(-1, 1)
    lr = LogisticRegression()
    lr.fit(logit_p, y_train)
    return lr


def sigmoid_calibrate_predict(lr, p):
    eps = 1e-6
    p_clip = np.clip(p, eps, 1 - eps)
    logit_p = np.log(p_clip / (1 - p_clip)).reshape(-1, 1)
    return lr.predict_proba(logit_p)[:, 1]


def isotonic_calibrate_fit(p_train, y_train):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(p_train, y_train)
    return iso


def calibration_slope_intercept(y, p):
    """Cox calibration regression: y ~ logit(p). slope=1,intercept=0 => perfectly calibrated."""
    eps = 1e-6
    p_clip = np.clip(p, eps, 1 - eps)
    logit_p = np.log(p_clip / (1 - p_clip)).reshape(-1, 1)
    lr = LogisticRegression()
    lr.fit(logit_p, y)
    return lr.coef_[0][0], lr.intercept_[0]


def reliability_table(y, p, n_bins=10):
    bins = pd.qcut(p, q=n_bins, duplicates="drop")
    df = pd.DataFrame({"y": y, "p": p, "bin": bins})
    grouped = df.groupby("bin", observed=True).agg(
        mean_predicted=("p", "mean"), frac_positive=("y", "mean"), n=("y", "size"))
    return grouped.reset_index(drop=True)


def compare_calibration_methods(oof_df, group_col, model_col, sex, model_name):
    y = oof_df["target_a"].values
    p = oof_df[model_col].values
    groups = oof_df[group_col].values

    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    briers_sigmoid, briers_isotonic, briers_raw = [], [], []

    for tr_idx, va_idx in cv.split(oof_df, y, groups):
        assert set(groups[tr_idx]) & set(groups[va_idx]) == set(), "PSU leakage in calibration CV!"
        p_tr, p_va = p[tr_idx], p[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        sig = sigmoid_calibrate_fit(p_tr, y_tr)
        p_va_sig = sigmoid_calibrate_predict(sig, p_va)
        briers_sigmoid.append(brier_score_loss(y_va, p_va_sig))

        iso = isotonic_calibrate_fit(p_tr, y_tr)
        p_va_iso = iso.predict(p_va)
        briers_isotonic.append(brier_score_loss(y_va, p_va_iso))

        briers_raw.append(brier_score_loss(y_va, p_va))

    result = {
        "model": model_name,
        "raw_brier_mean": np.mean(briers_raw), "raw_brier_std": np.std(briers_raw),
        "sigmoid_brier_mean": np.mean(briers_sigmoid), "sigmoid_brier_std": np.std(briers_sigmoid),
        "isotonic_brier_mean": np.mean(briers_isotonic), "isotonic_brier_std": np.std(briers_isotonic),
    }
    winner = "sigmoid" if result["sigmoid_brier_mean"] <= result["isotonic_brier_mean"] else "isotonic"
    result["winner"] = winner
    print(f"[{sex}] {model_name}: raw_brier={result['raw_brier_mean']:.5f} "
          f"sigmoid={result['sigmoid_brier_mean']:.5f} isotonic={result['isotonic_brier_mean']:.5f} "
          f"-> winner={winner}")
    return result


if __name__ == "__main__":
    import json

    for sex, group_col, oof_path in [
        ("women", "v021_psu", str(DATA_PROCESSED_DIR) + "/women_oof.pkl"),
        ("men", "mv021_psu", str(DATA_PROCESSED_DIR) + "/men_oof.pkl"),
    ]:
        oof_df = pd.read_pickle(oof_path)
        results = []
        for model_col, model_name in [("oof_LogisticRegression", "LogisticRegression"),
                                        ("oof_HistGradientBoosting", "HistGradientBoosting")]:
            r = compare_calibration_methods(oof_df, group_col, model_col, sex, model_name)
            # slope/intercept and reliability on raw OOF (uncalibrated) for diagnostic purposes
            slope, intercept = calibration_slope_intercept(
                oof_df["target_a"].values, oof_df[model_col].values)
            r["raw_calibration_slope"] = slope
            r["raw_calibration_intercept"] = intercept
            r["raw_roc_auc"] = roc_auc_score(oof_df["target_a"], oof_df[model_col])
            r["raw_pr_auc"] = average_precision_score(oof_df["target_a"], oof_df[model_col])
            results.append(r)

        with open(str(DATA_PROCESSED_DIR) + f"/{sex}_calibration_comparison.json", "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"saved {sex}_calibration_comparison.json\n")
