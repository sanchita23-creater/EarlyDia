"""
EarlyDia - Stage: Apply selected calibration + build screening threshold table.
Uses ONLY OOF training predictions. No test-set access.

Threshold selection rule (a documented screening-threshold CHOICE, not a clinically optimal
threshold): scan a 0.001-step grid of thresholds on the calibrated grouped-OOF probabilities
and take the HIGHEST threshold whose sensitivity is still >= 0.75. Applied to the OOF
predictions this yields 0.026 (women) and 0.027 (men) - the locked final thresholds, also
recorded in configs/config.py (DOCUMENTED_THRESHOLDS) and the production artifacts.
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
from src.calibration.calibration_comparison import (sigmoid_calibrate_fit, sigmoid_calibrate_predict,
                                     isotonic_calibrate_fit, reliability_table,
                                     calibration_slope_intercept)
from sklearn.metrics import brier_score_loss

SELECTED_MODEL = "HistGradientBoosting"
CALIB_METHOD = {"women": "isotonic", "men": "sigmoid"}

THRESHOLDS = [round(float(t), 3) for t in np.arange(0.001, 0.301, 0.001)]   # selection grid (0.001 step)
DISPLAY_THRESHOLDS = [0.010, 0.020, 0.030, 0.040, 0.050, 0.075, 0.100, 0.150, 0.200, 0.300]  # printed rows only


def fit_final_calibrator(oof_df, model_col, method):
    y = oof_df["target_a"].values
    p = oof_df[model_col].values
    if method == "isotonic":
        return isotonic_calibrate_fit(p, y), "isotonic"
    else:
        return sigmoid_calibrate_fit(p, y), "sigmoid"


def apply_calibrator(calibrator, method, p):
    if method == "isotonic":
        return calibrator.predict(p)
    else:
        return sigmoid_calibrate_predict(calibrator, p)


def threshold_table(y, p_calibrated, thresholds):
    rows = []
    n = len(y)
    for t in thresholds:
        pred = (p_calibrated >= t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        tn = ((pred == 0) & (y == 0)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        f1 = 2 * prec * sens / (prec + sens) if (prec is not np.nan and prec + sens > 0) else np.nan
        pred_pos_rate = (tp + fp) / n
        rows.append({"threshold": t, "sensitivity": sens, "specificity": spec,
                      "precision_ppv": prec, "npv": npv, "f1": f1,
                      "predicted_positive_rate": pred_pos_rate,
                      "tp": tp, "fp": fp, "tn": tn, "fn": fn})
    return pd.DataFrame(rows)


def select_threshold(table, min_sensitivity=0.75):
    """Screening rule: the HIGHEST threshold on the grid whose OOF calibrated sensitivity is
    still >= min_sensitivity (specificity never decreases as the threshold rises, so this is
    also the highest-specificity option meeting the floor). If no threshold clears the floor,
    fall back to the threshold with the highest achievable sensitivity."""
    eligible = table[table.sensitivity >= min_sensitivity]
    if len(eligible) == 0:
        return table.sort_values("sensitivity", ascending=False).iloc[0], False
    return eligible.sort_values("threshold").iloc[-1], True


if __name__ == "__main__":
    import json
    summary = {}

    for sex, group_col, oof_path in [
        ("women", "v021_psu", str(DATA_PROCESSED_DIR) + "/women_oof.pkl"),
        ("men", "mv021_psu", str(DATA_PROCESSED_DIR) + "/men_oof.pkl"),
    ]:
        oof_df = pd.read_pickle(oof_path)
        model_col = f"oof_{SELECTED_MODEL}"
        method = CALIB_METHOD[sex]

        calibrator, method_used = fit_final_calibrator(oof_df, model_col, method)
        p_calibrated = apply_calibrator(calibrator, method_used, oof_df[model_col].values)
        y = oof_df["target_a"].values

        brier_raw = brier_score_loss(y, oof_df[model_col].values)
        brier_calibrated = brier_score_loss(y, p_calibrated)
        slope, intercept = calibration_slope_intercept(y, p_calibrated)
        reliability = reliability_table(y, p_calibrated, n_bins=10)

        print(f"=== {sex} | model={SELECTED_MODEL} | calibration={method_used} ===")
        print(f"  Brier (raw OOF): {brier_raw:.5f}")
        print(f"  Brier (calibrated OOF): {brier_calibrated:.5f}")
        print(f"  Calibration slope/intercept (post-calibration, ~1/~0 = well calibrated): "
              f"{slope:.3f} / {intercept:.3f}")
        print("  Reliability table (post-calibration):")
        print(reliability.to_string(index=False))
        print()

        table = threshold_table(y, p_calibrated, THRESHOLDS)
        print("  Threshold table (calibrated OOF probabilities):")
        print(table[table.threshold.isin(DISPLAY_THRESHOLDS)].to_string(index=False))
        print()

        chosen_row, floor_achieved = select_threshold(table, min_sensitivity=0.75)
        print(f"  SELECTED THRESHOLD: {chosen_row.threshold:.3f} "
              f"(sensitivity floor 0.75 {'achieved' if floor_achieved else 'NOT achieved - used best available'})")
        print(f"    sensitivity={chosen_row.sensitivity:.3f} specificity={chosen_row.specificity:.3f} "
              f"precision/PPV={chosen_row.precision_ppv:.3f} NPV={chosen_row.npv:.3f} "
              f"F1={chosen_row.f1:.3f} predicted_positive_rate={chosen_row.predicted_positive_rate:.3f}")
        print()

        table.to_csv(str(DATA_PROCESSED_DIR) + f"/{sex}_threshold_table.csv", index=False)
        reliability.to_csv(str(DATA_PROCESSED_DIR) + f"/{sex}_reliability_table.csv", index=False)

        summary[sex] = {
            "selected_model": SELECTED_MODEL,
            "calibration_method": method_used,
            "brier_raw_oof": float(brier_raw),
            "brier_calibrated_oof": float(brier_calibrated),
            "calibration_slope": float(slope),
            "calibration_intercept": float(intercept),
            "selected_threshold": float(chosen_row.threshold),
            "sensitivity_floor_target": 0.75,
            "sensitivity_floor_achieved": bool(floor_achieved),
            "threshold_metrics": {
                "sensitivity": float(chosen_row.sensitivity),
                "specificity": float(chosen_row.specificity),
                "precision_ppv": float(chosen_row.precision_ppv),
                "npv": float(chosen_row.npv),
                "f1": float(chosen_row.f1),
                "predicted_positive_rate": float(chosen_row.predicted_positive_rate),
                "tp": int(chosen_row.tp), "fp": int(chosen_row.fp),
                "tn": int(chosen_row.tn), "fn": int(chosen_row.fn),
            },
            "threshold_selection_rule": (
                "Highest threshold (0.001-step grid) whose OOF calibrated sensitivity is >= 0.75; "
                "if none clears the floor, the threshold with the highest achievable sensitivity. Selected using ONLY grouped-CV OOF training "
                "predictions; the final test set was never accessed."
            ),
        }

    with open(str(DATA_PROCESSED_DIR) + "/threshold_selection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved threshold tables, reliability tables, and threshold_selection_summary.json")
