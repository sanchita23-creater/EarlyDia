"""
EarlyDia - Threshold review only. Uses existing OOF predictions + existing saved calibrators.
No retraining, no test-set access.
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from configs.config import (DATA_RAW_DIR, DATA_INTERIM_DIR, DATA_PROCESSED_DIR,
                            EXPLAINABILITY_REPORTS_DIR)
import pickle
import numpy as np
import pandas as pd
from src.calibration.calibration_comparison import sigmoid_calibrate_predict

REQUESTED_THRESHOLDS = [0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.050, 0.075, 0.100]
SENSITIVITY_FLOORS = [0.60, 0.70, 0.75, 0.80, 0.85]


def apply_saved_calibrator(artifact, p_raw):
    method = artifact["calibration_method"]
    if method == "isotonic":
        return artifact["calibrator"].predict(p_raw)
    else:
        return sigmoid_calibrate_predict(artifact["calibrator"], p_raw)


def threshold_table(y, p_calibrated, thresholds):
    rows = []
    n = len(y)
    for t in thresholds:
        pred = (p_calibrated >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        f1 = 2 * prec * sens / (prec + sens) if (prec and sens and (prec + sens) > 0) else np.nan
        pred_pos_rate = (tp + fp) / n
        rows.append({"threshold": t, "sensitivity": sens, "specificity": spec,
                      "precision_ppv": prec, "npv": npv, "f1": f1,
                      "predicted_positive_rate": pred_pos_rate,
                      "tp": tp, "fp": fp, "tn": tn, "fn": fn})
    return pd.DataFrame(rows)


def best_specificity_at_floor(y, p_calibrated, floor, search_grid=None):
    """Scan a fine threshold grid; among thresholds with sensitivity >= floor, return the
    one with highest specificity."""
    if search_grid is None:
        search_grid = np.arange(0.001, 0.301, 0.001)
    best = None
    for t in search_grid:
        pred = (p_calibrated >= t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        tn = ((pred == 0) & (y == 0)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        if sens >= floor:
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
            npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
            pred_pos_rate = (tp + fp) / len(y)
            if best is None or spec > best["specificity"]:
                best = {"threshold": t, "sensitivity": sens, "specificity": spec,
                        "precision_ppv": prec, "npv": npv, "predicted_positive_rate": pred_pos_rate}
    return best


if __name__ == "__main__":
    for sex, oof_path, artifact_path in [
        ("women", str(DATA_PROCESSED_DIR) + "/women_oof.pkl",
         str(DATA_PROCESSED_DIR) + "/women_production_pipeline.pkl"),
        ("men", str(DATA_PROCESSED_DIR) + "/men_oof.pkl",
         str(DATA_PROCESSED_DIR) + "/men_production_pipeline.pkl"),
    ]:
        oof_df = pd.read_pickle(oof_path)
        with open(artifact_path, "rb") as f:
            artifact = pickle.load(f)

        y = oof_df["target_a"].values
        p_raw = oof_df["oof_HistGradientBoosting"].values
        p_calibrated = apply_saved_calibrator(artifact, p_raw)

        table = threshold_table(y, p_calibrated, REQUESTED_THRESHOLDS)
        print(f"\n{'='*90}\n{sex.upper()} - threshold trade-off table (calibrated OOF, HGB, no retraining)\n{'='*90}")
        print(table.drop(columns=["tp", "fp", "tn", "fn"]).to_string(index=False))
        table.to_csv(str(DATA_PROCESSED_DIR) + f"/{sex}_threshold_review_table.csv", index=False)

        print(f"\n{sex.upper()} - best specificity achievable at each sensitivity floor")
        floor_rows = []
        for floor in SENSITIVITY_FLOORS:
            b = best_specificity_at_floor(y, p_calibrated, floor)
            if b:
                floor_rows.append({"sensitivity_floor": floor, **b})
                print(f"  floor>={floor:.2f}: threshold={b['threshold']:.3f} "
                      f"sensitivity={b['sensitivity']:.3f} specificity={b['specificity']:.3f} "
                      f"PPV={b['precision_ppv']:.3f} NPV={b['npv']:.3f} "
                      f"pred_pos_rate={b['predicted_positive_rate']:.3f}")
            else:
                print(f"  floor>={floor:.2f}: NOT ACHIEVABLE in search grid")
        pd.DataFrame(floor_rows).to_csv(str(DATA_PROCESSED_DIR) + f"/{sex}_sensitivity_floor_table.csv", index=False)
