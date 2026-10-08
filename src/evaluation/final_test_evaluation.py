"""
EarlyDia - FINAL UNTOUCHED TEST EVALUATION
Uses the saved production pipelines (base HGB model refit on full training data,
OOF-fit calibrator, and the newly documented 75%-sensitivity-floor thresholds).
This is the ONLY script in the entire project that opens women_test.pkl / men_test.pkl.
No tuning, no retraining, no threshold changes happen after this point.
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from configs.config import (DATA_RAW_DIR, DATA_INTERIM_DIR, DATA_PROCESSED_DIR,
                            EXPLAINABILITY_REPORTS_DIR)
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                              confusion_matrix)
from src.calibration.calibration_comparison import sigmoid_calibrate_predict, calibration_slope_intercept, reliability_table


def apply_calibrator(artifact, p_raw):
    if artifact["calibration_method"] == "isotonic":
        return artifact["calibrator"].predict(p_raw)
    else:
        return sigmoid_calibrate_predict(artifact["calibrator"], p_raw)


def compute_metrics(y_true, p_calibrated, threshold):
    y_pred = (p_calibrated >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if (ppv and sensitivity and (ppv + sensitivity) > 0) else np.nan
    n = len(y_true)
    n_pred_pos = int(tp + fp)

    roc_auc = roc_auc_score(y_true, p_calibrated) if len(np.unique(y_true)) > 1 else np.nan
    pr_auc = average_precision_score(y_true, p_calibrated) if len(np.unique(y_true)) > 1 else np.nan
    brier = brier_score_loss(y_true, p_calibrated)

    return {
        "n": int(n), "n_positive": int(y_true.sum()), "prevalence": float(y_true.mean()),
        "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else None,
        "pr_auc": float(pr_auc) if not np.isnan(pr_auc) else None,
        "sensitivity": float(sensitivity), "specificity": float(specificity),
        "ppv_precision": float(ppv), "npv": float(npv), "f1": float(f1),
        "brier_score": float(brier),
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        "n_predicted_positive": n_pred_pos,
        "pct_predicted_positive": float(n_pred_pos / n * 100),
    }


def evaluate_sex(sex, test_path, artifact_path, age_col):
    with open(artifact_path, "rb") as f:
        artifact = pickle.load(f)

    test_df = pd.read_pickle(test_path)
    feature_cols = artifact["feature_cols"]
    threshold = artifact["selected_threshold"]

    X_test = test_df[feature_cols]
    y_test = test_df["target_a"].astype(int).values

    p_raw = artifact["base_pipeline"].predict_proba(X_test)[:, 1]
    p_calibrated = apply_calibrator(artifact, p_raw)

    print(f"\n{'='*80}\n{sex.upper()} - FINAL TEST EVALUATION\n"
          f"model=HistGradientBoosting | calibration={artifact['calibration_method']} | "
          f"threshold={threshold}\n{'='*80}")

    overall = compute_metrics(y_test, p_calibrated, threshold)
    for k, v in overall.items():
        print(f"  {k}: {v}")

    slope, intercept = calibration_slope_intercept(y_test, p_calibrated)
    reliability = reliability_table(y_test, p_calibrated, n_bins=10)
    print(f"\n  Calibration slope/intercept (test set): {slope:.3f} / {intercept:.3f}")
    print("  Reliability table (test set):")
    print(reliability.to_string(index=False))

    overall["calibration_slope"] = float(slope)
    overall["calibration_intercept"] = float(intercept)

    # ---- age subgroup analysis ----
    subgroups = {}
    age = test_df[age_col].values
    for label, lo, hi in [("age_18_29", 18, 29), ("age_30_40", 30, 40)]:
        mask = (age >= lo) & (age <= hi)
        n_sub = mask.sum()
        if n_sub == 0:
            subgroups[label] = {"n": 0, "note": "no rows in this subgroup"}
            continue
        y_sub = y_test[mask]
        p_sub = p_calibrated[mask]
        m = compute_metrics(y_sub, p_sub, threshold)
        subgroups[label] = m
        print(f"\n  --- {label} (n={m['n']}, positive={m['n_positive']}, "
              f"prevalence={m['prevalence']*100:.2f}%) ---")
        print(f"    ROC-AUC={m['roc_auc']}, PR-AUC={m['pr_auc']}, "
              f"sensitivity={m['sensitivity']:.3f}, specificity={m['specificity']:.3f}, "
              f"PPV={m['ppv_precision']:.3f}, NPV={m['npv']:.3f}")

    return {
        "sex": sex,
        "model": "HistGradientBoosting",
        "calibration_method": artifact["calibration_method"],
        "threshold": threshold,
        "threshold_selection_rule": artifact.get("threshold_selection_rule", "see prior documentation"),
        "overall": overall,
        "reliability_table": reliability.to_dict(orient="records"),
        "age_subgroups": subgroups,
    }


if __name__ == "__main__":
    w_result = evaluate_sex("women", str(DATA_PROCESSED_DIR) + "/women_test.pkl",
                             str(DATA_PROCESSED_DIR) + "/women_production_pipeline.pkl", "age")
    m_result = evaluate_sex("men", str(DATA_PROCESSED_DIR) + "/men_test.pkl",
                             str(DATA_PROCESSED_DIR) + "/men_production_pipeline.pkl", "age")

    final_report = {"women": w_result, "men": m_result,
                     "note": "This is the ONLY evaluation performed on the final untouched test "
                             "sets. No retraining, recalibration, or threshold changes occurred "
                             "after this point."}
    with open(str(DATA_PROCESSED_DIR) + "/final_test_evaluation.json", "w") as f:
        json.dump(final_report, f, indent=2, default=str)
    print("\n\nSaved: data/processed/final_test_evaluation.json")
