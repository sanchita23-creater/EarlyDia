"""Subgroup performance from the locked final-test report (age x sex, plus overall).

Uses the locked sex-specific thresholds; no subgroup-specific thresholds are created.
Residence / education / state subgroups need row-level data and are produced by
`subgroup_from_rows()` when data/processed/*_test_predictions.* is present (see run_all).
"""
import csv

from src.research.common import OUT, THRESHOLDS, load_final_eval, confusion_metrics, ensure_dirs

COLS = ["sex", "subgroup", "n", "n_positive", "prevalence", "roc_auc", "pr_auc", "sensitivity", "specificity",
        "ppv", "npv", "f1", "brier_score", "false_positive_rate", "false_negative_rate",
        "calibration_slope", "calibration_intercept", "threshold_used", "note"]


def _row(sex, name, d):
    cm = d["confusion_matrix"]; m = confusion_metrics(cm["tp"], cm["fp"], cm["tn"], cm["fn"])
    return dict(sex=sex, subgroup=name, n=d["n"], n_positive=d["n_positive"], prevalence=d["prevalence"],
                roc_auc=d["roc_auc"], pr_auc=d["pr_auc"], sensitivity=m["sensitivity"], specificity=m["specificity"],
                ppv=m["ppv"], npv=m["npv"], f1=m["f1"], brier_score=d["brier_score"],
                false_positive_rate=m["fpr"], false_negative_rate=m["fnr"],
                calibration_slope=d.get("calibration_slope", ""), calibration_intercept=d.get("calibration_intercept", ""),
                threshold_used=THRESHOLDS[sex],
                note="" if "calibration_slope" in d else "subgroup calibration slope/intercept not stored in locked report")


def run():
    ensure_dirs()
    fe = load_final_eval(); rows = []
    for sex in ("women", "men"):
        rows.append(_row(sex, "overall", fe[sex]["overall"]))
        for name, d in fe[sex]["age_subgroups"].items():
            rows.append(_row(sex, name, d))
    with open(OUT / "subgroup_performance.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    return rows


if __name__ == "__main__":
    run()
