"""Calibration analysis from the LOCKED final-test reliability table (no recalibration).

Discrimination (ROC-AUC) and calibration (do predicted risks match observed frequencies)
are different properties; a model can rank people moderately and still be well calibrated.
Calibration is NEVER tuned here; production calibrators are untouched.
"""
import csv

import numpy as np

from src.research.common import OUT, load_final_eval, ensure_dirs


def run():
    ensure_dirs()
    fe = load_final_eval()
    rows = []
    for sex in ("women", "men"):
        ov = fe[sex]["overall"]
        bins = fe[sex]["reliability_table"]
        n = np.array([b["n"] for b in bins], float)
        p = np.array([b["mean_predicted"] for b in bins])
        o = np.array([b["frac_positive"] for b in bins])
        ece = float(np.sum(n * np.abs(p - o)) / n.sum())          # decile-binned ECE
        for i, b in enumerate(bins, 1):
            rows.append(dict(sex=sex, row_type="bin", bin=i, n=b["n"], mean_predicted=b["mean_predicted"],
                             observed_fraction=b["frac_positive"], abs_gap=abs(b["mean_predicted"] - b["frac_positive"])))
        rows.append(dict(sex=sex, row_type="summary", bin="", n=int(ov["n"]),
                         mean_predicted=float((n * p).sum() / n.sum()), observed_fraction=ov["prevalence"],
                         abs_gap=ece, brier_score=ov["brier_score"],
                         calibration_slope=ov["calibration_slope"], calibration_intercept=ov["calibration_intercept"]))
    cols = ["sex", "row_type", "bin", "n", "mean_predicted", "observed_fraction", "abs_gap",
            "brier_score", "calibration_slope", "calibration_intercept"]
    with open(OUT / "calibration_analysis.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    with open(OUT / "calibration_curves" / "reliability_points.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["sex", "mean_predicted", "observed_fraction", "n"])
        for r in rows:
            if r["row_type"] == "bin":
                w.writerow([r["sex"], r["mean_predicted"], r["observed_fraction"], r["n"]])
    return rows


if __name__ == "__main__":
    run()
