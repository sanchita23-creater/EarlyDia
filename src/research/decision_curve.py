"""Decision curve analysis (research-only) - EarlyDia vs treat-all vs treat-none.

Decision curve analysis evaluates potential decision usefulness and does not establish clinical utility.
It is NOT used to change the locked thresholds (women 0.026, men 0.027).

DATA LIMITATION: row-level locked test predictions are not shipped with the project (data/ is
empty). Two outputs are therefore produced:
  * EXACT net benefit at the locked operating threshold, from the locked test confusion matrix.
  * APPROXIMATE curve across thresholds from the 10-bin reliability table: each bin is treated
    as one risk stratum. This is a decile-resolution approximation, labelled as such.
If data/processed/<sex>_test_predictions.npz (y, p) exists, an exact curve is computed instead.
"""
import csv

import numpy as np

from src.research.common import OUT, THRESHOLDS, DATA_PROCESSED, load_final_eval, net_benefit, ensure_dirs

GRID = np.round(np.arange(0.005, 0.1001, 0.005), 4)


def _from_bins(bins, pt):
    n = np.array([b["n"] for b in bins], float)
    p = np.array([b["mean_predicted"] for b in bins])
    pos = n * np.array([b["frac_positive"] for b in bins])
    sel = p >= pt
    tp, fp = pos[sel].sum(), (n - pos)[sel].sum()
    N, P = n.sum(), pos.sum()
    return tp, fp, N, P


def run():
    ensure_dirs()
    fe = load_final_eval()
    rows = []
    for sex in ("women", "men"):
        ov = fe[sex]["overall"]
        cm = ov["confusion_matrix"]; N = ov["n"]; prev = ov["prevalence"]
        t = THRESHOLDS[sex]
        rows.append(dict(sex=sex, kind="exact_at_locked_threshold", threshold=t,
                         net_benefit_model=net_benefit(cm["tp"], cm["fp"], N, t),
                         net_benefit_treat_all=prev - (1 - prev) * t / (1 - t), net_benefit_treat_none=0.0,
                         source="locked final test confusion matrix"))
        exact = DATA_PROCESSED / f"{sex}_test_predictions.npz"
        if exact.exists():
            d = np.load(exact); y, p = d["y"], d["p"]; src = "exact (row-level locked predictions)"
        else:
            y = p = None; src = "APPROXIMATE (decile reliability bins)"
        for pt in GRID:
            if y is not None:
                sel = p >= pt; tp = int((sel & (y == 1)).sum()); fp = int((sel & (y == 0)).sum()); NN = len(y); prv = y.mean()
            else:
                tp, fp, NN, P = _from_bins(fe[sex]["reliability_table"], pt); prv = P / NN
            rows.append(dict(sex=sex, kind="curve", threshold=float(pt),
                             net_benefit_model=net_benefit(tp, fp, NN, pt),
                             net_benefit_treat_all=prv - (1 - prv) * pt / (1 - pt), net_benefit_treat_none=0.0, source=src))
    cols = ["sex", "kind", "threshold", "net_benefit_model", "net_benefit_treat_all", "net_benefit_treat_none", "source"]
    with open(OUT / "decision_curve.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    return rows


if __name__ == "__main__":
    run()
