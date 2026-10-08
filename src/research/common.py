"""Shared helpers for research modules (read-only access to locked artifacts)."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
OUT = REPORTS / "research"
FIG = OUT / "figures"
DATA_PROCESSED = ROOT / "data" / "processed"
THRESHOLDS = {"women": 0.026, "men": 0.027}   # locked; asserted against the manifest below
PSU_COL = {"women": "v021_psu", "men": "mv021_psu"}
PROD_LABEL = "Production model (locked)"
RESEARCH_LABEL = "Research analysis (not used by the production model)"


def ensure_dirs():
    for d in (OUT, FIG, OUT / "calibration_curves", OUT / "mapper"):
        d.mkdir(parents=True, exist_ok=True)


def load_final_eval():
    return json.loads((REPORTS / "final_evaluation" / "final_test_evaluation.json").read_text(encoding="utf-8"))


def assert_thresholds_locked():
    m = json.loads((ROOT / "models" / "production_pipeline_manifest.json").read_text(encoding="utf-8"))
    assert m["final_thresholds"] == THRESHOLDS, "threshold drift detected - research aborted"


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def confusion_metrics(tp, fp, tn, fn):
    d = lambda a, b: a / b if b else float("nan")
    sens, spec = d(tp, tp + fn), d(tn, tn + fp)
    ppv, npv = d(tp, tp + fp), d(tn, tn + fn)
    f1 = d(2 * ppv * sens, ppv + sens) if not (math.isnan(ppv) or math.isnan(sens)) else float("nan")
    return dict(sensitivity=sens, specificity=spec, ppv=ppv, npv=npv, f1=f1,
                fpr=1 - spec if not math.isnan(spec) else float("nan"),
                fnr=1 - sens if not math.isnan(sens) else float("nan"))


def net_benefit(tp, fp, n, pt):
    return tp / n - fp / n * (pt / (1 - pt))


def metrics_by_group(y, p, groups, thr, min_n=200, min_pos=10):
    """Per-group metrics at a LOCKED threshold. Groups below min_n/min_pos get metrics suppressed
    (flagged 'too small'), never silently reported. No group-specific thresholds."""
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    y = np.asarray(y); p = np.asarray(p); g = np.asarray(groups)
    rows = []
    for name in sorted(set(g[~_isnull(g)]), key=str):
        m = g == name
        yy, pp = y[m], p[m]
        n, pos = int(m.sum()), int(yy.sum())
        row = dict(group=name, n=n, n_positive=pos, prevalence=pos / n if n else float("nan"))
        if n < min_n or pos < min_pos or pos == n:
            row["note"] = f"too small (n<{min_n} or positives<{min_pos}) - metrics suppressed"
            rows.append(row); continue
        sel = pp >= thr
        tp = int((sel & (yy == 1)).sum()); fp = int((sel & (yy == 0)).sum())
        tn = int((~sel & (yy == 0)).sum()); fn = int((~sel & (yy == 1)).sum())
        row.update(confusion_metrics(tp, fp, tn, fn))
        row.update(roc_auc=roc_auc_score(yy, pp), pr_auc=average_precision_score(yy, pp),
                   brier_score=brier_score_loss(yy, pp), mean_predicted=float(pp.mean()),
                   flagged_rate=float(sel.mean()), error_rate=(fp + fn) / n)
        try:
            from src.calibration.calibration_utils import calibration_slope_intercept
            s, i = calibration_slope_intercept(yy, pp)
            row.update(calibration_slope=s, calibration_intercept=i)
        except Exception:
            pass
        rows.append(row)
    return rows


def _isnull(a):
    import pandas as pd
    return pd.isnull(a)
