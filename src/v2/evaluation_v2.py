"""
EarlyDia v2 - evaluation utilities (OOF-based; no test-set access in this module).

Calibration is CROSS-FITTED: the calibrator that scores fold k's OOF predictions is fit on the other
folds' OOF predictions only, so calibrated-OOF metrics (and thresholds chosen on them) are not
computed with a calibrator that saw the same rows. (v1 fit one calibrator on all OOF rows.)
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

AGE_BANDS = [(18, 24), (25, 29), (30, 34), (35, 40)]
THRESHOLD_GRID = [round(float(t), 3) for t in np.arange(0.001, 0.301, 0.001)]  # same grid as v1
_EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_apply_calibrator(method, p_tr, y_tr, p_new):
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_tr, y_tr).predict(p_new)
    lr = LogisticRegression().fit(_logit(p_tr), y_tr)
    return lr.predict_proba(_logit(p_new))[:, 1]


def crossfit_calibrate(y, p, folds, method):
    out = np.empty(len(p), float)
    for k in np.unique(folds):
        va = folds == k
        out[va] = fit_apply_calibrator(method, p[~va], y[~va], p[va])
    return out


def choose_calibration(y, p, folds):
    """Cross-fitted sigmoid vs isotonic, chosen by Brier (as in v1). Returns (method, p_cal, briers)."""
    res = {m: crossfit_calibrate(y, p, folds, m) for m in ("sigmoid", "isotonic")}
    briers = {m: brier_score_loss(y, v) for m, v in res.items()}
    best = min(briers, key=briers.get)
    return best, res[best], briers


def calib_slope_intercept(y, p):
    lr = LogisticRegression().fit(_logit(p), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def ece(y, p, n_bins=10):
    d = pd.DataFrame({"y": y, "p": p})
    d["b"] = pd.qcut(d.p.rank(method="first"), q=n_bins, labels=False)
    g = d.groupby("b").agg(n=("y", "size"), o=("y", "mean"), e=("p", "mean"))
    return float((g.n / g.n.sum() * (g.o - g.e).abs()).sum())


def threshold_for_sensitivity(y, p_cal, floor=0.75):
    """v1 rule: HIGHEST grid threshold whose sensitivity is still >= floor."""
    pos = np.sort(p_cal[y == 1])
    best = None
    for t in THRESHOLD_GRID:
        sens = 1.0 - np.searchsorted(pos, t, side="left") / len(pos)
        if sens >= floor:
            best = t
    return best if best is not None else THRESHOLD_GRID[0]


def confusion_metrics(y, flag):
    tp = int(((flag == 1) & (y == 1)).sum()); fp = int(((flag == 1) & (y == 0)).sum())
    tn = int(((flag == 0) & (y == 0)).sum()); fn = int(((flag == 0) & (y == 1)).sum())
    d = lambda a, b: a / b if b else float("nan")  # noqa: E731
    return dict(sensitivity=d(tp, tp + fn), specificity=d(tn, tn + fp), ppv=d(tp, tp + fp),
                npv=d(tn, tn + fn), flag_rate=(tp + fp) / len(y), tp=tp, fp=fp, tn=tn, fn=fn)


def flag_rate_at_sensitivity(y, p, floor=0.75):
    """Threshold-free ranking usefulness: share of people that must be flagged (taking the
    highest-scored first) to capture `floor` of all positives, and the PPV at that point."""
    order = np.argsort(-p, kind="stable")
    cum = np.cumsum(y[order])
    k = int(np.searchsorted(cum, np.ceil(floor * y.sum()), side="left")) + 1
    return k / len(y), float(cum[k - 1] / k)


def lift_table(y, p, fracs=(0.05, 0.10, 0.20)):
    order = np.argsort(-p, kind="stable")
    prev, rows = y.mean(), {}
    for f in fracs:
        k = int(round(f * len(y))); top = y[order[:k]]
        tag = f"top{int(f * 100)}"
        rows[f"{tag}_rate"] = float(top.mean())
        rows[f"{tag}_lift"] = float(top.mean() / prev)
        rows[f"{tag}_capture"] = float(top.sum() / y.sum())
    return rows


def band_masks(age):
    return {f"{lo}-{hi}": (age >= lo) & (age <= hi) for lo, hi in AGE_BANDS}


def evaluate_oof(y, p_raw, folds, age, floor=0.75):
    """Full OOF evaluation of one candidate. Returns (summary dict, p_cal, band DataFrame)."""
    method, p_cal, briers = choose_calibration(y, p_raw, folds)
    thr = threshold_for_sensitivity(y, p_cal, floor)
    cm = confusion_metrics(y, (p_cal >= thr).astype(int))
    slope, intercept = calib_slope_intercept(y, p_cal)
    fr, ppv_at = flag_rate_at_sensitivity(y, p_raw, floor)
    fold_auc = [roc_auc_score(y[folds == k], p_raw[folds == k]) for k in np.unique(folds)]
    s = dict(n=len(y), prevalence=float(y.mean()),
             roc_auc=float(roc_auc_score(y, p_raw)), roc_auc_fold_sd=float(np.std(fold_auc, ddof=1)),
             pr_auc=float(average_precision_score(y, p_raw)),
             calibration=method, brier_cal=float(briers[method]), cal_slope=slope, cal_intercept=intercept,
             ece=ece(y, p_cal), threshold=thr, **cm,
             flag_rate_at_sens=fr, ppv_at_sens=ppv_at, sens_floor=floor)
    s.update(lift_table(y, p_raw))
    rows = []
    for name, m in band_masks(age).items():
        yb, pb, cb = y[m], p_raw[m], p_cal[m]
        c = confusion_metrics(yb, (cb >= thr).astype(int))
        sl, ic = calib_slope_intercept(yb, cb)
        rows.append(dict(band=name, n=int(m.sum()), prevalence=float(yb.mean()),
                         roc_auc=float(roc_auc_score(yb, pb)), pr_auc=float(average_precision_score(yb, pb)),
                         sensitivity=c["sensitivity"], specificity=c["specificity"], ppv=c["ppv"],
                         flag_rate=c["flag_rate"], cal_slope=sl, cal_intercept=ic))
    return s, p_cal, pd.DataFrame(rows)


def age_band_thresholds(y, p_cal, age, floor=0.75):
    """Per-band threshold = v1 rule applied inside each band (OOF only). Returns (thr dict, summary,
    per-band DataFrame). Equalises sensitivity across bands; the cost shows up in flag rate."""
    thr, flag = {}, np.zeros(len(y), int)
    rows = []
    for name, m in band_masks(age).items():
        thr[name] = threshold_for_sensitivity(y[m], p_cal[m], floor)
        flag[m] = (p_cal[m] >= thr[name]).astype(int)
        c = confusion_metrics(y[m], flag[m])
        rows.append(dict(band=name, threshold=thr[name], n=int(m.sum()), **{k: c[k] for k in
                         ("sensitivity", "specificity", "ppv", "npv", "flag_rate")}))
    return thr, confusion_metrics(y, flag), pd.DataFrame(rows)


def psu_bootstrap_delta(y, p_new, p_ref, psu, n_boot=200, seed=42):
    """Paired cluster (PSU) bootstrap of AUC and AP differences, new - ref."""
    rng = np.random.default_rng(seed)
    order = np.argsort(psu, kind="stable")
    ids, starts = np.unique(psu[order], return_index=True)
    bounds = np.append(starts, len(psu))
    members = [order[bounds[i]:bounds[i + 1]] for i in range(len(ids))]
    d_auc, d_ap = [], []
    for _ in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(ids), len(ids))])
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        d_auc.append(roc_auc_score(yb, p_new[idx]) - roc_auc_score(yb, p_ref[idx]))
        d_ap.append(average_precision_score(yb, p_new[idx]) - average_precision_score(yb, p_ref[idx]))
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))  # noqa: E731
    return dict(d_auc=float(np.mean(d_auc)), d_auc_lo=q(d_auc)[0], d_auc_hi=q(d_auc)[1],
                d_ap=float(np.mean(d_ap)), d_ap_lo=q(d_ap)[0], d_ap_hi=q(d_ap)[1], n_boot=len(d_auc))


def fold_consistency(y, p_new, p_ref, folds):
    """Number of folds (of 5) in which the candidate beats the reference on AUC and on AP."""
    a = ap = 0
    for k in np.unique(folds):
        m = folds == k
        a += roc_auc_score(y[m], p_new[m]) > roc_auc_score(y[m], p_ref[m])
        ap += average_precision_score(y[m], p_new[m]) > average_precision_score(y[m], p_ref[m])
    return int(a), int(ap)
