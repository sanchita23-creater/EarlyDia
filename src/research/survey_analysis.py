"""Survey-weighted DESCRIPTIVE estimates of Target A prevalence (research-only).

Label: 'Survey-weighted descriptive estimate' - NOT a model prediction. Weights (v005/mv005, /1e6),
PSU and strata are used here only; they are never fed to the production model.
CI: Taylor-linearised ratio estimator with strata/PSU (with-replacement), lonely PSUs contribute 0.
Requires data/processed/{women,men}_target_a.pkl (not distributed). Without them, run() records status.
"""
import csv

import numpy as np
import pandas as pd

from src.research.common import OUT, DATA_PROCESSED, ensure_dirs, write_json

COLS = {"women": dict(w="v005", psu="v021", strat="v022", age="v012", res="v025", edu="v106", state="v024", bmi="v445"),
        "men": dict(w="mv005", psu="mv021", strat="mv022", age="mv012", res="mv025", edu="mv106", state="mv024", bmi=None)}


def weighted_prevalence(y, w, psu, strata, z=1.96):
    """Ratio estimator p = sum(w*y)/sum(w) with linearised variance. Returns (p, lo, hi, n)."""
    y = np.asarray(y, float); w = np.asarray(w, float)
    df = pd.DataFrame(dict(y=y, w=w, psu=np.asarray(psu), s=np.asarray(strata)))
    W = df.w.sum(); p = (df.w * df.y).sum() / W
    df["u"] = df.w * (df.y - p) / W
    var = 0.0
    for _, g in df.groupby("s"):
        tot = g.groupby("psu").u.sum()
        m = len(tot)
        if m > 1:
            var += m / (m - 1) * ((tot - tot.mean()) ** 2).sum()
    se = float(np.sqrt(var))
    return float(p), max(0.0, float(p - z * se)), min(1.0, float(p + z * se)), int(len(df))


def _groups(d, c, sex):
    age = d[c["age"]]
    out = {"overall": pd.Series("all", index=d.index),
           "age_group": pd.cut(age, [17, 24, 29, 34, 40], labels=["18-24", "25-29", "30-34", "35-40"]).astype(str),
           "residence": d[c["res"]].map({1: "urban", 2: "rural"}).fillna("missing"),
           "education": d[c["edu"]].astype(str)}
    if c["state"] in d:
        out["state_code"] = d[c["state"]].astype(str)
    if sex == "women":
        bmi = d[c["bmi"]].where(d[c["bmi"]] != 9998) / 100.0
        out["bmi_group"] = pd.cut(bmi, [0, 18.5, 25, 30, 100], labels=["<18.5", "18.5-24.9", "25-29.9", ">=30"],
                                  right=False).astype(str)
    return out


def run():
    ensure_dirs(); rows = []
    for sex, c in COLS.items():
        path = DATA_PROCESSED / f"{sex}_target_a.pkl"
        if not path.exists():
            write_json(OUT / "survey_weighted_prevalence.json",
                       {"status": "NOT_RUN", "reason": f"{path.name} not present (NFHS microdata is not distributed)."})
            return None
        d = pd.read_pickle(path)
        d = d[d.target_a.isin([0, 1]) & d.exclusion_reason_final.isna()]
        for gname, g in _groups(d, c, sex).items():
            for level in sorted(g.unique()):
                m = g == level
                if m.sum() < 30:
                    continue
                p, lo, hi, n = weighted_prevalence(d.loc[m, "target_a"], d.loc[m, c["w"]] / 1e6, d.loc[m, c["psu"]], d.loc[m, c["strat"]])
                rows.append(dict(sex=sex, grouping=gname, level=level, unweighted_n=n,
                                 unweighted_prevalence=float(d.loc[m, "target_a"].mean()),
                                 weighted_prevalence=p, ci95_low=lo, ci95_high=hi,
                                 label="Survey-weighted descriptive estimate (not a model prediction)"))
    with open(OUT / "survey_weighted_prevalence.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    write_json(OUT / "survey_weighted_prevalence.json", {"status": "OK", "rows": rows})
    return rows
