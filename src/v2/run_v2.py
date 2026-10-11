"""
EarlyDia v2 - experiment runner (DEVELOPMENT DATA ONLY: v1 training split, PSU-grouped 5-fold CV).
Never reads the held-out test split. Results are appended to reports/v2/{sex}_experiments.csv.

  python -m src.v2.run_v2 batch   --sex women --group baseline
  python -m src.v2.run_v2 batch   --sex women --group state       # state, state_native
  python -m src.v2.run_v2 batch   --sex women --group wealth      # wealth_v190, wealth_v190a
  python -m src.v2.run_v2 batch   --sex women --group tune        # tune_B..E
  python -m src.v2.run_v2 batch   --sex women --group eng         # eng, compact, eng_compact
  python -m src.v2.run_v2 oof     --sex women --name my_combo --features state,eng --model hgb_C
  python -m src.v2.run_v2 blend   --sex women --names tune_C,lr_base --name blend_C_lr
  python -m src.v2.run_v2 compare --sex women --names state,tune_C --ref v1_repro --boot 200
  python -m src.v2.run_v2 bands   --sex women --name tune_C --floor 0.75
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse
import time
import numpy as np
import pandas as pd

from src.v2 import data_v2 as D
from src.v2 import evaluation_v2 as E
from src.v2.features_v2 import build_pipeline, parse_flags, prepare_X

EXPERIMENTS = {
    "v1_repro":     ("none", "hgb_default"),   # must reproduce v1 (AUC ~0.68 women OOF) - sanity anchor
    "lr_base":      ("none", "lr"),
    "state":        ("state", "hgb_default"),
    "state_native": ("state_native", "hgb_default"),
    "wealth_v190":  ("wealth_v190", "hgb_default"),
    "wealth_v190a": ("wealth_v190a", "hgb_default"),
    "tune_B":       ("none", "hgb_B"),
    "tune_C":       ("none", "hgb_C"),
    "tune_D":       ("none", "hgb_D"),
    "tune_E":       ("none", "hgb_E"),
    "eng":          ("eng", "hgb_default"),
    "compact":      ("compact", "hgb_default"),
    "eng_compact":  ("eng,compact", "hgb_default"),
    "rf_base":      ("state", "rf"),
    "xgb_default":  ("state", "xgb_default"),
    "xgb_A":        ("state", "xgb_A"),
    "xgb_B":        ("state", "xgb_B"),
    "xgb_C":        ("state", "xgb_C"),
    "xgb_opt_1":    ("state,eng", "xgb_opt_1"),
    "xgb_opt_2":    ("state,eng", "xgb_opt_2"),
    "xgb_opt_3":    ("state,eng", "xgb_opt_3"),
    "xgb_opt_4":    ("state,eng", "xgb_opt_4"),
    "xgb_opt_5":    ("state,eng", "xgb_opt_5"),
    "xgb_opt_6":    ("state,eng", "xgb_opt_6"),
    "xgb_opt_7":    ("state,eng", "xgb_opt_7"),
    "xgb_opt_8":    ("state,eng", "xgb_opt_8"),
    "xgb_opt_9":    ("state,eng", "xgb_opt_9"),
    "xgb_opt_10":   ("state,eng", "xgb_opt_10"),
}
GROUPS = {"baseline": ["v1_repro", "lr_base"], "state": ["state", "state_native"],
          "wealth": ["wealth_v190", "wealth_v190a"], "tune": ["tune_B", "tune_C", "tune_D", "tune_E"],
          "eng": ["eng", "compact", "eng_compact"],
          "xgb": ["xgb_default", "xgb_A", "xgb_B", "xgb_C", "xgb_eng"],
          "xgb_tune": [f"xgb_opt_{i}" for i in range(1, 11)],
          "rf": ["rf_base"]}
V1_LOCKED_OOF_REF = {  # configs/threshold_selection_summary.json (locked v1, for sanity comparison only)
    "women": dict(threshold=0.026, sensitivity=0.7548, specificity=0.4829, ppv=0.0464, flag_rate=0.5248),
    "men": dict(threshold=0.027, sensitivity=0.7623, specificity=0.4693, ppv=0.0522, flag_rate=0.5393)}
KEY_COLS = ["roc_auc", "pr_auc", "flag_rate_at_sens", "ppv_at_sens", "brier_cal", "cal_slope", "ece",
            "top10_rate", "top10_lift"]


def _paths(sex, name):
    D.V2_DATA_DIR.mkdir(parents=True, exist_ok=True)
    D.V2_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return D.V2_DATA_DIR / f"{sex}_oof_{name}.pkl", D.V2_REPORTS_DIR / f"{sex}_experiments.csv"


def _load_oof(sex, name):
    return pd.read_pickle(_paths(sex, name)[0])


def _save_row(sex, name, features, model, summary):
    _, csv = _paths(sex, name)
    row = pd.DataFrame([dict(name=name, features=features, model=model, **summary)])
    if csv.exists():
        old = pd.read_csv(csv)
        row = pd.concat([old[old["name"] != name], row], ignore_index=True)
    row.to_csv(csv, index=False)


def run_oof(sex, name, features, model, force=False):
    pkl, _ = _paths(sex, name)
    dev = D.load_dev(sex)
    folds = D.make_folds(dev, sex)
    y = dev["target_a"].astype(int).values
    flags = parse_flags(features)
    if pkl.exists() and not force:
        p = pd.read_pickle(pkl)["p_raw"].values
        print(f"[{sex}:{name}] loaded cached OOF ({pkl.name})")
    else:
        X = prepare_X(dev, sex, flags)
        p = np.full(len(dev), np.nan)
        for k in range(D.N_SPLITS):
            tr, va = folds != k, folds == k
            t0 = time.time()
            pipe = build_pipeline(sex, flags, model).fit(X[tr], y[tr])
            p[va] = pipe.predict_proba(X[va])[:, 1]
            print(f"[{sex}:{name}] fold {k} fitted in {time.time() - t0:.0f}s")
        assert not np.isnan(p).any()
        pd.DataFrame({"y": y, "psu": dev[D.GROUP_COL[sex]].values, "fold": folds,
                      "age": dev["age"].values, "p_raw": p}, index=dev.index).to_pickle(pkl)
    summary, p_cal, bands = E.evaluate_oof(y, p, folds, dev["age"].values)
    _save_row(sex, name, features, model, summary)
    return summary, bands


def _show(rows, cols):
    df = pd.DataFrame(rows).set_index("name")[cols]
    with pd.option_context("display.width", 220, "display.max_columns", 50):
        print(df.round(4).to_string())


def cmd_oof(a):
    s, bands = run_oof(a.sex, a.name, a.features, a.model, a.force)
    _show([dict(name=a.name, **s)], KEY_COLS)
    print(f"\nglobal threshold {s['threshold']} (rule: highest with OOF sens>=0.75): "
          f"sens={s['sensitivity']:.3f} spec={s['specificity']:.3f} ppv={s['ppv']:.4f} "
          f"npv={s['npv']:.4f} flag_rate={s['flag_rate']:.3f}")
    print("\nage bands at the global threshold:\n", bands.round(4).to_string(index=False))


def cmd_batch(a):
    rows = []
    for name in GROUPS[a.group]:
        f, m = EXPERIMENTS[name]
        s, _ = run_oof(a.sex, name, f, m, a.force)
        rows.append(dict(name=name, features=f, model=m, **s))
    _show(rows, ["features", "model"] + KEY_COLS)
    if a.group == "baseline":
        print("\nlocked v1 OOF reference (sanity check for v1_repro):", V1_LOCKED_OOF_REF[a.sex])


def cmd_blend(a):
    names = a.names.split(",")
    oofs = [_load_oof(a.sex, n) for n in names]
    base = oofs[0]
    for o in oofs[1:]:
        assert (o["fold"].values == base["fold"].values).all() and (o.index == base.index).all()
    z = np.mean([E._logit(o["p_raw"].values).ravel() for o in oofs], axis=0)
    p = 1 / (1 + np.exp(-z))
    out = base.copy(); out["p_raw"] = p
    pkl, _ = _paths(a.sex, a.name); out.to_pickle(pkl)
    s, _, _ = E.evaluate_oof(base["y"].values, p, base["fold"].values, base["age"].values)
    _save_row(a.sex, a.name, "blend:" + "+".join(names), "mean-logit", s)
    _show([dict(name=a.name, **s)], KEY_COLS)


def cmd_compare(a):
    ref = _load_oof(a.sex, a.ref)
    y, folds, psu = ref["y"].values, ref["fold"].values, ref["psu"].values
    rows = []
    for n in [a.ref] + a.names.split(","):
        o = _load_oof(a.sex, n)
        s, _, _ = E.evaluate_oof(y, o["p_raw"].values, folds, ref["age"].values)
        r = dict(name=n, **{k: s[k] for k in KEY_COLS})
        if n != a.ref:
            r["folds_auc_win"], r["folds_ap_win"] = E.fold_consistency(y, o["p_raw"].values, ref["p_raw"].values, folds)
            if a.boot:
                b = E.psu_bootstrap_delta(y, o["p_raw"].values, ref["p_raw"].values, psu, a.boot)
                r.update(d_auc=b["d_auc"], d_auc_ci=f"[{b['d_auc_lo']:+.4f},{b['d_auc_hi']:+.4f}]",
                         d_ap=b["d_ap"], d_ap_ci=f"[{b['d_ap_lo']:+.4f},{b['d_ap_hi']:+.4f}]")
        rows.append(r)
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(pd.DataFrame(rows).set_index("name").round(4).to_string())
    print("\nfold wins = # of 5 CV folds where the candidate beats the reference. A bootstrap CI that "
          "includes 0 means no demonstrated improvement.")


def cmd_bands(a):
    o = _load_oof(a.sex, a.name)
    y, folds, age = o["y"].values, o["fold"].values, o["age"].values
    s, p_cal, bands = E.evaluate_oof(y, o["p_raw"].values, folds, age, a.floor)
    thr, overall, per_band = E.age_band_thresholds(y, p_cal, age, a.floor)
    print(f"GLOBAL threshold {s['threshold']}: overall sens={s['sensitivity']:.3f} spec={s['specificity']:.3f} "
          f"ppv={s['ppv']:.4f} flag_rate={s['flag_rate']:.3f}")
    print(bands[["band", "n", "prevalence", "sensitivity", "specificity", "ppv", "flag_rate",
                 "cal_slope", "cal_intercept"]].round(4).to_string(index=False))
    print(f"\nAGE-BAND thresholds (per-band sens>={a.floor}): overall sens={overall['sensitivity']:.3f} "
          f"spec={overall['specificity']:.3f} ppv={overall['ppv']:.4f} flag_rate={overall['flag_rate']:.3f}")
    print(per_band.round(4).to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("oof", "batch", "blend", "compare", "bands"):
        p = sub.add_parser(c); p.add_argument("--sex", choices=["women", "men"], required=True)
        if c in ("oof", "batch"):
            p.add_argument("--force", action="store_true")
        if c == "oof":
            p.add_argument("--name", required=True); p.add_argument("--features", default="none")
            p.add_argument("--model", default="hgb_default")
        if c == "batch":
            p.add_argument("--group", choices=sorted(GROUPS), required=True)
        if c == "blend":
            p.add_argument("--names", required=True); p.add_argument("--name", required=True)
        if c == "compare":
            p.add_argument("--names", required=True); p.add_argument("--ref", default="v1_repro")
            p.add_argument("--boot", type=int, default=0)
        if c == "bands":
            p.add_argument("--name", required=True); p.add_argument("--floor", type=float, default=0.75)
    a = ap.parse_args(argv)
    {"oof": cmd_oof, "batch": cmd_batch, "blend": cmd_blend, "compare": cmd_compare, "bands": cmd_bands}[a.cmd](a)


if __name__ == "__main__":
    main()
