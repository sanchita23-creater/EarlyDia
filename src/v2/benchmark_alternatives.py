"""
EarlyDia v2 - benchmark alternative models (CatBoost, LightGBM, ExtraTrees) on DEVELOPMENT data.

Uses the same PSU-grouped 5-fold CV, same features, same evaluation as run_v2.py.
Never reads the held-out test split.

  python -m src.v2.benchmark_alternatives run    --sex women --group catboost
  python -m src.v2.benchmark_alternatives run    --sex women --group lgbm
  python -m src.v2.benchmark_alternatives run    --sex women --group extra_trees
  python -m src.v2.benchmark_alternatives run    --sex women --group all
  python -m src.v2.benchmark_alternatives blend  --sex women --names cb_A,lgbm_A --name blend_cb_lgbm
  python -m src.v2.benchmark_alternatives summary --sex women
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
from src.v2.features_v2 import build_preprocessor_v2, parse_flags, prepare_X

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

CATBOOST_CONFIGS = {
    "cb_default": dict(depth=6, learning_rate=0.05, iterations=300,
                       auto_class_weights="Balanced", random_seed=42, verbose=0),
    "cb_A": dict(depth=4, learning_rate=0.05, iterations=300,
                 auto_class_weights="Balanced", random_seed=42, verbose=0,
                 l2_leaf_reg=5.0, min_data_in_leaf=200),
    "cb_B": dict(depth=6, learning_rate=0.03, iterations=500,
                 auto_class_weights="Balanced", random_seed=42, verbose=0,
                 l2_leaf_reg=10.0, min_data_in_leaf=300, subsample=0.8),
    "cb_C": dict(depth=4, learning_rate=0.03, iterations=500,
                 auto_class_weights="Balanced", random_seed=42, verbose=0,
                 l2_leaf_reg=3.0, min_data_in_leaf=100, border_count=128),
    "cb_D": dict(depth=8, learning_rate=0.05, iterations=200,
                 auto_class_weights="Balanced", random_seed=42, verbose=0,
                 l2_leaf_reg=1.0, min_data_in_leaf=50, random_strength=2.0),
}

LIGHTGBM_CONFIGS = {
    "lgbm_default": dict(n_estimators=300, learning_rate=0.05, max_depth=6,
                          num_leaves=31, min_child_samples=200, subsample=0.8,
                          colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=5.0,
                          is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1),
    "lgbm_A": dict(n_estimators=300, learning_rate=0.05, max_depth=4,
                    num_leaves=15, min_child_samples=300, subsample=0.8,
                    colsample_bytree=0.8, reg_alpha=5.0, reg_lambda=10.0,
                    is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1),
    "lgbm_B": dict(n_estimators=500, learning_rate=0.03, max_depth=6,
                    num_leaves=31, min_child_samples=200, subsample=0.7,
                    colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=5.0,
                    is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1),
    "lgbm_C": dict(n_estimators=500, learning_rate=0.03, max_depth=4,
                    num_leaves=15, min_child_samples=100, subsample=0.8,
                    colsample_bytree=0.8, reg_alpha=2.0, reg_lambda=3.0,
                    is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1),
    "lgbm_D": dict(n_estimators=200, learning_rate=0.05, max_depth=8,
                    num_leaves=63, min_child_samples=50, subsample=0.8,
                    colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=1.0,
                    is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1),
}

XGBOOST_CONFIGS = {
    "xgb_default": dict(n_estimators=300, learning_rate=0.05, max_depth=6,
                         min_child_weight=200, subsample=0.8, colsample_bytree=0.8,
                         reg_alpha=1.0, reg_lambda=5.0, scale_pos_weight=1.0,
                         random_state=42, n_jobs=-1, verbosity=0,
                         eval_metric="logloss"),
    "xgb_A": dict(n_estimators=300, learning_rate=0.05, max_depth=4,
                   min_child_weight=300, subsample=0.8, colsample_bytree=0.8,
                   reg_alpha=5.0, reg_lambda=10.0, scale_pos_weight=1.0,
                   random_state=42, n_jobs=-1, verbosity=0,
                   eval_metric="logloss"),
    "xgb_B": dict(n_estimators=500, learning_rate=0.03, max_depth=6,
                   min_child_weight=200, subsample=0.7, colsample_bytree=0.7,
                   reg_alpha=0.5, reg_lambda=5.0, scale_pos_weight=1.0,
                   random_state=42, n_jobs=-1, verbosity=0,
                   eval_metric="logloss"),
    "xgb_C": dict(n_estimators=500, learning_rate=0.03, max_depth=4,
                   min_child_weight=100, subsample=0.8, colsample_bytree=0.8,
                   reg_alpha=2.0, reg_lambda=3.0, scale_pos_weight=1.0,
                   random_state=42, n_jobs=-1, verbosity=0,
                   eval_metric="logloss"),
    "xgb_D": dict(n_estimators=200, learning_rate=0.05, max_depth=8,
                   min_child_weight=50, subsample=0.8, colsample_bytree=0.6,
                   reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=1.0,
                   random_state=42, n_jobs=-1, verbosity=0,
                   eval_metric="logloss"),
}

EXTRA_TREES_CONFIGS = {
    "et_default": dict(n_estimators=500, max_depth=None, min_samples_leaf=10,
                       class_weight="balanced_subsample", random_state=42, n_jobs=-1),
    "et_A": dict(n_estimators=500, max_depth=15, min_samples_leaf=20,
                 class_weight="balanced_subsample", random_state=42, n_jobs=-1,
                 max_features="sqrt"),
    "et_B": dict(n_estimators=300, max_depth=20, min_samples_leaf=50,
                 class_weight="balanced_subsample", random_state=42, n_jobs=-1),
}


def _build_clf(model_name):
    if model_name in CATBOOST_CONFIGS:
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**CATBOOST_CONFIGS[model_name])
    if model_name in LIGHTGBM_CONFIGS:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**LIGHTGBM_CONFIGS[model_name])
    if model_name in XGBOOST_CONFIGS:
        from xgboost import XGBClassifier
        return XGBClassifier(**XGBOOST_CONFIGS[model_name])
    if model_name in EXTRA_TREES_CONFIGS:
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(**EXTRA_TREES_CONFIGS[model_name])
    raise ValueError(f"unknown benchmark model: {model_name}")


def _build_pipeline(sex, flags, model_name):
    from sklearn.pipeline import Pipeline
    pre = build_preprocessor_v2(sex, flags)
    clf = _build_clf(model_name)
    return Pipeline([("pre", pre), ("clf", clf)])


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------

EXPERIMENTS = {}
for name in CATBOOST_CONFIGS:
    EXPERIMENTS[name] = ("state", name)
for name in LIGHTGBM_CONFIGS:
    EXPERIMENTS[name] = ("state", name)
for name in XGBOOST_CONFIGS:
    EXPERIMENTS[name] = ("state", name)
for name in EXTRA_TREES_CONFIGS:
    EXPERIMENTS[name] = ("state", name)

EXPERIMENTS.update({
    "cb_state_native":   ("state_native", "cb_default"),
    "cb_state_eng":      ("state,eng", "cb_default"),
    "cb_state_wealth":   ("state,wealth_v190", "cb_default"),
    "cb_A_state_eng":    ("state,eng", "cb_A"),
    "cb_B_state_eng":    ("state,eng", "cb_B"),
    "cb_C_state_eng":    ("state,eng", "cb_C"),
    "cb_D_state_eng":    ("state,eng", "cb_D"),
    "lgbm_state_eng":    ("state,eng", "lgbm_default"),
    "lgbm_A_state_eng":  ("state,eng", "lgbm_A"),
    "lgbm_B_state_eng":  ("state,eng", "lgbm_B"),
    "lgbm_C_state_eng":  ("state,eng", "lgbm_C"),
    "lgbm_D_state_eng":  ("state,eng", "lgbm_D"),
    "et_state":          ("state", "et_default"),
    "et_A_state":        ("state", "et_A"),
    "et_B_state":        ("state", "et_B"),
})

GROUPS = {
    "catboost": ["cb_default", "cb_A", "cb_B", "cb_C", "cb_D"],
    "lgbm": ["lgbm_default", "lgbm_A", "lgbm_B", "lgbm_C", "lgbm_D"],
    "xgboost": ["xgb_default", "xgb_A", "xgb_B", "xgb_C", "xgb_D"],
    "extra_trees": ["et_default", "et_A", "et_B"],
    "catboost_ext": ["cb_state_native", "cb_state_eng", "cb_state_wealth",
                     "cb_A_state_eng", "cb_B_state_eng", "cb_C_state_eng", "cb_D_state_eng"],
    "lgbm_ext": ["lgbm_state_eng", "lgbm_A_state_eng", "lgbm_B_state_eng",
                 "lgbm_C_state_eng", "lgbm_D_state_eng"],
    "extra_trees_ext": ["et_state", "et_A_state", "et_B_state"],
    "all": (["cb_default", "cb_A", "cb_B", "cb_C", "cb_D"]
            + ["lgbm_default", "lgbm_A", "lgbm_B", "lgbm_C", "lgbm_D"]
            + ["xgb_default", "xgb_A", "xgb_B", "xgb_C", "xgb_D"]
            + ["et_default", "et_A", "et_B"]),
}

REPORT_CSV = "benchmark_alternatives.csv"
KEY_COLS = ["roc_auc", "roc_auc_fold_sd", "pr_auc", "flag_rate_at_sens", "ppv_at_sens",
            "brier_cal", "cal_slope", "ece", "top5_lift", "top10_lift", "top20_capture"]


def _paths(sex, name):
    D.V2_DATA_DIR.mkdir(parents=True, exist_ok=True)
    D.V2_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return (D.V2_DATA_DIR / f"{sex}_oof_bench_{name}.pkl",
            D.V2_REPORTS_DIR / REPORT_CSV)


def run_oof(sex, name, features, model, force=False):
    pkl, _ = _paths(sex, name)
    dev = D.load_dev(sex)
    folds = D.make_folds(dev, sex)
    y = dev["target_a"].astype(int).values
    flags = parse_flags(features)

    if pkl.exists() and not force:
        oof = pd.read_pickle(pkl)
        p = oof["p_raw"].values
        print(f"[{sex}:{name}] loaded cached OOF ({pkl.name})")
    else:
        X = prepare_X(dev, sex, flags)
        p = np.full(len(dev), np.nan)
        for k in range(D.N_SPLITS):
            tr, va = folds != k, folds == k
            t0 = time.time()
            pipe = _build_pipeline(sex, flags, model).fit(X.iloc[tr] if hasattr(X, "iloc") else X[tr],
                                                           y[tr])
            p[va] = pipe.predict_proba(X.iloc[va] if hasattr(X, "iloc") else X[va])[:, 1]
            elapsed = time.time() - t0
            print(f"[{sex}:{name}] fold {k} fitted in {elapsed:.0f}s")
        assert not np.isnan(p).any(), f"{name}: NaN in OOF predictions"
        pd.DataFrame({"y": y, "psu": dev[D.GROUP_COL[sex]].values, "fold": folds,
                      "age": dev["age"].values, "p_raw": p}, index=dev.index).to_pickle(pkl)

    summary, p_cal, bands = E.evaluate_oof(y, p, folds, dev["age"].values)
    _save_row(sex, name, features, model, summary)
    return summary, bands


def _save_row(sex, name, features, model, summary):
    _, csv = _paths(sex, name)
    row = pd.DataFrame([dict(name=name, features=features, model=model, sex=sex, **summary)])
    if csv.exists():
        old = pd.read_csv(csv)
        old = old[~((old["name"] == name) & (old["sex"] == sex))]
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(csv, index=False)


def cmd_run(a):
    names = GROUPS[a.group]
    rows = []
    for name in names:
        features, model = EXPERIMENTS[name]
        s, _ = run_oof(a.sex, name, features, model, a.force)
        rows.append(dict(name=name, features=features, model=model, **s))
    df = pd.DataFrame(rows).set_index("name")[KEY_COLS]
    with pd.option_context("display.width", 260, "display.max_columns", 60):
        print(f"\n=== {a.sex.upper()} benchmark: {a.group} ===")
        print(df.round(4).to_string())
    best_idx = df["roc_auc"].idxmax()
    print(f"\nBest ROC-AUC: {best_idx} = {df.loc[best_idx, 'roc_auc']:.4f}")


def cmd_blend(a):
    names = a.names.split(",")
    oofs = []
    for n in names:
        pkl, _ = _paths(a.sex, n)
        if not pkl.exists():
            raise SystemExit(f"OOF for {n} not found at {pkl}. Run it first.")
        oofs.append(pd.read_pickle(pkl))
    base = oofs[0]
    for o in oofs[1:]:
        assert (o.index == base.index).all() and (o["fold"].values == base["fold"].values).all()
    z = np.mean([E._logit(o["p_raw"].values).ravel() for o in oofs], axis=0)
    p = 1 / (1 + np.exp(-z))
    out = base.copy()
    out["p_raw"] = p
    pkl_out, _ = _paths(a.sex, a.name)
    out.to_pickle(pkl_out)
    s, _, _ = E.evaluate_oof(base["y"].values, p, base["fold"].values, base["age"].values)
    _save_row(a.sex, a.name, "blend:" + "+".join(names), "mean-logit-bench", s)
    row = pd.DataFrame([dict(name=a.name, features="blend:" + "+".join(names),
                             model="mean-logit-bench", **s)])
    with pd.option_context("display.width", 260, "display.max_columns", 60):
        print(f"\n=== {a.sex.upper()} blend: {a.name} ===")
        print(row.set_index("name")[KEY_COLS].round(4).to_string())


def cmd_summary(a):
    _, csv = _paths(a.sex, "")
    if not csv.exists():
        raise SystemExit(f"No benchmark results yet: {csv}")
    df = pd.read_csv(csv)
    df = df[df["sex"] == a.sex].set_index("name")
    if df.empty:
        raise SystemExit(f"No results for sex={a.sex}")
    df = df.sort_values("roc_auc", ascending=False)
    show_cols = ["features", "model"] + KEY_COLS
    available = [c for c in show_cols if c in df.columns]
    with pd.option_context("display.width", 280, "display.max_columns", 60):
        print(f"\n=== {a.sex.upper()} BENCHMARK SUMMARY (sorted by ROC-AUC) ===")
        print(df[available].round(4).to_string())

    best = df.index[0]
    best_auc = df.loc[best, "roc_auc"]
    print(f"\nTop model: {best} (ROC-AUC = {best_auc:.4f})")

    v2_rows = pd.read_csv(D.V2_REPORTS_DIR / f"{a.sex}_experiments.csv")
    if not v2_rows.empty:
        v2_best = v2_rows.loc[v2_rows["roc_auc"].idxmax()]
        print(f"Previous v2 best: {v2_best['name']} (ROC-AUC = {v2_best['roc_auc']:.4f})")
        gap = best_auc - v2_best["roc_auc"]
        print(f"Gap to v2 best: {gap:+.4f}")

    target = 0.85
    print(f"\nTarget ROC-AUC: {target}")
    print(f"Gap from best benchmark model to target: {target - best_auc:.4f}")


def cmd_compare_v2(a):
    """Compare all benchmark models against the v2 reference via paired bootstrap."""
    _, csv = _paths(a.sex, "")
    if not csv.exists():
        raise SystemExit(f"No benchmark results: {csv}")
    bench = pd.read_csv(csv)
    bench = bench[bench["sex"] == a.sex].set_index("name")

    ref_name = a.ref
    ref_pkl, _ = _paths(a.sex, ref_name)
    if not ref_pkl.exists():
        ref_pkl_v2 = D.V2_DATA_DIR / f"{a.sex}_oof_{ref_name}.pkl"
        if ref_pkl_v2.exists():
            ref_pkl = ref_pkl_v2
        else:
            raise SystemExit(f"Reference OOF not found: {ref_name}")

    ref = pd.read_pickle(ref_pkl)
    y, folds, psu = ref["y"].values, ref["fold"].values, ref["psu"].values
    p_ref = ref["p_raw"].values

    rows = []
    for name in bench.index:
        pkl, _ = _paths(a.sex, name)
        if not pkl.exists():
            continue
        o = pd.read_pickle(pkl)
        p_new = o["p_raw"].values
        fold_wins_auc, fold_wins_ap = E.fold_consistency(y, p_new, p_ref, folds)
        boot = E.psu_bootstrap_delta(y, p_new, p_ref, psu, a.boot)
        auc = float(pd.read_csv(D.V2_REPORTS_DIR / REPORT_CSV)
                    .query(f"name == {name!r} and sex == {a.sex!r}")["roc_auc"].iloc[0])
        rows.append(dict(name=name, roc_auc=auc,
                         fold_wins_auc=f"{fold_wins_auc}/5",
                         fold_wins_ap=f"{fold_wins_ap}/5",
                         d_auc=f"{boot['d_auc']:+.4f}",
                         d_auc_ci=f"[{boot['d_auc_lo']:+.4f},{boot['d_auc_hi']:+.4f}]",
                         d_ap=f"{boot['d_ap']:+.4f}",
                         d_ap_ci=f"[{boot['d_ap_lo']:+.4f},{boot['d_ap_hi']:+.4f}]"))
    df = pd.DataFrame(rows).set_index("name").sort_values("roc_auc", ascending=False)
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(f"\n=== {a.sex.upper()}: benchmark models vs reference '{ref_name}' ===")
        print(df.to_string())
    print("\nCI includes 0 => no demonstrated improvement over reference.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--sex", choices=["women", "men"], required=True)
    p_run.add_argument("--group", choices=sorted(GROUPS), required=True)
    p_run.add_argument("--force", action="store_true")

    p_blend = sub.add_parser("blend")
    p_blend.add_argument("--sex", choices=["women", "men"], required=True)
    p_blend.add_argument("--names", required=True)
    p_blend.add_argument("--name", required=True)

    p_sum = sub.add_parser("summary")
    p_sum.add_argument("--sex", choices=["women", "men"], required=True)

    p_cmp = sub.add_parser("compare_v2")
    p_cmp.add_argument("--sex", choices=["women", "men"], required=True)
    p_cmp.add_argument("--ref", default="v1_repro")
    p_cmp.add_argument("--boot", type=int, default=200)

    a = ap.parse_args(argv)
    {"run": cmd_run, "blend": cmd_blend, "summary": cmd_summary,
     "compare_v2": cmd_compare_v2}[a.cmd](a)


if __name__ == "__main__":
    main()
