"""
EarlyDia v2 - FINAL one-shot evaluation on the held-out test split.

Run ONLY after a candidate has been chosen on development data (grouped-CV OOF) via run_v2.py.
  python -m src.v2.final_v2 --sex women --name tune_C --confirm-test-access

Safeguards
  * requires --confirm-test-access;
  * writes reports/v2/TEST_ACCESSED_{sex}.json and REFUSES a second run with a different candidate
    (no choosing among candidates on the test set);
  * the locked v1 artifact is loaded READ-ONLY (never re-saved) and is first checked to reproduce its
    own locked report, which also proves the test frame is the same one v1 used;
  * v2 refit data, calibrator and thresholds all come from the training split / its OOF predictions.
Outputs go to models_v2/ and reports/v2/ only.
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import argparse
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from configs.config import PRODUCTION_PIPELINE_PATHS, FINAL_EVAL_REPORTS_DIR
from configs.sklearn_compat import load_artifact
from src.v2 import data_v2 as D
from src.v2 import evaluation_v2 as E
from src.v2.features_v2 import build_pipeline, parse_flags, prepare_X


def _metrics(y, p_raw, p_cal, flag, age):
    cm = E.confusion_metrics(y, flag)
    slope, icpt = E.calib_slope_intercept(y, p_cal)
    out = dict(n=len(y), n_pos=int(y.sum()), prevalence=float(y.mean()),
               roc_auc=float(roc_auc_score(y, p_raw)), pr_auc=float(average_precision_score(y, p_raw)),
               brier=float(brier_score_loss(y, p_cal)), cal_slope=slope, cal_intercept=icpt,
               ece=E.ece(y, p_cal), **cm)
    out.update(E.lift_table(y, p_raw))
    bands = []
    for name, m in E.band_masks(age).items():
        c = E.confusion_metrics(y[m], flag[m])
        bands.append(dict(band=name, n=int(m.sum()), prevalence=float(y[m].mean()),
                          roc_auc=float(roc_auc_score(y[m], p_raw[m])),
                          pr_auc=float(average_precision_score(y[m], p_raw[m])),
                          **{k: c[k] for k in ("sensitivity", "specificity", "ppv", "npv", "flag_rate")}))
    return out, bands


def _apply(method, calibrator, p):
    if method == "isotonic":
        return calibrator.predict(p)
    from src.calibration.calibration_utils import sigmoid_calibrate_predict
    return sigmoid_calibrate_predict(calibrator, p)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sex", choices=["women", "men"], required=True)
    ap.add_argument("--name", required=True, help="candidate name in reports/v2/{sex}_experiments.csv")
    ap.add_argument("--floor", type=float, default=0.75)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--confirm-test-access", action="store_true")
    a = ap.parse_args(argv)
    if not a.confirm_test_access:
        raise SystemExit("Refusing: pass --confirm-test-access to open the held-out test split.")
    sex = a.sex
    D.V2_REPORTS_DIR.mkdir(parents=True, exist_ok=True); D.V2_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    lock = D.V2_REPORTS_DIR / f"TEST_ACCESSED_{sex}.json"
    if lock.exists() and json.loads(lock.read_text())["name"] != a.name:
        raise SystemExit(f"Test split already used for candidate {json.loads(lock.read_text())['name']!r}; "
                         f"selecting a different candidate after seeing test results is not allowed.")

    exp = pd.read_csv(D.V2_REPORTS_DIR / f"{sex}_experiments.csv").set_index("name").loc[a.name]
    if str(exp["model"]) == "mean-logit":
        raise SystemExit("Blend candidates are not supported by the final script yet.")
    flags, model = parse_flags(None if exp["features"] == "none" else exp["features"]), exp["model"]

    # ---- development side: everything below uses TRAIN rows / OOF only ----
    pred, tr_idx, te_idx = D._full(sex)
    dev = pred.iloc[tr_idx]
    oof = pd.read_pickle(D.V2_DATA_DIR / f"{sex}_oof_{a.name}.pkl")
    assert (oof.index == dev.index).all(), "cached OOF does not match the dev rows"
    y_dev, p_oof, folds, age_dev = oof["y"].values, oof["p_raw"].values, oof["fold"].values, oof["age"].values
    method, p_cal_oof, _ = E.choose_calibration(y_dev, p_oof, folds)
    thr = E.threshold_for_sensitivity(y_dev, p_cal_oof, a.floor)
    band_thr, _, _ = E.age_band_thresholds(y_dev, p_cal_oof, age_dev, a.floor)
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_oof, y_dev)
    else:
        from src.calibration.calibration_utils import sigmoid_calibrate_fit
        calibrator = sigmoid_calibrate_fit(p_oof, y_dev)
    X_dev = prepare_X(dev, sex, flags)
    pipe = build_pipeline(sex, flags, model).fit(X_dev, y_dev)
    artifact = dict(base_pipeline=pipe, feature_cols=list(X_dev.columns),
                    feature_flags=sorted(flags), model_name=model, calibration_method=method,
                    calibrator=calibrator, selected_threshold=thr, age_band_thresholds=band_thr,
                    sens_floor=a.floor, sex=sex, version="v2-candidate", candidate=a.name,
                    note="Screening/association model on cross-sectional NFHS-5 data; not diagnostic, not future-disease prediction.")
    with open(D.V2_MODELS_DIR / f"{sex}_v2_pipeline.pkl", "wb") as f:
        pickle.dump(artifact, f)

    # ---- the one and only test-split access ----
    lock.write_text(json.dumps({"name": a.name, "features": exp["features"], "model": model}))
    test = pred.iloc[te_idx]
    y, age, psu = test["target_a"].astype(int).values, test["age"].values, test[D.GROUP_COL[sex]].values
    p2_raw = pipe.predict_proba(prepare_X(test, sex, flags))[:, 1]
    p2_cal = _apply(method, calibrator, p2_raw)
    band_flag = np.zeros(len(y), int)
    for name, m in E.band_masks(age).items():
        band_flag[m] = (p2_cal[m] >= band_thr[name]).astype(int)

    v1 = load_artifact(PRODUCTION_PIPELINE_PATHS[sex])           # read-only
    p1_raw = v1["base_pipeline"].predict_proba(test[v1["feature_cols"]])[:, 1]
    p1_cal = _apply(v1["calibration_method"], v1["calibrator"], p1_raw)
    locked = json.loads((FINAL_EVAL_REPORTS_DIR / "final_test_evaluation.json").read_text())[sex]["overall"]
    assert abs(roc_auc_score(y, p1_cal) - locked["roc_auc"]) < 5e-4, \
        "v1 does not reproduce its locked test report - test frame/pipeline mismatch; stop."

    res = {}
    res["v1_locked_threshold"], b1 = _metrics(y, p1_raw, p1_cal, (p1_cal >= v1["selected_threshold"]).astype(int), age)
    res["v2_global_threshold"], b2 = _metrics(y, p2_raw, p2_cal, (p2_cal >= thr).astype(int), age)
    res["v2_age_band_thresholds"], b3 = _metrics(y, p2_raw, p2_cal, band_flag, age)
    boot = E.psu_bootstrap_delta(y, p2_raw, p1_raw, psu, a.boot)
    for k, (fr, ppv) in {"v1": E.flag_rate_at_sensitivity(y, p1_raw, a.floor),
                         "v2": E.flag_rate_at_sensitivity(y, p2_raw, a.floor)}.items():
        res[f"{k}_flag_rate_at_{int(a.floor*100)}pct_sens_DESCRIPTIVE"] = dict(flag_rate=fr, ppv=ppv)
    report = dict(sex=sex, candidate=a.name, features=exp["features"], model=model, calibration=method,
                  v2_global_threshold=thr, v2_age_band_thresholds=band_thr, v1_threshold=v1["selected_threshold"],
                  metrics=res, paired_psu_bootstrap_v2_minus_v1=boot,
                  age_bands=dict(v1=b1, v2_global=b2, v2_band=b3),
                  interpretation="Screening/association model; no claim of diagnosis or future-disease prediction.")
    (D.V2_REPORTS_DIR / f"final_{sex}_{a.name}.json").write_text(json.dumps(report, indent=2, default=float))

    cols = ["roc_auc", "pr_auc", "sensitivity", "specificity", "ppv", "npv", "flag_rate", "brier", "cal_slope",
            "cal_intercept", "ece", "top5_lift", "top10_lift", "top10_rate", "top20_capture"]
    tab = pd.DataFrame({k: {c: v[c] for c in cols} for k, v in res.items() if isinstance(v, dict) and "roc_auc" in v}).T
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(f"\n=== {sex.upper()} held-out TEST: v1 (locked) vs v2 candidate '{a.name}' ===")
        print(tab.round(4).to_string())
        print("\npaired PSU bootstrap (v2 - v1):", {k: round(v, 4) for k, v in boot.items()})
        for lab, b in (("v1", b1), ("v2 global thr", b2), ("v2 band thr", b3)):
            print(f"\nage bands - {lab}\n", pd.DataFrame(b).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
