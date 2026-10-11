"""
EarlyDia - Stage: Grouped CV baseline model comparison (training data ONLY)
CV strategy: StratifiedGroupKFold(n_splits=5), grouped by PSU, stratified by target_a.
Preprocessing (imputation + encoding + scaling) lives INSIDE each fold's pipeline - fit only
on that fold's training portion, so there is no preprocessing leakage across folds.
NO threshold tuning, NO calibration, NO final-test-set access at this stage.
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from configs.config import (DATA_RAW_DIR, DATA_INTERIM_DIR, DATA_PROCESSED_DIR,
                            EXPLAINABILITY_REPORTS_DIR)
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, recall_score,
                              precision_score, f1_score, brier_score_loss, confusion_matrix)
from sklearn.calibration import calibration_curve

SEED = 42
N_SPLITS = 5

# ---- feature groupings (documented, no new features - just preprocessing routing) ----
NUM_MEDIAN_COMMON = ["age", "avg_systolic_bp", "avg_diastolic_bp",
                      "n_valid_systolic_readings", "n_valid_diastolic_readings"]
NUM_CONSTANT0 = ["alc_duration_years"]          # NaN is structurally 0 (non-drinker)
CAT_CONSTANT0 = ["alc_frequency"]               # NaN is structurally "non-drinker" category

WOMEN_TOBACCO = ["tob_v463a","tob_v463b","tob_v463c","tob_v463d","tob_v463f","tob_v463g",
                  "tob_v463j","tob_v463k","tob_v463l","tob_v463x","tob_v463z","tob_v463aa"]
MEN_TOBACCO = ["tob_mv463a","tob_mv463b","tob_mv463c","tob_mv463d","tob_mv463f","tob_mv463g",
               "tob_mv463x","tob_mv463z"]
WOMEN_ALC_TYPE = ["alc_s722a","alc_s722b","alc_s722c","alc_s722d","alc_s722e","alc_s722x"]
MEN_ALC_TYPE = ["alc_sm621a","alc_sm621b","alc_sm621c","alc_sm621d","alc_sm621e","alc_sm621x"]
WOMEN_DIET = [f"diet_s731{c}" for c in "abcdefghi"]
MEN_DIET = [f"diet_sm630{c}" for c in "abcdefghi"]
COMMON_CAT_BASE = ["alc_drinks", "education", "residence"]

# V3 features: physical activity, hypertension, fasting context, interactions
V3_NUM_MEDIAN = ["physical_activity", "hypertension_self_report",
                 "glucose_context_sb53", "glucose_context_sb54", "glucose_context_sb55", "glucose_context_sb73"]
V3_MEN_NUM_MEDIAN = ["physical_activity", "hypertension_self_report",
                     "glucose_context_smb53", "glucose_context_smb54", "glucose_context_smb55", "glucose_context_smb73"]
V3_INTERACTIONS = ["age_x_bmi", "bmi_x_systolic", "bmi_x_diastolic", "age_x_systolic", "age_x_diastolic"]
V3_MEN_INTERACTIONS = ["age_x_systolic", "age_x_diastolic"]

# V4 features: comorbidities, BP medication, anthropometric (waist/hip)
V4_NUM_MEDIAN = ["bp_medication", "waist_circumference", "hip_circumference",
                 "waist_hip_ratio", "n_comorbidities",
                 "comorbidity_respiratory", "comorbidity_thyroid",
                 "comorbidity_heart", "comorbidity_cancer", "comorbidity_kidney"]

# V5 features: BP aggregates, waist interactions, diet diversity, BMI categories
V5_NUM_MEDIAN = ["mean_arterial_pressure", "pulse_pressure",
                 "waist_x_age", "waist_x_bp_med", "comorbidity_x_waist",
                 "diet_diversity_score"]
V5_BMI_CATS = ["bmi_underweight", "bmi_normal", "bmi_overweight", "bmi_obese"]

# V6 features: waist-to-height ratio, tobacco chewing, age categories
V6_NUM_MEDIAN = ["waist_height_ratio", "tobacco_chewing"]
V6_AGE_CATS = ["age_18_24", "age_25_34", "age_35_40"]


def get_feature_lists(sex):
    if sex == "women":
        num_median = (NUM_MEDIAN_COMMON + ["bmi"] + V3_NUM_MEDIAN + V3_INTERACTIONS +
                      V4_NUM_MEDIAN + V5_NUM_MEDIAN + V5_BMI_CATS + V6_NUM_MEDIAN + V6_AGE_CATS)
        cat_no_missing = WOMEN_TOBACCO + WOMEN_ALC_TYPE + WOMEN_DIET + COMMON_CAT_BASE
    else:
        num_median = (NUM_MEDIAN_COMMON + V3_MEN_NUM_MEDIAN + V3_MEN_INTERACTIONS +
                      V4_NUM_MEDIAN + V5_NUM_MEDIAN + V6_AGE_CATS)
        cat_no_missing = MEN_TOBACCO + MEN_ALC_TYPE + MEN_DIET + COMMON_CAT_BASE
    return num_median, NUM_CONSTANT0, CAT_CONSTANT0, cat_no_missing


def build_preprocessor(sex):
    num_median, num_const0, cat_const0, cat_no_missing = get_feature_lists(sex)

    num_median_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    num_const0_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=0)),
        ("scale", StandardScaler()),
    ])
    cat_const0_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=0)),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),  # safety net; verified 0% missing
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    pre = ColumnTransformer([
        ("num_median", num_median_pipe, num_median),
        ("num_const0", num_const0_pipe, num_const0),
        ("cat_const0", cat_const0_pipe, cat_const0),
        ("cat", cat_pipe, cat_no_missing),
    ])
    all_feature_cols = num_median + num_const0 + cat_const0 + cat_no_missing
    return pre, all_feature_cols


def get_models():
    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=SEED),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=SEED),
    }


def compute_fold_metrics(y_true, y_proba):
    y_pred = (y_proba >= 0.5).astype(int)  # default threshold - NOT tuned at this stage
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)
    return dict(roc_auc=roc_auc, pr_auc=pr_auc, sensitivity=sensitivity, specificity=specificity,
                precision=precision, f1=f1, brier=brier)


def run_grouped_cv(train_df, sex, group_col, only_fold=None):
    preprocessor, feature_cols = build_preprocessor(sex)
    models = get_models()
    y = train_df["target_a"].astype(int).values
    groups = train_df[group_col].values
    X = train_df[feature_cols]

    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    results = {name: [] for name in models}
    calib_info = {name: [] for name in models}

    for fold_i, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups)):
        if only_fold is not None and fold_i != only_fold:
            continue
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # sanity: no group overlap within this fold
        assert set(groups[tr_idx]) & set(groups[va_idx]) == set(), "PSU leakage within CV fold!"

        for name, model in models.items():
            pipe = Pipeline([("pre", preprocessor), ("clf", model)])
            t0 = time.time()
            pipe.fit(X_tr, y_tr)
            y_proba = pipe.predict_proba(X_va)[:, 1]
            dt = time.time() - t0
            m = compute_fold_metrics(y_va, y_proba)
            m["fold"] = fold_i
            m["fit_seconds"] = dt
            results[name].append(m)

            frac_pos, mean_pred = calibration_curve(y_va, y_proba, n_bins=5, strategy="quantile")
            calib_info[name].append({"fold": fold_i, "mean_predicted": mean_pred.tolist(),
                                      "frac_positive": frac_pos.tolist()})
            print(f"  [{sex}] fold {fold_i} {name}: ROC-AUC={m['roc_auc']:.4f} "
                  f"PR-AUC={m['pr_auc']:.4f} sens={m['sensitivity']:.3f} spec={m['specificity']:.3f} "
                  f"prec={m['precision']:.3f} f1={m['f1']:.3f} brier={m['brier']:.4f} ({dt:.1f}s)")

    return results, calib_info


def summarize(results):
    rows = []
    for name, fold_list in results.items():
        df = pd.DataFrame(fold_list)
        row = {"model": name}
        for metric in ["roc_auc", "pr_auc", "sensitivity", "specificity", "precision", "f1", "brier"]:
            row[f"{metric}_mean"] = df[metric].mean()
            row[f"{metric}_std"] = df[metric].std()
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    women_train = pd.read_pickle(str(DATA_PROCESSED_DIR) + "/women_train.pkl")
    men_train = pd.read_pickle(str(DATA_PROCESSED_DIR) + "/men_train.pkl")

    print("=" * 70)
    print("WOMEN grouped CV (StratifiedGroupKFold, n_splits=5, group=v021_psu)")
    print("=" * 70)
    w_results, w_calib = run_grouped_cv(women_train, "women", "v021_psu")
    w_summary = summarize(w_results)
    print()
    print(w_summary.to_string(index=False))

    print()
    print("=" * 70)
    print("MEN grouped CV (StratifiedGroupKFold, n_splits=5, group=mv021_psu)")
    print("=" * 70)
    m_results, m_calib = run_grouped_cv(men_train, "men", "mv021_psu")
    m_summary = summarize(m_results)
    print()
    print(m_summary.to_string(index=False))

    w_summary.to_csv(str(DATA_PROCESSED_DIR) + "/women_cv_results.csv", index=False)
    m_summary.to_csv(str(DATA_PROCESSED_DIR) + "/men_cv_results.csv", index=False)

    with open(str(DATA_PROCESSED_DIR) + "/women_cv_fold_details.json", "w") as f:
        json.dump({"results": w_results, "calibration": w_calib}, f, indent=2, default=str)
    with open(str(DATA_PROCESSED_DIR) + "/men_cv_fold_details.json", "w") as f:
        json.dump({"results": m_results, "calibration": m_calib}, f, indent=2, default=str)

    config = {
        "seed": SEED, "cv_strategy": "StratifiedGroupKFold", "n_splits": N_SPLITS,
        "group_col_women": "v021_psu", "group_col_men": "mv021_psu",
        "threshold_used_for_sens_spec_precision_f1": 0.5,
        "note": "Default 0.5 threshold used ONLY for baseline comparison metrics; "
                "no threshold tuning performed at this stage.",
    }
    with open(str(DATA_PROCESSED_DIR) + "/cv_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\nSaved: women_cv_results.csv, men_cv_results.csv, fold-level JSON details, cv_config.json")
