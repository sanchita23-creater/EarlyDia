"""
EarlyDia - Explainability stage.
IMPORTANT ENVIRONMENT LIMITATION: the `shap` package is not installed and this session has
no network access to install it. True SHAP TreeExplainer output (per-instance attributions,
summary/beeswarm plots) could NOT be produced. As a documented, clearly-labeled substitute,
this script computes sklearn's model-agnostic PERMUTATION IMPORTANCE for the selected HGB
production pipelines, plus Logistic Regression coefficients as the interpretable reference
requested. Neither of these is SHAP and neither is presented as SHAP anywhere in outputs.

Computed on a sample of TRAINING data (not the locked final test set), consistent with
"do not use the final test set to modify, select, or tune anything" - explainability
artifacts are descriptive outputs, and using training data avoids any ambiguity about
touching the locked test set again.

Wording discipline: all outputs use "predictive contribution" / "association" /
"model importance" language. No feature is described as causing diabetes-range status.
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from configs.config import (DATA_RAW_DIR, DATA_INTERIM_DIR, DATA_PROCESSED_DIR,
                            EXPLAINABILITY_REPORTS_DIR)
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.models.grouped_cv_baseline import build_preprocessor, SEED

SAMPLE_SIZE = 25000  # for permutation importance compute-time reasons; documented, not the full 347k rows
N_REPEATS = 10


def get_ohe_feature_names(preprocessor):
    """Recover human-readable feature names after ColumnTransformer + OneHotEncoder."""
    return preprocessor.get_feature_names_out()


def compute_permutation_importance(artifact, train_df, sex, sample_size=SAMPLE_SIZE, seed=SEED):
    feature_cols = artifact["feature_cols"]
    rng = np.random.RandomState(seed)
    if len(train_df) > sample_size:
        sample_idx = rng.choice(train_df.index, size=sample_size, replace=False)
        sample = train_df.loc[sample_idx]
    else:
        sample = train_df

    X = sample[feature_cols]
    y = sample["target_a"].astype(int)

    pipe = artifact["base_pipeline"]  # already fit on FULL training data - NOT refit here
    result = permutation_importance(pipe, X, y, scoring="roc_auc", n_repeats=N_REPEATS,
                                     random_state=seed, n_jobs=-1)

    df = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    print(f"[{sex}] permutation importance computed on {len(sample)} training rows "
          f"({N_REPEATS} repeats, scoring=ROC-AUC)")
    return df


def fit_lr_reference(train_df, sex):
    """LR fit on FULL training data - interpretable reference only, NOT the production model."""
    preprocessor, feature_cols = build_preprocessor(sex)
    lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)
    pipe = Pipeline([("pre", preprocessor), ("clf", lr)])
    X = train_df[feature_cols]
    y = train_df["target_a"].astype(int)
    pipe.fit(X, y)

    feature_names = get_ohe_feature_names(pipe.named_steps["pre"])
    coefs = pipe.named_steps["clf"].coef_[0]
    odds_ratios = np.exp(coefs)

    df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefs,
        "odds_ratio": odds_ratios,
    }).sort_values("coefficient", key=abs, ascending=False).reset_index(drop=True)
    print(f"[{sex}] Logistic Regression reference fit on full training data "
          f"({len(feature_names)} encoded features)")
    return df


if __name__ == "__main__":
    for sex, group_col in [("women", "v021_psu"), ("men", "mv021_psu")]:
        train = pd.read_pickle(str(DATA_PROCESSED_DIR) + f"/{sex}_train.pkl")
        with open(str(DATA_PROCESSED_DIR) + f"/{sex}_production_pipeline.pkl", "rb") as f:
            artifact = pickle.load(f)

        perm_df = compute_permutation_importance(artifact, train, sex)
        perm_df.to_csv(str(EXPLAINABILITY_REPORTS_DIR) + f"/{sex}_permutation_importance.csv", index=False)

        print(f"\n[{sex}] TOP 15 features by permutation importance (model importance, "
              f"NOT causal, NOT SHAP):")
        print(perm_df.head(15).to_string(index=False))

        lr_df = fit_lr_reference(train, sex)
        lr_df.to_csv(str(EXPLAINABILITY_REPORTS_DIR) + f"/{sex}_lr_coefficients.csv", index=False)
        print(f"\n[{sex}] TOP 15 |coefficient| LR reference (associations, NOT causal):")
        print(lr_df.head(15).to_string(index=False))
        print()
