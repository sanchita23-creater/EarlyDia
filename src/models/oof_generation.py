"""
EarlyDia - Stage: Grouped OOF probability generation for candidate models (LR, HGB)
Same StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) as the baseline CV stage.
Each row's OOF probability comes from a fold that did NOT train on that row.
Random Forest excluded from OOF/calibration (documented as clearly weaker in baseline grouped CV
on both ROC-AUC and PR-AUC, and its 0.5-threshold collapse is not the basis for this exclusion -
its threshold-independent metrics were the weaker ones).
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from configs.config import (DATA_RAW_DIR, DATA_INTERIM_DIR, DATA_PROCESSED_DIR,
                            EXPLAINABILITY_REPORTS_DIR)
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from src.models.grouped_cv_baseline import build_preprocessor, SEED, N_SPLITS
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier


def get_candidate_models():
    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=SEED),
    }


def generate_oof(train_df, sex, group_col):
    preprocessor, feature_cols = build_preprocessor(sex)
    models = get_candidate_models()
    y = train_df["target_a"].astype(int).values
    groups = train_df[group_col].values
    X = train_df[feature_cols]
    n = len(train_df)

    oof = {name: np.full(n, np.nan) for name in models}
    fold_assignment = np.full(n, -1)

    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    for fold_i, (tr_idx, va_idx) in enumerate(cv.split(X, y, groups)):
        assert set(groups[tr_idx]) & set(groups[va_idx]) == set(), \
            f"PSU leakage within CV fold {fold_i}!"
        fold_assignment[va_idx] = fold_i
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr = y[tr_idx]

        for name, model in models.items():
            pipe = Pipeline([("pre", preprocessor), ("clf", model)])
            pipe.fit(X_tr, y_tr)
            oof[name][va_idx] = pipe.predict_proba(X_va)[:, 1]
        print(f"  [{sex}] fold {fold_i} done")

    assert (fold_assignment >= 0).all(), "some rows never assigned to a validation fold!"
    for name in models:
        assert not np.isnan(oof[name]).any(), f"some OOF predictions missing for {name}!"

    result = pd.DataFrame({
        "target_a": y,
        group_col: groups,
        "cv_fold": fold_assignment,
    }, index=train_df.index)
    for name in models:
        result[f"oof_{name}"] = oof[name]
    return result


if __name__ == "__main__":
    import sys
    sex = sys.argv[1] if len(sys.argv) > 1 else "both"

    if sex in ("women", "both"):
        women_train = pd.read_pickle(str(DATA_PROCESSED_DIR) + "/women_train.pkl")
        w_oof = generate_oof(women_train, "women", "v021_psu")
        w_oof.to_pickle(str(DATA_PROCESSED_DIR) + "/women_oof.pkl")
        print("saved women_oof.pkl", w_oof.shape)

    if sex in ("men", "both"):
        men_train = pd.read_pickle(str(DATA_PROCESSED_DIR) + "/men_train.pkl")
        m_oof = generate_oof(men_train, "men", "mv021_psu")
        m_oof.to_pickle(str(DATA_PROCESSED_DIR) + "/men_oof.pkl")
        print("saved men_oof.pkl", m_oof.shape)
