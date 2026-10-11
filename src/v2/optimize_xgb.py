"""
EarlyDia - Bounded XGBoost Optimization (Stage 6)

Performs a randomized hyperparameter search on the development data, using
the exact same PSU-grouped stratified cross-validation (5-fold) to ensure
no data leakage and a fair comparison.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, roc_auc_score, average_precision_score
import xgboost as xgb

from src.v2 import data_v2 as D
from src.v2.features_v2 import build_preprocessor_v2, SEED

def optimize_xgboost(sex, n_iter, flags):
    print(f"Loading development data for {sex}...")
    pred, tr_idx, te_idx = D._full(sex)
    dev = pred.iloc[tr_idx]
    
    y = dev["target_a"].astype(int).values
    groups = dev[D.GROUP_COL[sex]].values
    
    pre = build_preprocessor_v2(sex, flags)
    
    clf = xgb.XGBClassifier(
        random_state=SEED,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    
    # Bounded parameter space for optimization
    param_distributions = {
        "clf__n_estimators": [100, 200, 300, 400],
        "clf__max_depth": [3, 4, 5, 6],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "clf__min_child_weight": [1, 5, 10, 20, 50],
        "clf__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "clf__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "clf__reg_alpha": [0, 0.5, 1, 2, 5],
        "clf__reg_lambda": [1, 5, 10, 20],
        "clf__gamma": [0, 0.1, 0.5, 1.0],
        # Approximate 1/prevalence weights, plus unweighted
        "clf__scale_pos_weight": [1.0, 10.0, 15.0, 20.0, 25.0, 30.0],
    }
    
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision"
    }
    
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        refit="roc_auc", # Primary optimization objective
        random_state=SEED,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    from src.v2.features_v2 import prepare_X
    X = prepare_X(dev, sex, flags)
    
    print(f"Running randomized search ({n_iter} iterations) on {len(X)} rows...")
    search.fit(X, y, groups=groups)
    
    res = pd.DataFrame(search.cv_results_)
    res = res.sort_values("rank_test_roc_auc")
    
    out_dir = D.V2_REPORTS_DIR / "optimization"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"xgb_opt_{sex}_{'_'.join(sorted(flags))}.csv"
    res.to_csv(out_path, index=False)
    
    print(f"\nTop 5 results for {sex} ({flags}):")
    cols = ["params", "mean_test_roc_auc", "mean_test_pr_auc", "mean_train_roc_auc"]
    for i, row in res.head(5).iterrows():
        print(f"Rank {row['rank_test_roc_auc']}: ROC-AUC={row['mean_test_roc_auc']:.4f} "
              f"(Train: {row['mean_train_roc_auc']:.4f}) | PR-AUC={row['mean_test_pr_auc']:.4f}")
        print(f"Params: {row['params']}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sex", choices=["women", "men"], required=True)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--flags", default="state", help="Comma-separated feature flags")
    args = parser.parse_args()
    
    flags = {f.strip() for f in args.flags.split(",") if f.strip() and f.strip() != "none"}
    optimize_xgboost(args.sex, args.n_iter, flags)
