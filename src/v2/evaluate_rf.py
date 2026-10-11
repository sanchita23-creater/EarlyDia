"""
EarlyDia - Random Forest Baseline (Stage 7)

Evaluates Random Forest using the same PSU-grouped stratified cross-validation
on the development data, to serve as a baseline comparison.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from src.v2 import data_v2 as D
from src.v2.features_v2 import build_preprocessor_v2, SEED, parse_flags

def evaluate_rf(sex, flags):
    print(f"Loading development data for {sex}...")
    pred, tr_idx, te_idx = D._full(sex)
    dev = pred.iloc[tr_idx]
    
    y = dev["target_a"].astype(int).values
    groups = dev[D.GROUP_COL[sex]].values
    
    pre = build_preprocessor_v2(sex, flags)
    
    # Balanced subsample is critical for imbalanced data in RF
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1
    )
    
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision"
    }
    
    from src.v2.features_v2 import prepare_X
    X = prepare_X(dev, sex, flags)
    
    print(f"Evaluating Random Forest (flags={flags}) on {len(X)} rows...")
    res = cross_validate(
        pipe, X, y, groups=groups, cv=cv, scoring=scoring, 
        return_train_score=True, n_jobs=1
    )
    
    print(f"Random Forest ({sex}) Results:")
    print(f"Mean Test ROC-AUC: {np.mean(res['test_roc_auc']):.4f} +/- {np.std(res['test_roc_auc']):.4f}")
    print(f"Mean Train ROC-AUC: {np.mean(res['train_roc_auc']):.4f}")
    print(f"Mean Test PR-AUC: {np.mean(res['test_pr_auc']):.4f} +/- {np.std(res['test_pr_auc']):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sex", choices=["women", "men"], required=True)
    parser.add_argument("--flags", default="state", help="Comma-separated feature flags")
    args = parser.parse_args()
    
    flags = parse_flags(args.flags)
    evaluate_rf(args.sex, flags)
