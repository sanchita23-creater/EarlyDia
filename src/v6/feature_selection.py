"""
EarlyDia V6 - Feature Selection via Model Importance
======================================================
Uses iterative feature removal based on model importance to identify
and drop noisy features that hurt generalization.

Strategy:
1. Train a model (CatBoost/LightGBM) with all features
2. Get feature importances
3. Iteratively remove lowest-importance features
4. Track ROC-AUC at each step
5. Pick the feature set that maximizes ROC-AUC
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from catboost import CatBoostClassifier

from configs.config import DATA_PROCESSED_DIR
from src.v2.data_v2 import load_dev, make_folds, GROUP_COL
from src.v2.features_v2 import prepare_X, build_preprocessor_v2
from src.models.grouped_cv_baseline import get_feature_lists

SEED = 42
N_SPLITS = 5


def get_feature_importance(X, y, groups, folds, feature_names):
    """Train CatBoost and return feature importances."""
    importances = np.zeros(X.shape[1])
    
    for k in range(N_SPLITS):
        train_idx = folds != k
        val_idx = folds == k
        
        model = CatBoostClassifier(
            learning_rate=0.05,
            iterations=300,
            depth=4,
            l2_leaf_reg=5.0,
            min_data_in_leaf=20,
            auto_class_weights='Balanced',
            random_seed=SEED,
            verbose=0,
            allow_writing_files=False,
        )
        model.fit(X[train_idx], y[train_idx])
        importances += model.feature_importances_
    
    importances /= N_SPLITS
    return importances


def run_cv_with_features(X, y, groups, folds, feature_mask):
    """Run CV with a subset of features (defined by boolean mask)."""
    X_sub = X[:, feature_mask]
    roc_scores = []
    
    for k in range(N_SPLITS):
        train_idx = folds != k
        val_idx = folds == k
        
        model = CatBoostClassifier(
            learning_rate=0.05,
            iterations=300,
            depth=4,
            l2_leaf_reg=5.0,
            min_data_in_leaf=20,
            auto_class_weights='Balanced',
            random_seed=SEED,
            verbose=0,
            allow_writing_files=False,
        )
        model.fit(X[train_idx][:, feature_mask], y[train_idx])
        proba = model.predict_proba(X[val_idx][:, feature_mask])[:, 1]
        roc = roc_auc_score(y[val_idx], proba)
        roc_scores.append(roc)
    
    return np.mean(roc_scores), np.std(roc_scores)


def iterative_feature_selection(sex, X, y, groups, folds, feature_names,
                                 min_features=20, remove_step=5):
    """Iteratively remove lowest-importance features.
    
    Args:
        sex: 'women' or 'men'
        X: feature matrix
        y: labels
        groups: PSU groups
        folds: fold assignments
        feature_names: list of feature names
        min_features: minimum number of features to keep
        remove_step: number of features to remove at each step
    
    Returns:
        dict with results
    """
    print(f"\n{'='*60}")
    print(f"FEATURE SELECTION - {sex.upper()}")
    print(f"{'='*60}")
    print(f"Starting with {X.shape[1]} features")
    
    n_features = X.shape[1]
    feature_mask = np.ones(n_features, dtype=bool)
    
    # Baseline: all features
    base_roc, base_std = run_cv_with_features(X, y, groups, folds, feature_mask)
    print(f"\nBaseline (all {n_features} features): ROC-AUC = {base_roc:.4f} (+/- {base_std:.4f})")
    
    # Track results
    history = [{'n_features': n_features, 'roc_mean': base_roc, 'roc_std': base_std,
                'removed': []}]
    
    best_roc = base_roc
    best_mask = feature_mask.copy()
    best_n = n_features
    
    current_mask = feature_mask.copy()
    current_names = list(feature_names)
    
    step = 0
    while current_mask.sum() > min_features:
        step += 1
        
        # Get feature importances with current feature set
        X_sub = X[:, current_mask]
        importances = get_feature_importance(X_sub, y, groups, folds, current_names)
        
        # Find lowest-importance features
        n_to_remove = min(remove_step, int(current_mask.sum()) - min_features)
        if n_to_remove <= 0:
            break
        
        sorted_idx = np.argsort(importances)
        remove_idx = sorted_idx[:n_to_remove]
        
        # Map back to original indices
        active_idx = np.where(current_mask)[0]
        remove_original = active_idx[remove_idx]
        
        # Remove features
        current_mask[remove_original] = False
        removed_names = [current_names[i] for i in remove_idx]
        
        # Evaluate
        roc_mean, roc_std = run_cv_with_features(X, y, groups, folds, current_mask)
        
        print(f"Step {step}: removed {n_to_remove} features ({current_mask.sum()} remaining), "
              f"ROC-AUC = {roc_mean:.4f} (+/- {roc_std:.4f})")
        print(f"  Removed: {removed_names}")
        
        history.append({
            'n_features': int(current_mask.sum()),
            'roc_mean': roc_mean,
            'roc_std': roc_std,
            'removed': removed_names,
        })
        
        # Track best
        if roc_mean > best_roc:
            best_roc = roc_mean
            best_mask = current_mask.copy()
            best_n = int(current_mask.sum())
            print(f"  ** NEW BEST **")
    
    # Get final feature names
    best_feature_names = [feature_names[i] for i in range(len(feature_names)) if best_mask[i]]
    
    print(f"\n{'='*60}")
    print(f"FEATURE SELECTION RESULTS - {sex.upper()}")
    print(f"{'='*60}")
    print(f"Best: {best_n} features, ROC-AUC = {best_roc:.4f}")
    print(f"Baseline: {n_features} features, ROC-AUC = {base_roc:.4f}")
    print(f"Improvement: {best_roc - base_roc:+.4f}")
    print(f"\nKept features ({best_n}):")
    for name in best_feature_names:
        print(f"  - {name}")
    
    return {
        'best_n_features': best_n,
        'best_roc': best_roc,
        'best_mask': best_mask,
        'best_feature_names': best_feature_names,
        'baseline_roc': base_roc,
        'baseline_n': n_features,
        'history': history,
    }


def main():
    """Run feature selection for both sexes."""
    results = {}
    
    for sex in ['women', 'men']:
        print(f"\n\n{'#'*60}")
        print(f"# LOADING DATA - {sex.upper()}")
        print(f"{'#'*60}")
        
        dev = load_dev(sex)
        y = dev['target_a'].astype(int).values
        groups = dev[GROUP_COL[sex]].values
        folds = make_folds(dev, sex)
        
        # Prepare features
        X_df = prepare_X(dev, sex, set())
        pre = build_preprocessor_v2(sex, set())
        X = pre.fit_transform(X_df)
        
        # Get feature names (after preprocessing expansion)
        # For simplicity, use column indices since preprocessing expands categorical vars
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        print(f"X shape: {X.shape}, y prevalence: {y.mean():.4f}")
        
        result = iterative_feature_selection(sex, X, y, groups, folds, feature_names,
                                              min_features=20, remove_step=5)
        results[sex] = result
        
        print(f"\n{sex.upper()}: Best ROC-AUC = {result['best_roc']:.4f} "
              f"with {result['best_n_features']} features "
              f"(improvement: {result['best_roc'] - result['baseline_roc']:+.4f})")
    
    return results


if __name__ == "__main__":
    main()
