"""
EarlyDia V6 - Ensemble Stacking
================================
Combines OOF predictions from multiple base models (CatBoost, LightGBM, XGB)
using a meta-learner (logistic regression) to improve ROC-AUC.

Strategy:
1. Load OOF predictions from each base model
2. Create meta-features (one column per base model)
3. Train meta-learner on OOF predictions (same PSU-grouped CV)
4. Evaluate stacked ensemble vs individual models
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from configs.config import DATA_PROCESSED_DIR, REPORTS_DIR
from src.v2.data_v2 import load_dev, make_folds, GROUP_COL

SEED = 42
N_SPLITS = 5


def load_oof_predictions(sex, model_family):
    """Load OOF predictions for a model family (cb, lgbm, xgb)."""
    oof_dir = DATA_PROCESSED_DIR / "v2"
    results = {}
    for f in sorted(oof_dir.glob(f"{sex}_oof_bench_{model_family}_*.pkl")):
        model_name = f.stem.replace(f"{sex}_oof_bench_", "")
        data = pickle.load(open(f, 'rb'))
        results[model_name] = {
            'y': data['y'].values,
            'psu': data['psu'].values,
            'fold': data['fold'].values,
            'p_raw': data['p_raw'].values,
        }
    return results


def create_meta_features(oof_dict, selected_models=None):
    """Create meta-feature matrix from OOF predictions.
    
    Args:
        oof_dict: dict of {model_name: {y, psu, fold, p_raw}}
        selected_models: list of model names to include (None = all)
    
    Returns:
        X_meta: meta-feature matrix (n_samples, n_models)
        y: true labels
        psu: PSU groups
        fold: fold assignments
    """
    if selected_models is None:
        selected_models = list(oof_dict.keys())
    
    # Use first model as reference for y, psu, fold
    ref_model = selected_models[0]
    y = oof_dict[ref_model]['y']
    psu = oof_dict[ref_model]['psu']
    fold = oof_dict[ref_model]['fold']
    
    # Stack predictions as meta-features
    X_meta = np.column_stack([oof_dict[m]['p_raw'] for m in selected_models])
    
    return X_meta, y, psu, fold, selected_models


def train_meta_learner(X_meta, y, psu, fold, n_splits=N_SPLITS, seed=SEED):
    """Train meta-learner using same PSU-grouped CV strategy.
    
    For each fold, train meta-learner on other folds' predictions, predict on this fold.
    This gives us out-of-fold meta-predictions for evaluation.
    """
    meta_oof_proba = np.zeros(len(y))
    meta_model = LogisticRegression(class_weight='balanced', random_state=seed, max_iter=1000)
    
    for k in range(n_splits):
        train_idx = fold != k
        val_idx = fold == k
        
        X_train, y_train = X_meta[train_idx], y[train_idx]
        X_val = X_meta[val_idx]
        
        meta_model.fit(X_train, y_train)
        meta_oof_proba[val_idx] = meta_model.predict_proba(X_val)[:, 1]
    
    return meta_oof_proba, meta_model


def evaluate_stacking(sex, model_families=['cb', 'lgbm']):
    """Evaluate stacking ensemble for a given sex.
    
    Args:
        sex: 'women' or 'men'
        model_families: list of model families to include in ensemble
    
    Returns:
        dict with results
    """
    print(f"\n{'='*60}")
    print(f"STACKING ENSEMBLE - {sex.upper()}")
    print(f"{'='*60}")
    
    # Load OOF predictions from each model family
    all_oof = {}
    for family in model_families:
        oof = load_oof_predictions(sex, family)
        all_oof.update(oof)
        print(f"Loaded {len(oof)} {family} models")
    
    # Select best model from each family (by ROC-AUC)
    selected_models = []
    for family in model_families:
        family_models = [m for m in all_oof.keys() if m.startswith(family)]
        if not family_models:
            continue
        
        # Compute ROC-AUC for each model in family
        best_model = None
        best_roc = 0
        for m in family_models:
            roc = roc_auc_score(all_oof[m]['y'], all_oof[m]['p_raw'])
            if roc > best_roc:
                best_roc = roc
                best_model = m
        
        selected_models.append(best_model)
        print(f"Best {family}: {best_model} (ROC-AUC={best_roc:.4f})")
    
    # Create meta-features
    X_meta, y, psu, fold, model_names = create_meta_features(all_oof, selected_models)
    print(f"\nMeta-feature matrix: {X_meta.shape}")
    print(f"Models in ensemble: {model_names}")
    
    # Train meta-learner
    print("\nTraining meta-learner (logistic regression)...")
    meta_oof_proba, meta_model = train_meta_learner(X_meta, y, psu, fold)
    
    # Evaluate
    roc = roc_auc_score(y, meta_oof_proba)
    pr = average_precision_score(y, meta_oof_proba)
    
    print(f"\n{'='*60}")
    print(f"STACKING RESULTS - {sex.upper()}")
    print(f"{'='*60}")
    print(f"Meta-learner: LogisticRegression")
    print(f"Base models: {', '.join(model_names)}")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC:  {pr:.4f}")
    print(f"Gap to 0.85: {0.85 - roc:.4f}")
    
    # Compare to best individual model
    best_individual_roc = max(roc_auc_score(y, all_oof[m]['p_raw']) for m in model_names)
    improvement = roc - best_individual_roc
    print(f"\nBest individual: {best_individual_roc:.4f}")
    print(f"Improvement: {improvement:+.4f}")
    
    # Save results
    results = {
        'sex': sex,
        'meta_learner': 'LogisticRegression',
        'base_models': model_names,
        'roc_auc': roc,
        'pr_auc': pr,
        'best_individual_roc': best_individual_roc,
        'improvement': improvement,
        'meta_oof_proba': meta_oof_proba,
    }
    
    return results


def main():
    """Run stacking for both sexes."""
    results = {}
    
    for sex in ['women', 'men']:
        try:
            results[sex] = evaluate_stacking(sex, model_families=['cb', 'lgbm'])
        except Exception as e:
            print(f"Error for {sex}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for sex, res in results.items():
        if res:
            print(f"{sex.upper()}: ROC-AUC = {res['roc_auc']:.4f} (gap to 0.85: {0.85 - res['roc_auc']:.4f})")
    
    return results


if __name__ == "__main__":
    main()
