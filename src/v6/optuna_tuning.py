"""
EarlyDia V6 - Optuna Hyperparameter Tuning
============================================
Bayesian optimization over CatBoost and LightGBM hyperparameters using
PSU-grouped 5-fold CV (same strategy as all other experiments).

Search space is designed to explore:
- Tree depth / leaf complexity (model capacity)
- Learning rate vs n_estimators tradeoff
- Regularization (L1/L2, subsample, colsample)
- Class imbalance handling
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import pickle
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from catboost import CatBoostClassifier
import lightgbm as lgb

from configs.config import DATA_PROCESSED_DIR, REPORTS_DIR
from src.v2.data_v2 import load_dev, make_folds, GROUP_COL, V2_DATA_DIR
from src.v2.features_v2 import prepare_X, build_preprocessor_v2

SEED = 42
N_SPLITS = 5
N_TRIALS = 30  # Reduced for speed; can increase later


def run_cv(X, y, groups, folds, model_fn):
    """Run PSU-grouped 5-fold CV and return mean ROC-AUC + OOF predictions."""
    oof_proba = np.zeros(len(y))
    roc_scores = []

    for k in range(N_SPLITS):
        train_idx = folds != k
        val_idx = folds == k

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = model_fn()
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        oof_proba[val_idx] = proba

        roc = roc_auc_score(y_val, proba)
        roc_scores.append(roc)

    return np.mean(roc_scores), np.std(roc_scores), oof_proba


def tune_catboost(sex, X, y, groups, folds, n_trials=N_TRIALS):
    """Optuna tuning for CatBoost."""
    print(f"\n{'='*60}")
    print(f"OPTUNA CATBOOST TUNING - {sex.upper()}")
    print(f"{'='*60}")

    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
            'depth': trial.suggest_int('depth', 3, 7),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 15.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.5),
            'random_strength': trial.suggest_float('random_strength', 0.5, 5.0, log=True),
            'iterations': trial.suggest_int('iterations', 150, 300),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
            'auto_class_weights': 'Balanced',
            'random_seed': SEED,
            'verbose': 0,
            'allow_writing_files': False,
        }

        def model_fn():
            return CatBoostClassifier(**params)

        roc_mean, _, _ = run_cv(X, y, groups, folds, model_fn)
        return roc_mean

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest trial: ROC-AUC = {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # Train final model with best params on all folds to get OOF
    best_params = study.best_params
    best_params.update({
        'auto_class_weights': 'Balanced',
        'random_seed': SEED,
        'verbose': 0,
        'allow_writing_files': False,
    })

    def best_model_fn():
        return CatBoostClassifier(**best_params)

    roc_mean, roc_std, oof_proba = run_cv(X, y, groups, folds, best_model_fn)
    pr = average_precision_score(y, oof_proba)

    print(f"\nFinal CV: ROC-AUC = {roc_mean:.4f} (+/- {roc_std:.4f})")
    print(f"PR-AUC = {pr:.4f}")

    return {
        'best_roc': roc_mean,
        'roc_std': roc_std,
        'pr_auc': pr,
        'params': best_params,
        'oof_proba': oof_proba,
        'study': study,
    }


def tune_lightgbm(sex, X, y, groups, folds, n_trials=N_TRIALS):
    """Optuna tuning for LightGBM."""
    print(f"\n{'='*60}")
    print(f"OPTUNA LIGHTGBM TUNING - {sex.upper()}")
    print(f"{'='*60}")

    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 80),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 10.0, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 150, 300),
            'random_state': SEED,
            'n_jobs': -1,
            'verbosity': -1,
            'is_unbalance': True,
        }

        def model_fn():
            return lgb.LGBMClassifier(**params)

        roc_mean, _, _ = run_cv(X, y, groups, folds, model_fn)
        return roc_mean

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest trial: ROC-AUC = {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    best_params = study.best_params
    best_params.update({
        'random_state': SEED,
        'n_jobs': -1,
        'verbosity': -1,
        'is_unbalance': True,
    })

    def best_model_fn():
        return lgb.LGBMClassifier(**best_params)

    roc_mean, roc_std, oof_proba = run_cv(X, y, groups, folds, best_model_fn)
    pr = average_precision_score(y, oof_proba)

    print(f"\nFinal CV: ROC-AUC = {roc_mean:.4f} (+/- {roc_std:.4f})")
    print(f"PR-AUC = {pr:.4f}")

    return {
        'best_roc': roc_mean,
        'roc_std': roc_std,
        'pr_auc': pr,
        'params': best_params,
        'oof_proba': oof_proba,
        'study': study,
    }


def main():
    """Run Optuna tuning for both sexes."""
    results = {}

    for sex in ['women', 'men']:
        print(f"\n\n{'#'*60}")
        print(f"# LOADING DATA - {sex.upper()}")
        print(f"{'#'*60}")

        dev = load_dev(sex)
        y = dev['target_a'].astype(int).values
        groups = dev[GROUP_COL[sex]].values
        folds = make_folds(dev, sex)

        # Prepare features with empty flags (base v6 features are in the predictor)
        X_df = prepare_X(dev, sex, set())
        pre = build_preprocessor_v2(sex, set())
        X = pre.fit_transform(X_df)

        print(f"X shape: {X.shape}, y prevalence: {y.mean():.4f}")

        # Tune CatBoost
        cb_result = tune_catboost(sex, X, y, groups, folds)

        # Tune LightGBM
        lgbm_result = tune_lightgbm(sex, X, y, groups, folds)

        # Pick best
        if cb_result['best_roc'] >= lgbm_result['best_roc']:
            best = cb_result
            best_name = 'catboost'
        else:
            best = lgbm_result
            best_name = 'lightgbm'

        results[sex] = {
            'catboost': cb_result,
            'lightgbm': lgbm_result,
            'best_name': best_name,
            'best_roc': best['best_roc'],
            'best_pr': best['pr_auc'],
        }

        print(f"\n{'='*60}")
        print(f"SUMMARY - {sex.upper()}")
        print(f"{'='*60}")
        print(f"CatBoost:  ROC-AUC = {cb_result['best_roc']:.4f}")
        print(f"LightGBM:  ROC-AUC = {lgbm_result['best_roc']:.4f}")
        print(f"BEST:      {best_name} = {best['best_roc']:.4f} (gap to 0.85: {0.85 - best['best_roc']:.4f})")

        # Save OOF predictions
        V2_DATA_DIR.mkdir(parents=True, exist_ok=True)
        oof_path = V2_DATA_DIR / f"{sex}_oof_optuna_{best_name}.pkl"
        oof_data = pd.DataFrame({
            'y': y,
            'psu': groups,
            'fold': folds,
            'p_raw': best['oof_proba'],
        })
        oof_data.to_pickle(oof_path)
        print(f"Saved OOF: {oof_path}")

    # Final summary
    print(f"\n\n{'#'*60}")
    print(f"# FINAL RESULTS")
    print(f"{'#'*60}")
    for sex, res in results.items():
        print(f"{sex.upper()}: ROC-AUC = {res['best_roc']:.4f} ({res['best_name']})  "
              f"gap to 0.85: {0.85 - res['best_roc']:.4f}")

    return results


if __name__ == "__main__":
    main()
