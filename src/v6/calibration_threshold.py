"""
EarlyDia V6 - Calibration + Threshold Optimization
====================================================
Applies probability calibration and finds optimal classification thresholds
for the best models.

Calibration methods:
- Sigmoid (Platt scaling): parametric, works well with enough data
- Isotonic: non-parametric, more flexible but needs more data

Threshold optimization:
- Youden's J (max sensitivity + specificity - 1)
- Target sensitivity (e.g., 0.75, 0.80)
- Max F1 score
- Max balanced accuracy
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import pickle
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                              precision_recall_curve, f1_score, confusion_matrix,
                              balanced_accuracy_score)
from sklearn.model_selection import StratifiedGroupKFold

from configs.config import DATA_PROCESSED_DIR
from src.v2.data_v2 import load_dev, make_folds, GROUP_COL

SEED = 42
N_SPLITS = 5


def calibrate_oof(y, folds, base_proba, method='sigmoid'):
    """Calibrate OOF predictions using cross-calibration.
    
    For each fold, calibrate on other folds' predictions, apply to this fold.
    This gives properly out-of-fold calibrated probabilities.
    """
    cal_proba = np.zeros(len(y))
    
    for k in range(N_SPLITS):
        train_idx = folds != k
        val_idx = folds == k
        
        # Reshape for CalibratedClassifierCV (needs 2D array)
        X_train = base_proba[train_idx].reshape(-1, 1)
        y_train = y[train_idx]
        X_val = base_proba[val_idx].reshape(-1, 1)
        
        # Fit calibrator
        calibrator = CalibratedClassifierCV(
            LogisticRegression(),
            method=method,
            cv=3,
        )
        calibrator.fit(X_train, y_train)
        cal_proba[val_idx] = calibrator.predict_proba(X_val)[:, 1]
    
    return cal_proba


def find_optimal_threshold(y, proba, strategy='youden'):
    """Find optimal classification threshold.
    
    Strategies:
    - 'youden': maximize J = sensitivity + specificity - 1
    - 'f1': maximize F1 score
    - 'balanced_accuracy': maximize balanced accuracy
    - 'target_sens_75': threshold that gives ~75% sensitivity
    - 'target_sens_80': threshold that gives ~80% sensitivity
    """
    fpr, tpr, thresholds = roc_curve(y, proba)
    prec, rec, pr_thresholds = precision_recall_curve(y, proba)
    
    if strategy == 'youden':
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_idx]
        best_sens = tpr[best_idx]
        best_spec = 1 - fpr[best_idx]
    
    elif strategy == 'f1':
        f1_scores = 2 * (prec * rec) / (prec + rec + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_threshold = pr_thresholds[best_idx]
        best_sens = rec[best_idx]
        best_prec = prec[best_idx]
        best_spec = None
    
    elif strategy == 'balanced_accuracy':
        bal_acc = (tpr + (1 - fpr)) / 2
        best_idx = np.argmax(bal_acc)
        best_threshold = thresholds[best_idx]
        best_sens = tpr[best_idx]
        best_spec = 1 - fpr[best_idx]
    
    elif strategy.startswith('target_sens_'):
        target_sens = float(strategy.split('_')[-1]) / 100.0
        # Find threshold closest to target sensitivity
        idx = np.argmin(np.abs(tpr - target_sens))
        best_threshold = thresholds[idx]
        best_sens = tpr[idx]
        best_spec = 1 - fpr[idx]
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return best_threshold, best_sens, best_spec


def evaluate_at_threshold(y, proba, threshold):
    """Compute full metrics at a given threshold."""
    preds = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = f1_score(y, preds)
    bal_acc = balanced_accuracy_score(y, preds)
    flag_rate = preds.mean()
    
    return {
        'threshold': threshold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'f1': f1,
        'balanced_accuracy': bal_acc,
        'flag_rate': flag_rate,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
    }


def analyze_model(sex, model_name, y, folds, base_proba):
    """Full calibration + threshold analysis for one model."""
    print(f"\n{'='*60}")
    print(f"CALIBRATION ANALYSIS - {sex.upper()} / {model_name}")
    print(f"{'='*60}")
    
    # Base metrics (uncalibrated)
    base_roc = roc_auc_score(y, base_proba)
    base_pr = average_precision_score(y, base_proba)
    print(f"\nUncalibrated: ROC-AUC={base_roc:.4f}, PR-AUC={base_pr:.4f}")
    
    # Calibrate with sigmoid
    sig_proba = calibrate_oof(y, folds, base_proba, method='sigmoid')
    sig_roc = roc_auc_score(y, sig_proba)
    sig_pr = average_precision_score(y, sig_proba)
    print(f"Sigmoid cal:  ROC-AUC={sig_roc:.4f}, PR-AUC={sig_pr:.4f}")
    
    # Calibrate with isotonic
    iso_proba = calibrate_oof(y, folds, base_proba, method='isotonic')
    iso_roc = roc_auc_score(y, iso_proba)
    iso_pr = average_precision_score(y, iso_proba)
    print(f"Isotonic cal: ROC-AUC={iso_roc:.4f}, PR-AUC={iso_pr:.4f}")
    
    # Pick best calibration (by ROC-AUC)
    results = [
        ('uncalibrated', base_proba, base_roc, base_pr),
        ('sigmoid', sig_proba, sig_roc, sig_pr),
        ('isotonic', iso_proba, iso_roc, iso_pr),
    ]
    best_cal_name, best_cal_proba, best_cal_roc, best_cal_pr = max(results, key=lambda x: x[2])
    
    print(f"\nBest calibration: {best_cal_name} (ROC-AUC={best_cal_roc:.4f})")
    
    # Find optimal thresholds
    print(f"\nThreshold optimization:")
    strategies = ['youden', 'f1', 'balanced_accuracy', 'target_sens_75', 'target_sens_80']
    threshold_results = {}
    
    for strategy in strategies:
        threshold, sens, spec = find_optimal_threshold(y, best_cal_proba, strategy)
        metrics = evaluate_at_threshold(y, best_cal_proba, threshold)
        threshold_results[strategy] = metrics
        print(f"  {strategy:25s}: threshold={threshold:.4f}, "
              f"sens={metrics['sensitivity']:.3f}, spec={metrics['specificity']:.3f}, "
              f"F1={metrics['f1']:.3f}, flag_rate={metrics['flag_rate']:.3f}")
    
    return {
        'model_name': model_name,
        'best_calibration': best_cal_name,
        'best_proba': best_cal_proba,
        'roc_auc': best_cal_roc,
        'pr_auc': best_cal_pr,
        'thresholds': threshold_results,
    }


def main():
    """Run calibration + threshold optimization for best models."""
    results = {}
    
    for sex in ['women', 'men']:
        print(f"\n\n{'#'*60}")
        print(f"# {sex.upper()}")
        print(f"{'#'*60}")
        
        dev = load_dev(sex)
        y = dev['target_a'].astype(int).values
        folds = make_folds(dev, sex)
        
        # Load best OOF predictions
        oof_dir = DATA_PROCESSED_DIR / "v2"
        
        # Try to load Optuna results first, fall back to benchmark results
        best_model = None
        best_proba = None
        best_roc = 0
        
        for f in sorted(oof_dir.glob(f"{sex}_oof_*.pkl")):
            data = pickle.load(open(f, 'rb'))
            if isinstance(data, pd.DataFrame) and 'p_raw' in data.columns:
                model_name = f.stem
                proba = data['p_raw'].values
                roc = roc_auc_score(y, proba)
                if roc > best_roc:
                    best_roc = roc
                    best_model = model_name
                    best_proba = proba
        
        if best_proba is None:
            print(f"No OOF predictions found for {sex}")
            continue
        
        print(f"Best model: {best_model} (ROC-AUC={best_roc:.4f})")
        
        result = analyze_model(sex, best_model, y, folds, best_proba)
        results[sex] = result
    
    # Final summary
    print(f"\n\n{'#'*60}")
    print(f"# FINAL CALIBRATION SUMMARY")
    print(f"{'#'*60}")
    for sex, res in results.items():
        if res:
            print(f"\n{sex.upper()}:")
            print(f"  Best calibration: {res['best_calibration']}")
            print(f"  ROC-AUC: {res['roc_auc']:.4f}")
            print(f"  PR-AUC:  {res['pr_auc']:.4f}")
            youden = res['thresholds']['youden']
            print(f"  Youden threshold: {youden['threshold']:.4f}")
            print(f"    -> sens={youden['sensitivity']:.3f}, spec={youden['specificity']:.3f}, "
                  f"F1={youden['f1']:.3f}")
    
    return results


if __name__ == "__main__":
    main()
