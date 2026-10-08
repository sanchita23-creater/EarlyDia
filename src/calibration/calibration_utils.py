"""
EarlyDia - shared calibration utilities (sigmoid/Platt scaling helpers, calibration
diagnostics). Used both during training-stage calibration comparison and at inference
time to apply an already-fit sigmoid calibrator. Contains NO fitting-on-the-fly logic
for inference use - isotonic calibrators are used directly via their own .predict();
sigmoid calibrators are small fitted LogisticRegression objects already saved in the
production artifact, and this module only supplies the matching predict-time transform.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def sigmoid_calibrate_fit(p_train, y_train):
    """Platt scaling: logistic regression of y on logit(p). Training-time only."""
    eps = 1e-6
    p_clip = np.clip(p_train, eps, 1 - eps)
    logit_p = np.log(p_clip / (1 - p_clip)).reshape(-1, 1)
    lr = LogisticRegression()
    lr.fit(logit_p, y_train)
    return lr


def sigmoid_calibrate_predict(lr, p):
    """Apply an already-fit sigmoid calibrator. Safe for inference use (no fitting)."""
    eps = 1e-6
    p = np.asarray(p, dtype="float64")
    p_clip = np.clip(p, eps, 1 - eps)
    logit_p = np.log(p_clip / (1 - p_clip)).reshape(-1, 1)
    return lr.predict_proba(logit_p)[:, 1]


def calibration_slope_intercept(y, p):
    """Cox calibration regression: y ~ logit(p). slope=1,intercept=0 => perfectly calibrated."""
    eps = 1e-6
    p_clip = np.clip(p, eps, 1 - eps)
    logit_p = np.log(p_clip / (1 - p_clip)).reshape(-1, 1)
    lr = LogisticRegression()
    lr.fit(logit_p, y)
    return lr.coef_[0][0], lr.intercept_[0]


def reliability_table(y, p, n_bins=10):
    bins = pd.qcut(p, q=n_bins, duplicates="drop")
    df = pd.DataFrame({"y": y, "p": p, "bin": bins})
    grouped = df.groupby("bin", observed=True).agg(
        mean_predicted=("p", "mean"), frac_positive=("y", "mean"), n=("y", "size"))
    return grouped.reset_index(drop=True)
