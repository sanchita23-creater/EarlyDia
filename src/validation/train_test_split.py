"""
EarlyDia - Stage: Cluster-aware train/test split
No PSU/cluster may appear in both train and test. ~80/20 split, target prevalence preserved
within a documented tolerance. Fixed, reproducible seed selection procedure.
"""
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
from configs.config import (DATA_RAW_DIR, DATA_INTERIM_DIR, DATA_PROCESSED_DIR,
                            EXPLAINABILITY_REPORTS_DIR)
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import json

SEED_CANDIDATES = list(range(42, 42 + 50))  # deterministic search order
TEST_SIZE = 0.20
PREVALENCE_REL_TOLERANCE = 0.15  # train/test prevalence must be within +/-15% (relative) of overall


def find_group_split(df, group_col, target_col, seed_candidates, test_size, tol):
    y = df[target_col].values
    groups = df[group_col].values
    overall_prev = y.mean()

    for seed in seed_candidates:
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(gss.split(df, y, groups))
        train_prev = y[train_idx].mean()
        test_prev = y[test_idx].mean()
        train_ok = abs(train_prev - overall_prev) / overall_prev <= tol
        test_ok = abs(test_prev - overall_prev) / overall_prev <= tol
        if train_ok and test_ok:
            return {
                "seed": seed,
                "train_idx": train_idx,
                "test_idx": test_idx,
                "overall_prev": overall_prev,
                "train_prev": train_prev,
                "test_prev": test_prev,
            }
    raise RuntimeError("No candidate seed satisfied the prevalence tolerance - widen search or tolerance.")


def split_and_report(df, group_col, target_col, label):
    result = find_group_split(df, group_col, target_col, SEED_CANDIDATES, TEST_SIZE, PREVALENCE_REL_TOLERANCE)
    train = df.iloc[result["train_idx"]].copy()
    test = df.iloc[result["test_idx"]].copy()

    train_psus = set(train[group_col].unique())
    test_psus = set(test[group_col].unique())
    overlap = train_psus & test_psus

    print(f"=== {label}: cluster-aware split (selected seed={result['seed']}) ===")
    print(f"  train rows: {len(train)}  ({len(train)/len(df)*100:.1f}%)")
    print(f"  test rows:  {len(test)}  ({len(test)/len(df)*100:.1f}%)")
    print(f"  train unique PSUs: {len(train_psus)}")
    print(f"  test unique PSUs:  {len(test_psus)}")
    print(f"  PSU overlap between train/test: {len(overlap)} (must be 0)")
    print(f"  overall prevalence: {result['overall_prev']*100:.3f}%")
    print(f"  train prevalence:   {result['train_prev']*100:.3f}%")
    print(f"  test prevalence:    {result['test_prev']*100:.3f}%")
    assert len(overlap) == 0, f"LEAKAGE: {len(overlap)} PSUs appear in both train and test for {label}!"
    print("  PSU disjointness assertion: PASSED")
    print()

    return train, test, result


if __name__ == "__main__":
    women = pd.read_pickle(str(DATA_PROCESSED_DIR) + "/women_predictors.pkl")
    men = pd.read_pickle(str(DATA_PROCESSED_DIR) + "/men_predictors.pkl")

    w_train, w_test, w_result = split_and_report(women, "v021_psu", "target_a", "WOMEN")
    m_train, m_test, m_result = split_and_report(men, "mv021_psu", "target_a", "MEN")

    w_train.to_pickle(str(DATA_PROCESSED_DIR) + "/women_train.pkl")
    w_test.to_pickle(str(DATA_PROCESSED_DIR) + "/women_test.pkl")
    m_train.to_pickle(str(DATA_PROCESSED_DIR) + "/men_train.pkl")
    m_test.to_pickle(str(DATA_PROCESSED_DIR) + "/men_test.pkl")

    split_config = {
        "method": "GroupShuffleSplit, group=PSU, test_size=0.20",
        "seed_search_candidates": SEED_CANDIDATES,
        "prevalence_relative_tolerance": PREVALENCE_REL_TOLERANCE,
        "women": {
            "selected_seed": w_result["seed"],
            "train_rows": len(w_train), "test_rows": len(w_test),
            "train_psus": int(w_train.v021_psu.nunique()), "test_psus": int(w_test.v021_psu.nunique()),
            "overall_prevalence": w_result["overall_prev"],
            "train_prevalence": w_result["train_prev"],
            "test_prevalence": w_result["test_prev"],
        },
        "men": {
            "selected_seed": m_result["seed"],
            "train_rows": len(m_train), "test_rows": len(m_test),
            "train_psus": int(m_train.mv021_psu.nunique()), "test_psus": int(m_test.mv021_psu.nunique()),
            "overall_prevalence": m_result["overall_prev"],
            "train_prevalence": m_result["train_prev"],
            "test_prevalence": m_result["test_prev"],
        },
    }
    with open(str(DATA_PROCESSED_DIR) + "/split_config.json", "w") as f:
        json.dump(split_config, f, indent=2)
    print("Saved: women_train.pkl, women_test.pkl, men_train.pkl, men_test.pkl, split_config.json")
