"""
EarlyDia - Bounded XGBoost Custom Optimization
Works by injecting dynamic configs into features_v2.XGB_CONFIGS and using run_oof.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import random
import pandas as pd
from itertools import product

from src.v2.features_v2 import XGB_CONFIGS
from src.v2.run_v2 import run_oof, _load_oof, _paths, EXPERIMENTS
from src.v2.evaluation_v2 import compute_summary

def run_grid(sex, flags_str, n_iter):
    flags = {f.strip() for f in flags_str.split(",") if f.strip() and f.strip() != "none"}
    flags_arg = flags_str
    
    # 1. Generate search space
    space = dict(
        scale_pos_weight=[10.0, 15.0, 20.0, 25.0, 30.0],
        learning_rate=[0.01, 0.03, 0.05, 0.1],
        n_estimators=[100, 200, 300],
        max_depth=[3, 4, 5, 6],
        min_child_weight=[1, 5, 10, 20],
        subsample=[0.6, 0.8, 1.0],
        colsample_bytree=[0.6, 0.8, 1.0],
        reg_alpha=[0, 1.0, 5.0],
        reg_lambda=[1.0, 5.0, 10.0]
    )
    
    keys = list(space.keys())
    combinations = list(product(*[space[k] for k in keys]))
    random.seed(42)
    random.shuffle(combinations)
    
    selected = combinations[:n_iter]
    
    results = []
    
    for i, comb in enumerate(selected):
        kw = dict(zip(keys, comb))
        name = f"xgb_opt_{i}"
        XGB_CONFIGS[name] = kw
        
        print(f"\n[{i+1}/{n_iter}] Testing {name}: {kw}")
        try:
            run_oof(sex, name, flags_arg, name, force=True)
            df = _load_oof(sex, name)
            summary = compute_summary(df, sex)
            summary["params"] = str(kw)
            summary["name"] = name
            results.append(summary)
            print(f"   -> ROC-AUC: {summary['roc_auc']:.4f} | PR-AUC: {summary['pr_auc']:.4f}")
        except Exception as e:
            print(f"   -> Error: {e}")
            
    df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    print("\n=== TOP 5 CONFIGS ===")
    print(df[["name", "roc_auc", "pr_auc", "params"]].head(5).to_string(index=False))
    
    out = Path("reports") / "v2" / f"optimization_{sex}.csv"
    df.to_csv(out, index=False)
    print(f"Saved to {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sex", choices=["women", "men"], required=True)
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument("--flags", default="state", help="Comma-separated feature flags")
    args = parser.parse_args()
    run_grid(args.sex, args.flags, args.n_iter)
