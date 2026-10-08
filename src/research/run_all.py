"""Generate every research artifact that can be produced from what is present.

Usage:  python -m src.research.run_all
Reads locked artifacts/reports, writes ONLY to reports/research/. Row-level analyses (survey weights,
state, Mapper/TDA) run automatically when data/processed/*.pkl exist and are otherwise recorded as NOT_RUN.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.research import (calibration_analysis, decision_curve, subgroup_analysis, target_sensitivity, idrs_analysis,  # noqa: E402
                          feature_importance, literature, benchmark_baselines, survey_analysis, state_holdout, figures, research_report)
from src.research.common import assert_thresholds_locked, DATA_PROCESSED, ensure_dirs  # noqa: E402


def mapper_on_real_data(sex, n_max=8000, B=30):
    """Builds Mapper on a TRAIN subsample (predictors only), runs stability + null, writes reports/research/mapper/."""
    import json
    import numpy as np, pandas as pd
    from src.models.grouped_cv_baseline import get_feature_lists
    from src.research import mapper_analysis as M, mapper_stability as S, mapper_null_model as N
    from src.research.common import OUT, write_json
    path = DATA_PROCESSED / f"{sex}_train.pkl"
    if not path.exists():
        return {"status": "NOT_RUN", "reason": f"{path.name} not present"}
    df = pd.read_pickle(path)
    num, c0, cc0, cats = get_feature_lists(sex)
    num_cols, cat_cols = num + c0, cc0 + cats
    sub = df.sample(min(n_max, len(df)), random_state=42)                  # target not used for sampling
    X, names, ref = M.encode_predictors(sub, num_cols, cat_cols)
    cfg = M.MapperConfig()
    stab = S.assess(X, cfg, n_boot=5)
    null = N.run_null(X, cfg, B=B)
    prof = M.characterise(stab["reference"]["point_community"], y=sub.target_a.to_numpy(), features=sub[["age"]])
    prof["stable"] = stab["stable"] if len(prof) == len(stab["stable"]) else None
    prof.to_csv(OUT / "mapper" / f"{sex}_communities.csv", index=False)
    res = {"status": "OK", "n_used": len(sub), "config": cfg.__dict__, "mean_ari": stab["mean_ari"], "mean_nmi": stab["mean_nmi"],
           "community_jaccard": stab["community_jaccard"], "stable": stab["stable"], "null": {k: v for k, v in null.items() if k != "null_values"}}
    write_json(OUT / "mapper" / f"{sex}_mapper_summary.json", res)
    return res


def main():
    ensure_dirs(); assert_thresholds_locked()
    target_sensitivity.run(); calibration_analysis.run(); decision_curve.run(); subgroup_analysis.run()
    idrs_analysis.run(); feature_importance.run(); literature.run(); benchmark_baselines.run()
    survey_analysis.run(); state_holdout.run()
    extra = {}
    for sex in ("women", "men"):
        r = mapper_on_real_data(sex)
        if r["status"] == "OK":
            extra[f"mapper_{sex}"] = "DONE"
    figures.run()
    research_report.run(extra)
    print("research outputs written to reports/research/")


if __name__ == "__main__":
    main()
