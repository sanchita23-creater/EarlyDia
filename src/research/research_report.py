"""Assemble research_summary.json with explicit run/not-run status and claim boundaries."""
import json

from src.research.common import OUT, load_final_eval, write_json, ensure_dirs

CLAIMS = {
    "established_by_this_project": ["production model performance on the locked test set", "calibration (reliability table, slope, intercept, Brier)",
                                    "cluster-aware validation design", "observed age-subgroup performance", "observed error patterns"],
    "exploratory": ["Mapper communities", "TDA structures", "subgroup patterns", "null-model comparisons", "feature-space heterogeneity"],
    "not_established": ["clinical diagnosis", "future diabetes prediction", "causality", "biological diabetes subtypes",
                        "clinical utility", "prospective validity", "external validity"],
}


def run(extra_status=None):
    ensure_dirs()
    fe = load_final_eval()
    def slim(s):
        o = fe[s]["overall"]
        return {k: o[k] for k in ("n", "n_positive", "prevalence", "roc_auc", "pr_auc", "sensitivity", "specificity", "ppv_precision", "npv",
                                  "brier_score", "calibration_slope", "calibration_intercept")} | {"threshold": fe[s]["threshold"],
                "calibration": fe[s]["calibration_method"], "model": fe[s]["model"]}
    ages = {s: {k: {m: v[m] for m in ("n", "prevalence", "roc_auc", "sensitivity", "specificity")} for k, v in fe[s]["age_subgroups"].items()} for s in ("women", "men")}
    summary = {
        "production_model": {"label": "Production model (locked) - final test metrics, unchanged", "women": slim("women"), "men": slim("men")},
        "age_subgroups_locked_threshold": ages,
        "key_finding": ("Overall sensitivity (~0.76) hides large age heterogeneity at the locked threshold: ages 18-29 sensitivity is "
                        "~0.46 (women) / ~0.39 (men) versus ~0.93 / ~0.94 at ages 30-40, reflecting the pooled threshold applied to a "
                        "population whose predicted risks differ with age. No subgroup-specific thresholds were created."),
        "analysis_status": {
            "target_sensitivity": "DONE - full replication NOT possible from retained extraction",
            "calibration": "DONE (from locked reliability table)",
            "decision_curve": "DONE - exact at locked threshold; curve APPROXIMATE (decile bins)",
            "subgroups_age_sex": "DONE (locked report)",
            "baseline_comparison": "DONE - recorded grouped-CV values, not recomputed",
            "idrs": "COMPATIBILITY ONLY - IDRS not reproducible (3/4 components missing)",
            "feature_importance": "DONE (permutation importance, not SHAP)",
            "survey_weighted_prevalence": "NOT RUN - needs data/processed (code ready: src/research/survey_analysis.py)",
            "state_holdout": "NOT RUN - needs row-level data (code ready)",
            "mapper_tda_stability_null": "NOT RUN on real data - needs row-level training data (code ready, unit-tested on synthetic data only)",
            "residence_education_subgroups": "NOT RUN - needs row-level data",
            "roc_pr_curves": "NOT GENERATED - need row-level predictions",
        } | (extra_status or {}),
        "claim_boundaries": CLAIMS,
        "positioning": ("EarlyDia is a leakage-aware, cluster-aware, calibrated, sensitivity-oriented screening research prototype using NFHS-5 data, "
                        "with explicit target-definition limitations, subgroup/error analysis, survey-aware descriptive analysis, decision-curve evaluation, "
                        "and exploratory topological feature-space analysis."),
    }
    write_json(OUT / "research_summary.json", summary)
    return summary
