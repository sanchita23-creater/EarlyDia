"""Structured literature benchmarking. Only facts verified from retrieved open-access text are stated;
everything else is marked 'not verified' so it cannot be mistaken for a claim."""
import json

from src.research.common import OUT, ensure_dirs

NV = "Not verified in this build - complete from the full text"

SAHADEVAN = {
    "reference": "Sahadevan et al., J Glob Health 2023 (open access: https://pmc.ncbi.nlm.nih.gov/articles/PMC10704946/)",
    "population": {"EarlyDia": "NFHS-5, ages 18-40, non-pregnant women + men, modelled separately", "paper": "NFHS-5 men and women"},
    "age_range": {"EarlyDia": "18-40", "paper": "15-49"},
    "target_definition": {"EarlyDia": "Target A: known diabetes OR valid glucose >=200 -> positive; explicit-No flags + glucose <=140 -> negative",
                          "paper": "Undiagnosed diabetes: 'no'/'don't know' to diagnosed diabetes AND (fasting >=126 mg/dL with >=8 h fasting OR random >=200 mg/dL)"},
    "glucose_definition": {"EarlyDia": "Single capillary glucose, >=200 positive, <=140 negative, 141-199 excluded", "paper": "Fasting >=126 (>=8 h) or random >=200"},
    "known_diabetes_handling": {"EarlyDia": "Known diabetes counted as positive", "paper": "Known/diagnosed diabetes is the comparison/excluded group; outcome is undiagnosed only"},
    "survey_weighting": {"EarlyDia": "Production model unweighted; weighted descriptive analysis is research-only", "paper": NV},
    "exclusions": {"EarlyDia": "Pregnancy (women), missing/ambiguous Target A", "paper": NV},
    "prevalence": {"EarlyDia": "~3.0-3.2% Target A (women test 3.03%); NOT comparable", "paper": NV},
}
GELDSETZER = {
    "reference": "Geldsetzer et al., JAMA Intern Med 2018",
    "use": "Historical/contextual Indian prevalence evidence only. Its national prevalence is NOT a target EarlyDia should reproduce.",
    "differences": ["Different survey (NFHS-4, 2015-16 vs NFHS-5, 2019-21)", "Adults 18+ vs EarlyDia 18-40",
                    "Different outcome definition (diabetes prevalence vs a model-ready screening label that excludes 141-199 mg/dL)",
                    "Prevalence study vs prediction/screening prototype"],
    "note": "Specific numeric details were not re-verified in this build.",
}
BARMAN_FIELDS = ["dataset", "population", "target", "predictors", "validation", "group_PSU_handling", "calibration",
                 "SMOTE", "model_types", "explainability", "thresholding", "DCA", "subgroup_evaluation"]
EARLYDIA_FIELDS = {
    "dataset": "NFHS-5 (2019-21) individual + men's recode", "population": "India, ages 18-40",
    "target": "Target A (locked)", "predictors": "age, BMI (women), BP, tobacco, alcohol, diet, education, residence",
    "validation": "Group-aware 80/20 split, StratifiedGroupKFold CV", "group_PSU_handling": "PSU grouping; no PSU in both train and test",
    "calibration": "Women isotonic, men sigmoid/Platt, fit on grouped OOF", "SMOTE": "None (class weighting)",
    "model_types": "HistGradientBoosting (LR baseline in model selection)", "explainability": "Permutation importance",
    "thresholding": "Sensitivity floor 0.75 on OOF: women 0.026, men 0.027", "DCA": "Research-only (reports/research/decision_curve.csv)",
    "subgroup_evaluation": "Age x sex (locked); others need row-level data",
}


def run():
    ensure_dirs()
    barman = {"reference": "Barman et al., Discover Public Health 2025",
              "caution": "Methodological differences that warrant consideration include the items below, once verified from the paper. No leakage claim is made.",
              "rows": {k: {"EarlyDia": EARLYDIA_FIELDS[k], "paper": NV} for k in BARMAN_FIELDS}}
    out = {"sahadevan_2023": SAHADEVAN, "geldsetzer_2018": GELDSETZER, "barman_2025": barman}
    (OUT / "literature_benchmark.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = ["# Literature benchmarking", "", "## Sahadevan et al. 2023", f"{SAHADEVAN['reference']}", "",
          "| Aspect | EarlyDia | Paper |", "|---|---|---|"]
    for k, v in SAHADEVAN.items():
        if isinstance(v, dict):
            md.append(f"| {k.replace('_', ' ')} | {v['EarlyDia']} | {v['paper']} |")
    md += ["", "## Geldsetzer et al. 2018", GELDSETZER["use"], ""] + [f"- {d}" for d in GELDSETZER["differences"]]
    md += ["", GELDSETZER["note"], "", "## Barman et al. 2025", barman["caution"], "",
           "| Aspect | EarlyDia | Paper |", "|---|---|---|"]
    md += [f"| {k.replace('_', ' ')} | {v['EarlyDia']} | {v['paper']} |" for k, v in barman["rows"].items()]
    (OUT / "literature_benchmark.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
