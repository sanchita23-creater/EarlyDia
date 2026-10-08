"""Alternative literature-aligned target (Sahadevan et al. 2023) - feasibility check.

Alternative literature-aligned target - NOT used by the EarlyDia production model.

Definition in the paper (retrieved from the open-access full text, PMC10704946): answered
'no' or "don't know" to diagnosed diabetes AND either opportunistic fasting glucose >=126 mg/dL
with >=8 h fasting, OR random glucose >=200 mg/dL; ages 15-49, women and men.

This module does NOT fabricate that target. It inspects what the retained extraction
(src/data/extract_nfhs.py) contains and records whether each ingredient is available.
"""
import csv

from src.data.extract_nfhs import WOMEN_COLUMNS, MEN_COLUMNS
from src.research.common import OUT, ensure_dirs, write_json

# The extraction keeps sb53/sb54/sb55/sb73 (women) and smb53/.../smb73 (men) as
# "fasting/measurement context", but NO project file or codebook excerpt in this repository
# states what each one means. Variable meanings must not be guessed -> status 'unverified'.
ROWS = [
    ("random/point glucose value (mg/dL)", "sb74 / smb74", "AVAILABLE (used by production Target A)"),
    ("known-diabetes self-report", "s728a, sb56, sb57 / sm627a, smb56, smb57", "AVAILABLE (used by production Target A)"),
    ("'don't know' diabetes answer kept distinct from 'no'", "s728a / sm627a raw codes", "AVAILABLE in raw interim file, but production Target A drops these rows as ambiguous"),
    ("fasting indicator", "sb53/sb54/sb55/sb73 (women), smb53/.../smb73 (men) retained as 'context'", "UNVERIFIED - meaning not documented in project; not guessed"),
    ("hours since last food (>=8 h rule)", "unknown", "UNVERIFIED - cannot confirm a duration variable exists among retained columns"),
    ("age 15-49 population", "v012 / mv012 (all 15-49/54 rows kept in interim file)", "AVAILABLE in interim file only (production restricts to 18-40)"),
    ("row-level data in this build environment", "data/interim, data/processed", "NOT PRESENT (raw NFHS microdata is not redistributed)"),
]


def run():
    ensure_dirs()
    present_ctx = [c for c in WOMEN_COLUMNS if c in ("sb53", "sb54", "sb55", "sb73")]
    rows = [dict(ingredient=a, variables=b, status=c) for a, b, c in ROWS]
    with open(OUT / "target_definition_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ingredient", "variables", "status"])
        w.writeheader(); w.writerows(rows)
    result = {
        "label": "Alternative literature-aligned target - NOT used by the EarlyDia production model",
        "full_replication_possible": False,
        "reason": ("Fasting status/duration variables cannot be verified from the retained extraction "
                   "documentation, and no row-level data is present to inspect their codes. No alternative "
                   "target was constructed and no alternative prevalence is reported."),
        "retained_context_columns_women": present_ctx,
        "ingredients": rows,
        "allowed_uses": ["prevalence sensitivity", "target-definition comparison", "robustness analysis"],
        "forbidden_uses": ["retraining", "threshold changes", "calibration changes", "inference changes", "final test set changes"],
        "to_complete": ("On a machine with data/interim/*.pkl and the NFHS-5 codebook: confirm sb53-sb73 meanings, then "
                        "implement the definition here (research-only) and report prevalence for ages 15-49 and 18-40."),
    }
    write_json(OUT / "target_definition_comparison.json", result)
    md = ["# Target definition comparison", "",
          "**Alternative literature-aligned target - NOT used by the EarlyDia production model.**", "",
          "| Aspect | EarlyDia Target A (production, locked) | Sahadevan et al. 2023 (J Glob Health) |", "|---|---|---|",
          "| Population | NFHS-5, ages 18-40, women non-pregnant + men | NFHS-5, ages 15-49, women and men |",
          "| Known diabetes | any of sb56/sb57/s728a (women), smb56/smb57/sm627a (men) = Yes -> positive | 'diagnosed' variable; undiagnosed = answered 'no' or 'don't know' |",
          "| Positive (glucose) | valid glucose >= 200 mg/dL, OR any known-diabetes flag | fasting >=126 mg/dL with >=8 h fasting, OR random >=200 mg/dL (among those not diagnosed) |",
          "| Negative | all flags explicit No, valid glucose, glucose <= 140 | not diagnosed and below the above glucose criteria |",
          "| Ambiguous handling | 141-199 mg/dL without known diabetes -> excluded | not excluded by glucose band |",
          "| Includes already-diagnosed people | Yes (known diabetes = positive) | No (outcome is *undiagnosed* diabetes) |", "",
          "## Replication status", "",
          "Full replication was **not possible** from the retained extraction: " + result["reason"], "",
          "Consequently no alternative-target prevalence is reported here. Prevalence from the two definitions must not be "
          "compared as like-for-like: the populations, age ranges, treatment of already-diagnosed people and glucose rules differ.", "",
          "This alternative target may only ever be used for prevalence sensitivity, target comparison and robustness analysis. "
          "It must never be used to retrain, rethreshold, recalibrate or alter inference.", ""]
    (OUT / "target_definition_comparison.md").write_text("\n".join(md), encoding="utf-8")
    return result


if __name__ == "__main__":
    run()
