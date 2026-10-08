"""Research-only comparison with the Mohan et al. (JAPI 2005) Simplified Indian Diabetes Risk Score.

IDRS components (age, abdominal obesity/waist, physical activity, family history) are listed
below with their availability in EarlyDia's retained NFHS variables. Only age exists, so a
faithful IDRS cannot be computed and NO IDRS performance numbers are produced or invented.
Point values are intentionally not reproduced here because they were not verified against the paper in this build.
"""
import csv

from src.research.common import OUT, ensure_dirs

ROWS = [
    ("Age", "participant age in years", "AVAILABLE", "v012 / mv012 (model feature 'age')",
     "Available, but an age-only score is not the IDRS."),
    ("Abdominal obesity (waist circumference)", "waist circumference in cm", "UNAVAILABLE", "",
     "No waist/hip variable retained or administered in the extraction (extract_nfhs.py notes 'no waist/hip'). BMI is NOT a substitute."),
    ("Physical activity", "activity level (exercise / occupational)", "UNAVAILABLE", "",
     "No physical-activity variable retained."),
    ("Family history of diabetes", "parental diabetes", "UNAVAILABLE", "",
     "No family-history variable retained."),
]


def run():
    ensure_dirs()
    with open(OUT / "idrs_compatibility.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idrs_variable", "required_information", "earlydia_availability", "exact_mapping", "reason"])
        w.writerows(ROWS)
    with open(OUT / "idrs_comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["status", "sensitivity", "specificity", "ppv", "npv", "roc_auc", "pr_auc", "reason"])
        w.writerow(["NOT_COMPUTED", "", "", "", "", "", "",
                    "3 of 4 IDRS components unavailable (waist, physical activity, family history); a faithful IDRS cannot be reproduced and EarlyDia was not modified."])


if __name__ == "__main__":
    run()
