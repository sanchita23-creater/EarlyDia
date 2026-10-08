"""Re-export the existing permutation-importance results (NOT SHAP) as a research table.
Permutation importance measures contribution to model discrimination, not causation."""
import csv

from src.research.common import OUT, REPORTS, ensure_dirs

DIET = {"a": "Milk or curd", "b": "Pulses or beans", "c": "Dark green leafy vegetables", "d": "Fruits", "e": "Eggs",
        "f": "Fish", "g": "Chicken or meat", "h": "Fried food", "i": "Aerated drinks"}
NAMES = {"age": "Age", "bmi": "BMI", "avg_systolic_bp": "Systolic blood pressure", "avg_diastolic_bp": "Diastolic blood pressure",
         "education": "Education", "residence": "Residence",
         "n_valid_systolic_readings": "(survey artefact) valid systolic readings",
         "n_valid_diastolic_readings": "(survey artefact) valid diastolic readings"}


def readable(f):
    if f in NAMES:
        return NAMES[f]
    if f.startswith("diet_"):
        return "Diet: " + DIET.get(f[-1], f)
    return f


def run(top=15):
    ensure_dirs(); rows = []
    for sex in ("women", "men"):
        with open(REPORTS / "explainability" / f"{sex}_permutation_importance.csv", encoding="utf-8") as f:
            for i, r in enumerate(csv.DictReader(f), 1):
                if i > top:
                    break
                rows.append(dict(sex=sex, rank=i, feature=r["feature"], label=readable(r["feature"]),
                                 importance_mean=float(r["importance_mean"]), importance_std=float(r["importance_std"]),
                                 method="permutation importance (not SHAP)"))
    with open(OUT / "feature_importance.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows


if __name__ == "__main__":
    run()
