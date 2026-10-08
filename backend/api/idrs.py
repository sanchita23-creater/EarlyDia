"""Indian Diabetes Risk Score (IDRS) - an INDEPENDENT, rule-based comparator.

Source: Mohan V, Deepa R, Deepa M, Somannavar S, Datta M. A simplified Indian Diabetes Risk
Score for screening for undiagnosed diabetic subjects. J Assoc Physicians India 2005;53:759-63.
The scoring table below was cross-checked against several published reproductions of the
original table (age, waist by sex, physical activity, family history; risk bands <30 / 30-50 / >=60).

IMPORTANT
- IDRS is NOT a feature of the EarlyDia machine-learning models. Its inputs (waist, physical
  activity, family history) are never passed to them. The models, thresholds and feature
  contract are untouched; this module only adds a second, published screening tool that the
  results page shows next to the ML result.
- It is a screening aid, not a diagnosis. Wording elsewhere must stay "associated with" /
  "screening", never "causes" or "predicts diabetes".
"""
from typing import Dict, List, Optional

CITATION = ("Mohan V et al. A simplified Indian Diabetes Risk Score for screening for "
            "undiagnosed diabetic subjects. J Assoc Physicians India 2005;53:759-63.")

# Waist (cm) bounds accepted by the API (sanity range, not a clinical statement).
WAIST_RANGE_CM = (50.0, 200.0)

# Physical-activity levels, scored exactly as in the original table.
ACTIVITY_OPTIONS = [
    {"id": "regular_both", "label": "Regular exercise AND strenuous work", "points": 0},
    {"id": "regular_or_strenuous", "label": "Regular exercise OR strenuous work", "points": 20},
    {"id": "sedentary", "label": "No exercise and sedentary work", "points": 30},
]
FAMILY_OPTIONS = [
    {"id": "none", "label": "No family history", "points": 0},
    {"id": "one_parent", "label": "One parent has diabetes", "points": 10},
    {"id": "both_parents", "label": "Both parents have diabetes", "points": 20},
]
_ACT = {o["id"]: o for o in ACTIVITY_OPTIONS}
_FAM = {o["id"]: o for o in FAMILY_OPTIONS}

# (female cut-offs, male cut-offs): waist >= first -> 10 points, >= second -> 20 points
_WAIST_CUTS = {"women": (80.0, 90.0), "men": (90.0, 100.0)}

MAX_SCORE = 100


def age_points(age: int) -> int:
    if age < 35:
        return 0
    if age < 50:
        return 20
    return 30


def waist_points(sex: str, waist_cm: float) -> int:
    mid, high = _WAIST_CUTS[sex]
    if waist_cm >= high:
        return 20
    if waist_cm >= mid:
        return 10
    return 0


def risk_category(score: int) -> str:
    """<30 low, 30-50 moderate, >=60 high. (Possible totals are multiples of 10.)"""
    if score >= 60:
        return "high"
    if score >= 30:
        return "moderate"
    return "low"


_CATEGORY_TEXT = {
    "low": "Lower IDRS risk band (score under 30)",
    "moderate": "Moderate IDRS risk band (score 30-50)",
    "high": "Higher IDRS risk band (score 60 or more)",
}


def validate(sex: str, waist_cm: Optional[float], activity: Optional[str],
             family: Optional[str]) -> List[str]:
    """Errors for the OPTIONAL IDRS answers. All three must be given together, or none."""
    given = [waist_cm is not None, activity is not None, family is not None]
    if not any(given):
        return []
    errs = []
    if not all(given):
        errs.append("For the IDRS comparison, answer waist circumference, physical activity "
                    "and family history together (or leave all three blank).")
        return errs
    lo, hi = WAIST_RANGE_CM
    if not (lo <= waist_cm <= hi):
        errs.append(f"Waist circumference should be between {lo:.0f} and {hi:.0f} cm.")
    if activity not in _ACT:
        errs.append("Select a valid physical activity level.")
    if family not in _FAM:
        errs.append("Select a valid family history option.")
    return errs


def compute_idrs(sex: str, age: int, waist_cm: float, activity: str, family: str) -> Dict:
    """Return the IDRS total, band and per-component breakdown. Inputs must already be valid."""
    comps = [
        {"label": "Age", "answer": f"{age} years", "points": age_points(age)},
        {"label": "Waist circumference", "answer": f"{waist_cm:g} cm", "points": waist_points(sex, waist_cm)},
        {"label": "Physical activity", "answer": _ACT[activity]["label"], "points": _ACT[activity]["points"]},
        {"label": "Family history", "answer": _FAM[family]["label"], "points": _FAM[family]["points"]},
    ]
    score = sum(c["points"] for c in comps)
    cat = risk_category(score)
    return {
        "score": score,
        "max_score": MAX_SCORE,
        "category": cat,
        "category_label": _CATEGORY_TEXT[cat],
        "components": comps,
        "citation": CITATION,
        "note": ("IDRS is a separate, published rule-based screening tool. Its inputs "
                 "(waist, physical activity, family history) are not used by the EarlyDia "
                 "machine-learning model. Neither is a diagnosis."),
    }


def idrs_config() -> Dict:
    return {
        "waist_range_cm": list(WAIST_RANGE_CM),
        "activity": [{"value": o["id"], "label": o["label"]} for o in ACTIVITY_OPTIONS],
        "family_history": [{"value": o["id"], "label": o["label"]} for o in FAMILY_OPTIONS],
        "waist_note": {"women": "Points start at 80 cm", "men": "Points start at 90 cm"},
        "citation": CITATION,
    }
