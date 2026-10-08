"""
EarlyDia - Personalized Recommendations Module
==============================================
Builds structured, safely-worded guidance from the user's answers and the screening result.

Architecture note: strictly decoupled from the ML pipeline. It consumes the `ui` answers dict
and the PredictionResult and returns display data. It does NOT diagnose, prescribe medication,
promise prevention/cure, or claim that any single factor caused the screening result.

Terminology: the result is always a "diabetes-range screening classification" (internal note: the target
is the NFHS-5-derived Target A documented in the README).
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.app.input_spec import compute_bmi, bmi_band
from src.app import summaries as S


@dataclass
class RecommendationItem:
    category: str            # e.g. "Tobacco Use", "Blood Pressure", "Physical Activity"
    title: str
    summary: str
    action_items: List[str]
    status_tag: str
    tag_type: str            # "alert" | "ok" | "info" | "neutral"


@dataclass
class RecommendationReport:
    sex: str
    calibrated_probability: float
    threshold_used: float
    predicted_class: int
    is_flagged: bool
    summary_headline: str
    clinical_next_steps: RecommendationItem
    lifestyle_recommendations: List[RecommendationItem] = field(default_factory=list)
    general_wellness: List[RecommendationItem] = field(default_factory=list)
    factors_considered: List[Dict[str, str]] = field(default_factory=list)


def _clinical_item(is_flagged: bool) -> RecommendationItem:
    if is_flagged:
        return RecommendationItem(
            category="Clinical Next Steps",
            title="Consider discussing this result with a healthcare professional",
            summary=("Your screening probability is at or above the screening threshold. This is NOT a medical "
                     "diagnosis of diabetes. It means the information you gave is associated with a higher estimated "
                     "probability for the diabetes-range screening target."),
            action_items=[
                "Consider booking a routine visit with a qualified doctor to talk about this screening result.",
                "Ask whether a laboratory test such as fasting plasma glucose or HbA1c is appropriate for you - "
                "only a clinical test can show whether blood sugar is in a healthy range.",
                "Share your blood-pressure readings and any family health history with the doctor.",
                "Check with a healthcare professional before starting supplements or restrictive diets.",
            ],
            status_tag="Follow-up suggested", tag_type="alert")
    return RecommendationItem(
        category="Clinical Next Steps",
        title="Keep up routine health check-ups",
        summary=("Your screening probability is below the screening threshold. Please note that this does not "
                 "guarantee the absence of diabetes or prediabetes - EarlyDia has not ruled anything out and "
                 "is not a substitute for a clinical blood test."),
        action_items=[
            "Continue regular adult health check-ups, including blood pressure and blood-glucose checks as "
            "advised by your doctor.",
            "Seek medical advice promptly if you notice persistent excessive thirst, frequent urination, "
            "unexplained tiredness or unexplained weight change.",
        ],
        status_tag="Routine monitoring", tag_type="ok")


def _bp_item(ui: Dict[str, Any]):
    n_sys = int(ui.get("n_sys", 0) or 0)
    sys_bp, dia_bp = ui.get("avg_sys"), ui.get("avg_dia")
    if n_sys == 0 or sys_bp is None or dia_bp is None:
        return RecommendationItem(
            category="Blood Pressure", title="Get a blood-pressure measurement",
            summary="You did not have a recent blood-pressure measurement, so the model handled it as missing. "
                    "Blood pressure needs an actual measurement - it cannot be judged from how you feel.",
            action_items=[
                "Have your blood pressure measured with a validated cuff at a clinic, pharmacy or health camp.",
                "Rest quietly for about 5 minutes, seated, before the reading.",
                "Re-run EarlyDia with the measured values if you want a more complete screening.",
            ],
            status_tag="Measurement needed", tag_type="info")
    sys_bp, dia_bp = float(sys_bp), float(dia_bp)
    shown = f"{sys_bp:.0f}/{dia_bp:.0f} mmHg"
    if sys_bp >= 140 or dia_bp >= 90:
        return RecommendationItem(
            category="Blood Pressure", title="Your blood-pressure readings are on the higher side",
            summary=(f"The readings you entered ({shown}) are at or above 140/90 mmHg. A few readings do not make "
                     "a diagnosis, but it is worth getting them checked."),
            action_items=[
                "Consider monitoring your blood pressure and discussing persistently elevated readings with a "
                "healthcare professional.",
                "Moderating salt (general guidance: under about 5 g of salt a day) is a widely recommended habit.",
                "Fruits and vegetables are good everyday sources of potassium unless a doctor has advised otherwise.",
            ],
            status_tag="Worth a check", tag_type="alert")
    if sys_bp >= 130 or dia_bp >= 80:
        return RecommendationItem(
            category="Blood Pressure", title="Your blood pressure is slightly above the ideal range",
            summary=f"The readings you entered ({shown}) are above 120/80 mmHg but below 140/90 mmHg.",
            action_items=[
                "Check your blood pressure from time to time to see how it changes.",
                "A diet rich in vegetables, fruit and whole grains with less salt is a commonly recommended pattern.",
            ],
            status_tag="Borderline", tag_type="info")
    return RecommendationItem(
        category="Blood Pressure", title="Your blood-pressure readings look in the usual range",
        summary=f"The readings you entered ({shown}) are below 130/80 mmHg.",
        action_items=["Keep checking blood pressure at routine health visits.",
                      "Regular physical activity and a balanced diet support healthy blood pressure."],
        status_tag="Optimal Range", tag_type="ok")


def _tobacco_item(sex: str, ui: Dict[str, Any]) -> RecommendationItem:
    prods = S.tobacco_products(sex, ui)
    if prods:
        names = ", ".join(p[0].lower() + p[1:] for p in prods)
        return RecommendationItem(
            category="Tobacco Use", title="Avoiding tobacco is one of the most useful general health steps",
            summary=("Your responses indicate tobacco use (" + names + "). Avoiding tobacco is an important general "
                     "health measure and is associated with lower type 2 diabetes risk."),
            action_items=[
                "A healthcare professional can explain evidence-based support for quitting.",
                "In India, the National Tobacco Quitline offers free support: 1800-11-2356.",
                "Writing down when cravings happen can make it easier to plan substitutes.",
            ],
            status_tag="Cessation Advised", tag_type="alert")
    return RecommendationItem(
        category="Tobacco Use", title="Staying tobacco-free",
        summary="You reported no tobacco use. Staying tobacco-free is good for general health.",
        action_items=["Keep avoiding tobacco, and avoid second-hand smoke where you can."],
        status_tag="Protective Habit", tag_type="ok")


def _alcohol_item(sex: str, ui: Dict[str, Any]) -> RecommendationItem:
    if ui.get("drinks"):
        years = float(ui.get("alc_duration", 0.0) or 0.0)
        freq = ui.get("alc_freq")
        how = S.ALCOHOL_FREQ_OPTIONS.get(freq, "").lower()
        return RecommendationItem(
            category="Alcohol Intake", title="Keeping alcohol intake low",
            summary=(f"You reported drinking alcohol{(' ' + how) if how else ''} for about {years:.0f} years. "
                     "Drinking less is generally better for blood pressure, weight and overall health."),
            action_items=[
                "If you drink, staying within recognised low-risk limits and keeping alcohol-free days is advisable.",
                "Avoid binge drinking.",
                "A healthcare professional can help if cutting down feels difficult.",
            ],
            status_tag="Moderation Advised", tag_type="info")
    return RecommendationItem(
        category="Alcohol Intake", title="Not drinking alcohol",
        summary="You reported that you do not drink alcohol.",
        action_items=["There is no health reason to start drinking alcohol."],
        status_tag="Protective Habit", tag_type="ok")


def _diet_item(sex: str, ui: Dict[str, Any]) -> RecommendationItem:
    v = lambda letter: S.diet_value(sex, ui, letter)   # noqa: E731
    tips, noted = [], []
    if v("h") in (1, 2):
        noted.append("fried food " + ("daily" if v("h") == 1 else "weekly"))
        tips.append("Consider swapping some fried snacks for roasted chana, boiled pulses, nuts or fresh fruit.")
    if v("i") in (1, 2):
        noted.append("aerated drinks " + ("daily" if v("i") == 1 else "weekly"))
        tips.append("Consider cutting back on sugary and aerated drinks - water, buttermilk or unsweetened "
                    "lime water are simple swaps.")
    if v("d") in (0, 3):
        noted.append("fruit " + ("never" if v("d") == 0 else "only occasionally"))
        tips.append("Consider adding a serving of whole seasonal fruit on most days (whole fruit, not juice).")
    if v("c") in (0, 3):
        noted.append("dark green leafy vegetables " + ("never" if v("c") == 0 else "only occasionally"))
        tips.append("Consider including dark green leafy vegetables more often for fibre and micronutrients.")
    if v("b") in (0, 3):
        noted.append("pulses " + ("never" if v("b") == 0 else "only occasionally"))
        tips.append("Pulses and beans are an affordable source of fibre and protein worth including regularly.")
    if tips:
        summary = ("From your answers (" + "; ".join(noted) + "), these are areas where small changes could help. "
                   "A varied diet with more vegetables, whole fruit and high-fibre foods, and fewer highly processed "
                   "foods, is the general pattern recommended for metabolic health.")
        tag, ttype = "Room to improve", "info"
    else:
        tips = ["Keep a varied diet built around vegetables, whole fruit, pulses and whole grains.",
                "Keep an eye on how often processed or deep-fried foods and sugary drinks creep in."]
        summary = "Your food answers do not show the specific patterns this guidance looks for. Keep it up."
        tag, ttype = "Looks balanced", "ok"
    return RecommendationItem(category="Dietary Pattern", title="Everyday food choices",
                              summary=summary, action_items=tips, status_tag=tag, tag_type=ttype)


def _weight_item(sex: str, ui: Dict[str, Any]) -> RecommendationItem:
    bmi = ui.get("bmi")
    if sex == "women" and bmi is not None:
        bmi = float(bmi)
        band = bmi_band(bmi)
        if bmi >= 23.0:
            return RecommendationItem(
                category="Weight-Related Habits", title=f"Your BMI is {bmi:.1f} kg/m²",
                summary=(f"Using the WHO cut-offs for Asian populations, this is in the '{band.lower()}' band "
                         "(23.0 or more). BMI is a rough indicator and does not describe body composition or health on its own."),
                action_items=[
                    "Gradual, sustainable changes - more daily movement and balanced portions - are generally "
                    "recommended over crash diets.",
                    "A doctor or registered dietitian can suggest a plan that suits you.",
                ],
                status_tag="Habits to consider", tag_type="info")
        if bmi < 18.5:
            return RecommendationItem(
                category="Weight-Related Habits", title=f"Your BMI is {bmi:.1f} kg/m²",
                summary="This is below the typical range (18.5-22.9). BMI is only a rough indicator.",
                action_items=["Consider talking with a healthcare professional about whether your eating "
                              "provides enough energy and nutrients."],
                status_tag="Worth discussing", tag_type="info")
        return RecommendationItem(
            category="Weight-Related Habits", title=f"Your BMI is {bmi:.1f} kg/m²",
            summary="This falls in the typical range (18.5-22.9) for the WHO Asian-population cut-offs.",
            action_items=["Balanced eating and regular activity help maintain a steady weight."],
            status_tag="Typical range", tag_type="ok")
    return RecommendationItem(
        category="Weight-Related Habits", title="Healthy weight habits",
        summary="BMI is not used for this screening group, so no BMI-based comment is made here.",
        action_items=["Healthy weight and waist size can be checked at a routine health visit.",
                      "Balanced eating and regular activity help maintain a steady weight."],
        status_tag="General guidance", tag_type="neutral")


def _general_items() -> List[RecommendationItem]:
    return [
        RecommendationItem(
            category="Physical Activity", title="Regular movement",
            summary="Regular physical activity supports blood-sugar control, blood pressure and general health.",
            action_items=[
                "WHO guidance for adults is at least 150 minutes of moderate activity a week "
                "(for example, brisk walking for 30 minutes on 5 days).",
                "Add muscle-strengthening activity on 2 or more days a week.",
                "Break up long periods of sitting with short walks.",
            ],
            status_tag="General guidance", tag_type="info"),
        RecommendationItem(
            category="General Preventive Health", title="Sleep, stress and check-ups",
            summary="Everyday habits and regular check-ups are the foundation of preventive health.",
            action_items=[
                "Most adults do best with about 7 to 9 hours of sleep a night.",
                "Walks, breathing exercises or time with friends can help manage stress.",
                "Ask your doctor how often blood pressure and blood-glucose checks are right for you.",
            ],
            status_tag="General guidance", tag_type="info"),
    ]


def _factors(sex: str, ui: Dict[str, Any]) -> List[Dict[str, str]]:
    f: List[Dict[str, str]] = []
    age = ui.get("age")
    f.append({"factor": "Age", "value": f"{age} years" if age is not None else "-", "note": ""})
    if sex == "women":
        bmi = ui.get("bmi")
        if bmi is None and ui.get("height_cm") and ui.get("weight_kg"):
            bmi = compute_bmi(ui["height_cm"], ui["weight_kg"])
        if bmi is not None:
            extra = (f" (height {ui['height_cm']:.0f} cm, weight {ui['weight_kg']:.1f} kg)"
                     if ui.get("height_cm") and ui.get("weight_kg") else "")
            f.append({"factor": "Body Mass Index (BMI)", "value": f"{float(bmi):.1f} kg/m²{extra}", "note": ""})
        else:
            f.append({"factor": "Body Mass Index (BMI)", "value": "Not provided", "note": ""})
    else:
        f.append({"factor": "Body Mass Index (BMI)",
                  "value": "Not used for this screening group", "note": ""})
    f.append({"factor": "Blood pressure", "value": S.bp_summary(ui), "note": ""})
    f.append({"factor": "Tobacco", "value": S.tobacco_summary(sex, ui), "note": ""})
    f.append({"factor": "Alcohol", "value": S.alcohol_summary(sex, ui), "note": ""})
    f.append({"factor": "Diet", "value": "Your answers for 9 food groups (shown below)", "note": ""})
    f.append({"factor": "Education", "value": S.education_label(ui.get("education")), "note": ""})
    f.append({"factor": "Residence", "value": S.residence_label(ui.get("residence")), "note": ""})
    return f


def generate_recommendations(sex: str, ui_inputs: Dict[str, Any], pred_result: Any) -> RecommendationReport:
    """Build the recommendation report. `ui_inputs` follows input_spec.ui_from_answers();
    missing keys are tolerated. `pred_result` is a PredictionResult (or any object exposing
    calibrated_probability, threshold_used and predicted_class)."""
    cal_prob = float(pred_result.calibrated_probability)
    threshold = float(pred_result.threshold_used)
    is_flagged = bool(pred_result.predicted_class == 1)
    prob_pct, th_pct = cal_prob * 100, threshold * 100

    headline = (f"Screening result: {'At or Above' if is_flagged else 'Below'} the screening threshold "
                f"({prob_pct:.2f}% vs {th_pct:.2f}%)")

    ui = dict(ui_inputs)
    lifestyle = [
        _bp_item(ui), _tobacco_item(sex, ui), _alcohol_item(sex, ui),
        _diet_item(sex, ui), _weight_item(sex, ui),
    ]
    return RecommendationReport(
        sex=sex, calibrated_probability=cal_prob, threshold_used=threshold,
        predicted_class=1 if is_flagged else 0, is_flagged=is_flagged, summary_headline=headline,
        clinical_next_steps=_clinical_item(is_flagged),
        lifestyle_recommendations=lifestyle, general_wellness=_general_items(),
        factors_considered=_factors(sex, ui),
    )
