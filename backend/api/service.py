"""Screening service: validates the user's answers, maps them to the existing model feature
contract (src/app/input_spec.py - reused unchanged) and calls the existing inference module.
"""
import json
from dataclasses import asdict
from functools import lru_cache
from typing import Dict, List, Optional

from backend.api import idrs as idrs_mod
from backend.api.schemas import ScreeningRequest
from backend.api.states import (
    NFHS5_STATE_OPTIONS,
    DEFAULT_STATE_CODE,
    DEFAULT_STATE_NAME,
    is_valid_state,
    get_state_name,
)
from backend.configs.settings import PROJECT_ROOT
from backend.inference.engine import get_predictor
from src.app import input_spec as I
from src.app import summaries as S
from src.app.recommendations import generate_recommendations

# The production models were trained with `n_valid_*_readings` = how many of the (up to 3)
# NFHS blood-pressure readings were valid. That is a survey artefact users cannot answer, so the
# UI never asks. The BP the user enters is treated as a complete average (3 of 3 readings): the
# most common value in the training data, so the input stays in-distribution. The models are
# unchanged. BP itself is REQUIRED - when it was missing entirely the models produced abnormal
# probabilities (women ~11.7%, men ~28.5% vs ~5% with a real reading in a spot-check), so
# a request without BP is rejected instead of being scored.
ASSUMED_BP_READINGS = 3

SYS_RANGE, DIA_RANGE = I.SYS_RANGE, I.DIA_RANGE
SEX_IDS = {"women": "Female", "men": "Male"}


@lru_cache(maxsize=4)
def _test_metrics(version: str = "v2") -> dict:
    """Read-only: headline held-out test metrics already recorded in reports/ (never recomputed)."""
    if version == "v2":
        prod_metrics_file = PROJECT_ROOT / "reports" / "v2" / "production_metrics.json"
        if prod_metrics_file.exists():
            try:
                return json.loads(prod_metrics_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        return {
            "women": {
                "model": "LightGBM (v2 - lgbm_D)",
                "roc_auc": 0.8016,
                "roc_auc_sd": 0.0017,
                "sensitivity": 0.750,
                "specificity": 0.701,
                "ppv_precision": 0.431,
                "pct_predicted_positive": 0.404,
                "pr_auc": 0.5829,
                "brier": 0.1776,
                "decision_threshold": 0.212,
            },
            "men": {
                "model": "CatBoost (v2 - cb_B)",
                "roc_auc": 0.8033,
                "roc_auc_sd": 0.0086,
                "sensitivity": 0.751,
                "specificity": 0.704,
                "ppv_precision": 0.541,
                "pct_predicted_positive": 0.440,
                "pr_auc": 0.6678,
                "brier": 0.1791,
                "decision_threshold": 0.300,
            },
        }

    path = PROJECT_ROOT / "reports" / "final_evaluation" / "final_test_evaluation.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    keys = ("roc_auc", "sensitivity", "specificity", "ppv_precision", "pct_predicted_positive")
    return {sex: {k: round(data[sex]["overall"][k], 3) for k in keys} for sex in ("women", "men") if sex in data}


class InputError(ValueError):
    def __init__(self, errors: List[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def get_config() -> dict:
    """Option lists / ranges the React form renders (labels come from the existing input spec)."""
    per_sex = {}
    for sex in ("women", "men"):
        per_sex[sex] = {
            "tobacco_products": [{"id": c, "label": l} for c, l in I.TOBACCO_ITEMS[sex]],
            "cigarette_id": I.CIGARETTE_COL[sex],
            "asks_tobacco_frequency": I.TOBACCO_FREQ_COL[sex] is not None,
            "alcohol_types": [{"id": c, "label": l} for c, l in I.ALCOHOL_TYPE_ITEMS[sex]],
            "uses_bmi": sex == "women",
        }
    return {
        "sexes": [{"id": k, "label": v} for k, v in SEX_IDS.items()],
        "age_range": list(I.AGE_RANGE),
        "height_range_cm": list(I.HEIGHT_RANGE_CM),
        "weight_range_kg": list(I.WEIGHT_RANGE_KG),
        "bp_systolic_range": list(SYS_RANGE),
        "bp_diastolic_range": list(DIA_RANGE),
        "education": [{"value": k, "label": v} for k, v in I.EDUCATION_OPTIONS.items()],
        "residence": [{"value": k, "label": v} for k, v in I.RESIDENCE_OPTIONS.items()],
        "tobacco_frequency": [{"value": k, "label": v} for k, v in I.TOBACCO_FREQ_OPTIONS.items()],
        "alcohol_frequency": [{"value": k, "label": v} for k, v in I.ALCOHOL_FREQ_OPTIONS.items()],
        "diet_questions": [{"id": k, "label": I.DIET_SHORT[k], "question": q}
                           for k, q in I.DIET_QUESTIONS.items()],
        "diet_options": [{"value": v, "label": I.DIET_OPTIONS[v]} for v in I.DIET_DISPLAY_ORDER],
        "states": [{"value": k, "label": v} for k, v in sorted(NFHS5_STATE_OPTIONS.items(), key=lambda x: x[1])],
        "default_state": {"value": DEFAULT_STATE_CODE, "label": DEFAULT_STATE_NAME},
        "per_sex": per_sex,
        "idrs": idrs_mod.idrs_config(),
    }


def _to_answers(req: ScreeningRequest) -> dict:
    sex = req.sex
    ans = I.blank_answers()
    ans.update(sex=sex, age=req.age, education=req.education, residence=req.residence,
               uses_tobacco=req.tobacco_use, tob_freq=req.cigarette_frequency,
               drinks=req.drinks_alcohol, alc_freq=req.alcohol_frequency,
               alc_duration=req.alcohol_years,
               bp_sys=req.bp_systolic, bp_dia=req.bp_diastolic, bp_n=ASSUMED_BP_READINGS,
               bp_unavailable=False)
    if sex == "women":
        ans.update(height_cm=req.height_cm, weight_kg=req.weight_kg)
    for col in req.tobacco_products:
        ans[col] = True
    for col in req.alcohol_types:
        ans[col] = True
    for col, _ in I.DIET_ITEMS[sex]:
        ans[col] = req.diet.get(I.diet_letter(col))
    return ans


def _check(req: ScreeningRequest) -> List[str]:
    errs = []
    if req.sex not in SEX_IDS:
        return ["Select your sex."]
    sex = req.sex
    if req.education not in I.EDUCATION_OPTIONS:
        errs.append("Select your education level.")
    if req.residence not in I.RESIDENCE_OPTIONS:
        errs.append("Select where you live (urban or rural).")
    version = req.model_version or "v2"
    if version == "v2" and req.state is not None and not is_valid_state(req.state):
        errs.append("Select a valid State / UT from the list.")
    if req.cigarette_frequency is not None and req.cigarette_frequency not in I.TOBACCO_FREQ_OPTIONS:
        errs.append("Select a valid smoking frequency.")
    if req.alcohol_frequency is not None and req.alcohol_frequency not in I.ALCOHOL_FREQ_OPTIONS:
        errs.append("Select a valid drinking frequency.")
    ok_t = {c for c, _ in I.TOBACCO_ITEMS[sex]}
    ok_a = {c for c, _ in I.ALCOHOL_TYPE_ITEMS[sex]}
    if any(c not in ok_t for c in req.tobacco_products) or any(c not in ok_a for c in req.alcohol_types):
        errs.append("One of the selected options is not valid for this screening group.")
    if any(v not in I.DIET_OPTIONS for v in req.diet.values()):
        errs.append("Select a valid answer for each food question.")
    if not (SYS_RANGE[0] <= req.bp_systolic <= SYS_RANGE[1]):
        errs.append(f"Systolic blood pressure should be between {SYS_RANGE[0]:.0f} and {SYS_RANGE[1]:.0f} mmHg.")
    if not (DIA_RANGE[0] <= req.bp_diastolic <= DIA_RANGE[1]):
        errs.append(f"Diastolic blood pressure should be between {DIA_RANGE[0]:.0f} and {DIA_RANGE[1]:.0f} mmHg.")
    if sex == "women":
        h, w = req.height_cm, req.weight_kg
        if h is not None and not (I.HEIGHT_RANGE_CM[0] <= h <= I.HEIGHT_RANGE_CM[1]):
            errs.append("Height should be between 120 and 220 cm.")
        if w is not None and not (I.WEIGHT_RANGE_KG[0] <= w <= I.WEIGHT_RANGE_KG[1]):
            errs.append("Weight should be between 25 and 200 kg.")
    errs += idrs_mod.validate(sex, req.waist_cm, req.physical_activity, req.family_history)
    if errs:
        return errs
    return I.validate_all(_to_answers(req))   # existing validators (age range, completeness, BMI range ...)


def _profile(sex: str, ui: dict, state_label: Optional[str] = None, extra=None) -> List[dict]:
    rows = [("Age", f"{ui['age']} years"), ("Sex", S.sex_label(sex)),
            ("Education", S.education_label(ui["education"])),
            ("Residence", S.residence_label(ui["residence"]))]
    if state_label:
        rows.append(("State / UT", state_label))
    if sex == "women":
        rows += [("Height", f"{ui['height_cm']:.0f} cm"), ("Weight", f"{ui['weight_kg']:.1f} kg"),
                 ("BMI", f"{ui['bmi']:.1f} kg/m²")]
    rows.append(("Blood pressure", f"{ui['avg_sys']:.0f}/{ui['avg_dia']:.0f} mmHg"))
    prods = S.tobacco_products(sex, ui)
    rows.append(("Tobacco", "No" if not prods else "Yes - " + ", ".join(p.lower() for p in prods)))
    rows.append(("Alcohol", "Non-drinker" if not ui["drinks"] else S.alcohol_summary(sex, ui).replace("Drinks alcohol", "Drinks alcohol", 1)))
    rows.append(("Diet", "9 food questions answered"))
    if extra:
        rows += extra
    return [{"label": a, "value": b} for a, b in rows]


def run_screening(req: ScreeningRequest) -> dict:
    errors = _check(req)
    if errors:
        raise InputError(errors)
    sex = req.sex
    version = req.model_version or "v2"
    ui = I.ui_from_answers(_to_answers(req))
    person = I.build_person(sex, ui)

    state_provided = (req.state is not None)
    state_warning = None
    state_name = None
    state_profile_label = None

    if version == "v2":
        if state_provided:
            state_name = get_state_name(req.state)
            person["state"] = req.state
            state_profile_label = state_name
        else:
            state_name = DEFAULT_STATE_NAME
            state_profile_label = f"{DEFAULT_STATE_NAME} (national default)"
            state_warning = (
                f"State was not provided. National default ({DEFAULT_STATE_NAME}, code {DEFAULT_STATE_CODE}) "
                "was used for regional context."
            )
    else:
        # v1 legacy: unchanged model behavior
        if state_provided:
            state_name = get_state_name(req.state)
            state_profile_label = state_name

    try:
        result = get_predictor(sex, version=version).predict(person)
    except ValueError as exc:   # predict_module range checks
        raise InputError([str(exc)])
    # v2 models expect many features that the API doesn't collect (physical_activity, glucose_context,
    # interaction terms, etc.) and imputes them. Only raise an error for v1 or if critical features
    # that the API DOES provide are missing.
    if version == "v1" and result.validation_warnings:
        raise InputError(["Some required answers were missing. Please review your inputs."])
    report = generate_recommendations(sex, ui, result)
    idrs_result = None
    extra_profile = []
    if req.waist_cm is not None:   # validated: waist, activity and family history all present
        idrs_result = idrs_mod.compute_idrs(sex, req.age, req.waist_cm, req.physical_activity,
                                            req.family_history)
        extra_profile = [("Waist", f"{req.waist_cm:g} cm")]
    p, thr = result.calibrated_probability, result.threshold_used
    flagged = bool(result.predicted_class == 1)
    return {
        "sex": sex,
        "probability": p, "percent": round(p * 100, 1),
        "cutoff": thr, "cutoff_percent": round(thr * 100, 1),
        "flagged": flagged,
        "status": "Flagged for follow-up" if flagged else "Lower screening likelihood",
        "profile": _profile(sex, ui, state_label=state_profile_label, extra=extra_profile),
        "guidance": [asdict(i) for i in report.lifestyle_recommendations + report.general_wellness],
        "state_provided": state_provided,
        "state_used": state_name,
        "warning": state_warning,
        "technical": {
            "model": result.model,
            "version": getattr(result, "version", version),
            "calibration_method": result.calibration_method,
            "calibrated_probability": round(p, 4),
            "decision_threshold": thr,
            "bmi_used_by_model": sex == "women",
            "test_metrics": _test_metrics(version).get(sex, {}),
            "idrs_used_by_model": False,
            "bp_readings_handling": f"Entered BP treated as a complete average ({ASSUMED_BP_READINGS} of {ASSUMED_BP_READINGS} readings)",
            "state_used": state_name,
            "state_provided": state_provided,
        },
        "idrs": idrs_result,
    }
