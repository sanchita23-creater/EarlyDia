"""
EarlyDia - end-to-end tests of the multi-step product (headless Streamlit AppTest) plus the
UI -> model input contract. These run the REAL app script and the REAL production predictor.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app import input_spec as I
from src.app.gauge import render_semicircular_gauge, pct_to_fraction
from src.inference.predict_module import EarlyDiaPredictor

APP = str(ROOT / "src" / "app" / "app.py")


def _answers(sex, **over):
    a = I.blank_answers()
    a.update(sex=sex, age=30, education=2, residence=1, uses_tobacco=False, drinks=False,
             bp_sys=118.0, bp_dia=76.0, bp_n=3)
    if sex == "women":
        a.update(height_cm=160.0, weight_kg=60.0)
    for col, _ in I.DIET_ITEMS[sex]:
        a[col] = 2
    a.update(over)
    return a


# ---------------- input contract ----------------
def test_bmi_from_height_weight():
    assert I.compute_bmi(160, 60) == 23.4
    assert I.compute_bmi(None, 60) is None and I.compute_bmi(160, 0) is None


@pytest.mark.parametrize("sex", ["women", "men"])
def test_ui_answers_map_exactly_to_model_feature_contract(sex):
    ans = _answers(sex)
    assert I.validate_all(ans) == []
    ui = I.ui_from_answers(ans)
    person = I.build_person(sex, ui)
    cols = set(EarlyDiaPredictor(sex).feature_cols)
    assert set(person) == cols, (sorted(set(person) ^ cols))
    if sex == "men":
        assert "bmi" not in person and "bmi" not in ui          # no BMI is ever sent to the men's model
    else:
        assert person["bmi"] == 23.4


@pytest.mark.parametrize("sex", ["women", "men"])
def test_real_inference_for_both_models_with_locked_thresholds(sex):
    ui = I.ui_from_answers(_answers(sex))
    res = EarlyDiaPredictor(sex).predict(I.build_person(sex, ui))
    assert 0 <= res.calibrated_probability <= 1
    assert res.threshold_used == {"women": 0.026, "men": 0.027}[sex]
    assert res.predicted_class == int(res.calibrated_probability >= res.threshold_used)


def test_men_validation_does_not_require_height_weight_and_women_does():
    assert I.validate_health(_answers("men")) == []
    bad = _answers("women", height_cm=None)
    assert any("height" in e.lower() for e in I.validate_health(bad))


def test_validation_blocks_incomplete_and_inconsistent_answers():
    assert I.validate_profile(I.blank_answers())
    assert any("Years of drinking" in e for e in
               I.validate_lifestyle(_answers("men", drinks=True, alc_freq=1, alc_duration=45, age=30)))
    assert any("Systolic" in e for e in I.validate_health(_answers("men", bp_sys=80.0, bp_dia=90.0)))
    assert I.validate_health(_answers("men", bp_unavailable=True, bp_sys=None, bp_dia=None, bp_n=None)) == []


# ---------------- gauge ----------------
def test_gauge_is_semicircle_with_needle_threshold_and_no_accuracy_wording():
    svg = render_semicircular_gauge(2.72, 2.60, True, False)
    for must in ("<svg", "<polygon", ">Threshold</text>", "2.60%", "2.72%", "Screening probability", "aria-label"):
        assert must in svg
    low = svg.lower()
    assert "accuracy" not in low.replace("not accuracy", "") and "confidence" not in low and "certain" not in low
    assert pct_to_fraction(0) == 0 and pct_to_fraction(100) == 1
    assert pct_to_fraction(2.60) < pct_to_fraction(2.72) < pct_to_fraction(5)   # needle past threshold, ordered
    # 97.28 % must not be implied: a 2.72 % reading sits near the LEFT end of the dial
    assert pct_to_fraction(2.72) < 0.2


# ---------------- full app flow (headless) ----------------
def _at():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def _click(at, label):
    [b for b in at.button if b.label.startswith(label)][0].click().run()


def _fill_widgets(at, ans):
    """Set each widget the way a user would, via the keys the app uses."""
    for name, val in ans.items():
        key = f"w_{name}"
        for coll in (at.radio, at.selectbox, at.number_input, at.checkbox):
            for w in coll:
                if w.key == key and val is not None:
                    w.set_value(val)
    at.run()


def test_full_flow_women_then_results_recommendations_and_reset():
    at = _at()
    assert any("Early diabetes risk screening" in m.value for m in at.markdown)
    assert any(b.label.startswith("Start Screening") for b in at.button)
    _click(at, "Start Screening")
    ans = _answers("women")
    _fill_widgets(at, {k: ans[k] for k in ("sex", "age", "education", "residence")})
    _click(at, "Continue to Lifestyle")
    assert at.session_state["page"] == "lifestyle"
    _fill_widgets(at, {k: v for k, v in ans.items() if k in ("uses_tobacco", "drinks") or k.startswith("diet_")})
    _click(at, "Continue to Health")
    assert at.session_state["page"] == "health"
    _fill_widgets(at, {k: ans[k] for k in ("height_cm", "weight_kg", "bp_sys", "bp_dia", "bp_n")})
    assert any("Your BMI: 23.4" in m.value for m in at.markdown)           # BMI shown, never typed
    _click(at, "Continue to Review")
    assert at.session_state["page"] == "review"
    _click(at, "Run EarlyDia Screening")
    assert at.session_state["page"] == "result" and not at.exception
    res = at.session_state["result"]
    assert res.threshold_used == 0.026 and res.sex == "women"
    page = " ".join(m.value for m in at.markdown)
    assert "Result:" in page and "Your EarlyDia Result" in page and "<svg" in page and "View Personalized Recommendations" in " ".join(b.label for b in at.button)
    _click(at, "View Personalized Recommendations")
    assert at.session_state["page"] == "recs" and not at.exception
    assert "Personalized recommendations" in " ".join(m.value for m in at.markdown)
    _click(at, "Start New Screening")
    assert at.session_state["page"] == "landing"
    assert at.session_state["result"] is None and at.session_state["report"] is None
    assert at.session_state["ans"] == I.blank_answers()
    assert not [k for k in list(at.session_state) if k.startswith("w_")]


def test_run_screening_without_answers_never_calls_model():
    at = _at()
    _click(at, "Start Screening")
    _click(at, "Continue to Lifestyle")
    assert at.session_state["page"] == "profile" and at.error      # blocked with a visible message


def test_result_and_recs_pages_have_safe_empty_states():
    at = _at()
    at.session_state["page"] = "result"
    at.run()
    assert not at.exception and any("No screening result yet" in m.value for m in at.markdown)
    at.session_state["page"] = "recs"
    at.run()
    assert not at.exception and any("No screening result yet" in m.value for m in at.markdown)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ---------------- user-facing wording audit ----------------
BANNED = ("nfhs", "dataset", "respondent", "derived data", "research prototype", "survey sample", "survey")


def _all_visible_text(at) -> str:
    parts = []
    for coll in (at.markdown, at.caption, at.error, at.warning, at.info, at.success):
        parts += [getattr(e, "value", "") for e in coll]
    for coll in (at.button, at.radio, at.selectbox, at.number_input, at.checkbox, at.toggle):
        for e in coll:
            parts.append(getattr(e, "label", "") or "")
            parts.append(getattr(e, "help", "") or "")
    return " ".join(parts).lower()


@pytest.mark.parametrize("sex", ["women", "men"])
def test_no_dataset_wording_on_any_user_facing_page(sex):
    ans = _answers(sex, uses_tobacco=True, **({"tob_v463a": True} if sex == "women" else {"tob_mv463a": True}),
                   tob_freq=1, drinks=True, alc_freq=2, alc_duration=3.0)
    at = _at()
    seen = {"landing": _all_visible_text(at)}
    _click(at, "Start Screening"); seen["profile"] = _all_visible_text(at)
    _fill_widgets(at, {k: ans[k] for k in ("sex", "age", "education", "residence")})
    _click(at, "Continue to Lifestyle")
    _fill_widgets(at, {k: v for k, v in ans.items() if k in ("uses_tobacco", "drinks") or k.startswith("diet_")})
    tob = {k: v for k, v in ans.items() if k.startswith("tob_")}
    _fill_widgets(at, tob)
    _fill_widgets(at, tob)            # second pass: the frequency question only appears once cigarettes are ticked
    _fill_widgets(at, {k: ans[k] for k in ("alc_freq", "alc_duration")})
    seen["lifestyle"] = _all_visible_text(at)
    _click(at, "Continue to Health"); seen["health"] = _all_visible_text(at)
    keys = ("bp_sys", "bp_dia", "bp_n") + (("height_cm", "weight_kg") if sex == "women" else ())
    _fill_widgets(at, {k: ans[k] for k in keys})
    _click(at, "Continue to Review"); seen["review"] = _all_visible_text(at)
    _click(at, "Run EarlyDia Screening"); seen["result"] = _all_visible_text(at)
    assert at.session_state["page"] == "result", seen["review"][:400]
    _click(at, "View Personalized Recommendations"); seen["recs"] = _all_visible_text(at)
    assert at.session_state["page"] == "recs"
    for page, text in seen.items():
        hits = [w for w in BANNED if w in text]
        assert not hits, f"{page}: user-facing text contains {hits}"


def test_landing_is_short_and_has_no_medical_disclaimer_block():
    text = _all_visible_text(_at())
    for must in ("early diabetes risk screening", "about 3 minutes", "how it works", "tell us about yourself",
                 "complete your screening", "get your result"):
        assert must in text
    for gone in ("what the result means", "what earlydia does not do", "research"):
        assert gone not in text


def test_stepper_text_is_never_glued_together():
    from src.app.ui_components import stepper_html
    html = stepper_html("result")
    assert "Step 5: </span>Result" in html and "Step 6: </span>Guidance" in html
    assert "5Result" not in html and "6Guidance" not in html


def test_result_page_text_and_food_table():
    at = _at()
    _click(at, "Start Screening")
    ans = _answers("men")
    _fill_widgets(at, {k: ans[k] for k in ("sex", "age", "education", "residence")})
    _click(at, "Continue to Lifestyle")
    _fill_widgets(at, {k: v for k, v in ans.items() if k in ("uses_tobacco", "drinks") or k.startswith("diet_")})
    _click(at, "Continue to Health")
    _fill_widgets(at, {k: ans[k] for k in ("bp_sys", "bp_dia", "bp_n")})
    _click(at, "Continue to Review"); _click(at, "Run EarlyDia Screening")
    page = " ".join(m.value for m in at.markdown)
    for must in ("Estimated diabetes-related screening probability", "What does this percentage mean?",
                 "Why is there a threshold?", "Factors considered", "Not used for this screening group",
                 "average of 3 reported readings", "Technical details", "Milk or curd", "Aerated drinks",
                 "does not mean that you have diabetes" if at.session_state["result"].predicted_class else
                 "does not rule out diabetes"):
        assert must in page, must
    assert "Your food answers" not in page and "HistGradientBoosting" in page      # model name only in tech details
    assert page.index("HistGradientBoosting") > page.index("Technical details")
