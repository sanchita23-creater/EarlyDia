"""
EarlyDia - Tests for recommendations and gauge components.
Verifies that recommendation logic dynamically reflects user inputs and prediction
results without touching any ML pipeline, and that the semicircular gauge renders properly.
"""
import sys
from pathlib import Path
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app.recommendations import generate_recommendations, RecommendationReport
from src.app.gauge import render_semicircular_gauge


@dataclass
class MockPredictionResult:
    calibrated_probability: float
    threshold_used: float
    predicted_class: int
    model: str = "HistGradientBoosting"
    calibration_method: str = "isotonic"


def test_flagged_clinical_next_steps():
    mock_res = MockPredictionResult(
        calibrated_probability=0.035,
        threshold_used=0.026,
        predicted_class=1
    )
    ui_inputs = {
        "age": 32, "bmi": 28.0, "avg_sys": 142.0, "avg_dia": 92.0,
        "n_sys": 3, "n_dia": 3, "drinks": True, "alc_duration": 4.0,
        "tobacco": {"tob_v463a": True}, "diet": {"diet_s731h": 1}
    }
    report = generate_recommendations("women", ui_inputs, mock_res)
    assert report.is_flagged is True
    assert "At or Above" in report.summary_headline
    assert report.clinical_next_steps.tag_type == "alert"
    assert "NOT a medical diagnosis" in report.clinical_next_steps.summary
    assert any("fasting plasma glucose" in a.lower() for a in report.clinical_next_steps.action_items)


def test_unflagged_clinical_next_steps():
    mock_res = MockPredictionResult(
        calibrated_probability=0.015,
        threshold_used=0.026,
        predicted_class=0
    )
    ui_inputs = {
        "age": 25, "bmi": 21.0, "avg_sys": 118.0, "avg_dia": 76.0,
        "n_sys": 3, "n_dia": 3, "drinks": False,
        "tobacco": {}, "diet": {}
    }
    report = generate_recommendations("women", ui_inputs, mock_res)
    assert report.is_flagged is False
    assert "Below" in report.summary_headline
    assert report.clinical_next_steps.tag_type == "ok"
    assert "does not guarantee the absence of diabetes" in report.clinical_next_steps.summary


def test_recommendations_change_with_lifestyle_inputs():
    mock_res = MockPredictionResult(calibrated_probability=0.020, threshold_used=0.027, predicted_class=0)

    # Case A: Smoker, drinker, high BP
    ui_a = {
        "age": 35, "avg_sys": 145.0, "avg_dia": 95.0, "n_sys": 3, "n_dia": 3,
        "drinks": True, "alc_duration": 8.0, "tobacco": {"tob_mv463a": True}, "diet": {}
    }
    rep_a = generate_recommendations("men", ui_a, mock_res)
    categories_a = {item.category: item for item in rep_a.lifestyle_recommendations}

    assert "Tobacco Use" in categories_a
    assert categories_a["Tobacco Use"].status_tag == "Cessation Advised"
    assert "Alcohol Intake" in categories_a
    assert categories_a["Alcohol Intake"].status_tag == "Moderation Advised"
    assert "Blood Pressure" in categories_a
    assert categories_a["Blood Pressure"].status_tag == "Worth a check"

    # Case B: Non-smoker, non-drinker, healthy BP
    ui_b = {
        "age": 35, "avg_sys": 118.0, "avg_dia": 75.0, "n_sys": 3, "n_dia": 3,
        "drinks": False, "tobacco": {}, "diet": {}
    }
    rep_b = generate_recommendations("men", ui_b, mock_res)
    categories_b = {item.category: item for item in rep_b.lifestyle_recommendations}

    assert categories_b["Tobacco Use"].status_tag == "Protective Habit"
    assert categories_b["Alcohol Intake"].status_tag == "Protective Habit"
    assert categories_b["Blood Pressure"].status_tag == "Optimal Range"


def test_factors_considered_sex_difference():
    mock_res = MockPredictionResult(calibrated_probability=0.025, threshold_used=0.027, predicted_class=0)
    rep_w = generate_recommendations("women", {"age": 28, "bmi": 24.0}, mock_res)
    rep_m = generate_recommendations("men", {"age": 28}, mock_res)

    factors_w = {f["factor"]: f["value"] for f in rep_w.factors_considered}
    factors_m = {f["factor"]: f["value"] for f in rep_m.factors_considered}

    assert "24.0 kg/m²" in factors_w["Body Mass Index (BMI)"]
    assert "Not used for this screening group" in factors_m["Body Mass Index (BMI)"]


def test_semicircular_gauge_svg():
    for dark_mode in [False, True]:
        for is_flagged in [False, True]:
            prob = 3.2 if is_flagged else 1.8
            svg = render_semicircular_gauge(prob, 2.6, is_flagged, dark_mode)
            assert "<svg" in svg
            assert "viewBox" in svg
            assert "Threshold</text>" in svg and "2.60%" in svg
            assert f"{prob:.2f}%" in svg


def test_react_probability_scale_gauge():
    jsx_path = ROOT / "frontend" / "src" / "components" / "ProbabilityScale.jsx"
    assert jsx_path.exists(), "ProbabilityScale.jsx must exist"
    content = jsx_path.read_text(encoding="utf-8")

    # Real React component rendering inline SVG
    assert "<svg" in content
    assert "gauge-svg" in content
    assert "SCALE_MAX = 15" in content

    # Clamping needle at 15% while displaying real probability number
    assert "clampedPercent" in content
    assert "needleFrac" in content

    # Cutoff tick & dynamic label (not hardcoded)
    assert "Screening cutoff" in content
    assert "gauge-cutoff-tick" in content
    assert "cutoffLabel" in content

    # Calm zones in navy/white/black palette, no alarming red / danger wording
    assert "gauge-zone-below" in content
    assert "gauge-zone-above" in content
    lower_c = content.lower()
    assert "danger" not in lower_c
    assert "alarm" not in lower_c

    # Accessible aria-label with disclaimer and status
    assert "aria-label=" in content
    assert "This does not mean you have diabetes" in content


def test_results_why_these_inputs_table():
    results_path = ROOT / "frontend" / "src" / "pages" / "Results.jsx"
    assert results_path.exists(), "Results.jsx must exist"
    content = results_path.read_text(encoding="utf-8")

    # Why these inputs section inside Technical Details
    assert "Why these inputs" in content
    assert "These are published associations, not proof that any factor caused your result" in content
    assert "Model importance values are in the repo reports" in content

    # Must contain citations from docs/RESEARCH_EVIDENCE.md
    for tag in ["[ADA-2]", "[INDIAB-1]", "[EMDIN-2015]", "[WILLI-2007]"]:
        assert tag in content, f"Expected citation tag {tag} in Results.jsx"



if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
