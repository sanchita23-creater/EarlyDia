"""End-to-end tests of the FastAPI layer (React -> FastAPI -> existing inference -> production models).
Uses the REAL production predictor; nothing is mocked."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from backend.api.main import app  # noqa: E402
from src.app import input_spec as I  # noqa: E402
from src.inference.predict_module import EarlyDiaPredictor  # noqa: E402
from src.inference.predict_module_v2 import EarlyDiaPredictorV2  # noqa: E402

THRESHOLDS_V1 = {"women": 0.026, "men": 0.027}
THRESHOLDS_V2 = {"women": 0.024, "men": 0.029}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def body(_sex, **over):
    sex = _sex
    b = dict(sex=sex, age=30, education=2, residence=1, tobacco_use=False, drinks_alcohol=False,
             diet={k: 2 for k in "abcdefghi"}, bp_systolic=120, bp_diastolic=78,
             height_cm=160 if sex == "women" else None, weight_kg=58.1 if sex == "women" else None)
    b.update(over)
    return b


def test_health_and_config(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["models"] == {"women": "isotonic", "men": "sigmoid"}
    cfg = client.get("/api/config").json()
    assert cfg["per_sex"]["women"]["uses_bmi"] and not cfg["per_sex"]["men"]["uses_bmi"]
    assert len(cfg["diet_questions"]) == 9


@pytest.mark.parametrize("sex", ["women", "men"])
def test_prediction_matches_direct_inference_and_locked_thresholds(client, sex):
    # Test v1 backwards compatibility
    r_v1 = client.post("/api/predict", json=body(sex, model_version="v1"))
    assert r_v1.status_code == 200, r_v1.text
    out_v1 = r_v1.json()
    ans = I.blank_answers()
    ans.update(sex=sex, age=30, education=2, residence=1, uses_tobacco=False, drinks=False,
               bp_sys=120.0, bp_dia=78.0, bp_n=3)
    if sex == "women":
        ans.update(height_cm=160.0, weight_kg=58.1)
    for col, _ in I.DIET_ITEMS[sex]:
        ans[col] = 2
    direct_v1 = EarlyDiaPredictor(sex).predict(I.build_person(sex, I.ui_from_answers(ans)))
    assert out_v1["probability"] == pytest.approx(direct_v1.calibrated_probability)
    assert out_v1["cutoff"] == THRESHOLDS_V1[sex] == direct_v1.threshold_used
    assert out_v1["flagged"] == bool(direct_v1.predicted_class)
    assert out_v1["status"] in ("Flagged for follow-up", "Lower screening likelihood")

    # Test v2 production screening (default)
    r_v2 = client.post("/api/predict", json=body(sex))
    assert r_v2.status_code == 200, r_v2.text
    out_v2 = r_v2.json()
    direct_v2 = EarlyDiaPredictorV2(sex).predict(I.build_person(sex, I.ui_from_answers(ans)))
    assert out_v2["probability"] == pytest.approx(direct_v2.calibrated_probability)
    assert out_v2["cutoff"] == THRESHOLDS_V2[sex] == direct_v2.threshold_used
    assert out_v2["flagged"] == bool(direct_v2.predicted_class)
    assert out_v2["technical"]["version"] == "v2"


def test_men_profile_has_no_bmi_and_women_has_bmi(client):
    men = client.post("/api/predict", json=body("men", height_cm=170, weight_kg=70)).json()
    assert "BMI" not in [p["label"] for p in men["profile"]]
    assert men["technical"]["bmi_used_by_model"] is False
    women = client.post("/api/predict", json=body("women")).json()
    assert dict((p["label"], p["value"]) for p in women["profile"])["BMI"] == "22.7 kg/m²"


@pytest.mark.parametrize("sex", ["women", "men"])
def test_missing_blood_pressure_is_rejected_not_scored(client, sex):
    for drop in ("bp_systolic", "bp_diastolic"):
        b = body(sex); b.pop(drop)
        r = client.post("/api/predict", json=b)
        assert r.status_code == 422
        assert "probability" not in r.json()
        assert "Traceback" not in r.text


@pytest.mark.parametrize("over", [
    dict(bp_systolic=300), dict(bp_diastolic=20), dict(bp_systolic=80, bp_diastolic=90),
    dict(age=17), dict(age=41), dict(sex="other"), dict(education=9), dict(residence=3),
    dict(diet={"a": 1}), dict(tobacco_use=True), dict(drinks_alcohol=True),
])
def test_invalid_input_gives_friendly_422(client, over):
    r = client.post("/api/predict", json=body("women", **over))
    assert r.status_code == 422
    j = r.json()
    assert j["message"] and "Traceback" not in r.text and "probability" not in j


def test_women_requires_height_and_weight(client):
    r = client.post("/api/predict", json=body("women", height_cm=None))
    assert r.status_code == 422


def test_tobacco_and_alcohol_mapping_reaches_model(client):
    cfg = client.get("/api/config").json()["per_sex"]["women"]
    b = body("women", tobacco_use=True, tobacco_products=[cfg["cigarette_id"]], cigarette_frequency=1,
             drinks_alcohol=True, alcohol_frequency=2, alcohol_years=5, alcohol_types=[cfg["alcohol_types"][2]["id"]])
    r = client.post("/api/predict", json=b)
    assert r.status_code == 200
    prof = dict((p["label"], p["value"]) for p in r.json()["profile"])
    assert prof["Tobacco"].startswith("Yes") and "cigarettes" in prof["Tobacco"].lower()
    assert "beer" in prof["Alcohol"].lower()


def test_response_never_contains_markup_or_raw_codes(client):
    text = client.post("/api/predict", json=body("women")).text
    for bad in ("<div", "<svg", "<path", "Education: 2", "Residence: 1"):
        assert bad not in text


def test_state_served_by_config(client):
    cfg = client.get("/api/config").json()
    assert "states" in cfg
    assert len(cfg["states"]) == 36  # NFHS-5 36 states/UTs (code 26 omitted)
    codes = [s["value"] for s in cfg["states"]]
    assert 9 in codes and 27 in codes and 33 in codes
    assert 26 not in codes
    assert cfg["default_state"] == {"value": 9, "label": "Uttar Pradesh"}


def test_state_required_in_ui_validation():
    answers_js = (ROOT / "frontend" / "src" / "utils" / "answers.js").read_text(encoding="utf-8")
    # Verify state is defined in emptyAnswers, checked in validateProfile, included in buildPayload & reviewSections
    assert "Select your State / UT." in answers_js
    assert "state: num(a.state)" in answers_js
    assert '"State / UT"' in answers_js


def test_invalid_state_code_rejected(client):
    for bad_code in [0, 26, 99, -1]:
        r = client.post("/api/predict", json=body("women", state=bad_code))
        assert r.status_code == 422
        assert any("State / UT" in err for err in r.json().get("errors", []))


def test_state_omitted_returns_warning_and_national_default(client):
    r = client.post("/api/predict", json=body("women", state=None))
    assert r.status_code == 200
    data = r.json()
    assert data["state_provided"] is False
    assert data["state_used"] == "Uttar Pradesh"
    assert data["warning"] is not None
    assert "Uttar Pradesh" in data["warning"]
    assert "code 9" in data["warning"]


def test_v1_unaffected_by_state(client):
    # Omitted state in v1
    r_v1_omitted = client.post("/api/predict", json=body("women", model_version="v1", state=None))
    assert r_v1_omitted.status_code == 200
    data_omitted = r_v1_omitted.json()
    assert data_omitted["cutoff"] == 0.026
    assert data_omitted["warning"] is None
    assert data_omitted["technical"]["version"] == "v1"

    # Provided state in v1 does not change model output
    r_v1_state = client.post("/api/predict", json=body("women", model_version="v1", state=27))
    assert r_v1_state.status_code == 200
    data_state = r_v1_state.json()
    assert data_state["cutoff"] == 0.026
    assert data_state["probability"] == data_omitted["probability"]


@pytest.mark.parametrize("sex", ["women", "men"])
def test_same_profile_with_two_states_gives_different_probabilities(client, sex):
    # Maharashtra (27) vs Tamil Nadu (33)
    r_mah = client.post("/api/predict", json=body(sex, state=27))
    r_tn = client.post("/api/predict", json=body(sex, state=33))
    assert r_mah.status_code == 200 and r_tn.status_code == 200
    d_mah = r_mah.json()
    d_tn = r_tn.json()
    assert d_mah["state_provided"] is True
    assert d_mah["state_used"] == "Maharashtra"
    assert d_mah["warning"] is None
    assert d_tn["state_provided"] is True
    assert d_tn["state_used"] == "Tamil Nadu"
    assert d_tn["warning"] is None
    assert d_mah["probability"] != d_tn["probability"], f"Probabilities should differ between states for {sex}"


def test_results_page_displays_model_version_and_screening_threshold(client):
    r = client.post("/api/predict", json=body("women"))
    assert r.status_code == 200
    data = r.json()
    assert "cutoff" in data and data["cutoff"] == 0.024
    assert data["technical"]["version"] == "v2"

    results_jsx = (ROOT / "frontend" / "src" / "pages" / "Results.jsx").read_text(encoding="utf-8")
    assert 'Model {t.version || "v2"} - screening threshold {r.cutoff}' in results_jsx


if __name__ == "__main__":      # lets `python run_tests.py` execute this file too
    sys.exit(pytest.main([__file__, "-q"]))

