"""IDRS comparator: scoring table (Mohan et al. 2005), validation, API integration, and the
guarantee that IDRS inputs never change the ML model output."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.api import idrs  # noqa: E402


# ---- scoring table -------------------------------------------------------------------------
@pytest.mark.parametrize("age,pts", [(18, 0), (34, 0), (35, 20), (40, 20), (49, 20), (50, 30)])
def test_age_points(age, pts):
    assert idrs.age_points(age) == pts


@pytest.mark.parametrize("sex,waist,pts", [
    ("women", 79.9, 0), ("women", 80, 10), ("women", 89.9, 10), ("women", 90, 20),
    ("men", 89.9, 0), ("men", 90, 10), ("men", 99.9, 10), ("men", 100, 20),
])
def test_waist_points_are_sex_specific(sex, waist, pts):
    assert idrs.waist_points(sex, waist) == pts


@pytest.mark.parametrize("score,cat", [(0, "low"), (20, "low"), (30, "moderate"), (50, "moderate"),
                                       (60, "high"), (100, "high")])
def test_risk_bands(score, cat):
    assert idrs.risk_category(score) == cat


def test_minimum_and_maximum_total():
    lo = idrs.compute_idrs("women", 25, 70, "regular_both", "none")
    assert lo["score"] == 0 and lo["category"] == "low"
    hi = idrs.compute_idrs("men", 55, 110, "sedentary", "both_parents")
    assert hi["score"] == idrs.MAX_SCORE == 100 and hi["category"] == "high"


def test_breakdown_sums_to_total_and_cites_source():
    r = idrs.compute_idrs("men", 38, 95, "regular_or_strenuous", "one_parent")
    assert [c["points"] for c in r["components"]] == [20, 10, 20, 10]
    assert r["score"] == sum(c["points"] for c in r["components"]) == 60
    assert "Mohan" in r["citation"] and "not used by the EarlyDia" in r["note"]


# ---- validation ----------------------------------------------------------------------------
def test_validation_all_or_none():
    assert idrs.validate("women", None, None, None) == []
    assert idrs.validate("women", 85, None, None)               # partial -> error
    assert idrs.validate("women", 85, "sedentary", "none") == []
    assert idrs.validate("women", 5, "sedentary", "none")       # waist out of range
    assert idrs.validate("women", 85, "bogus", "none")
    assert idrs.validate("women", 85, "sedentary", "bogus")


# ---- API integration -----------------------------------------------------------------------
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from backend.api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _body(sex, **over):
    b = dict(sex=sex, age=36, education=2, residence=1, tobacco_use=False, drinks_alcohol=False,
             diet={k: 2 for k in "abcdefghi"}, bp_systolic=124, bp_diastolic=80,
             height_cm=160 if sex == "women" else None, weight_kg=60 if sex == "women" else None)
    b.update(over)
    return b


def test_config_exposes_idrs_options(client):
    cfg = client.get("/api/config").json()["idrs"]
    assert [o["value"] for o in cfg["activity"]] == ["regular_both", "regular_or_strenuous", "sedentary"]
    assert len(cfg["family_history"]) == 3 and cfg["waist_range_cm"][0] < cfg["waist_range_cm"][1]


@pytest.mark.parametrize("sex", ["women", "men"])
def test_without_idrs_answers_response_is_unchanged(client, sex):
    r = client.post("/api/predict", json=_body(sex))
    assert r.status_code == 200 and r.json()["idrs"] is None


@pytest.mark.parametrize("sex", ["women", "men"])
def test_idrs_does_not_change_ml_output(client, sex):
    base = client.post("/api/predict", json=_body(sex)).json()
    with_idrs = client.post("/api/predict", json=_body(
        sex, waist_cm=104, physical_activity="sedentary", family_history="both_parents")).json()
    assert with_idrs["probability"] == base["probability"]
    assert with_idrs["flagged"] == base["flagged"]
    assert with_idrs["cutoff"] == base["cutoff"]
    assert with_idrs["idrs"]["score"] == 20 + 20 + 30 + 20 and with_idrs["idrs"]["category"] == "high"
    assert with_idrs["technical"]["idrs_used_by_model"] is False


def test_partial_idrs_answers_rejected(client):
    r = client.post("/api/predict", json=_body("women", waist_cm=85))
    assert r.status_code == 422 and any("IDRS" in e for e in r.json()["errors"])


def test_invalid_idrs_option_rejected(client):
    r = client.post("/api/predict", json=_body("men", waist_cm=95, physical_activity="x", family_history="none"))
    assert r.status_code == 422
