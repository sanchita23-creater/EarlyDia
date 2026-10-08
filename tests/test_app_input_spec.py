"""EarlyDia - tests for the app input specification (labels -> feature columns -> predictor).
Because Streamlit cannot be launched in the dev container, these tests pin the mapping that
the app relies on: every model feature is populated, labels match the NFHS-5 codebook, and
UI-consistency rules mirror training-time preprocessing."""
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import PRODUCTION_PIPELINE_PATHS
from src.app.input_spec import (TOBACCO_ITEMS, ALCOHOL_TYPE_ITEMS, DIET_ITEMS, build_person)
from src.inference.predict_module import EarlyDiaPredictor


def _feature_cols(sex):
    with open(PRODUCTION_PIPELINE_PATHS[sex], "rb") as f:
        return pickle.load(f)["feature_cols"]


def _default_ui(sex):
    ui = {"age": 30, "avg_sys": 120.0, "avg_dia": 78.0, "n_sys": 3, "n_dia": 3,
          "tobacco": {}, "tob_freq": None, "drinks": False, "alc_freq": None, "alc_duration": 0.0,
          "alc_types": {}, "diet": {c: 2 for c, _ in DIET_ITEMS[sex]},
          "education": 2, "residence": 1}
    if sex == "women":
        ui["bmi"] = 23.0
    return ui


def test_person_keys_exactly_match_model_features():
    for sex in ["women", "men"]:
        person = build_person(sex, _default_ui(sex))
        assert set(person) == set(_feature_cols(sex)), (
            f"{sex}: app-built keys differ from model features: "
            f"{set(person) ^ set(_feature_cols(sex))}")


def test_default_person_passes_predictor_for_both_sexes():
    for sex in ["women", "men"]:
        result = EarlyDiaPredictor(sex).predict(build_person(sex, _default_ui(sex)))
        assert 0.0 <= result.calibrated_probability <= 1.0
        assert result.validation_warnings == []


def test_alcohol_type_order_matches_codebook():
    expected = ["tadi", "country", "beer", "wine", "hard", "other"]
    for sex in ["women", "men"]:
        labels = [lab.lower() for _, lab in ALCOHOL_TYPE_ITEMS[sex]]
        for exp, lab in zip(expected, labels):
            assert exp in lab, f"{sex}: alcohol type order wrong: {labels}"
        suffixes = [c[-1] for c, _ in ALCOHOL_TYPE_ITEMS[sex]]
        assert suffixes == list("abcdex")


def test_tobacco_labels_match_codebook_by_sex():
    w = dict(TOBACCO_ITEMS["women"]); m = dict(TOBACCO_ITEMS["men"])
    assert "cigar" in w["tob_v463f"].lower() and "hookah" in w["tob_v463g"].lower()
    assert "gutkha" in w["tob_v463j"].lower() and "khaini" in w["tob_v463k"].lower()
    assert "paan" in w["tob_v463l"].lower()
    assert "hookah" in m["tob_mv463f"].lower() and "gutkha" in m["tob_mv463g"].lower()
    assert len(w) == 10 and len(m) == 7


def test_diet_labels_and_order():
    expected = ["milk", "pulses", "leafy", "fruit", "egg", "fish", "chicken", "fried", "aerated"]
    for sex in ["women", "men"]:
        labels = [lab.lower() for _, lab in DIET_ITEMS[sex]]
        for exp, lab in zip(expected, labels):
            assert exp in lab
        assert len(labels) == 9


def test_selecting_beer_sets_only_beer_column():
    ui = _default_ui("women"); ui["drinks"] = True; ui["alc_freq"] = 2; ui["alc_duration"] = 3
    ui["alc_types"] = {"alc_s722c": True}
    p = build_person("women", ui)
    assert p["alc_s722c"] == 1
    assert sum(p[c] for c, _ in ALCOHOL_TYPE_ITEMS["women"]) == 1
    assert p["alc_drinks"] == 1 and p["alc_frequency"] == 2 and p["alc_duration_years"] == 3.0


def test_hookah_maps_to_correct_column_per_sex():
    uw = _default_ui("women"); uw["tobacco"] = {"tob_v463g": True}
    um = _default_ui("men"); um["tobacco"] = {"tob_mv463f": True}
    assert build_person("women", uw)["tob_v463g"] == 1
    assert build_person("men", um)["tob_mv463f"] == 1


def test_non_drinker_alcohol_fields_zeroed_even_if_ticked():
    ui = _default_ui("men"); ui["drinks"] = False; ui["alc_freq"] = 1
    ui["alc_duration"] = 12; ui["alc_types"] = {"alc_sm621c": True}
    p = build_person("men", ui)
    assert p["alc_drinks"] == 0 and p["alc_frequency"] == 0 and p["alc_duration_years"] == 0.0
    assert all(p[c] == 0 for c, _ in ALCOHOL_TYPE_ITEMS["men"])


def test_none_tobacco_indicator_is_derived_and_consistent():
    p = build_person("women", _default_ui("women"))
    assert p["tob_v463z"] == 1
    ui = _default_ui("women"); ui["tobacco"] = {"tob_v463c": True}
    assert build_person("women", ui)["tob_v463z"] == 0


def test_cigarette_frequency_only_when_smoking_cigarettes_women_only():
    ui = _default_ui("women"); ui["tob_freq"] = 1
    assert build_person("women", ui)["tob_v463aa"] == 0
    ui["tobacco"] = {"tob_v463a": True}
    assert build_person("women", ui)["tob_v463aa"] == 1
    assert "tob_mv463aa" not in build_person("men", _default_ui("men"))


def test_zero_valid_bp_readings_makes_average_missing():
    ui = _default_ui("men"); ui["n_sys"] = 0
    p = build_person("men", ui)
    assert p["avg_systolic_bp"] is None and p["avg_diastolic_bp"] == 78.0
    result = EarlyDiaPredictor("men").predict(p)
    assert len(result.validation_warnings) == 1


def test_men_person_has_no_bmi_and_women_do():
    assert "bmi" not in build_person("men", _default_ui("men"))
    assert build_person("women", _default_ui("women"))["bmi"] == 23.0


def test_app_delegates_to_predict_module_and_has_no_model_logic():
    app_dir = ROOT / "src" / "app"
    state_src = (app_dir / "ui_state.py").read_text(encoding="utf-8")
    assert "EarlyDiaPredictor" in state_src and "build_person" in state_src
    src = "\n".join((app_dir / f).read_text(encoding="utf-8") for f in
                    ("app.py", "views.py", "ui_state.py", "ui_components.py", "gauge.py", "summaries.py"))
    for forbidden in ["predict_proba", ".fit(", "pickle", "HistGradientBoosting(", "LogisticRegression("]:
        assert forbidden not in src, f"app.py contains model/preprocessing logic: {forbidden}"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t(); print(f"[PASS] {t.__name__}"); passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {type(e).__name__}: {e}"); failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
