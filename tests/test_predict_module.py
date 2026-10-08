"""
EarlyDia - Tests for the production prediction module.
Run: python3 -m pytest tests/test_predict_module.py -v
     or: python3 tests/test_predict_module.py
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.inference.predict_module import EarlyDiaPredictor

VALID_WOMAN = {
    "age": 32, "avg_systolic_bp": 118, "avg_diastolic_bp": 76,
    "n_valid_systolic_readings": 3, "n_valid_diastolic_readings": 3,
    "bmi": 24.5, "alc_duration_years": 0, "alc_frequency": 0,
    "tob_v463a": 0, "tob_v463b": 0, "tob_v463c": 0, "tob_v463d": 0, "tob_v463f": 0,
    "tob_v463g": 0, "tob_v463j": 0, "tob_v463k": 0, "tob_v463l": 0, "tob_v463x": 0,
    "tob_v463z": 1, "tob_v463aa": 0,
    "alc_s722a": 0, "alc_s722b": 0, "alc_s722c": 0, "alc_s722d": 0, "alc_s722e": 0, "alc_s722x": 0,
    "diet_s731a": 1, "diet_s731b": 2, "diet_s731c": 3, "diet_s731d": 0, "diet_s731e": 1,
    "diet_s731f": 2, "diet_s731g": 3, "diet_s731h": 0, "diet_s731i": 1,
    "alc_drinks": 0, "education": 2, "residence": 1,
}

VALID_MAN = {
    "age": 35, "avg_systolic_bp": 124, "avg_diastolic_bp": 80,
    "n_valid_systolic_readings": 3, "n_valid_diastolic_readings": 3,
    "alc_duration_years": 5, "alc_frequency": 2,
    "tob_mv463a": 0, "tob_mv463b": 0, "tob_mv463c": 0, "tob_mv463d": 0,
    "tob_mv463f": 0, "tob_mv463g": 0, "tob_mv463x": 0, "tob_mv463z": 1,
    "alc_sm621a": 1, "alc_sm621b": 0, "alc_sm621c": 0, "alc_sm621d": 0, "alc_sm621e": 0, "alc_sm621x": 0,
    "diet_sm630a": 1, "diet_sm630b": 2, "diet_sm630c": 3, "diet_sm630d": 0, "diet_sm630e": 1,
    "diet_sm630f": 2, "diet_sm630g": 3, "diet_sm630h": 0, "diet_sm630i": 1,
    "alc_drinks": 1, "education": 3, "residence": 2,
}


def test_valid_women_input():
    predictor = EarlyDiaPredictor("women")
    result = predictor.predict(VALID_WOMAN)
    assert 0.0 <= result.calibrated_probability <= 1.0
    assert result.predicted_class in (0, 1)
    assert abs(result.threshold_used - 0.026) < 1e-9
    assert result.sex == "women"
    assert result.calibration_method == "isotonic"


def test_valid_men_input():
    predictor = EarlyDiaPredictor("men")
    result = predictor.predict(VALID_MAN)
    assert 0.0 <= result.calibrated_probability <= 1.0
    assert result.predicted_class in (0, 1)
    assert abs(result.threshold_used - 0.027) < 1e-9
    assert result.sex == "men"
    assert result.calibration_method == "sigmoid"


def test_missing_value_handling():
    predictor = EarlyDiaPredictor("women")
    incomplete = dict(VALID_WOMAN)
    del incomplete["bmi"]
    del incomplete["avg_diastolic_bp"]
    result = predictor.predict(incomplete)
    assert 0.0 <= result.calibrated_probability <= 1.0
    assert len(result.validation_warnings) == 2


def test_invalid_age_rejected():
    predictor = EarlyDiaPredictor("women")
    bad = dict(VALID_WOMAN); bad["age"] = 5
    try:
        predictor.predict(bad)
        assert False, "should have raised ValueError"
    except ValueError:
        pass


def test_invalid_bp_rejected():
    predictor = EarlyDiaPredictor("women")
    bad = dict(VALID_WOMAN); bad["avg_systolic_bp"] = 500
    try:
        predictor.predict(bad)
        assert False, "should have raised ValueError"
    except ValueError:
        pass


def test_invalid_category_rejected():
    predictor = EarlyDiaPredictor("women")
    bad = dict(VALID_WOMAN); bad["education"] = 9
    try:
        predictor.predict(bad)
        assert False, "should have raised ValueError"
    except ValueError:
        pass


def test_invalid_type_rejected():
    predictor = EarlyDiaPredictor("women")
    bad = dict(VALID_WOMAN); bad["age"] = "thirty-two"
    try:
        predictor.predict(bad)
        assert False, "should have raised ValueError"
    except ValueError:
        pass


def test_threshold_behavior():
    predictor = EarlyDiaPredictor("women")
    result = predictor.predict(VALID_WOMAN)
    expected_class = int(result.calibrated_probability >= result.threshold_used)
    assert result.predicted_class == expected_class
    assert int(0.026 >= 0.026) == 1
    assert int(0.0259 >= 0.026) == 0


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed, failed = 0, 0
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
