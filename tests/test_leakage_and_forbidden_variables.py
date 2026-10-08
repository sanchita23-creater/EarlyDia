"""
EarlyDia - Leakage and forbidden-variable checks on the LOCKED production artifacts.
Re-verifies (does not re-derive) the guarantees established during predictor construction:
no target-construction variable, no waist/hip, no men's BMI, no wealth, no hypertension
self-report, no pregnancy variable ever appears in either sex's feature list.
"""
import pickle
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import PRODUCTION_PIPELINE_PATHS

TARGET_CONSTRUCTION_VARS = {"sb56", "sb57", "s728a", "sb74", "smb56", "smb57", "sm627a", "smb74"}
FORBIDDEN_SUBSTRINGS = {
    "waist": "waist measurement",
    "hip": "hip measurement",
    "v445": "women's BMI variable name leaking into men's features",
    "v190": "wealth quintile",
    "s728b": "hypertension self-report",
    "sm627b": "hypertension self-report (men)",
    "v213": "pregnancy status",
    "v214": "pregnancy duration",
    "v215": "menstrual period timing (pregnancy-adjacent)",
}


def _load_feature_cols(sex):
    with open(PRODUCTION_PIPELINE_PATHS[sex], "rb") as f:
        artifact = pickle.load(f)
    return artifact["feature_cols"]


def test_no_target_construction_variables_in_features():
    for sex in ["women", "men"]:
        cols = set(_load_feature_cols(sex))
        leaked = cols & TARGET_CONSTRUCTION_VARS
        assert not leaked, f"{sex}: target-construction variables leaked into X: {leaked}"


def test_no_forbidden_variables_in_features():
    for sex in ["women", "men"]:
        cols = _load_feature_cols(sex)
        for col in cols:
            for bad_substr, description in FORBIDDEN_SUBSTRINGS.items():
                assert bad_substr not in col.lower(), (
                    f"{sex}: forbidden variable pattern '{bad_substr}' ({description}) "
                    f"found in feature column '{col}'")


def test_men_have_no_bmi_feature():
    cols = _load_feature_cols("men")
    assert "bmi" not in cols, "men's feature list must not contain a BMI predictor"


def test_women_have_bmi_feature():
    cols = _load_feature_cols("women")
    assert "bmi" in cols, "women's feature list should contain a BMI predictor"


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
