"""EarlyDia - tests for src/data/extract_nfhs.py.
Raw NFHS data is not present in this repo, so most of what can be verified here is
structural: column lists are well-formed and match what the downstream pipeline expects,
and the script fails clearly (not silently) when raw files are absent. One exception: the
men's file WAS available while building this project and the full extraction was run and
compared byte-for-byte against the file the real pipeline used (see docs/DATA_REPRODUCTION.md
and the git history of this test for that one-time verification); that is not repeatable
here without the raw file, so it is not re-asserted as a live test."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.extract_nfhs import WOMEN_COLUMNS, MEN_COLUMNS, SPECS, find_raw_file
from configs.config import PRODUCTION_PIPELINE_PATHS
import pickle


def test_column_lists_have_no_duplicates():
    assert len(WOMEN_COLUMNS) == len(set(WOMEN_COLUMNS))
    assert len(MEN_COLUMNS) == len(set(MEN_COLUMNS))


def test_target_construction_variables_are_extracted():
    for col in ["sb56", "sb57", "s728a", "sb74"]:
        assert col in WOMEN_COLUMNS
    for col in ["smb56", "smb57", "sm627a", "smb74"]:
        assert col in MEN_COLUMNS


def test_all_production_features_are_extractable():
    """Every column the production models actually use must trace back to a raw variable
    this script extracts (after the tob_/alc_/diet_ prefixes predictor construction adds)."""
    for sex, raw_cols in [("women", WOMEN_COLUMNS), ("men", MEN_COLUMNS)]:
        with open(PRODUCTION_PIPELINE_PATHS[sex], "rb") as f:
            feature_cols = pickle.load(f)["feature_cols"]
        for fc in feature_cols:
            raw_name = fc
            for prefix in ("tob_", "alc_", "diet_"):
                if raw_name.startswith(prefix):
                    raw_name = raw_name[len(prefix):]
            if raw_name in ("age",):
                raw_name = "v012" if sex == "women" else "mv012"
            elif raw_name in ("bmi",):
                raw_name = "v445"
            elif raw_name in ("education",):
                raw_name = "v106" if sex == "women" else "mv106"
            elif raw_name in ("residence",):
                raw_name = "v025" if sex == "women" else "mv025"
            elif raw_name in ("avg_systolic_bp", "avg_diastolic_bp", "n_valid_systolic_readings",
                              "n_valid_diastolic_readings"):
                continue  # derived from multiple BP reading columns, checked separately
            elif raw_name == "drinks":
                raw_name = "s720" if sex == "women" else "sm619"
            elif raw_name == "frequency":
                raw_name = "s721" if sex == "women" else "sm620"
            elif raw_name == "duration_years":
                raw_name = "s723" if sex == "women" else "sm622"
            assert raw_name in raw_cols, f"{sex}: feature '{fc}' has no traceable raw source column"


def test_no_pregnancy_variable_extracted_for_men():
    assert not any("213" in c or "214" in c or "215" in c for c in MEN_COLUMNS), (
        "men's column list must not include any pregnancy-related variable "
        "(mv213 is the PARTNER's pregnancy status and must never be used)")


def test_missing_raw_file_raises_clear_error():
    try:
        find_raw_file(Path("/nonexistent_dir_for_test"), "IAIR7EFL.DTA")
        assert False, "should have raised FileNotFoundError"
    except FileNotFoundError as e:
        assert "DHS Program" in str(e)


def test_specs_reference_expected_locked_counts():
    assert SPECS["women"]["expected_age_18_40"] == 510_289
    assert SPECS["men"]["expected_age_18_40"] == 64_918


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t(); print(f"[PASS] {t.__name__}"); passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}"); failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
