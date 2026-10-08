"""
EarlyDia - Artifact existence and loading tests. Verifies the project's saved artifacts
are present and structurally intact, without re-running or modifying any training.
"""
import json
import pickle
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import (PRODUCTION_PIPELINE_PATHS, CONFIGS_DIR, REPORTS_DIR,
                             EXPLAINABILITY_REPORTS_DIR, FINAL_EVAL_REPORTS_DIR, MODELS_DIR)

REQUIRED_ARTIFACT_KEYS = {"base_pipeline", "feature_cols", "calibration_method",
                           "calibrator", "selected_threshold", "sex", "random_seed"}


def test_production_pipelines_exist_and_load():
    for sex in ["women", "men"]:
        path = PRODUCTION_PIPELINE_PATHS[sex]
        assert path.exists(), f"missing production pipeline artifact: {path}"
        with open(path, "rb") as f:
            artifact = pickle.load(f)
        missing_keys = REQUIRED_ARTIFACT_KEYS - set(artifact.keys())
        assert not missing_keys, f"{sex} artifact missing keys: {missing_keys}"
        assert hasattr(artifact["base_pipeline"], "predict_proba"), \
            f"{sex} base_pipeline has no predict_proba method"


def test_config_files_exist():
    for fname in ["reproducibility_manifest.json", "split_config.json", "cv_config.json",
                  "model_selection.json", "threshold_selection_summary.json"]:
        path = CONFIGS_DIR / fname
        assert path.exists(), f"missing config file: {path}"
        with open(path) as f:
            json.load(f)  # must be valid JSON


def test_explainability_reports_exist():
    for sex in ["women", "men"]:
        for fname in [f"{sex}_permutation_importance.csv", f"{sex}_lr_coefficients.csv",
                      f"{sex}_permutation_importance_barplot.png"]:
            path = EXPLAINABILITY_REPORTS_DIR / fname
            assert path.exists(), f"missing explainability artifact: {path}"


def test_final_evaluation_report_exists():
    path = FINAL_EVAL_REPORTS_DIR / "final_test_evaluation.json"
    assert path.exists(), f"missing final evaluation report: {path}"
    with open(path) as f:
        report = json.load(f)
    assert "women" in report and "men" in report


def test_no_raw_data_in_project():
    """Guards against raw NFHS microdata accidentally being committed."""
    raw_dir = MODELS_DIR.parent / "data" / "raw"
    dta_files = list(raw_dir.glob("*.DTA")) + list(raw_dir.glob("*.dta"))
    assert len(dta_files) == 0, f"raw NFHS .DTA files found in data/raw/: {dta_files}"


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
