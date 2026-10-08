"""EarlyDia - reproducibility audit: the manifest must record everything needed to
understand how the final model was produced, and must agree with the saved artifacts."""
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import (CONFIGS_DIR, PRODUCTION_PIPELINE_PATHS, RANDOM_SEED, FINAL_EVAL_REPORTS_DIR)

REQUIRED_MANIFEST_KEYS = ["random_seed", "population", "target_definition", "predictor_list",
                          "train_test_split", "grouped_cv", "selected_models", "calibration",
                          "thresholds_LOCKED", "final_test_evaluation", "known_environment_limitations"]


def _manifest():
    return json.loads((CONFIGS_DIR / "reproducibility_manifest.json").read_text())


def _artifact(sex):
    with open(PRODUCTION_PIPELINE_PATHS[sex], "rb") as f:
        return pickle.load(f)


def test_manifest_has_all_required_sections():
    m = _manifest()
    missing = [k for k in REQUIRED_MANIFEST_KEYS if k not in m]
    assert not missing, f"manifest missing sections: {missing}"


def test_manifest_seed_matches_config_and_artifacts():
    m = _manifest()
    assert m["random_seed"] == RANDOM_SEED
    for sex in ["women", "men"]:
        assert _artifact(sex)["random_seed"] == RANDOM_SEED


def test_manifest_predictor_lists_match_artifacts():
    m = _manifest()
    for sex in ["women", "men"]:
        assert m["predictor_list"][f"{sex}_feature_cols"] == _artifact(sex)["feature_cols"]


def test_manifest_models_and_calibration_match_artifacts():
    m = _manifest()
    for sex in ["women", "men"]:
        a = _artifact(sex)
        assert m["selected_models"][sex]["model"] == "HistGradientBoosting"
        assert type(a["base_pipeline"].named_steps["clf"]).__name__ == "HistGradientBoostingClassifier"
        assert m["selected_models"][sex]["calibration_method"] == a["calibration_method"]


def test_split_and_cv_configs_record_seed_and_method():
    split = json.loads((CONFIGS_DIR / "split_config.json").read_text())
    cv = json.loads((CONFIGS_DIR / "cv_config.json").read_text())
    assert "GroupShuffleSplit" in split["method"]
    assert split["women"]["selected_seed"] == RANDOM_SEED and split["men"]["selected_seed"] == RANDOM_SEED
    assert cv["cv_strategy"] == "StratifiedGroupKFold" and cv["n_splits"] == 5 and cv["seed"] == RANDOM_SEED


def test_final_evaluation_record_uses_locked_configuration():
    ev = json.loads((FINAL_EVAL_REPORTS_DIR / "final_test_evaluation.json").read_text())
    for sex in ["women", "men"]:
        a = _artifact(sex)
        assert abs(ev[sex]["threshold"] - a["selected_threshold"]) < 1e-9
        assert ev[sex]["calibration_method"] == a["calibration_method"]
        assert ev[sex]["model"] == "HistGradientBoosting"


def test_final_evaluation_numbers_unchanged():
    """Locked results: spot-check key values recorded at evaluation time."""
    ev = json.loads((FINAL_EVAL_REPORTS_DIR / "final_test_evaluation.json").read_text())
    assert ev["women"]["overall"]["n"] == 86669 and ev["men"]["overall"]["n"] == 11408
    assert round(ev["women"]["overall"]["roc_auc"], 3) == 0.685
    assert round(ev["men"]["overall"]["roc_auc"], 3) == 0.665


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
