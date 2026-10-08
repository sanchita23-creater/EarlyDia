"""
EarlyDia - Tests that the locked threshold configuration is consistent everywhere it
appears (production artifact, config.py documentation constant, reproducibility manifest).
There should be exactly ONE functional source of truth (the artifact); this test guards
against the documentation-only constant silently drifting out of sync with it, and
against the reproducibility manifest silently drifting (which already happened once -
see the correction note in configs/threshold_selection_summary.json).
"""
import json
import pickle
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import DOCUMENTED_THRESHOLDS, PRODUCTION_PIPELINE_PATHS, CONFIGS_DIR


def _load_artifact_threshold(sex):
    with open(PRODUCTION_PIPELINE_PATHS[sex], "rb") as f:
        artifact = pickle.load(f)
    return artifact["selected_threshold"]


def test_artifact_threshold_matches_documented_constant():
    for sex in ["women", "men"]:
        artifact_threshold = _load_artifact_threshold(sex)
        documented = DOCUMENTED_THRESHOLDS[sex]
        assert abs(artifact_threshold - documented) < 1e-9, (
            f"{sex}: artifact threshold {artifact_threshold} != documented constant "
            f"{documented} - these must never drift apart")


def test_reproducibility_manifest_matches_artifact():
    manifest_path = CONFIGS_DIR / "reproducibility_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    for sex in ["women", "men"]:
        artifact_threshold = _load_artifact_threshold(sex)
        manifest_threshold = manifest["thresholds_LOCKED"][sex]
        assert abs(artifact_threshold - manifest_threshold) < 1e-9, (
            f"{sex}: artifact threshold {artifact_threshold} != manifest threshold "
            f"{manifest_threshold}")


def test_locked_threshold_values_exact():
    """The final thresholds are exactly 0.026 (women) and 0.027 (men)."""
    assert abs(_load_artifact_threshold("women") - 0.026) < 1e-9
    assert abs(_load_artifact_threshold("men") - 0.027) < 1e-9


def test_no_stale_threshold_in_configs_or_scripts():
    """The final thresholds must be stated consistently; no config/manifest may describe
    a provisional or superseded value, and final_refit must not hardcode one."""
    import re
    for path in list((CONFIGS_DIR).glob("*.json")) + list((CONFIGS_DIR.parent / "models").glob("*.json")):
        text = path.read_text().lower()
        assert "provisional" not in text and "stale" not in text and "0.020" not in text, path.name
    refit = (CONFIGS_DIR.parent / "src" / "models" / "final_refit.py").read_text()
    assert "DOCUMENTED_THRESHOLDS" in refit and not re.search(r"SELECTED_THRESHOLD\s*=\s*\{[^}]*\d\.\d", refit)


def test_threshold_summary_metrics_are_internally_consistent():
    import json
    summary = json.loads((CONFIGS_DIR / "threshold_selection_summary.json").read_text())
    for sex in ["women", "men"]:
        s = summary[sex]
        assert abs(s["selected_threshold"] - DOCUMENTED_THRESHOLDS[sex]) < 1e-9
        m = s["threshold_metrics_on_calibrated_oof"]
        tp, fp, tn, fn = m["tp"], m["fp"], m["tn"], m["fn"]
        assert abs(m["sensitivity"] - tp / (tp + fn)) < 1e-3
        assert abs(m["specificity"] - tn / (tn + fp)) < 1e-3
        assert abs(m["precision_ppv"] - tp / (tp + fp)) < 1e-3
        assert m["sensitivity"] >= 0.75


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
