"""Research-layer guards: production untouched, isolation, outputs present, no leakage, Mapper behaviour."""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

BEFORE = json.loads((ROOT / "reports" / "integrity" / "production_artifact_hashes_before.json").read_text(encoding="utf-8"))


def _sha(rel):
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def test_production_artifacts_and_target_a_unchanged():
    for rel, rec in BEFORE["artifacts"].items():
        assert _sha(rel) == rec["sha256"], f"{rel} changed since the pre-change snapshot"


def test_thresholds_unchanged():
    m = json.loads((ROOT / "models" / "production_pipeline_manifest.json").read_text(encoding="utf-8"))
    assert m["final_thresholds"] == {"women": 0.026, "men": 0.027}
    assert BEFORE["thresholds"] == m["final_thresholds"]


def test_production_code_does_not_import_research():
    for rel in ("src/inference", "src/app", "src/models", "src/calibration", "src/evaluation", "backend"):
        for f in (ROOT / rel).rglob("*.py"):
            assert not re.search(r"(from|import)\s+src\.research", f.read_text(encoding="utf-8")), f"{f} imports research"


def test_research_never_writes_to_models_or_configs_or_touches_test_pickles():
    for f in (ROOT / "src" / "research").glob("*.py"):
        t = f.read_text(encoding="utf-8")
        assert not re.search(r"(to_pickle|dump|write_text|write_bytes|open)\([^)]*(models|configs)[/\\\"']", t), f.name
        assert "women_test.pkl" not in t and "men_test.pkl" not in t, f"{f.name} reads the final test pickles"


def test_no_random_cv_in_production_source():
    from src.research.validation_checks import scan_for_random_cv
    assert scan_for_random_cv() == []


def test_psu_overlap_helpers():
    from src.research.validation_checks import psu_overlap, assert_no_psu_overlap, cv_fold_psu_overlaps
    assert psu_overlap([1, 2], [2, 3]) == {2}
    assert_no_psu_overlap([1, 2], [3])
    g = np.array([1, 1, 2, 2, 3, 3])
    assert cv_fold_psu_overlaps(g, [(np.array([0, 1, 2, 3]), np.array([4, 5]))]) == [0]
    assert cv_fold_psu_overlaps(g, [(np.array([0, 2, 4]), np.array([1, 3, 5]))]) == [3]   # a random split leaks PSUs


def test_research_outputs_generated():
    out = ROOT / "reports" / "research"
    for name in ("target_definition_comparison.csv", "target_definition_comparison.json", "target_definition_comparison.md",
                 "baseline_comparison.csv", "calibration_analysis.csv", "decision_curve.csv", "subgroup_performance.csv",
                 "idrs_compatibility.csv", "idrs_comparison.csv", "research_summary.json", "feature_importance.csv"):
        assert (out / name).exists(), name
    assert len(list((out / "figures").glob("*.png"))) >= 6


def test_alternative_target_not_fabricated():
    r = json.loads((ROOT / "reports" / "research" / "target_definition_comparison.json").read_text(encoding="utf-8"))
    assert r["full_replication_possible"] is False and "NOT used by the EarlyDia production model" in r["label"]


def test_summary_matches_locked_final_eval():
    fe = json.loads((ROOT / "reports" / "final_evaluation" / "final_test_evaluation.json").read_text(encoding="utf-8"))
    s = json.loads((ROOT / "reports" / "research" / "research_summary.json").read_text(encoding="utf-8"))
    for sex in ("women", "men"):
        assert s["production_model"][sex]["roc_auc"] == fe[sex]["overall"]["roc_auc"]
        assert s["production_model"][sex]["threshold"] == fe[sex]["threshold"]


def test_dca_exact_net_benefit_formula():
    from src.research.common import net_benefit
    assert abs(net_benefit(10, 20, 100, 0.2) - (0.1 - 0.2 * 0.25)) < 1e-12


def _blobs(seed=1):
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal(m, 1, (500, 5)) for m in (0, 3, 6)])


def test_mapper_reproducible_and_target_free():
    import pandas as pd
    from src.research.mapper_analysis import build_mapper, encode_predictors
    X = _blobs()
    a, b = build_mapper(X), build_mapper(X)
    assert (a["point_community"] == b["point_community"]).all() and len(a["communities"]) >= 2
    df = pd.DataFrame({"age": np.arange(50.0), "c": np.arange(50) % 3, "target_a": np.arange(50) % 2})
    Z1, _, _ = encode_predictors(df, ["age"], ["c"])
    df["target_a"] = 1 - df["target_a"]
    Z2, _, _ = encode_predictors(df, ["age"], ["c"])
    assert np.array_equal(Z1, Z2), "encoding depends on the target"
    from src.models.grouped_cv_baseline import get_feature_lists
    for sex in ("women", "men"):
        assert "target_a" not in sum(map(list, get_feature_lists(sex)), [])


def test_mapper_stability_and_null_model():
    from src.research.mapper_stability import assess
    from src.research.mapper_null_model import run_null, VERDICT_NOT_SUPPORTED
    X = _blobs()
    st = assess(X, n_boot=2)
    assert st["mean_ari"] > 0.5 and all(st["stable"])
    nm = run_null(X, B=8)
    assert nm["observed"] > nm["null_mean"] and len(nm["null_values"]) == 8
    rng = np.random.default_rng(3)
    G = rng.normal(size=(1500, 5)) @ rng.normal(size=(5, 5))      # pure covariance structure
    assert run_null(G, B=8)["empirical_p"] >= 0.05 and VERDICT_NOT_SUPPORTED


def test_survey_weighted_estimator_sanity():
    from src.research.survey_analysis import weighted_prevalence
    rng = np.random.default_rng(0); n = 6000
    y = (rng.random(n) < 0.05).astype(int)
    p, lo, hi, nn = weighted_prevalence(y, np.ones(n), rng.integers(0, 300, n), rng.integers(0, 10, n))
    assert lo <= p <= hi and abs(p - y.mean()) < 1e-9 and nn == n


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print(f"[PASS] {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"[FAIL] {name}: {e!r}")
    print(f"{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
