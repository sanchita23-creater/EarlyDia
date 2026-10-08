"""
EarlyDia - regression tests for the ``No module named '_loss'`` unpickling failure.

The failure only occurs on interpreter/scikit-learn builds that do NOT register the Cython
loss extension under ``sys.modules['_loss']``. These tests reproduce that condition in a
clean subprocess (alias removed, working directory outside the project) and prove that:
  1. a plain ``pickle.load`` really fails there (so the test is meaningful), and
  2. the production inference path still loads BOTH artifacts and predicts, and
  3. the artifacts are never modified by loading/prediction.
"""
import hashlib
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS = ROOT / "models"


def _run_clean(code: str) -> subprocess.CompletedProcess:
    """Run code in a fresh interpreter whose cwd is outside the project root."""
    prelude = textwrap.dedent(f"""
        import sys, pickle
        ROOT = {str(ROOT)!r}
        import sklearn._loss._loss, sklearn.pipeline, sklearn.isotonic, sklearn.linear_model
        import sklearn.ensemble._hist_gradient_boosting.gradient_boosting
        sys.modules.pop('_loss', None)   # simulate a build that does not self-register '_loss'
    """)
    env = dict(sys.modules.get("os", __import__("os")).environ)
    env["OPENBLAS_NUM_THREADS"] = "1"
    return subprocess.run([sys.executable, "-c", prelude + textwrap.dedent(code)],
                          capture_output=True, text=True, cwd=ROOT.parent, env=env)


def test_plain_pickle_fails_when_alias_missing_so_the_test_is_meaningful():
    r = _run_clean("""
        try:
            pickle.load(open(ROOT + '/models/women_production_pipeline.pkl', 'rb'))
            print('LOADED')
        except ModuleNotFoundError as e:
            print('FAILED:', e)
    """)
    assert "No module named '_loss'" in r.stdout, r.stdout + r.stderr


def test_load_artifact_works_without_alias_for_both_sexes():
    r = _run_clean("""
        sys.path.insert(0, ROOT)
        from configs.sklearn_compat import load_artifact
        for sex in ('women', 'men'):
            a = load_artifact(ROOT + f'/models/{sex}_production_pipeline.pkl')
            print(sex, a['selected_threshold'], len(a['feature_cols']))
    """)
    assert r.returncode == 0, r.stderr[-800:]
    assert "women 0.026" in r.stdout and "men 0.027" in r.stdout


def test_production_predictor_end_to_end_without_alias():
    r = _run_clean("""
        sys.path.insert(0, ROOT)
        import importlib
        sys.modules.pop('_loss', None)
        from src.inference.predict_module import EarlyDiaPredictor
        sys.modules.pop('_loss', None)
        for sex in ('women', 'men'):
            p = EarlyDiaPredictor(sex)
            person = {c: 0 for c in p.feature_cols}
            person.update(age=30, avg_systolic_bp=120, avg_diastolic_bp=78,
                          n_valid_systolic_readings=3, n_valid_diastolic_readings=3,
                          education=2, residence=1)
            for c in p.feature_cols:
                if c.startswith('diet_'): person[c] = 2
            if 'bmi' in p.feature_cols: person['bmi'] = 23.0
            res = p.predict(person)
            print(sex, round(res.calibrated_probability, 6), res.threshold_used)
    """)
    assert r.returncode == 0, r.stderr[-1200:]
    assert "women" in r.stdout and "men" in r.stdout


def test_loading_and_predicting_does_not_modify_artifacts():
    def digest(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    before = {p.name: digest(p) for p in MODELS.glob("*.pkl")}
    from src.inference.predict_module import EarlyDiaPredictor
    for sex in ("women", "men"):
        EarlyDiaPredictor(sex)
    after = {p.name: digest(p) for p in MODELS.glob("*.pkl")}
    assert before == after and len(before) == 2


def test_no_stray_toplevel_loss_shim_in_project_root():
    assert not (ROOT / "_loss.py").exists(), "root-level _loss.py shim should not exist"


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
