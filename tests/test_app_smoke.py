"""
EarlyDia - Streamlit app smoke test.
This environment does not have `streamlit` installed (no network access to install it
during development), so a live app run could not be tested here. This test does what
CAN be verified without streamlit: (1) the app file has valid Python syntax, (2) all
non-streamlit imports it depends on resolve correctly, (3) if streamlit IS available in
the environment running this test, it also attempts a real import of the app module.
"""
import ast
import importlib.util
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

APP_PATH = _PROJECT_ROOT / "src" / "app" / "app.py"
THEME_PATH = _PROJECT_ROOT / "src" / "app" / "theme.py"


def test_app_file_has_valid_syntax():
    with open(APP_PATH) as f:
        source = f.read()
    ast.parse(source)  # raises SyntaxError if invalid


def test_theme_file_has_valid_syntax():
    with open(THEME_PATH) as f:
        source = f.read()
    ast.parse(source)


def test_theme_module_imports_and_runs():
    """theme.py has no streamlit dependency - can be fully imported and exercised."""
    from src.app.theme import get_css
    css_light = get_css(dark_mode=False)
    css_dark = get_css(dark_mode=True)
    assert "<style>" in css_light and "<style>" in css_dark
    assert css_light != css_dark


def test_app_imports_predict_module_correctly():
    """Verify the app's dependency on predict_module resolves without needing streamlit -
    by checking the import statement is present and predict_module itself loads fine."""
    source = (APP_PATH.parent / "ui_state.py").read_text()
    assert "from src.inference.predict_module import EarlyDiaPredictor" in source
    from src.inference.predict_module import EarlyDiaPredictor  # must not raise
    assert EarlyDiaPredictor is not None


def test_streamlit_availability_and_conditional_full_import():
    spec = importlib.util.find_spec("streamlit")
    if spec is None:
        print("  streamlit is NOT installed in this environment - skipping live app "
              "import test. Run `pip install streamlit` and re-run this test for a full "
              "smoke test before deployment.")
        return
    # if streamlit IS available, attempt a real import of the app module
    import importlib.util as ilu
    module_spec = ilu.spec_from_file_location("earlydia_app", APP_PATH)
    module = ilu.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)  # will raise if the app has import/runtime errors


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
