"""EarlyDia - project hygiene: no machine-specific paths, no secrets, no raw data, deps declared."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEXT_EXT = {".py", ".json", ".md", ".txt", ".csv", ".toml", ".yaml", ".yml"}
SECRET_PATTERN = re.compile(r"(api[_-]?key|secret[_-]?key|password\s*=|bearer\s+[a-z0-9]|sk-[a-z0-9]{10,})", re.I)


# Third-party / generated directories (e.g. frontend/node_modules after `npm install`, the Vite
# build output, virtual environments) are not project source, so they are not scanned.
_VENDORED = {"node_modules", "dist", ".venv", "venv", "__pycache__"}


def _text_files():
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix in TEXT_EXT and not (set(p.relative_to(ROOT).parts) & _VENDORED):
            yield p


def test_no_machine_specific_absolute_paths():
    needle = "/home/" + "claude"  # split so this file does not flag itself
    offenders = [str(p.relative_to(ROOT)) for p in _text_files()
                 if needle in p.read_text(errors="ignore")]
    assert not offenders, f"machine-specific paths found in: {offenders}"


def test_no_secrets_in_project():
    offenders = []
    for p in _text_files():
        if p.name == "test_project_hygiene.py":
            continue
        if SECRET_PATTERN.search(p.read_text(errors="ignore")):
            offenders.append(str(p.relative_to(ROOT)))
    assert not offenders, f"possible secrets found in: {offenders}"


def test_no_raw_nfhs_files_anywhere():
    bad = [p for p in ROOT.rglob("*") if p.suffix.lower() in {".dta", ".dct"} and p.is_file()]
    assert not bad, f"raw NFHS files present: {bad}"


def test_requirements_declares_core_dependencies():
    text = (ROOT / "requirements.txt").read_text().lower()
    for dep in ["pandas", "numpy", "scikit-learn", "streamlit"]:
        assert dep in text, f"{dep} missing from requirements.txt"


def test_gitignore_excludes_raw_and_processed_data():
    text = (ROOT / ".gitignore").read_text()
    assert "data/raw/*" in text and "data/processed/*" in text


def test_predict_module_does_not_reference_test_data_or_fit():
    src = (ROOT / "src" / "inference" / "predict_module.py").read_text()
    code_lines = [l for l in src.splitlines() if not l.strip().startswith("#")]
    code = "\n".join(code_lines)
    assert "_test.pkl" not in code.replace('"""', "").split("class EarlyDiaPredictor")[1]
    assert ".fit(" not in code


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
