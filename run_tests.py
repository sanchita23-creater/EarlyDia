"""Runs every tests/test_*.py file (pytest-free fallback). Usage: python run_tests.py
If pytest is installed, `python -m pytest tests -v` works equally well."""
import subprocess, sys
from pathlib import Path

root = Path(__file__).resolve().parent
total_fail = 0
for f in sorted((root / "tests").glob("test_*.py")):
    r = subprocess.run([sys.executable, str(f)], capture_output=True, text=True, cwd=root)
    print(f"\n##### {f.name}\n{r.stdout.strip()}")
    if r.returncode != 0:
        total_fail += 1
        print(r.stderr.strip()[-800:])
print("\nOVERALL:", "ALL TEST FILES PASSED" if total_fail == 0 else f"{total_fail} TEST FILE(S) FAILED")
sys.exit(1 if total_fail else 0)
