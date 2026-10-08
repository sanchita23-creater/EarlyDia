import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.v2.run_v2 import run_oof, _show, KEY_COLS

candidates = [
    ("lr_state", "state", "lr"),
    ("state_native_B", "state_native", "hgb_B"),
    ("state_native_C", "state_native", "hgb_C"),
    ("state_native_D", "state_native", "hgb_D"),
    ("state_native_eng", "state_native,eng", "hgb_default"),
    ("state_native_eng_C", "state_native,eng", "hgb_C"),
    ("state_native_wealth", "state_native,wealth_v190", "hgb_default"),
]

rows = []
for name, features, model in candidates:
    print(f"\n--- Running women: {name} ({features}, {model}) ---")
    s, _ = run_oof("women", name, features, model, force=False)
    rows.append(dict(name=name, **s))

print("\n=== WOMEN EXPERIMENT RESULTS ===")
_show(rows, KEY_COLS)
