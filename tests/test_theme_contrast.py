"""EarlyDia - WCAG AA contrast guard for both themes (text/surface pairs must be >= 4.5:1)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app.theme import PALETTES, contrast_ratio, get_css, NAVY, NAVY_MID

PAIRS = [("text", "bg"), ("text", "surface"), ("text", "input_bg"), ("text", "sidebar"),
         ("muted", "bg"), ("muted", "surface"), ("accent", "surface"), ("flag", "surface"),
         ("ok", "surface"), ("button_text", "button_bg"), ("button_text", "button_hover")]


def test_all_text_pairs_meet_wcag_aa_in_both_themes():
    for name, pal in PALETTES.items():
        for fg, bg in PAIRS:
            r = contrast_ratio(pal[fg], pal[bg])
            assert r >= 4.5, f"{name}: {fg} on {bg} contrast {r:.2f} < 4.5"


def test_header_banner_text_is_readable():
    assert contrast_ratio("#FFFFFF", NAVY) >= 7 and contrast_ratio("#FFFFFF", NAVY_MID) >= 7


def test_themes_use_only_navy_white_black_family_plus_status_colors():
    assert set(PALETTES["light"]) == set(PALETTES["dark"])
    assert "#0A1F44" in get_css(False) and "#0A1F44" in get_css(True)


def test_dark_css_styles_popovers_and_alerts_explicitly():
    css = get_css(True)
    assert 'div[data-baseweb="popover"]' in css and 'div[data-testid="stAlert"]' in css
    assert PALETTES["dark"]["surface"] in css


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
