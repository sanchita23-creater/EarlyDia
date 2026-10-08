"""
EarlyDia - plain-language summaries of the user's answers (pure functions, no streamlit).

Used by the Review page, the Results "factors considered" section and the recommendations,
so every screen describes the same answers in the same words. Works on the `ui` dict produced
by input_spec.ui_from_answers() (and tolerates partial dicts, e.g. in unit tests).
"""
from src.app.input_spec import (
    TOBACCO_ITEMS, ALCOHOL_TYPE_ITEMS, DIET_ITEMS, DIET_OPTIONS, DIET_SHORT, DIET_DISPLAY_ORDER,
    ALCOHOL_FREQ_OPTIONS, TOBACCO_FREQ_OPTIONS, EDUCATION_OPTIONS, RESIDENCE_OPTIONS,
    CIGARETTE_COL, SEX_OPTIONS, diet_letter,
)


def sex_label(sex: str) -> str:
    return SEX_OPTIONS.get(sex, "-")


def education_label(code) -> str:
    return EDUCATION_OPTIONS.get(code, "-") if code is not None else "-"


def residence_label(code) -> str:
    return RESIDENCE_OPTIONS.get(code, "-") if code is not None else "-"


def tobacco_products(sex: str, ui: dict) -> list:
    flags = ui.get("tobacco", {}) or {}
    return [label for col, label in TOBACCO_ITEMS[sex] if flags.get(col)]


def tobacco_summary(sex: str, ui: dict) -> str:
    prods = tobacco_products(sex, ui)
    if not prods:
        return "No tobacco use"
    text = ", ".join(p[0].lower() + p[1:] if i else p for i, p in enumerate(prods))
    if ui.get("tob_freq") and (ui.get("tobacco", {}) or {}).get(CIGARETTE_COL[sex]):
        text += f" (cigarettes: {TOBACCO_FREQ_OPTIONS[ui['tob_freq']].lower()})"
    return text


def alcohol_types(sex: str, ui: dict) -> list:
    flags = ui.get("alc_types", {}) or {}
    return [label for col, label in ALCOHOL_TYPE_ITEMS[sex] if flags.get(col)]


def alcohol_summary(sex: str, ui: dict) -> str:
    if not ui.get("drinks"):
        return "Does not drink alcohol"
    parts = []
    if ui.get("alc_freq"):
        parts.append(ALCOHOL_FREQ_OPTIONS[ui["alc_freq"]].lower())
    years = ui.get("alc_duration")
    if years is not None:
        parts.append(f"for {years:.0f} year{'s' if round(years) != 1 else ''}")
    text = "Drinks alcohol" + (f" - {', '.join(parts)}" if parts else "")
    types = alcohol_types(sex, ui)
    if types:
        text += f" ({', '.join(types)})"
    return text


def diet_answers(sex: str, ui: dict) -> list:
    """[(short label, answer text)] in questionnaire order."""
    diet = ui.get("diet", {}) or {}
    return [(DIET_SHORT[diet_letter(col)], DIET_OPTIONS[diet[col]])
            for col, _ in DIET_ITEMS[sex] if col in diet]


def diet_counts_summary(sex: str, ui: dict) -> str:
    diet = ui.get("diet", {}) or {}
    counts = {code: 0 for code in DIET_OPTIONS}
    for col, _ in DIET_ITEMS[sex]:
        if col in diet:
            counts[diet[col]] += 1
    if not any(counts.values()):
        return "-"
    return " · ".join(f"{DIET_OPTIONS[c]}: {counts[c]}" for c in DIET_DISPLAY_ORDER)


def diet_value(sex: str, ui: dict, letter: str):
    """Answer code for the diet item with NFHS letter a..i, or None."""
    diet = ui.get("diet", {}) or {}
    for col, _ in DIET_ITEMS[sex]:
        if diet_letter(col) == letter and col in diet:
            return diet[col]
    return None


def bp_summary(ui: dict) -> str:
    n = ui.get("n_sys", 0)
    if not n or ui.get("avg_sys") is None or ui.get("avg_dia") is None:
        return "Not provided"
    return (f"{ui['avg_sys']:.0f} / {ui['avg_dia']:.0f} mmHg · average of {n} reported "
            f"reading{'s' if n != 1 else ''}")
