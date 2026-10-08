"""
EarlyDia - input specification for the app.

Single source of truth mapping user-facing labels to the model's feature columns.
Labels were verified against the NFHS-5 codebook variable labels (NOT written from memory):
note that the meaning of tobacco items differs by sex (e.g. women's v463f = cigars,
v463g = hookah; men's mv463f = hookah, mv463g = gutkha/paan masala), and that the alcohol
type order is Tadi/madi, Country liquor, Beer, Wine, Hard liquor, Other.

This module has NO streamlit dependency so it can be unit-tested. The app renders it;
build_person() converts UI values into the dict expected by predict_module.
"""

TOBACCO_ITEMS = {
    "women": [
        ("tob_v463a", "Smokes cigarettes"),
        ("tob_v463b", "Smokes pipe"),
        ("tob_v463c", "Chews tobacco"),
        ("tob_v463d", "Uses snuff"),
        ("tob_v463f", "Smokes cigars, cheroots or cigarillos"),
        ("tob_v463g", "Smokes water pipe / hookah"),
        ("tob_v463j", "Uses gutkha / paan masala with tobacco"),
        ("tob_v463k", "Uses khaini"),
        ("tob_v463l", "Uses paan with tobacco"),
        ("tob_v463x", "Uses other tobacco"),
    ],
    "men": [
        ("tob_mv463a", "Smokes cigarettes"),
        ("tob_mv463b", "Smokes pipe"),
        ("tob_mv463c", "Uses chewing tobacco"),
        ("tob_mv463d", "Uses snuff"),
        ("tob_mv463f", "Smokes hookah"),
        ("tob_mv463g", "Uses gutkha / paan masala with tobacco"),
        ("tob_mv463x", "Uses other tobacco"),
    ],
}
TOBACCO_NONE_COL = {"women": "tob_v463z", "men": "tob_mv463z"}
TOBACCO_FREQ_COL = {"women": "tob_v463aa", "men": None}   # men's equivalent not administered
CIGARETTE_COL = {"women": "tob_v463a", "men": "tob_mv463a"}

ALCOHOL_TYPE_ITEMS = {
    "women": [("alc_s722a", "Tadi / madi (toddy)"), ("alc_s722b", "Country liquor"),
              ("alc_s722c", "Beer"), ("alc_s722d", "Wine"),
              ("alc_s722e", "Hard liquor"), ("alc_s722x", "Other")],
    "men": [("alc_sm621a", "Tadi / madi (toddy)"), ("alc_sm621b", "Country liquor"),
            ("alc_sm621c", "Beer"), ("alc_sm621d", "Wine"),
            ("alc_sm621e", "Hard liquor"), ("alc_sm621x", "Other")],
}

_DIET_LABELS = ["Milk or curd", "Pulses or beans", "Dark green leafy vegetables", "Fruits",
                "Eggs", "Fish", "Chicken or meat", "Fried food", "Aerated drinks"]
DIET_ITEMS = {
    "women": [(f"diet_s731{c}", lab) for c, lab in zip("abcdefghi", _DIET_LABELS)],
    "men": [(f"diet_sm630{c}", lab) for c, lab in zip("abcdefghi", _DIET_LABELS)],
}
DIET_OPTIONS = {0: "Never", 1: "Daily", 2: "Weekly", 3: "Occasionally"}
ALCOHOL_FREQ_OPTIONS = {1: "Almost every day", 2: "About once a week", 3: "Less than once a week"}
TOBACCO_FREQ_OPTIONS = {1: "Every day", 2: "Some days"}
EDUCATION_OPTIONS = {0: "No education", 1: "Primary", 2: "Secondary", 3: "Higher"}
RESIDENCE_OPTIONS = {1: "Urban", 2: "Rural"}


def build_person(sex: str, ui: dict) -> dict:
    """Convert plain UI values into the input dict for EarlyDiaPredictor.predict().

    Expected keys in `ui`: age, avg_sys, avg_dia, n_sys, n_dia, tobacco (dict col->bool),
    tob_freq (1/2 or None), drinks (bool), alc_freq (1/2/3 or None), alc_duration,
    alc_types (dict col->bool), diet (dict col->int), education, residence, and bmi (women).

    Consistency rules mirror training-time preprocessing:
      * average BP is missing (None) when zero valid readings are reported;
      * non-drinkers get alcohol frequency/duration/type = 0 (structural, as in NFHS skip pattern);
      * "no tobacco" indicator is DERIVED from the item flags, so it can never contradict them.
    """
    if sex not in ("women", "men"):
        raise ValueError(f"sex must be 'women' or 'men', got {sex!r}")

    person = {
        "age": ui["age"],
        "avg_systolic_bp": ui["avg_sys"] if ui["n_sys"] > 0 else None,
        "avg_diastolic_bp": ui["avg_dia"] if ui["n_dia"] > 0 else None,
        "n_valid_systolic_readings": ui["n_sys"],
        "n_valid_diastolic_readings": ui["n_dia"],
        "education": ui["education"],
        "residence": ui["residence"],
    }
    if sex == "women":
        person["bmi"] = ui["bmi"]

    any_tobacco = False
    for col, _ in TOBACCO_ITEMS[sex]:
        flag = int(bool(ui["tobacco"].get(col, False)))
        person[col] = flag
        any_tobacco = any_tobacco or bool(flag)
    person[TOBACCO_NONE_COL[sex]] = int(not any_tobacco)
    freq_col = TOBACCO_FREQ_COL[sex]
    if freq_col:
        smokes_cig = bool(ui["tobacco"].get(CIGARETTE_COL[sex], False))
        person[freq_col] = int(ui["tob_freq"]) if (smokes_cig and ui.get("tob_freq")) else 0

    drinks = bool(ui["drinks"])
    person["alc_drinks"] = int(drinks)
    person["alc_frequency"] = int(ui["alc_freq"]) if (drinks and ui.get("alc_freq")) else 0
    person["alc_duration_years"] = float(ui["alc_duration"]) if drinks else 0.0
    for col, _ in ALCOHOL_TYPE_ITEMS[sex]:
        person[col] = int(bool(ui["alc_types"].get(col, False))) if drinks else 0

    for col, _ in DIET_ITEMS[sex]:
        person[col] = int(ui["diet"][col])
    return person


# ===========================================================================
# Product-UI layer (pure functions, no streamlit): human-readable questions, BMI from
# height/weight, validation, and the mapping from the UI's flat "answers" dict to the
# `ui` dict consumed by build_person(). NOTHING here changes the model feature contract:
# the column names, codes and sex-specific feature sets above are untouched.
# ===========================================================================

SEX_OPTIONS = {"women": "Female", "men": "Male"}

HEIGHT_RANGE_CM = (120.0, 220.0)
WEIGHT_RANGE_KG = (25.0, 200.0)
BMI_MODEL_RANGE = (10.0, 70.0)        # mirrors predict_module.VALID_RANGES["bmi"]
SYS_RANGE = (70.0, 270.0)             # mirrors predict_module.VALID_RANGES
DIA_RANGE = (40.0, 150.0)
AGE_RANGE = (18, 40)

# Human-readable diet questions, keyed by the NFHS item letter (a..i). The underlying
# feature mapping (diet_s731a.. / diet_sm630a..) is unchanged: letter -> existing column.
DIET_QUESTIONS = {
    "a": "How often do you have milk or curd?",
    "b": "How often do you eat pulses or beans?",
    "c": "How often do you eat dark green leafy vegetables?",
    "d": "How often do you eat fruits?",
    "e": "How often do you eat eggs?",
    "f": "How often do you eat fish?",
    "g": "How often do you eat chicken or meat?",
    "h": "How often do you eat fried food?",
    "i": "How often do you have aerated (fizzy) drinks?",
}
DIET_SHORT = {
    "a": "Milk or curd", "b": "Pulses or beans", "c": "Dark green leafy vegetables",
    "d": "Fruits", "e": "Eggs", "f": "Fish", "g": "Chicken or meat",
    "h": "Fried food", "i": "Aerated drinks",
}
# Display order for answers (values/codes are the NFHS codes in DIET_OPTIONS)
DIET_DISPLAY_ORDER = [1, 2, 3, 0]          # Daily, Weekly, Occasionally, Never


def diet_letter(col: str) -> str:
    """'diet_s731h' / 'diet_sm630h' -> 'h'."""
    return col[-1]


def compute_bmi(height_cm, weight_kg):
    """BMI = kg / m^2, rounded to 1 decimal (the precision shown to the user and passed to
    the women's model). Returns None when either value is missing/non-positive."""
    try:
        h, w = float(height_cm), float(weight_kg)
    except (TypeError, ValueError):
        return None
    if h <= 0 or w <= 0:
        return None
    return round(w / ((h / 100.0) ** 2), 1)


def bmi_band(bmi: float) -> str:
    """WHO Asian-population BMI cut-offs (descriptive label only)."""
    if bmi < 18.5:
        return "Below the typical range"
    if bmi < 23.0:
        return "Typical range"
    if bmi < 27.5:
        return "Above the typical range"
    return "Well above the typical range"


def blank_answers() -> dict:
    """Fresh, completely unanswered state (no pre-filled clinical assumptions)."""
    ans = {
        "sex": None, "age": None, "education": None, "residence": None,
        "uses_tobacco": None, "tob_freq": None,
        "drinks": None, "alc_freq": None, "alc_duration": None,
        "height_cm": None, "weight_kg": None,
        "bp_unavailable": False, "bp_sys": None, "bp_dia": None, "bp_n": None,
    }
    for sex in ("women", "men"):
        for col, _ in TOBACCO_ITEMS[sex]:
            ans[col] = False
        for col, _ in ALCOHOL_TYPE_ITEMS[sex]:
            ans[col] = False
        for col, _ in DIET_ITEMS[sex]:
            ans[col] = None
    return ans


def validate_profile(ans: dict) -> list:
    errs = []
    if ans.get("sex") not in ("women", "men"):
        errs.append("Select your sex.")
    if ans.get("age") is None:
        errs.append("Enter your age (18-40).")
    elif not (AGE_RANGE[0] <= ans["age"] <= AGE_RANGE[1]):
        errs.append("EarlyDia is designed for ages 18-40.")
    if ans.get("education") is None:
        errs.append("Select your education level.")
    if ans.get("residence") is None:
        errs.append("Select where you live (urban or rural).")
    return errs


def validate_lifestyle(ans: dict) -> list:
    errs = []
    sex = ans.get("sex")
    if sex not in ("women", "men"):
        return ["Complete the Profile step first."]
    if ans.get("uses_tobacco") is None:
        errs.append("Tell us whether you currently use tobacco.")
    elif ans["uses_tobacco"]:
        if not any(ans.get(c) for c, _ in TOBACCO_ITEMS[sex]):
            errs.append("Select at least one tobacco product you use.")
        elif TOBACCO_FREQ_COL[sex] and ans.get(CIGARETTE_COL[sex]) and ans.get("tob_freq") is None:
            errs.append("Tell us how often you smoke cigarettes.")
    if ans.get("drinks") is None:
        errs.append("Tell us whether you drink alcohol.")
    elif ans["drinks"]:
        if ans.get("alc_freq") is None:
            errs.append("Select how often you drink alcohol.")
        if ans.get("alc_duration") is None:
            errs.append("Enter for how many years you have been drinking.")
        elif ans.get("age") is not None and ans["alc_duration"] > ans["age"]:
            errs.append("Years of drinking cannot be greater than your age.")
    missing = [c for c, _ in DIET_ITEMS[sex] if ans.get(c) is None]
    if missing:
        errs.append(f"Answer all 9 food questions ({len(missing)} left).")
    return errs


def validate_health(ans: dict) -> list:
    errs = []
    sex = ans.get("sex")
    if sex not in ("women", "men"):
        return ["Complete the Profile step first."]
    if sex == "women":
        h, w = ans.get("height_cm"), ans.get("weight_kg")
        if h is None or w is None:
            errs.append("Enter your height and weight so BMI can be calculated.")
        else:
            bmi = compute_bmi(h, w)
            if bmi is None or not (BMI_MODEL_RANGE[0] <= bmi <= BMI_MODEL_RANGE[1]):
                errs.append("Those height/weight values give a BMI outside the supported range "
                            "(10-70). Please check them.")
    if not ans.get("bp_unavailable"):
        s, d, n = ans.get("bp_sys"), ans.get("bp_dia"), ans.get("bp_n")
        if s is None or d is None:
            errs.append("Enter your systolic and diastolic blood pressure, or tick "
                        "\"I don't have a recent blood pressure measurement\".")
        else:
            if s <= d:
                errs.append("Systolic pressure (top number) should be higher than diastolic (bottom number).")
        if n is None:
            errs.append("Select how many readings your blood pressure is based on.")
    return errs


def validate_all(ans: dict) -> list:
    return validate_profile(ans) + validate_lifestyle(ans) + validate_health(ans)


def ui_from_answers(ans: dict) -> dict:
    """Flat UI answers -> the `ui` dict expected by build_person() / generate_recommendations().

    * Women: BMI is computed from height/weight and passed to the model.
    * Men: NO bmi key is produced (the men's NFHS-5 model has no BMI feature).
    * BP: one "number of readings" answer applies to both systolic and diastolic counts;
      "no measurement" -> 0 readings (average treated as missing, imputed by the pipeline).
    """
    sex = ans["sex"]
    n = 0 if ans.get("bp_unavailable") else int(ans["bp_n"])
    uses_tob = bool(ans.get("uses_tobacco"))
    drinks = bool(ans.get("drinks"))
    ui = {
        "age": int(ans["age"]),
        "avg_sys": float(ans["bp_sys"]) if n > 0 else None,
        "avg_dia": float(ans["bp_dia"]) if n > 0 else None,
        "n_sys": n, "n_dia": n,
        "tobacco": {c: bool(ans.get(c)) and uses_tob for c, _ in TOBACCO_ITEMS[sex]},
        "drinks": drinks,
        "alc_freq": int(ans["alc_freq"]) if drinks else None,
        "alc_duration": float(ans["alc_duration"]) if drinks else 0.0,
        "alc_types": {c: bool(ans.get(c)) and drinks for c, _ in ALCOHOL_TYPE_ITEMS[sex]},
        "diet": {c: int(ans[c]) for c, _ in DIET_ITEMS[sex]},
        "education": int(ans["education"]),
        "residence": int(ans["residence"]),
    }
    smokes = bool(ui["tobacco"].get(CIGARETTE_COL[sex]))
    ui["tob_freq"] = int(ans["tob_freq"]) if (TOBACCO_FREQ_COL[sex] and smokes and ans.get("tob_freq")) else None
    if sex == "women":
        ui["bmi"] = compute_bmi(ans["height_cm"], ans["weight_kg"])
        ui["height_cm"] = float(ans["height_cm"])
        ui["weight_kg"] = float(ans["weight_kg"])
    return ui
