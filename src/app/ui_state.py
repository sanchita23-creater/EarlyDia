"""
EarlyDia - session-state and navigation logic for the multi-step app.

Design notes
------------
* All user answers live in ONE dict (`st.session_state["ans"]`), keyed by model-feature-neutral
  names (see input_spec.blank_answers). Widgets use keys "w_<name>" and are re-hydrated from
  that dict whenever a page is shown, because Streamlit drops the state of widgets that are not
  rendered in a run. This is what makes Back / Edit navigation lossless.
* Navigation uses button callbacks (`on_click`) so state changes before the next script run.
* `reset_screening()` clears every answer, widget key, result and edit flag, so a new person
  never sees the previous person's data. Only the light/dark preference survives.
"""
import streamlit as st

from src.app.input_spec import (
    blank_answers, validate_profile, validate_lifestyle, validate_health, validate_all,
    ui_from_answers, build_person,
)
from src.app.recommendations import generate_recommendations

PAGES = ("landing", "profile", "lifestyle", "health", "review", "result", "recs")
_VALIDATORS = {"profile": validate_profile, "lifestyle": validate_lifestyle, "health": validate_health}
_NEXT = {"profile": "lifestyle", "lifestyle": "health", "health": "review"}
_PREV = {"lifestyle": "profile", "health": "lifestyle", "review": "health"}


@st.cache_resource(show_spinner=False)
def get_predictor(sex_key: str):
    """Load a production pipeline once per server process (read-only artifacts)."""
    from src.inference.predict_module import EarlyDiaPredictor
    return EarlyDiaPredictor(sex_key)


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("page", "landing")
    ss.setdefault("dark_mode", False)
    if "ans" not in ss:
        ss["ans"] = blank_answers()
    for key, default in (("result", None), ("report", None), ("snapshot", None), ("errors", []),
                         ("edit_return", False), ("edit_sex", None)):
        ss.setdefault(key, default)


def sync_answers() -> None:
    """Copy current widget values into the persistent answers dict."""
    ss = st.session_state
    for name in list(ss["ans"]):
        key = f"w_{name}"
        if key in ss:
            ss["ans"][name] = ss[key]


def goto(page: str) -> None:
    assert page in PAGES, page
    st.session_state["page"] = page
    st.session_state["errors"] = []


def reset_screening() -> None:
    """Start New Screening: wipe all answers, widget state, results and edit flags."""
    ss = st.session_state
    for key in [k for k in ss if k.startswith("w_")]:
        del ss[key]
    ss["ans"] = blank_answers()
    ss["result"] = ss["report"] = ss["snapshot"] = None
    ss["edit_return"] = False
    ss["edit_sex"] = None
    ss["errors"] = []
    ss["page"] = "landing"


# ---- callbacks -------------------------------------------------------------
def cb_start() -> None:
    goto("profile")


def cb_continue(section: str) -> None:
    """Validate the current section, then advance (or return to Review when editing)."""
    ss = st.session_state
    sync_answers()
    errs = _VALIDATORS[section](ss["ans"])
    if errs:
        ss["errors"] = errs
        return
    if ss["edit_return"] and ss["ans"]["sex"] == ss["edit_sex"]:
        ss["edit_return"] = False
        goto("review")
        return
    # a changed sex invalidates sex-specific answers: walk through the remaining steps again
    ss["edit_return"] = False
    goto(_NEXT[section])


def cb_back(section: str) -> None:
    ss = st.session_state
    sync_answers()
    if ss["edit_return"]:
        ss["edit_return"] = False
        goto("review")
    else:
        goto(_PREV[section])


def cb_edit(section: str) -> None:
    ss = st.session_state
    sync_answers()
    ss["edit_return"] = True
    ss["edit_sex"] = ss["ans"]["sex"]
    goto(section)


def cb_back_to_review() -> None:
    goto("review")


def cb_modify_from_result() -> None:
    """Leaving the result to edit answers invalidates it (it is recomputed on the next run)."""
    ss = st.session_state
    ss["result"] = ss["report"] = ss["snapshot"] = None
    goto("review")


def cb_run_screening() -> None:
    ss = st.session_state
    sync_answers()
    errs = validate_all(ss["ans"])
    if errs:
        ss["errors"] = errs
        return
    try:
        ans = ss["ans"]
        sex_key = ans["sex"]
        ui = ui_from_answers(ans)
        person = build_person(sex_key, ui)
        result = get_predictor(sex_key).predict(person)
        report = generate_recommendations(sex_key, ui, result)
    except Exception as exc:  # surfaced to the user, never silently swallowed
        ss["errors"] = [f"Could not run the screening: {exc}"]
        return
    ss["result"], ss["report"], ss["snapshot"] = result, report, ui
    goto("result")


def cb_view_recs() -> None:
    goto("recs")


def cb_back_to_result() -> None:
    goto("result")


def cb_new_screening() -> None:
    reset_screening()
