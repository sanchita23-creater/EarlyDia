"""
EarlyDia - page renderers for the multi-step Streamlit app.

Pages: landing -> profile -> lifestyle -> health -> review -> result -> recs.
Pure presentation: inference runs through ui_state.cb_run_screening() (which calls the
production prediction module); no model or preprocessing logic lives here.
"""
import streamlit as st

from src.app import ui_components as C
from src.app import summaries as S
from src.app import ui_state as U
from src.app.gauge import render_semicircular_gauge
from src.app.input_spec import (
    TOBACCO_ITEMS, TOBACCO_FREQ_COL, CIGARETTE_COL, ALCOHOL_TYPE_ITEMS, DIET_ITEMS, DIET_OPTIONS,
    DIET_QUESTIONS, DIET_DISPLAY_ORDER, ALCOHOL_FREQ_OPTIONS, TOBACCO_FREQ_OPTIONS,
    EDUCATION_OPTIONS, RESIDENCE_OPTIONS, SEX_OPTIONS, AGE_RANGE, HEIGHT_RANGE_CM, WEIGHT_RANGE_KG,
    SYS_RANGE, DIA_RANGE, compute_bmi, bmi_band, ui_from_answers, diet_letter,
)

DISCLAIMER = ("<b>EarlyDia is a screening tool, not a diagnostic test.</b> It does not diagnose diabetes and "
              "should not replace professional medical evaluation or clinical testing.")


def md(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# widget helpers: keys are "w_<answer name>"; values are mirrored into session_state["ans"]
# ---------------------------------------------------------------------------
def _key(name: str) -> str:
    key = f"w_{name}"
    ans_val = st.session_state["ans"].get(name)
    if key not in st.session_state and ans_val is not None:
        st.session_state[key] = ans_val
    return key


def w_radio(name, label, options, fmt=str, horizontal=True, help=None):
    v = st.radio(label, options, index=None, key=_key(name), format_func=fmt,
                 horizontal=horizontal, help=help)
    st.session_state["ans"][name] = v
    return v


def w_select(name, label, options, fmt=str, placeholder="Select...", help=None):
    v = st.selectbox(label, options, index=None, key=_key(name), format_func=fmt,
                     placeholder=placeholder, help=help)
    st.session_state["ans"][name] = v
    return v


def w_number(name, label, lo, hi, step, placeholder="", fmt=None, help=None):
    kw = {"format": fmt} if fmt else {}
    v = st.number_input(label, min_value=lo, max_value=hi, value=None, step=step, key=_key(name),
                        placeholder=placeholder, help=help, **kw)
    st.session_state["ans"][name] = v
    return v


def w_check(name, label):
    v = st.checkbox(label, key=_key(name))
    st.session_state["ans"][name] = v
    return v


def _yes_no(v):
    return "Yes" if v else "No"


def _sex_key():
    return st.session_state["ans"].get("sex")


def _show_errors():
    errs = st.session_state.get("errors") or []
    if errs:
        st.error("**Please fix the following to continue**\n\n" + "\n".join(f"- {e}" for e in errs))


def _nav(section: str, continue_label: str = "Continue"):
    """Back / Continue row (or Back to review / Save changes when editing from Review)."""
    editing = st.session_state["edit_return"]
    c1, c2 = st.columns(2, gap="medium")
    if section != "profile" or editing:
        c1.button("← Back to review" if editing else "← Back", key=f"back_{section}", type="secondary",
                  on_click=U.cb_back, args=(section,))
    c2.button("Save changes → Review" if editing else continue_label, key=f"cont_{section}",
              type="primary", on_click=U.cb_continue, args=(section,))


# ---------------------------------------------------------------------------
# Landing
# ---------------------------------------------------------------------------
def view_landing():
    md(C.hero_html())
    c1, _ = st.columns([1, 2], gap="medium")
    c1.button("Start Screening →", key="start_top", type="primary", on_click=U.cb_start)

    md(C.section_title("How it works"))
    md('<div class="ed-grid-3">'
       + C.how_step(1, "Tell us about yourself", "Answer a few questions about your profile and lifestyle.")
       + C.how_step(2, "Complete your screening", "Provide information about your everyday habits and health-related factors.")
       + C.how_step(3, "Get your result", "Receive a personalised screening estimate and general healthy-habit guidance.")
       + '</div>')
    c1, _ = st.columns([1, 2], gap="medium")
    c1.button("Start Screening →", key="start_bottom", type="primary", on_click=U.cb_start)
    st.caption("EarlyDia is a screening tool, not a diagnostic test.")


# ---------------------------------------------------------------------------
# Step 1 - Profile
# ---------------------------------------------------------------------------
def view_profile():
    md(C.page_head("Step 1 of 4 · Profile", "Tell us a little about you",
                   "Four quick questions. Women and men are screened with separate models, so we need your "
                   "sex first."))
    _show_errors()
    with st.container(border=True):
        md(C.section_title("About you", "user"))
        a, b = st.columns(2, gap="large")
        with a:
            w_radio("sex", "Biological sex", ["women", "men"], fmt=lambda k: SEX_OPTIONS[k],
                    help="EarlyDia uses a separate screening model for women and for men.")
        with b:
            w_number("age", f"Age ({AGE_RANGE[0]}-{AGE_RANGE[1]})", AGE_RANGE[0], AGE_RANGE[1], 1,
                     placeholder="e.g. 29", help="EarlyDia is designed for adults aged 18 to 40.")
    with st.container(border=True):
        md(C.section_title("Background", "users"))
        a, b = st.columns(2, gap="large")
        with a:
            w_select("education", "Highest education completed", list(EDUCATION_OPTIONS),
                     fmt=lambda k: EDUCATION_OPTIONS[k])
        with b:
            w_radio("residence", "Where do you live?", list(RESIDENCE_OPTIONS),
                    fmt=lambda k: RESIDENCE_OPTIONS[k])
    _nav("profile", "Continue to Lifestyle →")


# ---------------------------------------------------------------------------
# Step 2 - Lifestyle
# ---------------------------------------------------------------------------
def view_lifestyle():
    sex = _sex_key()
    ans = st.session_state["ans"]
    md(C.page_head("Step 2 of 4 · Lifestyle", "Your everyday habits",
                   "Tobacco, alcohol and food habits. Answer for how things are now."))
    _show_errors()

    with st.container(border=True):
        md(C.section_title("Tobacco", "wind"))
        uses = w_radio("uses_tobacco", "Do you currently use any form of tobacco?", [True, False], fmt=_yes_no)
        if uses:
            st.markdown("**Which products do you use?** Select all that apply.")
            cols = st.columns(2, gap="large")
            for i, (col, label) in enumerate(TOBACCO_ITEMS[sex]):
                with cols[i % 2]:
                    w_check(col, label)
            if TOBACCO_FREQ_COL[sex] and ans.get(CIGARETTE_COL[sex]):
                w_radio("tob_freq", "How often do you smoke cigarettes?", list(TOBACCO_FREQ_OPTIONS),
                        fmt=lambda k: TOBACCO_FREQ_OPTIONS[k])

    with st.container(border=True):
        md(C.section_title("Alcohol", "coffee"))
        drinks = w_radio("drinks", "Do you drink alcohol?", [True, False], fmt=_yes_no)
        if drinks:
            a, b = st.columns(2, gap="large")
            with a:
                w_radio("alc_freq", "How often do you drink?", list(ALCOHOL_FREQ_OPTIONS),
                        fmt=lambda k: ALCOHOL_FREQ_OPTIONS[k], horizontal=False)
            with b:
                max_years = float(ans["age"]) if ans.get("age") else 40.0
                w_number("alc_duration", "For how many years have you been drinking?", 0.0, max_years, 1.0,
                         placeholder="e.g. 4", fmt="%.0f")
            st.markdown("**Which types do you drink?** Select all that apply.")
            cols = st.columns(3, gap="large")
            for i, (col, label) in enumerate(ALCOHOL_TYPE_ITEMS[sex]):
                with cols[i % 3]:
                    w_check(col, label)

    with st.container(border=True):
        md(C.section_title("Food habits", "apple"))
        st.caption("How often do you usually eat or drink each of these?")
        for col, _ in DIET_ITEMS[sex]:
            w_radio(col, DIET_QUESTIONS[diet_letter(col)], DIET_DISPLAY_ORDER, fmt=lambda k: DIET_OPTIONS[k])
        answered = sum(1 for col, _ in DIET_ITEMS[sex] if ans.get(col) is not None)
        st.caption(f"{answered} of 9 answered")
    _nav("lifestyle", "Continue to Health →")


# ---------------------------------------------------------------------------
# Step 3 - Health
# ---------------------------------------------------------------------------
def view_health():
    sex = _sex_key()
    ans = st.session_state["ans"]
    md(C.page_head("Step 3 of 4 · Health", "Body measurements and blood pressure",
                   "Blood pressure needs a real measurement - it can't be answered from how you feel."))
    _show_errors()

    with st.container(border=True):
        md(C.section_title("Height, weight and BMI" if sex == "women" else "Body measurements", "scale"))
        if sex == "women":
            a, b = st.columns(2, gap="large")
            with a:
                w_number("height_cm", "Height (cm)", HEIGHT_RANGE_CM[0], HEIGHT_RANGE_CM[1], 1.0,
                         placeholder="e.g. 160", fmt="%.0f")
            with b:
                w_number("weight_kg", "Weight (kg)", WEIGHT_RANGE_KG[0], WEIGHT_RANGE_KG[1], 0.5,
                         placeholder="e.g. 58", fmt="%.1f")
            bmi = compute_bmi(ans.get("height_cm"), ans.get("weight_kg"))
            if bmi is not None:
                md(f'<div class="ed-kpi"><div class="l">Your BMI</div><div class="v">Your BMI: {bmi:.1f}'
                   f' <span class="ed-muted" style="font-weight:600;font-size:.9rem">kg/m² · {C._e(bmi_band(bmi))}</span></div></div>')
            else:
                st.caption("Enter your height and weight - your BMI is calculated for you.")
        else:
            md(C.note("<b>No height or weight needed.</b> BMI is not used for the men's screening group, so there is "
                      "nothing to enter here."))

    with st.container(border=True):
        md(C.section_title("Blood pressure", "pulse"))
        st.markdown("Use a recent reading from a home monitor, clinic or pharmacy. If you measured more than "
                    "once, enter the **average**.")
        none = w_check("bp_unavailable", "I don't have a recent blood pressure measurement")
        if none:
            md(C.callout("No problem - blood pressure will be treated as missing and the model fills it in the "
                         "standard way it learned during training. The result may be less personalised.", False))
        else:
            a, b = st.columns(2, gap="large")
            with a:
                w_number("bp_sys", "Systolic (top number, mmHg)", SYS_RANGE[0], SYS_RANGE[1], 1.0,
                         placeholder="e.g. 118", fmt="%.0f")
            with b:
                w_number("bp_dia", "Diastolic (bottom number, mmHg)", DIA_RANGE[0], DIA_RANGE[1], 1.0,
                         placeholder="e.g. 76", fmt="%.0f")
            w_radio("bp_n", "How many readings is this average based on?", [1, 2, 3],
                    fmt=lambda k: f"{k} reading{'s' if k > 1 else ''}")
    _nav("health", "Continue to Review →")


# ---------------------------------------------------------------------------
# Step 4 - Review
# ---------------------------------------------------------------------------
def _review_card(title, icon_name, pairs, section, full=()):
    with st.container(border=True):
        a, b = st.columns([4, 1], gap="small")
        a.markdown(C.section_title(title, icon_name), unsafe_allow_html=True)
        b.button("Edit", key=f"edit_{section}", type="secondary", on_click=U.cb_edit, args=(section,))
        md(C.kv_grid(pairs, full))


def view_review():
    ans = st.session_state["ans"]
    md(C.page_head("Step 4 of 4 · Review", "Check your answers",
                   "Use Edit to change anything - the rest of your answers are kept."))
    _show_errors()
    errs_present = bool(U.validate_all(ans))
    if errs_present:
        st.warning("Some answers are missing. Use the Edit buttons to complete them.")
        _nav_review_back_only()
        return
    sex = ans["sex"]
    ui = ui_from_answers(ans)

    _review_card("Profile", "user", [
        ("Sex", S.sex_label(sex)), ("Age", f"{ui['age']} years"),
        ("Education", S.education_label(ui["education"])), ("Residence", S.residence_label(ui["residence"]))],
        "profile")
    _review_card("Lifestyle", "apple", [
        ("Tobacco", S.tobacco_summary(sex, ui)), ("Alcohol", S.alcohol_summary(sex, ui)),
        ("Diet", S.diet_counts_summary(sex, ui))], "lifestyle", full=("Tobacco", "Alcohol", "Diet"))
    bmi_txt = (f"{ui['bmi']:.1f} kg/m² (height {ui['height_cm']:.0f} cm, weight {ui['weight_kg']:.1f} kg)"
               if sex == "women" else "Not used for this screening group")
    _review_card("Health", "pulse", [("BMI", bmi_txt), ("Blood pressure", S.bp_summary(ui))],
                 "health", full=("BMI", "Blood pressure"))

    md(C.note("Your answers are used only to compute this one screening in your current session."))
    c1, c2 = st.columns(2, gap="medium")
    c1.button("← Back", key="back_review", type="secondary", on_click=U.goto, args=("health",))
    c2.button("Run EarlyDia Screening →", key="run", type="primary", on_click=U.cb_run_screening)


def _nav_review_back_only():
    st.button("← Back", key="back_review_only", type="secondary", on_click=U.goto, args=("health",))


# ---------------------------------------------------------------------------
# Step 5 - Result
# ---------------------------------------------------------------------------
def _empty_state(message: str):
    md(C.card("Nothing to show yet", message, "info"))
    st.button("Go to the review step", key="empty_review", type="primary", on_click=U.goto, args=("review",))


def view_result():
    res, rep, ui = st.session_state["result"], st.session_state["report"], st.session_state["snapshot"]
    if res is None or rep is None:
        md(C.page_head("Result", "No screening result yet"))
        _empty_state("Complete the steps and run the screening to see your result.")
        return
    flagged = res.predicted_class == 1
    prob_pct, th_pct = res.calibrated_probability * 100, res.threshold_used * 100
    sex = res.sex
    group = "Women" if sex == "women" else "Men"
    dark = st.session_state["dark_mode"]

    md(C.page_head("Result", "Your EarlyDia Result"))
    gauge = render_semicircular_gauge(prob_pct, th_pct, flagged, dark)
    md(C.result_card(prob_pct, th_pct, flagged, gauge))
    st.caption("This is the model's estimated probability for the diabetes-range screening target, based on the "
               "information you provided.")

    if flagged:
        md(C.callout("<b>Important:</b> A flagged result does not mean that you have diabetes. EarlyDia is a screening "
                     "tool, not a diagnostic test. Consider discussing your result with a qualified healthcare "
                     "professional and asking whether clinical testing is appropriate.", True, html=True))
    else:
        md(C.callout("<b>Important:</b> A result below the screening threshold does not rule out diabetes and does "
                     "not replace clinical testing.", False, html=True))

    md('<div class="ed-grid-2">'
       + C.card("What does this percentage mean?",
                f"Your result of {prob_pct:.2f}% is a model-estimated screening probability based on the information "
                "you provided. It is used to determine whether your responses fall above or below the screening "
                "threshold. It is not a blood-sugar measurement, a diagnosis, or a prediction of whether you will "
                "develop diabetes in the future.", "info")
       + C.card("Why is there a threshold?",
                f"EarlyDia compares your estimated screening probability with a predefined screening threshold. For "
                f"this screening, the threshold is {th_pct:.2f}%. Results at or above the threshold are classified as "
                "Flagged for Follow-up. Results below it are classified as Below Screening Threshold. The threshold is "
                "set low on purpose so that most diabetes-range cases are caught, which means many flagged results "
                "will not turn out to be diabetes.", "sliders")
       + '</div>')

    md(C.section_title("Factors considered", "clipboard"))
    md(C.factor_grid(rep.factors_considered))
    md('<div class="ed-card" style="margin-top:.4rem"><h3 style="margin-top:0!important">Diet: your responses</h3>'
       + C.diet_table(S.diet_answers(sex, ui)) + '</div>')
    st.caption("These are the factors the screening considered when producing your result. They show what "
               "information was used - not what caused the result.")

    if res.validation_warnings and ui.get("n_sys", 0) == 0:
        md(C.note("<b>No blood-pressure measurement was provided,</b> so that part was filled in using the "
                  "standard handling of missing values. The result may be less personalised."))

    md(C.details_html("Technical details",
        '<ul>'
        f'<li><b>Screening group:</b> {group}</li>'
        '<li><b>Model:</b> HistGradientBoosting, with a separate model for women and for men.</li>'
        f'<li><b>Threshold:</b> {th_pct:.2f}% for {group.lower()}. It is the highest value that still flags at least '
        '75% of diabetes-range cases in cross-validated training predictions.</li>'
        '<li><b>Gauge scale:</b> the dial runs from 0% to 100% on a non-linear scale that spreads out low '
        'probabilities so the needle and threshold can be told apart. It does not show accuracy or certainty.</li>'
        '<li><b>What the target means:</b> a reported diabetes diagnosis, or a single random blood-glucose reading of '
        '200 mg/dL or higher - not a fasting or laboratory diagnosis. Random readings vary with meals and timing.</li>'
        '<li><b>Limits:</b> designed for ages 18-40; it shows statistical associations, not causes; it has not been '
        'validated clinically; BMI is not used for men.</li>'
        '</ul>'))

    md(C.section_title("Want guidance based on your responses?", "heart"))
    st.button("View Personalized Recommendations →", key="go_recs", type="primary", on_click=U.cb_view_recs)
    c2, c3 = st.columns(2, gap="medium")
    c2.button("Edit my answers", key="edit_from_result", type="secondary", on_click=U.cb_modify_from_result)
    c3.button("Start New Screening", key="new_from_result", type="secondary", on_click=U.cb_new_screening)
    md(C.note(DISCLAIMER))


# ---------------------------------------------------------------------------
# Step 6 - Recommendations
# ---------------------------------------------------------------------------
def view_recs():
    res, rep = st.session_state["result"], st.session_state["report"]
    if res is None or rep is None:
        md(C.page_head("Guidance", "No screening result yet"))
        _empty_state("Run the screening first - guidance is based on your answers and result.")
        return
    flagged = rep.is_flagged
    md(C.page_head("Guidance", "Personalized recommendations",
                   "General healthy-habit guidance based on your answers. It is not medical advice and does not "
                   "depend on any single factor causing your result."))
    state = "flag" if flagged else "ok"
    cls = "Flagged for Follow-up" if flagged else "Below Screening Threshold"
    md(f'<div class="ed-card" style="margin-bottom:1rem"><span class="ed-badge {state}" style="margin:0">'
       f'{C.icon("alert" if flagged else "check")}Result: {cls}</span>'
       f'<p style="margin-top:.6rem">Estimated screening probability {rep.calibrated_probability*100:.2f}% · '
       f'screening threshold {rep.threshold_used*100:.2f}%</p></div>')

    md(C.section_title("Clinical next steps", "stethoscope"))
    md(C.rec_card(rep.clinical_next_steps))
    md(C.section_title("Based on your responses", "sliders"))
    for item in rep.lifestyle_recommendations:
        md(C.rec_card(item))
    md(C.section_title("General health", "heart"))
    for item in rep.general_wellness:
        md(C.rec_card(item))
    md(C.note("<b>About this guidance.</b> These are general, widely recommended healthy-habit suggestions. They are "
              "not a diagnosis, treatment plan or prescription, cannot promise that any condition will be prevented "
              "or cured, and are no substitute for advice from a qualified healthcare professional."))
    c1, c2 = st.columns(2, gap="medium")
    c1.button("← Back to result", key="back_to_result", type="secondary", on_click=U.cb_back_to_result)
    c2.button("Start New Screening", key="new_from_recs", type="primary", on_click=U.cb_new_screening)


VIEWS = {"landing": view_landing, "profile": view_profile, "lifestyle": view_lifestyle,
         "health": view_health, "review": view_review, "result": view_result, "recs": view_recs}
