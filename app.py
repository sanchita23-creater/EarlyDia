"""
app.py
------
EarlyDia — Early Diabetes Risk Screening Through Non-Invasive Insights
Phase 2: Complete, polished, multi-step Streamlit web application.
Connected directly to the saved Phase 1 Random Forest pipeline (`models/selected_model.joblib`).
"""

import time
import os
import sys

# Ensure src/ is discoverable
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import streamlit as st
import pandas as pd
import joblib

# Import Phase 2 helpers
from src.feature_mapping import (
    calculate_bmi,
    map_age_to_bracket,
    build_model_input_dataframe,
    GEN_HLTH_OPTIONS,
    GEN_HLTH_DESCRIPTIONS,
    AGE_BRACKETS
)
from src.insights import generate_profile_insights
from src.ui_helpers import (
    apply_theme,
    render_top_navigation,
    render_stepper,
    render_medical_disclaimer,
    create_risk_gauge,
    render_model_performance_section
)


# ==============================================================================
# STREAMLIT CONFIG & STATE INITIALIZATION
# ==============================================================================

st.set_page_config(
    page_title="EarlyDia — Early Diabetes Risk Screening",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Initialize Session State Variables
if "theme" not in st.session_state:
    st.session_state.theme = "light"

if "current_page" not in st.session_state:
    st.session_state.current_page = 1

if "show_model_info" not in st.session_state:
    st.session_state.show_model_info = False

# Default Form State
if "profile_data" not in st.session_state:
    st.session_state.profile_data = {
        # Personal Profile (Step 1)
        "age_years": 32,
        "sex": "Female",
        "height_cm": 168.0,
        "weight_kg": 65.0,
        
        # Lifestyle Profile (Step 2)
        "phys_activity": "Yes",
        "fruits": "Yes",
        "veggies": "Yes",
        "smoker": "No",
        "hvy_alcohol": "No",
        
        # Health Profile (Step 3)
        "gen_hlth": "Very Good",
        "ment_hlth_days": 2,
        "phys_hlth_days": 1,
        "high_bp": "No",
        "high_chol": "No",
        "stroke": "No",
        "heart_disease": "No",
        "diff_walk": "No"
    }

if "assessment_result" not in st.session_state:
    st.session_state.assessment_result = None


# Apply custom styling based on active theme
apply_theme(st.session_state.theme)


# ==============================================================================
# MODEL LOADER (CACHED)
# ==============================================================================

@st.cache_resource(show_spinner=False)
def get_cached_model():
    """
    Loads and caches the Phase 1 serialized Random Forest pipeline.
    Avoids reloading on every interaction.
    """
    model_path = os.path.join(current_dir, "models", "selected_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Selected model not found at {model_path}")
    return joblib.load(model_path)


# ==============================================================================
# NAVIGATION CONTROLLER HELPERS
# ==============================================================================

def goto_page(page_num: int):
    st.session_state.current_page = page_num
    st.rerun()


def reset_assessment():
    """Resets assessment state and returns to landing page."""
    st.session_state.current_page = 1
    st.session_state.assessment_result = None
    st.session_state.show_model_info = False
    st.rerun()


# Render Top Branding & Navigation Bar
render_top_navigation(st.session_state.theme)

# Conditional Model Info Drawer (accessible anytime)
if st.session_state.show_model_info:
    render_model_performance_section(st.session_state.theme)
    if st.button("✕ Close Model Information", key="close_model_info_btn"):
        st.session_state.show_model_info = False
        st.rerun()
    st.markdown("<hr style='border:0; border-top:1px solid var(--border-color); margin: 1.5rem 0;'>", unsafe_allow_html=True)


# ==============================================================================
# PAGE 1: LANDING PAGE
# ==============================================================================

if st.session_state.current_page == 1:
    # Hero Section
    st.markdown(
        """
        <div class="earlydia-hero">
            <div class="earlydia-badge">NON-INVASIVE RISK SCREENING PROTOTYPE</div>
            <h1 style="font-size: 2.5rem; margin-top: 0.4rem; margin-bottom: 0.6rem;">EarlyDia</h1>
            <h3 style="color: var(--accent-blue) !important; font-weight: 600; margin-top: 0; margin-bottom: 1.2rem;">
                Early Diabetes Risk Screening Through Non-Invasive Insights
            </h3>
            <p style="font-size: 1.05rem; line-height: 1.6; max-width: 680px; margin-bottom: 1.6rem;">
                EarlyDia uses Machine Learning and accessible health and lifestyle information to provide 
                an early diabetes risk screening indication without relying on diabetes-specific laboratory tests.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Big Start Assessment Button
    col_cta1, col_cta2, col_cta3 = st.columns([1, 2, 1])
    with col_cta2:
        if st.button("🚀 START ASSESSMENT", key="landing_start_btn"):
            goto_page(2)

    st.markdown("<br>", unsafe_allow_html=True)

    # 3 Value Proposition Feature Cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="earlydia-card" style="height: 100%;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">📋</div>
                <h4 style="margin-bottom: 0.5rem;">Non-Invasive Inputs</h4>
                <p style="font-size: 0.88rem; line-height: 1.5;">
                    Uses accessible information that users can report about themselves—zero blood draws, needles, or fasting tests required.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div class="earlydia-card" style="height: 100%;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🤖</div>
                <h4 style="margin-bottom: 0.5rem;">ML-Powered Analysis</h4>
                <p style="font-size: 0.88rem; line-height: 1.5;">
                    Uses a trained Machine Learning model (Random Forest Pipeline) evaluated on 50,736 held-out instances to identify risk patterns.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            """
            <div class="earlydia-card" style="height: 100%;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">💡</div>
                <h4 style="margin-bottom: 0.5rem;">Early Risk Insights</h4>
                <p style="font-size: 0.88rem; line-height: 1.5;">
                    Provides a risk-oriented screening indication to support awareness, encourage timely health checks, and highlight habits for attention.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Expandable Model Information Preview
    with st.expander("📊 View Model Performance & Phase 1 Validation (LR vs RF)"):
        render_model_performance_section(st.session_state.theme)

    render_medical_disclaimer()


# ==============================================================================
# PAGE 2: PERSONAL PROFILE (STEP 1 OF 4)
# ==============================================================================

elif st.session_state.current_page == 2:
    render_stepper(step=1, total_steps=4, step_name="Personal Profile")

    st.markdown(
        """
        <div class="earlydia-card">
            <h3 style="margin-top: 0; margin-bottom: 0.25rem;">Demographic & Anthropometric Indicators</h3>
            <p style="font-size: 0.88rem; margin-bottom: 0;">
                Your height and weight are used exclusively to automatically compute your Body Mass Index (BMI).
                Height and weight themselves are not model features.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        # Age Input
        user_age = st.number_input(
            "Age (in years):",
            min_value=18,
            max_value=120,
            value=int(st.session_state.profile_data.get("age_years", 32)),
            step=1,
            help="Enter your chronological age in years (18 years or older for adult screening)."
        )
        st.session_state.profile_data["age_years"] = user_age
        
        # Live display of corresponding CDC BRFSS bracket
        bracket_code, bracket_label = map_age_to_bracket(user_age)
        st.markdown(
            f"""
            <div style="font-size: 0.82rem; color: var(--text-muted); margin-top: -0.5rem; margin-bottom: 1rem;">
                CDC BRFSS Bracket: <strong>{bracket_label}</strong> (Category code: {bracket_code})
            </div>
            """,
            unsafe_allow_html=True
        )

        # Biological Sex Input
        sex_options = ["Female", "Male"]
        cur_sex = st.session_state.profile_data.get("sex", "Female")
        sex_idx = sex_options.index(cur_sex) if cur_sex in sex_options else 0
        user_sex = st.radio(
            "Biological Sex:",
            options=sex_options,
            index=sex_idx,
            horizontal=True,
            help="Biological sex influences metabolic fat distribution and hormone pathways."
        )
        st.session_state.profile_data["sex"] = user_sex

    with col2:
        # Height Input (cm)
        user_height = st.number_input(
            "Height (centimetres):",
            min_value=100.0,
            max_value=250.0,
            value=float(st.session_state.profile_data.get("height_cm", 168.0)),
            step=0.5,
            help="Enter height in centimetres (e.g. 170 cm = 5 ft 7 in)."
        )
        st.session_state.profile_data["height_cm"] = user_height

        # Weight Input (kg)
        user_weight = st.number_input(
            "Weight (kilograms):",
            min_value=30.0,
            max_value=250.0,
            value=float(st.session_state.profile_data.get("weight_kg", 65.0)),
            step=0.5,
            help="Enter weight in kilograms (e.g. 70 kg = 154 lbs)."
        )
        st.session_state.profile_data["weight_kg"] = user_weight

    # Live BMI Calculation Display
    bmi_val, bmi_cat, bmi_color = calculate_bmi(user_height, user_weight)
    
    st.markdown(
        f"""
        <div class="earlydia-card" style="margin-top: 1rem; text-align: center; border-left: 5px solid {bmi_color};">
            <div style="font-size: 0.82rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em;">
                YOUR CALCULATED BODY MASS INDEX (BMI)
            </div>
            <div style="font-size: 2.6rem; font-weight: 800; color: {bmi_color}; margin: 0.2rem 0;">
                {bmi_val:.1f} <span style="font-size: 1rem; font-weight: 600; color: var(--text-secondary);">kg/m²</span>
            </div>
            <div style="display: inline-block; background: {bmi_color}22; color: {bmi_color}; font-weight: 700; padding: 0.25rem 0.9rem; border-radius: 9999px; font-size: 0.88rem;">
                Classification: {bmi_cat}
            </div>
            <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 0.5rem;">
                Computed live: {user_weight:.1f} kg / ({user_height/100:.2f} m)² — Passed directly to ML model as BMI feature.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Navigation Buttons
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("← Back to Overview", key="step1_back_btn"):
            goto_page(1)
    with col_nav2:
        if st.button("Continue to Lifestyle Profile →", key="step1_next_btn"):
            goto_page(3)


# ==============================================================================
# PAGE 3: LIFESTYLE PROFILE (STEP 2 OF 4)
# ==============================================================================

elif st.session_state.current_page == 3:
    render_stepper(step=2, total_steps=4, step_name="Lifestyle Profile")

    st.markdown(
        """
        <div class="earlydia-card">
            <h3 style="margin-top: 0; margin-bottom: 0.25rem;">Daily Habits & Lifestyle Factors</h3>
            <p style="font-size: 0.88rem; margin-bottom: 0;">
                Questions and definitions match the CDC Behavioral Risk Factor Surveillance System (BRFSS) 
                and Phase 1 model feature specifications exactly.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 1. Physical Activity
    st.markdown(
        """
        <div class="earlydia-card-interactive">
            <div style="display: flex; gap: 0.6rem; align-items: center;">
                <span style="font-size: 1.4rem;">🏃</span>
                <div>
                    <strong>Physical Activity (PhysActivity)</strong>
                    <div style="font-size: 0.84rem; color: var(--text-secondary);">
                        Did you participate in any physical activity or exercise during the past 30 days other than your regular job?
                        <em>(Includes brisk walking, running, sports, gardening, or intentional workouts)</em>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    cur_pa = st.session_state.profile_data.get("phys_activity", "Yes")
    pa_choice = st.radio(
        "Physical Activity in past 30 days outside work:",
        options=["Yes", "No"],
        index=0 if cur_pa == "Yes" else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="radio_pa"
    )
    st.session_state.profile_data["phys_activity"] = pa_choice

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Fruit Consumption
    st.markdown(
        """
        <div class="earlydia-card-interactive">
            <div style="display: flex; gap: 0.6rem; align-items: center;">
                <span style="font-size: 1.4rem;">🍎</span>
                <div>
                    <strong>Fruit Intake (Fruits)</strong>
                    <div style="font-size: 0.84rem; color: var(--text-secondary);">
                        Do you consume fruit 1 or more times per day?
                        <em>(Includes fresh, frozen, or canned whole fruits)</em>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    cur_fr = st.session_state.profile_data.get("fruits", "Yes")
    fr_choice = st.radio(
        "Fruit 1 or more times per day:",
        options=["Yes", "No"],
        index=0 if cur_fr == "Yes" else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="radio_fr"
    )
    st.session_state.profile_data["fruits"] = fr_choice

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Vegetable Consumption
    st.markdown(
        """
        <div class="earlydia-card-interactive">
            <div style="display: flex; gap: 0.6rem; align-items: center;">
                <span style="font-size: 1.4rem;">🥦</span>
                <div>
                    <strong>Vegetable Intake (Veggies)</strong>
                    <div style="font-size: 0.84rem; color: var(--text-secondary);">
                        Do you consume vegetables 1 or more times per day?
                        <em>(Includes raw salads, cooked vegetables, and legumes)</em>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    cur_vg = st.session_state.profile_data.get("veggies", "Yes")
    vg_choice = st.radio(
        "Vegetables 1 or more times per day:",
        options=["Yes", "No"],
        index=0 if cur_vg == "Yes" else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="radio_vg"
    )
    st.session_state.profile_data["veggies"] = vg_choice

    st.markdown("<br>", unsafe_allow_html=True)

    # 4. Tobacco Smoking Exposure
    st.markdown(
        """
        <div class="earlydia-card-interactive">
            <div style="display: flex; gap: 0.6rem; align-items: center;">
                <span style="font-size: 1.4rem;">🚭</span>
                <div>
                    <strong>Lifetime Tobacco Smoking History (Smoker)</strong>
                    <div style="font-size: 0.84rem; color: var(--text-secondary);">
                        Have you smoked at least 100 cigarettes in your entire life?
                        <em>[Standard CDC Benchmark Note: 5 packs = 100 cigarettes]</em>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    cur_sm = st.session_state.profile_data.get("smoker", "No")
    sm_choice = st.radio(
        "Smoked >= 100 cigarettes in entire lifetime:",
        options=["No", "Yes"],
        index=0 if cur_sm == "No" else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="radio_sm"
    )
    st.session_state.profile_data["smoker"] = sm_choice

    st.markdown("<br>", unsafe_allow_html=True)

    # 5. Heavy Alcohol Consumption
    st.markdown(
        """
        <div class="earlydia-card-interactive">
            <div style="display: flex; gap: 0.6rem; align-items: center;">
                <span style="font-size: 1.4rem;">🍷</span>
                <div>
                    <strong>Heavy Alcohol Consumption (HvyAlcoholConsump)</strong>
                    <div style="font-size: 0.84rem; color: var(--text-secondary);">
                        Do you meet the CDC definition for heavy drinking?
                        <em>(Adult men having more than 14 alcoholic drinks per week; adult women having more than 7 drinks per week)</em>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    cur_al = st.session_state.profile_data.get("hvy_alcohol", "No")
    al_choice = st.radio(
        "Heavy alcohol consumption (>14 drinks/wk men, >7 women):",
        options=["No", "Yes"],
        index=0 if cur_al == "No" else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="radio_al"
    )
    st.session_state.profile_data["hvy_alcohol"] = al_choice

    # Navigation Buttons
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("← Back to Personal Profile", key="step2_back_btn"):
            goto_page(2)
    with col_nav2:
        if st.button("Continue to Health Profile →", key="step2_next_btn"):
            goto_page(4)


# ==============================================================================
# PAGE 4: HEALTH PROFILE (STEP 3 OF 4)
# ==============================================================================

elif st.session_state.current_page == 4:
    render_stepper(step=3, total_steps=4, step_name="Health Profile")

    st.markdown(
        """
        <div class="earlydia-card">
            <h3 style="margin-top: 0; margin-bottom: 0.25rem;">General Well-Being & Medical History</h3>
            <p style="font-size: 0.88rem; margin-bottom: 0;">
                All questions are self-reportable. <strong>Zero clinical lab tests</strong> (no blood draws, fasting glucose, or HbA1c tests) are required.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 1. General Health
    st.markdown("#### 1. Self-Assessed Overall Health (GenHlth)")
    st.caption("Would you say that in general your health is:")
    gen_options = list(GEN_HLTH_OPTIONS.keys())
    cur_gen = st.session_state.profile_data.get("gen_hlth", "Very Good")
    gen_idx = gen_options.index(cur_gen) if cur_gen in gen_options else 1
    
    selected_gen = st.select_slider(
        "General health status:",
        options=gen_options,
        value=cur_gen,
        key="slider_gen_hlth",
        label_visibility="collapsed"
    )
    st.session_state.profile_data["gen_hlth"] = selected_gen
    
    # Description callout for selected health rating
    st.markdown(
        f"""
        <div style="font-size: 0.83rem; background: var(--bg-card-sec); border-radius: 8px; padding: 0.5rem 0.8rem; margin-top: -0.5rem; margin-bottom: 1.2rem;">
            <strong>{selected_gen} Health:</strong> {GEN_HLTH_DESCRIPTIONS[selected_gen]} (Scale: {GEN_HLTH_OPTIONS[selected_gen]} of 5)
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2. Mental and Physical Health Days
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.markdown("#### 2. Mental Well-Being (MentHlth)")
        st.caption("Days during past 30 days when mental health (stress, depression, emotions) was not good (0–30 days):")
        ment_days = st.slider(
            "Days poor mental health:",
            min_value=0,
            max_value=30,
            value=int(st.session_state.profile_data.get("ment_hlth_days", 2)),
            step=1,
            label_visibility="collapsed",
            key="slider_ment"
        )
        st.session_state.profile_data["ment_hlth_days"] = ment_days
        st.markdown(f"<div style='font-size:0.85rem; font-weight:700; color:var(--accent-blue); text-align:right;'>{ment_days} days reported</div>", unsafe_allow_html=True)

    with col_h2:
        st.markdown("#### 3. Physical Illness Burden (PhysHlth)")
        st.caption("Days during past 30 days when physical health (illness and injury) was not good (0–30 days):")
        phys_days = st.slider(
            "Days poor physical health:",
            min_value=0,
            max_value=30,
            value=int(st.session_state.profile_data.get("phys_hlth_days", 1)),
            step=1,
            label_visibility="collapsed",
            key="slider_phys"
        )
        st.session_state.profile_data["phys_hlth_days"] = phys_days
        st.markdown(f"<div style='font-size:0.85rem; font-weight:700; color:var(--accent-blue); text-align:right;'>{phys_days} days reported</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border:0; border-top:1px solid var(--border-color); margin: 1.2rem 0;'>", unsafe_allow_html=True)
    st.markdown("#### 4. Medical History & Mobility Questions")
    st.caption("Have you ever been told by a doctor, nurse, or health professional that you had any of the following?")

    # Medical History Questions
    mh_col1, mh_col2 = st.columns(2)

    with mh_col1:
        # High Blood Pressure
        cur_hbp = st.session_state.profile_data.get("high_bp", "No")
        hbp_choice = st.radio(
            "🩺 High Blood Pressure (HighBP):",
            options=["No", "Yes"],
            index=0 if cur_hbp == "No" else 1,
            horizontal=True,
            help="Ever told by a health professional that your blood pressure is high?"
        )
        st.session_state.profile_data["high_bp"] = hbp_choice

        # History of Stroke
        cur_str = st.session_state.profile_data.get("stroke", "No")
        str_choice = st.radio(
            "🧠 History of Stroke (Stroke):",
            options=["No", "Yes"],
            index=0 if cur_str == "No" else 1,
            horizontal=True,
            help="Ever told by a health professional that you had a stroke?"
        )
        st.session_state.profile_data["stroke"] = str_choice

        # Serious Difficulty Walking
        cur_dw = st.session_state.profile_data.get("diff_walk", "No")
        dw_choice = st.radio(
            "🚶 Serious Difficulty Walking or Stairs (DiffWalk):",
            options=["No", "Yes"],
            index=0 if cur_dw == "No" else 1,
            horizontal=True,
            help="Do you have serious difficulty walking or climbing stairs?"
        )
        st.session_state.profile_data["diff_walk"] = dw_choice

    with mh_col2:
        # High Cholesterol
        cur_hch = st.session_state.profile_data.get("high_chol", "No")
        hch_choice = st.radio(
            "🩸 High Blood Cholesterol (HighChol):",
            options=["No", "Yes"],
            index=0 if cur_hch == "No" else 1,
            horizontal=True,
            help="Ever told by a health professional that your blood cholesterol is high?"
        )
        st.session_state.profile_data["high_chol"] = hch_choice

        # Heart Disease or Heart Attack
        cur_hd = st.session_state.profile_data.get("heart_disease", "No")
        hd_choice = st.radio(
            "❤️ Heart Disease or Heart Attack (HeartDiseaseorAttack):",
            options=["No", "Yes"],
            index=0 if cur_hd == "No" else 1,
            horizontal=True,
            help="Ever reported coronary heart disease (CHD) or myocardial infarction (heart attack)?"
        )
        st.session_state.profile_data["heart_disease"] = hd_choice

    # Navigation Buttons
    st.markdown("<br>", unsafe_allow_html=True)
    col_nav1, col_nav2 = st.columns(2)
    with col_nav1:
        if st.button("← Back to Lifestyle Profile", key="step3_back_btn"):
            goto_page(3)
    with col_nav2:
        if st.button("Review Your Profile →", key="step3_next_btn"):
            goto_page(5)


# ==============================================================================
# PAGE 5: REVIEW PROFILE (STEP 4 OF 4)
# ==============================================================================

elif st.session_state.current_page == 5:
    render_stepper(step=4, total_steps=4, step_name="Review Your Profile")

    data = st.session_state.profile_data
    h_cm = data.get("height_cm", 168.0)
    w_kg = data.get("weight_kg", 65.0)
    age = data.get("age_years", 32)
    bmi_val, bmi_cat, bmi_col = calculate_bmi(h_cm, w_kg)
    bracket_code, bracket_lbl = map_age_to_bracket(age)

    st.markdown(
        """
        <div class="earlydia-card">
            <h3 style="margin-top: 0; margin-bottom: 0.25rem;">Review Your Self-Reported Indicators</h3>
            <p style="font-size: 0.88rem; margin-bottom: 0;">
                Please verify your information before running the Machine Learning risk screening model.
                You may return to any previous section to adjust your responses.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 3 Summary Cards
    col_r1, col_r2, col_r3 = st.columns(3)

    with col_r1:
        st.markdown(
            f"""
            <div class="earlydia-card" style="height: 100%;">
                <div class="earlydia-badge">SECTION 1</div>
                <h4 style="margin: 0.3rem 0;">Personal Profile</h4>
                <hr style="border:0; border-top:1px solid var(--border-color); margin: 0.6rem 0;">
                <p style="font-size: 0.86rem; line-height: 1.6; margin: 0;">
                    <strong>Age:</strong> {age} years<br>
                    <span style="font-size: 0.78rem; color: var(--text-muted);">(Bracket: {bracket_lbl})</span><br>
                    <strong>Sex:</strong> {data.get('sex', 'Female')}<br>
                    <strong>Height:</strong> {h_cm:.1f} cm<br>
                    <strong>Weight:</strong> {w_kg:.1f} kg<br>
                    <strong>Calculated BMI:</strong> <span style="color:{bmi_col}; font-weight:800;">{bmi_val:.1f}</span> ({bmi_cat})
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("✏️ Edit Personal", key="edit_sec1_btn"):
            goto_page(2)

    with col_r2:
        st.markdown(
            f"""
            <div class="earlydia-card" style="height: 100%;">
                <div class="earlydia-badge">SECTION 2</div>
                <h4 style="margin: 0.3rem 0;">Lifestyle Profile</h4>
                <hr style="border:0; border-top:1px solid var(--border-color); margin: 0.6rem 0;">
                <p style="font-size: 0.86rem; line-height: 1.6; margin: 0;">
                    <strong>Physical Activity:</strong> {data.get('phys_activity', 'Yes')}<br>
                    <strong>Daily Fruits:</strong> {data.get('fruits', 'Yes')}<br>
                    <strong>Daily Veggies:</strong> {data.get('veggies', 'Yes')}<br>
                    <strong>Smoker (100+ cigs):</strong> {data.get('smoker', 'No')}<br>
                    <strong>Heavy Alcohol:</strong> {data.get('hvy_alcohol', 'No')}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("✏️ Edit Lifestyle", key="edit_sec2_btn"):
            goto_page(3)

    with col_r3:
        st.markdown(
            f"""
            <div class="earlydia-card" style="height: 100%;">
                <div class="earlydia-badge">SECTION 3</div>
                <h4 style="margin: 0.3rem 0;">Health Profile</h4>
                <hr style="border:0; border-top:1px solid var(--border-color); margin: 0.6rem 0;">
                <p style="font-size: 0.86rem; line-height: 1.6; margin: 0;">
                    <strong>General Health:</strong> {data.get('gen_hlth', 'Very Good')}<br>
                    <strong>Poor Mental Days:</strong> {data.get('ment_hlth_days', 0)} / 30<br>
                    <strong>Poor Physical Days:</strong> {data.get('phys_hlth_days', 0)} / 30<br>
                    <strong>High Blood Pressure:</strong> {data.get('high_bp', 'No')}<br>
                    <strong>High Cholesterol:</strong> {data.get('high_chol', 'No')}<br>
                    <strong>Stroke History:</strong> {data.get('stroke', 'No')}<br>
                    <strong>Heart Disease/Attack:</strong> {data.get('heart_disease', 'No')}<br>
                    <strong>Difficulty Walking:</strong> {data.get('diff_walk', 'No')}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("✏️ Edit Health", key="edit_sec3_btn"):
            goto_page(4)

    st.markdown("<br>", unsafe_allow_html=True)

    # Large Analyze Button
    col_a1, col_a2, col_a3 = st.columns([1, 2, 1])
    with col_a2:
        if st.button("🔬 ANALYZE MY PROFILE", key="review_analyze_btn"):
            goto_page(6)


# ==============================================================================
# PAGE 6: ML ANALYSIS (PROCESSING SEQUENCE)
# ==============================================================================

elif st.session_state.current_page == 6:
    st.markdown(
        """
        <div class="earlydia-card" style="text-align: center; padding: 2.5rem 1.5rem;">
            <div class="earlydia-badge">MODEL EXECUTION IN PROGRESS</div>
            <h2 style="margin-top: 0.5rem; margin-bottom: 0.5rem;">EarlyDia Machine Learning Engine</h2>
            <p style="font-size: 0.95rem; color: var(--text-secondary); max-width: 520px; margin: 0 auto 1.5rem auto;">
                Evaluating your non-invasive indicators using the trained Phase 1 Random Forest pipeline...
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    status_placeholder = st.empty()
    progress_bar = st.progress(0)

    # Polished execution sequence
    steps = [
        (20, "✓ Profile information received from session state"),
        (45, "✓ Computing BMI and mapping age category (_AGEG5YR)"),
        (70, "✓ Formatting 16 self-reported features strictly in model order"),
        (90, "✓ Querying serialized Random Forest pipeline (`models/selected_model.joblib`)"),
        (100, "✓ Computing ML Risk Score and assigning prototype screening tier")
    ]

    for pct, step_text in steps:
        time.sleep(0.35)
        progress_bar.progress(pct)
        status_placeholder.markdown(
            f"""
            <div style="background: var(--bg-card); border: 1px solid var(--border-color); border-radius: 12px; padding: 1rem 1.5rem; margin-top: 1rem; box-shadow: var(--box-shadow);">
                <span style="color: #10b981; font-weight: 700;">{step_text}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Retrieve data and construct model input features
    data = st.session_state.profile_data
    h_cm = data.get("height_cm", 168.0)
    w_kg = data.get("weight_kg", 65.0)
    user_age = data.get("age_years", 32)
    
    bmi_val, _, _ = calculate_bmi(h_cm, w_kg)
    age_bracket, _ = map_age_to_bracket(user_age)

    # Strict 0/1 encodings matching Phase 1 documentation
    encoded_features = {
        "BMI": float(bmi_val),
        "GenHlth": int(GEN_HLTH_OPTIONS.get(data.get("gen_hlth", "Good"), 3)),
        "MentHlth": int(data.get("ment_hlth_days", 0)),
        "PhysHlth": int(data.get("phys_hlth_days", 0)),
        "Age": int(age_bracket),
        "HighBP": 1 if data.get("high_bp") == "Yes" else 0,
        "HighChol": 1 if data.get("high_chol") == "Yes" else 0,
        "Smoker": 1 if data.get("smoker") == "Yes" else 0,
        "Stroke": 1 if data.get("stroke") == "Yes" else 0,
        "HeartDiseaseorAttack": 1 if data.get("heart_disease") == "Yes" else 0,
        "PhysActivity": 1 if data.get("phys_activity") == "Yes" else 0,
        "Fruits": 1 if data.get("fruits") == "Yes" else 0,
        "Veggies": 1 if data.get("veggies") == "Yes" else 0,
        "HvyAlcoholConsump": 1 if data.get("hvy_alcohol") == "Yes" else 0,
        "DiffWalk": 1 if data.get("diff_walk") == "Yes" else 0,
        "Sex": 1 if data.get("sex") == "Male" else 0
    }

    # Build DataFrame matching exact 16 feature names and order
    input_df = build_model_input_dataframe(encoded_features)

    # Load saved model without retraining
    model = get_cached_model()

    # Predict Class & Probability
    pred_class = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    risk_score = float(probabilities[1])  # Class 1: Prediabetes or Diabetes
    risk_pct = round(risk_score * 100, 1)

    # Map to Prototype Screening Tiers (Strict Phase 1 Rules)
    if risk_pct < 30.0:
        tier_name = "LOW RISK INDICATION"
        tier_class = "tier-card-low"
        tier_icon = "🟢"
        interpretation = (
            "The model identified a lower-risk pattern based on the information provided. "
            "This screening result does not replace regular health screening or professional medical advice."
        )
    elif risk_pct <= 60.0:
        tier_name = "MODERATE / ELEVATED RISK INDICATION"
        tier_class = "tier-card-moderate"
        tier_icon = "🟡"
        interpretation = (
            "The model identified an intermediate risk pattern based on the information provided. "
            "Additional health screening may help provide a clearer assessment."
        )
    else:
        tier_name = "HIGH RISK INDICATION"
        tier_class = "tier-card-high"
        tier_icon = "🔴"
        interpretation = (
            "The model identified a higher-risk pattern based on the information provided. "
            "Consider consulting a healthcare professional for appropriate clinical screening."
        )

    # Generate Rule-Based Transparent Insights
    full_profile_dict = dict(encoded_features)
    full_profile_dict["height_cm"] = h_cm
    full_profile_dict["weight_kg"] = w_kg
    full_profile_dict["age_years"] = user_age
    insights = generate_profile_insights(full_profile_dict)

    # Save to session state
    st.session_state.assessment_result = {
        "risk_score": risk_score,
        "risk_pct": risk_pct,
        "tier_name": tier_name,
        "tier_class": tier_class,
        "tier_icon": tier_icon,
        "interpretation": interpretation,
        "insights": insights,
        "encoded_features": encoded_features
    }

    time.sleep(0.4)
    goto_page(7)


# ==============================================================================
# PAGE 7: RISK ASSESSMENT RESULT
# ==============================================================================

elif st.session_state.current_page == 7:
    result = st.session_state.assessment_result
    if result is None:
        goto_page(1)

    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 1.25rem;">
            <div class="earlydia-badge">ANALYSIS COMPLETE</div>
            <h1 style="font-size: 2.2rem; margin-top: 0.3rem; margin-bottom: 0.2rem;">YOUR EARLYDIA RISK ASSESSMENT</h1>
            <p style="font-size: 0.95rem; color: var(--text-secondary);">
                Preliminary Machine Learning-generated risk indication based on non-invasive indicators
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Plotly Circular Gauge Indicator
    gauge_fig = create_risk_gauge(result["risk_pct"], result["tier_name"], st.session_state.theme)
    st.plotly_chart(gauge_fig, use_container_width=True)

    # Exact phrasing required: "Your ML-generated risk score based on the provided profile is XX%."
    st.markdown(
        f"""
        <div style="text-align: center; margin-top: -0.5rem; margin-bottom: 1.5rem;">
            <div style="font-size: 1.15rem; font-weight: 700; color: var(--text-primary);">
                Your ML-generated risk score based on the provided profile is {result['risk_pct']}%.
            </div>
            <div style="font-size: 0.84rem; color: var(--text-muted); margin-top: 0.25rem;">
                Evaluated against the trained Phase 1 Random Forest decision forest.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Result Tier Card
    st.markdown(
        f"""
        <div class="{result['tier_class']}">
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.6rem;">
                <span style="font-size: 1.8rem;">{result['tier_icon']}</span>
                <div>
                    <div style="font-size: 0.78rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.85;">
                        PROTOTYPE SCREENING TIER
                    </div>
                    <div style="font-size: 1.45rem; font-weight: 800; letter-spacing: -0.01em;">
                        {result['tier_name']}
                    </div>
                </div>
            </div>
            <p style="font-size: 0.95rem; line-height: 1.55; margin-bottom: 0.8rem;">
                {result['interpretation']}
            </p>
            <div style="font-size: 0.8rem; opacity: 0.85; border-top: 1px dashed currentColor; padding-top: 0.5rem;">
                ℹ️ <em>These thresholds are prototype screening thresholds and are not clinical diagnostic cutoffs.</em>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    render_medical_disclaimer()

    st.markdown("<br>", unsafe_allow_html=True)

    # Navigation Buttons
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        if st.button("🔍 View Personalized Insights →", key="res_insights_btn"):
            goto_page(8)
    with col_res2:
        if st.button("🔄 Take a New Assessment", key="res_new_btn"):
            reset_assessment()


# ==============================================================================
# PAGE 8: PERSONALIZED INSIGHTS
# ==============================================================================

elif st.session_state.current_page == 8:
    result = st.session_state.assessment_result
    data = st.session_state.profile_data
    if result is None:
        goto_page(1)

    st.markdown(
        """
        <div style="margin-bottom: 1.25rem;">
            <div class="earlydia-badge">TRANSPARENT PROFILE OBSERVATIONS</div>
            <h1 style="font-size: 2.2rem; margin-top: 0.3rem; margin-bottom: 0.2rem;">YOUR PROFILE AT A GLANCE</h1>
            <p style="font-size: 0.95rem; color: var(--text-secondary);">
                Educational observations and actionable factors to discuss with a healthcare professional
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Quick Biometric Metric Cards
    h_cm = data.get("height_cm", 168.0)
    w_kg = data.get("weight_kg", 65.0)
    user_age = data.get("age_years", 32)
    bmi_val, bmi_cat, bmi_col = calculate_bmi(h_cm, w_kg)
    bracket_code, bracket_lbl = map_age_to_bracket(user_age)

    col_b1, col_b2, col_b3, col_b4 = st.columns(4)
    with col_b1:
        st.markdown(
            f"""
            <div class="earlydia-metric-box">
                <div class="earlydia-metric-val">{user_age}</div>
                <div class="earlydia-metric-lbl">Age (Years)</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_b2:
        st.markdown(
            f"""
            <div class="earlydia-metric-box">
                <div class="earlydia-metric-val" style="color:{bmi_col};">{bmi_val:.1f}</div>
                <div class="earlydia-metric-lbl">BMI ({bmi_cat})</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_b3:
        st.markdown(
            f"""
            <div class="earlydia-metric-box">
                <div class="earlydia-metric-val">{result['risk_pct']}%</div>
                <div class="earlydia-metric-lbl">ML Risk Score</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_b4:
        st.markdown(
            f"""
            <div class="earlydia-metric-box">
                <div class="earlydia-metric-val" style="font-size: 1.25rem; line-height: 2;">{data.get('gen_hlth', 'Good')}</div>
                <div class="earlydia-metric-lbl">General Health</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<hr style='border:0; border-top:1px solid var(--border-color); margin: 1.5rem 0;'>", unsafe_allow_html=True)

    # Explicit Distinction Banner
    st.markdown(
        """
        <div style="background: var(--bg-card-sec); border-left: 4px solid var(--accent-blue); border-radius: 10px; padding: 0.9rem 1.25rem; margin-bottom: 1.5rem; font-size: 0.88rem;">
            <strong>Important Framing Distinction:</strong><br>
            The <strong>ML Risk Assessment</strong> reflects the statistical risk pattern identified by the trained model.
            The <strong>Profile Observations</strong> below are educational, rule-based observations from preventive healthcare guidelines.
            EarlyDia <em>never claims that any single factor directly caused your score</em>.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Section 1: Factors for Attention
    st.markdown("### 🔍 Factors for Attention")
    st.markdown(
        """
        <div style="font-size: 0.9rem; color: var(--text-secondary); margin-top: -0.5rem; margin-bottom: 1rem;">
            Based on your provided profile, the following factors may deserve additional attention:
        </div>
        """,
        unsafe_allow_html=True
    )

    factors = result["insights"]["factors_for_attention"]
    if factors:
        for factor in factors:
            st.markdown(
                f"""
                <div class="earlydia-card" style="border-left: 4px solid #f59e0b; margin-bottom: 0.9rem;">
                    <div style="display: flex; gap: 0.75rem; align-items: flex-start;">
                        <span style="font-size: 1.5rem;">{factor['icon']}</span>
                        <div style="width: 100%;">
                            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                                <h4 style="margin: 0; font-size: 1.05rem;">{factor['title']}</h4>
                                <span class="earlydia-badge" style="margin-bottom: 0;">{factor['category']}</span>
                            </div>
                            <p style="font-size: 0.88rem; line-height: 1.5; margin: 0.4rem 0;">
                                {factor['observation']}
                            </p>
                            <div style="font-size: 0.82rem; color: var(--accent-blue); font-weight: 600; background: rgba(56, 189, 248, 0.08); padding: 0.45rem 0.75rem; border-radius: 8px;">
                                💡 <strong>Actionable Discussion Point:</strong> {factor['suggestion']}
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            """
            <div class="earlydia-card" style="border-left: 4px solid #10b981;">
                <div style="display: flex; gap: 0.75rem; align-items: center;">
                    <span style="font-size: 1.5rem;">🎉</span>
                    <div>
                        <h4 style="margin: 0;">No High-Priority Adverse Factors Identified</h4>
                        <p style="font-size: 0.88rem; margin: 0.25rem 0 0 0;">
                            Your self-reported profile exhibits primarily healthy baseline indicators across physical activity, diet, and cardiovascular history.
                        </p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Section 2: Positive Habits
    st.markdown("### ✨ Positive Lifestyle Habits")
    pos_habits = result["insights"]["positive_habits"]
    if pos_habits:
        col_p1, col_p2 = st.columns(2)
        for i, habit in enumerate(pos_habits):
            target_col = col_p1 if i % 2 == 0 else col_p2
            with target_col:
                st.markdown(
                    f"""
                    <div class="earlydia-card" style="height: calc(100% - 1rem); border-left: 4px solid #10b981; margin-bottom: 0.8rem;">
                        <div style="display: flex; gap: 0.6rem; align-items: center; margin-bottom: 0.35rem;">
                            <span style="font-size: 1.25rem;">{habit['icon']}</span>
                            <strong style="font-size: 0.95rem;">{habit['title']}</strong>
                        </div>
                        <p style="font-size: 0.84rem; line-height: 1.45; margin: 0;">
                            {habit['observation']}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    render_medical_disclaimer()

    st.markdown("<br>", unsafe_allow_html=True)

    # Take New Assessment Button
    col_rst1, col_rst2, col_rst3 = st.columns([1, 2, 1])
    with col_rst2:
        if st.button("🔄 TAKE A NEW ASSESSMENT", key="insights_reset_btn"):
            reset_assessment()
