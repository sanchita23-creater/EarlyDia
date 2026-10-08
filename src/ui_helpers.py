"""
src/ui_helpers.py
-----------------
UI components, custom CSS theming (Light and Dark Mode), Plotly risk gauge,
and visual presentation helpers for EarlyDia Phase 2.
"""

from typing import Dict, Any, List, Optional
import streamlit as st
import plotly.graph_objects as go

# ==============================================================================
# CSS THEME INJECTION
# ==============================================================================

def get_theme_css(theme: str = "light") -> str:
    """
    Returns custom CSS tailored for the EarlyDia Navy Blue + White healthcare aesthetic.
    Supports persistent Light and Dark themes with high readability and polished micro-interactions.
    """
    if theme == "dark":
        bg_color = "#070d18"
        bg_card = "#0f1a30"
        bg_card_secondary = "#162544"
        border_color = "#1e3357"
        text_primary = "#f1f5f9"
        text_secondary = "#94a3b8"
        text_muted = "#64748b"
        accent_blue = "#38bdf8"
        accent_blue_hover = "#0284c7"
        nav_bg = "#0c1527"
        badge_bg = "rgba(56, 189, 248, 0.15)"
        badge_text = "#38bdf8"
        box_shadow = "0 8px 24px rgba(0, 0, 0, 0.45)"
        btn_gradient = "linear-gradient(135deg, #1e40af 0%, #0284c7 100%)"
        btn_gradient_hover = "linear-gradient(135deg, #1d4ed8 0%, #0369a1 100%)"
        hero_gradient = "linear-gradient(135deg, #0b172e 0%, #11254a 100%)"
    else:
        # Default Light Mode
        bg_color = "#f8fafc"
        bg_card = "#ffffff"
        bg_card_secondary = "#f1f5f9"
        border_color = "#e2e8f0"
        text_primary = "#0a2540"
        text_secondary = "#475569"
        text_muted = "#64748b"
        accent_blue = "#1d4ed8"
        accent_blue_hover = "#1e40af"
        nav_bg = "#ffffff"
        badge_bg = "rgba(29, 78, 216, 0.08)"
        badge_text = "#1d4ed8"
        box_shadow = "0 4px 16px rgba(10, 37, 64, 0.06)"
        btn_gradient = "linear-gradient(135deg, #0a2540 0%, #1e3a8a 100%)"
        btn_gradient_hover = "linear-gradient(135deg, #0f3460 0%, #2563eb 100%)"
        hero_gradient = "linear-gradient(135deg, #ffffff 0%, #f0f7ff 100%)"

    return f"""
    <style>
        /* Google Fonts Import */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

        /* Root Variables */
        :root {{
            --bg-color: {bg_color};
            --bg-card: {bg_card};
            --bg-card-sec: {bg_card_secondary};
            --border-color: {border_color};
            --text-primary: {text_primary};
            --text-secondary: {text_secondary};
            --text-muted: {text_muted};
            --accent-blue: {accent_blue};
            --accent-hover: {accent_blue_hover};
            --box-shadow: {box_shadow};
        }}

        /* Global App Styling */
        html, body, [class*="css"] {{
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            color: {text_primary};
        }}
        
        .stApp {{
            background-color: {bg_color};
            color: {text_primary};
            transition: background-color 0.3s ease, color 0.3s ease;
        }}

        /* Hide Streamlit default header decoration bar */
        header[data-testid="stHeader"] {{
            background-color: transparent !important;
        }}
        
        #MainMenu, footer {{
            visibility: hidden;
        }}

        /* Typography */
        h1, h2, h3, h4, h5, h6 {{
            color: {text_primary} !important;
            font-weight: 700;
            letter-spacing: -0.02em;
        }}

        p, span, label {{
            color: {text_secondary};
        }}

        /* EarlyDia Top Header Bar */
        .earlydia-topbar {{
            background: {nav_bg};
            border-bottom: 1px solid {border_color};
            padding: 0.85rem 1.5rem;
            border-radius: 14px;
            margin-bottom: 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: {box_shadow};
        }}

        /* Hero Banner */
        .earlydia-hero {{
            background: {hero_gradient};
            border: 1px solid {border_color};
            border-radius: 20px;
            padding: 2.8rem 2.2rem;
            margin-bottom: 2rem;
            box-shadow: {box_shadow};
            position: relative;
            overflow: hidden;
        }}

        /* Clean Healthcare Cards */
        .earlydia-card {{
            background: {bg_card};
            border: 1px solid {border_color};
            border-radius: 16px;
            padding: 1.6rem;
            margin-bottom: 1.2rem;
            box-shadow: {box_shadow};
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        }}
        .earlydia-card:hover {{
            transform: translateY(-2px);
            border-color: {accent_blue};
            box-shadow: 0 10px 25px rgba(10, 37, 64, 0.12);
        }}

        .earlydia-card-interactive {{
            background: {bg_card};
            border: 1px solid {border_color};
            border-radius: 16px;
            padding: 1.4rem;
            margin-bottom: 1rem;
            box-shadow: {box_shadow};
        }}

        /* Metric / Value Highlights */
        .earlydia-metric-box {{
            background: {bg_card_secondary};
            border: 1px solid {border_color};
            border-radius: 12px;
            padding: 1rem 1.25rem;
            text-align: center;
        }}
        .earlydia-metric-val {{
            font-size: 2rem;
            font-weight: 800;
            color: {accent_blue};
            line-height: 1.1;
        }}
        .earlydia-metric-lbl {{
            font-size: 0.82rem;
            font-weight: 600;
            color: {text_muted};
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.35rem;
        }}

        /* Feature Pill / Badges */
        .earlydia-badge {{
            display: inline-block;
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 700;
            background: {badge_bg};
            color: {badge_text};
            letter-spacing: 0.02em;
            margin-bottom: 0.6rem;
        }}

        /* Disclaimer Banner */
        .earlydia-disclaimer {{
            background: {"rgba(15, 23, 42, 0.6)" if theme == "dark" else "#f8fafc"};
            border: 1px solid {border_color};
            border-left: 4px solid #f59e0b;
            border-radius: 10px;
            padding: 0.9rem 1.2rem;
            font-size: 0.82rem;
            line-height: 1.45;
            color: {text_muted};
            margin-top: 1.8rem;
        }}

        /* Buttons Styling */
        div.stButton > button {{
            background: {btn_gradient} !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.65rem 1.5rem !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            letter-spacing: 0.01em !important;
            box-shadow: 0 4px 12px rgba(10, 37, 64, 0.2) !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
            width: 100%;
        }}
        div.stButton > button:hover {{
            background: {btn_gradient_hover} !important;
            box-shadow: 0 6px 18px rgba(10, 37, 64, 0.3) !important;
            transform: translateY(-1px) !important;
        }}

        /* Secondary / Back Buttons */
        div.stButton.btn-secondary > button,
        div[data-testid="column"]:first-child div.stButton > button:not(.btn-primary-forced) {{
            /* Can be specifically overridden in code when needed */
        }}

        /* Input Controls and Labels */
        .stSelectbox label, .stSlider label, .stRadio label, .stNumberInput label {{
            color: {text_primary} !important;
            font-weight: 600 !important;
            font-size: 0.92rem !important;
            margin-bottom: 0.25rem !important;
        }}
        
        .stSlider > div {{
            padding-top: 0.5rem;
        }}

        /* Radio Buttons */
        div[data-testid="stRadio"] > div {{
            gap: 0.75rem;
        }}
        div[data-testid="stRadio"] label span {{
            color: {text_primary} !important;
            font-size: 0.9rem !important;
        }}

        /* Stepper Header */
        .stepper-container {{
            background: {bg_card};
            border: 1px solid {border_color};
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.75rem;
            box-shadow: {box_shadow};
        }}
        .stepper-header-title {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
        }}
        .stepper-step-count {{
            font-size: 0.8rem;
            font-weight: 800;
            color: {accent_blue};
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}
        .stepper-step-name {{
            font-size: 1.15rem;
            font-weight: 700;
            color: {text_primary};
        }}
        .stepper-bar-bg {{
            width: 100%;
            height: 8px;
            background: {bg_card_secondary};
            border-radius: 9999px;
            overflow: hidden;
        }}
        .stepper-bar-fill {{
            height: 100%;
            background: {btn_gradient};
            border-radius: 9999px;
            transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        /* Status Tiers */
        .tier-card-low {{
            background: {"rgba(16, 185, 129, 0.12)" if theme == "dark" else "#ecfdf5"};
            border: 2px solid #10b981;
            border-radius: 16px;
            padding: 1.6rem;
            color: {"#a7f3d0" if theme == "dark" else "#065f46"};
        }}
        .tier-card-moderate {{
            background: {"rgba(245, 158, 11, 0.12)" if theme == "dark" else "#fffbeb"};
            border: 2px solid #f59e0b;
            border-radius: 16px;
            padding: 1.6rem;
            color: {"#fde68a" if theme == "dark" else "#92400e"};
        }}
        .tier-card-high {{
            background: {"rgba(239, 68, 68, 0.12)" if theme == "dark" else "#fef2f2"};
            border: 2px solid #ef4444;
            border-radius: 16px;
            padding: 1.6rem;
            color: {"#fca5a5" if theme == "dark" else "#991b1b"};
        }}
    </style>
    """


def apply_theme(theme: str = "light") -> None:
    """Injects the theme CSS into the current Streamlit app page."""
    css = get_theme_css(theme)
    st.markdown(css, unsafe_allow_html=True)


# ==============================================================================
# UI COMPONENTS
# ==============================================================================

def render_top_navigation(theme: str = "light") -> None:
    """
    Renders top app branding bar with logo, theme toggle button, and quick actions.
    """
    col1, col2, col3 = st.columns([6, 3, 3])
    with col1:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-top: 0.25rem;">
                <span style="font-size: 1.65rem;">🩺</span>
                <div>
                    <span style="font-size: 1.35rem; font-weight: 800; letter-spacing: -0.03em; color: var(--text-primary);">EarlyDia</span>
                    <span style="font-size: 0.72rem; font-weight: 700; background: rgba(56, 189, 248, 0.2); color: var(--accent-blue); padding: 0.15rem 0.5rem; border-radius: 9999px; margin-left: 0.5rem;">AI PRE-SCREENING</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        # Quick Model Info button
        if st.button("ℹ️ Model Info", key="top_model_info_btn", help="Inspect real Phase 1 validation metrics"):
            st.session_state.show_model_info = not st.session_state.get("show_model_info", False)
            st.rerun()
    with col3:
        # Theme toggle button
        theme_icon = "☀️ Light" if theme == "dark" else "🌙 Dark"
        if st.button(f"{theme_icon}", key="top_theme_toggle_btn", help="Toggle between Light and Dark mode"):
            st.session_state.theme = "light" if theme == "dark" else "dark"
            st.rerun()


def render_stepper(step: int, total_steps: int = 4, step_name: str = "") -> None:
    """
    Renders the multi-step progress bar (Step X of 4).
    """
    pct = int((step / total_steps) * 100)
    st.markdown(
        f"""
        <div class="stepper-container">
            <div class="stepper-header-title">
                <div>
                    <div class="stepper-step-count">STEP {step} OF {total_steps}</div>
                    <div class="stepper-step-name">{step_name}</div>
                </div>
                <div style="font-weight: 800; font-size: 1rem; color: var(--accent-blue);">{pct}%</div>
            </div>
            <div class="stepper-bar-bg">
                <div class="stepper-bar-fill" style="width: {pct}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_medical_disclaimer() -> None:
    """Renders the required non-diagnostic medical disclaimer banner."""
    st.markdown(
        """
        <div class="earlydia-disclaimer">
            <strong>⚠️ Preliminary Screening Disclaimer:</strong> EarlyDia is an exploratory Machine Learning-based 
            preliminary risk screening system designed for informational and educational purposes. 
            It is <strong>not a medical diagnosis</strong> and does not replace regular clinical testing (such as fasting glucose or HbA1c) 
            or professional consultation with a qualified medical healthcare provider.
        </div>
        """,
        unsafe_allow_html=True
    )


# ==============================================================================
# PLOTLY RISK GAUGE CHART
# ==============================================================================

def create_risk_gauge(risk_score_pct: float, risk_tier: str, theme: str = "light") -> go.Figure:
    """
    Creates a modern circular arc gauge in Plotly visualizing the ML Risk Score.
    
    Tiers & Thresholds:
      - Below 30%: Low Risk Indication (Green / Emerald)
      - 30% to 60%: Moderate / Elevated Risk Indication (Amber)
      - Above 60%: High Risk Indication (Red / Rose)
    """
    is_dark = (theme == "dark")
    bg_color = "rgba(0, 0, 0, 0)"
    font_color = "#f8fafc" if is_dark else "#0a2540"
    
    if risk_score_pct < 30.0:
        bar_color = "#10b981"  # Emerald
    elif risk_score_pct <= 60.0:
        bar_color = "#f59e0b"  # Amber
    else:
        bar_color = "#ef4444"  # Rose / Crimson

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score_pct,
        number={
            "suffix": "%",
            "font": {"size": 48, "color": font_color, "family": "Plus Jakarta Sans, sans-serif"}
        },
        title={
            "text": "<b>ML RISK SCORE</b><br><span style='font-size:0.85em;color:#64748b;'>Prototype Screening Scale</span>",
            "font": {"size": 16, "color": font_color, "family": "Plus Jakarta Sans, sans-serif"}
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 2,
                "tickcolor": "#94a3b8",
                "tickvals": [0, 30, 60, 100],
                "ticktext": ["0%", "30%", "60%", "100%"],
                "tickfont": {"size": 12, "color": font_color}
            },
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": "rgba(255, 255, 255, 0.05)" if is_dark else "rgba(0, 0, 0, 0.05)",
            "borderwidth": 1,
            "bordercolor": "#1e3357" if is_dark else "#cbd5e1",
            "steps": [
                {"range": [0, 30], "color": "rgba(16, 185, 129, 0.18)"},
                {"range": [30, 60], "color": "rgba(245, 158, 11, 0.18)"},
                {"range": [60, 100], "color": "rgba(239, 68, 68, 0.18)"}
            ],
            "threshold": {
                "line": {"color": bar_color, "width": 4},
                "thickness": 0.85,
                "value": risk_score_pct
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font={"color": font_color, "family": "Plus Jakarta Sans"},
        margin=dict(l=25, r=25, t=50, b=25),
        height=280
    )
    return fig


# ==============================================================================
# MODEL PERFORMANCE MODAL / DRAWER
# ==============================================================================

def render_model_performance_section(theme: str = "light") -> None:
    """
    Displays the empirical Phase 1 evaluation metrics from reports/model_results.txt
    comparing Logistic Regression vs Random Forest.
    """
    st.markdown(
        """
        <div class="earlydia-card" style="margin-bottom: 1.5rem; border-left: 4px solid var(--accent-blue);">
            <div class="earlydia-badge">PHASE 1 EMPIRICAL EVALUATION</div>
            <h3 style="margin-top: 0.2rem; margin-bottom: 0.4rem;">Model Evaluation & Selection Justification</h3>
            <p style="font-size: 0.9rem; margin-bottom: 1rem;">
                Evaluated on the CDC Diabetes Health Indicators held-out test split 
                <strong>(20% Stratified Split, n = 50,736 instances)</strong> without data leakage.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Actual Phase 1 metrics from reports/model_results.txt
    metrics_data = [
        {"metric": "Accuracy", "lr": "73.03%", "rf": "72.60%", "diff": "-0.43%", "focus": "Overall classification rate"},
        {"metric": "Precision", "lr": "30.95%", "rf": "30.79%", "diff": "-0.16%", "focus": "True positive rate over positive predictions"},
        {"metric": "Recall (Sensitivity)", "lr": "75.97%", "rf": "77.45%", "diff": "+1.48% ★", "focus": "Critical for screening: minimizing missed positive cases"},
        {"metric": "F1 Score", "lr": "43.98%", "rf": "44.06%", "diff": "+0.08% ★", "focus": "Harmonic balance between precision and recall"},
        {"metric": "ROC-AUC", "lr": "0.8170", "rf": "0.8220", "diff": "+0.0050 ★", "focus": "Overall discriminative power across all thresholds"}
    ]

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(
            """
            <div class="earlydia-card" style="text-align: center;">
                <div style="font-size: 0.85rem; font-weight: 700; color: var(--text-muted);">BASELINE CANDIDATE</div>
                <h4 style="margin: 0.4rem 0;">Logistic Regression</h4>
                <div style="font-size: 1.8rem; font-weight: 800; color: var(--text-primary);">0.8170</div>
                <div style="font-size: 0.78rem; color: var(--text-muted);">ROC-AUC Score</div>
                <hr style="border: 0; border-top: 1px solid var(--border-color); margin: 0.8rem 0;">
                <div style="display: flex; justify-content: space-around; font-size: 0.85rem;">
                    <div><strong>Recall:</strong> 75.97%</div>
                    <div><strong>F1 Score:</strong> 43.98%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_m2:
        st.markdown(
            """
            <div class="earlydia-card" style="text-align: center; border: 2px solid #2563eb; background: rgba(37, 99, 235, 0.04);">
                <div class="earlydia-badge" style="background: #2563eb; color: #ffffff;">SELECTED WINNER</div>
                <h4 style="margin: 0.4rem 0; color: #2563eb !important;">Random Forest Pipeline</h4>
                <div style="font-size: 1.8rem; font-weight: 800; color: #2563eb;">0.8220</div>
                <div style="font-size: 0.78rem; color: var(--text-muted);">ROC-AUC Score (+0.0050)</div>
                <hr style="border: 0; border-top: 1px solid var(--border-color); margin: 0.8rem 0;">
                <div style="display: flex; justify-content: space-around; font-size: 0.85rem;">
                    <div><strong>Recall:</strong> <span style="color: #10b981; font-weight:700;">77.45%</span></div>
                    <div><strong>F1 Score:</strong> <span style="color: #10b981; font-weight:700;">44.06%</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Detailed Comparison Table
    table_rows = ""
    for row in metrics_data:
        highlight = "background: rgba(37, 99, 235, 0.08); font-weight: 700;" if "★" in row["diff"] else ""
        table_rows += f"""
        <tr style="{highlight}">
            <td style="padding: 0.65rem 1rem; border-bottom: 1px solid var(--border-color);"><strong>{row['metric']}</strong><br><small style="color: var(--text-muted);">{row['focus']}</small></td>
            <td style="padding: 0.65rem 1rem; border-bottom: 1px solid var(--border-color); text-align: center;">{row['lr']}</td>
            <td style="padding: 0.65rem 1rem; border-bottom: 1px solid var(--border-color); text-align: center; color: #2563eb;"><strong>{row['rf']}</strong></td>
            <td style="padding: 0.65rem 1rem; border-bottom: 1px solid var(--border-color); text-align: center;">{row['diff']}</td>
        </tr>
        """

    st.markdown(
        f"""
        <div class="earlydia-card" style="padding: 0; overflow: hidden;">
            <table style="width: 100%; border-collapse: collapse; font-size: 0.88rem;">
                <thead>
                    <tr style="background: var(--bg-card-sec); border-bottom: 1px solid var(--border-color);">
                        <th style="padding: 0.75rem 1rem; text-align: left;">Metric</th>
                        <th style="padding: 0.75rem 1rem; text-align: center;">Logistic Regression</th>
                        <th style="padding: 0.75rem 1rem; text-align: center;">Random Forest (Selected)</th>
                        <th style="padding: 0.75rem 1rem; text-align: center;">Difference</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.5; margin-top: 0.8rem;">
            <strong>Clinical Screening Selection Rationale:</strong><br>
            The final model was selected based on its overall performance, with particular consideration given to 
            <strong>Recall (77.45% vs 75.97%)</strong> and <strong>F1 Score (44.06% vs 43.98%)</strong> because EarlyDia is designed as an 
            early risk screening tool. In preliminary screening, capturing true positive risk cases to encourage timely medical follow-up 
            is clinically more vital than minimizing false positives.
        </div>
        """,
        unsafe_allow_html=True
    )
