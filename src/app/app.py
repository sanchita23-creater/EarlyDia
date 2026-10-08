"""
EarlyDia - Streamlit application entry point.

Research/prototype screening tool built on NFHS-5. ALL inference goes through the production
prediction module (src/inference/predict_module.py); nothing here trains, re-fits or alters a model.

Run with:  streamlit run src/app/app.py
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="EarlyDia | Diabetes-Range Screening",
                   page_icon="🩺", layout="centered", initial_sidebar_state="collapsed")

from src.app import ui_components as C  # noqa: E402
from src.app import ui_state as U  # noqa: E402
from src.app.theme import get_css  # noqa: E402
from src.app.views import VIEWS  # noqa: E402


def main() -> None:
    U.init_state()
    left, right = st.columns([3, 1], vertical_alignment="center")
    left.markdown(C.brand_html(), unsafe_allow_html=True)
    right.toggle("Dark mode", key="dark_mode")
    st.markdown(get_css(st.session_state["dark_mode"]), unsafe_allow_html=True)

    page = st.session_state["page"]
    if page != "landing":
        st.markdown(C.stepper_html(page if page != "recs" else "recs"), unsafe_allow_html=True)
    VIEWS[page]()


main()
