"""
EarlyDia - theme definitions for the Streamlit app (navy / white / black identity).

Two explicit palettes (light, dark). Palette values are plain data so their text/background
contrast can be unit-tested (WCAG). Dark-mode overrides also target Streamlit's dropdown
popovers, alerts, bordered containers and metrics explicitly, because those render outside the
main container and would otherwise keep light-theme surfaces with light text.

No streamlit import here - this module is pure data + string building.
"""
from string import Template

NAVY = "#0A1F44"
NAVY_MID = "#14335E"

PALETTES = {
    "light": {
        "bg": "#F4F6FA", "surface": "#FFFFFF", "surface_alt": "#EEF2F8", "sidebar": "#FFFFFF",
        "text": "#0B0C0E", "muted": "#4A5565", "border": "#D3DBE8", "track": "#E9EEF6",
        "accent": "#1C4E9C", "button_bg": NAVY, "button_hover": "#1C4E9C", "button_text": "#FFFFFF",
        "input_bg": "#FFFFFF", "flag": "#B3261E", "ok": "#1E7A46",
        "flag_bg": "#FCE9E7", "ok_bg": "#E6F4EC",
    },
    "dark": {
        "bg": "#0B1220", "surface": "#141E33", "surface_alt": "#1A2744", "sidebar": "#0F182B",
        "text": "#F2F5FA", "muted": "#B7C1D6", "border": "#34466B", "track": "#0F1830",
        "accent": "#8DB2F5", "button_bg": "#2A5BB5", "button_hover": "#3B70D0", "button_text": "#FFFFFF",
        "input_bg": "#0F1830", "flag": "#FF8A80", "ok": "#5FD09A",
        "flag_bg": "#3A1F26", "ok_bg": "#12302A",
    },
}

_CSS = Template(r"""
<style>
html, body, .stApp { font-family: 'Inter','Segoe UI',-apple-system,BlinkMacSystemFont,sans-serif; }
:root { --ed-ok: $ok; --ed-flag: $flag; --ed-accent: $accent; --ed-muted: $muted; }
.stApp { background-color: $bg; color: $text; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 940px; }

/* chrome: no sidebar, no Deploy/menu/footer - this is a single-purpose product screen */
section[data-testid="stSidebar"], div[data-testid="stSidebarCollapsedControl"],
div[data-testid="collapsedControl"], div[data-testid="stToolbar"], #MainMenu, footer,
div[data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; height: 0; pointer-events: none; }
header[data-testid="stHeader"] * { pointer-events: none; }

/* text */
.stApp p, .stApp li, .stApp label, .stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stWidgetLabel"] * { color: $text; font-size: 1rem; line-height: 1.55; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: $text !important; font-weight: 700; letter-spacing: -0.01em; }
.ed-muted, .ed-muted * { color: $muted !important; }
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] * { color: $muted !important; opacity: 1 !important; }
hr { border-color: $border; }


/* Streamlit gives headings big default padding - reset for our own headings */
.stApp .ed-title, .stApp .ed-hero h1, .stApp .ed-hero h2, .stApp .ed-card h3 { padding: 0 !important; }
.ed-hero .ed-pill svg { width: 15px; height: 15px; stroke: #FFFFFF; fill: none; stroke-width: 2.4; stroke-linecap: round; stroke-linejoin: round; flex: 0 0 auto; }
.ed-hero .ed-pill { white-space: nowrap; }
.ed-card svg, .ed-result svg { max-width: 100%; }
div[data-testid="stToggle"] { display: flex; justify-content: flex-end; }

/* explicit colours beat the generic .stApp p rule */
.stApp .ed-sub, .stApp .ed-card p, .stApp .ed-muted { color: $muted !important; }
.stApp .ed-hero p { color: #E6EDF9 !important; }
.stApp .ed-rec .sum, .stApp .ed-rec li, .stApp .ed-list li, .stApp .ed-note, .stApp .ed-note b { color: $text !important; }
div[data-testid="stVerticalBlockBorderWrapper"] .ed-section-title { margin-top: .1rem; }

/* visually hidden text for assistive tech / copy-paste (keeps '5. Result' readable) */
.ed-sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
.ed-how-num { display: inline-flex; align-items: center; justify-content: center; width: 34px; height: 34px; border-radius: 50%;
    background: $button_bg; color: $button_text; font-weight: 800; border: 1px solid $button_bg; }
.stApp .ed-how-num { color: $button_text !important; }
.ed-details { background: $surface; border: 1px solid $border; border-radius: 12px; margin: .9rem 0; }
.ed-details summary { cursor: pointer; padding: .8rem 1.1rem; font-weight: 700; color: $text; list-style-position: inside; }
.ed-details summary:focus-visible { outline: 3px solid $accent; outline-offset: 2px; border-radius: 10px; }
.ed-details-body { padding: 0 1.2rem 1rem 1.2rem; }
.ed-details-body ul { margin: .3rem 0 0 0; padding-left: 1.2rem; }
.ed-details-body li, .ed-details-body p { color: $text; font-size: .95rem; margin-bottom: .35rem; }
.ed-diet { display: grid; grid-template-columns: repeat(2, 1fr); gap: .15rem 1.6rem; }
.ed-diet-row { display: flex; justify-content: space-between; gap: 1rem; padding: .42rem 0; border-bottom: 1px solid $border; }
.ed-diet-row span { color: $muted; font-size: .95rem; } .ed-diet-row b { color: $text; font-size: .95rem; text-align: right; }

/* top bar */
.ed-brand { display: flex; align-items: center; gap: .6rem; font-weight: 800; font-size: 1.25rem; color: $text; letter-spacing: -0.02em; }
.ed-brand .ed-logo { width: 34px; height: 34px; border-radius: 9px; background: linear-gradient(135deg, $navy 0%, $navy_mid 100%);
    display: inline-flex; align-items: center; justify-content: center; border: 1px solid $border; }
.ed-brand .ed-logo svg { width: 20px; height: 20px; stroke: #FFFFFF; fill: none; stroke-width: 2.2; stroke-linecap: round; stroke-linejoin: round; }
.ed-brand small { font-weight: 600; color: $muted; font-size: .8rem; letter-spacing: 0; margin-left: .2rem; }

/* stepper */
.ed-stepper { display: flex; align-items: center; gap: .35rem; background: $surface; border: 1px solid $border;
    border-radius: 14px; padding: .6rem .8rem; margin: .6rem 0 1.4rem 0; overflow-x: auto; }
.ed-step { display: flex; align-items: center; gap: .5rem; flex: 1 1 auto; white-space: nowrap; color: $muted; font-weight: 600; font-size: .9rem; }
.ed-step + .ed-step::before { content: ""; flex: 1 1 14px; min-width: 10px; height: 2px; background: $border; margin-right: .35rem; border-radius: 2px; }
.ed-step .ed-num { width: 26px; height: 26px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center;
    font-size: .8rem; font-weight: 800; background: $surface_alt; color: $muted; border: 1px solid $border; flex: 0 0 auto; }
.ed-step .ed-num svg { width: 14px; height: 14px; stroke: currentColor; fill: none; stroke-width: 3.2; stroke-linecap: round; stroke-linejoin: round; }
.ed-step.active { color: $text; }
.ed-step.active .ed-num { background: $button_bg; color: $button_text; border-color: $button_bg; }
.ed-step.done { color: $text; }
.ed-step.done .ed-num { background: $ok; color: $surface; border-color: $ok; }
.ed-step.done + .ed-step::before, .ed-step.active + .ed-step::before { background: $border; }

/* page head */
.ed-eyebrow { color: $accent; font-weight: 800; font-size: .78rem; letter-spacing: .12em; text-transform: uppercase; margin-bottom: .2rem; }
.ed-title { color: $text; font-weight: 800; font-size: 1.9rem; letter-spacing: -0.02em; line-height: 1.2; margin: 0 0 .35rem 0; }
.ed-sub { color: $muted; font-size: 1.04rem; margin: 0 0 1.2rem 0; max-width: 62ch; }

/* landing hero */
.ed-hero { background: linear-gradient(135deg, $navy 0%, $navy_mid 100%); border: 1px solid $border; border-radius: 20px;
    padding: 2.6rem 2.4rem; margin: .4rem 0 1.4rem 0; }
.ed-hero .ed-pill { display: inline-flex; align-items: center; gap: .45rem; background: rgba(255,255,255,.12); color: #FFFFFF;
    border: 1px solid rgba(255,255,255,.28); border-radius: 999px; padding: .25rem .8rem; font-size: .8rem; font-weight: 700; letter-spacing: .04em; }
.ed-hero h1 { color: #FFFFFF !important; font-size: 3rem; line-height: 1.05; margin: .9rem 0 .3rem 0; font-weight: 800; letter-spacing: -0.03em; }
.ed-hero h2 { color: #DCE6F7 !important; font-size: 1.35rem; font-weight: 600; margin: 0 0 .9rem 0; letter-spacing: -0.01em; }
.ed-hero p { color: #E6EDF9 !important; font-size: 1.08rem; max-width: 60ch; margin: 0; }

/* cards */
.ed-grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.2rem; }
.ed-grid-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-bottom: 1.2rem; }
.ed-card { background: $surface; border: 1px solid $border; border-radius: 14px; padding: 1.25rem 1.35rem; }
.ed-card h3 { margin: .6rem 0 .35rem 0 !important; font-size: 1.08rem !important; }
.ed-card p { margin: 0; color: $muted; font-size: .96rem; }
.ed-ico { width: 40px; height: 40px; border-radius: 11px; display: inline-flex; align-items: center; justify-content: center;
    background: $surface_alt; border: 1px solid $border; }
.ed-ico svg { width: 21px; height: 21px; stroke: $accent; fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }
.ed-ico.ok svg { stroke: $ok; } .ed-ico.flag svg { stroke: $flag; }
.ed-list { list-style: none; padding: 0; margin: .6rem 0 0 0; }
.ed-list li { display: flex; gap: .6rem; align-items: flex-start; padding: .32rem 0; color: $text; font-size: .98rem; }
.ed-list li svg { width: 18px; height: 18px; flex: 0 0 18px; margin-top: .22rem; fill: none; stroke-width: 2.4; stroke-linecap: round; stroke-linejoin: round; }
.ed-list.yes li svg { stroke: $ok; } .ed-list.no li svg { stroke: $flag; }
.ed-section-title { display: flex; align-items: center; gap: .6rem; margin: 1.6rem 0 .6rem 0; font-size: 1.25rem; font-weight: 800; color: $text; letter-spacing: -0.01em; }

/* disclaimer / callouts */
.ed-note { background: $surface; border: 1px solid $border; border-left: 5px solid $accent; border-radius: 10px;
    padding: .95rem 1.2rem; font-size: .95rem; color: $text; margin: 1rem 0; }
.ed-note b { color: $text; }
.ed-callout-ok, .ed-callout-flag { border-radius: 10px; padding: .9rem 1.2rem; font-size: .98rem; margin: .8rem 0; border: 1px solid $border; }
.ed-callout-ok { background: $ok_bg; border-left: 5px solid $ok; }
.ed-callout-flag { background: $flag_bg; border-left: 5px solid $flag; }
.ed-callout-ok, .ed-callout-flag, .ed-callout-ok *, .ed-callout-flag * { color: $text !important; }

/* results */
.ed-result { background: $surface; border: 1px solid $border; border-radius: 18px; padding: 1.8rem 1.6rem 1.2rem 1.6rem; text-align: center; }
.ed-result { margin-bottom: .6rem; }
.ed-result.flag { border: 2px solid $flag; } .ed-result.ok { border: 2px solid $ok; }
.ed-result .ed-kicker { color: $muted; font-size: .8rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }
.ed-big { font-size: 3.6rem; font-weight: 800; line-height: 1.05; letter-spacing: -0.03em; color: $text; margin: .2rem 0 0 0; }
.ed-badge { display: inline-flex; align-items: center; justify-content: center; text-align: center; gap: .45rem; border-radius: 999px; padding: .4rem 1rem; font-weight: 800; font-size: 1rem; margin-top: .8rem; }
.ed-badge.flag { background: $flag_bg; color: $flag; border: 1px solid $flag; }
.ed-badge.ok { background: $ok_bg; color: $ok; border: 1px solid $ok; }
.ed-badge svg { width: 18px; height: 18px; fill: none; stroke: currentColor; stroke-width: 2.4; stroke-linecap: round; stroke-linejoin: round; }
.ed-kpis { display: grid; grid-template-columns: repeat(2, 1fr); gap: .8rem; margin-top: 1.1rem; text-align: left; }
.ed-kpi { background: $surface_alt; border: 1px solid $border; border-radius: 12px; padding: .7rem .9rem; }
.ed-kpi .l { color: $muted; font-size: .76rem; font-weight: 700; letter-spacing: .05em; text-transform: uppercase; }
.ed-kpi .v { color: $text; font-size: 1.12rem; font-weight: 800; margin-top: .1rem; }
.ed-legend { display: flex; justify-content: center; gap: 1.2rem; flex-wrap: wrap; margin: .2rem 0 .3rem 0; color: $muted; font-size: .85rem; }
.ed-legend i { display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin-right: .35rem; vertical-align: -1px; }
.ed-factors { display: grid; grid-template-columns: repeat(2, 1fr); gap: .7rem; margin: .6rem 0; }
.ed-factor { background: $surface; border: 1px solid $border; border-radius: 12px; padding: .75rem 1rem; }
.ed-factor .n { color: $muted; font-size: .76rem; font-weight: 700; letter-spacing: .05em; text-transform: uppercase; }
.ed-factor .v { color: $text; font-size: 1.02rem; font-weight: 700; margin-top: .1rem; }

/* review */
.ed-kv { display: grid; grid-template-columns: repeat(2, 1fr); gap: .55rem 1.2rem; }
.ed-kv .k { color: $muted; font-size: .76rem; font-weight: 700; letter-spacing: .05em; text-transform: uppercase; }
.ed-kv .v { color: $text; font-size: 1.02rem; font-weight: 600; }
.ed-kv .full { grid-column: 1 / -1; }

/* recommendations */
.ed-rec { background: $surface; border: 1px solid $border; border-radius: 14px; padding: 1.15rem 1.35rem; margin-bottom: .9rem; }
.ed-rec-head { display: flex; justify-content: space-between; align-items: flex-start; gap: .8rem; flex-wrap: wrap; margin-bottom: .35rem; }
.ed-rec-title { display: flex; align-items: center; gap: .65rem; font-weight: 800; font-size: 1.08rem; color: $text; }
.ed-rec .sum { color: $text; margin: .2rem 0 .5rem 0; font-size: .98rem; }
.ed-rec ul { margin: .2rem 0 0 0; padding-left: 1.2rem; }
.ed-rec li { color: $text; font-size: .96rem; margin-bottom: .3rem; }
.ed-tag { display: inline-block; border-radius: 999px; padding: .15rem .7rem; font-size: .74rem; font-weight: 800; letter-spacing: .03em; border: 1px solid $border; white-space: nowrap; }
.ed-tag-alert { background: $flag_bg; color: $flag; border-color: $flag; }
.ed-tag-ok { background: $ok_bg; color: $ok; border-color: $ok; }
.ed-tag-info, .ed-tag-neutral { background: $surface_alt; color: $accent; border-color: $border; }

/* buttons: primary = filled navy, secondary = outlined; full width inside their column */
div[data-testid="stButton"] > button, .stButton > button { width: 100%; border-radius: 10px; font-weight: 700; font-size: 1rem; padding: .65rem 1.2rem; }
button[kind="primary"], button[data-testid="stBaseButton-primary"] { background-color: $button_bg !important; color: $button_text !important; border: 1px solid $button_bg !important; }
button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover { background-color: $button_hover !important; border-color: $button_hover !important; }
button[kind="primary"] *, button[data-testid="stBaseButton-primary"] * { color: $button_text !important; }
button[kind="secondary"], button[data-testid="stBaseButton-secondary"] { background-color: transparent !important; border: 1.5px solid $accent !important; }
button[kind="secondary"] *, button[data-testid="stBaseButton-secondary"] * { color: $accent !important; }
button[kind="secondary"]:hover, button[data-testid="stBaseButton-secondary"]:hover { background-color: $surface_alt !important; }
button:focus-visible { outline: 3px solid $accent !important; outline-offset: 2px; }

/* inputs */
.stNumberInput input, .stTextInput input, div[data-baseweb="select"] > div {
    background-color: $input_bg !important; color: $text !important;
    border: 1px solid $border !important; border-radius: 10px !important; font-size: 1rem !important; }
.stNumberInput input::placeholder, div[data-baseweb="select"] input::placeholder { color: $muted !important; opacity: 1; }
div[data-baseweb="select"] span, div[data-baseweb="select"] div { color: $text !important; }
div[data-testid="stNumberInputContainer"] button { background-color: $input_bg !important; color: $text !important; border-color: $border !important; }
div[data-testid="stNumberInputContainer"] button svg, div[data-baseweb="select"] svg { fill: $text !important; }
.stCheckbox span, .stRadio label span { color: $text !important; }
div[data-testid="stRadio"] div[role="radiogroup"] { gap: .5rem; }
div[data-testid="stRadio"] div[role="radiogroup"] > label { background: $input_bg; border: 1px solid $border; border-radius: 10px; padding: .45rem .95rem; }
div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) { border-color: $accent; background: $surface_alt; box-shadow: inset 0 0 0 1px $accent; }
div[data-testid="stCheckbox"] label { padding: .1rem 0; }
div[data-testid="stToggle"] label * { color: $text !important; }

/* bordered containers used as form cards */
div[data-testid="stVerticalBlockBorderWrapper"] { background: $surface; border-color: $border !important; border-radius: 14px; }

/* dropdown popovers render in a portal - style explicitly so text never sits on the wrong surface */
div[data-baseweb="popover"], div[data-baseweb="popover"] ul, div[data-baseweb="menu"] { background-color: $surface !important; border: 1px solid $border; }
div[data-baseweb="popover"] li, div[data-baseweb="popover"] li *, div[data-baseweb="menu"] li * { color: $text !important; background-color: $surface; }
div[data-baseweb="popover"] li:hover, div[data-baseweb="popover"] li[aria-selected="true"] { background-color: $border !important; }

/* metrics */
[data-testid="stMetric"] { background: $surface; border: 1px solid $border; border-radius: 10px; padding: .8rem 1rem; }
[data-testid="stMetricValue"] * { color: $text !important; }
[data-testid="stMetricLabel"] * { color: $muted !important; }

/* alerts (validation messages) */
div[data-testid="stAlert"] { background-color: $surface !important; border: 1px solid $border; border-left: 5px solid $flag; border-radius: 10px; }
div[data-testid="stAlert"] * { color: $text !important; }

/* expander */
div[data-testid="stExpander"] { background: $surface; border: 1px solid $border; border-radius: 12px; }
div[data-testid="stExpander"] * { color: $text; }

@media (max-width: 720px) {
  .ed-grid-3, .ed-grid-2, .ed-factors, .ed-kv, .ed-diet { grid-template-columns: 1fr; }
  .ed-kpis { grid-template-columns: 1fr 1fr; gap: .5rem; }
  .ed-hero { padding: 1.7rem 1.3rem; } .ed-hero h1 { font-size: 2.2rem; } .ed-hero h2 { font-size: 1.1rem; }
  .ed-result { padding: 1.1rem .6rem .9rem .6rem; }
  .ed-result .ed-kicker { font-size: .72rem; }
  .ed-big { font-size: 2.8rem; } .ed-title { font-size: 1.55rem; }
  .ed-step span.ed-lbl { display: none; } .ed-step.active span.ed-lbl { display: inline; }
}
</style>
""")


def get_css(dark_mode: bool) -> str:
    c = PALETTES["dark" if dark_mode else "light"]
    return _CSS.substitute(**c, navy=NAVY, navy_mid=NAVY_MID)


def _lum(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    rgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast_ratio(fg: str, bg: str) -> float:
    """WCAG 2.x contrast ratio between two hex colors."""
    a, b = _lum(fg), _lum(bg)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)
