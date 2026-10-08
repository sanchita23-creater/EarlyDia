"""
EarlyDia - HTML building blocks for the product UI (pure functions, no streamlit import).

Every function returns a compact HTML string with no blank lines and no indentation, because
Markdown treats indented lines/blank lines inside raw HTML as code blocks. Dynamic text is
escaped. Icons are inline stroke SVGs (24x24 grid) so they inherit colour from the theme CSS.
"""
from html import escape as _e

_ICONS = {
    "pulse": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "user": '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "users": '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    "check": '<polyline points="20 6 9 17 4 12"/>',
    "x": '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>',
    "info": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
    "heart": '<path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>',
    "clipboard": '<path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><rect x="8" y="2" width="8" height="4" rx="1"/>',
    "sliders": '<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/>',
    "alert": '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "droplet": '<path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/>',
    "wind": '<path d="M9.59 4.59A2 2 0 1 1 11 8H2m10.59 11.41A2 2 0 1 0 14 16H2m15.73-8.27A2.5 2.5 0 1 1 19.5 12H2"/>',
    "coffee": '<path d="M18 8h1a4 4 0 0 1 0 8h-1"/><path d="M2 8h16v9a4 4 0 0 1-4 4H6a4 4 0 0 1-4-4V8z"/><line x1="6" y1="1" x2="6" y2="4"/><line x1="10" y1="1" x2="10" y2="4"/><line x1="14" y1="1" x2="14" y2="4"/>',
    "apple": '<path d="M12 6c-2-2-6-1-7 2-1 4 1 9 4 11 1 .7 2 .5 3 0 1 .5 2 .7 3 0 3-2 5-7 4-11-1-3-5-4-7-2z"/><path d="M12 6c0-2 1-3 3-4"/>',
    "scale": '<path d="M12 3v18"/><path d="M5 7h14"/><path d="M5 7l-3 7a3 3 0 0 0 6 0z"/><path d="M19 7l-3 7a3 3 0 0 0 6 0z"/>',
    "run": '<circle cx="13" cy="4" r="2"/><path d="M4 22l4-7 3 2 2-5 3 3 4 0"/><path d="M9 8l4-1 3 3"/>',
    "calendar": '<rect x="3" y="4" width="18" height="18" rx="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/>',
    "stethoscope": '<path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6 6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"/><path d="M8 15v1a6 6 0 0 0 6 6 6 6 0 0 0 6-6v-4"/><circle cx="20" cy="10" r="2"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "arrow": '<line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>',
}

CATEGORY_ICON = {
    "Blood Pressure": "pulse", "Tobacco Use": "wind", "Alcohol Intake": "coffee",
    "Dietary Pattern": "apple", "Weight-Related Habits": "scale", "Physical Activity": "run",
    "General Preventive Health": "calendar", "Clinical Next Steps": "stethoscope",
}

STEPS = [("profile", "Profile"), ("lifestyle", "Lifestyle"), ("health", "Health"),
         ("review", "Review"), ("result", "Result"), ("recs", "Guidance")]


def icon(name: str) -> str:
    return (f'<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">{_ICONS[name]}</svg>')


def icon_box(name: str, tone: str = "") -> str:
    return f'<span class="ed-ico {tone}">{icon(name)}</span>'


def brand_html() -> str:
    return f'<div class="ed-brand"><span class="ed-logo">{icon("droplet")}</span>EarlyDia</div>'


def stepper_html(current_key: str) -> str:
    keys = [k for k, _ in STEPS]
    cur = keys.index(current_key) if current_key in keys else -1
    items = []
    for i, (key, label) in enumerate(STEPS):
        cls = "active" if i == cur else ("done" if i < cur else "")
        num = icon("check") if i < cur else str(i + 1)
        aria = ' aria-current="step"' if i == cur else ""
        items.append(
            f'<div class="ed-step {cls}"{aria} title="Step {i + 1}: {_e(label)}">'
            f'<span class="ed-num" aria-hidden="true">{num}</span>'
            f'<span class="ed-lbl"><span class="ed-sr">Step {i + 1}: </span>{_e(label)}</span></div>')
    return f'<nav class="ed-stepper" aria-label="Progress">{"".join(items)}</nav>'


def page_head(eyebrow: str, title: str, sub: str = "") -> str:
    sub_html = f'<p class="ed-sub">{_e(sub)}</p>' if sub else ""
    return (f'<div class="ed-eyebrow">{_e(eyebrow)}</div><h1 class="ed-title">{_e(title)}</h1>{sub_html}')


def section_title(text: str, icon_name: str = "") -> str:
    ico = icon_box(icon_name) if icon_name else ""
    return f'<div class="ed-section-title">{ico}{_e(text)}</div>'


def card(title: str, body: str, icon_name: str = "", tone: str = "") -> str:
    ico = icon_box(icon_name, tone) if icon_name else ""
    return f'<div class="ed-card">{ico}<h3>{_e(title)}</h3><p>{_e(body)}</p></div>'


def checklist(items, kind: str) -> str:
    """kind: 'yes' (green check) or 'no' (red cross)."""
    ic = icon("check" if kind == "yes" else "x")
    lis = "".join(f"<li>{ic}<span>{_e(t)}</span></li>" for t in items)
    return f'<ul class="ed-list {kind}">{lis}</ul>'


def note(html_text: str) -> str:
    """Disclaimer-style note. `html_text` may contain simple <b> markup written by us."""
    return f'<div class="ed-note">{html_text}</div>'


def callout(text: str, flagged: bool, html: bool = False) -> str:
    body = text if html else _e(text)
    return f'<div class="{"ed-callout-flag" if flagged else "ed-callout-ok"}">{body}</div>'


def kv_grid(pairs, full_keys=()) -> str:
    cells = []
    for k, v in pairs:
        cls = ' class="full"' if k in full_keys else ""
        cells.append(f'<div{cls}><div class="k">{_e(k)}</div><div class="v">{_e(v)}</div></div>')
    return f'<div class="ed-kv">{"".join(cells)}</div>'


def factor_grid(factors) -> str:
    cells = "".join(f'<div class="ed-factor"><div class="n">{_e(f["factor"])}</div>'
                    f'<div class="v">{_e(f["value"])}</div></div>' for f in factors)
    return f'<div class="ed-factors">{cells}</div>'


def rec_card(item) -> str:
    ico = icon_box(CATEGORY_ICON.get(item.category, "info"))
    lis = "".join(f"<li>{_e(a)}</li>" for a in item.action_items)
    return (f'<div class="ed-rec"><div class="ed-rec-head"><div class="ed-rec-title">{ico}'
            f'<span>{_e(item.category)}</span></div>'
            f'<span class="ed-tag ed-tag-{_e(item.tag_type)}">{_e(item.status_tag)}</span></div>'
            f'<div style="font-weight:700;margin:.15rem 0 .1rem 0">{_e(item.title)}</div>'
            f'<div class="sum">{_e(item.summary)}</div><ul>{lis}</ul></div>')


def hero_html() -> str:
    return ('<div class="ed-hero"><span class="ed-pill">' + icon("clock") + '&nbsp;About 3 minutes</span>'
            '<h1>EarlyDia</h1><h2>Early diabetes risk screening</h2>'
            '<p>A short, private questionnaire that uses information about your profile, lifestyle and '
            'health-related factors to provide a personalised diabetes-related screening estimate.</p></div>')


def how_step(n: int, title: str, body: str) -> str:
    return (f'<div class="ed-card"><span class="ed-how-num">{n}</span><h3>{_e(title)}</h3>'
            f'<p>{_e(body)}</p></div>')


def details_html(title: str, inner_html: str) -> str:
    """Native <details> disclosure (no Streamlit expander, so it renders identically everywhere)."""
    return (f'<details class="ed-details"><summary>{_e(title)}</summary>'
            f'<div class="ed-details-body">{inner_html}</div></details>')


def diet_table(rows) -> str:
    cells = "".join(f'<div class="ed-diet-row"><span>{_e(k)}</span><b>{_e(v)}</b></div>' for k, v in rows)
    return f'<div class="ed-diet">{cells}</div>'


def result_card(prob_pct: float, th_pct: float, flagged: bool, gauge_svg: str) -> str:
    state = "flag" if flagged else "ok"
    badge_icon = icon("alert" if flagged else "check")
    label = "Flagged for Follow-up" if flagged else "Below Screening Threshold"
    legend = ('<div class="ed-legend"><span><i style="background:var(--ed-ok);opacity:.6"></i>Below threshold</span>'
              '<span><i style="background:var(--ed-flag);opacity:.4"></i>At or above threshold</span></div>')
    return (f'<div class="ed-result {state}"><div class="ed-kicker">Estimated diabetes-related screening probability</div>'
            f'{gauge_svg}{legend}'
            f'<div class="ed-badge {state}">{badge_icon}Result: {label}</div>'
            f'<div class="ed-kpis">'
            f'<div class="ed-kpi"><div class="l">Screening probability</div><div class="v">{prob_pct:.2f}%</div></div>'
            f'<div class="ed-kpi"><div class="l">Screening threshold</div><div class="v">{th_pct:.2f}%</div></div>'
            f'</div></div>')
