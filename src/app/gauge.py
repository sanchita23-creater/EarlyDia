"""
EarlyDia - semicircular screening gauge (inline SVG, no external assets).

The dial spans 0-100 % of calibrated screening probability. Because real probabilities for
this population are small (the locked thresholds are 2.6 % and 2.7 %), a *linear* 0-100 %
dial would squeeze both the needle and the threshold marker into the first few degrees and
make them unreadable. The scale is therefore a square-root scale (labelled ticks at
0, 1, 5, 10, 25, 50, 100 %), which keeps the full 0-100 % range while spreading out the low end.

The gauge shows ONE number: the model's calibrated screening probability. It deliberately
carries no "accuracy", "confidence" or "certainty" wording, and the colouring only separates
"below the sex-specific threshold" from "at or above it".
"""
import math

from src.app.theme import PALETTES

TICKS_PCT = (0, 1, 5, 10, 25, 50, 100)


def pct_to_fraction(pct: float) -> float:
    """Square-root scale on [0, 100] %, returning a position in [0, 1] along the arc."""
    pct = min(100.0, max(0.0, float(pct)))
    return math.sqrt(pct / 100.0)


def _pt(cx, cy, r, frac):
    ang = math.pi + frac * math.pi
    return cx + r * math.cos(ang), cy + r * math.sin(ang)


def _band(cx, cy, r_out, r_in, f1, f2):
    x1o, y1o = _pt(cx, cy, r_out, f1)
    x2o, y2o = _pt(cx, cy, r_out, f2)
    x2i, y2i = _pt(cx, cy, r_in, f2)
    x1i, y1i = _pt(cx, cy, r_in, f1)
    return (f"M {x1o:.2f} {y1o:.2f} A {r_out} {r_out} 0 0 1 {x2o:.2f} {y2o:.2f} "
            f"L {x2i:.2f} {y2i:.2f} A {r_in} {r_in} 0 0 0 {x1i:.2f} {y1i:.2f} Z")


def render_semicircular_gauge(prob_pct: float, th_pct: float, is_flagged: bool,
                              dark_mode: bool) -> str:
    """Return an SVG string (single block, no blank lines, safe inside st.markdown)."""
    pal = PALETTES["dark" if dark_mode else "light"]
    w, h = 560, 366
    cx, cy = 280, 255
    r_out, r_in = 200, 160

    f_p = pct_to_fraction(prob_pct)
    f_t = pct_to_fraction(th_pct)
    c_state = pal["flag"] if is_flagged else pal["ok"]

    parts = []
    parts.append(f'<svg viewBox="-30 0 {w + 60} {h}" width="100%" role="img" xmlns="http://www.w3.org/2000/svg" '
                 f'style="display:block;max-width:620px;margin:0 auto;font-family:Inter,Segoe UI,-apple-system,sans-serif" '
                 f'aria-label="Screening probability {prob_pct:.2f} percent; sex-specific screening threshold {th_pct:.2f} percent; '
                 f'{"at or above" if is_flagged else "below"} the threshold. Non-linear scale from 0 to 100 percent.">')
    parts.append(f'<title>Estimated screening probability {prob_pct:.2f}% vs threshold {th_pct:.2f}%</title>')
    # track + zones
    parts.append(f'<path d="{_band(cx, cy, r_out, r_in, 0, 1)}" fill="{pal["track"]}" stroke="{pal["border"]}" stroke-width="1"/>')
    parts.append(f'<path d="{_band(cx, cy, r_out, r_in, 0, f_t)}" fill="{pal["ok"]}" fill-opacity="0.55"/>')
    parts.append(f'<path d="{_band(cx, cy, r_out, r_in, f_t, 1)}" fill="{pal["flag"]}" fill-opacity="0.28"/>')
    # ticks + labels (outside the band)
    for t in TICKS_PCT:
        f = pct_to_fraction(t)
        x1, y1 = _pt(cx, cy, r_out + 2, f)
        x2, y2 = _pt(cx, cy, r_out + 10, f)
        lx, ly = _pt(cx, cy, r_out + 24, f)
        parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{pal["muted"]}" stroke-width="1.5"/>')
        anchor = "end" if f < 0.04 else ("start" if f > 0.96 else "middle")
        parts.append(f'<text x="{lx:.1f}" y="{ly + 4:.1f}" text-anchor="{anchor}" fill="{pal["muted"]}" font-size="18" font-weight="700">{t}%</text>')
    # threshold marker: bold notch through the band + labelled leader OUTSIDE the dial,
    # so the label can never collide with the needle even when the two are almost equal
    tx1, ty1 = _pt(cx, cy, r_in - 8, f_t)
    tx2, ty2 = _pt(cx, cy, r_out + 8, f_t)
    parts.append(f'<line x1="{tx1:.2f}" y1="{ty1:.2f}" x2="{tx2:.2f}" y2="{ty2:.2f}" stroke="{pal["text"]}" stroke-width="3.5" stroke-linecap="round"/>')
    lx1, ly1 = _pt(cx, cy, r_out + 8, f_t)
    lx2, ly2 = _pt(cx, cy, r_out + 46, f_t)
    parts.append(f'<line x1="{lx1:.1f}" y1="{ly1:.1f}" x2="{lx2:.1f}" y2="{ly2:.1f}" stroke="{pal["text"]}" stroke-width="1.2" stroke-dasharray="3 3"/>')
    anchor_t = "end" if f_t < 0.5 else "start"
    dx = -8 if f_t < 0.5 else 8
    parts.append(f'<text x="{lx2 + dx:.1f}" y="{ly2 - 2:.1f}" text-anchor="{anchor_t}" fill="{pal["text"]}" font-size="17" font-weight="700">Threshold</text>')
    parts.append(f'<text x="{lx2 + dx:.1f}" y="{ly2 + 17:.1f}" text-anchor="{anchor_t}" fill="{pal["text"]}" font-size="17" font-weight="800">{th_pct:.2f}%</text>')
    # needle (tapered) + hub
    nx, ny = _pt(cx, cy, r_out - 22, f_p)
    ang = math.pi + f_p * math.pi
    bx1, by1 = cx + 9 * math.cos(ang + math.pi / 2), cy + 9 * math.sin(ang + math.pi / 2)
    bx2, by2 = cx + 9 * math.cos(ang - math.pi / 2), cy + 9 * math.sin(ang - math.pi / 2)
    parts.append(f'<polygon points="{bx1:.2f},{by1:.2f} {nx:.2f},{ny:.2f} {bx2:.2f},{by2:.2f}" fill="{c_state}" stroke="{pal["surface"]}" stroke-width="1.5" stroke-linejoin="round"/>')
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="15" fill="{c_state}" stroke="{pal["surface"]}" stroke-width="3"/>')
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="5" fill="{pal["surface"]}"/>')
    # readout
    parts.append(f'<text x="{cx}" y="{cy + 62}" text-anchor="middle" fill="{pal["text"]}" font-size="52" font-weight="800" letter-spacing="-0.5">{prob_pct:.2f}%</text>')
    parts.append(f'<text x="{cx}" y="{cy + 88}" text-anchor="middle" fill="{pal["muted"]}" font-size="18" font-weight="600">Screening probability</text>')
    parts.append('</svg>')
    return "".join(parts)
