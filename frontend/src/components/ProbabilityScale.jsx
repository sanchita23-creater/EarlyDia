// EarlyDia - Semicircular screening probability gauge (inline SVG, scale 0-100%).
// Pure presentation component rendering colorful risk zones: Green (Low), Yellow (Medium), Orange (Elevated), Red (High).
import React from "react";

export const SCALE_MAX = 100;

export const formatPercent = (v) => {
  const num = Number(v);
  if (isNaN(num)) return "0.0%";
  if (num > 0 && num < 0.05) return "<0.1%";
  return `${num.toFixed(1)}%`;
};

// Generates SVG path for a circular band (upper half-circle annulus) between fractions f1 and f2 on [0, 1].
// Clockwise sweep from left (f=0) through top (f=0.5) to right (f=1).
export const makeArcBand = (cx, cy, rOut, rIn, f1, f2) => {
  const pt = (rad, f) => [
    cx - rad * Math.cos(f * Math.PI),
    cy - rad * Math.sin(f * Math.PI),
  ];
  const [x1o, y1o] = pt(rOut, f1);
  const [x2o, y2o] = pt(rOut, f2);
  const [x2i, y2i] = pt(rIn, f2);
  const [x1i, y1i] = pt(rIn, f1);

  return (
    `M ${x1o.toFixed(2)} ${y1o.toFixed(2)} ` +
    `A ${rOut} ${rOut} 0 0 1 ${x2o.toFixed(2)} ${y2o.toFixed(2)} ` +
    `L ${x2i.toFixed(2)} ${y2i.toFixed(2)} ` +
    `A ${rIn} ${rIn} 0 0 0 ${x1i.toFixed(2)} ${y1i.toFixed(2)} Z`
  );
};

export const getRiskTier = (percent, cutoff) => {
  const p = Number(percent) || 0;
  if (p >= 60) return { key: "very-high", label: "Very High", color: "#EF4444" };
  if (p >= 40) return { key: "high", label: "High", color: "#FB923C" };
  if (p >= 20) return { key: "moderate", label: "Moderate", color: "#FACC15" };
  return { key: "low", label: "Low", color: "#10B981" };
};

export default function ProbabilityScale({ percent = 0, cutoffPercent = 21.2, flagged = false }) {
  const numPercent = Number(percent) || 0;
  const numCutoff = Number(cutoffPercent) || 21.2;

  const cutoffFrac = Math.min(1, Math.max(0, numCutoff / SCALE_MAX));
  const clampedPercent = Math.min(SCALE_MAX, Math.max(0, numPercent));
  const needleFrac = clampedPercent / SCALE_MAX;

  // Gauge dimensions
  const cx = 240;
  const cy = 195;
  const rOut = 145;
  const rIn = 105;

  // Screening cutoff marker tick coordinates
  const cutCos = Math.cos(cutoffFrac * Math.PI);
  const cutSin = Math.sin(cutoffFrac * Math.PI);
  const tx1 = cx - (rIn - 6) * cutCos;
  const ty1 = cy - (rIn - 6) * cutSin;
  const tx2 = cx - (rOut + 8) * cutCos;
  const ty2 = cy - (rOut + 8) * cutSin;
  const leaderX = cx - (rOut + 28) * cutCos;
  const leaderY = cy - (rOut + 28) * cutSin;

  // Needle tip and hub base coordinates
  const needleAngle = needleFrac * Math.PI;
  const nCos = Math.cos(needleAngle);
  const nSin = Math.sin(needleAngle);
  const tipX = cx - (rOut - 10) * nCos;
  const tipY = cy - (rOut - 10) * nSin;

  const baseW = 6;
  const b1x = cx - baseW * nSin;
  const b1y = cy + baseW * nCos;
  const b2x = cx + baseW * nSin;
  const b2y = cy - baseW * nCos;

  const valueLabel = formatPercent(numPercent);
  const cutoffLabel = formatPercent(numCutoff);
  const isClamped = numPercent > SCALE_MAX;
  const tier = getRiskTier(numPercent, numCutoff);

  const ariaText = `Your estimated screening likelihood is ${valueLabel}. ` +
    `Screening cutoff is ${cutoffLabel}. ` +
    `Risk Tier: ${tier.label}. ` +
    `Status: ${flagged ? "At or above cutoff" : "Below cutoff"}. ` +
    `This does not mean you have diabetes.`;

  return (
    <div className="gauge-container" role="img" aria-label={ariaText}>
      <svg
        className="gauge-svg"
        viewBox="0 0 480 250"
        xmlns="http://www.w3.org/2000/svg"
        aria-hidden="true"
      >
        <title>{`Screening probability ${valueLabel} vs cutoff ${cutoffLabel}`}</title>

        {/* Outer track background */}
        <path
          d={makeArcBand(cx, cy, rOut, rIn, 0, 1)}
          className="gauge-track"
        />

        {/* Colorful Risk Zones along the semicircle arc */}
        {/* Low: 0-20% (Green) */}
        <path
          d={makeArcBand(cx, cy, rOut, rIn, 0, 0.20)}
          className="gauge-zone-band gauge-zone-green gauge-zone-below"
        />
        {/* Medium: 20-40% (Yellow) */}
        <path
          d={makeArcBand(cx, cy, rOut, rIn, 0.20, 0.40)}
          className="gauge-zone-band gauge-zone-yellow"
        />
        {/* Elevated: 40-60% (Orange) */}
        <path
          d={makeArcBand(cx, cy, rOut, rIn, 0.40, 0.60)}
          className="gauge-zone-band gauge-zone-orange"
        />
        {/* High: 60-100% (Red) */}
        <path
          d={makeArcBand(cx, cy, rOut, rIn, 0.60, 1.0)}
          className="gauge-zone-band gauge-zone-red gauge-zone-above"
        />

        {/* Axis ticks: 0%, 20%, 40%, 60%, 80%, 100% */}
        {[0, 20, 40, 60, 80, 100].map((val) => {
          const f = val / SCALE_MAX;
          const cos = Math.cos(f * Math.PI);
          const sin = Math.sin(f * Math.PI);
          const xInner = cx - (rIn + 2) * cos;
          const yInner = cy - (rIn + 2) * sin;
          const xOuter = cx - (rIn + 10) * cos;
          const yOuter = cy - (rIn + 10) * sin;
          const xText = cx - (rOut + 14) * cos;
          const yText = cy - (rOut + 14) * sin;
          const anchor = f < 0.1 ? "end" : f > 0.9 ? "start" : "middle";

          return (
            <g key={val} className="gauge-tick-group">
              <line
                x1={xInner.toFixed(2)}
                y1={yInner.toFixed(2)}
                x2={xOuter.toFixed(2)}
                y2={yOuter.toFixed(2)}
                className="gauge-tick-line"
              />
              <text
                x={xText.toFixed(2)}
                y={(yText + (f < 0.1 || f > 0.9 ? 4 : 0)).toFixed(2)}
                textAnchor={anchor}
                className="gauge-axis-label"
              >
                {val}%
              </text>
            </g>
          );
        })}

        {/* Clearly visible tick at screening cutoff */}
        <line
          x1={tx1.toFixed(2)}
          y1={ty1.toFixed(2)}
          x2={tx2.toFixed(2)}
          y2={ty2.toFixed(2)}
          className="gauge-cutoff-tick"
        />
        <line
          x1={tx2.toFixed(2)}
          y1={ty2.toFixed(2)}
          x2={leaderX.toFixed(2)}
          y2={leaderY.toFixed(2)}
          className="gauge-cutoff-leader"
        />

        {/* Dynamic cutoff label */}
        <text
          x={(leaderX - 6).toFixed(2)}
          y={(leaderY - 4).toFixed(2)}
          textAnchor={cutoffFrac < 0.35 ? "end" : "middle"}
          className="gauge-cutoff-label"
        >
          {`Screening cutoff ${cutoffLabel}`}
        </text>

        {/* Needle indicator styled to match active category color */}
        <polygon
          points={`${b1x.toFixed(2)},${b1y.toFixed(2)} ${tipX.toFixed(2)},${tipY.toFixed(2)} ${b2x.toFixed(2)},${b2y.toFixed(2)}`}
          className="gauge-needle"
          style={{ fill: tier.color }}
        />
        <circle cx={cx} cy={cy} r="10" className="gauge-hub-outer" style={{ fill: tier.color }} />
        <circle cx={cx} cy={cy} r="4" className="gauge-hub-inner" />

        {/* Centered probability number */}
        <text x={cx} y={cy - 40} textAnchor="middle" className="gauge-readout-value">
          {valueLabel}
        </text>
        <text x={cx} y={cy - 18} textAnchor="middle" className="gauge-readout-sub">
          Screening probability
        </text>
        <text
          x={cx}
          y={cy + 22}
          textAnchor="middle"
          className={`gauge-tier-pill gauge-tier-${tier.key}`}
          style={{ fill: tier.color }}
        >
          {`${tier.label}${flagged ? " · Flagged" : ""}`}
        </text>

        {isClamped && (
          <text x={cx} y={cy + 38} textAnchor="middle" className="gauge-clamped-note">
            (Scale max 100%; needle clamped)
          </text>
        )}
      </svg>

      {/* Colorful Risk zones & Cutoff legend */}
      <div className="gauge-legend">
        <span className="gauge-legend-item">
          <span className="gauge-legend-swatch swatch-green" />
          <span>Low (&lt;20%)</span>
        </span>
        <span className="gauge-legend-item">
          <span className="gauge-legend-swatch swatch-yellow" />
          <span>Moderate (20–40%)</span>
        </span>
        <span className="gauge-legend-item">
          <span className="gauge-legend-swatch swatch-orange" />
          <span>High (40–60%)</span>
        </span>
        <span className="gauge-legend-item">
          <span className="gauge-legend-swatch swatch-red" />
          <span>Very High (≥60%)</span>
        </span>
        <span className="gauge-legend-item">
          <span className="gauge-legend-swatch swatch-cutoff" />
          <span>Cutoff (≥{cutoffLabel})</span>
        </span>
      </div>
    </div>
  );
}
