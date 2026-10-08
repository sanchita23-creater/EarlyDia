// Simple LINEAR probability visual (not a gauge): 0% .... max%, with the screening cutoff
// and the user's result marked. Pure presentation - no model logic.
const niceMax = (pct, cutoff) => {
  const top = Math.max(pct, cutoff) * 1.15;
  if (top <= 10) return 10;
  return Math.ceil(top / 5) * 5;
};

export default function ProbabilityScale({ percent, cutoffPercent, flagged }) {
  const max = niceMax(percent, cutoffPercent);
  const pos = (v) => Math.min(100, Math.max(0, (v / max) * 100));
  const label = (v) => `${v.toFixed(1)}%`;
  return (
    <div className="scale" role="img"
      aria-label={`Your estimated screening likelihood is ${label(percent)}. The screening cutoff is ${label(cutoffPercent)}.`}>
      <div className="scale-marker-label" style={{ "--p": `${pos(percent)}%` }}>
        <span>Your result</span><strong>{label(percent)}</strong>
      </div>
      <div className="scale-track">
        <div className="scale-zone-low" style={{ width: `${pos(cutoffPercent)}%` }} />
        <div className="scale-zone-high" style={{ left: `${pos(cutoffPercent)}%` }} />
        <div className="scale-cutoff" style={{ left: `${pos(cutoffPercent)}%` }} />
        <div className={`scale-dot ${flagged ? "is-flagged" : "is-ok"}`} style={{ left: `${pos(percent)}%` }} />
      </div>
      <div className="scale-axis">
        <span>0%</span>
        <span>{max}%</span>
      </div>
      <div className="scale-cutoff-label" style={{ left: `${pos(cutoffPercent)}%` }}>
        <span>Screening cutoff</span><strong>{label(cutoffPercent)}</strong>
      </div>
    </div>
  );
}
