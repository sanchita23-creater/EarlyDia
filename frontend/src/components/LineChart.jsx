// Small dependency-free SVG charts (axes, lines, reference marks). Presentational only.
export function LineChart({ series, xLabel, yLabel, vline, diag, height = 240, title }) {
  const W = 420, H = height, m = { l: 52, r: 12, t: 12, b: 38 };
  const pts = series.flatMap((s) => s.data);
  if (!pts.length) return <p className="muted">No data.</p>;
  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y).concat(diag ? [0] : []);
  const x0 = Math.min(...xs, 0), x1 = Math.max(...xs), y0 = Math.min(...ys, 0), y1 = Math.max(...ys);
  const sx = (v) => m.l + ((v - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
  const sy = (v) => H - m.b - ((v - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);
  const path = (d) => d.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(" ");
  const ticks = (a, b) => Array.from({ length: 5 }, (_, i) => a + ((b - a) * i) / 4);
  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title || yLabel}>
        {ticks(y0, y1).map((t) => (
          <g key={`y${t}`}><line x1={m.l} x2={W - m.r} y1={sy(t)} y2={sy(t)} className="grid" />
            <text x={m.l - 6} y={sy(t) + 4} textAnchor="end" className="tick">{t.toFixed(3)}</text></g>
        ))}
        {ticks(x0, x1).map((t) => (
          <text key={`x${t}`} x={sx(t)} y={H - m.b + 16} textAnchor="middle" className="tick">{t.toFixed(3)}</text>
        ))}
        {diag && <line x1={sx(x0)} y1={sy(x0)} x2={sx(Math.min(x1, y1))} y2={sy(Math.min(x1, y1))} className="ref" />}
        {y0 < 0 && <line x1={m.l} x2={W - m.r} y1={sy(0)} y2={sy(0)} className="ref" />}
        {vline != null && <line x1={sx(vline)} x2={sx(vline)} y1={m.t} y2={H - m.b} className="vline" />}
        {series.map((s, i) => (
          <g key={s.name}>
            <path d={path(s.data)} className={`line line-${i}`} fill="none" />
            {s.dots && s.data.map((p, j) => <circle key={j} cx={sx(p.x)} cy={sy(p.y)} r="3.5" className={`dot dot-${i}`} />)}
          </g>
        ))}
        <text x={(W + m.l) / 2} y={H - 4} textAnchor="middle" className="axis">{xLabel}</text>
        <text transform={`translate(12 ${H / 2}) rotate(-90)`} textAnchor="middle" className="axis">{yLabel}</text>
      </svg>
      <figcaption className="legend">{series.map((s, i) => <span key={s.name}><i className={`sw sw-${i}`} />{s.name}</span>)}</figcaption>
    </figure>
  );
}

export function BarList({ items, valueLabel }) {
  const max = Math.max(...items.map((i) => i.value), 1e-9);
  return (
    <ul className="barlist" aria-label={valueLabel}>
      {items.map((i) => (
        <li key={i.label}><span className="bl-label">{i.label}</span>
          <span className="bl-track"><span className="bl-fill" style={{ width: `${(i.value / max) * 100}%` }} /></span>
          <span className="bl-val">{i.value.toFixed(3)}</span></li>
      ))}
    </ul>
  );
}
