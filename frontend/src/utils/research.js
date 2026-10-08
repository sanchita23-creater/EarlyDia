// Pure helpers for the Research page (no ML logic: only reshaping already-generated report rows).
export const num = (v) => (v === "" || v === undefined || v === null ? null : Number(v));
export const pct = (v, d = 1) => (num(v) === null || Number.isNaN(num(v)) ? "n/a" : `${(num(v) * 100).toFixed(d)}%`);
export const fix = (v, d = 3) => (num(v) === null || Number.isNaN(num(v)) ? "n/a" : num(v).toFixed(d));

export function reliabilityPoints(rows, sex) {
  return rows.filter((r) => r.sex === sex && r.row_type === "bin")
    .map((r) => ({ x: num(r.mean_predicted), y: num(r.observed_fraction), n: num(r.n) }));
}
export function calibrationSummary(rows, sex) {
  return rows.find((r) => r.sex === sex && r.row_type === "summary") || null;
}
export function decisionCurve(rows, sex) {
  const c = rows.filter((r) => r.sex === sex && r.kind === "curve");
  return {
    model: c.map((r) => ({ x: num(r.threshold), y: num(r.net_benefit_model) })),
    all: c.map((r) => ({ x: num(r.threshold), y: num(r.net_benefit_treat_all) })),
    source: c[0]?.source || "",
    exact: rows.find((r) => r.sex === sex && r.kind === "exact_at_locked_threshold") || null,
  };
}
export function subgroupRows(rows) {
  return rows.filter((r) => r.subgroup !== "overall");
}
export function topFeatures(rows, sex, k = 12) {
  return rows.filter((r) => r.sex === sex).slice(0, k)
    .map((r) => ({ label: r.label, value: num(r.importance_mean), sd: num(r.importance_std) }));
}
