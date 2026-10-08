import { describe, it, expect } from "vitest";
import { pct, fix, reliabilityPoints, decisionCurve, topFeatures } from "./research";

describe("research helpers", () => {
  it("formats numbers and missing values", () => {
    expect(pct("0.0304")).toBe("3.0%");
    expect(pct("")).toBe("n/a");
    expect(fix("0.68507")).toBe("0.685");
  });
  it("extracts reliability bins per sex", () => {
    const rows = [
      { sex: "women", row_type: "bin", mean_predicted: "0.01", observed_fraction: "0.02", n: "10" },
      { sex: "men", row_type: "bin", mean_predicted: "0.03", observed_fraction: "0.04", n: "5" },
      { sex: "women", row_type: "summary", mean_predicted: "0.03" },
    ];
    expect(reliabilityPoints(rows, "women")).toEqual([{ x: 0.01, y: 0.02, n: 10 }]);
  });
  it("splits decision-curve rows", () => {
    const rows = [
      { sex: "women", kind: "curve", threshold: "0.02", net_benefit_model: "0.01", net_benefit_treat_all: "0.005", source: "APPROX" },
      { sex: "women", kind: "exact_at_locked_threshold", threshold: "0.026", net_benefit_model: "0.0097" },
    ];
    const d = decisionCurve(rows, "women");
    expect(d.model).toHaveLength(1);
    expect(d.exact.threshold).toBe("0.026");
  });
  it("limits top features", () => {
    const rows = Array.from({ length: 20 }, (_, i) => ({ sex: "men", label: `f${i}`, importance_mean: String(i), importance_std: "0" }));
    expect(topFeatures(rows, "men", 5)).toHaveLength(5);
  });
});
