import { describe, it, expect } from "vitest";
import React from "react";
import { renderToString } from "react-dom/server";
import ProbabilityScale, { formatPercent, makeArcBand, SCALE_MAX } from "./ProbabilityScale";

describe("ProbabilityScale (semicircular gauge component)", () => {
  it("formats percentages correctly with rounding and low values", () => {
    expect(formatPercent(3.6)).toBe("3.6%");
    expect(formatPercent(0.03)).toBe("<0.1%");
    expect(formatPercent(15)).toBe("15.0%");
    expect(formatPercent(24.5)).toBe("24.5%");
  });

  it("generates valid SVG arc path data for upper semicircle", () => {
    const path = makeArcBand(240, 195, 145, 105, 0, 0.5);
    expect(path).toContain("M ");
    expect(path).toContain("A 145 145 0 0 1");
    expect(path).toContain("A 105 105 0 0 0");
    expect(path).toContain("Z");
  });

  it("renders inline SVG with semicircular gauge elements", () => {
    const html = renderToString(
      <ProbabilityScale percent={3.6} cutoffPercent={2.4} flagged={true} />
    );
    expect(html).toContain("<svg");
    expect(html).toContain("gauge-svg");
    expect(html).toContain("viewBox=\"0 0 480 250\"");
    expect(html).toContain("gauge-track");
    expect(html).toContain("gauge-zone-below");
    expect(html).toContain("gauge-zone-above");
    expect(html).toContain("gauge-needle");
    expect(html).toContain("3.6%");
    expect(html).toContain("Screening cutoff 2.4%");
  });

  it("dynamically formats cutoff marker without hardcoding", () => {
    const html29 = renderToString(
      <ProbabilityScale percent={2.1} cutoffPercent={2.9} flagged={false} />
    );
    expect(html29).toContain("Screening cutoff 2.9%");

    const html26 = renderToString(
      <ProbabilityScale percent={2.8} cutoffPercent={2.6} flagged={true} />
    );
    expect(html26).toContain("Screening cutoff 2.6%");
  });

  it("clamps needle when probability exceeds 15% while displaying real number", () => {
    const html = renderToString(
      <ProbabilityScale percent={18.5} cutoffPercent={2.4} flagged={true} />
    );
    expect(html).toContain("18.5%");
    expect(html).toContain("needle clamped");
  });

  it("includes accessible aria-label with cutoff, value, and disclaimer", () => {
    const html = renderToString(
      <ProbabilityScale percent={3.6} cutoffPercent={2.4} flagged={true} />
    );
    expect(html).toContain("aria-label=");
    expect(html).toContain("Your estimated screening likelihood is 3.6%");
    expect(html).toContain("Screening cutoff is 2.4%");
    expect(html).toContain("At or above cutoff");
    expect(html).toContain("This does not mean you have diabetes");
  });

  it("defines scale maximum as 15%", () => {
    expect(SCALE_MAX).toBe(15);
  });
});
