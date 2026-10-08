import { describe, it, expect } from "vitest";
import { computeBmi, buildPayload, emptyAnswers, validateHealth } from "./answers";

const cfg = { bp_systolic_range: [70, 270], bp_diastolic_range: [40, 150],
  height_range_cm: [120, 220], weight_range_kg: [25, 200] };

describe("answers utils", () => {
  it("computes BMI from height/weight", () => {
    expect(computeBmi("160", "58.1")).toBe(22.7);
    expect(computeBmi("", "58")).toBeNull();
  });
  it("men payload never carries height/weight", () => {
    const a = { ...emptyAnswers(), sex: "men", heightCm: "170", weightKg: "70", bpSys: "120", bpDia: "80" };
    const p = buildPayload(a);
    expect(p.height_cm).toBeNull();
    expect(p.weight_kg).toBeNull();
  });
  it("requires blood pressure", () => {
    const a = { ...emptyAnswers(), sex: "men" };
    expect(validateHealth(a, cfg).length).toBe(2);
  });
});

import { idrsComplete, validateIdrs } from "./answers";
const cfgIdrs = { idrs: { waist_range_cm: [50, 200] } };
describe("IDRS (optional, all-or-none)", () => {
  it("is skipped silently when untouched and omitted from the payload", () => {
    const a = { ...emptyAnswers(), sex: "men" };
    expect(validateIdrs(a, cfgIdrs)).toEqual([]);
    const p = buildPayload(a);
    expect(p.waist_cm).toBeNull(); expect(p.physical_activity).toBeNull(); expect(p.family_history).toBeNull();
  });
  it("flags a partial answer", () => {
    const a = { ...emptyAnswers(), sex: "women", waistCm: "85" };
    expect(validateIdrs(a, cfgIdrs).length).toBe(1);
  });
  it("sends all three when complete", () => {
    const a = { ...emptyAnswers(), sex: "women", waistCm: "85", activity: "sedentary", familyHistory: "none" };
    expect(idrsComplete(a)).toBe(true);
    expect(validateIdrs(a, cfgIdrs)).toEqual([]);
    expect(buildPayload(a)).toMatchObject({ waist_cm: 85, physical_activity: "sedentary", family_history: "none" });
  });
  it("rejects an out-of-range waist", () => {
    const a = { ...emptyAnswers(), sex: "men", waistCm: "10", activity: "sedentary", familyHistory: "none" };
    expect(validateIdrs(a, cfgIdrs).length).toBe(1);
  });
});
