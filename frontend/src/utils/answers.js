// Answer state shape, light client-side checks (for friendly UX only - the server validates
// again) and conversion of answers to the API payload.
export const emptyAnswers = () => ({
  sex: null, age: "", state: null, education: null, residence: null,
  tobaccoUse: null, tobaccoProducts: [], cigFreq: null,
  drinks: null, alcFreq: null, alcYears: "", alcTypes: [],
  diet: {},
  heightCm: "", weightKg: "",
  bpSys: "", bpDia: "",
  // Optional IDRS comparison (independent published score; NOT used by the ML model)
  waistCm: "", activity: null, familyHistory: null,
});

const num = (v) => (v === "" || v === null || v === undefined ? null : Number(v));

export function computeBmi(heightCm, weightKg) {
  const h = num(heightCm), w = num(weightKg);
  if (!h || !w || h <= 0 || w <= 0) return null;
  return Math.round((w / ((h / 100) ** 2)) * 10) / 10; // display only; the server recomputes
}

export function bmiBand(bmi) {
  if (bmi < 18.5) return "Below the typical range";
  if (bmi < 23) return "Typical range";
  if (bmi < 27.5) return "Above the typical range";
  return "Well above the typical range";
}

export function validateProfile(a, cfg) {
  const e = [];
  const [lo, hi] = cfg?.age_range ?? [18, 40];
  if (!a.sex) e.push("Select your sex.");
  const age = num(a.age);
  if (age === null) e.push(`Enter your age (${lo}-${hi}).`);
  else if (!Number.isInteger(age) || age < lo || age > hi) e.push(`EarlyDia is designed for ages ${lo}-${hi}.`);
  if (a.state === null || a.state === "" || a.state === undefined) e.push("Select your State / UT.");
  if (a.education === null) e.push("Select your education level.");
  if (a.residence === null) e.push("Select where you live.");
  return e;
}

export function validateLifestyle(a, cfg) {
  const e = [];
  if (!a.sex) return ["Complete the Profile step first."];
  const ps = cfg.per_sex[a.sex];
  if (a.tobaccoUse === null) e.push("Tell us whether you currently use tobacco.");
  else if (a.tobaccoUse) {
    if (a.tobaccoProducts.length === 0) e.push("Select at least one tobacco product.");
    else if (ps.asks_tobacco_frequency && a.tobaccoProducts.includes(ps.cigarette_id) && a.cigFreq === null)
      e.push("Tell us how often you smoke cigarettes.");
  }
  if (a.drinks === null) e.push("Tell us whether you drink alcohol.");
  else if (a.drinks) {
    if (a.alcFreq === null) e.push("Select how often you drink alcohol.");
    const y = num(a.alcYears);
    if (y === null) e.push("Enter for how many years you have been drinking.");
    else if (y < 0 || y > Number(a.age || 90)) e.push("Years of drinking cannot be negative or greater than your age.");
  }
  const missing = cfg.diet_questions.filter((q) => a.diet[q.id] === undefined).length;
  if (missing) e.push(`Answer all ${cfg.diet_questions.length} food questions (${missing} left).`);
  return e;
}

export function validateHealth(a, cfg) {
  const e = [];
  if (!a.sex) return ["Complete the Profile step first."];
  if (a.sex === "women") {
    const h = num(a.heightCm), w = num(a.weightKg);
    const [hl, hh] = cfg.height_range_cm, [wl, wh] = cfg.weight_range_kg;
    if (h === null || h < hl || h > hh) e.push(`Enter your height (${hl}-${hh} cm).`);
    if (w === null || w < wl || w > wh) e.push(`Enter your weight (${wl}-${wh} kg).`);
    if (!e.length) {
      const bmi = computeBmi(h, w);
      if (bmi < 10 || bmi > 70) e.push("Those height and weight values look unusual. Please check them.");
    }
  }
  const s = num(a.bpSys), d = num(a.bpDia);
  const [sl, sh] = cfg.bp_systolic_range, [dl, dh] = cfg.bp_diastolic_range;
  if (s === null || s < sl || s > sh) e.push(`Enter your systolic (top) blood pressure (${sl}-${sh} mmHg).`);
  if (d === null || d < dl || d > dh) e.push(`Enter your diastolic (bottom) blood pressure (${dl}-${dh} mmHg).`);
  if (s !== null && d !== null && s <= d && !e.length) e.push("Systolic (top number) should be higher than diastolic (bottom number).");
  e.push(...validateIdrs(a, cfg));
  return e;
}

// IDRS answers are optional, but all-or-none (mirrors the server rule).
export function idrsStarted(a) {
  return num(a.waistCm) !== null || !!a.activity || !!a.familyHistory;
}
export function idrsComplete(a) {
  return num(a.waistCm) !== null && !!a.activity && !!a.familyHistory;
}
export function validateIdrs(a, cfg) {
  if (!idrsStarted(a)) return [];
  const e = [];
  if (!idrsComplete(a)) {
    e.push("For the IDRS comparison, answer waist, physical activity and family history together - or clear them to skip it.");
    return e;
  }
  const [lo, hi] = cfg?.idrs?.waist_range_cm ?? [50, 200];
  const w = num(a.waistCm);
  if (w < lo || w > hi) e.push(`Enter a waist circumference between ${lo} and ${hi} cm.`);
  return e;
}

export function buildPayload(a) {
  const women = a.sex === "women";
  return {
    sex: a.sex,
    age: num(a.age),
    state: num(a.state),
    education: a.education,
    residence: a.residence,
    tobacco_use: !!a.tobaccoUse,
    tobacco_products: a.tobaccoUse ? a.tobaccoProducts : [],
    cigarette_frequency: a.tobaccoUse ? a.cigFreq : null,
    drinks_alcohol: !!a.drinks,
    alcohol_frequency: a.drinks ? a.alcFreq : null,
    alcohol_years: a.drinks ? num(a.alcYears) : null,
    alcohol_types: a.drinks ? a.alcTypes : [],
    diet: a.diet,
    height_cm: women ? num(a.heightCm) : null,
    weight_kg: women ? num(a.weightKg) : null,
    bp_systolic: num(a.bpSys),
    bp_diastolic: num(a.bpDia),
    waist_cm: idrsComplete(a) ? num(a.waistCm) : null,
    physical_activity: idrsComplete(a) ? a.activity : null,
    family_history: idrsComplete(a) ? a.familyHistory : null,
  };
}

// Plain-language review summary (labels come from the backend config).
export function reviewSections(a, cfg) {
  const label = (list, key, v) => list.find((o) => o[key] === v)?.label ?? "-";
  const ps = cfg.per_sex[a.sex];
  const prods = ps.tobacco_products.filter((p) => a.tobaccoProducts.includes(p.id)).map((p) => p.label.toLowerCase());
  const types = ps.alcohol_types.filter((p) => a.alcTypes.includes(p.id)).map((p) => p.label.toLowerCase());
  const bmi = computeBmi(a.heightCm, a.weightKg);
  const tobacco = !a.tobaccoUse ? "No"
    : `Yes - ${prods.join(", ")}` + (a.cigFreq ? ` (cigarettes: ${label(cfg.tobacco_frequency, "value", a.cigFreq).toLowerCase()})` : "");
  const alcohol = !a.drinks ? "Non-drinker"
    : `${label(cfg.alcohol_frequency, "value", a.alcFreq)}, for ${a.alcYears} year${Number(a.alcYears) === 1 ? "" : "s"}`
      + (types.length ? ` (${types.join(", ")})` : "");
  const health = [];
  if (a.sex === "women") {
    health.push(["Height", `${a.heightCm} cm`], ["Weight", `${a.weightKg} kg`],
      ["BMI", bmi ? `${bmi.toFixed(1)} kg/m²` : "-"]);
  }
  health.push(["Blood pressure", `${a.bpSys}/${a.bpDia} mmHg`]);
  if (idrsComplete(a)) {
    health.push(["Waist", `${a.waistCm} cm`],
      ["Physical activity", label(cfg.idrs.activity, "value", a.activity)],
      ["Family history", label(cfg.idrs.family_history, "value", a.familyHistory)]);
  }
  const stateLabel = cfg?.states?.find((s) => s.value === num(a.state))?.label ?? "-";
  return [
    { key: "profile", title: "Profile", route: "/screening/profile", rows: [
      ["Age", `${a.age} years`], ["Sex", label(cfg.sexes, "id", a.sex)],
      ["State / UT", stateLabel],
      ["Education", label(cfg.education, "value", a.education)],
      ["Residence", label(cfg.residence, "value", a.residence)] ] },
    { key: "lifestyle", title: "Lifestyle", route: "/screening/lifestyle", rows: [
      ["Tobacco", tobacco], ["Alcohol", alcohol], ["Diet", `${Object.keys(a.diet).length} food questions answered`] ] },
    { key: "health", title: "Health", route: "/screening/health", rows: health },
  ];
}
