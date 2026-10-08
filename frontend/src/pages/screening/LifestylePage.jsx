import { useState } from "react";
import StepShell from "../../components/StepShell";
import { ChoiceGroup, ChipGroup, NumberField, SegmentedRow } from "../../components/Controls";
import { useScreening } from "../../context/ScreeningContext";
import useStepNav from "../../utils/useStepNav";
import { validateLifestyle } from "../../utils/answers";

const YES_NO = [{ value: true, label: "Yes" }, { value: false, label: "No" }];

export default function LifestylePage() {
  const { answers: a, update, config } = useScreening();
  const nav = useStepNav("lifestyle");
  const [show, setShow] = useState(false);
  const errors = validateLifestyle(a, config);
  const ps = config.per_sex[a.sex];
  const smokesCig = a.tobaccoUse && a.tobaccoProducts.includes(ps.cigarette_id);
  const next = () => (errors.length ? setShow(true) : nav.next());

  return (
    <StepShell step="lifestyle" title="Your lifestyle" subtitle="Tobacco, alcohol and everyday food habits."
      errors={show ? errors : []} onBack={nav.back} onNext={next} backLabel={nav.editing ? "Cancel" : "Back"}
      nextLabel={nav.editing ? "Save" : "Continue"}>
      <section className="block">
        <h2 className="block-title">Tobacco</h2>
        <ChoiceGroup label="Do you currently use any tobacco?" options={YES_NO} value={a.tobaccoUse}
          onChange={(v) => update(v ? { tobaccoUse: true } : { tobaccoUse: false, tobaccoProducts: [], cigFreq: null })} columns={2} />
        {a.tobaccoUse && (
          <>
            <ChipGroup label="Which products?" hint="Select all that apply." options={ps.tobacco_products}
              values={a.tobaccoProducts}
              onChange={(v) => update({ tobaccoProducts: v, cigFreq: v.includes(ps.cigarette_id) ? a.cigFreq : null })} />
            {smokesCig && ps.asks_tobacco_frequency && (
              <ChoiceGroup label="How often do you smoke cigarettes?" options={config.tobacco_frequency}
                value={a.cigFreq} onChange={(v) => update({ cigFreq: v })} columns={2} />
            )}
          </>
        )}
      </section>

      <section className="block">
        <h2 className="block-title">Alcohol</h2>
        <ChoiceGroup label="Do you drink alcohol?" options={YES_NO} value={a.drinks}
          onChange={(v) => update(v ? { drinks: true } : { drinks: false, alcFreq: null, alcYears: "", alcTypes: [] })} columns={2} />
        {a.drinks && (
          <>
            <ChoiceGroup label="How often?" options={config.alcohol_frequency} value={a.alcFreq}
              onChange={(v) => update({ alcFreq: v })} columns={3} />
            <NumberField label="For how many years have you been drinking?" unit="years" min={0} max={40} step={1}
              inputMode="numeric" value={a.alcYears} onChange={(v) => update({ alcYears: v })} />
            <ChipGroup label="What do you usually drink?" hint="Optional - select all that apply."
              options={ps.alcohol_types} values={a.alcTypes} onChange={(v) => update({ alcTypes: v })} />
          </>
        )}
      </section>

      <section className="block">
        <h2 className="block-title">Diet</h2>
        <p className="hint">How often do you have each of these?</p>
        <div className="seg-list">
          {config.diet_questions.map((q) => (
            <SegmentedRow key={q.id} label={q.label} options={config.diet_options} value={a.diet[q.id]}
              onChange={(v) => update({ diet: { ...a.diet, [q.id]: v } })} />
          ))}
        </div>
        <p className="progress-note">
          {Object.keys(a.diet).length} of {config.diet_questions.length} answered
        </p>
      </section>
    </StepShell>
  );
}
