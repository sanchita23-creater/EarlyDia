import { useState } from "react";
import StepShell from "../../components/StepShell";
import { NumberField, Group, ChoiceGroup } from "../../components/Controls";
import { useScreening } from "../../context/ScreeningContext";
import useStepNav from "../../utils/useStepNav";
import { validateHealth, computeBmi, bmiBand, idrsStarted } from "../../utils/answers";

export default function HealthPage() {
  const { answers: a, update, config } = useScreening();
  const nav = useStepNav("health");
  const [show, setShow] = useState(false);
  const errors = validateHealth(a, config);
  const women = a.sex === "women";
  const bmi = women ? computeBmi(a.heightCm, a.weightKg) : null;
  const next = () => (errors.length ? setShow(true) : nav.next());

  return (
    <StepShell step="health" title="Your health measurements"
      subtitle={women ? "Height, weight and blood pressure, plus an optional risk-score section." : "Your blood pressure, plus an optional risk-score section."}
      errors={show ? errors : []} onBack={nav.back} onNext={next} backLabel={nav.editing ? "Cancel" : "Back"}
      nextLabel={nav.editing ? "Save" : "Continue"}>
      {women && (
        <section className="block">
          <div className="two-col">
            <NumberField label="Height" unit="cm" min={120} max={220} placeholder="e.g. 160"
              value={a.heightCm} onChange={(v) => update({ heightCm: v })} />
            <NumberField label="Weight" unit="kg" min={25} max={200} placeholder="e.g. 58"
              value={a.weightKg} onChange={(v) => update({ weightKg: v })} />
          </div>
          <div className="bmi-card" aria-live="polite">
            <span className="bmi-label">Your BMI</span>
            {bmi ? (
              <>
                <strong className="bmi-value">{bmi.toFixed(1)} <small>kg/m²</small></strong>
                <span className="bmi-band">{bmiBand(bmi)}</span>
              </>
            ) : (
              <span className="bmi-band">Calculated automatically from your height and weight</span>
            )}
          </div>
        </section>
      )}
      {!women && (
        <p className="hint" role="note">
          BMI is not collected for men: the NFHS-5 men's questionnaire has no BMI, so the men's model does not use it.
        </p>
      )}
      <section className="block">
        <Group label="Blood pressure"
          hint="Use your most recent measurement. If you don't have one, a clinic or pharmacy can measure it - it is needed for the screening.">
          <div className="two-col">
            <NumberField label="Top number (systolic)" unit="mmHg" min={70} max={270} step={1} inputMode="numeric"
              placeholder="e.g. 120" value={a.bpSys} onChange={(v) => update({ bpSys: v })} />
            <NumberField label="Bottom number (diastolic)" unit="mmHg" min={40} max={150} step={1} inputMode="numeric"
              placeholder="e.g. 80" value={a.bpDia} onChange={(v) => update({ bpDia: v })} />
          </div>
        </Group>
      </section>
      <section className="block idrs-block" aria-label="Optional IDRS comparison">
        <h2 className="block-title">Optional: Indian Diabetes Risk Score (IDRS)</h2>
        <p className="hint">
          IDRS is a separate, published screening score (Mohan et al., 2005). These three answers are used <strong>only</strong> to
          show an IDRS result next to the EarlyDia result - they are <strong>not</strong> given to the machine-learning model.
          Answer all three, or leave them blank to skip.
        </p>
        <NumberField label="Waist circumference" unit="cm" min={config.idrs.waist_range_cm[0]} max={config.idrs.waist_range_cm[1]}
          placeholder={women ? "e.g. 82" : "e.g. 92"} hint="Measured around the belly button, standing, after breathing out."
          value={a.waistCm} onChange={(v) => update({ waistCm: v })} />
        <ChoiceGroup label="Physical activity" options={config.idrs.activity} value={a.activity}
          onChange={(v) => update({ activity: v })} columns={1} />
        <ChoiceGroup label="Family history of diabetes" options={config.idrs.family_history} value={a.familyHistory}
          onChange={(v) => update({ familyHistory: v })} columns={1} />
        {idrsStarted(a) && (
          <button type="button" className="btn btn-link"
            onClick={() => update({ waistCm: "", activity: null, familyHistory: null })}>Clear IDRS answers</button>
        )}
      </section>
    </StepShell>
  );
}
