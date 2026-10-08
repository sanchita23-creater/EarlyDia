import { useState } from "react";
import StepShell from "../../components/StepShell";
import { ChoiceGroup, NumberField, SelectField } from "../../components/Controls";
import { useScreening } from "../../context/ScreeningContext";
import useStepNav from "../../utils/useStepNav";
import { validateProfile } from "../../utils/answers";

export default function ProfilePage() {
  const { answers: a, update, config } = useScreening();
  const nav = useStepNav("profile");
  const [show, setShow] = useState(false);
  const errors = validateProfile(a, config);

  const setSex = (sex) => {
    // Option lists differ by sex, so clear sex-specific selections when it changes.
    if (sex !== a.sex) update({ sex, tobaccoProducts: [], cigFreq: null, alcTypes: [] });
  };
  const next = () => (errors.length ? setShow(true) : nav.next());

  return (
    <StepShell step="profile" title="Tell us about yourself" subtitle="A few quick questions."
      errors={show ? errors : []} onBack={nav.back} onNext={next} backLabel={nav.editing ? "Cancel" : "Back"}
      nextLabel={nav.editing ? "Save" : "Continue"}>
      <ChoiceGroup label="Sex" options={config.sexes.map((s) => ({ value: s.id, label: s.label }))}
        value={a.sex} onChange={setSex} columns={2} />
      <NumberField label="Age" unit="years" min={config.age_range[0]} max={config.age_range[1]} step={1}
        inputMode="numeric" placeholder={`${config.age_range[0]}-${config.age_range[1]}`}
        value={a.age} onChange={(v) => update({ age: v })} />
      <SelectField label="State / Union Territory"
        hint="Contextual regional marker used for screening risk stratification"
        options={config.states || []} value={a.state}
        onChange={(v) => update({ state: v ? Number(v) : null })}
        placeholder="Select your State / UT" />
      <ChoiceGroup label="Highest education" options={config.education} value={a.education}
        onChange={(v) => update({ education: v })} columns={2} />
      <ChoiceGroup label="Where do you live?" options={config.residence} value={a.residence}
        onChange={(v) => update({ residence: v })} columns={2} />
    </StepShell>
  );
}
