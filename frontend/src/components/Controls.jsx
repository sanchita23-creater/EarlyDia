import { useId } from "react";

export function Alert({ kind = "error", title, children }) {
  return (
    <div className={`alert alert-${kind}`} role={kind === "error" ? "alert" : "status"}>
      {title && <strong>{title}</strong>}
      {children}
    </div>
  );
}

export function Field({ label, hint, children, id }) {
  return (
    <div className="field">
      {label && <label className="label" htmlFor={id}>{label}</label>}
      {hint && <p className="hint">{hint}</p>}
      {children}
    </div>
  );
}

export function Group({ label, hint, children }) {
  const id = useId();
  return (
    <div className="field" role="group" aria-labelledby={id}>
      <span className="label" id={id}>{label}</span>
      {hint && <p className="hint">{hint}</p>}
      {children}
    </div>
  );
}

// Single choice rendered as selectable cards.
export function ChoiceGroup({ options, value, onChange, columns, label, hint, valueKey = "value" }) {
  return (
    <Group label={label} hint={hint}>
      <div className="choice-grid" style={columns ? { "--cols": columns } : undefined} role="radiogroup">
        {options.map((o) => {
          const v = o[valueKey];
          const on = value === v;
          return (
            <button type="button" key={String(v)} role="radio" aria-checked={on}
              className={`choice ${on ? "is-on" : ""}`} onClick={() => onChange(v)}>
              {o.label}
            </button>
          );
        })}
      </div>
    </Group>
  );
}

// Multi choice rendered as chips.
export function ChipGroup({ options, values, onChange, label, hint }) {
  const toggle = (id) =>
    onChange(values.includes(id) ? values.filter((x) => x !== id) : [...values, id]);
  return (
    <Group label={label} hint={hint}>
      <div className="chips">
        {options.map((o) => {
          const on = values.includes(o.id);
          return (
            <button type="button" key={o.id} role="checkbox" aria-checked={on}
              className={`chip ${on ? "is-on" : ""}`} onClick={() => toggle(o.id)}>
              {o.label}
            </button>
          );
        })}
      </div>
    </Group>
  );
}

export function NumberField({ label, hint, unit, value, onChange, min, max, step = "any", placeholder, inputMode = "decimal" }) {
  const id = useId();
  return (
    <Field label={label} hint={hint} id={id}>
      <div className="input-wrap">
        <input id={id} className="input" type="number" inputMode={inputMode} min={min} max={max} step={step}
          placeholder={placeholder} value={value} onChange={(e) => onChange(e.target.value)} />
        {unit && <span className="unit">{unit}</span>}
      </div>
    </Field>
  );
}

// One compact diet question with 4 segmented answers.
export function SegmentedRow({ label, options, value, onChange }) {
  const id = useId();
  return (
    <div className="seg-row" role="radiogroup" aria-labelledby={id}>
      <span className="seg-label" id={id}>{label}</span>
      <div className="seg">
        {options.map((o) => (
          <button type="button" key={o.value} role="radio" aria-checked={value === o.value}
            className={`seg-btn ${value === o.value ? "is-on" : ""}`} onClick={() => onChange(o.value)}>
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export function Collapsible({ title, children, defaultOpen = false }) {
  return (
    <details className="collapsible" open={defaultOpen}>
      <summary>{title}</summary>
      <div className="collapsible-body">{children}</div>
    </details>
  );
}

export function SelectField({ label, hint, value, onChange, options = [], placeholder = "Select an option" }) {
  const id = useId();
  return (
    <Field label={label} hint={hint} id={id}>
      <div className="input-wrap select-wrap">
        <select
          id={id}
          className="input select"
          value={value === null || value === undefined ? "" : String(value)}
          onChange={(e) => onChange(e.target.value)}
        >
          <option value="" disabled>
            {placeholder}
          </option>
          {options.map((o) => (
            <option key={String(o.value)} value={String(o.value)}>
              {o.label}
            </option>
          ))}
        </select>
      </div>
    </Field>
  );
}
