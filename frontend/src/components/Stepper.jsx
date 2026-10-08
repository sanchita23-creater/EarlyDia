import { CheckIcon } from "./Icons";

export const STEPS = [
  { key: "profile", label: "Profile" },
  { key: "lifestyle", label: "Lifestyle" },
  { key: "health", label: "Health" },
  { key: "review", label: "Review" },
  { key: "results", label: "Results" },
  { key: "guidance", label: "Guidance" },
];

export default function Stepper({ current }) {
  const idx = STEPS.findIndex((s) => s.key === current);
  return (
    <nav className="stepper" aria-label="Progress">
      <ol>
        {STEPS.map((s, i) => {
          const state = i < idx ? "done" : i === idx ? "current" : "todo";
          return (
            <li key={s.key} className={`step step-${state}`} aria-current={state === "current" ? "step" : undefined}>
              <span className="step-dot">{state === "done" ? <CheckIcon width={14} height={14} strokeWidth={3} /> : i + 1}</span>
              <span className="step-label">{s.label}</span>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
