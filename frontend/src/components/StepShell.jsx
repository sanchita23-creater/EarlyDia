import Stepper from "./Stepper";
import { Alert } from "./Controls";

export default function StepShell({ step, title, subtitle, errors = [], onBack, onNext, nextLabel = "Continue",
  backLabel = "Back", loading = false, children, serverError }) {
  return (
    <div className="container narrow">
      <Stepper current={step} />
      <section className="panel">
        <h1 className="page-title">{title}</h1>
        {subtitle && <p className="page-sub">{subtitle}</p>}
        <div className="form-body">{children}</div>
        {errors.length > 0 && (
          <Alert title="Please fix the following">
            <ul>{errors.map((e) => <li key={e}>{e}</li>)}</ul>
          </Alert>
        )}
        {serverError}
        <div className="actions">
          <button type="button" className="btn btn-secondary" onClick={onBack} disabled={loading}>{backLabel}</button>
          <button type="button" className="btn btn-primary" onClick={onNext} disabled={loading}>
            {loading ? <><span className="spinner" aria-hidden="true" /> Working...</> : nextLabel}
          </button>
        </div>
      </section>
    </div>
  );
}
