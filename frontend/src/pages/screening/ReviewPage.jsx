import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import StepShell from "../../components/StepShell";
import { Alert } from "../../components/Controls";
import { useScreening } from "../../context/ScreeningContext";
import { buildPayload, reviewSections } from "../../utils/answers";
import { predict } from "../../services/api";

export default function ReviewPage() {
  const { answers: a, config, setResult } = useScreening();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const sections = reviewSections(a, config);

  const run = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await predict(buildPayload(a));
      setResult(res);
      navigate("/results");
    } catch (e) {
      setError({ message: e.message, errors: e.errors || [] });
    } finally {
      setLoading(false);
    }
  };

  return (
    <StepShell step="review" title="Review your answers" subtitle="Check everything looks right, or edit a section."
      onBack={() => navigate("/screening/health")} onNext={run} nextLabel="Start Screening" loading={loading}
      serverError={error && (
        <Alert title={error.message}>
          {error.errors.length > 0 && <ul>{error.errors.map((m) => <li key={m}>{m}</li>)}</ul>}
        </Alert>
      )}>
      <div className="review">
        {sections.map((s) => (
          <section className="review-card" key={s.key}>
            <div className="review-head">
              <h2>{s.title}</h2>
              <Link className="btn btn-ghost btn-sm" to={`${s.route}?edit=1`} aria-label={`Edit ${s.title}`}>Edit</Link>
            </div>
            <dl className="kv">
              {s.rows.map(([k, v]) => (
                <div key={k}><dt>{k}</dt><dd>{v}</dd></div>
              ))}
            </dl>
          </section>
        ))}
      </div>
    </StepShell>
  );
}
