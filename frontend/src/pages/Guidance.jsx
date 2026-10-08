import { Link, Navigate, useNavigate } from "react-router-dom";
import Stepper from "../components/Stepper";
import { useScreening } from "../context/ScreeningContext";

export default function Guidance() {
  const { result: r, reset } = useScreening();
  const navigate = useNavigate();
  if (!r) return <Navigate to="/screening/profile" replace />;

  return (
    <div className="container narrow">
      <Stepper current="guidance" />
      <section className="panel">
        <h1 className="page-title">Next Steps &amp; General Guidance</h1>
        <p className="page-sub">General guidance based on your answers.</p>

        <div className={`callout ${r.flagged ? "callout-flag" : "callout-ok"}`}>
          <h2>{r.flagged ? "Your result was flagged for follow-up" : "Your result was in the lower screening range"}</h2>
          {r.flagged ? (
            <ul className="plain-list">
              <li>This is not a diagnosis. It means your answers resemble a group that was more likely to have the research-defined outcome.</li>
              <li>Consider discussing this result with a qualified healthcare professional.</li>
              <li>Ask whether a blood test such as HbA1c or fasting plasma glucose is appropriate for you.</li>
              <li>Share your blood pressure readings and any family health history with them.</li>
            </ul>
          ) : (
            <ul className="plain-list">
              <li>A lower screening result does not rule out diabetes or prediabetes.</li>
              <li>Continue regular health check-ups, including blood pressure and blood glucose checks as your doctor advises.</li>
              <li>Seek medical advice if you notice persistent thirst, frequent urination, unexplained tiredness or unexplained weight change.</li>
            </ul>
          )}
        </div>

        <h2 className="section-title">Everyday habits</h2>
        <div className="guide-list">
          {r.guidance.map((g) => (
            <details className="guide-item" key={g.category}>
              <summary>
                <span className="guide-cat">{g.category}</span>
                <span className="guide-title">{g.title}</span>
              </summary>
              <div className="guide-body">
                <p>{g.summary}</p>
                <ul className="plain-list">{g.action_items.map((x) => <li key={x}>{x}</li>)}</ul>
              </div>
            </details>
          ))}
        </div>

        <div className="actions">
          <Link to="/results" className="btn btn-secondary">Back to result</Link>
          <button type="button" className="btn btn-primary" onClick={() => { reset(); navigate("/"); }}>Start over</button>
        </div>
      </section>
    </div>
  );
}
