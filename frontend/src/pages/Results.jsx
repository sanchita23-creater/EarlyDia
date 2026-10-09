import { Link, Navigate } from "react-router-dom";
import Stepper from "../components/Stepper";
import ProbabilityScale from "../components/ProbabilityScale";
import { Collapsible } from "../components/Controls";
import { useScreening } from "../context/ScreeningContext";

const pct = (v) => (v > 0 && v < 0.05 ? "<0.1%" : `${v.toFixed(1)}%`);

const INPUT_RESEARCH_EVIDENCE = [
  {
    input: "Age",
    research: "Diabetes becomes more common with age; screening is recommended from age 35 or earlier with risk factors.",
    citation: "[ADA-2], [INDIAB-1]",
  },
  {
    input: "Sex",
    research: "Prevalence and risk-factor patterns differ by sex in Indian population data; survey items also differ.",
    citation: "[NFHS-5], [INDIAB-1]",
  },
  {
    input: "Body mass index (BMI)",
    research: "Overweight is a recognised risk factor; lower screening cut-off (23 kg/m²) for Asian populations (women's model).",
    citation: "[ADA-2], [WHO-BMI]",
  },
  {
    input: "Blood pressure",
    research: "Hypertension is a recognised screening factor; higher usual blood pressure is statistically associated with diabetes.",
    citation: "[ADA-2], [EMDIN-2015]",
  },
  {
    input: "Tobacco use",
    research: "Active smoking is associated with higher type 2 diabetes incidence in systematic reviews and meta-analyses.",
    citation: "[WILLI-2007], [PAN-2015]",
  },
  {
    input: "Alcohol intake",
    research: "Alcohol intake patterns are associated with diabetes risk, with heavy drinking linked to higher risk.",
    citation: "[KNOTT-2015], [BALIUNAS-2009]",
  },
  {
    input: "Diet (food frequency)",
    research: "Fruit/vegetable intake is linked with lower risk; sugary drinks and frequent fried foods with higher risk.",
    citation: "[CARTER-2010], [SCHWING-2017], [IMAMURA-2015], [CAHILL-2014]",
  },
  {
    input: "Education & residence",
    research: "Diabetes prevalence varies across states, between urban and rural settings, and across socioeconomic backgrounds.",
    citation: "[INDIAB-1]",
  },
];

export default function Results() {
  const { result: r } = useScreening();
  if (!r) return <Navigate to="/screening/profile" replace />;
  const t = r.technical || {};
  const m = t.test_metrics || {};
  const calib = t.calibration_method === "isotonic" ? "isotonic regression" : "sigmoid (Platt) scaling";

  return (
    <div className="container narrow">
      <Stepper current="results" />
      <section className="panel result">
        <h1 className="page-title">Your Screening Result</h1>

        <div className="result-main">
          <p className="result-kicker">Estimated diabetes-related screening likelihood</p>
          <p className="result-number" aria-label={`${r.percent} percent`}>{pct(r.percent)}</p>
          <div className="result-status">
            <span className="result-status-label">Result</span>
            <span className={`pill pill-lg ${r.flagged ? "pill-flag" : "pill-ok"}`}>{r.status}</span>
          </div>
        </div>

        <ProbabilityScale percent={r.percent} cutoffPercent={r.cutoff_percent} flagged={r.flagged} />

        <div className="result-explain">
          <p>
            Based on the information you provided, the model estimates a {pct(r.percent)} screening likelihood for the
            research-defined outcome used by EarlyDia.
          </p>
          <p><strong>This does not mean that you have diabetes.</strong> EarlyDia is a screening research prototype, not a diagnostic test.</p>
          <p className="cutoff-note">
            <strong>Screening cutoff: {pct(r.cutoff_percent)}.</strong> Results at or above this cutoff are flagged for
            follow-up because the model was evaluated using a sensitivity-oriented screening rule.
          </p>
          <p className="model-meta-note">
            Model {t.version || "v2"} - screening threshold {r.cutoff}
          </p>
          {r.state_used && (
            <p className="state-note">
              State used: {r.state_used}. State is a regional marker from NFHS-5, not a cause of risk.
            </p>
          )}
          {r.warning && (
            <p className="state-warning-note">
              <strong>Notice:</strong> {r.warning}
            </p>
          )}
        </div>

        <div className="next-box">
          <h2>What to do next</h2>
          {r.flagged ? (
            <p>Consider discussing this result with a qualified healthcare professional. Only a blood test, such as HbA1c or fasting glucose, can show how your blood sugar actually is.</p>
          ) : (
            <p>A lower screening result does not rule out diabetes. Keep up routine health check-ups and speak to a healthcare professional if you have any concerns.</p>
          )}
          <Link to="/guidance" className="btn btn-primary">View general guidance</Link>
        </div>

        {r.idrs && (
          <section className="idrs-card" aria-label="Indian Diabetes Risk Score">
            <h2>Indian Diabetes Risk Score (IDRS)</h2>
            <p className="idrs-score">
              <strong>{r.idrs.score}</strong> <small>/ {r.idrs.max_score}</small>
              <span className={`pill pill-lg ${r.idrs.category === "high" ? "pill-flag" : "pill-ok"}`}>{r.idrs.category_label}</span>
            </p>
            <dl className="kv kv-compact">
              {r.idrs.components.map((c) => (
                <div key={c.label}><dt>{c.label}</dt><dd>{c.answer} <span className="idrs-pts">+{c.points}</span></dd></div>
              ))}
            </dl>
            <p className="hint">{r.idrs.note}</p>
            <p className="hint">Source: {r.idrs.citation}</p>
          </section>
        )}

        <section className="profile-compact" aria-label="Your screening profile">
          <h2>Your Screening Profile</h2>
          <dl className="kv kv-compact">
            {r.profile.filter((p) => !["Sex"].includes(p.label)).map((p) => (
              <div key={p.label}><dt>{p.label}</dt><dd>{p.value}</dd></div>
            ))}
          </dl>
        </section>

        <div className="more">
          <Collapsible title="Technical Details">
            <dl className="kv">
              <div><dt>Model</dt><dd>{t.model} (separate model for {r.sex === "women" ? "women" : "men"})</dd></div>
              {t.state_used && <div><dt>State / UT</dt><dd>{t.state_used} {t.state_provided === false ? "(national default)" : ""}</dd></div>}
              <div><dt>Probability calibration</dt><dd>{calib}</dd></div>
              <div><dt>Calibrated probability</dt><dd>{(r.probability * 100).toFixed(2)}%</dd></div>
              <div><dt>Decision threshold</dt><dd>{(r.cutoff * 100).toFixed(1)}%</dd></div>
              <div><dt>Blood pressure handling</dt><dd>{t.bp_readings_handling}</dd></div>
              {m.roc_auc !== undefined && (
                <div><dt>Held-out test results</dt>
                  <dd>ROC-AUC {m.roc_auc}; sensitivity {(m.sensitivity * 100).toFixed(0)}%; specificity {(m.specificity * 100).toFixed(0)}%; precision {(m.ppv_precision * 100).toFixed(1)}%</dd></div>
              )}
            </dl>
            <p className="small">The cutoff was chosen on training data only, as the highest value that keeps sensitivity at or above 75%. It is a screening trade-off, not a clinically optimal value.</p>

            <div className="evidence-section">
              <h3>Why these inputs</h3>
              <p className="evidence-note">
                These are published associations, not proof that any factor caused your result. Model importance values are in the repo reports.
              </p>
              <div className="table-wrap">
                <table className="evidence-table">
                  <thead>
                    <tr>
                      <th scope="col">Input</th>
                      <th scope="col">What published research says</th>
                      <th scope="col">Citation tag</th>
                    </tr>
                  </thead>
                  <tbody>
                    {INPUT_RESEARCH_EVIDENCE.map((item) => (
                      <tr key={item.input}>
                        <td className="col-input">{item.input}</td>
                        <td className="col-research">{item.research}</td>
                        <td className="col-citation">{item.citation}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </Collapsible>
          <Collapsible title="Research Methodology">
            <ul className="plain-list">
              <li>Data: NFHS-5 (India, 2019-21), adults aged 18-40, women and men modelled separately.</li>
              <li>Outcome: a research-defined group (reported diabetes, or random blood glucose of 200 mg/dL or more), not a clinical diagnosis.</li>
              <li>Validation: data was split by survey cluster, so people from the same village never appear on both sides of a split.</li>
              <li>Regional context: State is used as a contextual marker based on NFHS-5 sampling groups.</li>
              <li>Model: gradient-boosted trees with probability calibration; the final test set was used only once.</li>
            </ul>
          </Collapsible>
          <Collapsible title="Limitations">
            <ul className="plain-list">
              <li>The outcome comes from a single survey and a random (non-fasting) glucose reading, so it is not a clinical diagnosis.</li>
              <li>Discrimination is modest, and many flagged results are false alarms because the cutoff favours catching cases.</li>
              <li>State is a regional marker capturing survey-level geographic variation, not an individual cause of risk.</li>
              <li>Results show statistical associations in survey data. They do not show that any single answer caused the result.</li>
              <li>Designed for ages 18-40 only, and not clinically validated.</li>
            </ul>
          </Collapsible>
        </div>

        <div className="actions">
          <Link to="/screening/review" className="btn btn-secondary">Edit answers</Link>
        </div>
      </section>
    </div>
  );
}
