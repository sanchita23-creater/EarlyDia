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
          <Collapsible title="Technical Details & Production Models">
            <dl className="kv">
              <div><dt>Model evaluated</dt><dd>{t.model || (r.sex === "women" ? "LightGBM (v2 - lgbm_D)" : "CatBoost (v2 - cb_B)")} (separate model for {r.sex === "women" ? "women" : "men"})</dd></div>
              <div><dt>Calibrated probability</dt><dd>{(r.probability * 100).toFixed(2)}%</dd></div>
              <div><dt>Screening decision threshold</dt><dd>{(r.cutoff * 100).toFixed(1)}% (chosen at &ge;75% sensitivity floor on dev folds)</dd></div>
              <div><dt>Probability calibration</dt><dd>{calib}</dd></div>
              <div><dt>Blood pressure handling</dt><dd>{t.bp_readings_handling}</dd></div>
              {t.state_used && <div><dt>State / UT context</dt><dd>{t.state_used} {t.state_provided === false ? "(national default)" : ""}</dd></div>}
              {m.roc_auc !== undefined && (
                <div><dt>Held-out test metrics</dt>
                  <dd>ROC-AUC {m.roc_auc} &bull; Sensitivity {(m.sensitivity * 100).toFixed(0)}% &bull; Specificity {(m.specificity * 100).toFixed(0)}% &bull; Precision {(m.ppv_precision * 100).toFixed(1)}%</dd></div>
              )}
            </dl>

            <h3 className="tech-subhead">Approved Production Models &amp; Verified Benchmarks</h3>
            <p className="small">
              Verified against saved pipelines (<code>models_v2/</code>) and evaluation reports (<code>reports/v2/production_metrics.json</code>).
              Outdated experimental models (including earlier XGBoost candidates and v1 baselines) are deprecated in the model registry.
            </p>
            <div className="table-wrap">
              <table className="tech-table">
                <thead>
                  <tr>
                    <th scope="col">Cohort</th>
                    <th scope="col">Model Architecture</th>
                    <th scope="col">ROC-AUC</th>
                    <th scope="col">PR-AUC</th>
                    <th scope="col">Cutoff</th>
                    <th scope="col">Sensitivity</th>
                    <th scope="col">Specificity</th>
                    <th scope="col">Status</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><strong>Women</strong></td>
                    <td>LightGBM (<code>lgbm_D</code>)</td>
                    <td>0.8016 &plusmn; 0.0017</td>
                    <td>0.5829</td>
                    <td>21.2%</td>
                    <td>75.0%</td>
                    <td>70.1%</td>
                    <td><span className="tech-badge tech-badge-approved">Approved</span></td>
                  </tr>
                  <tr>
                    <td><strong>Men</strong></td>
                    <td>CatBoost (<code>cb_B</code>)</td>
                    <td>0.8033 &plusmn; 0.0086</td>
                    <td>0.6678</td>
                    <td>30.0%</td>
                    <td>75.1%</td>
                    <td>70.4%</td>
                    <td><span className="tech-badge tech-badge-best">Best Model</span></td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="small">
              <em>Benchmark comparison note:</em> For men, CatBoost <code>cb_B</code> demonstrated the highest discrimination (ROC-AUC 0.8033),
              outperforming LightGBM (<code>lgbm_C</code> ROC-AUC 0.7988) and XGBoost (<code>xgb_D</code> ROC-AUC 0.8001; earlier <code>xgb_B</code> 0.6792).
              For women, LightGBM <code>lgbm_D</code> was the top-performing model (ROC-AUC 0.8016).
            </p>

            <div className="evidence-section">
              <h3>Why these inputs (Published Evidence)</h3>
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

          <Collapsible title="Dataset, Target &amp; Preprocessing">
            <h3 className="tech-subhead">NFHS-5 Dataset &amp; Population</h3>
            <ul className="plain-list">
              <li><strong>Data Source:</strong> National Family Health Survey-5 (NFHS-5, 2019&ndash;2021), India.</li>
              <li><strong>Study Cohort:</strong> Adults aged 18&ndash;40 years; pregnant women excluded.</li>
              <li><strong>Sample Size:</strong> 199,514 women (development cohort; 46,328 positive; 23.22% prevalence; 164 features) and 29,151 men (development cohort; 9,259 positive; 31.76% prevalence; 145 features).</li>
              <li><strong>Sex-Separated Modeling:</strong> Modeled completely independently due to differing biological risk profiles, anthropometric markers (e.g. BMI recorded for women in NFHS-5), and differing survey questionnaires.</li>
            </ul>

            <h3 className="tech-subhead">Target Definition (Target A)</h3>
            <ul className="plain-list">
              <li><strong>Positive (Class 1):</strong> Any known-diabetes flag (self-reported diabetes <code>s728a</code>/<code>sm627a</code>, told high blood sugar <code>sb56</code>/<code>smb56</code>, taking medication <code>sb57</code>/<code>smb57</code>) OR random capillary blood glucose &ge; 141 mg/dL.</li>
              <li><strong>Negative (Class 0):</strong> All flags explicitly 0 AND random blood glucose &le; 100 mg/dL (or &le; 140 mg/dL with zero flags).</li>
              <li><strong>Gray Zone Excluded:</strong> Blood glucose 101&ndash;140 mg/dL with no flags excluded during training to eliminate ambiguous label boundary noise.</li>
              <li><strong>Survey Endpoint:</strong> Research-defined screening outcome based on cross-sectional survey data, NOT a formal clinical diagnosis.</li>
            </ul>

            <h3 className="tech-subhead">Preprocessing &amp; Leakage Prevention</h3>
            <ul className="plain-list">
              <li><strong>Cluster-Aware Pipeline:</strong> All imputation, encoding, and scaling fitted strictly fold-by-fold inside cross-validation splits to guarantee zero data leakage.</li>
              <li><strong>Missing Value Handling:</strong> Median imputation for continuous variables; distinct missing indicators for categorical variables.</li>
              <li><strong>Categorical Encoding:</strong> One-hot encoding for nominal categories (education, residence, state, diet frequencies, tobacco, alcohol).</li>
              <li><strong>Harmonized Blood Pressure:</strong> Survey records up to 3 valid readings; entered BP is treated as a complete 3-of-3 average to remain in-distribution.</li>
            </ul>
          </Collapsible>

          <Collapsible title="Feature Engineering &amp; Prediction Flow">
            <h3 className="tech-subhead">Feature Engineering (V1&ndash;V6 Progression)</h3>
            <ul className="plain-list">
              <li><strong>Total Dimensions:</strong> 164 features for women; 145 features for men (after one-hot encoding).</li>
              <li><strong>V1 Demographics &amp; Anthropometrics:</strong> Age, BMI (women), average SBP and DBP, tobacco products &amp; frequency, alcohol frequency &amp; duration, 9-item food intake frequency.</li>
              <li><strong>V3&ndash;V5 Hemodynamics &amp; Interactions:</strong> Pulse pressure (SBP &minus; DBP), mean arterial pressure (MAP), waist-to-hip ratio, waist-to-height ratio, age &times; BMI, age &times; systolic BP, waist &times; age.</li>
              <li><strong>Key Predictors:</strong> Hemodynamics (pulse pressure, SBP), central adiposity (waist-hip ratio), measurement timing context, age interactions, and geographic state.</li>
            </ul>

            <h3 className="tech-subhead">End-to-End Prediction Flow</h3>
            <div className="tech-flow">
              <div className="tech-flow-step">
                <span className="tech-flow-num">1</span>
                <div><strong>User Questionnaire:</strong> User provides answers across age, sex, anthropometry, blood pressure, lifestyle, diet, and state/UT.</div>
              </div>
              <div className="tech-flow-step">
                <span className="tech-flow-num">2</span>
                <div><strong>Validation:</strong> Client &amp; server validate plausible ranges (e.g., SBP 70&ndash;260 mmHg, DBP 40&ndash;150 mmHg).</div>
              </div>
              <div className="tech-flow-step">
                <span className="tech-flow-num">3</span>
                <div><strong>Feature Vector Mapping:</strong> Answers mapped to the sex-specific feature schema (164 features for women, 145 for men).</div>
              </div>
              <div className="tech-flow-step">
                <span className="tech-flow-num">4</span>
                <div><strong>Model Inference:</strong> Gradient-boosted decision tree pipeline (LightGBM for women, CatBoost for men) computes raw score.</div>
              </div>
              <div className="tech-flow-step">
                <span className="tech-flow-num">5</span>
                <div><strong>Probability Calibration:</strong> Platt sigmoid scaling converts raw score into calibrated screening probability.</div>
              </div>
              <div className="tech-flow-step">
                <span className="tech-flow-num">6</span>
                <div><strong>Cutoff Evaluation:</strong> Evaluated against screening decision threshold (21.2% women, 30.0% men).</div>
              </div>
              <div className="tech-flow-step">
                <span className="tech-flow-num">7</span>
                <div><strong>Visual Presentation:</strong> Semicircular gauge displays risk tier (Low, Moderate, High, Very High) with color-matched needle and evidence-based guidance.</div>
              </div>
            </div>
          </Collapsible>

          <Collapsible title="Validation Methodology &amp; Risk Thresholds">
            <h3 className="tech-subhead">Evaluation &amp; Validation Methodology</h3>
            <ul className="plain-list">
              <li><strong>Cluster-Grouped CV:</strong> 5-fold <code>StratifiedGroupKFold</code> grouped by Primary Sampling Unit (PSU) / survey cluster and stratified by outcome.</li>
              <li><strong>Spatial Leakage Protection:</strong> Participants from the same village/cluster never appear across both train and validation splits.</li>
              <li><strong>Calibration Selection:</strong> Sigmoid (Platt) scaling selected out-of-fold for reliable empirical alignment across risk percentiles.</li>
              <li><strong>Locked Held-Out Test:</strong> Final test set used exactly once on held-out clusters, guarding against overfitting or threshold hunting.</li>
            </ul>

            <h3 className="tech-subhead">Risk Thresholds &amp; Gauge Categories</h3>
            <ul className="plain-list">
              <li><strong>Screening Floor Strategy:</strong> Decision cutoffs chosen on training data at the highest value maintaining &ge;75% sensitivity floor.</li>
              <li><strong>Women Cutoff (21.2%):</strong> Sensitivity 75.0%, Specificity 70.1%, Precision (PPV) 43.1%, Flag Rate 40.4%.</li>
              <li><strong>Men Cutoff (30.0%):</strong> Sensitivity 75.1%, Specificity 70.4%, Precision (PPV) 54.1%, Flag Rate 44.0%.</li>
              <li><strong>Semicircular Gauge Risk Tiers:</strong>
                Low Risk (&lt;20%, Green), Moderate Risk (20&ndash;40%, Yellow), High Risk (40&ndash;60%, Orange), and Very High Risk (&ge;60%, Red).
              </li>
            </ul>
          </Collapsible>

          <Collapsible title="Limitations &amp; Clinical Boundaries">
            <ul className="plain-list">
              <li><strong>Non-Diagnostic Endpoint:</strong> Uses single random capillary blood glucose; no laboratory fasting plasma glucose, oral glucose tolerance test (OGTT), or venous HbA1c.</li>
              <li><strong>Screening Trade-Off:</strong> Prioritizing &ge;75% sensitivity produces false alarms (PPV 43&ndash;54%, ~40&ndash;44% flag rate). A flagged result requires formal lab testing, not an assumption of disease.</li>
              <li><strong>Geographic Context:</strong> State/UT is a survey cluster marker capturing broad regional prevalence variance, not an individual causal determinant.</li>
              <li><strong>Population Scope:</strong> Validated strictly for Indian adults aged 18&ndash;40 in NFHS-5; not clinically validated for pediatric, older, or non-Indian cohorts.</li>
              <li><strong>Observational Associations:</strong> Reflects statistical associations in cross-sectional survey data; answers do not prove individual clinical causality.</li>
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
