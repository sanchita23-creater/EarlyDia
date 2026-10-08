import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { LineChart, BarList } from "../components/LineChart";
import { getResearch } from "../services/api";
import { pct, fix, reliabilityPoints, calibrationSummary, decisionCurve, subgroupRows, topFeatures } from "../utils/research";

const TABS = ["Overview", "Dataset & target", "Validation", "Calibration", "Decision curve", "Subgroups", "Features", "Mapper / TDA", "Limitations", "References"];

function Badge({ kind, children }) { return <span className={`badge badge-${kind}`}>{children}</span>; }
function Group({ title, kind, children }) {
  return <section className={`card research-block research-${kind}`}><h2>{title}</h2>{children}</section>;
}

export default function Research() {
  const [tab, setTab] = useState(TABS[0]);
  const [data, setData] = useState({});
  const [err, setErr] = useState("");
  useEffect(() => {
    let alive = true;
    Promise.all([getResearch("summary"), getResearch("calibration"), getResearch("decision-curve"),
      getResearch("subgroups"), getResearch("feature-importance"), getResearch("target-definition")])
      .then(([summary, calibration, dca, subgroups, features, target]) => alive && setData({ summary, calibration: calibration.rows, dca: dca.rows, subgroups: subgroups.rows, features: features.rows, target }))
      .catch((e) => alive && setErr(e.message));
    return () => { alive = false; };
  }, []);

  if (err) return <div className="container"><div className="alert alert-error"><strong>Research artifacts unavailable</strong><p>{err}</p></div></div>;
  if (!data.summary) return <div className="container loading"><span className="spinner" aria-hidden="true" /> Loading research outputs...</div>;
  const s = data.summary, pm = s.production_model;

  return (
    <div className="container">
      <h1 className="page-title">Research</h1>
      <p className="muted">Research prototype documentation. Nothing on this page changes the production model.</p>
      <div className="tabs" role="tablist" aria-label="Research sections">
        {TABS.map((t) => <button key={t} role="tab" aria-selected={tab === t} className={`tab ${tab === t ? "tab-on" : ""}`} onClick={() => setTab(t)}>{t}</button>)}
      </div>

      {tab === "Overview" && (<>
        <Group title="Production model" kind="prod"><Badge kind="prod">Locked</Badge>
          <p>Sex-specific HistGradientBoosting models; women isotonic and men sigmoid calibration; thresholds 2.6% (women) and 2.7% (men). Metrics below are the unchanged final-test results.</p>
          <div className="table-wrap"><table><thead><tr><th>Sex</th><th>n</th><th>ROC-AUC</th><th>PR-AUC</th><th>Sensitivity</th><th>Specificity</th><th>PPV</th><th>Brier</th></tr></thead>
            <tbody>{["women", "men"].map((x) => <tr key={x}><td>{x}</td><td>{pm[x].n.toLocaleString()}</td><td>{fix(pm[x].roc_auc)}</td><td>{fix(pm[x].pr_auc)}</td><td>{fix(pm[x].sensitivity)}</td><td>{fix(pm[x].specificity)}</td><td>{fix(pm[x].ppv_precision)}</td><td>{fix(pm[x].brier_score, 4)}</td></tr>)}</tbody></table></div>
        </Group>
        <Group title="Key finding" kind="research"><Badge kind="research">Research analysis</Badge><p>{s.key_finding}</p></Group>
        <Group title="What was and was not run" kind="research">
          <ul>{Object.entries(s.analysis_status).map(([k, v]) => <li key={k}><strong>{k.replace(/_/g, " ")}:</strong> {v}</li>)}</ul>
        </Group>
        <Group title="Claim boundaries" kind="research">
          {Object.entries(s.claim_boundaries).map(([k, v]) => <div key={k}><h3>{k.replace(/_/g, " ")}</h3><ul>{v.map((i) => <li key={i}>{i}</li>)}</ul></div>)}
        </Group>
      </>)}

      {tab === "Dataset & target" && (<>
        <Group title="Dataset and population" kind="prod"><p>NFHS-5 (India, 2019-21) individual and men's recodes; ages 18-40; women and men modelled separately; pregnant women excluded.</p></Group>
        <Group title="Target A (production)" kind="prod"><p>Positive: any known-diabetes indicator or valid glucose of at least 200 mg/dL. Negative: all indicators explicitly No with valid glucose up to 140 mg/dL. Glucose 141-199 mg/dL without known diabetes is excluded as ambiguous. It is a research-defined survey outcome, not a clinical diagnosis.</p></Group>
        <Group title="Alternative literature-aligned target" kind="research"><Badge kind="research">Not used by the production model</Badge>
          <p>{data.target.reason}</p>
          <div className="table-wrap"><table><thead><tr><th>Ingredient</th><th>Variables</th><th>Status</th></tr></thead><tbody>{data.target.ingredients.map((r) => <tr key={r.ingredient}><td>{r.ingredient}</td><td>{r.variables}</td><td>{r.status}</td></tr>)}</tbody></table></div>
        </Group>
      </>)}

      {tab === "Validation" && (
        <Group title="Cluster-aware validation" kind="prod"><p>The train/test split is grouped by primary sampling unit (PSU), so no PSU appears in both sets, and cross-validation uses StratifiedGroupKFold with PSU groups. Naive random splits can be optimistic for clustered survey data because people in the same PSU resemble each other. Thresholds were chosen on out-of-fold training predictions only; the final test set was used once.</p><p className="muted">External, prospective and clinical validation have not been performed.</p></Group>
      )}

      {tab === "Calibration" && (<Group title="Calibration on the locked final test" kind="research"><Badge kind="research">Research analysis</Badge>
        <p>Calibration asks whether predicted risks match observed frequencies; discrimination (ROC-AUC) asks whether people are ranked correctly. They are different properties.</p>
        <div className="chart-row">{["women", "men"].map((x) => {
          const sm = calibrationSummary(data.calibration, x); const pts = reliabilityPoints(data.calibration, x);
          return <div key={x}><h3>{x}</h3>
            <LineChart title={`${x} calibration`} xLabel="Mean predicted" yLabel="Observed fraction" diag series={[{ name: "Deciles", data: pts, dots: true }]} />
            <p className="muted">Slope {fix(sm.calibration_slope)} · intercept {fix(sm.calibration_intercept)} · Brier {fix(sm.brier_score, 4)}</p></div>;
        })}</div></Group>)}

      {tab === "Decision curve" && (<Group title="Decision curve analysis" kind="research"><Badge kind="research">Research analysis</Badge>
        <p>Decision curve analysis evaluates potential decision usefulness and does not establish clinical utility. It does not change the locked thresholds.</p>
        <div className="chart-row">{["women", "men"].map((x) => {
          const d = decisionCurve(data.dca, x); const thr = Number(d.exact.threshold);
          return <div key={x}><h3>{x}</h3>
            <LineChart title={`${x} decision curve`} xLabel="Threshold probability" yLabel="Net benefit" vline={thr} series={[{ name: "EarlyDia", data: d.model }, { name: "Treat all", data: d.all }]} />
            <p className="muted">At the locked cutoff ({pct(thr)}): net benefit {fix(d.exact.net_benefit_model, 4)} vs treat-all {fix(d.exact.net_benefit_treat_all, 4)} (exact, from the test confusion matrix). Curve is {d.source}.</p></div>;
        })}</div></Group>)}

      {tab === "Subgroups" && (<Group title="Subgroup performance at locked thresholds" kind="research"><Badge kind="research">Research analysis</Badge>
        <p>Overall sensitivity hides a large age difference. No subgroup-specific thresholds exist. Residence, education and state subgroups require row-level data that is not in this build.</p>
        <div className="table-wrap"><table><thead><tr><th>Sex</th><th>Group</th><th>n</th><th>Prevalence</th><th>ROC-AUC</th><th>Sens.</th><th>Spec.</th><th>PPV</th><th>NPV</th><th>FNR</th></tr></thead>
          <tbody>{subgroupRows(data.subgroups).map((r) => <tr key={r.sex + r.subgroup}><td>{r.sex}</td><td>{r.subgroup.replace("age_", "age ").replace("_", "-")}</td><td>{Number(r.n).toLocaleString()}</td><td>{pct(r.prevalence)}</td><td>{fix(r.roc_auc)}</td><td>{fix(r.sensitivity)}</td><td>{fix(r.specificity)}</td><td>{fix(r.ppv)}</td><td>{fix(r.npv)}</td><td>{fix(r.false_negative_rate)}</td></tr>)}</tbody></table></div></Group>)}

      {tab === "Features" && (<Group title="Permutation importance (not SHAP)" kind="prod"><p>These values describe how much each feature contributed to model discrimination. They do not show that any feature causes diabetes.</p>
        <div className="chart-row">{["women", "men"].map((x) => <div key={x}><h3>{x}</h3><BarList valueLabel={`${x} importance`} items={topFeatures(data.features, x)} /></div>)}</div></Group>)}

      {tab === "Mapper / TDA" && (<Group title="Exploratory Mapper / TDA" kind="research"><Badge kind="research">Exploratory</Badge>
        <p>The Mapper, stability and covariance-matched null-model code is implemented and unit-tested on synthetic data. It has <strong>not</strong> been run on NFHS data in this build because the row-level training data is not distributed. No communities, subtypes or null-model results are reported. Any future communities would be called exploratory feature-space communities, never biological diabetes subtypes, and only stable ones would be discussed.</p></Group>)}

      {tab === "Limitations" && (<Group title="Limitations" kind="research"><ul>
        <li>Single capillary glucose reading; no HbA1c, fasting confirmation or OGTT.</li>
        <li>Target A excludes 141-199 mg/dL without known diabetes and treats known diabetes as positive.</li>
        <li>Low PPV and about half of people flagged at the locked cutoffs.</li>
        <li>Younger adults (18-29) have much lower sensitivity at the pooled cutoff.</li>
        <li>No external, prospective or clinical validation; no causal interpretation.</li>
      </ul><p><Link to="/about">How EarlyDia works</Link></p></Group>)}

      {tab === "References" && (<Group title="References" kind="research"><p>The full reference list is in <code>docs/REFERENCES.md</code>. Only the supplied papers are cited, and any detail that could not be verified is marked as unverified.</p></Group>)}
    </div>
  );
}
