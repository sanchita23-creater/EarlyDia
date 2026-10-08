import { Link } from "react-router-dom";

const FLOW = ["Your inputs", "Preprocessing", "Sex-specific model", "Screening result"];

export default function About() {
  return (
    <div className="container">
      <h1 className="page-title">How EarlyDia works</h1>
      <div className="about-grid">
        <article className="card">
          <h2>What is EarlyDia?</h2>
          <p>An ML-based diabetes-related screening research prototype.</p>
        </article>
        <article className="card">
          <h2>Data</h2>
          <p>NFHS-5 survey data (India, 2019-21), adults aged 18-40.</p>
        </article>
        <article className="card card-wide">
          <h2>How it works</h2>
          <ol className="flow">
            {FLOW.map((f) => <li key={f}>{f}</li>)}
          </ol>
        </article>
        <article className="card card-wide card-important">
          <h2>Important</h2>
          <p>EarlyDia is a screening tool, not a diagnosis.</p>
        </article>
      </div>
      <div className="center-actions">
        <Link to="/screening/profile" className="btn btn-primary btn-lg">Start Screening</Link>
      </div>
    </div>
  );
}
