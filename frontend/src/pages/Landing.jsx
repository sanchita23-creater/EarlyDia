import { Link } from "react-router-dom";
import ProbabilityScale from "../components/ProbabilityScale";
import { DatabaseIcon, BookIcon, SplitIcon } from "../components/Icons";

const POINTS = [
  { icon: DatabaseIcon, title: "NFHS-5 based", text: "Built on India's National Family Health Survey (2019-21)." },
  { icon: BookIcon, title: "Research-informed features", text: "Simple, self-reportable inputs drawn from diabetes research." },
  { icon: SplitIcon, title: "Sex-specific screening models", text: "Separate models for women and men." },
];

export default function Landing() {
  return (
    <div className="container landing">
      <section className="hero">
        <div className="hero-copy">
          <h1 className="hero-brand">EarlyDia</h1>
          <h2 className="hero-title">Understand Your Diabetes-Related Screening Risk</h2>
          <p className="hero-text">
            An ML-based screening prototype built using NFHS-5 survey data for adults aged 18-40.
          </p>
          <div className="hero-actions">
            <Link to="/screening/profile" className="btn btn-primary btn-lg">Start Screening</Link>
            <Link to="/about" className="btn btn-secondary btn-lg">How It Works</Link>
          </div>
        </div>
        <div className="hero-preview" aria-hidden="true">
          <div className="preview-card">
            <div className="preview-head">
              <span>Example result</span>
              <span className="pill pill-flag">Flagged for follow-up</span>
            </div>
            <ProbabilityScale percent={4.2} cutoffPercent={2.6} flagged />
          </div>
        </div>
      </section>
      <section className="points" aria-label="Highlights">
        {POINTS.map(({ icon: Icon, title, text }) => (
          <div className="point" key={title}>
            <span className="point-icon"><Icon /></span>
            <h3>{title}</h3>
            <p>{text}</p>
          </div>
        ))}
      </section>
    </div>
  );
}
