import { Navigate, Route, Routes, useParams } from "react-router-dom";
import Layout from "./components/Layout";
import { Alert } from "./components/Controls";
import { useScreening } from "./context/ScreeningContext";
import { validateProfile, validateLifestyle, validateHealth } from "./utils/answers";
import { ORDER } from "./utils/useStepNav";
import Landing from "./pages/Landing";
import About from "./pages/About";
import ProfilePage from "./pages/screening/ProfilePage";
import LifestylePage from "./pages/screening/LifestylePage";
import HealthPage from "./pages/screening/HealthPage";
import ReviewPage from "./pages/screening/ReviewPage";
import Results from "./pages/Results";
import Guidance from "./pages/Guidance";
import Research from "./pages/Research";

const PAGES = { profile: ProfilePage, lifestyle: LifestylePage, health: HealthPage, review: ReviewPage };
const CHECKS = [["profile", validateProfile], ["lifestyle", validateLifestyle], ["health", validateHealth]];

// Loads the form options from the API and stops people from jumping ahead of unanswered steps.
function ScreeningStep() {
  const { step } = useParams();
  const { answers, config, configError, loadConfig } = useScreening();
  const Page = PAGES[step];
  if (!Page) return <Navigate to="/screening/profile" replace />;
  if (configError) {
    return (
      <div className="container narrow">
        <Alert title="We couldn't load the screening form">
          <p>{configError}</p>
          <button type="button" className="btn btn-secondary btn-sm" onClick={loadConfig}>Try again</button>
        </Alert>
      </div>
    );
  }
  if (!config) return <div className="container narrow loading"><span className="spinner" aria-hidden="true" /> Loading...</div>;
  const target = ORDER.indexOf(step);
  for (const [key, check] of CHECKS) {
    if (ORDER.indexOf(key) < target && check(answers, config).length) {
      return <Navigate to={`/screening/${key}`} replace />;
    }
  }
  return <Page />;
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Landing />} />
        <Route path="/about" element={<About />} />
        <Route path="/screening" element={<Navigate to="/screening/profile" replace />} />
        <Route path="/screening/:step" element={<ScreeningStep />} />
        <Route path="/results" element={<Results />} />
        <Route path="/guidance" element={<Guidance />} />
        <Route path="/research" element={<Research />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
