import { useEffect, useState } from "react";
import { HashRouter, Link, Route, Routes, useLocation } from "react-router-dom";
import { api, mode } from "./api";
import { EngineDetail } from "./pages/EngineDetail";
import { FleetOverview } from "./pages/FleetOverview";
import { ModelComparisonPage } from "./pages/ModelComparison";

const ALL_SUBSETS = ["FD001", "FD002", "FD003", "FD004"];

const PIPELINE_STEPS = [
  "RAW SENSOR DATA",
  "HEALTH INDICATOR",
  "RUL PREDICTION",
  "FAILURE RISK",
  "MAINTENANCE REC.",
];

function AboutPanel() {
  return (
    <div className="about-panel">
      <div className="about-panel-inner">
        <div className="pipeline-strip">
          {PIPELINE_STEPS.map((step, i) => (
            <span key={step} style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span className="pipeline-step">{step}</span>
              {i < PIPELINE_STEPS.length - 1 && <span className="pipeline-arrow">&gt;&gt;&gt;</span>}
            </span>
          ))}
        </div>
        <div>
          <h4>What is RUL?</h4>
          Remaining Useful Life — predicted engine-cycles left before failure, from NASA's
          CMAPSS turbofan degradation dataset. Every prediction here comes from a model
          trained and evaluated against the real, uncensored ground truth in RUL_*.txt.
        </div>
        <div>
          <h4>Risk tiers</h4>
          CRITICAL (&le;15 cycles), HIGH (&le;30), MEDIUM (&le;60), LOW (&gt;60) — thresholds
          chosen relative to the 125-cycle cap models are trained against, so "low risk"
          means comfortably clear of the regime the model was tuned to be precise in.
        </div>
        <div>
          <h4>Which model?</h4>
          Five architectures (XGBoost, LSTM, TCN, Transformer, Weibull AFT survival) are
          compared per subset on the NASA PHM08 asymmetric score; the winner is served live.
          See the model comparison page for the grounding and limitations behind each one.
        </div>
        <div>
          <h4>The interval</h4>
          Split conformal prediction: a distribution-free 90% coverage guarantee around the
          point estimate, calibrated on held-out engines. Wider isn't worse — it's honest
          about where the model is less certain.
        </div>
        <div>
          <h4>About this demo</h4>
          {mode === "static"
            ? "Running in static mode: predictions are precomputed JSON snapshots over the fixed CMAPSS test set, no backend required. The FastAPI service behind these same predictions is in the repo — clone it and set VITE_API_BASE to run live."
            : "Connected to a live FastAPI backend serving these predictions in real time from the trained model artifacts."}
        </div>
      </div>
    </div>
  );
}

function HeaderNav() {
  const location = useLocation();
  return (
    <Link
      to="/models"
      className={`header-btn ${location.pathname === "/models" ? "active" : ""}`}
    >
      Models
    </Link>
  );
}

function App() {
  const [subset, setSubset] = useState("FD001");
  const [availableSubsets, setAvailableSubsets] = useState<string[]>(ALL_SUBSETS);
  const [apiOk, setApiOk] = useState<boolean | null>(null);
  const [aboutOpen, setAboutOpen] = useState(false);

  useEffect(() => {
    api
      .health()
      .then((res) => {
        setApiOk(true);
        if (res.available_subsets.length) setAvailableSubsets(res.available_subsets);
      })
      .catch(() => setApiOk(false));
  }, []);

  return (
    <HashRouter>
      <div className="app-shell">
        <header className="app-header">
          <div className="brand">
            <span className="brand-mark">
              AERO<span className="accent">RUL</span>
            </span>
            <span className="brand-sub">// Fleet Health Console</span>
          </div>
          <div className="header-right">
            <HeaderNav />
            <button
              type="button"
              className={`header-btn ${aboutOpen ? "active" : ""}`}
              onClick={() => setAboutOpen((o) => !o)}
            >
              {aboutOpen ? "✕ Close" : "? About"}
            </button>
            <div className="subset-group">
              <span className="subset-group-label">SUBSET</span>
              <div className="subset-buttons">
                {ALL_SUBSETS.map((s) => (
                  <button
                    key={s}
                    type="button"
                    className={`subset-btn ${s === subset ? "active" : ""}`}
                    disabled={!availableSubsets.includes(s)}
                    onClick={() => setSubset(s)}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
            <div className="status-dot-wrap">
              <span
                className={`status-dot ${
                  mode === "static" ? "ok" : apiOk ? "ok" : apiOk === false ? "down" : ""
                }`}
              />
              {apiOk === null
                ? "connecting"
                : mode === "static"
                  ? "static demo"
                  : apiOk
                    ? "api online"
                    : "api offline"}
            </div>
          </div>
        </header>

        {aboutOpen && <AboutPanel />}

        <main className="app-main">
          <Routes>
            <Route path="/" element={<FleetOverview subset={subset} />} />
            <Route path="/engine/:unitNumber" element={<EngineDetail subset={subset} />} />
            <Route path="/models" element={<ModelComparisonPage subset={subset} />} />
          </Routes>
        </main>
      </div>
    </HashRouter>
  );
}

export default App;
