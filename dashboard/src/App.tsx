import { useEffect, useState } from "react";
import { HashRouter, Route, Routes } from "react-router-dom";
import { api } from "./api";
import { EngineDetail } from "./pages/EngineDetail";
import { FleetOverview } from "./pages/FleetOverview";

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
          compared per subset on the NASA PHM08 asymmetric score; the winner is served live
          and shown as CHAMPION in the comparison table.
        </div>
        <div>
          <h4>The interval</h4>
          Split conformal prediction: a distribution-free 90% coverage guarantee around the
          point estimate, calibrated on held-out engines. Wider isn't worse — it's honest
          about where the model is less certain.
        </div>
      </div>
    </div>
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
            <button
              type="button"
              className={`header-btn ${aboutOpen ? "active" : ""}`}
              onClick={() => setAboutOpen((o) => !o)}
            >
              {aboutOpen ? "✕ Close" : "? About"}
            </button>
            <div className="select-wrap">
              <select
                className="native-select"
                value={subset}
                onChange={(e) => setSubset(e.target.value)}
              >
                {ALL_SUBSETS.map((s) => (
                  <option key={s} value={s} disabled={!availableSubsets.includes(s)}>
                    {s}
                  </option>
                ))}
              </select>
            </div>
            <div className="status-dot-wrap">
              <span className={`status-dot ${apiOk ? "ok" : apiOk === false ? "down" : ""}`} />
              {apiOk === null ? "connecting" : apiOk ? "api online" : "api offline"}
            </div>
          </div>
        </header>

        {aboutOpen && <AboutPanel />}

        <main className="app-main">
          <Routes>
            <Route path="/" element={<FleetOverview subset={subset} />} />
            <Route path="/engine/:unitNumber" element={<EngineDetail subset={subset} />} />
          </Routes>
        </main>
      </div>
    </HashRouter>
  );
}

export default App;
