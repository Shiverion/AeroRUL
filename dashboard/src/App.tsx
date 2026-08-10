import { useEffect, useState } from "react";
import { HashRouter, Route, Routes } from "react-router-dom";
import { api } from "./api";
import { EngineDetail } from "./pages/EngineDetail";
import { FleetOverview } from "./pages/FleetOverview";

const ALL_SUBSETS = ["FD001", "FD002", "FD003", "FD004"];

function App() {
  const [subset, setSubset] = useState("FD001");
  const [availableSubsets, setAvailableSubsets] = useState<string[]>(ALL_SUBSETS);
  const [apiOk, setApiOk] = useState<boolean | null>(null);

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
            <span className="brand-mark">AeroRUL</span>
            <span className="brand-sub">Turbofan fleet health</span>
          </div>
          <div className="header-right">
            <div className="subset-switcher">
              {ALL_SUBSETS.map((s) => (
                <button
                  key={s}
                  className={`subset-btn ${s === subset ? "active" : ""}`}
                  disabled={!availableSubsets.includes(s)}
                  onClick={() => setSubset(s)}
                >
                  {s}
                </button>
              ))}
            </div>
            <span className={`api-status ${apiOk ? "ok" : apiOk === false ? "down" : ""}`}>
              {apiOk === null ? "checking API…" : apiOk ? "API connected" : "API unreachable"}
            </span>
          </div>
        </header>

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
