import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api, type FleetEngineSummary, type RiskTier } from "../api";
import { RiskBadge } from "../components/RiskBadge";

const RISK_ORDER: RiskTier[] = ["critical", "high", "medium", "low"];

export function FleetOverview({ subset }: { subset: string }) {
  const [engines, setEngines] = useState<FleetEngineSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [riskFilter, setRiskFilter] = useState<RiskTier | "all">("all");

  useEffect(() => {
    setEngines(null);
    setError(null);
    api
      .fleet(subset)
      .then(setEngines)
      .catch((err) => setError(String(err.message ?? err)));
  }, [subset]);

  const counts = useMemo(() => {
    const c: Record<RiskTier, number> = { critical: 0, high: 0, medium: 0, low: 0 };
    engines?.forEach((e) => c[e.risk_tier]++);
    return c;
  }, [engines]);

  const filtered = useMemo(() => {
    if (!engines) return [];
    if (riskFilter === "all") return engines;
    return engines.filter((e) => e.risk_tier === riskFilter);
  }, [engines, riskFilter]);

  if (error) {
    return (
      <div className="panel error-panel">
        <p>Could not load fleet data: {error}</p>
        <p className="muted">Is the API running? See README for `uv run uvicorn api.main:app`.</p>
      </div>
    );
  }

  if (!engines) {
    return <div className="panel">Loading fleet…</div>;
  }

  return (
    <div>
      <div className="stat-row">
        {RISK_ORDER.map((tier) => (
          <button
            key={tier}
            className={`stat-card risk-${tier} ${riskFilter === tier ? "active" : ""}`}
            onClick={() => setRiskFilter(riskFilter === tier ? "all" : tier)}
          >
            <span className="stat-value">{counts[tier]}</span>
            <span className="stat-label">{tier}</span>
          </button>
        ))}
        <button
          className={`stat-card ${riskFilter === "all" ? "active" : ""}`}
          onClick={() => setRiskFilter("all")}
        >
          <span className="stat-value">{engines.length}</span>
          <span className="stat-label">total</span>
        </button>
      </div>

      <div className="panel">
        <table className="fleet-table">
          <thead>
            <tr>
              <th>Engine</th>
              <th>Latest cycle</th>
              <th>Predicted RUL</th>
              <th>True RUL</th>
              <th>Risk</th>
              <th>Recommendation</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((e) => (
              <tr key={e.unit_number}>
                <td>
                  <Link to={`/engine/${e.unit_number}`}>#{e.unit_number}</Link>
                </td>
                <td>{e.latest_cycle}</td>
                <td className="mono">{e.predicted_rul.toFixed(1)}</td>
                <td className="mono muted">{e.true_rul ?? "—"}</td>
                <td>
                  <RiskBadge tier={e.risk_tier} />
                </td>
                <td className="recommendation">{e.recommendation}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
