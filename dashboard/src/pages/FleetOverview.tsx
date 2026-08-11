import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api, type FleetEngineSummary, type ModelComparison, type RiskTier } from "../api";
import { InfoTip } from "../components/InfoTip";
import { RiskBadge } from "../components/RiskBadge";

const RISK_ORDER: RiskTier[] = ["critical", "high", "medium", "low"];

type SortKey = "unit_number" | "latest_cycle" | "predicted_rul" | "true_rul";

const BAR_TRACK_WIDTH = 90;

function IntervalBar({
  lower,
  upper,
  point,
  scaleMax,
}: {
  lower: number;
  upper: number;
  point: number;
  scaleMax: number;
}) {
  const toPx = (v: number) => Math.max(0, Math.min(1, v / scaleMax)) * BAR_TRACK_WIDTH;
  return (
    <div>
      <div className="interval-bar">
        <div
          className="interval-bar-fill"
          style={{ left: toPx(lower), width: Math.max(1, toPx(upper) - toPx(lower)) }}
        />
        <div className="interval-bar-point" style={{ left: toPx(point) }} />
      </div>
      <div className="interval-label">
        {lower.toFixed(0)}–{upper.toFixed(0)}
      </div>
    </div>
  );
}

function FleetTableSkeleton() {
  return (
    <div className="skeleton-block">
      {Array.from({ length: 8 }).map((_, i) => (
        <div key={i} className="skeleton-row" style={{ width: `${92 - (i % 3) * 8}%` }} />
      ))}
    </div>
  );
}

export function FleetOverview({ subset }: { subset: string }) {
  const [engines, setEngines] = useState<FleetEngineSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [riskFilter, setRiskFilter] = useState<RiskTier | "all">("all");
  const [comparison, setComparison] = useState<ModelComparison | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>("predicted_rul");
  const [sortAsc, setSortAsc] = useState(true);

  useEffect(() => {
    setEngines(null);
    setError(null);
    api
      .fleet(subset)
      .then(setEngines)
      .catch((err) => setError(String(err.message ?? err)));
    api.modelComparison(subset).then(setComparison).catch(() => setComparison(null));
  }, [subset]);

  const counts = useMemo(() => {
    const c: Record<RiskTier, number> = { critical: 0, high: 0, medium: 0, low: 0 };
    engines?.forEach((e) => c[e.risk_tier]++);
    return c;
  }, [engines]);

  const scaleMax = useMemo(() => {
    if (!engines?.length) return 150;
    return Math.max(...engines.map((e) => e.predicted_rul_upper ?? e.predicted_rul), 10);
  }, [engines]);

  const sorted = useMemo(() => {
    if (!engines) return [];
    const filtered = riskFilter === "all" ? engines : engines.filter((e) => e.risk_tier === riskFilter);
    const dir = sortAsc ? 1 : -1;
    return [...filtered].sort((a, b) => {
      const av = a[sortKey] ?? 0;
      const bv = b[sortKey] ?? 0;
      return (av - bv) * dir;
    });
  }, [engines, riskFilter, sortKey, sortAsc]);

  function toggleSort(key: SortKey) {
    if (key === sortKey) {
      setSortAsc((asc) => !asc);
    } else {
      setSortKey(key);
      setSortAsc(true);
    }
  }

  function sortArrow(key: SortKey) {
    if (key !== sortKey) return null;
    return <span className="sort-arrow">{sortAsc ? "▲" : "▼"}</span>;
  }

  if (error) {
    return (
      <div className="panel error-panel">
        <p>Fleet data request failed: {error}</p>
        <p className="muted">
          Is the API running? See README.md for `uv run uvicorn api.main:app`.
        </p>
      </div>
    );
  }

  return (
    <div>
      <div className="stat-row">
        {RISK_ORDER.map((tier) => (
          <button
            key={tier}
            className={`stat-card risk-${tier} ${riskFilter === tier ? "active" : ""}`}
            onClick={() => setRiskFilter(riskFilter === tier ? "all" : tier)}
            disabled={!engines}
          >
            <span className="stat-value">{engines ? counts[tier] : "–"}</span>
            <span className="stat-label">{tier}</span>
          </button>
        ))}
        <button
          className={`stat-card ${riskFilter === "all" ? "active" : ""}`}
          onClick={() => setRiskFilter("all")}
          disabled={!engines}
        >
          <span className="stat-value">{engines ? engines.length : "–"}</span>
          <span className="stat-label">total</span>
        </button>
      </div>

      {comparison && (
        <Link to="/models" className="champion-banner">
          <span>
            Served by <span className="mono">{comparison.champion.toUpperCase()}</span> — chosen
            over {Object.keys(comparison.metrics).length - 1} other models on this subset
          </span>
          <span className="champion-banner-cta">VIEW MODEL COMPARISON →</span>
        </Link>
      )}

      <div className="panel">
        <h3 className="bracket-title">
          <span className="bracket">[</span>
          <span className="title-text">FLEET STATUS — {subset}</span>
          <span className="bracket">]</span>
        </h3>

        {!engines ? (
          <FleetTableSkeleton />
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table className="fleet-table">
              <thead>
                <tr>
                  <th className="sortable" onClick={() => toggleSort("unit_number")}>
                    Engine {sortArrow("unit_number")}
                  </th>
                  <th className="sortable" onClick={() => toggleSort("latest_cycle")}>
                    Latest cycle {sortArrow("latest_cycle")}
                  </th>
                  <th className="sortable" onClick={() => toggleSort("predicted_rul")}>
                    Predicted RUL {sortArrow("predicted_rul")}
                  </th>
                  <th>
                    90% interval
                    <InfoTip>
                      Split conformal prediction interval — the true RUL falls inside this
                      range roughly 90% of the time, calibrated on held-out engines.
                    </InfoTip>
                  </th>
                  <th className="sortable" onClick={() => toggleSort("true_rul")}>
                    True RUL {sortArrow("true_rul")}
                  </th>
                  <th>Model</th>
                  <th>Risk</th>
                  <th>Recommendation</th>
                </tr>
              </thead>
              <tbody>
                {sorted.map((e) => (
                  <tr key={e.unit_number}>
                    <td>
                      <Link to={`/engine/${e.unit_number}`}>#{e.unit_number}</Link>
                    </td>
                    <td className="mono muted">{e.latest_cycle}</td>
                    <td className="mono">{e.predicted_rul.toFixed(1)}</td>
                    <td>
                      {e.predicted_rul_lower != null && e.predicted_rul_upper != null ? (
                        <IntervalBar
                          lower={e.predicted_rul_lower}
                          upper={e.predicted_rul_upper}
                          point={e.predicted_rul}
                          scaleMax={scaleMax}
                        />
                      ) : (
                        <span className="muted">—</span>
                      )}
                    </td>
                    <td className="mono muted">{e.true_rul ?? "—"}</td>
                    <td className="mono muted">{e.model_used}</td>
                    <td>
                      <RiskBadge tier={e.risk_tier} />
                    </td>
                    <td className="recommendation">{e.recommendation}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
