import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, type ModelComparison as ModelComparisonData } from "../api";
import { KNOWN_LIMITATIONS, MODEL_INFO, NASA_SCORE_EXPLAINER } from "../modelInfo";

function ComparisonSkeleton() {
  return (
    <div className="panel">
      <div className="skeleton-block">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="skeleton-row" style={{ width: `${90 - (i % 3) * 10}%` }} />
        ))}
      </div>
    </div>
  );
}

export function ModelComparisonPage({ subset }: { subset: string }) {
  const [comparison, setComparison] = useState<ModelComparisonData | null>(null);
  const [notFound, setNotFound] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setComparison(null);
    setNotFound(false);
    setError(null);
    api
      .modelComparison(subset)
      .then((res) => (res ? setComparison(res) : setNotFound(true)))
      .catch((err) => setError(String(err.message ?? err)));
  }, [subset]);

  if (error) {
    return (
      <div className="panel error-panel">
        <p>Model comparison request failed: {error}</p>
        <Link to="/" className="back-link">&larr; Back to fleet</Link>
      </div>
    );
  }

  if (notFound) {
    return (
      <div className="panel">
        <p className="muted">
          No comparison found for {subset}. Run{" "}
          <span className="mono">scripts/compare_models.py --subset {subset}</span> to
          generate one.
        </p>
        <Link to="/" className="back-link">&larr; Back to fleet</Link>
      </div>
    );
  }

  return (
    <div>
      <Link to="/" className="back-link">
        &larr; Back to fleet
      </Link>

      <div className="panel">
        <h3 className="bracket-title">
          <span className="bracket">[</span>
          <span className="title-text">MODEL COMPARISON — {subset}</span>
          <span className="bracket">]</span>
        </h3>
        <p className="explainer-text">{NASA_SCORE_EXPLAINER}</p>
      </div>

      {!comparison ? (
        <ComparisonSkeleton />
      ) : (
        <>
          <div className="panel">
            <table className="fleet-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>RMSE</th>
                  <th>MAE</th>
                  <th>NASA score</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(comparison.metrics)
                  .sort(([, a], [, b]) => a.nasa_score - b.nasa_score)
                  .map(([modelType, m]) => (
                    <tr key={modelType}>
                      <td className="mono">
                        {modelType}
                        {modelType === comparison.champion && (
                          <span className="champion-tag">champion</span>
                        )}
                      </td>
                      <td className="mono muted">{m.rmse.toFixed(2)}</td>
                      <td className="mono muted">{m.mae.toFixed(2)}</td>
                      <td className="mono muted">{m.nasa_score.toFixed(1)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>

          <div className="model-card-grid">
            {Object.entries(comparison.metrics)
              .sort(([, a], [, b]) => a.nasa_score - b.nasa_score)
              .map(([modelType, m]) => {
                const info = MODEL_INFO[modelType];
                const isChampion = modelType === comparison.champion;
                if (!info) return null;
                return (
                  <div key={modelType} className={`panel model-card ${isChampion ? "champion" : ""}`}>
                    <div className="model-card-header">
                      <h4>
                        {info.name}
                        {isChampion && <span className="champion-tag">champion</span>}
                      </h4>
                      <div className="model-card-metrics">
                        <span>
                          RMSE <b>{m.rmse.toFixed(1)}</b>
                        </span>
                        <span>
                          MAE <b>{m.mae.toFixed(1)}</b>
                        </span>
                        <span>
                          NASA <b>{m.nasa_score.toFixed(0)}</b>
                        </span>
                      </div>
                    </div>
                    <p className="model-card-tagline">{info.tagline}</p>
                    <div className="model-card-section">
                      <span className="model-card-label">// grounding</span>
                      <p>{info.grounding}</p>
                    </div>
                    <div className="model-card-section">
                      <span className="model-card-label">// limitations</span>
                      <p>{info.limitations}</p>
                    </div>
                  </div>
                );
              })}
          </div>

          <div className="panel">
            <h3 className="bracket-title">
              <span className="bracket">[</span>
              <span className="title-text">KNOWN LIMITATIONS</span>
              <span className="bracket">]</span>
            </h3>
            <div className="limitations-list">
              {KNOWN_LIMITATIONS.map((item) => (
                <div key={item.title} className="limitations-item">
                  <span className="model-card-label">// {item.title}</span>
                  <p>{item.body}</p>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
