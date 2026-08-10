import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, type EngineHistory } from "../api";
import { RiskBadge } from "../components/RiskBadge";

const SENSOR_LABELS: Record<string, string> = {
  s_2: "LPC outlet temp",
  s_3: "HPC outlet temp",
  s_4: "LPT outlet temp",
  s_7: "HPC outlet pressure",
  s_8: "Physical fan speed",
  s_9: "Physical core speed",
  s_11: "HPC outlet static pressure",
  s_12: "Fuel flow / Ps30 ratio",
  s_13: "Corrected fan speed",
  s_14: "Corrected core speed",
  s_15: "Bypass ratio",
  s_17: "Bleed enthalpy",
  s_20: "HPT coolant bleed",
  s_21: "LPT coolant bleed",
};

const DEFAULT_SENSORS = ["s_4", "s_11", "s_9"];

export function EngineDetail({ subset }: { subset: string }) {
  const { unitNumber } = useParams<{ unitNumber: string }>();
  const [engine, setEngine] = useState<EngineHistory | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedSensors, setSelectedSensors] = useState<string[]>(DEFAULT_SENSORS);

  useEffect(() => {
    if (!unitNumber) return;
    setEngine(null);
    setError(null);
    api
      .engine(subset, Number(unitNumber))
      .then(setEngine)
      .catch((err) => setError(String(err.message ?? err)));
  }, [subset, unitNumber]);

  if (error) {
    return (
      <div className="panel error-panel">
        <p>Could not load engine #{unitNumber}: {error}</p>
        <Link to="/">← Back to fleet</Link>
      </div>
    );
  }

  if (!engine) {
    return <div className="panel">Loading engine #{unitNumber}…</div>;
  }

  const chartData = engine.cycles.map((cycle, i) => {
    const row: Record<string, number> = { cycle };
    selectedSensors.forEach((s) => {
      row[s] = engine.sensors[s][i];
    });
    return row;
  });

  const availableSensors = Object.keys(engine.sensors).filter((s) => s in SENSOR_LABELS);
  const colors = ["#5fd0ff", "#ff9f5f", "#a1ff5f"];

  return (
    <div>
      <Link to="/" className="back-link">← Back to fleet</Link>

      <div className="panel engine-summary">
        <div>
          <h2>Engine #{engine.unit_number}</h2>
          <p className="muted">{engine.subset} · {engine.cycles.length} cycles observed</p>
        </div>
        <div className="engine-summary-stats">
          <div>
            <span className="stat-value">{engine.predicted_rul.toFixed(1)}</span>
            <span className="stat-label">predicted RUL</span>
          </div>
          <div>
            <span className="stat-value">{engine.true_rul ?? "—"}</span>
            <span className="stat-label">true RUL</span>
          </div>
          <RiskBadge tier={engine.risk_tier} />
        </div>
        <p className="recommendation-banner">{engine.recommendation}</p>
      </div>

      <div className="panel">
        <div className="sensor-picker">
          {availableSensors.map((s) => (
            <label key={s} className="sensor-checkbox">
              <input
                type="checkbox"
                checked={selectedSensors.includes(s)}
                onChange={(e) => {
                  setSelectedSensors((prev) =>
                    e.target.checked ? [...prev, s] : prev.filter((x) => x !== s)
                  );
                }}
              />
              {s} — {SENSOR_LABELS[s]}
            </label>
          ))}
        </div>

        <ResponsiveContainer width="100%" height={360}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2a3444" />
            <XAxis dataKey="cycle" stroke="#8b98ab" label={{ value: "cycle", position: "insideBottom", offset: -4, fill: "#8b98ab" }} />
            <YAxis stroke="#8b98ab" />
            <Tooltip
              contentStyle={{ background: "#151b26", border: "1px solid #2a3444", borderRadius: 8 }}
              labelStyle={{ color: "#e7ecf3" }}
            />
            {selectedSensors.map((s, i) => (
              <Line
                key={s}
                type="monotone"
                dataKey={s}
                name={`${s} — ${SENSOR_LABELS[s] ?? s}`}
                stroke={colors[i % colors.length]}
                dot={false}
                strokeWidth={2}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
