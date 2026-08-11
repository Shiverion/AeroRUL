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
import { InfoTip } from "../components/InfoTip";
import { MultiSelect } from "../components/MultiSelect";
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
const CHART_COLORS = ["#ff2a2a", "#eaeaea", "#4af626", "#8a8a8a"];

function EngineDetailSkeleton({ unitNumber }: { unitNumber?: string }) {
  return (
    <div className="panel">
      <p className="muted mono">LOADING ENGINE #{unitNumber}…</p>
      <div className="skeleton-block" style={{ marginTop: 12 }}>
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="skeleton-row" style={{ width: `${88 - i * 10}%` }} />
        ))}
      </div>
    </div>
  );
}

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
        <p>Engine #{unitNumber} request failed: {error}</p>
        <Link to="/" className="back-link">&larr; Back to fleet</Link>
      </div>
    );
  }

  if (!engine) {
    return <EngineDetailSkeleton unitNumber={unitNumber} />;
  }

  const chartData = engine.cycles.map((cycle, i) => {
    const row: Record<string, number> = { cycle };
    selectedSensors.forEach((s) => {
      row[s] = engine.sensors[s][i];
    });
    return row;
  });

  const availableSensors = Object.keys(engine.sensors).filter((s) => s in SENSOR_LABELS);
  const sensorOptions = availableSensors.map((s, i) => ({
    value: s,
    label: `${s} — ${SENSOR_LABELS[s]}`,
    color: CHART_COLORS[i % CHART_COLORS.length],
  }));

  return (
    <div>
      <Link to="/" className="back-link">
        &larr; Back to fleet
      </Link>

      <div className="panel engine-summary">
        <div>
          <h2>Engine #{engine.unit_number}</h2>
          <p className="muted">
            {engine.subset} // {engine.cycles.length} cycles observed
          </p>
        </div>
        <div className="engine-summary-stats">
          <div>
            <span className="stat-value">{engine.predicted_rul.toFixed(1)}</span>
            <span className="stat-label">predicted rul</span>
          </div>
          <div>
            <span className="stat-value">
              {engine.predicted_rul_lower != null && engine.predicted_rul_upper != null
                ? `${engine.predicted_rul_lower.toFixed(0)}–${engine.predicted_rul_upper.toFixed(0)}`
                : "—"}
            </span>
            <span className="stat-label">
              90% interval
              <InfoTip>
                Split conformal prediction interval — the true RUL falls inside this range
                roughly 90% of the time, calibrated on held-out engines.
              </InfoTip>
            </span>
          </div>
          <div>
            <span className="stat-value">{engine.true_rul ?? "—"}</span>
            <span className="stat-label">true rul</span>
          </div>
          <div>
            <span className="stat-value mono-small">{engine.model_used}</span>
            <span className="stat-label">
              model
              <InfoTip>
                The champion model for this subset — the architecture that scored lowest on
                NASA score across XGBoost, LSTM, TCN, Transformer, and survival analysis.
              </InfoTip>
            </span>
          </div>
          <RiskBadge tier={engine.risk_tier} />
        </div>
        <p className="recommendation-banner">{engine.recommendation}</p>
      </div>

      <div className="panel">
        <h3 className="bracket-title">
          <span className="bracket">[</span>
          <span className="title-text">SENSOR HISTORY</span>
          <span className="bracket">]</span>
        </h3>

        <MultiSelect
          label="SENSORS"
          options={sensorOptions}
          selected={selectedSensors}
          onChange={setSelectedSensors}
        />

        <ResponsiveContainer width="100%" height={360}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="2 3" stroke="#2a2a2a" />
            <XAxis
              dataKey="cycle"
              stroke="#5c5c5c"
              tick={{ fill: "#8a8a8a", fontSize: 11, fontFamily: "JetBrains Mono, monospace" }}
              label={{ value: "CYCLE", position: "insideBottom", offset: -4, fill: "#5c5c5c", fontSize: 11 }}
            />
            <YAxis stroke="#5c5c5c" tick={{ fill: "#8a8a8a", fontSize: 11, fontFamily: "JetBrains Mono, monospace" }} />
            <Tooltip
              contentStyle={{
                background: "#0a0a0a",
                border: "1px solid #444444",
                borderRadius: 0,
                fontFamily: "JetBrains Mono, monospace",
                fontSize: 12,
              }}
              labelStyle={{ color: "#eaeaea" }}
            />
            {selectedSensors.map((s) => (
              <Line
                key={s}
                type="monotone"
                dataKey={s}
                name={`${s} — ${SENSOR_LABELS[s] ?? s}`}
                stroke={CHART_COLORS[availableSensors.indexOf(s) % CHART_COLORS.length]}
                dot={false}
                strokeWidth={1.75}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
