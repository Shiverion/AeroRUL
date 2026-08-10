const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export type RiskTier = "low" | "medium" | "high" | "critical";

export interface FleetEngineSummary {
  unit_number: number;
  latest_cycle: number;
  predicted_rul: number;
  predicted_rul_lower: number | null;
  predicted_rul_upper: number | null;
  true_rul: number | null;
  risk_tier: RiskTier;
  recommendation: string;
  model_used: string;
}

export interface EngineHistory {
  unit_number: number;
  subset: string;
  cycles: number[];
  sensors: Record<string, number[]>;
  predicted_rul: number;
  predicted_rul_lower: number | null;
  predicted_rul_upper: number | null;
  true_rul: number | null;
  risk_tier: RiskTier;
  recommendation: string;
  model_used: string;
}

export interface HealthResponse {
  status: string;
  available_subsets: string[];
}

export interface ModelMetrics {
  rmse: number;
  mae: number;
  nasa_score: number;
}

export interface ModelComparison {
  champion: string;
  metrics: Record<string, ModelMetrics>;
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const detail = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status} ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => getJSON<HealthResponse>("/health"),
  fleet: (subset: string) => getJSON<FleetEngineSummary[]>(`/fleet/${subset}`),
  engine: (subset: string, unitNumber: number) =>
    getJSON<EngineHistory>(`/engine/${subset}/${unitNumber}`),
  modelComparison: async (subset: string): Promise<ModelComparison | null> => {
    const res = await fetch(`${API_BASE}/models/${subset}`);
    if (res.status === 404) return null;
    if (!res.ok) throw new Error(`${res.status} ${await res.text().catch(() => res.statusText)}`);
    return res.json() as Promise<ModelComparison>;
  },
};
