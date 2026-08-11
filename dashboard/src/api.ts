// No VITE_API_BASE at build time -> static mode: read the JSON snapshots in public/data/
// (see scripts/export_static_data.py) instead of hitting a live server. In dev, default to
// the local API so `npm run dev` works out of the box without any .env file; in a
// production build, no configured base means a genuinely backend-free deployment (Vercel).
const configuredBase = import.meta.env.VITE_API_BASE as string | undefined;
const API_BASE = configuredBase ?? (import.meta.env.DEV ? "http://localhost:8000" : undefined);
const STATIC_MODE = !API_BASE;
const STATIC_BASE = "/data";

export const mode: "live" | "static" = STATIC_MODE ? "static" : "live";

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

async function getStaticJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${STATIC_BASE}${path}`);
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: async (): Promise<HealthResponse> => {
    if (STATIC_MODE) {
      const manifest = await getStaticJSON<{ available_subsets: string[] }>("/manifest.json");
      return { status: "static", available_subsets: manifest.available_subsets };
    }
    return getJSON<HealthResponse>("/health");
  },

  fleet: (subset: string) =>
    STATIC_MODE
      ? getStaticJSON<FleetEngineSummary[]>(`/${subset}/fleet.json`)
      : getJSON<FleetEngineSummary[]>(`/fleet/${subset}`),

  engine: (subset: string, unitNumber: number) =>
    STATIC_MODE
      ? getStaticJSON<EngineHistory>(`/${subset}/engines/${unitNumber}.json`)
      : getJSON<EngineHistory>(`/engine/${subset}/${unitNumber}`),

  modelComparison: async (subset: string): Promise<ModelComparison | null> => {
    if (STATIC_MODE) {
      try {
        return await getStaticJSON<ModelComparison>(`/${subset}/models.json`);
      } catch {
        return null;
      }
    }
    const res = await fetch(`${API_BASE}/models/${subset}`);
    if (res.status === 404) return null;
    if (!res.ok) throw new Error(`${res.status} ${await res.text().catch(() => res.statusText)}`);
    return res.json() as Promise<ModelComparison>;
  },
};
