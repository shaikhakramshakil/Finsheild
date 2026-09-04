const API = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

async function req(path: string, init?: RequestInit) {
  const r = await fetch(`${API}${path}`, { ...init, headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) } });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

export const api = {
  health: () => req("/api/health"),
  metrics: () => req("/api/model/metrics"),
  generate: (scenario: string) => req(`/api/transactions/generate?scenario=${scenario}`, { method: "POST" }),
  list: () => req("/api/transactions"),
  get: (id: string) => req(`/api/transactions/${id}`),
  explain: (body: object) => req("/api/investigation/explain", { method: "POST", body: JSON.stringify(body) }),
  graph: (id: string) => req(`/api/graph/${id}`),
  identity: (uid: string) => req(`/api/identity/${uid}`),
  reset: () => req("/api/demo/reset", { method: "POST" }),
};

export type Score = {
  transaction_id: string; risk_score: number; risk_level: "LOW"|"MEDIUM"|"HIGH"|"CRITICAL";
  signals: {name:string;value:number;contribution:number}[]; rules: string[];
  behavioral_score: number; anomaly_score: number; xgb_score: number|null;
  graph_score: number|null; evidence: string[]; source: "LIVE_MODEL"|"DEMO_FALLBACK";
};
export type Rec = { transaction: Record<string, unknown>; context: {scenario:string} & Record<string, unknown>; score: Score };
