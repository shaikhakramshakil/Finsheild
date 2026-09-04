import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, type Rec } from "./api";

function Badge({ level }: { level: string }) {
  return <span className={`mono px-2 py-0.5 rounded border risk-${level}`}>{level}</span>;
}
function Src({ s }: { s: string }) {
  return <span className={`mono text-xs px-2 py-0.5 rounded border src-${s}`}>{s.replace("_", " ")}</span>;
}

export function Dashboard() {
  const [items, setItems] = useState<Rec[]>([]);
  const [metrics, setMetrics] = useState<any>(null);
  const [health, setHealth] = useState<any>(null);
  const [running, setRunning] = useState(false);

  async function refresh() {
    try { setItems((await api.list()).transactions ?? []); } catch {}
    try { setMetrics(await api.metrics()); } catch {}
    try { setHealth(await api.health()); } catch {}
  }
  useEffect(() => { refresh(); }, []);
  useEffect(() => {
    if (!running) return;
    const t = setInterval(async () => {
      const sc = Math.random() < 0.7 ? "normal" : "suspicious";
      try { await api.generate(sc); await refresh(); } catch {}
    }, 2500);
    return () => clearInterval(t);
  }, [running ]);

  const alerts = items.filter((r) => r.score.risk_score >= 0.6);
  return (
    <div className="p-6 max-w-6xl mx-auto space-y-6">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">FINSHEILD — Command Center</h1>
          <p className="text-sm opacity-70">Transaction → Detection → Evidence → Risk → Investigation → Decision</p>
        </div>
        <Src s={health ? (health.adapter === "mock" ? "DEMO_FALLBACK" : "LIVE_MODEL") : "DEMO_FALLBACK"} />
      </header>

      <nav className="flex gap-4 text-sm">
        <Link className="underline" to="/">Command Center</Link>
        <Link className="underline" to="/performance">Model Performance</Link>
        <Link className="underline" to="/architecture">Architecture</Link>
        <Link className="underline" to="/privacy/U-00001">Privacy Identity</Link>
      </nav>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[["Transactions processed", items.length],
          ["High-risk alerts", alerts.length],
          ["XGB ROC-AUC (real ULB)", metrics?.xgboost ? metrics.xgboost.roc_auc.toFixed(4) : "…"],
          ["XGB PR-AUC (real ULB)", metrics?.xgboost ? metrics.xgboost.pr_auc.toFixed(4) : "…"],
        ].map(([k, v]) => (
          <div key={k} className="card p-4"><div className="text-xs opacity-60">{k}</div>
            <div className="text-2xl font-semibold mono">{v}</div></div>
        ))}
      </div>

      <div className="card p-4 space-y-2">
        <div className="font-semibold">Demo Controls</div>
        <div className="flex flex-wrap gap-2 text-sm">
          <button className="border px-3 py-1 rounded" onClick={() => setRunning(!running)}>{running ? "Pause" : "Start simulation"}</button>
          <button className="border px-3 py-1 rounded" onClick={async () => { await api.reset(); refresh(); }}>Reset</button>
          {(["normal", "suspicious", "fraud_ring", "ambiguous"] as const).map((s) => (
            <button key={s} className="border px-3 py-1 rounded" onClick={async () => { await api.generate(s); refresh(); }}>
              Generate {s.replace("_", " ")}
            </button>
          ))}
        </div>
        <p className="text-xs opacity-60">Simulation data only — not real banking activity. Model: {health?.adapter ?? "…"} ({health?.model_status?.detail?.note ?? ""})</p>
      </div>

      <div className="card p-4">
        <div className="font-semibold mb-2">Recent alerts</div>
        <table className="w-full text-sm mono">
          <thead><tr className="text-left opacity-60"><th>ID</th><th>Amount</th><th>Risk</th><th>Score</th><th>Source</th></tr></thead>
          <tbody>
            {items.slice(0, 20).map((r) => (
              <tr key={r.transaction.transaction_id as string} className="border-t border-neutral-800">
                <td><Link className="underline" to={`/investigate/${r.transaction.transaction_id}`}>{r.transaction.transaction_id as string}</Link></td>
                <td>₹{r.transaction.amount as number}</td>
                <td><Badge level={r.score.risk_level} /></td>
                <td>{r.score.risk_score.toFixed(3)}</td>
                <td><Src s={r.score.source} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function Investigation() {
  const { id } = useParams();
  const nav = useNavigate();
  const [rec, setRec] = useState<Rec | null>(null);
  const [copilot, setCopilot] = useState<any>(null);
  const [graph, setGraph] = useState<any>(null);
  useEffect(() => {
    (async () => {
      if (!id) return;
      try { setRec(await api.get(id)); } catch { setRec(null); }
      try { setGraph(await api.graph(id)); } catch {}
    })();
  }, [id]);
  if (!rec) return <div className="p-6">Loading… <button className="underline" onClick={() => nav("/")}>back</button></div>;
  const t = rec.transaction as Record<string, any>;
  const s = rec.score;
  return (
    <div className="p-6 max-w-6xl mx-auto space-y-4">
      <Link className="underline text-sm" to="/">← Command Center</Link>
      <h2 className="text-xl font-bold mono">{t.transaction_id} <Badge level={s.risk_level} /> <Src s={s.source} /></h2>
      <div className="grid md:grid-cols-3 gap-4">
        <div className="card p-4 text-sm space-y-1">
          <div className="font-semibold">Transaction</div>
          {[["amount", `₹${t.amount}`], ["merchant", t.merchant], ["category", t.merchant_category],
            ["device", t.device_id], ["location", t.location], ["user", t.user_id], ["velocity", t.velocity]].map(([k, v]) => (
            <div key={k} className="flex justify-between mono"><span className="opacity-60">{k}</span><span>{v}</span></div>
          ))}
        </div>
        <div className="card p-4 text-sm space-y-1">
          <div className="font-semibold">Risk assessment</div>
          {[["fused risk", s.risk_score], ["xgb", s.xgb_score ?? "Not available"],
            ["anomaly", s.anomaly_score], ["behavioral", s.behavioral_score], ["graph", s.graph_score ?? "Not available"],
            ["rules", s.rules.join(", ") || "none"]].map(([k, v]) => (
            <div key={k} className="flex justify-between mono"><span className="opacity-60">{k}</span><span>{String(v)}</span></div>
          ))}
          <div className="pt-2"><div className="font-semibold">Why flagged?</div>
            <ul className="list-disc ml-5">{s.evidence.map((e: string) => <li key={e} className="mono">{e}</li>)}</ul></div>
        </div>
        <div className="card p-4 text-sm space-y-2">
          <div className="font-semibold">Investigation Copilot (explains engine evidence only)</div>
          <button className="border px-3 py-1 rounded" onClick={async () => {
            setCopilot(await api.explain({
              transaction_amount: t.amount, usual_amount: 4200, new_device: String(t.device_id).includes("NEW"),
              location_distance_km: 400, recent_transaction_count: t.velocity,
              xgboost_score: s.xgb_score ?? s.risk_score, anomaly_score: s.anomaly_score,
              triggered_rules: s.rules, graph_signals: { shared_device_accounts: 4 },
            }));
          }}>Run explanation</button>
          {copilot && <pre className="mono text-xs whitespace-pre-wrap">{JSON.stringify(copilot, null, 2)}</pre>}
        </div>
      </div>
      <div className="card p-4 text-sm">
        <div className="font-semibold">Graph intelligence (DEMO SIMULATION)</div>
        <pre className="mono text-xs">{JSON.stringify(graph, null, 2)}</pre>
      </div>
    </div>
  );
}

export function Performance() {
  const [m, setM] = useState<any>(null);
  useEffect(() => { api.metrics().then(setM).catch(() => {}); }, []);
  if (!m) return <div className="p-6">Loading…</div>;
  const row = (name: string, d: any) => (
    <tr className="border-t border-neutral-800 mono">
      <td>{name}</td><td>{d.roc_auc.toFixed(4)}</td><td>{d.pr_auc.toFixed(4)}</td>
      <td>{d.precision.toFixed(4)}</td><td>{d.recall.toFixed(4)}</td><td>{d.f1.toFixed(4)}</td>
    </tr>
  );
  return (
    <div className="p-6 max-w-5xl mx-auto space-y-4">
      <Link className="underline text-sm" to="/">← Command Center</Link>
      <h2 className="text-xl font-bold">Model Performance — REAL ULB BENCHMARK</h2>
      <p className="text-xs opacity-60">Read from {m.sources.xgboost}. Never mixed with synthetic/demo data.</p>
      <div className="card p-4">
        <table className="w-full text-sm">
          <thead><tr className="text-left opacity-60"><th>Model</th><th>ROC-AUC</th><th>PR-AUC</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
          <tbody>{row("Logistic Regression", m.logistic_regression)}{row("XGBoost", m.xgboost)}</tbody>
        </table>
      </div>
      <div className="card p-4 mono text-sm">
        XGBoost holdout @0.5: TP={m.xgboost.confusion_matrix.tp} FN={m.xgboost.confusion_matrix.fn} FP={m.xgboost.confusion_matrix.fp} TN={m.xgboost.confusion_matrix.tn}
      </div>
    </div>
  );
}

export function Architecture() {
  return (
    <div className="p-6 max-w-4xl mx-auto space-y-4">
      <Link className="underline text-sm" to="/">← Command Center</Link>
      <h2 className="text-xl font-bold">System Architecture</h2>
      <pre className="card p-4 mono text-xs overflow-x-auto">{`Transaction -> Feature Engineering
  -> [XGBoost | Behavioral | Rules | Anomaly | Graph]
  -> Risk Fusion -> SHAP/Evidence -> Investigation Copilot
  -> Investigator Decision -> Privacy Identity Layer

ML/Risk Engine -> Risk Decision -> Evidence -> LLM -> Explanation.
The LLM NEVER sets the fraud score.`}</pre>
    </div>
  );
}

export function Privacy() {
  const { uid } = useParams();
  const [d, setD] = useState<any>(null);
  useEffect(() => { if (uid) api.identity(uid).then(setD).catch(() => {}); }, [uid]);
  return (
    <div className="p-6 max-w-3xl mx-auto space-y-4">
      <Link className="underline text-sm" to="/">← Command Center</Link>
      <h2 className="text-xl font-bold">Privacy-Preserving Identity Representation</h2>
      <pre className="card p-4 mono text-sm">{JSON.stringify(d, null, 2)}</pre>
      <p className="text-xs opacity-60">Prototype identity tokenization — NOT a zero-knowledge proof.</p>
    </div>
  );
}
