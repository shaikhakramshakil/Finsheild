"""FastAPI app — demo-first, honest source labels, never silent failures."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .schemas import CopilotEvidence, CopilotResponse, Transaction
from .adapters.mock_adapter import MockMLAdapter
from .adapters.real_adapter import probe
from . import metrics_loader
from .services import store

adapter = MockMLAdapter()
app = FastAPI(title="Finsheild Demo API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    return {"status": "ok", "adapter": adapter.name, "model_status": probe()}


@app.get("/api/model/metrics")
def model_metrics():
    return metrics_loader.get_metrics()


@app.get("/api/model/status")
def model_status():
    return probe()


@app.post("/api/transaction/score")
def score_transaction(txn: Transaction, scenario: str = Query(default="normal")):
    if scenario not in store.SCENARIOS:
        raise HTTPException(400, f"unknown scenario {scenario!r}")
    ctx = {"scenario": scenario, "recent_transaction_count": txn.velocity}
    score = adapter.score(txn, ctx)
    return store.save_scored(txn, ctx, score.model_dump())


@app.post("/api/transactions/generate")
def generate(scenario: str = Query(default="normal"), seed: int | None = None):
    if scenario not in store.SCENARIOS:
        raise HTTPException(400, f"unknown scenario {scenario!r}")
    txn, ctx = store.make_transaction(scenario, seed)
    score = adapter.score(txn, ctx)
    return store.save_scored(txn, ctx, score.model_dump())


@app.get("/api/transactions")
def list_transactions(limit: int = 50):
    return {"transactions": store.list_all(limit), "kind": "DEMO_SIMULATION"}


@app.get("/api/transactions/{txn_id}")
def get_transaction(txn_id: str):
    rec = store.get(txn_id)
    if not rec:
        raise HTTPException(404, "unknown transaction")
    return rec


@app.post("/api/investigation/explain")
def explain(body: CopilotEvidence):
    """Copilot explains ENGINE evidence. Never overrides the risk decision."""
    lvl = "HIGH" if (body.xgboost_score > 0.6 or len(body.triggered_rules) >= 2) else (
        "MEDIUM" if (body.xgboost_score > 0.3 or body.triggered_rules) else "LOW")
    ftype = "ACCOUNT_TAKEOVER" if body.new_device and body.transaction_amount > 5 * body.usual_amount else (
        "VELOCITY_ABUSE" if body.recent_transaction_count >= 5 else "UNDETERMINED")
    summary = (
        f"Engine scored xgb={body.xgboost_score:.2f}, anomaly={body.anomaly_score:.2f}. "
        f"Amount {body.transaction_amount} vs usual {body.usual_amount}; "
        f"{'new device; ' if body.new_device else ''}"
        f"{body.recent_transaction_count} recent txns; rules={body.triggered_rules or 'none'}. "
        "This explanation is derived from engine evidence; the copilot does not set the risk score.")
    return CopilotResponse(
        risk_level=lvl, fraud_type=ftype, summary=summary,
        evidence=[f"rule:{r}" for r in body.triggered_rules] + [
            f"amount {body.transaction_amount} vs usual {body.usual_amount}",
            f"graph shared_device_accounts={body.graph_signals.get('shared_device_accounts', 0)}"],
        recommended_action="STEP_UP_AUTH" if lvl == "MEDIUM" else (
            "BLOCK_AND_REVIEW" if lvl == "HIGH" else "APPROVE"),
        source="DEMO_FALLBACK",
    ).model_dump()


@app.get("/api/graph/{txn_id}")
def graph(txn_id: str):
    return store.graph_for(txn_id)


@app.get("/api/identity/{user_id}")
def identity(user_id: str):
    return store.tokenize(user_id)


@app.post("/api/demo/reset")
def reset():
    store.reset()
    return {"status": "reset"}
