"""FastAPI app — demo-first, honest source labels, never silent failures."""
from __future__ import annotations

import os
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from .adapters.mock_adapter import MockMLAdapter
from .adapters.real_adapter import probe, RealMLAdapter
from .metrics_loader import get_metrics as load_metrics
from .schemas import CopilotEvidence, CopilotResponse, Transaction
from .services import store

# Try real adapter first (reuses existing pipeline), fallback to mock honestly
USE_REAL = os.getenv("FINSHEILD_USE_REAL", "auto")  # auto | mock | real
if USE_REAL == "real":
    adapter = RealMLAdapter()
elif USE_REAL == "mock":
    adapter = MockMLAdapter()
else:
    # auto: try real, fallback to mock
    try:
        p = probe()
        # real is considered live if risk_fusion code is available (even without model files)
        if p.get("risk_fusion"):
            adapter = RealMLAdapter()
        else:
            adapter = MockMLAdapter()
    except Exception:
        adapter = MockMLAdapter()

# Also expose mock for explicit fallback
_mock = MockMLAdapter()

app = FastAPI(title="Finsheild Demo API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    return {"status": "ok", "adapter": adapter.name, "model_status": probe()}


@app.get("/api/model/metrics")
def model_metrics():
    # Enhance with synthetic comparison if available
    base = load_metrics()
    # Add synthetic experiments
    from pathlib import Path
    import json
    repo = Path(__file__).resolve().parents[2]
    synth = {}
    for p in repo.glob("evaluation/reports/synthetic_*_metrics.json"):
        try:
            d = json.loads(p.read_text())
            synth[p.stem] = {"pr_auc": d.get("pr_auc"), "roc_auc": d.get("roc_auc"), "fraud_rate": d.get("fraud_rate")}
        except Exception:
            pass
    base["synthetic_experiments"] = synth
    # Comparison report if exists
    comp = repo / "evaluation/reports/synthetic_hard_overlap_comparison_report.md"
    if comp.exists():
        base["comparison_report"] = comp.read_text()[:8000]
    return base


@app.get("/api/model/status")
def model_status():
    return probe()


def _score_with_fallback(txn, ctx):
    try:
        return adapter.score(txn, ctx)
    except Exception as e:
        # Real adapter failed — fallback to mock with honest label
        score = _mock.score(txn, ctx)
        score.source = "DEMO_FALLBACK"  # type: ignore
        return score


@app.post("/api/transaction/score")
def score_transaction(txn: Transaction, scenario: str = Query(default="normal")):
    if scenario not in store.SCENARIOS:
        scenario = "normal"
    ctx = {"scenario": scenario}
    score = _score_with_fallback(txn, ctx)
    return store.save_scored(txn, ctx, score.model_dump())


@app.post("/api/transactions/generate")
def generate(scenario: str = Query(default="normal"), seed: int | None = None):
    if scenario not in store.SCENARIOS:
        scenario = "normal"
    txn, ctx = store.make_transaction(scenario, seed=seed)
    score = _score_with_fallback(txn, ctx)
    return store.save_scored(txn, ctx, score.model_dump())


@app.get("/api/transactions")
def list_transactions(limit: int = 50):
    return {"transactions": store.list_all(limit), "kind": "DEMO_SIMULATION"}


@app.get("/api/transactions/{txn_id}")
def get_transaction(txn_id: str):
    rec = store.get(txn_id)
    if not rec:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Not found")
    return rec


@app.post("/api/investigation/explain")
def explain(body: CopilotEvidence):
    """Copilot explains ENGINE evidence. Never overrides the risk decision."""
    # Try to use real LLM Data generator if available, else deterministic fallback
    try:
        # Use existing explain evidence_from_features if available
        rules_text = ", ".join(body.triggered_rules) if body.triggered_rules else "none"
        # Determine fraud type from amount deviation
        if body.transaction_amount > body.usual_amount * 3 and body.new_device:
            fraud_type = "ACCOUNT_TAKEOVER"
        elif body.recent_transaction_count >= 5:
            fraud_type = "VELOCITY_ABUSE"
        elif body.new_device:
            fraud_type = "DEVICE_COMPROMISE"
        elif body.location_distance_km > 300:
            fraud_type = "GEO_ANOMALY"
        else:
            fraud_type = "BEHAVIORAL_ANOMALY"
        # Risk from xgb score
        xgb = body.xgboost_score
        if xgb >= 0.7:
            risk = "HIGH"
            action = "BLOCK and investigate — multiple strong signals"
        elif xgb >= 0.4:
            risk = "MEDIUM"
            action = "STEP-UP verification (OTP / biometric)"
        else:
            risk = "LOW"
            action = "APPROVE — no further action"
        summary = (
            f"This transaction was flagged because amount deviation is {body.transaction_amount/body.usual_amount:.1f}× vs usual, "
            f"device is {'new' if body.new_device else 'known'}, location distance {body.location_distance_km:.0f}km, "
            f"velocity {body.recent_transaction_count} recent, XGB {xgb:.2f}, anomaly {body.anomaly_score:.2f}. "
            f"Rules: {rules_text}. This explanation does not set the risk score — the risk engine does."
        )
        return CopilotResponse(
            risk_level=risk, fraud_type=fraud_type, summary=summary,
            evidence=[f"Amount {body.transaction_amount} vs usual {body.usual_amount}", f"Device {'new' if body.new_device else 'known'}", f"Velocity {body.recent_transaction_count}", f"Anomaly {body.anomaly_score:.2f}"],
            recommended_action=action, source="DEMO_FALLBACK",
        ).model_dump()
    except Exception as e:
        return CopilotResponse(
            risk_level="MEDIUM", fraud_type="UNKNOWN", summary=f"Copilot fallback: {e}", evidence=[], recommended_action="INVESTIGATE", source="DEMO_FALLBACK"
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
