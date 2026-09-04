"""Backend API tests — contract + honest labels."""
from fastapi.testclient import TestClient

from app.backend.main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["adapter"] in ("mock", "real")


def test_metrics_are_real_values():
    r = client.get("/api/model/metrics")
    d = r.json()
    assert d["kind"] == "REAL_BENCHMARK"
    assert abs(d["xgboost"]["roc_auc"] - 0.9709) < 0.001
    assert d["xgboost"]["confusion_matrix"] == {"tn": 42643, "fp": 5, "fn": 16, "tp": 58}


def test_generate_and_investigate_flow():
    for sc in ("normal", "suspicious", "fraud_ring", "ambiguous"):
        r = client.post(f"/api/transactions/generate?scenario={sc}")
        assert r.status_code == 200, r.text
        rec = r.json()
        assert rec["score"]["source"] in ("DEMO_FALLBACK", "LIVE_MODEL")
        tid = rec["transaction"]["transaction_id"]
        assert client.get(f"/api/transactions/{tid}").status_code == 200
    r = client.get("/api/transactions")
    assert r.json()["kind"] == "DEMO_SIMULATION"


def test_risk_ordering():
    n = client.post("/api/transactions/generate?scenario=normal").json()["score"]["risk_score"]
    s = client.post("/api/transactions/generate?scenario=suspicious").json()["score"]["risk_score"]
    assert s > n


def test_copilot_never_sets_score():
    body = {"transaction_amount": 50000, "usual_amount": 4200, "new_device": True,
            "location_distance_km": 400, "recent_transaction_count": 8,
            "xgboost_score": 0.91, "anomaly_score": 0.83,
            "triggered_rules": ["NEW_DEVICE_HIGH_VALUE", "HIGH_VELOCITY"],
            "graph_signals": {"shared_device_accounts": 4}}
    r = client.post("/api/investigation/explain", json=body)
    assert r.status_code == 200
    assert "does not set the risk score" in r.json()["summary"]


def test_graph_and_identity_labels():
    tid = client.post("/api/transactions/generate?scenario=fraud_ring").json()["transaction"]["transaction_id"]
    assert client.get(f"/api/graph/{tid}").json()["kind"] == "DEMO_SIMULATION"
    ident = client.get("/api/identity/U-00001").json()
    assert "zero-knowledge" in ident["method"].lower() or "NOT" in ident["method"]
