"""MockMLAdapter — deterministic demo scorer.

Produces ScoreResults in the exact GUI contract WITHOUT touching the real
ML pipeline. Used until Agent 3 swaps in RealMLAdapter.
All outputs are labelled source=DEMO_FALLBACK by the API layer.
"""
from __future__ import annotations

import hashlib
import math

from ..schemas import ScoreResult, Signal, Transaction


def _stable_unit(key: str) -> float:
    h = hashlib.sha256(key.encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def risk_level_for(score: float) -> str:
    if score >= 0.8:
        return "CRITICAL"
    if score >= 0.6:
        return "HIGH"
    if score >= 0.3:
        return "MEDIUM"
    return "LOW"


class MockMLAdapter:
    name = "mock"

    def score(self, txn: Transaction, ctx: dict | None = None) -> ScoreResult:
        ctx = ctx or {}
        scenario = ctx.get("scenario", "normal")

        # Deterministic base per scenario so demos are reliable.
        base = {"normal": 0.08, "suspicious": 0.91, "fraud_ring": 0.87, "ambiguous": 0.55}.get(scenario, 0.15)
        jitter = (_stable_unit(txn.transaction_id) - 0.5) * 0.06
        risk = max(0.01, min(0.99, base + jitter))

        amt_dev = ctx.get("amount_deviation", 4.2 if scenario == "suspicious" else 0.3)
        vel = float(txn.velocity or ctx.get("recent_transaction_count", 1))

        signals = [
            Signal(name="amount_deviation", value=round(float(amt_dev), 2),
                   contribution=round(0.31 if risk > 0.5 else 0.05, 3)),
            Signal(name="velocity", value=vel,
                   contribution=round(0.24 if vel >= 5 else 0.03, 3)),
        ]
        rules: list[str] = []
        evidence: list[str] = []
        if scenario == "suspicious":
            rules = ["NEW_DEVICE_HIGH_VALUE", "HIGH_VELOCITY"]
            evidence = [
                f"High behavioral deviation (amount {txn.amount} vs usual ~4200)",
                f"Unusual transaction velocity ({int(vel)} recent txns)",
                "New device, location ~400km from usual",
            ]
        elif scenario == "fraud_ring":
            rules = ["SHARED_DEVICE_MULTI_ACCOUNT"]
            evidence = ["Shared device across 4 accounts — possible fraud ring"]
        elif scenario == "ambiguous":
            rules = ["MODERATE_VELOCITY", "SLIGHT_AMOUNT_DEVIATION"]
            evidence = ["No single extreme feature; several moderate signals combine"]
        else:
            evidence = ["Device matches previous activity", "Amount within usual range"]

        return ScoreResult(
            transaction_id=txn.transaction_id,
            risk_score=round(risk, 3),
            risk_level=risk_level_for(risk),  # type: ignore[arg-type]
            signals=signals,
            rules=rules,
            behavioral_score=round(max(0.0, min(1.0, risk - 0.08)), 3),
            anomaly_score=round(max(0.0, min(1.0, risk - 0.12)), 3),
            xgb_score=None,  # mock has no real XGB — GUI shows "Not available"
            graph_score=0.85 if scenario == "fraud_ring" else 0.05,
            evidence=evidence,
            source="DEMO_FALLBACK",
        )
