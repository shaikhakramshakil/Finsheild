"""Finsheild demo backend — shared API contract.

GUI -> MockMLAdapter (now) -> RealMLAdapter (later, Agent 3).
Both adapters return ScoreResult; the GUI never cares how it was produced.
"""
from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
SourceKind = Literal["LIVE_MODEL", "DEMO_FALLBACK"]
DataKind = Literal["REAL_BENCHMARK", "SYNTHETIC_EXPERIMENT", "DEMO_SIMULATION"]


class Signal(BaseModel):
    name: str
    value: float
    contribution: float


class ScoreResult(BaseModel):
    transaction_id: str
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    signals: list[Signal] = []
    rules: list[str] = []
    behavioral_score: float = 0.0
    anomaly_score: float = 0.0
    xgb_score: Optional[float] = None
    graph_score: Optional[float] = None
    evidence: list[str] = []
    source: SourceKind = "DEMO_FALLBACK"


class Transaction(BaseModel):
    transaction_id: str
    user_id: str
    amount: float
    timestamp: str
    merchant: str
    merchant_category: str
    device_id: str
    location: str
    velocity: int = 0


class ScoredTransaction(Transaction):
    score: ScoreResult


class InvestigationRequest(BaseModel):
    transaction_id: str


class CopilotEvidence(BaseModel):
    transaction_amount: float
    usual_amount: float
    new_device: bool
    location_distance_km: float
    recent_transaction_count: int
    xgboost_score: float
    anomaly_score: float
    triggered_rules: list[str] = []
    graph_signals: dict = {}


class CopilotResponse(BaseModel):
    risk_level: str
    fraud_type: str
    summary: str
    evidence: list[str]
    recommended_action: str
    source: SourceKind = "DEMO_FALLBACK"
