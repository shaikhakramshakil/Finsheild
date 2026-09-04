"""metrics_loader — reads REAL benchmark JSONs from the repo. Never hardcodes."""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
XGB_PATH = REPO_ROOT / "evaluation" / "reports" / "xgboost_metrics.json"
BASE_PATH = REPO_ROOT / "evaluation" / "reports" / "baseline_metrics.json"


def _load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def get_metrics() -> dict:
    xgb = _load(XGB_PATH)
    base = _load(BASE_PATH)
    return {
        "kind": "REAL_BENCHMARK",
        "label": "Real ULB Benchmark (284,807 rows, 0.17% fraud)",
        "xgboost": xgb,
        "logistic_regression": base,
        "sources": {"xgboost": str(XGB_PATH), "baseline": str(BASE_PATH)},
    }
