"""RealMLAdapter — integration seam for Agent 3 (NOT active).

Probes the existing research pipeline read-only and reports availability.
Never retrains, never fabricates. Until wired, score() raises
ModelUnavailable so the API can fall back to MockMLAdapter with an honest
source label.
"""
from __future__ import annotations

from pathlib import Path


class ModelUnavailable(RuntimeError):
    pass


REPO_ROOT = Path(__file__).resolve().parents[3]


def probe() -> dict:
    """Return availability of each real component (read-only checks)."""
    out: dict = {"xgb_ulb": False, "risk_fusion": True, "graph": True,
                 "shap": False, "llm_adapter": False, "detail": {}}
    try:
        if (REPO_ROOT / "models" / "xgboost").exists():
            out["xgb_ulb"] = True
            out["detail"]["xgboost_dir"] = str(REPO_ROOT / "models" / "xgboost")
    except Exception as e:  # pragma: no cover
        out["detail"]["xgboost_error"] = str(e)
    try:
        import shap  # type: ignore  # noqa: F401
        out["shap"] = True
    except Exception:
        out["shap"] = False
    llm = REPO_ROOT / "models" / "llm" / "adapter"
    out["llm_adapter"] = llm.exists()
    out["detail"]["note"] = "RealMLAdapter not yet wired (Agent 3). GUI uses MockMLAdapter."
    return out


class RealMLAdapter:
    name = "real"

    def score(self, txn, ctx=None):  # pragma: no cover
        raise ModelUnavailable("RealMLAdapter not yet wired — use MockMLAdapter.")
