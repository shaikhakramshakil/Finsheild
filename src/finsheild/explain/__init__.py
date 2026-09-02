"""Finsheild SHAP Explainability (Phase 11).

Re-exports the canonical explainability helpers.
"""

from finsheild.explain.explainer import (
    evidence_from_features,
    explain_batch,
    explain_transaction,
    top_evidence,
)

__all__ = [
    "explain_batch",
    "explain_transaction",
    "evidence_from_features",
    "top_evidence",
]
