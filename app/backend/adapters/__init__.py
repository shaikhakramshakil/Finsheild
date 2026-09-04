"""Adapter protocol — the single seam Agent 3 will swap."""
from __future__ import annotations

from typing import Protocol
from ..schemas import Transaction, ScoreResult


class MLAdapter(Protocol):
    name: str

    def score(self, txn: Transaction, ctx: dict | None = None) -> ScoreResult:
        ...
