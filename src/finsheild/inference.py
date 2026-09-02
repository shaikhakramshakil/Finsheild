"""Inference — single core (predict_proba) plus thin batch + record wrappers.

All paths come from `config.ProjectPaths` so the same code runs locally and in Colab.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from finsheild.data.preprocessing import FraudPreprocessor
from finsheild.model import predict_proba

logger = logging.getLogger(__name__)

EXPECTED_INPUT_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


@dataclass
class FraudPredictor:
    """Wraps a fitted estimator + scaler + tuned threshold. The single shape the app will consume."""

    model: Any
    preprocessor: FraudPreprocessor
    threshold: float
    feature_order: list[str]

    @classmethod
    def load(cls, model_path: str | Path, scaler_path: str | Path, threshold: float = 0.5) -> "FraudPredictor":
        import joblib
        model_path = Path(model_path)
        scaler_path = Path(scaler_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        model = joblib.load(model_path)
        pre = FraudPreprocessor.load(scaler_path)
        order = list(pre.scale_features) + [c for c in EXPECTED_INPUT_COLUMNS if c not in pre.scale_features]
        logger.info("Loaded predictor: model=%s, threshold=%.4f", type(model).__name__, threshold)
        return cls(model=model, preprocessor=pre, threshold=float(threshold), feature_order=order)

    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = set(EXPECTED_INPUT_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required input columns: {sorted(missing)}")
        return df[EXPECTED_INPUT_COLUMNS]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(fraud=1) for each row. Core shape — app uses this."""
        df = self._ensure_columns(X.copy())
        scaled = self.preprocessor.transform(df)
        order = self.preprocessor._feature_order or EXPECTED_INPUT_COLUMNS
        scaled = scaled[order]
        return predict_proba(self.model, scaled.to_numpy())

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return 0/1 predictions using the tuned threshold."""
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def predict_record(self, record: dict) -> dict:
        """Score a single transaction dict. Convenience wrapper for the future app."""
        df = pd.DataFrame([record])
        prob = float(self.predict_proba(df)[0])
        return {
            "fraud_prob": prob,
            "threshold": self.threshold,
            "is_fraud": int(prob >= self.threshold),
        }

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a DataFrame of transactions. Returns df + fraud_prob + is_fraud columns."""
        out = df.copy().reset_index(drop=True)
        out["fraud_prob"] = self.predict_proba(out)
        out["is_fraud"] = (out["fraud_prob"] >= self.threshold).astype(int)
        return out